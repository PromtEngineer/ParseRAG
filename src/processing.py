"""
Processing pipeline: Parse PDF → Chunk → Screenshot → Embed → Store in Milvus
"""
import os
from pathlib import Path

from google import genai
from liteparse import LiteParse
from pymilvus import MilvusClient, DataType

SCREENSHOT_DIR = "screenshots"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
COLLECTION_NAME = "parserag"
MILVUS_DB_PATH = "./milvus_lite.db"

_parser = LiteParse()


def parse_pdf(file_path: str) -> dict[int, str]:
    """Extract layout-aware text per page using LiteParse."""
    result = _parser.parse(file_path)
    pages = {}
    for page in result.pages:
        pages[page.pageNum] = page.text
    return pages


def take_screenshots(file_path: str, out_dir: str = SCREENSHOT_DIR) -> dict[int, str]:
    """Render page screenshots using LiteParse."""
    stem = Path(file_path).stem
    doc_dir = os.path.join(out_dir, stem)
    os.makedirs(doc_dir, exist_ok=True)
    result = _parser.screenshot(file_path, output_dir=doc_dir)
    screenshot_paths = {}
    for r in result.screenshots:
        screenshot_paths[r.page_num] = r.image_path
    return screenshot_paths


def chunk_text(text: str, chunk_size: int = 4096) -> list[str]:
    """Split text into chunks, respecting paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 > chunk_size:
            if current:
                chunks.append(current.strip())
            current = paragraph
        else:
            current = current + "\n\n" + paragraph if current else paragraph
    if current.strip():
        chunks.append(current.strip())
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using Google Gemini embedding model."""
    client = genai.Client()
    embeddings = []
    for i in range(0, len(texts), 5):
        batch = [t if t.strip() else "empty page" for t in texts[i:i+5]]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
        )
        for emb in result.embeddings:
            embeddings.append(emb.values)
    return embeddings


def get_milvus_client() -> MilvusClient:
    """Connect to Milvus Lite (local file-based)."""
    return MilvusClient(uri=MILVUS_DB_PATH)


def ensure_collection(client: MilvusClient):
    """Create the parserag collection if it doesn't exist."""
    if client.has_collection(COLLECTION_NAME):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("source_file", DataType.VARCHAR, max_length=1024)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("image_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("page_num", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"  Created collection '{COLLECTION_NAME}'.")


def insert_data(
    client: MilvusClient,
    source_file: str,
    chunks: list[str],
    embeddings: list[list[float]],
    image_paths: list[str],
    page_nums: list[int],
):
    """Insert chunk data into Milvus."""
    data = []
    for i in range(len(chunks)):
        data.append({
            "source_file": source_file,
            "text": chunks[i],
            "image_path": image_paths[i],
            "page_num": page_nums[i],
            "vector": embeddings[i],
        })
    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"  Inserted {result['insert_count']} records.")


def pipeline(file_path: str, screenshot_dir: str = SCREENSHOT_DIR, reset: bool = False):
    """Process a single PDF: parse → chunk → screenshot → embed → store."""
    print(f"Processing: {file_path}")

    client = get_milvus_client()
    if reset and client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print("  Dropped existing collection.")
    ensure_collection(client)

    # Step 1: Parse
    print("  Parsing PDF with LiteParse...")
    pages = parse_pdf(file_path)
    print(f"  Extracted text from {len(pages)} pages.")

    # Step 2: Screenshots
    print("  Taking page screenshots...")
    screenshot_paths = take_screenshots(file_path, screenshot_dir)
    print(f"  Saved {len(screenshot_paths)} screenshots.")

    # Step 3: Chunk
    all_chunks = []
    all_image_paths = []
    all_page_nums = []
    for page_num, text in pages.items():
        chunks = chunk_text(text)
        img_path = screenshot_paths.get(page_num, "")
        for chunk in chunks:
            all_chunks.append(chunk)
            all_image_paths.append(img_path)
            all_page_nums.append(page_num)
    print(f"  Created {len(all_chunks)} chunks.")

    # Step 4: Embed
    print("  Generating embeddings...")
    embeddings = embed_texts(all_chunks)
    print(f"  Generated {len(embeddings)} embeddings.")

    # Step 5: Store
    source = Path(file_path).name
    insert_data(client, source, all_chunks, embeddings, all_image_paths, all_page_nums)
    print(f"  Done: {source}")

    return len(all_chunks)


def pipeline_directory(directory: str, screenshot_dir: str = SCREENSHOT_DIR, reset: bool = True):
    """Process all PDFs in a directory."""
    pdf_files = sorted(Path(directory).glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return

    print(f"Found {len(pdf_files)} PDF files in {directory}")

    client = get_milvus_client()
    if reset and client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print("Dropped existing collection.")
    ensure_collection(client)

    total_chunks = 0
    for i, pdf in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf.name}")
        try:
            n = pipeline(str(pdf), screenshot_dir, reset=False)
            total_chunks += n
        except Exception as e:
            print(f"  ERROR: {e}")

    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"\nIndexing complete!")
    print(f"  Files processed: {len(pdf_files)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Collection rows: {stats.get('row_count', 0)}")
