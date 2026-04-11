"""
Search module: Vector search + image retrieval via Milvus
"""
import base64
from pathlib import Path

from google import genai

from .processing import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_milvus_client,
)


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    client = genai.Client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
    )
    return result.embeddings[0].values


def search(query: str, limit: int = 3) -> list[dict]:
    """
    Vector similarity search in Milvus.
    Returns list of dicts with 'text', 'image_path', 'page_num', 'distance'.
    """
    embedding = embed_query(query)
    client = get_milvus_client()

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        limit=limit,
        output_fields=["text", "image_path", "page_num", "source_file"],
        search_params={"metric_type": "COSINE"},
    )

    search_results = []
    for hits in results:
        for hit in hits:
            search_results.append({
                "text": hit["entity"]["text"],
                "image_path": hit["entity"]["image_path"],
                "page_num": hit["entity"]["page_num"],
                "source_file": hit["entity"].get("source_file", ""),
                "distance": hit["distance"],
            })
    return search_results


def get_image_base64(image_path: str) -> str:
    """Read a screenshot from disk and return as base64 string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")
