"""
FastAPI backend for ParseRAG Web UI.
Serves the frontend and provides SSE streaming for agent execution.
"""
import json
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import run_agent_stream
from src.processing import (
    pipeline,
    pipeline_directory,
    get_milvus_client,
    COLLECTION_NAME,
)

app = FastAPI(title="ParseRAG")

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Default PDF path
DEFAULT_PDF = "data/Medication_Side_Effect_Flyer.pdf"


# --- Models ---

class QueryRequest(BaseModel):
    question: str
    folder: str = ""


class ProcessRequest(BaseModel):
    file: str = DEFAULT_PDF


class IndexRequest(BaseModel):
    directory: str = "data/documents"
    reset: bool = True


# --- Routes ---

@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/api/status")
async def status():
    """Check if the Milvus collection has data."""
    try:
        client = get_milvus_client()
        if client.has_collection(COLLECTION_NAME):
            stats = client.get_collection_stats(COLLECTION_NAME)
            count = stats.get("row_count", 0)
            return {"processed": count > 0, "chunk_count": count}
    except Exception:
        pass
    return {"processed": False, "chunk_count": 0}


@app.post("/api/process")
async def process_pdf(req: ProcessRequest):
    """Run the PDF processing pipeline."""
    file_path = Path(req.file)
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {req.file}")
    try:
        await asyncio.to_thread(pipeline, str(file_path))
        client = get_milvus_client()
        stats = client.get_collection_stats(COLLECTION_NAME)
        count = stats.get("row_count", 0)
        return {"ok": True, "chunks": count}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/index")
async def index_directory(req: IndexRequest):
    """Index all PDFs in a directory."""
    dir_path = Path(req.directory)
    if not dir_path.exists():
        raise HTTPException(404, f"Directory not found: {req.directory}")
    try:
        await asyncio.to_thread(pipeline_directory, str(dir_path), "screenshots", req.reset)
        client = get_milvus_client()
        stats = client.get_collection_stats(COLLECTION_NAME)
        count = stats.get("row_count", 0)
        pdf_count = len(list(dir_path.glob("**/*.pdf")))
        return {"ok": True, "files": pdf_count, "chunks": count}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/query")
async def query_agent(req: QueryRequest):
    """Stream agent execution steps via SSE."""
    if not req.question.strip():
        raise HTTPException(400, "Question is required")

    folder = req.folder.strip() if req.folder else "."

    async def event_stream():
        async for event in run_agent_stream(req.question, folder=folder):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/folders")
async def list_folders(path: str = "."):
    """List folders in the given path for the folder browser."""
    try:
        base = Path(path).resolve()
        if not base.exists() or not base.is_dir():
            raise HTTPException(404, "Path not found")
        folders = sorted(
            f.name for f in base.iterdir()
            if f.is_dir() and not f.name.startswith(".")
        )
        parent = str(base.parent) if base != base.parent else None
        file_count = len([f for f in base.iterdir() if f.is_file()])
        return {
            "current": str(base),
            "parent": parent,
            "folders": folders,
            "files_count": file_count,
        }
    except PermissionError:
        raise HTTPException(403, "Permission denied")


@app.get("/api/screenshot/{path:path}")
async def get_screenshot(path: str):
    """Serve screenshot images."""
    file_path = Path(path)
    # Security: only serve from screenshots directory
    if not str(file_path).startswith("screenshots/"):
        raise HTTPException(403, "Access denied")
    if not file_path.exists():
        raise HTTPException(404, "Screenshot not found")
    return FileResponse(str(file_path), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
