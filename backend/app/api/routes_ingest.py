"""
backend/app/api/routes_ingest.py
---------------------------------
POST /ingest endpoint.

Accepts a file path (or file upload), runs it through the ingestion pipeline
(parse → chunk → index), and returns an ``IngestResponse``.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse


from app.config import settings
from app.ingestion import IndexManager, get_chunker, parse_document
from app.models import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Shared IndexManager (singleton-ish) ───────────────────────────────────────
# Instantiated once per process; can be replaced by proper DI in the future.

_index_manager = IndexManager(
    embedding_model=settings.embedding_model,
    persist_dir=settings.chroma_persist_dir,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _ingest_file(file_path: str, chunking_strategy: str, extra_metadata: dict | None) -> IngestResponse:
    """Run the full ingestion pipeline for a single file."""
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}",
        )

    logger.info("Parsing '%s' …", path.name)
    pages = parse_document(path)

    if not pages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No content could be extracted from '{path.name}'.",
        )

    # Attach any extra caller-supplied metadata to every page
    if extra_metadata:
        for page in pages:
            page.metadata.update(extra_metadata)

    chunker = get_chunker(
        strategy=chunking_strategy,  # type: ignore[arg-type]
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        similarity_threshold=settings.semantic_similarity_threshold,
        embedding_model=settings.embedding_model,
    )

    logger.info("Chunking with strategy '%s' …", chunking_strategy)
    chunks = chunker.chunk(pages)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Chunking produced 0 chunks — the document may be empty.",
        )

    logger.info("Indexing %d chunks …", len(chunks))
    _index_manager.add_documents(chunks)

    document_id = str(uuid.uuid4())
    logger.info(
        "Ingestion complete — doc_id=%s, chunks=%d, strategy=%s",
        document_id, len(chunks), chunking_strategy,
    )

    return IngestResponse(
        document_id=document_id,
        chunks_created=len(chunks),
        chunking_strategy=chunking_strategy,
        message="Document ingested successfully",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a document by file path",
    description=(
        "Parse, chunk, embed, and index a document located at *file_path* on "
        "the server's filesystem.  Use `POST /ingest/upload` to upload a file "
        "directly from the client."
    ),
)
async def ingest_by_path(request: IngestRequest) -> IngestResponse:
    """Ingest a document given its server-side file path."""
    return _ingest_file(
        file_path=request.file_path,
        chunking_strategy=request.chunking_strategy,
        extra_metadata=request.metadata,
    )


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest a document",
    description=(
        "Accept a multipart file upload, save it to a temporary location, "
        "run the ingestion pipeline, and return the result."
    ),
)
async def ingest_upload(
    file: UploadFile = File(...),
    chunking_strategy: str = settings.default_chunking,
) -> IngestResponse:
    """Upload a document and ingest it immediately."""
    # Persist to a temp directory under the configured chroma dir's parent
    tmp_dir = Path(settings.chroma_persist_dir).parent / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise the filename
    safe_name = Path(file.filename or "upload").name
    tmp_path = tmp_dir / safe_name

    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)

        response = _ingest_file(
            file_path=str(tmp_path),
            chunking_strategy=chunking_strategy,
            extra_metadata={"original_filename": safe_name},
        )
    finally:
        # Clean up the temp file after indexing
        if tmp_path.exists():
            os.remove(tmp_path)

    return response


@router.delete(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Clear all indexed documents",
    description="Delete all chunks from both the dense and sparse indexes.",
)
async def clear_index() -> JSONResponse:
    """Wipe the entire index (useful for testing / re-ingestion)."""
    _index_manager.clear()
    return JSONResponse(content={"message": "Index cleared."})


@router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Index statistics",
    description="Return chunk counts and index health information.",
)
async def index_stats() -> dict:
    """Return current index statistics."""
    return _index_manager.get_stats()
