from pydantic import BaseModel, Field
from typing import Optional, Literal


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: str = Field(..., description="Path to the document file to ingest")
    chunking_strategy: Literal["fixed", "semantic", "sentence_window"] = Field(
        default="semantic",
        description="Chunking strategy to use for splitting the document",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional extra metadata to attach to all chunks from this document",
    )


class QueryRequest(BaseModel):
    """Request model for querying the RAG pipeline."""
    query: str = Field(..., description="The question to answer")
    top_k: Optional[int] = Field(
        default=None,
        description="Number of chunks to retrieve before reranking (overrides settings default)",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of chunks to keep after reranking (overrides settings default)",
    )
    detect_hallucinations: bool = Field(
        default=True,
        description="Whether to run hallucination detection on the generated answer",
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID for multi-turn memory continuity.  "
            "A new UUID is auto-generated if omitted."
        ),
    )
