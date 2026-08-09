from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Citation(BaseModel):
    """A single source citation linked to a chunk."""
    chunk_id: str = Field(..., description="ID of the chunk being cited")
    source: str = Field(..., description="Source document name/path")
    page_num: Optional[int] = Field(default=None, description="Page number within the source")
    section: Optional[str] = Field(default=None, description="Section heading within the source")
    snippet: str = Field(..., description="Short excerpt from the cited chunk")


class HallucinationFlag(BaseModel):
    """Hallucination detection result for a single claim in the answer."""
    claim: str = Field(..., description="The individual claim extracted from the answer")
    label: Literal["entailment", "neutral", "contradiction"] = Field(
        ..., description="NLI label for this claim vs its cited context"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from the NLI model")
    cited_chunk_id: Optional[str] = Field(
        default=None, description="The chunk ID this claim was verified against"
    )
    flagged: bool = Field(
        ..., description="True if this claim is considered a potential hallucination"
    )


class QueryResponse(BaseModel):
    """Full response returned by POST /query."""
    answer: str = Field(..., description="The generated answer grounded in context")
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of source citations supporting the answer",
    )
    hallucination_flags: List[HallucinationFlag] = Field(
        default_factory=list,
        description="Per-claim hallucination detection results (empty if detection was skipped)",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Overall confidence score (fraction of claims that are entailed)",
    )
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    # ── Agentic metadata (new optional fields, backward-compatible) ───────────
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID used for this query (echoed back for client continuity)",
    )
    retrieval_source: Optional[str] = Field(
        default=None,
        description=(
            "How evidence was sourced: 'documents', 'web', 'cached', 'hybrid', or 'fallback'"
        ),
    )
    agent_rounds: Optional[int] = Field(
        default=None,
        description="Number of orchestrator tool-call rounds executed",
    )


class IngestResponse(BaseModel):
    """Response returned by POST /ingest."""
    document_id: str = Field(..., description="Unique identifier assigned to the ingested document")
    chunks_created: int = Field(..., description="Number of chunks created from the document")
    chunking_strategy: str = Field(..., description="Chunking strategy that was used")
    message: str = Field(default="Document ingested successfully")


class EvalRunSummary(BaseModel):
    """Summary of a single evaluation run stored in results/."""
    filename: str = Field(..., description="Result file name")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the run")
    num_questions: int = Field(..., description="Number of questions evaluated")
    faithfulness: Optional[float] = Field(default=None, description="RAGAS Faithfulness score")
    answer_relevancy: Optional[float] = Field(default=None, description="RAGAS Answer Relevancy score")
    context_precision: Optional[float] = Field(default=None, description="RAGAS Context Precision score")
    context_recall: Optional[float] = Field(default=None, description="RAGAS Context Recall score")


class EvalResponse(BaseModel):
    """Full response returned by POST /eval/run."""
    timestamp: str = Field(..., description="ISO-8601 timestamp of this evaluation run")
    elapsed_seconds: float = Field(..., description="Total wall-clock time for the evaluation run")
    num_questions: int = Field(..., description="Number of questions evaluated")
    faithfulness: Optional[float] = Field(default=None, description="RAGAS Faithfulness score (0–1)")
    answer_relevancy: Optional[float] = Field(default=None, description="RAGAS Answer Relevancy score (0–1)")
    context_precision: Optional[float] = Field(default=None, description="RAGAS Context Precision score (0–1)")
    context_recall: Optional[float] = Field(default=None, description="RAGAS Context Recall score (0–1)")
    output_path: str = Field(..., description="Path to the saved result JSON file")
    config: dict = Field(default_factory=dict, description="Pipeline configuration used for this run")
