"""
backend/app/api/routes_query.py
---------------------------------
POST /query endpoint.

Pipeline (agentic):
    QueryRequest
        → orchestrate()          [OpenAI tool-calling loop]
            ├─ reuse_cached_evidence  (SQLite session memory)
            ├─ retrieve_documents     (HybridRetriever + CrossEncoderReranker)
            └─ web_search             (Tavily)
        → CitationGenerator.generate(query, context_chunks)
        → HallucinationDetector.detect(answer, citations, context_chunks)
        → QueryResponse

The orchestrator has an internal fallback to direct retrieve→rerank if the
tool-calling loop fails or times out.  The /query endpoint always returns
a QueryResponse — 5xx errors are only raised for generation failures.

The endpoint is wired into FastAPI via ``app.main`` and exported from
``app.api.__init__``.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.generation import CitationGenerator, GenerationResult
from app.hallucination import HallucinationDetector
from app.models import QueryRequest, QueryResponse
from app.orchestrator import OrchestratorResult, orchestrate
from app.retrieval import CrossEncoderReranker, HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Singleton pipeline components ────────────────────────────────────────────
# Instantiated once per process; lazy model loading keeps startup fast.

_hybrid_retriever = HybridRetriever(
    embedding_model=settings.embedding_model,
    persist_dir=settings.chroma_persist_dir,
    rrf_k=settings.rrf_k,
)

_reranker = CrossEncoderReranker(model_name=settings.reranker_model)

_generator = CitationGenerator(
    api_key=settings.openai_api_key,
    model=settings.llm_model,
)

_detector = HallucinationDetector(
    model_name=settings.nli_model,
    threshold=settings.hallucination_threshold,
)


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about the indexed documents",
    description=(
        "Run the agentic RAG pipeline: OpenAI tool-calling orchestrator "
        "(hybrid retrieval, cross-encoder reranking, optional Tavily web search, "
        "SQLite session memory) → citation-grounded generation → hallucination detection. "
        "Falls back to direct retrieve→rerank internally if the agentic loop fails."
    ),
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    End-to-end agentic RAG query endpoint.

    1. Orchestrator: OpenAI tool-calling loop gathers evidence via:
       a) reuse_cached_evidence (SQLite session memory)
       b) retrieve_documents (HybridRetriever + CrossEncoderReranker)
       c) web_search (Tavily, if document evidence is weak or absent)
    2. Citation-grounded generation via OpenAI (plain-text + JSON extraction).
    3. Claim-level hallucination detection via NLI cross-encoder.
    """
    t_start = time.perf_counter()

    session_id = request.session_id or str(uuid.uuid4())
    top_k = request.top_k or settings.retrieval_top_k
    top_n = request.top_n or settings.rerank_top_n

    # ── 1. Agentic evidence gathering ─────────────────────────────────────────
    logger.info(
        "Query pipeline [session=%s]: starting orchestrator (max_rounds=%d) …",
        session_id,
        settings.agent_max_rounds,
    )
    orch_result: OrchestratorResult = await orchestrate(
        query=request.query,
        session_id=session_id,
        retriever=_hybrid_retriever,
        reranker=_reranker,
        top_k=top_k,
        top_n=top_n,
    )

    if not orch_result.chunks:
        logger.warning(
            "No evidence gathered for query: %r (session=%s, source=%s)",
            request.query, session_id, orch_result.retrieval_source,
        )
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return QueryResponse(
            answer="I couldn't find any relevant documents. Please ingest some documents first.",
            citations=[],
            hallucination_flags=[],
            confidence=None,
            latency_ms=total_ms,
            session_id=session_id,
            retrieval_source=orch_result.retrieval_source,
            agent_rounds=orch_result.agent_rounds,
        )

    # ── 2. Citation-grounded generation ──────────────────────────────────────
    logger.info(
        "Query pipeline: generating answer from %d evidence chunks …",
        len(orch_result.chunks),
    )
    try:
        result: GenerationResult = _generator.generate(
            query=request.query,
            context_chunks=orch_result.chunks,
        )
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Generation failed: {exc}",
        ) from exc

    # ── 3. Hallucination detection ────────────────────────────────────────────
    hallucination_flags: list = []
    confidence: float | None = None

    if request.detect_hallucinations and result.answer:
        logger.info("Query pipeline: running hallucination detection …")
        try:
            hallucination_flags = _detector.detect(
                answer=result.answer,
                citations=result.citations,
                context_chunks=orch_result.chunks,
            )
            if hallucination_flags:
                entailed = sum(
                    1 for f in hallucination_flags if f.label == "entailment"
                )
                confidence = round(entailed / len(hallucination_flags), 4)
        except Exception as exc:
            logger.warning(
                "Hallucination detection failed (%s) — returning empty flags.", exc
            )

    total_ms = (time.perf_counter() - t_start) * 1000.0

    flagged_count = sum(1 for f in hallucination_flags if f.flagged)
    logger.info(
        "Query pipeline complete: latency=%.1f ms, citations=%d, "
        "hallucination_flags=%d (flagged=%d), source=%s, rounds=%d.",
        total_ms,
        len(result.citations),
        len(hallucination_flags),
        flagged_count,
        orch_result.retrieval_source,
        orch_result.agent_rounds,
    )

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        hallucination_flags=hallucination_flags,
        confidence=confidence,
        latency_ms=total_ms,
        session_id=session_id,
        retrieval_source=orch_result.retrieval_source,
        agent_rounds=orch_result.agent_rounds,
    )
