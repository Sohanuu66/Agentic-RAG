"""
backend/app/api/routes_eval.py
--------------------------------
Evaluation API endpoints.

Routes
------
POST /eval/run
    Trigger an evaluation run on the golden Q&A dataset.
    Returns aggregate RAGAS scores and a summary.

GET /eval/results
    List all historical evaluation result files with their aggregate scores.

GET /eval/ablation/{experiment}
    Return the most recent ablation study results for 'chunking' or 'retrieval'.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EvalRunRequest(BaseModel):
    """Request body for POST /eval/run."""
    limit: Optional[int] = Field(
        default=None,
        description="Limit evaluation to first N questions (omit for full eval)",
        ge=1,
    )
    golden_qa_path: Optional[str] = Field(
        default=None,
        description="Override path to golden_qa.json (default: from .env)",
    )


class AggregateScores(BaseModel):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    note: Optional[str] = Field(default=None, alias="_note")

    model_config = {"populate_by_name": True}


class EvalRunResponse(BaseModel):
    """Response from POST /eval/run."""
    timestamp: str
    elapsed_seconds: float
    num_questions: int
    aggregate_scores: Dict[str, Any]
    output_path: str
    config: Dict[str, Any]


class EvalResultSummary(BaseModel):
    """Summary entry for GET /eval/results."""
    filename: str
    timestamp: str
    num_questions: int
    aggregate_scores: Dict[str, Any]
    config: Dict[str, Any]


class EvalResultsResponse(BaseModel):
    """Response from GET /eval/results."""
    count: int
    results: List[EvalResultSummary]


class AblationEntry(BaseModel):
    config: str
    scores: Dict[str, Any]
    extra: Dict[str, Any] = Field(default_factory=dict)


class AblationResponse(BaseModel):
    """Response from GET /eval/ablation/{experiment}."""
    experiment: str
    timestamp: str
    result_count: int
    results: List[Dict[str, Any]]
    comparison_table: str


# ---------------------------------------------------------------------------
# Helper: load results directory
# ---------------------------------------------------------------------------

def _results_dir() -> Path:
    return Path(settings.eval_results_dir)


def _list_result_files() -> List[Path]:
    """Return all eval_*.json files sorted newest first."""
    d = _results_dir()
    if not d.exists():
        return []
    files = sorted(d.glob("eval_*.json"), reverse=True)
    return files


def _list_ablation_files(experiment: str) -> List[Path]:
    """Return all ablation_<experiment>_*.json files sorted newest first."""
    d = _results_dir()
    if not d.exists():
        return []
    files = sorted(d.glob(f"ablation_{experiment}_*.json"), reverse=True)
    return files


def _format_markdown_table(results: List[Dict], experiment: str) -> str:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    lines = [
        f"## Ablation: {experiment}",
        "| Config | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for res in results:
        scores = res.get("scores", {})
        vals = []
        for m in metrics:
            v = scores.get(m)
            vals.append(f"{v:.4f}" if isinstance(v, float) else "N/A")
        lines.append(f"| {res.get('config', '?')} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/run",
    response_model=EvalRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run RAGAS evaluation on the golden Q&A dataset",
    description=(
        "Triggers a full evaluation run: runs each golden Q&A question through "
        "the RAG pipeline and computes Faithfulness, Answer Relevancy, "
        "Context Precision, and Context Recall using RAGAS. "
        "This is a synchronous endpoint — expect it to take several minutes for a full run. "
        "Use the `limit` parameter for a quick smoke test."
    ),
)
async def run_evaluation(request: EvalRunRequest) -> EvalRunResponse:
    """
    Trigger an evaluation run.

    Loads the full pipeline (retriever, reranker, generator) and runs every
    question in golden_qa.json through it, then computes RAGAS metrics.
    """
    logger.info(
        "Evaluation run requested (limit=%s).", request.limit
    )

    try:
        from evaluation.evaluate import RAGASEvaluator  # type: ignore[import]
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not import evaluation module: {exc}",
        ) from exc

    evaluator = RAGASEvaluator(
        golden_qa_path=request.golden_qa_path or settings.golden_qa_path,
        results_dir=settings.eval_results_dir,
        embedding_model=settings.embedding_model,
        persist_dir=settings.chroma_persist_dir,
        llm_api_key=settings.openai_api_key,
        llm_model=settings.llm_model,
        nli_model=settings.nli_model,
        hallucination_threshold=settings.hallucination_threshold,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_n=settings.rerank_top_n,
        rrf_k=settings.rrf_k,
        reranker_model=settings.reranker_model,
        limit=request.limit,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    try:
        result = evaluator.run()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Evaluation run failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}",
        ) from exc

    # Determine output path from results_dir (last written file)
    result_files = _list_result_files()
    out_path = str(result_files[0]) if result_files else "unknown"

    return EvalRunResponse(
        timestamp=result["timestamp"],
        elapsed_seconds=result["elapsed_seconds"],
        num_questions=result["config"]["num_questions"],
        aggregate_scores=result["aggregate_scores"],
        output_path=out_path,
        config=result["config"],
    )


@router.get(
    "/results",
    response_model=EvalResultsResponse,
    status_code=status.HTTP_200_OK,
    summary="List historical evaluation results",
    description="Returns a list of all past evaluation runs with aggregate RAGAS scores.",
)
async def list_results(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of results to return"),
) -> EvalResultsResponse:
    """List historical RAGAS evaluation results ordered newest first."""
    files = _list_result_files()[:limit]

    summaries: List[EvalResultSummary] = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            summaries.append(
                EvalResultSummary(
                    filename=f.name,
                    timestamp=data.get("timestamp", ""),
                    num_questions=data.get("config", {}).get("num_questions", 0),
                    aggregate_scores=data.get("aggregate_scores", {}),
                    config=data.get("config", {}),
                )
            )
        except Exception as exc:
            logger.warning("Could not parse result file '%s': %s", f.name, exc)

    return EvalResultsResponse(count=len(summaries), results=summaries)


@router.get(
    "/ablation/{experiment}",
    response_model=AblationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get ablation study results",
    description=(
        "Returns the most recent ablation comparison table for the given experiment. "
        "Valid experiments: 'chunking', 'retrieval'."
    ),
)
async def get_ablation_results(
    experiment: str,
) -> AblationResponse:
    """Return the most recent ablation result for 'chunking' or 'retrieval'."""
    if experiment not in ("chunking", "retrieval"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid experiment '{experiment}'. Choose from: chunking, retrieval",
        )

    files = _list_ablation_files(experiment)
    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No ablation results found for experiment '{experiment}'. "
                f"Run: python -m evaluation.ablation --experiment {experiment}"
            ),
        )

    # Load the most recent file
    with open(files[0], encoding="utf-8") as fh:
        data = json.load(fh)

    results = data.get("results", [])
    table = _format_markdown_table(results, experiment)

    return AblationResponse(
        experiment=experiment,
        timestamp=data.get("timestamp", ""),
        result_count=len(results),
        results=results,
        comparison_table=table,
    )


@router.get(
    "/baseline",
    status_code=status.HTTP_200_OK,
    summary="Get the current evaluation baseline",
    description="Returns the stored baseline RAGAS scores used for regression gating.",
)
async def get_baseline() -> Dict[str, Any]:
    """Return the current baseline.json."""
    baseline_path = Path(settings.baseline_path)
    if not baseline_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Baseline file not found at '{baseline_path}'.",
        )
    with open(baseline_path, encoding="utf-8") as fh:
        return json.load(fh)
