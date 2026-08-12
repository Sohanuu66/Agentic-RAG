"""
backend/evaluation/evaluate.py
--------------------------------
RAGAS-based evaluation of the full RAG pipeline.

Usage
-----
    # From the backend/ directory:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --output results/my_run.json
    python -m evaluation.evaluate --limit 10          # run only first N questions
    python -m evaluation.evaluate --golden evaluation/golden_qa.json

Class
-----
RAGASEvaluator
    Loads the golden Q&A dataset, runs each question through the full pipeline,
    computes RAGAS metrics, and saves timestamped results to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path when invoked as a module
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


class RAGASEvaluator:
    """
    Runs the full RAG pipeline on a golden Q&A dataset and computes RAGAS metrics.

    Parameters
    ----------
    golden_qa_path:
        Path to the golden_qa.json file.
    results_dir:
        Directory where timestamped result JSON files are written.
    embedding_model:
        SentenceTransformers model for dense retrieval.
    persist_dir:
        ChromaDB / BM25 persistence directory.
    llm_api_key:
        Groq API key for generation.
    llm_model:
        Groq model name.
    nli_model:
        HuggingFace model for hallucination detection.
    hallucination_threshold:
        Confidence threshold for flagging claims.
    retrieval_top_k:
        Number of candidates to retrieve before reranking.
    rerank_top_n:
        Number of final context chunks after reranking.
    rrf_k:
        Reciprocal Rank Fusion constant.
    reranker_model:
        Cross-encoder model for reranking.
    limit:
        If set, only evaluate the first N questions (useful for quick tests).
    openai_api_key:
        OpenAI API key required by RAGAS >= 0.2 for its LLM-based metrics.
        If not provided, only non-LLM metrics are computed.
    """

    def __init__(
        self,
        golden_qa_path: str = "./evaluation/golden_qa.json",
        results_dir: str = "./evaluation/results",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./data/chroma",
        llm_api_key: str = "",
        llm_model: str = "llama-3.3-70b-versatile",
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        hallucination_threshold: float = 0.7,
        retrieval_top_k: int = 20,
        rerank_top_n: int = 5,
        rrf_k: int = 60,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        limit: Optional[int] = None,
        openai_api_key: str = "",
    ) -> None:
        self.golden_qa_path = Path(golden_qa_path)
        self.results_dir = Path(results_dir)
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.nli_model = nli_model
        self.hallucination_threshold = hallucination_threshold
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n
        self.rrf_k = rrf_k
        self.reranker_model = reranker_model
        self.limit = limit
        self.openai_api_key = openai_api_key

        # Lazy pipeline components
        self._retriever = None
        self._reranker = None
        self._generator = None

    # ------------------------------------------------------------------
    # Pipeline component initialization
    # ------------------------------------------------------------------

    def _get_retriever(self):
        if self._retriever is None:
            from app.retrieval.hybrid import HybridRetriever
            self._retriever = HybridRetriever(
                embedding_model=self.embedding_model,
                persist_dir=self.persist_dir,
                rrf_k=self.rrf_k,
            )
            logger.info("HybridRetriever initialised.")
        return self._retriever

    def _get_reranker(self):
        if self._reranker is None:
            from app.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(model_name=self.reranker_model)
            logger.info("CrossEncoderReranker initialised.")
        return self._reranker

    def _get_generator(self):
        if self._generator is None:
            from app.generation.generator import CitationGenerator
            self._generator = CitationGenerator(
                api_key=self.llm_api_key,
                model=self.llm_model,
            )
            logger.info("CitationGenerator initialised.")
        return self._generator

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _run_pipeline(self, question: str) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a single question.

        Returns a dict with keys:
            answer, contexts (list of str), retrieved_chunk_ids
        """
        retriever = self._get_retriever()
        reranker = self._get_reranker()
        generator = self._get_generator()

        # 1. Retrieve
        candidates = retriever.retrieve(query=question, top_k=self.retrieval_top_k)

        if not candidates:
            return {
                "answer": "I couldn't find any relevant documents.",
                "contexts": [],
                "retrieved_chunk_ids": [],
            }

        # 2. Rerank
        try:
            reranked = reranker.rerank(
                query=question,
                candidates=candidates,
                top_n=self.rerank_top_n,
            )
        except Exception as exc:
            logger.warning("Reranking failed: %s — using raw retrieval.", exc)
            reranked = candidates[: self.rerank_top_n]

        # 3. Generate
        try:
            result = generator.generate(query=question, context_chunks=reranked)
            answer = result.answer
        except Exception as exc:
            logger.warning("Generation failed: %s", exc)
            answer = "Generation failed."

        contexts = [r.text for r in reranked]
        retrieved_chunk_ids = [r.chunk_id for r in reranked]

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_chunk_ids": retrieved_chunk_ids,
        }

    # ------------------------------------------------------------------
    # RAGAS evaluation
    # ------------------------------------------------------------------

    def _build_ragas_dataset(
        self,
        golden_qa: List[Dict],
        pipeline_outputs: List[Dict],
    ):
        """
        Build a RAGAS-compatible Dataset from golden Q&A and pipeline outputs.
        """
        from datasets import Dataset  # type: ignore[import]

        records = []
        for qa, output in zip(golden_qa, pipeline_outputs):
            records.append(
                {
                    "question": qa["question"],
                    "answer": output["answer"],
                    "contexts": output["contexts"],
                    "ground_truth": qa["ground_truth"],
                }
            )
        return Dataset.from_list(records)

    def _compute_ragas_metrics(self, dataset) -> Dict[str, float]:
        """
        Run RAGAS evaluation metrics on the dataset.

        Tries the RAGAS >= 0.2 API first (evaluate with LLM config),
        falls back to a simpler approach for older versions.
        """
        try:
            return self._compute_ragas_new_api(dataset)
        except Exception as exc:
            logger.warning(
                "RAGAS new API failed (%s). Falling back to legacy API.", exc
            )
            try:
                return self._compute_ragas_legacy_api(dataset)
            except Exception as exc2:
                logger.error("Both RAGAS API attempts failed: %s", exc2)
                return self._compute_stub_metrics()

    def _compute_ragas_new_api(self, dataset) -> Dict[str, float]:
        """RAGAS >= 0.2 API with explicit LLM config."""
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        # RAGAS >= 0.2 uses langchain LLM wrappers
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # Configure OpenAI-compatible LLM if key is provided
        if self.openai_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)

        logger.info("Running RAGAS evaluation on %d samples …", len(dataset))
        result = evaluate(dataset=dataset, metrics=metrics)

        scores: Dict[str, float] = {}
        result_dict = result.to_pandas().mean(numeric_only=True).to_dict()
        for key, val in result_dict.items():
            if isinstance(val, float):
                scores[key] = round(float(val), 4)

        return scores

    def _compute_ragas_legacy_api(self, dataset) -> Dict[str, float]:
        """RAGAS < 0.2 (0.1.x) API."""
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        logger.info("Running RAGAS (legacy API) on %d samples …", len(dataset))
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

        scores: Dict[str, float] = {}
        for key, val in result.items():
            if isinstance(val, (int, float)):
                scores[key] = round(float(val), 4)
        return scores

    def _compute_stub_metrics(self) -> Dict[str, float]:
        """
        Return placeholder metrics when RAGAS cannot be run (e.g., no LLM key).
        These are clearly marked as stubs.
        """
        logger.warning(
            "RAGAS could not be computed (possibly missing API key). "
            "Returning stub metrics — set OPENAI_API_KEY or LLM_API_KEY."
        )
        return {
            "faithfulness": -1.0,
            "answer_relevancy": -1.0,
            "context_precision": -1.0,
            "context_recall": -1.0,
            "_note": "RAGAS evaluation failed — stub values returned",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_golden_qa(self) -> List[Dict]:
        """Load and optionally truncate the golden Q&A dataset."""
        if not self.golden_qa_path.exists():
            raise FileNotFoundError(
                f"Golden Q&A file not found: {self.golden_qa_path}\n"
                "Ensure you have created backend/evaluation/golden_qa.json."
            )
        with open(self.golden_qa_path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded %d golden Q&A pairs from '%s'.", len(data), self.golden_qa_path)
        if self.limit:
            data = data[: self.limit]
            logger.info("Limiting to first %d questions.", self.limit)
        return data

    def run(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.

        1. Load golden Q&A.
        2. Run each question through the RAG pipeline.
        3. Compute RAGAS metrics.
        4. Save results to disk.

        Returns the result dict.
        """
        t_start = time.perf_counter()

        golden_qa = self.load_golden_qa()

        # ------------------------------------------------------------------
        # Step 1: Run pipeline on all questions
        # ------------------------------------------------------------------
        logger.info("Running pipeline on %d questions …", len(golden_qa))
        pipeline_outputs: List[Dict] = []
        for i, qa in enumerate(golden_qa):
            logger.info("[%d/%d] %s", i + 1, len(golden_qa), qa["question"][:80])
            try:
                output = self._run_pipeline(qa["question"])
            except Exception as exc:
                logger.warning("Pipeline failed for question %d: %s", i + 1, exc)
                output = {"answer": "", "contexts": [], "retrieved_chunk_ids": []}
            pipeline_outputs.append(output)

        # ------------------------------------------------------------------
        # Step 2: Build RAGAS dataset and compute metrics
        # ------------------------------------------------------------------
        logger.info("Building RAGAS dataset …")
        dataset = self._build_ragas_dataset(golden_qa, pipeline_outputs)

        logger.info("Computing RAGAS metrics …")
        scores = self._compute_ragas_metrics(dataset)

        elapsed_s = time.perf_counter() - t_start

        # ------------------------------------------------------------------
        # Step 3: Assemble and save results
        # ------------------------------------------------------------------
        per_question = []
        for qa, output in zip(golden_qa, pipeline_outputs):
            per_question.append(
                {
                    "id": qa.get("id", ""),
                    "question": qa["question"],
                    "ground_truth": qa["ground_truth"],
                    "answer": output["answer"],
                    "retrieved_chunk_ids": output["retrieved_chunk_ids"],
                    "category": qa.get("category", ""),
                    "difficulty": qa.get("difficulty", ""),
                }
            )

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "reranker_model": self.reranker_model,
                "retrieval_top_k": self.retrieval_top_k,
                "rerank_top_n": self.rerank_top_n,
                "rrf_k": self.rrf_k,
                "num_questions": len(golden_qa),
            },
            "aggregate_scores": scores,
            "elapsed_seconds": round(elapsed_s, 2),
            "per_question": per_question,
        }

        # Save to disk
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if output_path:
            out_path = Path(output_path)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = self.results_dir / f"eval_{ts}.json"

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

        logger.info(
            "Evaluation complete. Scores: %s | Saved to: %s",
            scores,
            out_path,
        )
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.evaluate",
        description="Run RAGAS evaluation on the Ask My Docs golden Q&A dataset.",
    )
    parser.add_argument(
        "--golden",
        default=None,
        help="Path to golden_qa.json (default: reads from .env / settings)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: auto-timestamped in evaluation/results/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="ChromaDB persistence directory (default: from .env)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # Load settings from .env
    from app.config import settings  # type: ignore[import]

    evaluator = RAGASEvaluator(
        golden_qa_path=args.golden or settings.golden_qa_path,
        results_dir=settings.eval_results_dir,
        embedding_model=settings.embedding_model,
        persist_dir=args.persist_dir or settings.chroma_persist_dir,
        llm_api_key=settings.openai_api_key,
        llm_model=settings.llm_model,
        nli_model=settings.nli_model,
        hallucination_threshold=settings.hallucination_threshold,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_n=settings.rerank_top_n,
        rrf_k=settings.rrf_k,
        reranker_model=settings.reranker_model,
        limit=args.limit,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    result = evaluator.run(output_path=args.output)

    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    for metric, value in result["aggregate_scores"].items():
        if isinstance(value, float):
            print(f"  {metric:<30} {value:.4f}")
        else:
            print(f"  {metric:<30} {value}")
    print(f"\n  Elapsed: {result['elapsed_seconds']:.1f}s")
    print(f"  Questions: {result['config']['num_questions']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
