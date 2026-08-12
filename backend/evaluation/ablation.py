"""
backend/evaluation/ablation.py
--------------------------------
Ablation study runner for the Ask My Docs RAG pipeline.

Sweeps two experiment dimensions:
    1. Chunking strategy: fixed-256, fixed-512, semantic, sentence_window
    2. Retrieval method: dense-only, bm25-only, hybrid (RRF), hybrid+rerank

For each configuration, it:
    - Re-indexes the corpus with the new settings
    - Runs the full evaluation pipeline
    - Logs RAGAS scores
    - Outputs a comparison table (markdown + JSON)

Usage
-----
    # From the backend/ directory:
    python -m evaluation.ablation --experiment chunking
    python -m evaluation.ablation --experiment retrieval
    python -m evaluation.ablation --experiment all
    python -m evaluation.ablation --experiment chunking --corpus ./data/corpus --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


# ---------------------------------------------------------------------------
# Experiment configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    name: str
    strategy: str
    chunk_size: int
    chunk_overlap: int
    semantic_threshold: float = 0.75


@dataclass
class RetrievalConfig:
    name: str
    retrieval_mode: str          # "dense" | "sparse" | "hybrid"
    use_reranking: bool = True
    retrieval_top_k: int = 20
    rerank_top_n: int = 5


# Pre-defined experiment configurations
CHUNKING_CONFIGS: List[ChunkingConfig] = [
    ChunkingConfig("fixed-256", strategy="fixed", chunk_size=256, chunk_overlap=25),
    ChunkingConfig("fixed-512", strategy="fixed", chunk_size=512, chunk_overlap=50),
    ChunkingConfig("semantic", strategy="semantic", chunk_size=512, chunk_overlap=0, semantic_threshold=0.75),
    ChunkingConfig("sentence_window", strategy="sentence_window", chunk_size=3, chunk_overlap=0),
]

RETRIEVAL_CONFIGS: List[RetrievalConfig] = [
    RetrievalConfig("dense-only", retrieval_mode="dense", use_reranking=False),
    RetrievalConfig("bm25-only", retrieval_mode="sparse", use_reranking=False),
    RetrievalConfig("hybrid-no-rerank", retrieval_mode="hybrid", use_reranking=False),
    RetrievalConfig("hybrid-rerank", retrieval_mode="hybrid", use_reranking=True),
]


# ---------------------------------------------------------------------------
# AblationRetriever – supports dense-only / sparse-only / hybrid modes
# ---------------------------------------------------------------------------

class AblationRetriever:
    """
    Wraps the retrieval components to support dense-only, sparse-only, or hybrid modes.
    """

    def __init__(
        self,
        retrieval_mode: str,
        use_reranking: bool,
        embedding_model: str,
        persist_dir: str,
        rrf_k: int = 60,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 20,
        top_n: int = 5,
    ) -> None:
        self.retrieval_mode = retrieval_mode
        self.use_reranking = use_reranking
        self.top_k = top_k
        self.top_n = top_n

        from app.retrieval.dense import DenseRetriever
        from app.retrieval.sparse import SparseRetriever
        from app.retrieval.hybrid import HybridRetriever

        if retrieval_mode == "dense":
            self._retriever = DenseRetriever(
                embedding_model=embedding_model, persist_dir=persist_dir
            )
        elif retrieval_mode == "sparse":
            self._retriever = SparseRetriever(persist_dir=persist_dir)
        else:
            self._retriever = HybridRetriever(
                embedding_model=embedding_model,
                persist_dir=persist_dir,
                rrf_k=rrf_k,
            )

        self._reranker = None
        if use_reranking:
            from app.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(model_name=reranker_model)

    def retrieve(self, query: str):
        candidates = self._retriever.retrieve(query=query, top_k=self.top_k)
        if self._reranker and candidates:
            try:
                candidates = self._reranker.rerank(
                    query=query, candidates=candidates, top_n=self.top_n
                )
            except Exception as exc:
                logger.warning("Reranker failed: %s — using raw results.", exc)
                candidates = candidates[: self.top_n]
        else:
            candidates = candidates[: self.top_n]
        return candidates


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """
    Runs ablation experiments across chunking and retrieval configurations.

    Parameters
    ----------
    corpus_dir:
        Directory containing corpus documents to ingest.
    golden_qa_path:
        Path to golden_qa.json.
    results_dir:
        Directory for writing ablation results.
    base_persist_dir:
        Base path for ChromaDB / BM25 storage; each config gets its own subdir.
    embedding_model:
        SentenceTransformers embedding model.
    llm_api_key:
        Groq API key.
    llm_model:
        Groq model name.
    reranker_model:
        Cross-encoder reranker model.
    limit:
        Limit evaluation to first N questions.
    openai_api_key:
        OpenAI key for RAGAS LLM-based metrics.
    """

    def __init__(
        self,
        corpus_dir: str = "./data/corpus",
        golden_qa_path: str = "./evaluation/golden_qa.json",
        results_dir: str = "./evaluation/results",
        base_persist_dir: str = "./data/ablation_chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_api_key: str = "",
        llm_model: str = "llama-3.3-70b-versatile",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
        limit: Optional[int] = None,
        openai_api_key: str = "",
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.golden_qa_path = Path(golden_qa_path)
        self.results_dir = Path(results_dir)
        self.base_persist_dir = Path(base_persist_dir)
        self.embedding_model = embedding_model
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.reranker_model = reranker_model
        self.rrf_k = rrf_k
        self.limit = limit
        self.openai_api_key = openai_api_key

    # ------------------------------------------------------------------
    # Corpus ingestion helpers
    # ------------------------------------------------------------------

    def _ingest_corpus(self, chunking_cfg: ChunkingConfig, persist_dir: Path) -> int:
        """
        Ingest all documents in corpus_dir with the given chunking config.
        Returns the number of chunks created.
        """
        from app.ingestion.parser import parse_document
        from app.ingestion.chunker import get_chunker
        from app.ingestion.indexer import IndexManager

        if not self.corpus_dir.exists() or not any(self.corpus_dir.iterdir()):
            logger.warning(
                "Corpus directory '%s' is empty — skipping ingestion.", self.corpus_dir
            )
            return 0

        chunker = get_chunker(
            strategy=chunking_cfg.strategy,
            chunk_size=chunking_cfg.chunk_size,
            overlap=chunking_cfg.chunk_overlap,
            similarity_threshold=chunking_cfg.semantic_threshold,
        )
        index_manager = IndexManager(
            embedding_model=self.embedding_model,
            persist_dir=str(persist_dir),
        )
        # Clear existing index for this config
        index_manager.clear()

        total_chunks = []
        for doc_path in self.corpus_dir.iterdir():
            if doc_path.is_file():
                try:
                    pages = parse_document(str(doc_path))
                    chunks = chunker.chunk(pages)
                    total_chunks.extend(chunks)
                    logger.info(
                        "Ingested '%s': %d chunks.", doc_path.name, len(chunks)
                    )
                except Exception as exc:
                    logger.warning("Failed to ingest '%s': %s", doc_path.name, exc)

        if total_chunks:
            index_manager.add_documents(total_chunks)

        logger.info(
            "Ingestion complete for config '%s': %d total chunks.",
            chunking_cfg.name,
            len(total_chunks),
        )
        return len(total_chunks)

    # ------------------------------------------------------------------
    # Single-config evaluation
    # ------------------------------------------------------------------

    def _evaluate_config(
        self,
        golden_qa: List[Dict],
        retriever: AblationRetriever,
        config_label: str,
    ) -> Dict[str, Any]:
        """Run the pipeline and compute RAGAS scores for one configuration."""
        from app.generation.generator import CitationGenerator
        from evaluation.evaluate import RAGASEvaluator

        generator = CitationGenerator(api_key=self.llm_api_key, model=self.llm_model)

        pipeline_outputs = []
        for i, qa in enumerate(golden_qa):
            logger.info(
                "[%s] [%d/%d] %s", config_label, i + 1, len(golden_qa), qa["question"][:60]
            )
            try:
                candidates = retriever.retrieve(qa["question"])
                result = generator.generate(
                    query=qa["question"], context_chunks=candidates
                )
                output = {
                    "answer": result.answer,
                    "contexts": [c.text for c in candidates],
                    "retrieved_chunk_ids": [c.chunk_id for c in candidates],
                }
            except Exception as exc:
                logger.warning("Pipeline error for Q%d: %s", i + 1, exc)
                output = {"answer": "", "contexts": [], "retrieved_chunk_ids": []}
            pipeline_outputs.append(output)

        # Build RAGAS dataset
        dummy_evaluator = RAGASEvaluator(openai_api_key=self.openai_api_key)
        dataset = dummy_evaluator._build_ragas_dataset(golden_qa, pipeline_outputs)
        scores = dummy_evaluator._compute_ragas_metrics(dataset)

        return {"config": config_label, "scores": scores}

    # ------------------------------------------------------------------
    # Experiment runners
    # ------------------------------------------------------------------

    def run_chunking_experiment(self) -> List[Dict[str, Any]]:
        """
        For each chunking config:
            1. Ingest corpus with that strategy.
            2. Evaluate with hybrid+rerank retrieval.
        Returns a list of result dicts.
        """
        golden_qa = self._load_golden_qa()
        results = []

        for cfg in CHUNKING_CONFIGS:
            logger.info("=" * 60)
            logger.info("CHUNKING ABLATION: %s", cfg.name)
            logger.info("=" * 60)

            persist_dir = self.base_persist_dir / f"chunking_{cfg.name}"

            chunk_count = self._ingest_corpus(cfg, persist_dir)
            if chunk_count == 0:
                logger.warning(
                    "Skipping eval for '%s' — no chunks in index.", cfg.name
                )
                results.append(
                    {
                        "config": cfg.name,
                        "chunking": asdict(cfg),
                        "chunk_count": 0,
                        "scores": {},
                        "error": "no chunks indexed",
                    }
                )
                continue

            retriever = AblationRetriever(
                retrieval_mode="hybrid",
                use_reranking=True,
                embedding_model=self.embedding_model,
                persist_dir=str(persist_dir),
                rrf_k=self.rrf_k,
                reranker_model=self.reranker_model,
            )

            eval_result = self._evaluate_config(
                golden_qa=golden_qa,
                retriever=retriever,
                config_label=cfg.name,
            )
            eval_result["chunking"] = asdict(cfg)
            eval_result["chunk_count"] = chunk_count
            results.append(eval_result)

        return results

    def run_retrieval_experiment(self) -> List[Dict[str, Any]]:
        """
        Ingest corpus once with fixed-512 chunking, then evaluate each retrieval config.
        Returns a list of result dicts.
        """
        golden_qa = self._load_golden_qa()

        # Ingest once with the default/best chunking config
        default_cfg = ChunkingConfig("fixed-512", strategy="fixed", chunk_size=512, chunk_overlap=50)
        persist_dir = self.base_persist_dir / "retrieval_base"
        chunk_count = self._ingest_corpus(default_cfg, persist_dir)

        if chunk_count == 0:
            logger.warning("No corpus documents found — retrieval ablation cannot run.")
            return [
                {
                    "config": rc.name,
                    "retrieval": asdict(rc),
                    "scores": {},
                    "error": "no chunks indexed",
                }
                for rc in RETRIEVAL_CONFIGS
            ]

        results = []
        for cfg in RETRIEVAL_CONFIGS:
            logger.info("=" * 60)
            logger.info("RETRIEVAL ABLATION: %s", cfg.name)
            logger.info("=" * 60)

            retriever = AblationRetriever(
                retrieval_mode=cfg.retrieval_mode,
                use_reranking=cfg.use_reranking,
                embedding_model=self.embedding_model,
                persist_dir=str(persist_dir),
                rrf_k=self.rrf_k,
                reranker_model=self.reranker_model,
                top_k=cfg.retrieval_top_k,
                top_n=cfg.rerank_top_n,
            )

            eval_result = self._evaluate_config(
                golden_qa=golden_qa,
                retriever=retriever,
                config_label=cfg.name,
            )
            eval_result["retrieval"] = asdict(cfg)
            eval_result["chunk_count"] = chunk_count
            results.append(eval_result)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_golden_qa(self) -> List[Dict]:
        with open(self.golden_qa_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if self.limit:
            data = data[: self.limit]
            logger.info("Golden Q&A limited to first %d questions.", self.limit)
        return data

    @staticmethod
    def _format_comparison_table(results: List[Dict], experiment: str) -> str:
        """Format ablation results as a markdown table."""
        lines = []
        lines.append(f"\n## Ablation Results: {experiment}\n")

        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        header = "| Config | " + " | ".join(metrics) + " |"
        sep = "|---|" + "---|" * len(metrics)
        lines.append(header)
        lines.append(sep)

        for res in results:
            scores = res.get("scores", {})
            row_vals = []
            for m in metrics:
                v = scores.get(m, "N/A")
                row_vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append(f"| {res['config']} | " + " | ".join(row_vals) + " |")

        return "\n".join(lines)

    def save_results(
        self,
        results: List[Dict],
        experiment: str,
        output_path: Optional[str] = None,
    ) -> Path:
        """Save ablation results to JSON and print the markdown table."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if output_path:
            out_path = Path(output_path)
        else:
            out_path = self.results_dir / f"ablation_{experiment}_{ts}.json"

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        table = self._format_comparison_table(results, experiment)
        print(table)
        logger.info("Ablation results saved to '%s'.", out_path)
        return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.ablation",
        description="Run ablation studies for chunking and/or retrieval configurations.",
    )
    parser.add_argument(
        "--experiment",
        choices=["chunking", "retrieval", "all"],
        default="all",
        help="Which ablation to run (default: all)",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to the corpus directory (default: from .env)",
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
        help="Limit evaluation to first N questions",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    import os
    args = _parse_args(argv)

    from app.config import settings  # type: ignore[import]

    runner = AblationRunner(
        corpus_dir=args.corpus or "./data/corpus",
        golden_qa_path=settings.golden_qa_path,
        results_dir=settings.eval_results_dir,
        base_persist_dir="./data/ablation_chroma",
        embedding_model=settings.embedding_model,
        llm_api_key=settings.openai_api_key,
        llm_model=settings.llm_model,
        reranker_model=settings.reranker_model,
        rrf_k=settings.rrf_k,
        limit=args.limit,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    if args.experiment in ("chunking", "all"):
        logger.info("Starting chunking ablation …")
        t0 = time.perf_counter()
        chunking_results = runner.run_chunking_experiment()
        runner.save_results(chunking_results, "chunking", output_path=args.output)
        logger.info("Chunking ablation done in %.1fs.", time.perf_counter() - t0)

    if args.experiment in ("retrieval", "all"):
        logger.info("Starting retrieval ablation …")
        t0 = time.perf_counter()
        retrieval_results = runner.run_retrieval_experiment()
        runner.save_results(retrieval_results, "retrieval", output_path=args.output)
        logger.info("Retrieval ablation done in %.1fs.", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
