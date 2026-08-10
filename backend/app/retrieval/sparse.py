"""
backend/app/retrieval/sparse.py
--------------------------------
Sparse (keyword) retrieval using BM25Okapi loaded from a pickle file.

The pickle format produced by ``build_sparse_index`` in
``app.ingestion.indexer`` is::

    {
        "bm25":      BM25Okapi,
        "chunk_ids": List[str],
        "texts":     List[str],
        "metadatas": List[dict],
    }
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List

from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)

_BM25_PICKLE_NAME = "bm25_index.pkl"


class SparseRetriever:
    """
    BM25-based keyword retriever.

    Parameters
    ----------
    persist_dir:
        Directory where the BM25 pickle file (``bm25_index.pkl``) lives.
        Must match ``persist_dir`` used when building the sparse index.
    """

    def __init__(self, persist_dir: str = "./data/chroma") -> None:
        self.persist_dir = persist_dir

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        """
        Load and return the BM25 pickle payload.

        Raises
        ------
        FileNotFoundError
            If the pickle file does not exist (no documents indexed yet).
        """
        pickle_path = Path(self.persist_dir) / _BM25_PICKLE_NAME
        if not pickle_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{pickle_path}'. "
                "Ingest documents first via POST /ingest."
            )
        with open(pickle_path, "rb") as fh:
            return pickle.load(fh)

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """
        Tokenize *query* and return the top-*k* BM25-scored chunks.

        Parameters
        ----------
        query:
            The natural-language query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List[RetrievalResult]
            Results ordered from highest to lowest BM25 score.
            An empty list is returned when the index is empty or missing.
        """
        try:
            payload = self._load_index()
        except FileNotFoundError as exc:
            logger.warning("SparseRetriever: %s — returning empty results.", exc)
            return []

        bm25 = payload["bm25"]
        chunk_ids: List[str] = payload["chunk_ids"]
        texts: List[str] = payload["texts"]
        metadatas: List[dict] = payload.get("metadatas", [{} for _ in chunk_ids])

        if not chunk_ids:
            logger.warning("SparseRetriever: BM25 index is empty.")
            return []

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)  # numpy array, len == len(chunk_ids)

        # Pair (index, score) and sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results: List[RetrievalResult] = []
        for idx, score in top:
            results.append(
                RetrievalResult(
                    chunk_id=chunk_ids[idx],
                    text=texts[idx],
                    metadata=metadatas[idx] if idx < len(metadatas) else {},
                    score=float(score),
                )
            )

        logger.debug(
            "SparseRetriever: returned %d results for query '%s …'",
            len(results),
            query[:50],
        )
        return results
