"""
backend/app/retrieval/hybrid.py
---------------------------------
Hybrid retrieval: fuse dense + sparse results with Reciprocal Rank Fusion (RRF).

Key references
--------------
* Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods" (SIGIR 2009).
* RRF formula: score(d) = Σ_r  1 / (k + rank_r(d))

Classes
-------
HybridRetriever
    Runs both DenseRetriever and SparseRetriever, fuses their ranked lists
    with RRF, deduplicates by chunk_id, and returns a merged ranking.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from app.retrieval.dense import DenseRetriever, RetrievalResult
from app.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: Sequence[List[RetrievalResult]],
    k: int = 60,
) -> List[RetrievalResult]:
    """
    Fuse multiple ranked result lists into one using Reciprocal Rank Fusion.

    Parameters
    ----------
    result_lists:
        One or more ordered lists of ``RetrievalResult``.  Each list is
        treated as one "voter".  Earlier positions (lower index) are
        considered more relevant.
    k:
        RRF constant that controls the influence of high-ranked documents.
        The standard value is 60 (Cormack et al., 2009).

    Returns
    -------
    List[RetrievalResult]
        Deduplicated list sorted by descending RRF score.
        For each unique ``chunk_id`` the ``text`` and ``metadata`` from
        the first occurrence are preserved; the ``score`` is the RRF score.

    Notes
    -----
    RRF formula for a document *d* across ranked lists *R*::

        rrf_score(d) = Σ_{r ∈ R}  1 / (k + rank_r(d))

    Ranks are 1-indexed.
    """
    # chunk_id → accumulated RRF score
    rrf_scores: Dict[str, float] = {}
    # chunk_id → RetrievalResult (for text/metadata reconstruction)
    registry: Dict[str, RetrievalResult] = {}

    for ranked_list in result_lists:
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in registry:
                registry[cid] = result

    # Build final list sorted by descending RRF score
    fused: List[RetrievalResult] = []
    for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        orig = registry[cid]
        fused.append(
            RetrievalResult(
                chunk_id=cid,
                text=orig.text,
                metadata=orig.metadata,
                score=rrf_score,
            )
        )

    return fused


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Runs dense and sparse retrieval in parallel and fuses results with RRF.

    Parameters
    ----------
    embedding_model:
        SentenceTransformers model for dense retrieval.
    persist_dir:
        ChromaDB / BM25 persistence directory.
    rrf_k:
        RRF constant (default 60).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./data/chroma",
        rrf_k: int = 60,
    ) -> None:
        self.rrf_k = rrf_k
        self._dense = DenseRetriever(
            embedding_model=embedding_model,
            persist_dir=persist_dir,
        )
        self._sparse = SparseRetriever(persist_dir=persist_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """
        Retrieve and fuse candidates from both dense and sparse retrievers.

        Parameters
        ----------
        query:
            The natural-language query string.
        top_k:
            Number of fused results to return.

        Returns
        -------
        List[RetrievalResult]
            Top-``top_k`` results by RRF score, deduplicated by ``chunk_id``.
        """
        logger.debug("HybridRetriever: fetching up to %d dense candidates …", top_k)
        dense_results = self._dense.retrieve(query, top_k=top_k)

        logger.debug("HybridRetriever: fetching up to %d sparse candidates …", top_k)
        sparse_results = self._sparse.retrieve(query, top_k=top_k)

        logger.debug(
            "HybridRetriever: fusing %d dense + %d sparse results with RRF(k=%d) …",
            len(dense_results),
            len(sparse_results),
            self.rrf_k,
        )
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )

        top = fused[:top_k]
        logger.debug("HybridRetriever: returning %d fused results.", len(top))
        return top
