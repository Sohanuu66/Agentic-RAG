"""
backend/app/retrieval/reranker.py
----------------------------------
Cross-encoder reranker using ``ms-marco-MiniLM-L-6-v2`` from the
``sentence-transformers`` library.

The cross-encoder scores each (query, chunk_text) pair jointly, giving
much more accurate relevance estimates than bi-encoder similarity.

Classes
-------
CrossEncoderReranker
    Loads the cross-encoder model and exposes ``rerank(query, candidates,
    top_n)``.
"""

from __future__ import annotations

import logging
from typing import List

from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Cross-encoder reranker.

    Parameters
    ----------
    model_name:
        HuggingFace model name / path.  Defaults to
        ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~80 MB).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model(self):
        """Lazily load the CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            logger.info("Loading cross-encoder model '%s' …", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_n: int = 5,
    ) -> List[RetrievalResult]:
        """
        Score every (query, chunk_text) pair and return the top-*n* chunks.

        Parameters
        ----------
        query:
            The natural-language query string.
        candidates:
            Candidate ``RetrievalResult`` objects to rerank (typically the
            output of ``HybridRetriever.retrieve``).
        top_n:
            Number of top results to return after reranking.

        Returns
        -------
        List[RetrievalResult]
            Top-``top_n`` results sorted by descending cross-encoder score.
            The ``score`` field is replaced with the raw logit from the
            cross-encoder.
        """
        if not candidates:
            return []

        model = self._get_model()

        pairs = [(query, c.text) for c in candidates]
        logger.debug(
            "CrossEncoderReranker: scoring %d pairs …", len(pairs)
        )
        scores = model.predict(pairs)  # numpy array of shape (len(pairs),)

        # Attach new scores and sort descending
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top: List[RetrievalResult] = []
        for result, score in scored[:top_n]:
            top.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    metadata=result.metadata,
                    score=float(score),
                )
            )

        logger.debug("CrossEncoderReranker: kept top %d results.", len(top))
        return top
