"""
backend/app/retrieval/dense.py
-------------------------------
Dense (vector) retrieval using ChromaDB ANN search.

Classes
-------
DenseRetriever
    Embeds a query with ``all-MiniLM-L6-v2`` and queries the ChromaDB
    collection for approximate nearest neighbours.

Data model
----------
RetrievalResult
    Common dataclass shared by all retrievers and the reranker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Shared data model ─────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Atomic retrieval unit returned by any retriever or the reranker.

    Attributes
    ----------
    chunk_id:
        Unique identifier of the chunk (matches the ID used at index time).
    text:
        Full text of the chunk.
    metadata:
        Original metadata dict stored alongside the chunk (source, page_num, …).
    score:
        Relevance score — higher is better for dense/reranked results;
        for BM25 this is a raw BM25 score.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


# ── DenseRetriever ────────────────────────────────────────────────────────────

_CHROMA_COLLECTION = "ask_my_docs"


class DenseRetriever:
    """
    ANN retriever backed by ChromaDB + SentenceTransformers.

    Parameters
    ----------
    embedding_model:
        SentenceTransformers model name used for query embedding.
        Must match the model used at index time.
    persist_dir:
        ChromaDB persistence directory.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./data/chroma",
    ) -> None:
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self._model = None
        self._collection = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model(self):
        """Lazily load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            logger.info("Loading embedding model '%s' …", self.embedding_model)
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def _get_collection(self):
        """Lazily initialise the ChromaDB client and return the collection."""
        if self._collection is None:
            import chromadb  # type: ignore[import]
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = client.get_or_create_collection(
                name=_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """
        Embed *query* and return the top-*k* nearest chunks from ChromaDB.

        Parameters
        ----------
        query:
            The natural-language query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List[RetrievalResult]
            Results ordered from most to least relevant (highest cosine
            similarity first).  The ``score`` field is the cosine distance
            converted to a similarity (``1 - distance``).
        """
        model = self._get_model()
        collection = self._get_collection()

        # Clamp top_k to the number of indexed chunks to avoid ChromaDB errors
        n_results = min(top_k, collection.count())
        if n_results == 0:
            logger.warning("DenseRetriever: ChromaDB collection is empty.")
            return []

        logger.debug("DenseRetriever: querying ChromaDB for top-%d chunks …", n_results)
        query_embedding = model.encode(query, show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: List[RetrievalResult] = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB uses cosine *distance* (0 = identical, 2 = opposite).
            # Convert to a similarity score in [0, 1].
            similarity = 1.0 - (dist / 2.0)
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=doc,
                    metadata=meta or {},
                    score=similarity,
                )
            )

        logger.debug("DenseRetriever: returned %d results.", len(retrieval_results))
        return retrieval_results
