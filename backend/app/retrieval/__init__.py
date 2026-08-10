"""
app.retrieval — hybrid retrieval + reranking pipeline

Stages:
    dense   → DenseRetriever.retrieve(query, top_k) — ChromaDB ANN search
    sparse  → SparseRetriever.retrieve(query, top_k) — BM25 keyword search
    hybrid  → HybridRetriever.retrieve(query, top_k) — RRF fusion of dense + sparse
    rerank  → CrossEncoderReranker.rerank(query, candidates, top_n) — final ranking
"""

from .dense import DenseRetriever, RetrievalResult
from .sparse import SparseRetriever
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .reranker import CrossEncoderReranker

__all__ = [
    "RetrievalResult",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
]
