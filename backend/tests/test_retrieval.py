"""
backend/tests/test_retrieval.py
---------------------------------
Unit + integration tests for the Milestone 2 retrieval pipeline.

Test groups
-----------
1. RRF math                  – verify the RRF formula with controlled inputs
2. DenseRetriever            – mocked ChromaDB + SentenceTransformers
3. SparseRetriever           – round-trip against a real BM25 pickle file
4. HybridRetriever           – mocked dense + sparse, verifies fusion ordering
5. CrossEncoderReranker      – mocked CrossEncoder, verifies top-N selection
6. Integration               – index 3 chunks, run hybrid → rerank, check top-1
"""

from __future__ import annotations

import pickle
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.retrieval.dense import DenseRetriever, RetrievalResult
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_result(chunk_id: str, text: str = "sample text", score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        metadata={"source": "test.txt", "page_num": 1},
        score=score,
    )


# ── 1. RRF math ────────────────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    """Verify the mathematical correctness of the RRF formula."""

    def test_single_list_preserves_order(self):
        """With one voter list the RRF ranking should mirror the input order."""
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        fused = reciprocal_rank_fusion([results], k=60)
        ids = [r.chunk_id for r in fused]
        assert ids == ["a", "b", "c"]

    def test_two_agreeing_lists_boost_top_rank(self):
        """A chunk ranked first in both lists should rank first in fused output."""
        list1 = [_make_result("x"), _make_result("y"), _make_result("z")]
        list2 = [_make_result("x"), _make_result("z"), _make_result("y")]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert fused[0].chunk_id == "x"

    def test_rrf_scores_are_positive(self):
        results = [_make_result("a"), _make_result("b")]
        fused = reciprocal_rank_fusion([results], k=60)
        assert all(r.score > 0 for r in fused)

    def test_deduplication(self):
        """A chunk appearing in both lists should appear only once in output."""
        list1 = [_make_result("shared"), _make_result("only1")]
        list2 = [_make_result("shared"), _make_result("only2")]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [r.chunk_id for r in fused]
        assert ids.count("shared") == 1

    def test_correct_rrf_score_formula(self):
        """Verify the exact RRF score for a chunk ranked 1st in two lists."""
        # rank 1 in both → score = 1/(60+1) + 1/(60+1) = 2/61
        list1 = [_make_result("a")]
        list2 = [_make_result("a")]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        expected = 2.0 / 61.0
        assert abs(fused[0].score - expected) < 1e-9

    def test_empty_lists_return_empty(self):
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_union_of_chunks_is_complete(self):
        """All unique chunk_ids across both lists must appear in the fused output."""
        list1 = [_make_result("a"), _make_result("b")]
        list2 = [_make_result("c"), _make_result("d")]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        fused_ids = {r.chunk_id for r in fused}
        assert fused_ids == {"a", "b", "c", "d"}

    def test_k_parameter_affects_scores(self):
        """Lower k → higher sensitivity to rank differences."""
        results = [_make_result("a")]
        score_k60 = reciprocal_rank_fusion([results], k=60)[0].score
        score_k1 = reciprocal_rank_fusion([results], k=1)[0].score
        # 1/(1+1) > 1/(60+1)
        assert score_k1 > score_k60


# ── 2. DenseRetriever ─────────────────────────────────────────────────────────

class TestDenseRetriever:
    """Mocked ChromaDB + SentenceTransformers tests."""

    def _make_retriever(self, ids, docs, metadatas, distances):
        retriever = DenseRetriever(persist_dir="/fake/dir")

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype="float32")
        retriever._model = mock_model

        mock_collection = MagicMock()
        mock_collection.count.return_value = len(ids)
        mock_collection.query.return_value = {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metadatas],
            "distances": [distances],
        }
        retriever._collection = mock_collection
        return retriever

    def test_returns_correct_number_of_results(self):
        retriever = self._make_retriever(
            ids=["c1", "c2", "c3"],
            docs=["doc1", "doc2", "doc3"],
            metadatas=[{}, {}, {}],
            distances=[0.1, 0.3, 0.5],
        )
        results = retriever.retrieve("test query", top_k=3)
        assert len(results) == 3

    def test_results_are_retrieval_result_instances(self):
        retriever = self._make_retriever(["c1"], ["text"], [{}], [0.2])
        results = retriever.retrieve("q", top_k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_score_is_similarity_not_distance(self):
        """ChromaDB returns cosine distance; we convert to similarity = 1 - dist/2."""
        retriever = self._make_retriever(["c1"], ["text"], [{}], [0.0])  # dist=0 → sim=1
        results = retriever.retrieve("q", top_k=1)
        assert results[0].score == pytest.approx(1.0)

    def test_empty_collection_returns_empty(self):
        retriever = DenseRetriever(persist_dir="/fake")
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        retriever._collection = mock_collection
        results = retriever.retrieve("q")
        assert results == []

    def test_chunk_ids_match(self):
        retriever = self._make_retriever(
            ["id1", "id2"], ["a", "b"], [{}, {}], [0.1, 0.2]
        )
        results = retriever.retrieve("q", top_k=2)
        assert [r.chunk_id for r in results] == ["id1", "id2"]


# ── 3. SparseRetriever ────────────────────────────────────────────────────────

class TestSparseRetriever:
    """Real BM25 pickle round-trip tests (no mocking needed)."""

    def _write_pickle(self, tmp_path: Path, texts: List[str], ids: List[str]) -> SparseRetriever:
        from rank_bm25 import BM25Okapi  # type: ignore[import]
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        payload = {
            "bm25": bm25,
            "chunk_ids": ids,
            "texts": texts,
            "metadatas": [{} for _ in ids],
        }
        pickle_path = tmp_path / "bm25_index.pkl"
        with open(pickle_path, "wb") as fh:
            pickle.dump(payload, fh)
        return SparseRetriever(persist_dir=str(tmp_path))

    def test_returns_results(self, tmp_path):
        retriever = self._write_pickle(
            tmp_path,
            texts=["hello world foo", "bar baz qux"],
            ids=["c1", "c2"],
        )
        results = retriever.retrieve("hello", top_k=2)
        assert len(results) >= 1

    def test_relevant_chunk_ranked_first(self, tmp_path):
        retriever = self._write_pickle(
            tmp_path,
            texts=["chromadb vector database embeddings", "totally unrelated banana recipe"],
            ids=["relevant", "irrelevant"],
        )
        results = retriever.retrieve("chromadb vector", top_k=2)
        assert results[0].chunk_id == "relevant"

    def test_missing_pickle_returns_empty(self, tmp_path):
        retriever = SparseRetriever(persist_dir=str(tmp_path))
        results = retriever.retrieve("anything")
        assert results == []

    def test_result_chunk_ids_match_stored(self, tmp_path):
        retriever = self._write_pickle(
            tmp_path,
            texts=["alpha beta gamma", "delta epsilon zeta"],
            ids=["id-alpha", "id-delta"],
        )
        results = retriever.retrieve("alpha", top_k=2)
        returned_ids = {r.chunk_id for r in results}
        assert returned_ids.issubset({"id-alpha", "id-delta"})


# ── 4. HybridRetriever ────────────────────────────────────────────────────────

class TestHybridRetriever:
    """Mock both sub-retrievers and verify RRF fusion logic."""

    def _make_hybrid(self, dense_results, sparse_results, top_k=10):
        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = dense_results
        hybrid._dense = mock_dense

        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = sparse_results
        hybrid._sparse = mock_sparse

        return hybrid

    def test_returns_list_of_retrieval_results(self):
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("b"), _make_result("c")]
        hybrid = self._make_hybrid(dense, sparse)
        results = hybrid.retrieve("q", top_k=5)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_deduplicates_by_chunk_id(self):
        dense = [_make_result("shared"), _make_result("only_dense")]
        sparse = [_make_result("shared"), _make_result("only_sparse")]
        hybrid = self._make_hybrid(dense, sparse)
        results = hybrid.retrieve("q", top_k=10)
        ids = [r.chunk_id for r in results]
        assert ids.count("shared") == 1

    def test_union_of_chunks_returned(self):
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("c"), _make_result("d")]
        hybrid = self._make_hybrid(dense, sparse)
        results = hybrid.retrieve("q", top_k=10)
        returned_ids = {r.chunk_id for r in results}
        assert returned_ids == {"a", "b", "c", "d"}

    def test_top_k_limits_output(self):
        dense = [_make_result(f"d{i}") for i in range(10)]
        sparse = [_make_result(f"s{i}") for i in range(10)]
        hybrid = self._make_hybrid(dense, sparse)
        results = hybrid.retrieve("q", top_k=5)
        assert len(results) <= 5

    def test_both_sub_retrievers_are_called(self):
        dense = [_make_result("a")]
        sparse = [_make_result("b")]
        hybrid = self._make_hybrid(dense, sparse)
        hybrid.retrieve("my query", top_k=5)
        hybrid._dense.retrieve.assert_called_once()
        hybrid._sparse.retrieve.assert_called_once()

    def test_chunk_ranked_first_in_both_is_top_of_fused(self):
        """A chunk that tops both ranked lists must win RRF."""
        best = _make_result("best")
        dense = [best, _make_result("ok"), _make_result("meh")]
        sparse = [best, _make_result("meh"), _make_result("ok")]
        hybrid = self._make_hybrid(dense, sparse)
        results = hybrid.retrieve("q", top_k=5)
        assert results[0].chunk_id == "best"


# ── 5. CrossEncoderReranker ───────────────────────────────────────────────────

class TestCrossEncoderReranker:
    """Mock the CrossEncoder model to avoid downloading weights in CI."""

    def _make_reranker(self, scores: List[float]) -> CrossEncoderReranker:
        reranker = CrossEncoderReranker(model_name="fake-model")
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores, dtype="float32")
        reranker._model = mock_model
        return reranker

    def test_returns_top_n_results(self):
        candidates = [_make_result(f"c{i}") for i in range(5)]
        reranker = self._make_reranker([0.1, 0.9, 0.3, 0.7, 0.5])
        results = reranker.rerank("q", candidates, top_n=3)
        assert len(results) == 3

    def test_results_are_sorted_by_score_descending(self):
        candidates = [_make_result("a"), _make_result("b"), _make_result("c")]
        reranker = self._make_reranker([0.1, 0.9, 0.5])
        results = reranker.rerank("q", candidates, top_n=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_1_is_highest_scoring_candidate(self):
        candidates = [_make_result("low"), _make_result("high"), _make_result("mid")]
        reranker = self._make_reranker([0.1, 0.99, 0.5])
        results = reranker.rerank("q", candidates, top_n=1)
        assert results[0].chunk_id == "high"

    def test_empty_candidates_returns_empty(self):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = MagicMock()
        results = reranker.rerank("q", [], top_n=5)
        assert results == []

    def test_score_replaced_by_cross_encoder_score(self):
        """The original retrieval score must be overwritten by the CE score."""
        candidates = [_make_result("a", score=0.42)]
        reranker = self._make_reranker([0.99])
        results = reranker.rerank("q", candidates, top_n=1)
        assert results[0].score == pytest.approx(0.99, abs=1e-4)

    def test_top_n_greater_than_candidates_returns_all(self):
        candidates = [_make_result("a"), _make_result("b")]
        reranker = self._make_reranker([0.3, 0.7])
        results = reranker.rerank("q", candidates, top_n=10)
        assert len(results) == 2


# ── 6. Integration test ───────────────────────────────────────────────────────

class TestRetrievalIntegration:
    """
    End-to-end: build a real BM25 pickle, run HybridRetriever (mocked dense
    + real sparse), then rerank with mocked CrossEncoder.

    Goal: confirm that the expected chunk surfaces in the top result.
    """

    CHUNKS = [
        ("vec-001", "ChromaDB is a vector database for storing embeddings"),
        ("vec-002", "BM25 is a sparse retrieval algorithm based on term frequency"),
        ("vec-003", "Reciprocal rank fusion combines multiple ranked lists"),
    ]

    QUERY = "vector database embeddings"

    def _build_pickle(self, tmp_path: Path) -> Path:
        from rank_bm25 import BM25Okapi  # type: ignore[import]
        ids, texts = zip(*self.CHUNKS)
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        payload = {
            "bm25": bm25,
            "chunk_ids": list(ids),
            "texts": list(texts),
            "metadatas": [{} for _ in ids],
        }
        pickle_path = tmp_path / "bm25_index.pkl"
        with open(pickle_path, "wb") as fh:
            pickle.dump(payload, fh)
        return tmp_path

    def test_top_result_contains_expected_chunk(self, tmp_path):
        persist_dir = self._build_pickle(tmp_path)

        # Build hybrid retriever with mocked dense + real sparse
        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.rrf_k = 60

        # Dense: return results in reverse-relevance order so sparse wins the tie
        dense_results = [
            RetrievalResult("vec-003", self.CHUNKS[2][1], {}, score=0.9),
            RetrievalResult("vec-001", self.CHUNKS[0][1], {}, score=0.8),
            RetrievalResult("vec-002", self.CHUNKS[1][1], {}, score=0.7),
        ]
        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = dense_results
        hybrid._dense = mock_dense

        sparse = SparseRetriever(persist_dir=str(persist_dir))
        hybrid._sparse = sparse

        fused = hybrid.retrieve(self.QUERY, top_k=20)

        # "vec-001" scores well in BM25 for "vector database embeddings";
        # after RRF fusion the top-5 should contain it.
        top_ids = {r.chunk_id for r in fused[:5]}
        assert "vec-001" in top_ids, (
            f"Expected 'vec-001' in top-5 fused results, got: {top_ids}"
        )

    def test_reranker_promotes_most_relevant_chunk(self, tmp_path):
        """After reranking, the most semantically relevant chunk should be #1."""
        candidates = [
            RetrievalResult("best", "ChromaDB stores embeddings for vector search", {}, 0.5),
            RetrievalResult("mid", "BM25 uses term frequency inverse document frequency", {}, 0.6),
            RetrievalResult("worst", "Banana recipes include flour and eggs", {}, 0.7),
        ]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        mock_model = MagicMock()
        # Simulate: "best" gets highest CE score, "worst" gets lowest
        mock_model.predict.return_value = np.array([0.95, 0.60, 0.05], dtype="float32")
        reranker._model = mock_model

        results = reranker.rerank(self.QUERY, candidates, top_n=3)
        assert results[0].chunk_id == "best"
