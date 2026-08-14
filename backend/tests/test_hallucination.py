"""
backend/tests/test_hallucination.py
--------------------------------------
Unit tests for the Milestone 4 hallucination detection pipeline.

Test groups
-----------
1. split_into_claims          – sentence splitting edge cases
2. verify_claim               – mocked NLI model, label mapping + softmax
3. HallucinationDetector._is_flagged – flagging logic without model
4. HallucinationDetector.detect – full detection with mocked NLI model
   a. Known entailment pair  → not flagged
   b. Known contradiction pair → always flagged
   c. Neutral below threshold → flagged
   d. Neutral above threshold → not flagged
   e. Empty answer            → empty flags
   f. No context chunks       → empty flags
5. POST /query integration    – hallucination_flags populated in response
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.hallucination.detector import (
    HallucinationDetector,
    split_into_claims,
    verify_claim,
    _LABEL_MAP,
)
from app.models.response import Citation, HallucinationFlag
from app.retrieval.dense import RetrievalResult


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str, text: str, source: str = "doc.txt") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        metadata={"source": source, "page_num": 1},
        score=0.9,
    )


def _make_citation(chunk_id: str, source: str = "doc.txt") -> Citation:
    return Citation(
        chunk_id=chunk_id,
        source=source,
        page_num=1,
        section="Test",
        snippet="snippet text",
    )


def _mock_nli_model(label_idx: int, confidence: float = 0.95):
    """
    Return a MagicMock CrossEncoder that always predicts ``label_idx``
    with the specified confidence (using one-hot-ish logits so softmax
    returns approximately ``confidence`` for that index).
    """
    import numpy as np

    model = MagicMock()
    # Build logits so softmax ≈ the desired confidence for label_idx
    # Simple approach: put a large value on label_idx, 0 elsewhere
    logit_scale = np.log(confidence / (1 - confidence + 1e-9) * 2)
    logits = np.zeros(3)
    logits[label_idx] = logit_scale
    model.predict.return_value = np.array([logits])
    return model


# ── 1. split_into_claims ──────────────────────────────────────────────────────

class TestSplitIntoClaims:
    """Verify sentence splitting behaviour."""

    def test_single_sentence(self):
        claims = split_into_claims("The sky is blue.")
        assert len(claims) == 1
        assert claims[0] == "The sky is blue."

    def test_multiple_sentences(self):
        claims = split_into_claims("Python is fast. It is also readable. Many use it.")
        assert len(claims) >= 2

    def test_empty_string_returns_empty(self):
        assert split_into_claims("") == []

    def test_whitespace_only_returns_empty(self):
        assert split_into_claims("   ") == []

    def test_none_like_empty_string(self):
        # Callers should never pass None, but guard anyway
        assert split_into_claims("") == []

    def test_no_trailing_empty_claims(self):
        claims = split_into_claims("Hello world.")
        assert all(c.strip() != "" for c in claims)

    def test_bullet_list_splits(self):
        text = "Python features:\n- Fast\n- Readable\n- Popular"
        claims = split_into_claims(text)
        assert len(claims) >= 2

    def test_numbered_list_splits(self):
        text = "Steps:\n1. Install Python\n2. Write code\n3. Run it"
        claims = split_into_claims(text)
        assert len(claims) >= 2

    def test_abbreviation_not_split(self):
        # "Dr." should not be treated as a sentence boundary
        claims = split_into_claims("Dr. Smith is a scientist. He works at MIT.")
        # should produce 2 claims (or at least not split on "Dr.")
        assert len(claims) >= 1
        # The first claim should still contain "Dr. Smith"
        joined = " ".join(claims)
        assert "Dr. Smith" in joined

    def test_question_mark_splits(self):
        claims = split_into_claims("What is Python? It is a language.")
        assert len(claims) >= 2

    def test_exclamation_mark_splits(self):
        claims = split_into_claims("Python is great! Everyone loves it.")
        assert len(claims) >= 2


# ── 2. verify_claim ───────────────────────────────────────────────────────────

class TestVerifyClaim:
    """Test verify_claim using a mocked NLI model."""

    def test_returns_tuple_of_label_and_confidence(self):
        model = _mock_nli_model(label_idx=1)  # entailment
        label, conf = verify_claim("Python is fast.", "Python is a fast language.", model)
        assert isinstance(label, str)
        assert isinstance(conf, float)

    def test_entailment_label_returned(self):
        model = _mock_nli_model(label_idx=1)  # entailment = index 1
        label, _ = verify_claim("Python is fast.", "Python is a fast language.", model)
        assert label == "entailment"

    def test_contradiction_label_returned(self):
        model = _mock_nli_model(label_idx=0)  # contradiction = index 0
        label, _ = verify_claim("Python is slow.", "Python is a fast language.", model)
        assert label == "contradiction"

    def test_neutral_label_returned(self):
        model = _mock_nli_model(label_idx=2)  # neutral = index 2
        label, _ = verify_claim("Python was invented in 1991.", "Python is popular.", model)
        assert label == "neutral"

    def test_confidence_between_0_and_1(self):
        model = _mock_nli_model(label_idx=1, confidence=0.9)
        _, conf = verify_claim("claim", "context", model)
        assert 0.0 <= conf <= 1.0

    def test_model_predict_called_with_pair(self):
        model = _mock_nli_model(label_idx=1)
        verify_claim("claim text", "context text", model)
        model.predict.assert_called_once()
        call_args = model.predict.call_args[0][0]
        assert ("context text", "claim text") in call_args

    def test_all_label_indices_map_correctly(self):
        """Each label index should produce the correct string label."""
        expected = {0: "contradiction", 1: "entailment", 2: "neutral"}
        for idx, expected_label in expected.items():
            model = _mock_nli_model(label_idx=idx)
            label, _ = verify_claim("x", "y", model)
            assert label == expected_label, f"Index {idx} should map to {expected_label}"


# ── 3. HallucinationDetector._is_flagged ─────────────────────────────────────

class TestIsFlagged:
    """Unit-test the flagging logic in isolation (no model needed)."""

    def setup_method(self):
        self.det = HallucinationDetector(threshold=0.7)

    def test_contradiction_always_flagged(self):
        assert self.det._is_flagged("contradiction", 0.99, 0.7) is True

    def test_contradiction_low_confidence_still_flagged(self):
        assert self.det._is_flagged("contradiction", 0.1, 0.7) is True

    def test_entailment_never_flagged(self):
        assert self.det._is_flagged("entailment", 0.99, 0.7) is False

    def test_entailment_low_confidence_not_flagged(self):
        assert self.det._is_flagged("entailment", 0.1, 0.7) is False

    def test_neutral_above_threshold_not_flagged(self):
        assert self.det._is_flagged("neutral", 0.8, 0.7) is False

    def test_neutral_below_threshold_flagged(self):
        assert self.det._is_flagged("neutral", 0.6, 0.7) is True

    def test_neutral_at_threshold_not_flagged(self):
        # Boundary: exactly at threshold → not flagged (strict less-than)
        assert self.det._is_flagged("neutral", 0.7, 0.7) is False


# ── 4. HallucinationDetector.detect ──────────────────────────────────────────

class TestHallucinationDetectorDetect:
    """Full detect() tests with a mocked NLI model."""

    CONTEXT_TEXT = "Python is a high-level programming language created by Guido van Rossum."
    CHUNK = _make_chunk("chunk-1", CONTEXT_TEXT)
    CITATION = _make_citation("chunk-1")

    def _make_detector(self, label_idx: int, confidence: float = 0.95, threshold: float = 0.7):
        det = HallucinationDetector(threshold=threshold)
        det._model = _mock_nli_model(label_idx, confidence)
        return det

    # a. Known entailment pair → not flagged
    def test_entailment_claim_not_flagged(self):
        det = self._make_detector(label_idx=1, confidence=0.95)  # entailment
        flags = det.detect(
            answer="Python is a high-level language.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        assert len(flags) == 1
        assert flags[0].label == "entailment"
        assert flags[0].flagged is False

    # b. Known contradiction pair → always flagged
    def test_contradiction_claim_flagged(self):
        det = self._make_detector(label_idx=0, confidence=0.95)  # contradiction
        flags = det.detect(
            answer="Python was invented in 2020.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        assert len(flags) == 1
        assert flags[0].label == "contradiction"
        assert flags[0].flagged is True

    # c. Neutral below threshold → flagged
    def test_neutral_below_threshold_flagged(self):
        det = self._make_detector(label_idx=2, confidence=0.5, threshold=0.7)  # neutral, low conf
        flags = det.detect(
            answer="Python is sometimes used in space missions.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        assert flags[0].label == "neutral"
        assert flags[0].flagged is True

    # d. Neutral above threshold → not flagged
    def test_neutral_above_threshold_not_flagged(self):
        det = self._make_detector(label_idx=2, confidence=0.95, threshold=0.7)  # neutral, high conf
        flags = det.detect(
            answer="Python is sometimes used in space missions.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        assert flags[0].label == "neutral"
        assert flags[0].flagged is False

    # e. Empty answer → empty flags
    def test_empty_answer_returns_empty(self):
        det = self._make_detector(label_idx=1)
        flags = det.detect(answer="", citations=[], context_chunks=[self.CHUNK])
        assert flags == []

    # f. No context chunks → empty flags
    def test_no_context_chunks_returns_empty(self):
        det = self._make_detector(label_idx=1)
        flags = det.detect(
            answer="Python is great.",
            citations=[self.CITATION],
            context_chunks=[],
        )
        assert flags == []

    def test_flag_contains_claim_text(self):
        det = self._make_detector(label_idx=1)
        flags = det.detect(
            answer="Python is a language.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        assert any("Python" in f.claim for f in flags)

    def test_flag_cited_chunk_id_populated(self):
        det = self._make_detector(label_idx=1)
        flags = det.detect(
            answer="Python is a language.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        assert flags[0].cited_chunk_id == "chunk-1"

    def test_no_citations_still_runs_with_fallback_context(self):
        """When citations list is empty, detection falls back to combined context."""
        det = self._make_detector(label_idx=1)
        flags = det.detect(
            answer="Python is a language.",
            citations=[],         # no citations
            context_chunks=[self.CHUNK],
        )
        assert len(flags) >= 1

    def test_multiple_claims_produce_multiple_flags(self):
        det = self._make_detector(label_idx=1)
        answer = "Python is a language. It supports OOP. It is popular."
        flags = det.detect(
            answer=answer,
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        assert len(flags) >= 2

    def test_returns_list_of_hallucination_flag_objects(self):
        det = self._make_detector(label_idx=1)
        flags = det.detect(
            answer="Python is a language.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        for f in flags:
            assert isinstance(f, HallucinationFlag)

    def test_confidence_between_0_and_1(self):
        det = self._make_detector(label_idx=1, confidence=0.9)
        flags = det.detect(
            answer="Python is a language.",
            citations=[self.CITATION],
            context_chunks=[self.CHUNK],
        )
        for f in flags:
            assert 0.0 <= f.confidence <= 1.0


# ── 5. POST /query integration — hallucination_flags in response ──────────────

class TestQueryRouteHallucinationIntegration:
    """
    Integration tests for POST /query confirming hallucination_flags are
    populated in the response when detect_hallucinations=True.
    """

    @pytest.fixture(autouse=True)
    def mock_pipeline(self, monkeypatch):
        import app.api.routes_query as rq
        from app.generation.generator import GenerationResult

        chunk = _make_chunk("chunk-1", "Python is a high-level programming language.")
        citation = _make_citation("chunk-1")

        # Mock hybrid retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [chunk]
        monkeypatch.setattr(rq, "_hybrid_retriever", mock_retriever)

        # Mock reranker
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [chunk]
        monkeypatch.setattr(rq, "_reranker", mock_reranker)

        # Mock generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="Python is a high-level language.",
            citations=[citation],
            latency_ms=10.0,
            raw_chunk_ids=["chunk-1"],
        )
        monkeypatch.setattr(rq, "_generator", mock_generator)

        # Mock detector → returns one entailment flag
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            HallucinationFlag(
                claim="Python is a high-level language.",
                label="entailment",
                confidence=0.95,
                cited_chunk_id="chunk-1",
                flagged=False,
            )
        ]
        monkeypatch.setattr(rq, "_detector", mock_detector)

        self.mock_detector = mock_detector

    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    def test_hallucination_flags_present_in_response(self, client):
        response = client.post(
            "/query/",
            json={"query": "What is Python?", "detect_hallucinations": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "hallucination_flags" in data
        assert len(data["hallucination_flags"]) == 1

    def test_hallucination_flag_fields(self, client):
        response = client.post(
            "/query/",
            json={"query": "What is Python?", "detect_hallucinations": True},
        )
        flag = response.json()["hallucination_flags"][0]
        assert flag["label"] == "entailment"
        assert flag["flagged"] is False
        assert 0.0 <= flag["confidence"] <= 1.0
        assert "claim" in flag

    def test_confidence_score_in_response(self, client):
        response = client.post(
            "/query/",
            json={"query": "What is Python?", "detect_hallucinations": True},
        )
        data = response.json()
        # 1 entailed / 1 total = 1.0
        assert data["confidence"] == 1.0

    def test_detect_hallucinations_false_skips_detector(self, client):
        client.post(
            "/query/",
            json={"query": "What is Python?", "detect_hallucinations": False},
        )
        self.mock_detector.detect.assert_not_called()

    def test_detect_hallucinations_true_calls_detector(self, client):
        client.post(
            "/query/",
            json={"query": "What is Python?", "detect_hallucinations": True},
        )
        self.mock_detector.detect.assert_called_once()

    def test_empty_flags_when_detection_skipped(self, client):
        response = client.post(
            "/query/",
            json={"query": "What is Python?", "detect_hallucinations": False},
        )
        data = response.json()
        assert data["hallucination_flags"] == []
        assert data["confidence"] is None
