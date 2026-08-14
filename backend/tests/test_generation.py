"""
backend/tests/test_generation.py
----------------------------------
Unit tests for the Milestone 3 generation pipeline.

Test groups
-----------
1. format_context          – prompt formatting utilities
2. CitationGenerator._extract_json  – JSON extraction (valid, fenced, preamble, missing keys)
3. CitationGenerator._validate_citations  – ghost-citation rejection
4. CitationGenerator.generate  – mock Groq call, end-to-end happy path
5. CitationGenerator.generate  – invalid chunk_id is rejected from response
6. CitationGenerator.generate  – Groq returns malformed JSON → graceful degradation
7. POST /query route           – integration test via TestClient (mocked Groq)
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.generation.generator import CitationGenerator, GenerationResult
from app.generation.prompts import format_context, build_messages, build_retry_messages, SYSTEM_PROMPT
from app.models.response import Citation
from app.retrieval.dense import RetrievalResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_chunk(
    chunk_id: str = "chunk-1",
    text: str = "The sky is blue.",
    source: str = "doc.txt",
    page_num: int = 1,
    section: str = "Introduction",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        metadata={"source": source, "page_num": page_num, "section": section},
        score=0.9,
    )


CHUNK_A = _make_chunk("chunk-A", "Python is a high-level programming language.", "python.txt", 1, "Overview")
CHUNK_B = _make_chunk("chunk-B", "It supports multiple programming paradigms.", "python.txt", 2, "Features")
CHUNK_C = _make_chunk("chunk-C", "FastAPI is a modern web framework for Python.", "fastapi.txt", 1, "Intro")

ALL_CHUNKS: List[RetrievalResult] = [CHUNK_A, CHUNK_B, CHUNK_C]


# ── 1. format_context ──────────────────────────────────────────────────────────

class TestFormatContext:
    """Verify that format_context produces well-structured context blocks."""

    def test_empty_chunks_returns_placeholder(self):
        result = format_context([])
        assert "No context available" in result

    def test_chunk_id_present_in_output(self):
        result = format_context([CHUNK_A])
        assert "chunk-A" in result

    def test_text_present_in_output(self):
        result = format_context([CHUNK_A])
        assert CHUNK_A.text.strip() in result

    def test_metadata_fields_present(self):
        result = format_context([CHUNK_A])
        assert "python.txt" in result     # source
        assert "Overview" in result       # section

    def test_multiple_chunks_all_ids_present(self):
        result = format_context(ALL_CHUNKS)
        for chunk in ALL_CHUNKS:
            assert chunk.chunk_id in result

    def test_chunks_numbered(self):
        result = format_context([CHUNK_A, CHUNK_B])
        assert "[1]" in result
        assert "[2]" in result


class TestBuildMessages:
    """Verify the messages list structure for the Groq API."""

    def test_returns_two_messages(self):
        msgs = build_messages("What is Python?", [CHUNK_A])
        assert len(msgs) == 2

    def test_system_role_first(self):
        msgs = build_messages("Q?", [CHUNK_A])
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_user_role_second(self):
        msgs = build_messages("Q?", [CHUNK_A])
        assert msgs[1]["role"] == "user"

    def test_query_in_user_message(self):
        msgs = build_messages("What is Python?", [CHUNK_A])
        assert "What is Python?" in msgs[1]["content"]

    def test_chunk_id_in_user_message(self):
        msgs = build_messages("Q?", [CHUNK_A])
        assert "chunk-A" in msgs[1]["content"]


# ── 2. _extract_json ────────────────────────────────────────────────────────────

class TestParseLlmJson:
    """Verify JSON extraction from raw LLM response text."""

    def setup_method(self):
        self.gen = CitationGenerator(api_key="test-key")

    def test_valid_json_parsed(self):
        raw = json.dumps({"answer": "hello", "citations": ["chunk-A"]})
        data = self.gen._extract_json(raw)
        assert data["answer"] == "hello"
        assert data["citations"] == ["chunk-A"]

    def test_markdown_fenced_json_stripped(self):
        raw = "```json\n{\"answer\": \"hi\", \"citations\": []}\n```"
        data = self.gen._extract_json(raw)
        assert data["answer"] == "hi"

    def test_plain_fenced_json_stripped(self):
        raw = "```\n{\"answer\": \"hi\", \"citations\": []}\n```"
        data = self.gen._extract_json(raw)
        assert data["answer"] == "hi"

    def test_preamble_text_before_json(self):
        """Model outputs reasoning text before the JSON object."""
        raw = 'Here is my answer:\n{"answer": "ok", "citations": []}'
        data = self.gen._extract_json(raw)
        assert data["answer"] == "ok"

    def test_missing_answer_key_raises(self):
        raw = json.dumps({"citations": []})
        with pytest.raises(ValueError, match="missing 'answer'"):
            self.gen._extract_json(raw)

    def test_missing_citations_key_raises(self):
        raw = json.dumps({"answer": "ok"})
        with pytest.raises(ValueError, match="missing 'citations'"):
            self.gen._extract_json(raw)

    def test_invalid_json_raises(self):
        raw = "not json at all"
        with pytest.raises(ValueError):
            self.gen._extract_json(raw)

    def test_empty_citations_list_accepted(self):
        raw = json.dumps({"answer": "ok", "citations": []})
        data = self.gen._extract_json(raw)
        assert data["citations"] == []


# ── 3. _validate_citations ─────────────────────────────────────────────────────

class TestValidateCitations:
    """Verify that ghost chunk_ids are rejected and valid ones are converted."""

    def setup_method(self):
        self.gen = CitationGenerator(api_key="test-key")
        self.ctx = {c.chunk_id: c for c in ALL_CHUNKS}

    def test_valid_id_produces_citation(self):
        citations = self.gen._validate_citations(["chunk-A"], self.ctx)
        assert len(citations) == 1
        assert citations[0].chunk_id == "chunk-A"

    def test_ghost_id_is_rejected(self):
        citations = self.gen._validate_citations(["ghost-999"], self.ctx)
        assert len(citations) == 0

    def test_mix_valid_and_ghost(self):
        citations = self.gen._validate_citations(["chunk-A", "ghost-999", "chunk-C"], self.ctx)
        ids = [c.chunk_id for c in citations]
        assert "chunk-A" in ids
        assert "chunk-C" in ids
        assert "ghost-999" not in ids
        assert len(citations) == 2

    def test_duplicate_ids_deduplicated(self):
        citations = self.gen._validate_citations(["chunk-A", "chunk-A"], self.ctx)
        assert len(citations) == 1

    def test_citation_fields_populated_from_chunk_metadata(self):
        citations = self.gen._validate_citations(["chunk-A"], self.ctx)
        c = citations[0]
        assert c.source == "python.txt"
        assert c.page_num == 1
        assert c.section == "Overview"
        assert "Python" in c.snippet   # text snippet from CHUNK_A

    def test_non_string_id_is_ignored(self):
        citations = self.gen._validate_citations([123, None], self.ctx)  # type: ignore
        assert citations == []

    def test_empty_id_list_returns_empty(self):
        citations = self.gen._validate_citations([], self.ctx)
        assert citations == []


# ── 4. CitationGenerator.generate — happy path ────────────────────────────────

class TestCitationGeneratorHappyPath:
    """Mock the Groq client and test end-to-end generation."""

    def _mock_groq_response(self, answer: str, citations: list) -> MagicMock:
        """Return a MagicMock that mimics a Groq ChatCompletion response."""
        raw_json = json.dumps({"answer": answer, "citations": citations})
        choice = MagicMock()
        choice.message.content = raw_json
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_generate_returns_generation_result(self):
        gen = CitationGenerator(api_key="test-key")
        mock_resp = self._mock_groq_response("Python is great.", ["chunk-A"])

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("What is Python?", ALL_CHUNKS)

        assert isinstance(result, GenerationResult)
        assert result.answer == "Python is great."
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk-A"
        assert result.latency_ms >= 0.0

    def test_generate_latency_is_positive(self):
        gen = CitationGenerator(api_key="test-key")
        mock_resp = self._mock_groq_response("Answer.", [])

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("Q?", ALL_CHUNKS)

        assert result.latency_ms > 0

    def test_generate_multiple_valid_citations(self):
        gen = CitationGenerator(api_key="test-key")
        mock_resp = self._mock_groq_response(
            "Python supports multiple paradigms.",
            ["chunk-A", "chunk-B"],
        )

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("Describe Python.", ALL_CHUNKS)

        assert len(result.citations) == 2
        cited_ids = {c.chunk_id for c in result.citations}
        assert cited_ids == {"chunk-A", "chunk-B"}


# ── 5. CitationGenerator.generate — invalid chunk_id rejected ─────────────────

class TestCitationGeneratorGhostRejection:
    """Ensure that a chunk_id not in the context cannot appear in the output."""

    def _mock_groq_response(self, answer: str, citations: list) -> MagicMock:
        raw_json = json.dumps({"answer": answer, "citations": citations})
        choice = MagicMock()
        choice.message.content = raw_json
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_ghost_id_not_in_result(self):
        gen = CitationGenerator(api_key="test-key")
        # Model claims to cite "ghost-xyz" which was NOT in the provided chunks
        mock_resp = self._mock_groq_response("Answer.", ["chunk-A", "ghost-xyz"])

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("Q?", ALL_CHUNKS)

        cited_ids = {c.chunk_id for c in result.citations}
        assert "ghost-xyz" not in cited_ids
        assert "chunk-A" in cited_ids

    def test_raw_chunk_ids_contains_ghost(self):
        """raw_chunk_ids preserves what the model actually returned."""
        gen = CitationGenerator(api_key="test-key")
        mock_resp = self._mock_groq_response("Answer.", ["ghost-xyz"])

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("Q?", [CHUNK_A])

        # raw_chunk_ids should still record what the model said
        assert "ghost-xyz" in result.raw_chunk_ids
        # but it must not appear in citations
        assert len(result.citations) == 0

    def test_all_ghost_ids_yields_no_citations(self):
        gen = CitationGenerator(api_key="test-key")
        mock_resp = self._mock_groq_response("Answer.", ["ghost-1", "ghost-2"])

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("Q?", [CHUNK_A])

        assert result.citations == []


# ── 6. CitationGenerator.generate — malformed JSON → graceful degradation ──────

class TestCitationGeneratorMalformedJSON:
    """If the LLM returns non-JSON, generate() should degrade gracefully."""

    def _mock_bad_response(self, content: str) -> MagicMock:
        choice = MagicMock()
        choice.message.content = content
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_malformed_json_returns_fallback_answer(self):
        gen = CitationGenerator(api_key="test-key")
        mock_resp = self._mock_bad_response("Sorry, I cannot answer that right now.")

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("Q?", [CHUNK_A])

        # Should not raise; returns graceful degradation
        assert isinstance(result, GenerationResult)
        assert "unable to generate" in result.answer.lower() or result.answer != ""
        assert result.citations == []

    def test_missing_citations_key_returns_fallback(self):
        gen = CitationGenerator(api_key="test-key")
        mock_resp = self._mock_bad_response(json.dumps({"answer": "ok"}))

        with patch.object(gen, "_get_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_client_fn.return_value = mock_client

            result = gen.generate("Q?", [CHUNK_A])

        assert result.citations == []


# ── 7. POST /query route — integration test via TestClient ─────────────────────

class TestQueryRoute:
    """
    Integration tests for POST /query using FastAPI TestClient.

    The orchestrator (app.orchestrator.orchestrate) and CitationGenerator
    (_generator) are mocked so the tests run without real indexes, models,
    or API keys.  The orchestrator is the new evidence-gathering entry point;
    mocking it at that boundary keeps all existing response-shape assertions
    intact while correctly reflecting the agentic architecture.
    """

    @pytest.fixture(autouse=True)
    def mock_pipeline(self, monkeypatch):
        """
        Replace the orchestrator and generator singletons with mocks.
        The orchestrator returns a fixed OrchestratorResult; the generator
        returns a fixed GenerationResult — together they cover the full route.
        """
        import app.api.routes_query as rq
        from app.orchestrator import OrchestratorResult

        # ── Mock orchestrator → returns CHUNK_A as assembled evidence ────────
        async def mock_orchestrate(query, session_id, retriever, reranker, top_k, top_n):
            return OrchestratorResult(
                chunks=[CHUNK_A],
                session_id=session_id,
                retrieval_source="documents",
                agent_rounds=1,
            )

        monkeypatch.setattr("app.api.routes_query.orchestrate", mock_orchestrate)

        # ── Mock generator → returns a valid GenerationResult ────────────────
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="Python is a high-level language.",
            citations=[
                Citation(
                    chunk_id="chunk-A",
                    source="python.txt",
                    page_num=1,
                    section="Overview",
                    snippet="Python is a high-level programming language.",
                )
            ],
            latency_ms=42.0,
            raw_chunk_ids=["chunk-A"],
        )
        monkeypatch.setattr(rq, "_generator", mock_generator)

        self.mock_generator = mock_generator

    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    def test_post_query_returns_200(self, client):
        response = client.post("/query/", json={"query": "What is Python?"})
        assert response.status_code == 200

    def test_response_has_answer(self, client):
        response = client.post("/query/", json={"query": "What is Python?"})
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Python is a high-level language."

    def test_response_has_citations(self, client):
        response = client.post("/query/", json={"query": "What is Python?"})
        data = response.json()
        assert "citations" in data
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "chunk-A"

    def test_response_has_latency_ms(self, client):
        response = client.post("/query/", json={"query": "What is Python?"})
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_response_has_session_id(self, client):
        """New field: session_id is echoed back in the response."""
        response = client.post("/query/", json={"query": "What is Python?"})
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] is not None

    def test_response_has_retrieval_source(self, client):
        """New field: retrieval_source indicates how evidence was gathered."""
        response = client.post("/query/", json={"query": "What is Python?"})
        data = response.json()
        assert "retrieval_source" in data
        assert data["retrieval_source"] == "documents"

    def test_response_has_agent_rounds(self, client):
        """New field: agent_rounds reports how many tool-call rounds ran."""
        response = client.post("/query/", json={"query": "What is Python?"})
        data = response.json()
        assert "agent_rounds" in data
        assert data["agent_rounds"] == 1

    def test_session_id_forwarded_when_provided(self, client):
        """If caller supplies a session_id, it must be echoed back unchanged."""
        response = client.post(
            "/query/", json={"query": "What is Python?", "session_id": "my-sess-123"}
        )
        data = response.json()
        assert data["session_id"] == "my-sess-123"

    def test_generator_called_with_evidence_chunks(self, client):
        """Generator must be called once with the chunks from the orchestrator."""
        client.post("/query/", json={"query": "What is Python?"})
        self.mock_generator.generate.assert_called_once()

    def test_empty_retrieval_returns_helpful_message(self, client, monkeypatch):
        """When orchestrator returns no chunks, return a helpful 200 (not 5xx)."""
        from app.orchestrator import OrchestratorResult

        async def mock_empty_orchestrate(query, session_id, retriever, reranker, top_k, top_n):
            return OrchestratorResult(
                chunks=[],
                session_id=session_id,
                retrieval_source="documents",
                agent_rounds=1,
            )

        monkeypatch.setattr("app.api.routes_query.orchestrate", mock_empty_orchestrate)
        response = client.post("/query/", json={"query": "Completely unknown topic"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        # Generator must NOT have been called when there is no evidence
        self.mock_generator.generate.assert_not_called()

    def test_custom_top_k_top_n_accepted(self, client):
        """Custom top_k/top_n values in the request must be accepted without errors."""
        response = client.post("/query/", json={"query": "Q?", "top_k": 7, "top_n": 3})
        assert response.status_code == 200
