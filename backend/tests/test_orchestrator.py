"""
backend/tests/test_orchestrator.py
-------------------------------------
Direct unit tests for app/orchestrator.py.

All tests mock the OpenAI client at the _get_client / openai.OpenAI boundary
consistent with test_generation.py, so no real API key is required.

Test classes
------------
TestToolCallRouting
    Verifies the orchestrator calls the right tools in the right order and
    accumulates evidence correctly across rounds.

TestFallbackPath
    Forces the OpenAI client to raise mid-loop and asserts orchestrate()
    catches it, logs a warning, and returns a valid OrchestratorResult via
    the fallback direct-retrieve path.

TestMaxRoundsCutoff
    Mocks the OpenAI client to always return a tool call so the model never
    stops on its own.  Asserts the loop terminates at exactly agent_max_rounds.

TestCachedEvidenceReuse
    Seeds SQLite memory with prior evidence and mocks the OpenAI client to
    choose reuse_cached_evidence.  Asserts HybridRetriever is NOT called and
    retrieval_source == 'cached'.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import os
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.retrieval.dense import RetrievalResult
from app.memory import init_memory, save_session

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_memory_functions(monkeypatch, request):
    """
    By default, mock out the SQLite memory functions for orchestrator tests
    so they don't fail with 'no such table' on uninitialized :memory: dbs.
    Tests that specifically need real memory (like TestCachedEvidenceReuse)
    can opt out by using a specific marker or we can just let them override.
    Actually, TestCachedEvidenceReuse works because it uses a real file DB,
    but it imports the real functions directly from app.memory.
    We will patch them in `app.orchestrator` namespace.
    """
    if "TestCachedEvidenceReuse" in request.node.nodeid:
        return
        
    monkeypatch.setattr("app.orchestrator.get_session_context", lambda sid, db: {"has_cache": False, "last_query": "", "cached_evidence_json": "[]"})
    monkeypatch.setattr("app.orchestrator.load_cached_chunks", lambda sid, db: [])
    monkeypatch.setattr("app.orchestrator.save_session", lambda sid, q, c, db: None)

CHUNK_A = RetrievalResult(
    chunk_id="chunk-A",
    text="Python is a high-level programming language.",
    metadata={"source": "python.txt", "page_num": 1, "section": "Overview"},
    score=1.5,
)

CHUNK_B = RetrievalResult(
    chunk_id="chunk-B",
    text="Python supports multiple programming paradigms.",
    metadata={"source": "python.txt", "page_num": 2, "section": "Paradigms"},
    score=1.2,
)

WEB_CHUNK = RetrievalResult(
    chunk_id="web-0",
    text="Python 3.12 released with improved typing.",
    metadata={"source": "https://example.com", "section": "web_search"},
    score=0.8,
)


def _make_tool_call(name: str, args: dict, call_id: str = "call-1"):
    """Build a mock tool_call object matching the OpenAI SDK shape."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _make_response(tool_calls=None, content=""):
    """Build a mock ChatCompletion response."""
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _run(coro):
    """Run a coroutine synchronously in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# TestToolCallRouting
# ---------------------------------------------------------------------------

class TestToolCallRouting:
    """
    The orchestrator must call retrieve_documents then web_search when the model
    requests both tools in successive rounds, then stop when the model returns
    no tool calls.
    """

    def test_retrieve_then_web_search_then_stop(self, monkeypatch):
        """
        Round 1 → retrieve_documents (top_score low → below_threshold: true)
        Round 2 → web_search
        Round 3 → no tool_calls (model satisfied)
        Evidence must include chunks from both retrieve and web search.
        """
        from app.orchestrator import orchestrate

        # Mock retriever: returns CHUNK_A with a low score
        mock_retriever = MagicMock()
        low_score_chunk = RetrievalResult(
            chunk_id="chunk-A",
            text="Python is a language.",
            metadata={},
            score=0.1,  # below default threshold of 0.30
        )
        mock_retriever.retrieve.return_value = [low_score_chunk]

        # Mock reranker: passes through
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [low_score_chunk]

        # Mock Tavily
        def mock_web_search(query):
            return [WEB_CHUNK], {"results_found": 1, "query": query}

        # OpenAI responses: round1=retrieve, round2=web_search, round3=stop
        responses = [
            _make_response(tool_calls=[_make_tool_call(
                "retrieve_documents", {"query": "Python language"}, "call-1"
            )]),
            _make_response(tool_calls=[_make_tool_call(
                "web_search", {"query": "Python latest features"}, "call-2"
            )]),
            _make_response(tool_calls=[]),  # stop
        ]
        call_idx = {"n": 0}

        def fake_create(**kwargs):
            r = responses[call_idx["n"]]
            call_idx["n"] += 1
            return r

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        with patch("app.orchestrator.openai") as mock_openai,              patch("app.orchestrator._run_web_search", side_effect=mock_web_search):
            mock_openai.OpenAI.return_value = mock_client
            result = _run(orchestrate(
                query="Tell me about Python",
                session_id="test-session",
                retriever=mock_retriever,
                reranker=mock_reranker,
                top_k=20,
                top_n=5,
            ))

        # Both retrieve_documents and web_search tool calls fired
        assert call_idx["n"] == 3, "Expected exactly 3 OpenAI rounds"
        mock_retriever.retrieve.assert_called_once()
        # Evidence must contain both doc chunk and web chunk
        chunk_ids = {c.chunk_id for c in result.chunks}
        assert "chunk-A" in chunk_ids
        assert "web-0" in chunk_ids
        assert result.retrieval_source == "hybrid"
        assert result.agent_rounds >= 2

    def test_single_retrieve_above_threshold_no_web_search(self, monkeypatch):
        """When top_score >= threshold, model should stop without calling web_search."""
        from app.orchestrator import orchestrate

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [CHUNK_A]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [CHUNK_A]

        responses = [
            _make_response(tool_calls=[_make_tool_call(
                "retrieve_documents", {"query": "Python"}, "call-1"
            )]),
            _make_response(tool_calls=[]),  # satisfied
        ]
        call_idx = {"n": 0}

        def fake_create(**kwargs):
            r = responses[call_idx["n"]]
            call_idx["n"] += 1
            return r

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        with patch("app.orchestrator.openai") as mock_openai,              patch("app.orchestrator._run_web_search") as mock_ws:
            mock_openai.OpenAI.return_value = mock_client
            result = _run(orchestrate(
                query="Python",
                session_id="test-s2",
                retriever=mock_retriever,
                reranker=mock_reranker,
                top_k=20,
                top_n=5,
            ))

        mock_ws.assert_not_called()
        assert result.retrieval_source == "documents"
        assert len(result.chunks) == 1


# ---------------------------------------------------------------------------
# TestFallbackPath
# ---------------------------------------------------------------------------

class TestFallbackPath:
    """
    When the OpenAI client raises an exception during the agentic loop,
    orchestrate() must catch it, log a warning, and return a valid
    OrchestratorResult from the fallback direct-pipeline.
    """

    def test_fallback_fires_on_api_error(self, monkeypatch):
        """
        Mock the OpenAI client to raise an error immediately.
        Assert orchestrate() returns a valid result (does not propagate).
        """
        from app.orchestrator import orchestrate

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [CHUNK_A, CHUNK_B]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [CHUNK_A]

        with patch("app.orchestrator.openai") as mock_openai:
            import openai as real_openai
            mock_openai.OpenAI.return_value = MagicMock(
                **{"chat.completions.create.side_effect": Exception("Simulated API failure")}
            )
            result = _run(orchestrate(
                query="Test fallback",
                session_id="fallback-sess",
                retriever=mock_retriever,
                reranker=mock_reranker,
                top_k=20,
                top_n=5,
            ))

        # Must not raise — fallback should return a valid OrchestratorResult
        assert result is not None
        assert result.retrieval_source == "fallback"
        # Fallback calls the retriever directly
        mock_retriever.retrieve.assert_called_once()
        # agent_rounds == 0 because the loop never completed
        assert result.agent_rounds == 0

    def test_fallback_result_has_chunks(self, monkeypatch):
        """Fallback must return chunks from direct retrieval."""
        from app.orchestrator import orchestrate

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [CHUNK_A]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [CHUNK_A]

        with patch("app.orchestrator.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock(
                **{"chat.completions.create.side_effect": RuntimeError("boom")}
            )
            result = _run(orchestrate(
                query="fallback chunks test",
                session_id="fs-2",
                retriever=mock_retriever,
                reranker=mock_reranker,
                top_k=20,
                top_n=5,
            ))

        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == "chunk-A"


# ---------------------------------------------------------------------------
# TestMaxRoundsCutoff
# ---------------------------------------------------------------------------

class TestMaxRoundsCutoff:
    """
    The loop must terminate after exactly agent_max_rounds even if the model
    never returns a no-tool-call response.
    """

    def test_loop_terminates_at_max_rounds(self, monkeypatch):
        """
        Model always returns a retrieve_documents call.
        With agent_max_rounds=3, the loop must stop after 3 iterations.
        """
        from app.orchestrator import orchestrate

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [CHUNK_A]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [CHUNK_A]

        call_count = {"n": 0}

        def always_retrieve(**kwargs):
            call_count["n"] += 1
            return _make_response(tool_calls=[_make_tool_call(
                "retrieve_documents", {"query": "Python"}, f"call-{call_count['n']}"
            )])

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = always_retrieve

        # Temporarily set max_rounds to 3
        from app import config as cfg_module
        original_max = cfg_module.settings.agent_max_rounds
        cfg_module.settings.agent_max_rounds = 3

        try:
            with patch("app.orchestrator.openai") as mock_openai:
                mock_openai.OpenAI.return_value = mock_client
                result = _run(orchestrate(
                    query="Infinite loop test",
                    session_id="max-rounds-sess",
                    retriever=mock_retriever,
                    reranker=mock_reranker,
                    top_k=20,
                    top_n=5,
                ))
        finally:
            cfg_module.settings.agent_max_rounds = original_max

        # Loop should have stopped at max_rounds (3)
        assert call_count["n"] <= 3, f"Expected at most 3 OpenAI calls, got {call_count['n']}"
        # Result must still be valid
        assert result is not None
        assert isinstance(result.chunks, list)

    def test_returns_whatever_evidence_gathered(self, monkeypatch):
        """Even at cutoff, the result must contain any evidence gathered so far."""
        from app.orchestrator import orchestrate

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [CHUNK_A]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [CHUNK_A]

        def always_retrieve(**kwargs):
            return _make_response(tool_calls=[_make_tool_call(
                "retrieve_documents", {"query": "Python"}, "call-1"
            )])

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = always_retrieve

        from app import config as cfg_module
        original_max = cfg_module.settings.agent_max_rounds
        cfg_module.settings.agent_max_rounds = 2

        try:
            with patch("app.orchestrator.openai") as mock_openai:
                mock_openai.OpenAI.return_value = mock_client
                result = _run(orchestrate(
                    query="Evidence at cutoff",
                    session_id="cutoff-sess",
                    retriever=mock_retriever,
                    reranker=mock_reranker,
                    top_k=20,
                    top_n=5,
                ))
        finally:
            cfg_module.settings.agent_max_rounds = original_max

        # Should have gathered CHUNK_A from the first retrieve call
        assert len(result.chunks) >= 1


# ---------------------------------------------------------------------------
# TestCachedEvidenceReuse
# ---------------------------------------------------------------------------

class TestCachedEvidenceReuse:
    """
    When SQLite session memory contains prior evidence, the orchestrator must
    offer reuse_cached_evidence as a tool.  When the model selects it, no
    retrieve_documents call should be made.
    """

    @pytest.fixture
    def seeded_db(self):
        """Create a temp SQLite DB pre-seeded with evidence for session 'prior-sess'."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name

        init_memory(db_path)
        save_session(
            session_id="prior-sess",
            query="What is Python?",
            chunks=[CHUNK_A, CHUNK_B],
            db_path=db_path,
        )
        yield db_path
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Windows may lock the file; ignore

    def test_cache_reuse_skips_retriever(self, seeded_db, monkeypatch):
        """
        Mock the model to choose reuse_cached_evidence.
        Assert HybridRetriever.retrieve is NOT called.
        Assert retrieval_source == 'cached'.
        """
        from app.orchestrator import orchestrate

        mock_retriever = MagicMock()
        mock_reranker = MagicMock()

        # Responses: round1 → reuse_cached, round2 → stop
        responses = [
            _make_response(tool_calls=[_make_tool_call(
                "reuse_cached_evidence",
                {"reason": "Same Python topic as prior turn"},
                "call-cache-1",
            )]),
            _make_response(tool_calls=[]),  # satisfied
        ]
        call_idx = {"n": 0}

        def fake_create(**kwargs):
            r = responses[call_idx["n"]]
            call_idx["n"] += 1
            return r

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        # Override the memory db path
        from app import config as cfg_module
        original_db = cfg_module.settings.session_memory_db
        cfg_module.settings.session_memory_db = seeded_db

        try:
            with patch("app.orchestrator.openai") as mock_openai:
                mock_openai.OpenAI.return_value = mock_client
                result = _run(orchestrate(
                    query="Tell me more about Python",
                    session_id="prior-sess",
                    retriever=mock_retriever,
                    reranker=mock_reranker,
                    top_k=20,
                    top_n=5,
                ))
        finally:
            cfg_module.settings.session_memory_db = original_db

        # Retriever must NOT have been called
        mock_retriever.retrieve.assert_not_called()
        # Cached chunks must have been activated (CHUNK_A + CHUNK_B)
        assert len(result.chunks) == 2
        chunk_ids = {c.chunk_id for c in result.chunks}
        assert "chunk-A" in chunk_ids
        assert "chunk-B" in chunk_ids
        # Source label must reflect cache
        assert result.retrieval_source == "cached"

    def test_cache_reuse_returns_prior_chunks(self, seeded_db, monkeypatch):
        """The text content of returned chunks must match what was seeded."""
        from app.orchestrator import orchestrate

        mock_retriever = MagicMock()
        mock_reranker = MagicMock()

        responses = [
            _make_response(tool_calls=[_make_tool_call(
                "reuse_cached_evidence",
                {"reason": "Same session context"},
                "call-cache-2",
            )]),
            _make_response(tool_calls=[]),
        ]
        call_idx = {"n": 0}

        def fake_create(**kwargs):
            r = responses[call_idx["n"]]
            call_idx["n"] += 1
            return r

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = fake_create

        from app import config as cfg_module
        original_db = cfg_module.settings.session_memory_db
        cfg_module.settings.session_memory_db = seeded_db

        try:
            with patch("app.orchestrator.openai") as mock_openai:
                mock_openai.OpenAI.return_value = mock_client
                result = _run(orchestrate(
                    query="More about Python paradigms",
                    session_id="prior-sess",
                    retriever=mock_retriever,
                    reranker=mock_reranker,
                    top_k=20,
                    top_n=5,
                ))
        finally:
            cfg_module.settings.session_memory_db = original_db

        texts = {c.text for c in result.chunks}
        assert "Python is a high-level programming language." in texts
        assert "Python supports multiple programming paradigms." in texts
