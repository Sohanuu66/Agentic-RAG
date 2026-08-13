"""
backend/app/orchestrator.py
----------------------------
Agentic evidence-gathering orchestrator for the /query pipeline.

The orchestrator replaces the unconditional retrieve→rerank linear flow with an
OpenAI tool-calling loop that lets the model decide how to gather evidence for
each query.  The raw retrieval/reranking/generation/detection modules are
unchanged — this module only decides *which* evidence-gathering tools to call.

Tools exposed to the model
--------------------------
1. reuse_cached_evidence(reason)
       Activate previously retrieved chunks from SQLite session memory.
       Returns early if the current session has usable cached evidence.

2. retrieve_documents(query)
       Run HybridRetriever + CrossEncoderReranker and return results.
       The tool result includes ``top_score`` and ``below_threshold`` so the
       model has a concrete signal for whether to supplement with web search
       (Addendum 2).

3. web_search(query)
       Call the Tavily API and convert results to RetrievalResult objects.
       The model should call this when retrieve_documents returns
       below_threshold=true or finds zero results.

Guardrails
----------
* Cold-start guard (Addendum 3): the loop cannot exit into generation with an
  empty evidence list on a first-turn query (no session_id match in SQLite).
  If the model tries to stop before retrieving anything, a reminder message
  is injected and the loop continues.

* Max-rounds cap: the for-loop runs at most ``agent_max_rounds`` iterations.
  Whatever evidence has been gathered so far is returned when the cap is hit.

Fallback (Addendum 5)
---------------------
Any exception in the agentic loop is caught, logged with the error message, and
the function falls back to a direct HybridRetriever+Reranker call so the
request always produces a response.

OrchestratorResult
------------------
The public ``orchestrate()`` coroutine returns an ``OrchestratorResult``
dataclass containing:
    chunks           List[RetrievalResult] — assembled evidence
    session_id       str
    retrieval_source str  — 'documents' | 'web' | 'cached' | 'hybrid' | 'fallback'
    agent_rounds     int  — number of tool-call rounds executed
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import openai  # type: ignore[import]

from app.config import settings
from app.memory import get_session_context, load_cached_chunks, save_session
from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """Return value of orchestrate()."""
    chunks: List[RetrievalResult] = field(default_factory=list)
    session_id: str = ""
    retrieval_source: str = "documents"
    agent_rounds: int = 0


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "reuse_cached_evidence",
            "description": (
                "Activate cached evidence from the previous turn of this session when it is "
                "still clearly and directly relevant to the current query. "
                "Prefer this for follow-up questions that rely on the same documents. "
                "Do NOT use it when the user asks for a different topic or new detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short explanation of why the cached evidence is still relevant.",
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Retrieve evidence from the indexed document collection using hybrid "
                "dense+sparse search followed by cross-encoder reranking. "
                "Use this when cached evidence is missing or not relevant. "
                "The result includes top_score and below_threshold — if below_threshold "
                "is true, consider calling web_search to supplement the evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Self-contained evidence request. "
                            "For follow-up questions rewrite to include omitted subject details."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web using Tavily and return relevant snippets as evidence. "
                "Call this when retrieve_documents returns below_threshold=true or finds "
                "zero results, OR when the query likely requires up-to-date information "
                "not present in the indexed documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query string.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an evidence-gathering orchestrator for a document-QA system.
Your sole job is to collect high-quality evidence chunks before a separate
citation-grounded generation step produces the final answer.

Available tools:
1. reuse_cached_evidence — use cached evidence from a prior turn of this session.
2. retrieve_documents    — hybrid search over the indexed document collection.
3. web_search            — Tavily web search for real-time or supplemental evidence.

Decision rules:
- If the session has cached evidence that is still relevant, call reuse_cached_evidence first.
- Always call retrieve_documents on a fresh query unless cache was reused.
- If retrieve_documents returns below_threshold: true OR chunks_found: 0, call web_search.
- Stop calling tools when you have enough evidence to answer the query.
- Do NOT attempt to generate an answer yourself — stop tool calls once evidence is ready.
"""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _run_retrieve_documents(
    query: str,
    retriever,
    reranker,
    top_k: int,
    top_n: int,
) -> tuple[List[RetrievalResult], dict]:
    """
    Synchronous: run hybrid retrieval + cross-encoder reranking.
    Returns (chunks, tool_result_dict).
    """
    candidates = retriever.retrieve(query=query, top_k=top_k)
    if not candidates:
        return [], {
            "chunks_found": 0,
            "top_score": 0.0,
            "below_threshold": True,
            "preview": "(no results)",
        }

    try:
        reranked = reranker.rerank(query=query, candidates=candidates, top_n=top_n)
    except Exception as exc:
        logger.warning("Reranker failed in orchestrator: %s — using raw candidates.", exc)
        reranked = candidates[:top_n]

    top_score = reranked[0].score if reranked else 0.0
    below_threshold = top_score < settings.web_search_score_threshold
    preview = reranked[0].text[:200].strip() if reranked else "(empty)"

    result = {
        "chunks_found": len(reranked),
        "top_score": round(top_score, 4),
        "below_threshold": below_threshold,
        "preview": preview,
    }
    return reranked, result


def _run_web_search(query: str) -> tuple[List[RetrievalResult], dict]:
    """
    Synchronous: call Tavily and convert results to RetrievalResult objects.
    Returns (chunks, tool_result_dict).
    """
    if not settings.tavily_api_key:
        return [], {"error": "Tavily API key not configured — web search unavailable."}

    try:
        from tavily import TavilyClient  # type: ignore[import]
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=5)
        results = response.get("results", [])
    except Exception as exc:
        logger.warning("Tavily web search failed: %s", exc)
        return [], {"error": str(exc)}

    chunks: List[RetrievalResult] = []
    for i, r in enumerate(results):
        chunks.append(
            RetrievalResult(
                chunk_id=f"web-{i}",
                text=r.get("content", r.get("snippet", "")),
                metadata={
                    "source": r.get("url", "web"),
                    "title": r.get("title", ""),
                    "section": "web_search",
                },
                score=r.get("score", 0.5),
            )
        )

    return chunks, {"results_found": len(chunks), "query": query}


# ---------------------------------------------------------------------------
# Fallback direct pipeline
# ---------------------------------------------------------------------------

async def _fallback_direct_retrieve(
    query: str,
    session_id: str,
    retriever,
    reranker,
    top_k: int,
    top_n: int,
) -> OrchestratorResult:
    """
    Direct retrieve→rerank path used when the agentic loop fails.
    Runs synchronously inside a thread executor so the event loop stays free.
    """
    def _run():
        candidates = retriever.retrieve(query=query, top_k=top_k)
        if not candidates:
            return []
        try:
            return reranker.rerank(query=query, candidates=candidates, top_n=top_n)
        except Exception:
            return candidates[:top_n]

    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(None, _run)
    return OrchestratorResult(
        chunks=chunks,
        session_id=session_id,
        retrieval_source="fallback",
        agent_rounds=0,
    )


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

async def orchestrate(
    query: str,
    session_id: str,
    retriever,
    reranker,
    top_k: int,
    top_n: int,
) -> OrchestratorResult:
    """
    Run the agentic evidence-gathering loop and return an OrchestratorResult.

    Parameters
    ----------
    query       : The user's natural-language question.
    session_id  : Session identifier for memory lookup.
    retriever   : HybridRetriever singleton (passed in from routes_query).
    reranker    : CrossEncoderReranker singleton.
    top_k       : Number of candidates to retrieve.
    top_n       : Number to keep after reranking.

    Returns
    -------
    OrchestratorResult with assembled evidence chunks.
    """
    try:
        return await _agentic_loop(query, session_id, retriever, reranker, top_k, top_n)
    except Exception as exc:
        # Addendum 5: log the caught exception so the fallback is observable
        logger.warning(
            "Orchestrator agentic loop failed, falling back to direct pipeline: %s", exc
        )
        return await _fallback_direct_retrieve(query, session_id, retriever, reranker, top_k, top_n)


async def _agentic_loop(
    query: str,
    session_id: str,
    retriever,
    reranker,
    top_k: int,
    top_n: int,
) -> OrchestratorResult:
    """Internal: the actual OpenAI tool-calling loop."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    loop = asyncio.get_running_loop()

    # ── Load session context ──────────────────────────────────────────────────
    session_ctx = get_session_context(session_id, settings.session_memory_db)
    cached_chunks: List[RetrievalResult] = (
        load_cached_chunks(session_id, settings.session_memory_db)
        if session_ctx["has_cache"]
        else []
    )

    # ── Orchestrator state ────────────────────────────────────────────────────
    state = {
        "evidence_chunks": [],
        "retrieval_attempted": False,
        "cache_used": False,
        "web_used": False,
        "sources_used": set(),
    }

    # ── Build initial messages ────────────────────────────────────────────────
    cache_summary = (
        f"Cached evidence available: {len(cached_chunks)} chunks from prior turn "
        f"(query: '{session_ctx['last_query'][:80]}')."
        if cached_chunks
        else "No cached evidence available for this session."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User query: {query}\n\n"
                f"Session context: {cache_summary}\n"
                "Gather the evidence needed to answer this query."
            ),
        },
    ]

    agent_rounds = 0

    # ── Tool-calling loop ─────────────────────────────────────────────────────
    for round_num in range(settings.agent_max_rounds):
        response = await loop.run_in_executor(
            None,
            lambda msgs=messages: client.chat.completions.create(
                model=settings.llm_model,
                messages=msgs,
                tools=_TOOL_SCHEMAS,
                tool_choice="auto",
            ),
        )

        tool_calls = response.choices[0].message.tool_calls or []

        # Addendum 3 — cold-start guardrail
        if (
            not tool_calls
            and not state["evidence_chunks"]
            and not state["cache_used"]
            and not state["retrieval_attempted"]
        ):
            logger.warning(
                "Orchestrator: model tried to stop with zero evidence on cold-start "
                "(round %d). Injecting retrieval reminder.", round_num
            )
            messages.append({
                "role": "user",
                "content": (
                    "You have not yet retrieved any evidence. "
                    "You MUST call retrieve_documents before finishing."
                ),
            })
            continue  # do not increment agent_rounds, do not break

        agent_rounds += 1

        if not tool_calls:
            # Model is satisfied with current evidence — exit loop
            break

        # Append the assistant's tool-call message to the conversation
        messages.append(response.choices[0].message)

        # ── Execute each tool call ────────────────────────────────────────────
        tool_results = []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if name == "reuse_cached_evidence":
                if state["evidence_chunks"]:
                    output = {"status": "evidence_already_active"}
                elif cached_chunks:
                    state["evidence_chunks"] = cached_chunks
                    state["cache_used"] = True
                    state["sources_used"].add("cached")
                    logger.info(
                        "Orchestrator: reusing %d cached chunks (reason: %s).",
                        len(cached_chunks),
                        args.get("reason", ""),
                    )
                    output = {
                        "status": "activated",
                        "chunks_activated": len(cached_chunks),
                        "reason": args.get("reason", ""),
                    }
                else:
                    output = {"status": "no_cache_available"}

            elif name == "retrieve_documents":
                if state["retrieval_attempted"]:
                    output = {"status": "retrieval_already_attempted"}
                elif state["evidence_chunks"]:
                    output = {"status": "evidence_already_active"}
                else:
                    state["retrieval_attempted"] = True
                    retrieve_query = args.get("query", query)
                    chunks, tool_output = await loop.run_in_executor(
                        None,
                        lambda q=retrieve_query: _run_retrieve_documents(
                            q, retriever, reranker, top_k, top_n
                        ),
                    )
                    if chunks:
                        state["evidence_chunks"].extend(chunks)
                        state["sources_used"].add("documents")
                    output = tool_output
                    logger.info(
                        "Orchestrator: retrieve_documents → %d chunks, top_score=%.4f, "
                        "below_threshold=%s.",
                        tool_output["chunks_found"],
                        tool_output["top_score"],
                        tool_output["below_threshold"],
                    )

            elif name == "web_search":
                if state["web_used"]:
                    output = {"status": "web_search_already_attempted"}
                else:
                    state["web_used"] = True
                    search_query = args.get("query", query)
                    chunks, tool_output = await loop.run_in_executor(
                        None,
                        lambda q=search_query: _run_web_search(q),
                    )
                    if chunks:
                        state["evidence_chunks"].extend(chunks)
                        state["sources_used"].add("web")
                    output = tool_output
                    logger.info(
                        "Orchestrator: web_search → %d results.", tool_output.get("results_found", 0)
                    )

            else:
                output = {"error": f"Unknown tool: {name}"}
                logger.warning("Orchestrator: unknown tool called: %s", name)

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(output),
            })

        messages.extend(tool_results)

    # ── Determine retrieval_source label ──────────────────────────────────────
    sources = state["sources_used"]
    if not sources:
        retrieval_source = "fallback"
    elif len(sources) == 1:
        retrieval_source = next(iter(sources))
    else:
        retrieval_source = "hybrid"

    final_chunks = state["evidence_chunks"]

    # ── Persist to session memory if new evidence was retrieved ───────────────
    if final_chunks and not state["cache_used"]:
        try:
            save_session(session_id, query, final_chunks, settings.session_memory_db)
        except Exception as exc:
            logger.warning("Failed to save session evidence to memory: %s", exc)

    logger.info(
        "Orchestrator complete: rounds=%d, chunks=%d, source=%s.",
        agent_rounds, len(final_chunks), retrieval_source,
    )

    return OrchestratorResult(
        chunks=final_chunks,
        session_id=session_id,
        retrieval_source=retrieval_source,
        agent_rounds=agent_rounds,
    )
