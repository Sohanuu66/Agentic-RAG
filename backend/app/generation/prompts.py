"""
backend/app/generation/prompts.py
-----------------------------------
Prompt templates and context-formatting utilities for the citation-grounded
generation step.

The system prompt instructs the model to:
  - Answer *only* from the provided context.
  - Return a strict JSON object with keys ``answer`` and ``citations``.
  - Never cite a chunk_id that was not in the supplied context.

Functions
---------
format_context(chunks)
    Turn a list of RetrievalResult objects into a numbered, ID-tagged
    context block that is safe to insert into the user turn.

build_messages(query, context_chunks)
    Assemble the full messages list (system + user) ready to send to the
    Groq chat API.

build_retry_messages(query, context_chunks)
    Simpler fallback prompt used when the primary call fails JSON validation.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.dense import RetrievalResult


# ── System prompt ──────────────────────────────────────────────────────────────
# NOTE: Keep this concise. Do NOT embed a JSON example here — Llama-3 tends to
# echo it verbatim, which confuses Groq's json_object validator. The schema is
# described only in words; the user turn carries a concrete example.

SYSTEM_PROMPT = """\
You are a precise, citation-grounded research assistant.

Rules you MUST follow:
1. Answer the user's question using ONLY the information in the CONTEXT block.
2. If the context does not contain enough information, respond with:
   {"answer": "I don't have enough information in the provided documents to answer this.", "citations": []}
3. Your entire response must be a single raw JSON object — no markdown fences,
   no preamble, no trailing text.  The object must have exactly two keys:
   "answer" (string) and "citations" (array of chunk_id strings).
4. Only include chunk_ids that appear verbatim in the CONTEXT block.
5. Do NOT invent facts or cite sources outside the CONTEXT block.
"""


# ── Retry system prompt (simpler, used as fallback) ───────────────────────────

RETRY_SYSTEM_PROMPT = """\
You are a JSON-only assistant. Respond with a single raw JSON object and nothing else.
The object must have exactly two keys: "answer" and "citations".
"answer" is a string. "citations" is an array of chunk_id strings from the CONTEXT.
"""


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(chunks: "List[RetrievalResult]") -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is rendered as::

        [chunk_id: <id>]
        Source: <source> | Page: <page> | Section: <section>
        <text>

    Parameters
    ----------
    chunks:
        List of ``RetrievalResult`` objects from the retrieval/reranking stage.

    Returns
    -------
    str
        A multi-line string ready to be embedded in the user message.
    """
    if not chunks:
        return "(No context available.)"

    lines: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page_num", "—")
        section = meta.get("section", "—")

        lines.append(f"[{i}] [chunk_id: {chunk.chunk_id}]")
        lines.append(f"Source: {source} | Page: {page} | Section: {section}")
        lines.append(chunk.text.strip())
        lines.append("")          # blank line between chunks

    return "\n".join(lines).rstrip()


# ── Message builder ───────────────────────────────────────────────────────────

def build_messages(query: str, context_chunks: "List[RetrievalResult]") -> list:
    """
    Build the ``messages`` list for the Groq chat completion API.

    The user turn carries a concrete JSON example so the model knows the exact
    output shape, without the system prompt repeating it (which caused Llama-3
    to echo the example and trip Groq's json_object validator).

    Parameters
    ----------
    query:
        The user's natural-language question.
    context_chunks:
        Reranked chunks to embed as context.

    Returns
    -------
    list
        ``[{"role": "system", ...}, {"role": "user", ...}]``
    """
    context_block = format_context(context_chunks)

    user_content = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        'Respond with a raw JSON object only. Example shape (do not copy — use real data):\n'
        '{"answer": "...", "citations": ["chunk_id_a", "chunk_id_b"]}'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_retry_messages(query: str, context_chunks: "List[RetrievalResult]") -> list:
    """
    Simplified fallback prompt used when the primary call fails JSON validation.

    Strips most instructions down to the bare minimum to maximise the chance
    of getting a parseable JSON blob from the model.
    """
    context_block = format_context(context_chunks)

    # Provide the chunk_ids explicitly so the model can copy them
    chunk_ids = [c.chunk_id for c in context_chunks]

    user_content = (
        f"CONTEXT:\n{context_block}\n\n"
        f"Available chunk_ids: {chunk_ids}\n\n"
        f"QUESTION: {query}\n\n"
        "Output ONLY a JSON object like: "
        '{"answer": "your answer here", "citations": ["chunk_id_1"]}'
    )

    return [
        {"role": "system", "content": RETRY_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
