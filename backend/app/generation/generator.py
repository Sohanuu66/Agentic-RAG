"""
backend/app/generation/generator.py
--------------------------------------
Citation-grounded answer generation using the OpenAI Chat Completions API
(gpt-4o-mini by default).

The generator:
  1. Formats context chunks into the prompt via ``build_messages``.
  2. Calls the OpenAI chat-completion endpoint in **plain-text mode** (no
     ``response_format`` constraint) to stay robust when the model emits a
     reasoning preamble before the JSON object.
  3. Extracts the JSON block from the raw text using a regex extractor that
     handles markdown fences and bare JSON objects.
  4. If parsing fails, retries once with a simpler prompt via
     ``build_retry_messages``.
  5. Validates that every cited chunk_id is actually present in the supplied
     context (reject any ghost citations).
  6. Converts valid citations to ``Citation`` Pydantic models and returns a
     ``GenerationResult`` with latency.

Why plain-text mode instead of json_object mode?
-------------------------------------------------
Using ``response_format={"type": "json_object"}`` can raise errors when the
model starts with a reasoning preamble before the JSON object.  Plain-text
mode + regex extraction is more robust and produces identical downstream
behaviour.

Classes
-------
GenerationResult
    Dataclass holding answer, citations, and wall-clock latency.
CitationGenerator
    Wraps the OpenAI client and exposes ``generate(query, context_chunks)``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List

from app.generation.prompts import build_messages, build_retry_messages
from app.models.response import Citation
from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TOKENS = 2048   # raised from 1024 to prevent mid-JSON truncation
_MAX_RETRIES = 1             # one retry with the simplified prompt


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Output of ``CitationGenerator.generate()``.

    Attributes
    ----------
    answer:
        The LLM-generated answer string (may contain markdown).
    citations:
        Validated ``Citation`` objects whose chunk_ids were present in the
        supplied context.
    latency_ms:
        Wall-clock time from the start of the Groq API call to parse
        completion, in milliseconds.
    raw_chunk_ids:
        The raw list of chunk_ids returned by the model before validation
        (useful for debugging / logging).
    """

    answer: str
    citations: List[Citation] = field(default_factory=list)
    latency_ms: float = 0.0
    raw_chunk_ids: List[str] = field(default_factory=list)


# ── CitationGenerator ─────────────────────────────────────────────────────────

class CitationGenerator:
    """
    Generate citation-grounded answers via the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key.
    model:
        OpenAI model name.  Defaults to ``gpt-4o-mini``.
    temperature:
        Sampling temperature.  Use 0 for fully deterministic outputs.
    max_tokens:
        Maximum tokens in the completion.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self):
        """Lazily instantiate the OpenAI client."""
        if self._client is None:
            import openai  # type: ignore[import]
            self._client = openai.OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialised (model=%s).", self.model)
        return self._client

    @staticmethod
    def _build_context_index(chunks: List[RetrievalResult]) -> dict[str, RetrievalResult]:
        """Map chunk_id → RetrievalResult for O(1) look-ups."""
        return {c.chunk_id: c for c in chunks}

    @staticmethod
    def _extract_json(content: str) -> dict:
        """
        Extract the first JSON object from *content*, handling:
          - Bare JSON objects  ``{ ... }``
          - Markdown fences    ```json\\n{ ... }\\n```
          - Preamble text      "Here is my answer:\\n{ ... }"

        Raises
        ------
        ValueError
            If no valid JSON object is found or required keys are missing.
        """
        text = content.strip()

        # 1. Try stripping markdown fences
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            re.DOTALL,
        )
        if fence_match:
            text = fence_match.group(1)

        # 2. Try extracting the first {...} block (handles preamble text)
        if not text.startswith("{"):
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response contains no valid JSON: {exc}\nRaw content:\n{content[:500]}"
            ) from exc

        if "answer" not in data:
            raise ValueError(
                f"LLM JSON missing 'answer' key. Got keys: {list(data.keys())}"
            )
        if "citations" not in data:
            raise ValueError(
                f"LLM JSON missing 'citations' key. Got keys: {list(data.keys())}"
            )

        return data

    @staticmethod
    def _validate_citations(
        raw_ids: List[str],
        context_index: dict[str, RetrievalResult],
    ) -> List[Citation]:
        """
        Reject any chunk_id not present in the supplied context and convert
        the remaining ones to ``Citation`` Pydantic models.

        This is the security / hallucination guard: the model must not be
        allowed to cite documents it was not given.

        Parameters
        ----------
        raw_ids:
            chunk_ids returned by the LLM.
        context_index:
            Map of chunk_id → RetrievalResult for all chunks in the context.

        Returns
        -------
        List[Citation]
            Only citations whose chunk_id exists in ``context_index``.
        """
        citations: List[Citation] = []
        seen: set[str] = set()

        for cid in raw_ids:
            if not isinstance(cid, str):
                logger.warning("Ignoring non-string citation: %r", cid)
                continue
            if cid in seen:
                continue
            seen.add(cid)

            if cid not in context_index:
                logger.warning(
                    "LLM cited chunk_id '%s' which is not in the provided context — REJECTED.",
                    cid,
                )
                continue

            chunk = context_index[cid]
            meta = chunk.metadata or {}

            citations.append(
                Citation(
                    chunk_id=cid,
                    source=str(meta.get("source", "unknown")),
                    page_num=meta.get("page_num"),
                    section=meta.get("section"),
                    snippet=chunk.text[:200].strip(),
                )
            )

        return citations

    def _call_openai(self, messages: list) -> str:
        """
        Make a single OpenAI chat completion call in plain-text mode.

        Returns the raw content string. Raises on API error.

        Plain-text mode is used deliberately (no ``response_format`` arg)
        to stay robust when the model emits a reasoning preamble before the
        JSON object — which can cause json_object mode to fail validation.
        """
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            # NOTE: response_format intentionally omitted — see module docstring
        )
        return response.choices[0].message.content or ""

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievalResult],
    ) -> GenerationResult:
        """
        Generate a citation-grounded answer for *query* given *context_chunks*.

        Steps:
          1. Format chunks into the prompt.
          2. Call Groq in plain-text mode (no json_object enforcement).
          3. Extract the JSON block from the raw response.
          4. If parsing fails, retry once with the simplified prompt.
          5. Reject any cited chunk_id not in our context.
          6. Return a ``GenerationResult`` with latency.

        Parameters
        ----------
        query:
            The natural-language question.
        context_chunks:
            Reranked list of ``RetrievalResult`` objects; these are the only
            chunks the model is allowed to cite.

        Returns
        -------
        GenerationResult
        """
        context_index = self._build_context_index(context_chunks)

        logger.info(
            "CitationGenerator: calling OpenAI model='%s' with %d context chunks …",
            self.model,
            len(context_chunks),
        )

        t0 = time.perf_counter()

        # ── Attempt 1: primary prompt ──────────────────────────────────────────
        raw_content = ""
        data: dict | None = None
        last_exc: Exception | None = None

        for attempt, messages in enumerate(
            [
                build_messages(query, context_chunks),
                build_retry_messages(query, context_chunks),
            ],
            start=1,
        ):
            if attempt > 1:
                logger.warning(
                    "CitationGenerator: attempt %d/%d — using simplified retry prompt.",
                    attempt,
                    _MAX_RETRIES + 1,
                )

            try:
                raw_content = self._call_openai(messages)
            except Exception as exc:
                logger.error("OpenAI API call failed (attempt %d): %s", attempt, exc)
                last_exc = exc
                if attempt > _MAX_RETRIES:
                    raise
                continue

            logger.debug(
                "OpenAI raw response attempt %d (%d chars): %.300s…",
                attempt,
                len(raw_content),
                raw_content,
            )

            try:
                data = self._extract_json(raw_content)
                break  # success — stop retrying
            except ValueError as exc:
                logger.warning(
                    "JSON extraction failed (attempt %d): %s",
                    attempt,
                    exc,
                )
                last_exc = exc
                if attempt > _MAX_RETRIES:
                    # Exhausted retries — return graceful degradation
                    logger.error(
                        "All %d generation attempts failed — returning fallback answer.",
                        _MAX_RETRIES + 1,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return GenerationResult(
                        answer=(
                            "I was unable to generate a structured response. "
                            "Please try rephrasing your question."
                        ),
                        citations=[],
                        latency_ms=latency_ms,
                        raw_chunk_ids=[],
                    )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if data is None:
            # Defensive — should not reach here, but satisfy type checker
            raise RuntimeError("Generation loop exited without data or exception") from last_exc

        answer: str = data.get("answer", "")
        raw_ids: List[str] = data.get("citations", [])

        # Validate citations — reject any chunk_id not in our context
        valid_citations = self._validate_citations(raw_ids, context_index)

        logger.info(
            "CitationGenerator: answer generated via OpenAI (latency=%.1f ms, "
            "raw_citations=%d, valid_citations=%d).",
            latency_ms,
            len(raw_ids),
            len(valid_citations),
        )

        return GenerationResult(
            answer=answer,
            citations=valid_citations,
            latency_ms=latency_ms,
            raw_chunk_ids=raw_ids,
        )
