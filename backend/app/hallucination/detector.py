"""
backend/app/hallucination/detector.py
---------------------------------------
Claim-level hallucination detection using a cross-encoder NLI model
(``cross-encoder/nli-deberta-v3-base`` by default).

Pipeline
--------
1. ``split_into_claims(answer)``
       Sentence-tokenise the answer text into individual claim strings.

2. ``verify_claim(claim, context)``
       Run the NLI model on the (context, claim) pair and return
       ``(label, confidence)`` where label ∈ {entailment, neutral, contradiction}.

3. ``HallucinationDetector.detect(answer, citations, context_chunks)``
       For each claim, locate its most relevant cited chunk (or fall back to
       the full context), run NLI, and build a ``HallucinationFlag``.
       Flagging rules:
         - ``contradiction``   → always flagged
         - ``neutral``         → flagged if confidence < threshold
         - ``entailment``      → never flagged

Classes
-------
HallucinationDetector
    Wraps the NLI cross-encoder and exposes the public ``detect()`` method.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.models.response import Citation, HallucinationFlag
from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# deberta-v3 label order returned by sentence-transformers CrossEncoder
# The model outputs logits in the order: [contradiction, entailment, neutral]
# (verified against the HuggingFace model card)
_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}


# ── Claim splitter ─────────────────────────────────────────────────────────────

def split_into_claims(answer: str) -> List[str]:
    """
    Split an answer string into individual claim sentences.

    Uses a two-pass approach compatible with Python 3.10's fixed-width
    lookbehind restriction:
      1. Split on sentence-ending punctuation (. ! ?) followed by whitespace.
      2. Re-join fragments that appear to be abbreviations (e.g. "Dr.", "Mr.").
      3. Also split on newline-separated bullet / numbered list items.

    Parameters
    ----------
    answer:
        The full answer text (possibly multi-sentence, may contain markdown).

    Returns
    -------
    List[str]
        Non-empty, stripped sentences.  Markdown bullet prefixes (``-``, ``*``,
        numbered lists) are preserved so the semantic content is unchanged.
    """
    if not answer or not answer.strip():
        return []

    text = answer.strip()

    # ── Pass 1: split on ". " / "! " / "? " boundaries ──────────────────────
    # We use a simple positive lookbehind on a single fixed character (. ! ?)
    # to stay within Python 3.10 re constraints.
    raw = re.split(r'(?<=[.!?])\s+', text)

    # ── Pass 2: re-join splits that were caused by known abbreviations ────────
    # e.g. "Dr." or "Mr." at the end of a fragment shouldn't break sentences.
    ABBREVS = re.compile(
        r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Fig|Eq|No|Vol|pp|cf|i\.e|e\.g)\.$',
        re.IGNORECASE,
    )
    merged: List[str] = []
    carry = ""
    for fragment in raw:
        if carry:
            fragment = carry + " " + fragment
            carry = ""
        if ABBREVS.search(fragment.rstrip()):
            carry = fragment
        else:
            merged.append(fragment)
    if carry:
        merged.append(carry)

    # ── Pass 3: split on newline-separated bullet / numbered list items ───────
    sentences: List[str] = []
    for fragment in merged:
        sub = re.split(r'\n+(?=[\-\*\d])', fragment)
        sentences.extend(sub)

    return [s.strip() for s in sentences if s.strip()]


# ── Low-level NLI helper ──────────────────────────────────────────────────────

def verify_claim(
    claim: str,
    context: str,
    model,  # sentence_transformers.CrossEncoder instance
) -> Tuple[str, float]:
    """
    Run NLI on a single (context, claim) pair.

    The NLI model takes ``[context, hypothesis]`` as input and outputs
    logits over three labels: contradiction / entailment / neutral.
    We apply softmax to convert logits to probabilities and return the
    argmax label with its probability as the confidence.

    Parameters
    ----------
    claim:
        The hypothesis sentence to verify.
    context:
        The premise text (the cited chunk content).
    model:
        An already-loaded ``sentence_transformers.CrossEncoder`` instance.

    Returns
    -------
    Tuple[str, float]
        ``(label, confidence)`` where label ∈ {``"entailment"``,
        ``"neutral"``, ``"contradiction"``} and confidence ∈ [0, 1].
    """
    import numpy as np

    scores = model.predict([(context, claim)])  # shape (1, 3)
    logits = scores[0]

    # Softmax for probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    label_idx = int(np.argmax(probs))
    label = _LABEL_MAP.get(label_idx, "neutral")
    confidence = float(probs[label_idx])

    return label, confidence


# ── HallucinationDetector ─────────────────────────────────────────────────────

class HallucinationDetector:
    """
    Claim-level hallucination detector backed by a cross-encoder NLI model.

    Parameters
    ----------
    model_name:
        HuggingFace model name / path.  Defaults to
        ``cross-encoder/nli-deberta-v3-base``.
    threshold:
        Confidence threshold below which a ``neutral`` claim is flagged as
        a potential hallucination.  ``contradiction`` claims are always
        flagged regardless of confidence.  Default: ``0.7``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_NLI_MODEL,
        threshold: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model(self):
        """Lazily load the NLI CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            logger.info(
                "Loading NLI model '%s' (first use only) …", self.model_name
            )
            self._model = CrossEncoder(self.model_name, num_labels=3)
        return self._model

    @staticmethod
    def _build_chunk_index(
        context_chunks: List[RetrievalResult],
    ) -> Dict[str, RetrievalResult]:
        return {c.chunk_id: c for c in context_chunks}

    @staticmethod
    def _find_cited_context(
        chunk_id: Optional[str],
        chunk_index: Dict[str, RetrievalResult],
        context_chunks: List[RetrievalResult],
    ) -> str:
        """
        Return the context text to verify a claim against.

        If ``chunk_id`` is provided and exists in the index we use that
        chunk.  Otherwise we concatenate all context chunks (truncated to
        keep NLI input manageable).
        """
        if chunk_id and chunk_id in chunk_index:
            return chunk_index[chunk_id].text

        # Fallback: join all context texts, truncated to ~2 000 chars
        combined = " ".join(c.text for c in context_chunks)
        return combined[:2000]

    @staticmethod
    def _is_flagged(label: str, confidence: float, threshold: float) -> bool:
        """Return True if this claim should be flagged as a hallucination."""
        if label == "contradiction":
            return True
        if label == "neutral" and confidence < threshold:
            return True
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        answer: str,
        citations: List[Citation],
        context_chunks: List[RetrievalResult],
    ) -> List[HallucinationFlag]:
        """
        Detect potential hallucinations in *answer* at the claim level.

        For each sentence in the answer we:
          1. Find which ``Citation`` covers it (use the first cited chunk as
             a proxy; more sophisticated attribution is left for later).
          2. Retrieve the cited chunk's text as the NLI premise.
          3. Run the NLI model.
          4. Build a ``HallucinationFlag`` and flag if label is
             ``contradiction`` or ``neutral`` below the confidence threshold.

        Parameters
        ----------
        answer:
            The generated answer string.
        citations:
            The validated ``Citation`` list produced by ``CitationGenerator``.
        context_chunks:
            The reranked context chunks passed to the generator.

        Returns
        -------
        List[HallucinationFlag]
            One entry per claim sentence.  Empty if the answer has no
            detectable sentences or the context is empty.
        """
        claims = split_into_claims(answer)
        if not claims:
            logger.debug("HallucinationDetector: no claims found in answer.")
            return []

        if not context_chunks:
            logger.warning(
                "HallucinationDetector: no context chunks supplied — skipping."
            )
            return []

        model = self._get_model()
        chunk_index = self._build_chunk_index(context_chunks)

        # Map claim index → cited chunk_id (round-robin over citations)
        # This is a simple heuristic; a smarter approach would attribute each
        # claim to the most semantically similar chunk.
        citation_ids: List[Optional[str]] = []
        if citations:
            for i in range(len(claims)):
                citation_ids.append(citations[i % len(citations)].chunk_id)
        else:
            citation_ids = [None] * len(claims)

        flags: List[HallucinationFlag] = []
        for claim, cited_id in zip(claims, citation_ids):
            context_text = self._find_cited_context(cited_id, chunk_index, context_chunks)

            try:
                label, confidence = verify_claim(claim, context_text, model)
            except Exception as exc:
                logger.error(
                    "NLI inference failed for claim %r: %s — defaulting to neutral.",
                    claim[:60],
                    exc,
                )
                label, confidence = "neutral", 0.0

            flagged = self._is_flagged(label, confidence, self.threshold)

            flags.append(
                HallucinationFlag(
                    claim=claim,
                    label=label,  # type: ignore[arg-type]
                    confidence=confidence,
                    cited_chunk_id=cited_id,
                    flagged=flagged,
                )
            )

            logger.debug(
                "Claim: %r  →  label=%s  conf=%.3f  flagged=%s",
                claim[:60],
                label,
                confidence,
                flagged,
            )

        flagged_count = sum(1 for f in flags if f.flagged)
        logger.info(
            "HallucinationDetector: %d/%d claims flagged.",
            flagged_count,
            len(flags),
        )
        return flags
