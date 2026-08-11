"""
app.hallucination — claim-level hallucination detection

Pipeline:
    1. split_into_claims(answer)           → list of individual sentences/claims
    2. verify_claim(claim, context)        → (label, confidence) via NLI model
    3. detect(answer, citations, chunks)   → list[HallucinationFlag]

NLI labels:
    entailment   — claim is supported by the cited chunk
    neutral      — claim is neither supported nor contradicted (flagged if low confidence)
    contradiction — claim directly conflicts with the cited chunk (always flagged)
"""

from .detector import HallucinationDetector, split_into_claims, verify_claim

__all__ = ["HallucinationDetector", "split_into_claims", "verify_claim"]
