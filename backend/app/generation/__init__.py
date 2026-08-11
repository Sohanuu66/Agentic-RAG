"""
app.generation — citation-grounded LLM generation

Components:
    prompts   → format_context(chunks) + system prompt template
    generator → CitationGenerator.generate(query, context_chunks) → GenerationResult
                  - calls Groq in JSON mode
                  - validates that every cited chunk_id exists in provided context
"""

from .generator import CitationGenerator, GenerationResult
from .prompts import format_context, build_messages, SYSTEM_PROMPT

__all__ = [
    "CitationGenerator",
    "GenerationResult",
    "format_context",
    "build_messages",
    "SYSTEM_PROMPT",
]
