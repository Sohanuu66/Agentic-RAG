"""
backend.tests — pytest test suite

Test modules:
    test_ingestion    → parser fixtures, chunker sizes/overlap, indexer integration
    test_retrieval    → RRF math unit test, top-5 contains expected chunk
    test_generation   → mock Groq response, JSON parsing, invalid chunk_id rejection
    test_hallucination → claim splitting, known entailment pair, known contradiction pair
"""
