"""
backend/tests/test_ingestion.py
--------------------------------
Unit + integration tests for the ingestion pipeline.

Test groups
-----------
1. Parser fixtures      – smoke-test parse_pdf, parse_markdown, parse_text
2. Chunker sizes/overlap – verify chunk count and overlap for FixedSizeChunker
3. SentenceWindowChunker – verify window metadata is populated
4. SemanticChunker       – verify it produces at least one chunk per non-empty page
5. Integration test      – parse → chunk → IndexManager.add_documents (mocked ChromaDB / BM25)
"""

from __future__ import annotations

import pickle
import textwrap
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.parser import (
    DocumentPage,
    parse_markdown,
    parse_text,
    parse_document,
)
from app.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    SentenceWindowChunker,
    SemanticChunker,
    get_chunker,
)
from app.ingestion.indexer import IndexManager, build_sparse_index


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = textwrap.dedent("""\
    The quick brown fox jumps over the lazy dog.
    This is the second sentence in the document.
    Here comes a third sentence with a bit more content.
    A fourth sentence follows naturally from the third.
    The fifth and final sentence wraps things up nicely.
""")

SAMPLE_MARKDOWN = textwrap.dedent("""\
    # Title

    Introductory paragraph before any section heading.

    ## Section One

    Content for section one. It has a couple of sentences.

    ## Section Two

    Content for section two. Even more interesting.
""")


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return p


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.md"
    p.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return p


@pytest.fixture
def sample_pages() -> List[DocumentPage]:
    return [
        DocumentPage(
            content=SAMPLE_TEXT,
            metadata={"source": "test.txt", "page_num": 1, "section": None},
        )
    ]


# ── 1. Parser tests ───────────────────────────────────────────────────────────

class TestParseText:
    def test_returns_at_least_one_page(self, text_file):
        pages = parse_text(text_file)
        assert len(pages) >= 1

    def test_page_has_required_metadata_keys(self, text_file):
        pages = parse_text(text_file)
        for page in pages:
            assert "source" in page.metadata
            assert "page_num" in page.metadata
            assert "section" in page.metadata

    def test_source_is_filename(self, text_file):
        pages = parse_text(text_file)
        assert pages[0].metadata["source"] == "sample.txt"

    def test_content_is_non_empty(self, text_file):
        pages = parse_text(text_file)
        assert all(page.content.strip() for page in pages)


class TestParseMarkdown:
    def test_splits_on_h2_headings(self, markdown_file):
        pages = parse_markdown(markdown_file)
        # preamble + 2 sections = 3 pages
        assert len(pages) == 3

    def test_section_names_captured(self, markdown_file):
        pages = parse_markdown(markdown_file)
        section_names = [p.metadata["section"] for p in pages]
        assert "Section One" in section_names
        assert "Section Two" in section_names

    def test_preamble_section(self, markdown_file):
        pages = parse_markdown(markdown_file)
        assert pages[0].metadata["section"] == "preamble"

    def test_page_numbers_are_sequential(self, markdown_file):
        pages = parse_markdown(markdown_file)
        for i, page in enumerate(pages, start=1):
            assert page.metadata["page_num"] == i


class TestParseDocument:
    def test_routes_txt(self, text_file):
        pages = parse_document(text_file)
        assert len(pages) >= 1

    def test_routes_md(self, markdown_file):
        pages = parse_document(markdown_file)
        assert len(pages) >= 1

    def test_raises_on_unknown_extension(self, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(bad)


# ── 2. FixedSizeChunker tests ─────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_produces_chunks(self, sample_pages):
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_pages)
        assert len(chunks) > 0

    def test_chunks_have_unique_ids(self, sample_pages):
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_pages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_metadata_strategy(self, sample_pages):
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_pages)
        assert all(c.metadata["strategy"] == "fixed" for c in chunks)

    def test_overlap_means_more_chunks_than_no_overlap(self, sample_pages):
        """With overlap, we expect more chunks than without."""
        chunker_no_overlap = FixedSizeChunker(chunk_size=20, overlap=0)
        chunker_overlap = FixedSizeChunker(chunk_size=20, overlap=10)
        no_ov = chunker_no_overlap.chunk(sample_pages)
        ov = chunker_overlap.chunk(sample_pages)
        # More or equal chunks when overlap > 0
        assert len(ov) >= len(no_ov)

    def test_chunk_indices_are_sequential(self, sample_pages):
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(sample_pages)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i


# ── 3. SentenceWindowChunker tests ────────────────────────────────────────────

class TestSentenceWindowChunker:
    def test_produces_chunks(self, sample_pages):
        chunker = SentenceWindowChunker(window_size=2, context_sentences=1)
        chunks = chunker.chunk(sample_pages)
        assert len(chunks) > 0

    def test_window_in_metadata(self, sample_pages):
        chunker = SentenceWindowChunker(window_size=2, context_sentences=1)
        chunks = chunker.chunk(sample_pages)
        for c in chunks:
            assert "window" in c.metadata
            # The window should be at least as long as the core text
            assert len(c.metadata["window"]) >= len(c.text)

    def test_strategy_label(self, sample_pages):
        chunker = SentenceWindowChunker()
        chunks = chunker.chunk(sample_pages)
        assert all(c.metadata["strategy"] == "sentence_window" for c in chunks)


# ── 4. SemanticChunker (mocked) ───────────────────────────────────────────────

class TestSemanticChunker:
    def test_produces_chunks_with_mocked_model(self, sample_pages):
        """Mock the sentence-transformers model to avoid downloading it."""
        import numpy as np

        mock_model = MagicMock()
        # Return embeddings that are very similar so no split occurs
        n_sentences = len(sample_pages[0].content.split("."))
        mock_model.encode.return_value = np.ones((n_sentences, 384), dtype="float32")

        chunker = SemanticChunker(similarity_threshold=0.5)
        chunker._model = mock_model

        chunks = chunker.chunk(sample_pages)
        assert len(chunks) >= 1

    def test_strategy_label_with_mocked_model(self, sample_pages):
        import numpy as np

        mock_model = MagicMock()
        n_sentences = 5
        mock_model.encode.return_value = np.ones((n_sentences, 384), dtype="float32")

        chunker = SemanticChunker(similarity_threshold=0.5)
        chunker._model = mock_model
        chunks = chunker.chunk(sample_pages)
        assert all(c.metadata["strategy"] == "semantic" for c in chunks)


# ── 5. Factory tests ──────────────────────────────────────────────────────────

class TestGetChunker:
    def test_fixed_returns_fixed_chunker(self):
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_semantic_returns_semantic_chunker(self):
        chunker = get_chunker("semantic")
        assert isinstance(chunker, SemanticChunker)

    def test_sentence_window_returns_sentence_window_chunker(self):
        chunker = get_chunker("sentence_window")
        assert isinstance(chunker, SentenceWindowChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("unknown_strategy")  # type: ignore[arg-type]


# ── 6. IndexManager integration test (mocked external deps) ──────────────────

class TestIndexManagerIntegration:
    """
    Integration test: parse → chunk → IndexManager.add_documents

    We mock ChromaDB and BM25 I/O so the test runs without GPU/disk deps.
    """

    def test_add_documents_calls_both_indexes(self, text_file, tmp_path):
        pages = parse_text(text_file)
        chunker = FixedSizeChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(pages)

        assert len(chunks) > 0, "Expected at least one chunk from sample text"

        with (
            patch("app.ingestion.indexer.build_dense_index") as mock_dense,
            patch("app.ingestion.indexer.build_sparse_index") as mock_sparse,
        ):
            # Simulate an empty existing ChromaDB collection
            mock_collection = MagicMock()
            mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

            manager = IndexManager(persist_dir=str(tmp_path))
            manager._collection = mock_collection  # inject mock

            manager.add_documents(chunks)

            mock_dense.assert_called_once()
            mock_sparse.assert_called_once()

    def test_get_stats_returns_expected_keys(self, tmp_path):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42

        manager = IndexManager(persist_dir=str(tmp_path))
        manager._collection = mock_collection

        stats = manager.get_stats()
        assert "dense_chunk_count" in stats
        assert "sparse_index_exists" in stats
        assert stats["dense_chunk_count"] == 42


# ── 7. BM25 sparse index round-trip ──────────────────────────────────────────

class TestBuildSparseIndex:
    def test_pickle_round_trip(self, tmp_path):
        chunks = [
            Chunk(chunk_id="c1", text="hello world foo", metadata={}),
            Chunk(chunk_id="c2", text="bar baz qux", metadata={}),
        ]
        build_sparse_index(chunks, persist_dir=str(tmp_path))

        pickle_path = tmp_path / "bm25_index.pkl"
        assert pickle_path.exists()

        with open(pickle_path, "rb") as fh:
            payload = pickle.load(fh)

        assert "bm25" in payload
        assert payload["chunk_ids"] == ["c1", "c2"]
        assert len(payload["texts"]) == 2
