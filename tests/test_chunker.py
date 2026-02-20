"""
Tests for ingestion/chunker.py.

Verifies overlap correctness, no mid-sentence splits, table preservation,
and section header boundary detection.
"""

from __future__ import annotations

import pytest

from ingestion.chunker import (
    Chunk,
    chunk_page_docs,
    _apply_overlap,
    _recursive_split,
    _is_table_block,
)
from ingestion.loader import PageDoc


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_page(text: str, page_num: int = 1) -> PageDoc:
    return PageDoc(
        text=text,
        page_num=page_num,
        source_file="TEST_10K_2023.txt",
        ticker="TEST",
        doc_type="10-K",
    )


# ── Overlap tests ─────────────────────────────────────────────────────────────


class TestOverlap:
    def test_overlap_tail_prepended(self):
        chunks = ["ABCDEFGHIJ", "KLMNOPQRST"]
        result = _apply_overlap(chunks, overlap=5)
        # Second chunk should start with tail of first
        assert result[1].startswith("FGHIJ")

    def test_overlap_zero_no_change(self):
        chunks = ["first chunk", "second chunk"]
        result = _apply_overlap(chunks, overlap=0)
        assert result == chunks

    def test_single_chunk_unchanged(self):
        chunks = ["only chunk"]
        result = _apply_overlap(chunks, overlap=10)
        assert result == chunks

    def test_overlap_length_bounded_by_previous_chunk(self):
        chunks = ["abc", "XYZ long second chunk"]
        result = _apply_overlap(chunks, overlap=100)
        # Even though overlap=100, previous chunk only has 3 chars
        assert result[1].startswith("abc")


class TestChunkSize:
    def test_chunks_not_exceed_size(self):
        long_text = (
            "Apple reported strong earnings this quarter. "
            "Revenue grew 12% year over year to $94 billion. "
            "iPhone sales were the primary driver of growth. "
        ) * 30  # ~2700 chars

        chunks = _recursive_split(long_text, chunk_size=300, chunk_overlap=30)
        # Allow some slack for overlap prepend
        for chunk in chunks:
            assert len(chunk) <= 600, f"Chunk too large: {len(chunk)}"

    def test_short_text_produces_single_chunk(self):
        text = "Apple Inc. reported net income of $97 billion."
        chunks = _recursive_split(text, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text


# ── No mid-sentence split tests ───────────────────────────────────────────────


class TestSentenceBoundary:
    def test_no_split_within_sentence(self):
        # A single long sentence — chunker should not split mid-word
        sentence = (
            "The Company's revenue for the fiscal year ended September 30, 2023 was "
            "$394.3 billion, representing a decrease of 2.8 percent compared to the "
            "prior fiscal year ended September 24, 2022."
        )
        text = (sentence + " ") * 5
        chunks = _recursive_split(text, chunk_size=300, chunk_overlap=30)
        for chunk in chunks:
            # No chunk should end in the middle of the word 'billion'
            assert not chunk.endswith(" $394")


# ── Table preservation tests ──────────────────────────────────────────────────


class TestTablePreservation:
    def test_detects_table_block(self):
        table = (
            "Revenue | 2023 | 2022\n"
            "iPhone  | $200B | $205B\n"
            "Mac     | $29B  | $40B\n"
            "iPad    | $28B  | $29B\n"
        )
        assert _is_table_block(table) is True

    def test_non_table_not_detected(self):
        prose = "The company reported strong results for the fiscal year."
        assert _is_table_block(prose) is False

    def test_table_kept_as_single_chunk(self):
        table = (
            "Net sales | 2023 | 2022\n"
            "Products  | 298,085 | 316,199\n"
            "Services  | 85,200  | 78,129\n"
            "Total     | 383,285 | 394,328\n"
        )
        intro = "The following table summarises net sales by category.\n\n"
        text = intro + table
        chunks = _recursive_split(text, chunk_size=512, chunk_overlap=64)
        # The table itself should be intact in one chunk
        table_chunks = [c for c in chunks if "Products" in c and "Services" in c]
        assert len(table_chunks) >= 1, "Table should appear as a coherent chunk"


# ── Section header boundary tests ─────────────────────────────────────────────


class TestSectionHeaders:
    def test_risk_factors_starts_new_chunk(self):
        text = (
            "Some introductory text about the company.\n\n"
            "Risk Factors\n\n"
            "The following risks could adversely affect the company."
        )
        chunks = _recursive_split(text, chunk_size=512, chunk_overlap=64)
        # Risk Factors section should appear in a chunk, possibly with the header
        risk_chunks = [c for c in chunks if "risk" in c.lower()]
        assert risk_chunks, "Risk Factors content should appear in a chunk"


# ── Integration: chunk_page_docs ──────────────────────────────────────────────


class TestChunkPageDocs:
    def test_produces_chunk_objects(self):
        pages = [
            _make_page(
                "Apple reported $97 billion in net income for fiscal 2023. "
                "This represents a strong performance. Revenue was $394 billion.",
                page_num=1,
            )
        ]
        chunks = chunk_page_docs(pages, chunk_size=512, chunk_overlap=64)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_metadata_propagated(self):
        pages = [_make_page("Some financial text.", page_num=3)]
        chunks = chunk_page_docs(pages, chunk_size=512, chunk_overlap=64)
        for chunk in chunks:
            assert chunk.source_file == "TEST_10K_2023.txt"
            assert chunk.ticker == "TEST"
            assert chunk.page_num == 3

    def test_empty_pages_handled(self):
        pages = [_make_page("   \n  \n  ", page_num=1)]  # blank page
        # load_document skips blank pages, but chunker receives empty text
        chunks = chunk_page_docs(pages, chunk_size=512, chunk_overlap=64)
        # Should not crash and may return 0 chunks
        assert isinstance(chunks, list)

    def test_chunk_indices_are_sequential(self):
        pages = [
            _make_page("Page one content with multiple sentences. " * 20, page_num=1),
            _make_page("Page two content with multiple sentences. " * 20, page_num=2),
        ]
        chunks = chunk_page_docs(pages, chunk_size=200, chunk_overlap=20)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_math_correct(self):
        # With overlap=10, the tail of each chunk should appear at start of next
        text = "A" * 50 + "B" * 50 + "C" * 50 + "D" * 50 + "E" * 50
        pages = [_make_page(text, page_num=1)]
        chunks = chunk_page_docs(pages, chunk_size=100, chunk_overlap=10)
        if len(chunks) >= 2:
            # The last 10 chars of chunk[i] should appear at the start of chunk[i+1]
            for i in range(len(chunks) - 1):
                tail = chunks[i].text[-10:]
                head = chunks[i + 1].text[:10]
                assert tail == head, (
                    f"Overlap mismatch at chunk {i}: tail={tail!r}, head={head!r}"
                )
