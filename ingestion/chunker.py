"""
FinRAG finance-aware text chunker.

Implements a recursive character splitter that:
- Respects sentence boundaries (no mid-sentence splits)
- Preserves finance section headers (Risk Factors, MD&A, etc.)
- Keeps tables and bullet lists intact as single chunks
- Produces configurable chunk_size / chunk_overlap windows

No external splitting library — implemented from scratch to demonstrate
RAG depth and to avoid LangChain dependency.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

# ── Finance section header patterns ───────────────────────────────────────────
# These headers are treated as hard split boundaries so they always start a
# new chunk — preserving semantic coherence in SEC filings.
_SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"ITEM\s+\d+[A-Z]?\.?"
    r"|Risk\s+Factors?"
    r"|Management['']s?\s+Discussion"
    r"|MD&A"
    r"|Financial\s+Statements?"
    r"|Notes?\s+to\s+(?:Consolidated\s+)?Financial"
    r"|Selected\s+Financial\s+Data"
    r"|Quantitative\s+(?:and\s+Qualitative\s+)?Disclosures?"
    r"|Executive\s+(?:Summary|Overview)"
    r"|Business\s+Overview"
    r"|(?:Consolidated\s+)?Balance\s+Sheet"
    r"|(?:Consolidated\s+)?(?:Statement|Statements)\s+of"
    r"|Liquidity\s+and\s+Capital"
    r"|Contractual\s+Obligations"
    r"|Critical\s+Accounting"
    r"|Forward.Looking\s+Statements?"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Table detector: rows containing multiple pipe chars or tab-separated numbers
_TABLE_LINE_RE = re.compile(r"(?:\|.*\||\t\S.*\t\S|\s{4,}\d[\d,\.]+\s{2,})")

# Sentence ending punctuation
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A text chunk with positional metadata."""

    text: str
    chunk_index: int
    source_file: str
    page_num: int
    ticker: str | None = None
    doc_type: str | None = None
    document_year: str | None = None


# ── Core splitting logic ──────────────────────────────────────────────────────


def _is_table_block(text: str) -> bool:
    """Return True if the text block looks like a financial table."""
    lines = text.splitlines()
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if _TABLE_LINE_RE.search(ln))
    return table_lines >= 2  # at least 2 table-like rows


def _split_on_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, preserving trailing spaces."""
    sentences = _SENTENCE_END_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* characters.

    Strategy (descending granularity):
    1. Split on finance section headers (hard boundaries).
    2. Split on paragraph boundaries (double newline).
    3. Split on single newlines.
    4. Split on sentence endings.
    5. Hard-split oversized sentences at word boundaries.

    Overlap is implemented as a sliding window: the tail of the previous chunk
    is prepended to the next chunk.
    """
    # Step 1 — section header split
    segments = _split_by_section_headers(text)

    chunks: list[str] = []
    pending = ""

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # If segment is a table, keep it as-is (don't fragment tables)
        if _is_table_block(seg):
            if pending:
                chunks.extend(_paragraph_split(pending, chunk_size, chunk_overlap))
                pending = ""
            chunks.append(seg[: chunk_size * 3])  # hard cap at 3× to avoid runaway
            continue

        candidate = (pending + " " + seg).strip() if pending else seg
        if len(candidate) <= chunk_size:
            pending = candidate
        else:
            if pending:
                chunks.extend(_paragraph_split(pending, chunk_size, chunk_overlap))
            pending = seg

    if pending:
        chunks.extend(_paragraph_split(pending, chunk_size, chunk_overlap))

    # Apply overlap sliding window
    return _apply_overlap(chunks, chunk_overlap)


def _split_by_section_headers(text: str) -> list[str]:
    """Split on finance section header lines, returning segment list."""
    parts = _SECTION_HEADER_RE.split(text)
    # The regex splits out the header itself; re-attach headers to following body
    if len(parts) <= 1:
        return parts

    result: list[str] = []
    # parts[0] is content before first header, then header/body alternate
    spans = list(_SECTION_HEADER_RE.finditer(text))
    prev_end = 0
    for match in spans:
        before = text[prev_end : match.start()]
        if before.strip():
            result.append(before)
        prev_end = match.start()
    result.append(text[prev_end:])
    return result


def _paragraph_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a segment on paragraphs, then sentences, producing ≤ chunk_size chunks."""
    if len(text) <= chunk_size:
        return [text]

    # Try paragraph split first
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks = _merge_into_chunks(paragraphs, chunk_size)

    # If still oversized, fall back to sentence split
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final.append(chunk)
        else:
            sentences = _split_on_sentences(chunk)
            final.extend(_merge_into_chunks(sentences, chunk_size))

    return final or [text[:chunk_size]]


def _merge_into_chunks(units: list[str], chunk_size: int) -> list[str]:
    """Greedily merge units into chunks of at most chunk_size characters."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = (current + " " + unit).strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single unit is already too large, hard-split at word boundary
            if len(unit) > chunk_size:
                chunks.extend(_word_split(unit, chunk_size))
                current = ""
            else:
                current = unit
    if current:
        chunks.append(current)
    return chunks


def _word_split(text: str, chunk_size: int) -> list[str]:
    """Last-resort: split oversized text at word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip() if current else word
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to each subsequent chunk."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append((prev_tail + " " + chunks[i]).strip())
    return result


# ── Public API ────────────────────────────────────────────────────────────────


def chunk_page_docs(
    page_docs: list,  # list[PageDoc] — avoid circular import
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Convert a list of ``PageDoc`` objects into a flat list of ``Chunk`` objects.

    Args:
        page_docs: Output from ``ingestion.loader.load_document``.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        Ordered list of ``Chunk`` objects ready for embedding.
    """
    all_chunks: list[Chunk] = []
    global_idx = 0

    for page in page_docs:
        raw_chunks = _recursive_split(page.text, chunk_size, chunk_overlap)
        for text in raw_chunks:
            if not text.strip():
                continue
            all_chunks.append(
                Chunk(
                    text=text,
                    chunk_index=global_idx,
                    source_file=page.source_file,
                    page_num=page.page_num,
                    ticker=page.ticker,
                    doc_type=page.doc_type,
                    document_year=page.document_year,
                )
            )
            global_idx += 1

    log.info(
        "chunker.complete",
        input_pages=len(page_docs),
        output_chunks=len(all_chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return all_chunks
