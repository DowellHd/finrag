"""
FinRAG document loader.

Loads PDF, TXT, and Markdown files into raw text + page-level metadata.
Uses PyMuPDF (fitz) as the primary PDF backend with pdfplumber as a fallback
when PyMuPDF cannot extract meaningful text (e.g. scanned PDFs with partial
OCR).

Path and size validation is delegated to ``security.validators`` — this module
performs NO file I/O before validation succeeds.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Filename pattern: AAPL_10K_2023.pdf → ticker=AAPL, doc_type=10K, year=2023
_FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z]{1,5})_(?P<doc_type>[A-Z0-9\-K]+)_(?P<year>\d{4})",
    re.IGNORECASE,
)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class PageDoc:
    """A single page of extracted text with associated metadata."""

    text: str
    page_num: int  # 1-indexed
    source_file: str
    ticker: Optional[str] = None
    doc_type: Optional[str] = None
    document_year: Optional[str] = None
    extra: dict = field(default_factory=dict)


# ── Metadata helpers ──────────────────────────────────────────────────────────


def _parse_filename_metadata(filename: str) -> dict[str, Optional[str]]:
    """Extract ticker, doc_type, and year from a structured filename.

    Expects format: ``<TICKER>_<DOCTYPE>_<YEAR>.ext``
    e.g. ``AAPL_10K_2023.pdf``.
    """
    m = _FILENAME_RE.match(Path(filename).stem)
    if m:
        return {
            "ticker": m.group("ticker").upper(),
            "doc_type": m.group("doc_type").upper(),
            "document_year": m.group("year"),
        }
    return {"ticker": None, "doc_type": None, "document_year": None}


# ── PDF loading ───────────────────────────────────────────────────────────────


def _load_pdf_pymupdf(path: Path) -> list[tuple[int, str]]:
    """Primary PDF extractor using PyMuPDF (fitz).

    Returns a list of (page_num_1indexed, page_text) tuples.
    Raises ``RuntimeError`` if fitz is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed. Run: pip install pymupdf"
        ) from exc

    pages: list[tuple[int, str]] = []
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF with PyMuPDF: {exc}") from exc

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append((i, text))
    doc.close()
    return pages


def _load_pdf_pdfplumber(path: Path) -> list[tuple[int, str]]:
    """Fallback PDF extractor using pdfplumber.

    Returns a list of (page_num_1indexed, page_text) tuples.
    Raises ``RuntimeError`` if pdfplumber is not installed.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        ) from exc

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def _load_pdf(path: Path) -> list[tuple[int, str]]:
    """Load a PDF file, trying PyMuPDF first and falling back to pdfplumber.

    Applies a quality heuristic: if PyMuPDF extracts fewer than 50 characters
    total, it likely failed (e.g. image-only PDF) and we try pdfplumber.
    """
    try:
        pages = _load_pdf_pymupdf(path)
        total_chars = sum(len(t) for _, t in pages)
        if total_chars < 50:
            log.info(
                "loader.pymupdf_low_content_fallback",
                path=str(path),
                chars_extracted=total_chars,
            )
            raise ValueError("Low content — try fallback")
        return pages
    except Exception as primary_exc:
        log.info(
            "loader.pdfplumber_fallback",
            path=str(path),
            reason=str(primary_exc),
        )
        try:
            return _load_pdf_pdfplumber(path)
        except Exception as fallback_exc:
            raise ValueError(
                f"Both PDF parsers failed. PyMuPDF: {primary_exc} | "
                f"pdfplumber: {fallback_exc}"
            ) from fallback_exc


# ── Plain text / Markdown loading ─────────────────────────────────────────────


def _load_text(path: Path) -> list[tuple[int, str]]:
    """Load a plain text or Markdown file as a single 'page'."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(1, text)]


# ── Public API ────────────────────────────────────────────────────────────────


def load_document(
    path: Path,
    *,
    ticker: str | None = None,
    doc_type: str | None = None,
) -> list[PageDoc]:
    """Load a document from disk into a list of ``PageDoc`` objects.

    This function assumes the path has already been validated by
    ``security.validators.validate_file_path``.

    Args:
        path: Resolved, validated ``pathlib.Path`` to the document.
        ticker: Ticker symbol override (e.g. "AAPL"). Falls back to filename parse.
        doc_type: Document type override (e.g. "10-K"). Falls back to filename parse.

    Returns:
        List of ``PageDoc`` objects, one per page (PDFs) or one for text files.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    filename = path.name
    meta = _parse_filename_metadata(filename)

    # Explicit arguments take precedence over filename-parsed values
    resolved_ticker = ticker or meta["ticker"]
    resolved_doc_type = doc_type or meta["doc_type"]
    document_year = meta["document_year"]

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_pages = _load_pdf(path)
    elif suffix in {".txt", ".md"}:
        raw_pages = _load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs: list[PageDoc] = []
    for page_num, text in raw_pages:
        stripped = text.strip()
        if not stripped:
            continue  # skip blank pages
        docs.append(
            PageDoc(
                text=stripped,
                page_num=page_num,
                source_file=filename,
                ticker=resolved_ticker,
                doc_type=resolved_doc_type,
                document_year=document_year,
            )
        )

    log.info(
        "loader.document_loaded",
        filename=filename,
        ticker=resolved_ticker,
        doc_type=resolved_doc_type,
        pages=len(docs),
    )
    return docs
