"""
FinRAG demo script: download and ingest a public EDGAR 10-K filing.

Downloads Apple's FY2023 10-K from SEC EDGAR (public domain) and ingests it
into the FinRAG vector store.

Usage:
    python scripts/ingest_sec_filing.py

Requires OPENAI_API_KEY in .env (for generation; not needed for ingestion only).

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# Apple FY2023 10-K — public domain, SEC EDGAR
EDGAR_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/"
    "000032019323000106/aapl-20230930.htm"
)

# Where we'll save the downloaded file
DATA_DIR = Path(__file__).parent.parent / "data" / "sample"
OUTPUT_FILENAME = "AAPL_10K_2023.txt"


def download_filing(url: str, dest: Path) -> None:
    """Download a filing from EDGAR and save to dest."""
    print(f"Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "FinRAG/0.1 research@example.com"})
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
        html_bytes = response.read()

    # Strip HTML tags for plain-text ingestion
    try:
        from html.parser import HTMLParser

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self._parts.append(stripped)

            def get_text(self) -> str:
                return "\n".join(self._parts)

        parser = _TextExtractor()
        parser.feed(html_bytes.decode("utf-8", errors="replace"))
        text = parser.get_text()
    except Exception as exc:
        print(f"HTML stripping failed ({exc}), saving raw bytes as text.")
        text = html_bytes.decode("utf-8", errors="replace")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(f"Saved: {dest} ({dest.stat().st_size:,} bytes)")


def main() -> None:
    output_path = DATA_DIR / OUTPUT_FILENAME

    if not output_path.exists():
        download_filing(EDGAR_URL, output_path)
    else:
        print(f"File already exists: {output_path} — skipping download.")

    # Add project root to path so we can import finrag modules
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.rag_pipeline import build_pipeline

    print("\nBuilding FinRAG pipeline...")
    pipeline = build_pipeline()

    print(f"\nIngesting: {output_path}")
    result = pipeline.ingest(
        str(output_path),
        ticker="AAPL",
        doc_type="10-K",
    )

    print(
        f"\nIngestion complete:\n"
        f"  Source file : {result.source_file}\n"
        f"  Total chunks: {result.total_chunks}\n"
        f"  Stored      : {result.chunks_stored}\n"
        f"  Skipped (dup): {result.chunks_skipped}\n"
        f"\nRun a query:\n"
        f"  python -m finrag query \"What were Apple's main risk factors in 2023?\""
    )


if __name__ == "__main__":
    main()
