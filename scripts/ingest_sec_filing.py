"""
FinRAG demo script: download and ingest real public EDGAR 10-K filings.

Downloads each company's FY2023 10-K from SEC EDGAR (public domain) and
ingests it into the FinRAG vector store. Covers a diverse set of sectors —
tech, banking, energy, healthcare, and retail — rather than a single company.

The exact filing URL for each company (accession number + primary document)
is resolved dynamically via SEC's submissions API (data.sec.gov) rather than
hardcoded, since accession numbers are opaque and would otherwise need to be
guessed or manually refreshed.

Usage:
    python scripts/ingest_sec_filing.py

Requires OPENAI_API_KEY in .env (for generation; not needed for ingestion only).

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# Ticker -> (CIK, doc_type). CIKs verified against SEC's official
# company_tickers.json mapping. For each, we pull the 10-K whose reporting
# period falls in calendar year 2023 (or, for companies like Walmart whose
# own fiscal-year labeling differs, the 10-K they call "fiscal 2023").
COMPANIES: dict[str, int] = {
    "AAPL": 320193,
    "MSFT": 789019,
    "AMZN": 1018724,
    "GOOGL": 1652044,
    "NVDA": 1045810,
    "META": 1326801,
    "NFLX": 1065280,
    "JPM": 19617,
    "XOM": 34088,
    "JNJ": 200406,
    "WMT": 104169,
}

USER_AGENT = "FinRAG/0.1 research@example.com"

# Where we'll save the downloaded filings
DATA_DIR = Path(__file__).parent.parent / "data" / "sample"


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310
        return json.loads(response.read())


def find_fy2023_10k_url(cik: int) -> Optional[str]:
    """Locate a company's FY2023 10-K filing via SEC's submissions API.

    Searches the "recent" filings window first, then falls back to older
    paginated submission files for large filers (e.g. banks) whose recent
    window is dominated by other filing types (8-Ks, debt prospectuses).
    """

    def _search(entries: dict) -> Optional[str]:
        forms = entries["form"]
        for i, form in enumerate(forms):
            if form == "10-K" and entries["reportDate"][i].startswith("2023"):
                accession = entries["accessionNumber"][i].replace("-", "")
                doc = entries["primaryDocument"][i]
                return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
        return None

    submissions = _fetch_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    url = _search(submissions["filings"]["recent"])
    if url:
        return url

    for extra_file in submissions["filings"].get("files", []):
        older = _fetch_json(f"https://data.sec.gov/submissions/{extra_file['name']}")
        url = _search(older)
        if url:
            return url

    return None


def download_filing(url: str, dest: Path) -> None:
    """Download a filing from EDGAR and save to dest as stripped plain text."""
    print(f"Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
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
    # Add project root to path so we can import finrag modules
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.rag_pipeline import build_pipeline

    print("Building FinRAG pipeline...")
    pipeline = build_pipeline()

    for ticker, cik in COMPANIES.items():
        output_path = DATA_DIR / f"{ticker}_10K_2023.txt"

        if not output_path.exists():
            print(f"\nResolving FY2023 10-K URL for {ticker} (CIK {cik})...")
            url = find_fy2023_10k_url(cik)
            if url is None:
                print(f"  Could not locate a FY2023 10-K for {ticker}, skipping.", file=sys.stderr)
                continue
            download_filing(url, output_path)
        else:
            print(f"\nFile already exists: {output_path} — skipping download.")

        print(f"Ingesting: {output_path}")
        result = pipeline.ingest(
            str(output_path),
            ticker=ticker,
            doc_type="10-K",
        )
        print(
            f"  Total chunks: {result.total_chunks} | "
            f"Stored: {result.chunks_stored} | "
            f"Skipped (dup): {result.chunks_skipped}"
        )

    print(
        "\nIngestion complete for all companies.\n"
        "Run a query:\n"
        '  python -m finrag query "What were Apple\'s main risk factors in 2023?"'
    )


if __name__ == "__main__":
    main()
