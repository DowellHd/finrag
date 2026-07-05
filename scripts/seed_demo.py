"""
Seed the demo vector store with FY2023 10-K excerpts across a diverse set of
companies (tech, banking, energy, healthcare, retail).

Run once during container startup (via entrypoint.sh). Safe to call multiple
times — checks doc_count first and skips if already seeded.

Usage:
    python scripts/seed_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog

from config.settings import Settings
from pipeline.rag_pipeline import build_pipeline

log = structlog.get_logger(__name__)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

# Ticker -> sample excerpt filename. Covers tech, banking, energy, healthcare,
# and retail so the demo corpus supports cross-sector queries.
SAMPLE_FILES = {
    "AAPL": "AAPL_10K_2023_excerpt.txt",
    "MSFT": "MSFT_10K_2023_excerpt.txt",
    "AMZN": "AMZN_10K_2023_excerpt.txt",
    "GOOGL": "GOOGL_10K_2023_excerpt.txt",
    "NVDA": "NVDA_10K_2023_excerpt.txt",
    "META": "META_10K_2023_excerpt.txt",
    "NFLX": "NFLX_10K_2023_excerpt.txt",
    "JPM": "JPM_10K_2023_excerpt.txt",
    "XOM": "XOM_10K_2023_excerpt.txt",
    "JNJ": "JNJ_10K_2023_excerpt.txt",
    "WMT": "WMT_10K_2023_excerpt.txt",
}


def main() -> None:
    settings = Settings()
    pipeline = build_pipeline(settings)

    doc_count = pipeline.store.count()
    if doc_count > 0:
        log.info("seed_demo.already_seeded", doc_count=doc_count)
        print(f"[seed_demo] Already seeded ({doc_count} chunks). Skipping.")
        return

    total_stored = 0
    total_skipped = 0
    total_chunks = 0
    for ticker, filename in SAMPLE_FILES.items():
        sample_file = SAMPLE_DIR / filename
        if not sample_file.exists():
            log.error("seed_demo.file_not_found", path=str(sample_file))
            print(f"[seed_demo] ERROR: Sample file not found: {sample_file}", file=sys.stderr)
            sys.exit(1)

        print(f"[seed_demo] Ingesting {sample_file.name} ...")
        result = pipeline.ingest(
            str(sample_file),
            ticker=ticker,
            doc_type="10-K",
        )
        total_stored += result.chunks_stored
        total_skipped += result.chunks_skipped
        total_chunks += result.total_chunks

    print(
        f"[seed_demo] Done — stored={total_stored}, "
        f"skipped={total_skipped}, total={total_chunks}"
    )


if __name__ == "__main__":
    main()
