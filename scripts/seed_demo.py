"""
Seed the demo vector store with Apple's FY2023 10-K excerpt.

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

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample" / "AAPL_10K_2023_excerpt.txt"


def main() -> None:
    settings = Settings()
    pipeline = build_pipeline(settings)

    doc_count = pipeline.store.count()
    if doc_count > 0:
        log.info("seed_demo.already_seeded", doc_count=doc_count)
        print(f"[seed_demo] Already seeded ({doc_count} chunks). Skipping.")
        return

    if not SAMPLE_FILE.exists():
        log.error("seed_demo.file_not_found", path=str(SAMPLE_FILE))
        print(f"[seed_demo] ERROR: Sample file not found: {SAMPLE_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"[seed_demo] Ingesting {SAMPLE_FILE.name} ...")
    result = pipeline.ingest(
        str(SAMPLE_FILE),
        ticker="AAPL",
        doc_type="10-K",
    )
    print(
        f"[seed_demo] Done — stored={result.chunks_stored}, "
        f"skipped={result.chunks_skipped}, total={result.total_chunks}"
    )


if __name__ == "__main__":
    main()
