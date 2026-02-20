"""
FinRAG CLI entry points.

Usage:
    python -m finrag ingest ./data/sample/AAPL_10K_2023.pdf --ticker AAPL --doc-type 10-K
    python -m finrag query "What were Apple's main risk factors in 2023?"

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import asyncio
import sys

import structlog

log = structlog.get_logger(__name__)


def _setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for human-readable CLI output."""
    import logging

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def cmd_ingest(args) -> None:
    """Run the document ingestion pipeline from the CLI."""
    from pipeline.rag_pipeline import build_pipeline

    _setup_logging()
    pipeline = build_pipeline()

    print(f"Ingesting: {args.file_path}")
    result = pipeline.ingest(
        args.file_path,
        ticker=args.ticker or None,
        doc_type=args.doc_type or None,
    )

    print(
        f"\nIngestion complete:\n"
        f"  Source file : {result.source_file}\n"
        f"  Total chunks: {result.total_chunks}\n"
        f"  Stored      : {result.chunks_stored}\n"
        f"  Skipped (dup): {result.chunks_skipped}"
    )


def cmd_query(args) -> None:
    """Run a RAG query from the CLI."""
    from pipeline.rag_pipeline import build_pipeline

    _setup_logging()
    pipeline = build_pipeline()

    async def _run():
        result = await pipeline.query(args.question)
        print(f"\nAnswer:\n{result.answer}")
        if result.sources:
            print("\nSources:")
            for src in result.sources:
                ticker_str = f" [{src.ticker}]" if src.ticker else ""
                print(f"  - {src.source_file}{ticker_str}, page {src.page_num}")
        print(f"\nLatency: {result.latency_ms:.0f}ms | Request ID: {result.request_id}")

    asyncio.run(_run())


def main() -> None:
    """Parse CLI arguments and dispatch to sub-commands."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="finrag",
        description="FinRAG — Finance-domain RAG system for document Q&A",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ──
    ingest_p = sub.add_parser("ingest", help="Ingest a financial document")
    ingest_p.add_argument("file_path", help="Path to PDF, TXT, or MD file")
    ingest_p.add_argument("--ticker", default="", help="Ticker symbol (e.g. AAPL)")
    ingest_p.add_argument("--doc-type", dest="doc_type", default="", help="Document type (e.g. 10-K)")
    ingest_p.set_defaults(func=cmd_ingest)

    # ── query ──
    query_p = sub.add_parser("query", help="Ask a financial question")
    query_p.add_argument("question", help="Question to ask (max 500 chars)")
    query_p.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
