"""
FinRAG integration smoke test.

Ingests the bundled AAPL 10-K excerpt, then runs a query and asserts:
1. The answer contains an expected financial term
2. At least one source citation is returned
3. The source file references the ingested document

This test uses real BGE embeddings on a small corpus (no mocking of core logic).
It does NOT call the OpenAI API — the LLM is replaced with a fake that echoes
context, so the test is fully offline.

Requires:
    pip install sentence-transformers chromadb

Set FINRAG_SKIP_INTEGRATION=1 to skip if running in a resource-constrained CI.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip if explicitly disabled or if heavy deps are unavailable
pytestmark = pytest.mark.skipif(
    os.getenv("FINRAG_SKIP_INTEGRATION") == "1",
    reason="FINRAG_SKIP_INTEGRATION=1 set",
)

DATA_DIR = Path(__file__).parent.parent / "data" / "sample"
EXCERPT_FILE = DATA_DIR / "AAPL_10K_2023_excerpt.txt"


@pytest.fixture(scope="module")
def tmp_chroma(tmp_path_factory):
    """Isolated ChromaDB directory for integration tests."""
    return str(tmp_path_factory.mktemp("chroma"))


@pytest.fixture(scope="module")
def tmp_models(tmp_path_factory):
    """Isolated model cache directory."""
    return str(tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="module")
def mock_settings(tmp_chroma, tmp_models):
    """Settings pointing at temporary dirs — no real .env required."""
    settings = MagicMock()
    settings.chroma_persist_dir = tmp_chroma
    settings.chroma_collection = "test_finrag"
    settings.chroma_auth_token = None
    settings.embedding_model = "BAAI/bge-small-en-v1.5"
    settings.model_cache_dir = tmp_models
    settings.openai_api_key = MagicMock()
    settings.openai_api_key.get_secret_value.return_value = "sk-test"
    settings.openai_model = "gpt-4o-mini"
    settings.chunk_size = 512
    settings.chunk_overlap = 64
    settings.top_k = 3
    settings.mmr_lambda = 0.7
    settings.max_file_size_bytes = 52_428_800
    return settings


@pytest.fixture(scope="module")
def fake_llm():
    """Fake LLM that returns the user message as the 'answer' (echoes context)."""
    from generation.llm import LLMResponse

    llm = MagicMock()

    async def _generate(system_prompt, user_message, *, request_id=""):
        # Echo back a snippet so we can assert against expected terms
        return LLMResponse(
            content=user_message[:500],
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o-mini",
            latency_ms=10.0,
        )

    llm.generate = _generate
    return llm


@pytest.fixture(scope="module")
def pipeline(mock_settings, fake_llm):
    """Fully wired pipeline with real embedder + ChromaDB, fake LLM."""
    from ingestion.embedder import BGEEmbedder
    from pipeline.rag_pipeline import RAGPipeline
    from store.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(
        persist_dir=mock_settings.chroma_persist_dir,
        collection_name=mock_settings.chroma_collection,
    )
    embedder = BGEEmbedder(
        model_name=mock_settings.embedding_model,
        cache_dir=mock_settings.model_cache_dir,
    )
    return RAGPipeline(
        settings=mock_settings,
        store=store,
        embedder=embedder,
        llm=fake_llm,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestIngestAndQuery:
    def test_excerpt_file_exists(self):
        assert EXCERPT_FILE.exists(), f"Sample file missing: {EXCERPT_FILE}"

    def test_ingest_excerpt(self, pipeline):
        result = pipeline.ingest(str(EXCERPT_FILE), ticker="AAPL", doc_type="10-K")
        assert result.chunks_stored > 0, "Expected at least one chunk to be stored"
        assert result.total_chunks > 0
        assert result.chunks_stored + result.chunks_skipped == result.total_chunks

    def test_store_count_positive_after_ingest(self, pipeline):
        count = pipeline.store.count()
        assert count > 0, "Store should have chunks after ingestion"

    def test_query_returns_result(self, pipeline):
        result = asyncio.run(
            pipeline.query("What were Apple's risk factors in fiscal 2023?")
        )
        assert result.answer, "Answer should not be empty"
        assert isinstance(result.sources, list)
        assert result.latency_ms > 0

    def test_query_sources_cite_excerpt(self, pipeline):
        result = asyncio.run(
            pipeline.query("What is Apple's net income?")
        )
        source_files = [s.source_file for s in result.sources]
        assert any("AAPL" in f or "excerpt" in f for f in source_files), (
            f"Expected source to reference AAPL excerpt, got: {source_files}"
        )

    def test_query_answer_contains_financial_term(self, pipeline):
        """The fake LLM echoes context, so the answer should contain terms from the doc."""
        result = asyncio.run(
            pipeline.query("What was Apple's revenue in 2023?")
        )
        # The echoed context should contain financial terms from the excerpt
        answer_lower = result.answer.lower()
        financial_terms = ["apple", "revenue", "billion", "2023", "net", "income"]
        matched = [t for t in financial_terms if t in answer_lower]
        assert matched, (
            f"Expected financial terms in answer. Answer: {result.answer[:200]}"
        )

    def test_deduplication_on_second_ingest(self, pipeline):
        """Re-ingesting the same document should skip all chunks."""
        result = pipeline.ingest(str(EXCERPT_FILE), ticker="AAPL", doc_type="10-K")
        assert result.chunks_skipped == result.total_chunks, (
            "All chunks should be skipped on re-ingest (SHA256 deduplication)"
        )
        assert result.chunks_stored == 0

    def test_query_request_id_non_empty(self, pipeline):
        result = asyncio.run(
            pipeline.query("What is the gross margin?")
        )
        assert result.request_id, "Request ID should be set"
