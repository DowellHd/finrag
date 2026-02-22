"""
FinRAG pipeline orchestrator.

Coordinates document ingestion and retrieval-augmented generation for financial
documents. Built as a companion to SSB (Smart Strategies Builder) to explore
document-grounded Q&A as a potential future capability of SSB's market analysis
assistant.

The pipeline exposes two high-level operations:
- ``ingest(path, ticker, doc_type)``: load → chunk → embed → store
- ``query(question)``: sanitise → embed → retrieve → MMR → generate → respond
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from config.settings import Settings
from generation.llm import OpenAILLM, LLMResponse
from generation.prompt_builder import build_prompt
from ingestion.chunker import chunk_page_docs, Chunk
from ingestion.embedder import BGEEmbedder, OpenAIEmbedder
from ingestion.loader import load_document
from retrieval.retriever import Retriever, RetrievedChunk
from security.validators import validate_file_path, sanitize_query
from store.base import AbstractVectorStore

log = structlog.get_logger(__name__)


# ── Response models ────────────────────────────────────────────────────────────


@dataclass
class SourceRef:
    """A source citation for an answer chunk."""

    source_file: str
    page_num: int
    ticker: str | None
    doc_type: str | None


@dataclass
class IngestResult:
    """Result of a document ingestion operation."""

    source_file: str
    chunks_stored: int
    chunks_skipped: int  # deduplication hits
    total_chunks: int


@dataclass
class QueryResult:
    """Result of a RAG query."""

    answer: str
    sources: list[SourceRef]
    latency_ms: float
    request_id: str
    prompt_tokens: int
    completion_tokens: int


# ── SHA256 chunk deduplication ─────────────────────────────────────────────────


def _chunk_hash(text: str) -> str:
    """SHA256 hash of chunk text — used for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ── Pipeline ──────────────────────────────────────────────────────────────────


class RAGPipeline:
    """End-to-end FinRAG orchestrator.

    Args:
        settings: Application settings (from pydantic-settings).
        store: Vector store backend (ChromaDB or pgvector adapter).
        embedder: BGEEmbedder instance.
        llm: Async OpenAI LLM wrapper.
    """

    def __init__(
        self,
        settings: Settings,
        store: AbstractVectorStore,
        embedder: BGEEmbedder,
        llm: OpenAILLM,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.retriever = Retriever(
            store=store,
            embedder=embedder,
            top_k=settings.top_k,
            mmr_lambda=settings.mmr_lambda,
        )

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def ingest(
        self,
        file_path: str,
        *,
        ticker: str | None = None,
        doc_type: str | None = None,
    ) -> IngestResult:
        """Ingest a document into the vector store.

        Pipeline:
        1. Security: validate path + extension + size.
        2. Load pages from PDF/TXT.
        3. Chunk pages into overlapping segments.
        4. Embed chunks in batches.
        5. SHA256 deduplicate — skip already-stored chunks.
        6. Upsert new chunks to the vector store.

        Args:
            file_path: Raw path string (validated before any I/O).
            ticker: Ticker symbol override (e.g. "AAPL").
            doc_type: Document type override (e.g. "10-K").

        Returns:
            ``IngestResult`` with store/skip counts.
        """
        # 1 — Security validation (before any file I/O)
        validated_path = validate_file_path(
            file_path,
            max_size_bytes=self.settings.max_file_size_bytes,
        )

        # 2 — Load
        pages = load_document(validated_path, ticker=ticker, doc_type=doc_type)

        # 3 — Chunk
        chunks: list[Chunk] = chunk_page_docs(
            pages,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        if not chunks:
            log.warning("pipeline.no_chunks_produced", file=str(validated_path))
            return IngestResult(
                source_file=validated_path.name,
                chunks_stored=0,
                chunks_skipped=0,
                total_chunks=0,
            )

        # 4 — Embed
        texts, embeddings = self.embedder.embed_chunks(chunks, show_progress=True)

        # 5 — Deduplication + upsert
        new_ids: list[str] = []
        new_embeddings: list[list[float]] = []
        new_texts: list[str] = []
        new_metadatas: list[dict[str, Any]] = []
        skipped = 0

        for chunk, text, embedding in zip(chunks, texts, embeddings):
            chunk_id = _chunk_hash(text)

            if self.store.exists(chunk_id):
                skipped += 1
                continue

            new_ids.append(chunk_id)
            new_embeddings.append(embedding.tolist())
            new_texts.append(text)
            new_metadatas.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": chunk.source_file,
                    "page_num": chunk.page_num,
                    "ticker": chunk.ticker or "",
                    "doc_type": chunk.doc_type or "",
                    "document_year": chunk.document_year or "",
                    "chunk_hash": chunk_id,
                }
            )

        if new_ids:
            self.store.upsert(new_ids, new_embeddings, new_texts, new_metadatas)

        result = IngestResult(
            source_file=validated_path.name,
            chunks_stored=len(new_ids),
            chunks_skipped=skipped,
            total_chunks=len(chunks),
        )
        log.info(
            "pipeline.ingest_complete",
            source_file=result.source_file,
            stored=result.chunks_stored,
            skipped=result.chunks_skipped,
        )
        return result

    # ── Query ──────────────────────────────────────────────────────────────────

    async def query(self, question: str) -> QueryResult:
        """Answer a financial question from the document store.

        Pipeline:
        1. Sanitise + validate query (security layer).
        2. Retrieve top-k chunks via MMR.
        3. Build prompt with trust boundary.
        4. Generate answer via LLM with retry.
        5. Return structured response with source citations.

        Args:
            question: Raw user question string.

        Returns:
            ``QueryResult`` with answer, sources, and latency.
        """
        request_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()

        # 1 — Sanitise
        clean_query = sanitize_query(question)

        # 2 — Retrieve
        retrieved: list[RetrievedChunk] = self.retriever.retrieve(clean_query)

        # 3 — Build prompt
        system_prompt, user_message = build_prompt(clean_query, retrieved)

        # 4 — Generate
        llm_response: LLMResponse = await self.llm.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            request_id=request_id,
        )

        # 5 — Build source citations (deduplicated)
        seen: set[tuple] = set()
        sources: list[SourceRef] = []
        for chunk in retrieved:
            key = (chunk.source_file, chunk.page_num)
            if key not in seen:
                seen.add(key)
                sources.append(
                    SourceRef(
                        source_file=chunk.source_file,
                        page_num=chunk.page_num,
                        ticker=chunk.ticker,
                        doc_type=chunk.doc_type,
                    )
                )

        latency_ms = (time.monotonic() - t0) * 1_000

        log.info(
            "pipeline.query_complete",
            request_id=request_id,
            sources=len(sources),
            latency_ms=round(latency_ms, 1),
        )

        return QueryResult(
            answer=llm_response.content,
            sources=sources,
            latency_ms=latency_ms,
            request_id=request_id,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def build_pipeline(settings: Settings | None = None) -> RAGPipeline:
    """Construct a fully wired ``RAGPipeline`` from application settings.

    Args:
        settings: Optional pre-built settings object. Loads from env if None.

    Returns:
        Ready-to-use ``RAGPipeline`` instance.
    """
    from store.chroma_store import ChromaVectorStore

    if settings is None:
        settings = Settings()

    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
        auth_token=(
            settings.chroma_auth_token.get_secret_value()
            if settings.chroma_auth_token
            else None
        ),
    )

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )

    llm = OpenAILLM(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    return RAGPipeline(settings=settings, store=store, embedder=embedder, llm=llm)
