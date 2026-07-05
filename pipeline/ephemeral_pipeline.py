"""
FinRAG ephemeral query pipeline.

Answers questions about a single ad-hoc uploaded document held in an
``EphemeralSession`` — mirrors ``pipeline.rag_pipeline.RAGPipeline.query()``'s
retrieval/generation steps, but sources candidates directly from the
session's in-memory chunks/embeddings instead of a vector-store round trip.
This deliberately bypasses ``RAGPipeline``/``Retriever``/``AbstractVectorStore``
entirely — ``retrieval.retriever.mmr_rerank`` is a pure function that needs
only a query embedding and a matching candidate-embeddings array, so no new
vector-store backend is needed for a single ephemeral document.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import numpy as np
import structlog
from starlette.concurrency import run_in_threadpool

from generation.llm import OpenAILLM
from generation.prompt_builder import build_prompt
from pipeline.ephemeral_store import EphemeralSession
from pipeline.rag_pipeline import SourceRef
from retrieval.retriever import RetrievedChunk, mmr_rerank
from security.validators import sanitize_query
from store.base import StoredChunk

log = structlog.get_logger(__name__)


@dataclass
class EphemeralQueryResult:
    """Result of an ephemeral-document query."""

    answer: str
    sources: list[SourceRef]  # ticker/doc_type are always None for uploads
    latency_ms: float
    request_id: str
    truncated_notice: str | None


def retrieve_from_session(
    session: EphemeralSession,
    query_embedding: np.ndarray,
    *,
    top_k: int,
    mmr_lambda: float,
    fetch_multiplier: int = 3,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant, diverse chunks from a single ephemeral session.

    Mirrors ``retrieval.retriever.Retriever.retrieve()``'s candidate-fetch +
    MMR-rerank steps, but computes relevance directly against the session's
    in-memory embeddings instead of querying a vector store.
    """
    n = len(session.chunks)
    if n == 0:
        return []

    # Both query_embedding and session.embeddings are already L2-normalised
    # (by OpenAIEmbedder), so the dot product is cosine similarity.
    rel_scores = session.embeddings @ query_embedding  # shape (n,)

    fetch_k = min(n, top_k * fetch_multiplier)
    order = np.argsort(-rel_scores)[:fetch_k]

    candidates = [
        StoredChunk(
            chunk_id=f"{session.session_id}:{i}",
            text=session.chunks[i].text,
            score=float(rel_scores[i]),
            metadata={
                "source_file": session.filename,
                "page_num": session.chunks[i].page_num,
            },
        )
        for i in order
    ]
    candidate_embeddings = session.embeddings[order]

    reranked = mmr_rerank(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        candidates=candidates,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
    )

    results: list[RetrievedChunk] = []
    for i, chunk in enumerate(reranked):
        meta = chunk.metadata
        # Same simplified position-weighted score as Retriever.retrieve().
        mmr_score = chunk.score * (mmr_lambda**i)
        results.append(
            RetrievedChunk(
                text=chunk.text,
                score=chunk.score,
                mmr_score=mmr_score,
                chunk_id=chunk.chunk_id,
                source_file=meta["source_file"],
                page_num=meta["page_num"],
                ticker=None,
                doc_type=None,
                metadata=meta,
            )
        )
    return results


async def answer_ephemeral_query(
    session: EphemeralSession,
    question: str,
    *,
    embedder,
    llm: OpenAILLM,
    top_k: int,
    mmr_lambda: float,
) -> EphemeralQueryResult:
    """Answer a question about a single ephemeral (uploaded) document.

    Pipeline: sanitise -> embed query -> retrieve+MMR -> build prompt ->
    generate. Reuses the same security/generation building blocks as the
    permanent-corpus flow unchanged.
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.monotonic()

    clean_query = sanitize_query(question)

    # embedder.encode_query is a synchronous (blocking) OpenAI call — run it
    # off the event loop, same as the upload endpoint does for embed_chunks.
    query_embedding = await run_in_threadpool(embedder.encode_query, clean_query)

    retrieved = retrieve_from_session(
        session, query_embedding, top_k=top_k, mmr_lambda=mmr_lambda
    )

    system_prompt, user_message = build_prompt(clean_query, retrieved)

    llm_response = await llm.generate(
        system_prompt=system_prompt,
        user_message=user_message,
        request_id=request_id,
    )

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
                    ticker=None,
                    doc_type=None,
                )
            )

    truncated_notice = (
        f"This answer is based on the first {len(session.chunks)} of "
        f"{session.total_chunks_before_cap} text chunks extracted from your "
        f"document (large document truncated)."
        if session.truncated
        else None
    )

    latency_ms = (time.monotonic() - t0) * 1_000

    log.info(
        "ephemeral_pipeline.query_complete",
        request_id=request_id,
        session_id=session.session_id,
        sources=len(sources),
        latency_ms=round(latency_ms, 1),
    )

    return EphemeralQueryResult(
        answer=llm_response.content,
        sources=sources,
        latency_ms=latency_ms,
        request_id=request_id,
        truncated_notice=truncated_notice,
    )
