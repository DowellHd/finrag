"""
FinRAG MMR retriever.

Implements Maximal Marginal Relevance (MMR) reranking from scratch — no
LangChain dependency. MMR balances relevance and diversity to reduce redundant
chunks in the retrieved context window.

MMR formula (Carbonell & Goldstein, 1998):
    MMR = argmax[ λ · sim(q, d_i) − (1−λ) · max_{d_j ∈ S} sim(d_i, d_j) ]

where:
- q  = query embedding
- d_i = candidate document embedding
- S  = already-selected documents
- λ  = relevance/diversity trade-off (1.0 = pure relevance, 0.0 = pure diversity)

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from store.base import AbstractVectorStore, StoredChunk

log = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with combined relevance + MMR score."""

    text: str
    score: float          # original cosine similarity to query
    mmr_score: float      # post-MMR score (λ·rel − (1-λ)·redundancy)
    chunk_id: str
    source_file: str
    page_num: int
    ticker: str | None
    doc_type: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D unit vectors (dot product for normalised)."""
    return float(np.dot(a, b))


def _cosine_matrix(
    candidates: np.ndarray,  # shape (n, dim)
    selected: np.ndarray,    # shape (m, dim)
) -> np.ndarray:
    """Pairwise cosine similarity matrix: candidates × selected → (n, m)."""
    # Both are L2-normalised → dot product = cosine similarity
    return candidates @ selected.T


def mmr_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list[StoredChunk],
    *,
    top_k: int = 5,
    mmr_lambda: float = 0.7,
) -> list[StoredChunk]:
    """Re-rank candidate chunks using MMR.

    Args:
        query_embedding: L2-normalised query vector, shape (dim,).
        candidate_embeddings: L2-normalised embeddings for each candidate,
                              shape (n, dim). Must align with ``candidates``.
        candidates: StoredChunk objects from the vector store query.
        top_k: Number of chunks to select.
        mmr_lambda: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        Re-ordered list of at most ``top_k`` ``StoredChunk`` objects.
    """
    if not candidates:
        return []

    n = len(candidates)
    k = min(top_k, n)

    # Relevance scores: query · each candidate (dot product of unit vectors)
    rel_scores = candidate_embeddings @ query_embedding  # shape (n,)

    selected_indices: list[int] = []
    selected_embeddings: list[np.ndarray] = []
    remaining = list(range(n))

    for _ in range(k):
        if not remaining:
            break

        if not selected_embeddings:
            # First selection: pure relevance
            best_idx = max(remaining, key=lambda i: rel_scores[i])
        else:
            sel_matrix = np.stack(selected_embeddings, axis=0)  # (m, dim)

            best_score = -np.inf
            best_idx = remaining[0]

            for i in remaining:
                rel = mmr_lambda * rel_scores[i]
                # Max similarity to already-selected chunks
                sims_to_selected = _cosine_matrix(
                    candidate_embeddings[i : i + 1], sel_matrix
                )  # (1, m)
                redundancy = (1.0 - mmr_lambda) * float(sims_to_selected.max())
                score = rel - redundancy
                if score > best_score:
                    best_score = score
                    best_idx = i

        selected_indices.append(best_idx)
        selected_embeddings.append(candidate_embeddings[best_idx])
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


class Retriever:
    """Top-k cosine retrieval with MMR reranking.

    Args:
        store: Any ``AbstractVectorStore`` implementation.
        embedder: BGEEmbedder (or compatible) for encoding queries.
        top_k: Final number of chunks to return after MMR.
        mmr_lambda: MMR relevance/diversity trade-off.
        fetch_multiplier: Retrieve ``top_k * fetch_multiplier`` candidates
                          before MMR to give the reranker a larger pool.
    """

    def __init__(
        self,
        store: AbstractVectorStore,
        embedder,
        *,
        top_k: int = 5,
        mmr_lambda: float = 0.7,
        fetch_multiplier: int = 3,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.fetch_multiplier = fetch_multiplier

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve the most relevant, diverse chunks for a query.

        Pipeline:
        1. Encode query with BGE prefix.
        2. Fetch ``top_k * fetch_multiplier`` candidates from the store.
        3. MMR rerank to ``top_k`` chunks.
        4. Return ``RetrievedChunk`` objects.

        Args:
            query: Sanitised user question.

        Returns:
            List of ``RetrievedChunk`` objects ordered by MMR score.
        """
        query_embedding = self.embedder.encode_query(query)  # (dim,) float32

        fetch_k = self.top_k * self.fetch_multiplier
        raw_candidates = self.store.query(query_embedding.tolist(), top_k=fetch_k)

        if not raw_candidates:
            log.info("retriever.no_candidates", query_len=len(query))
            return []

        # Use stored embeddings returned by the vector store when available
        # (avoids a second embedder call, which matters for API-based embedders).
        # Fall back to re-encoding if any embedding is missing.
        if all(c.embedding is not None for c in raw_candidates):
            candidate_embeddings = np.array(
                [c.embedding for c in raw_candidates], dtype=np.float32
            )
        else:
            candidate_texts = [c.text for c in raw_candidates]
            candidate_embeddings = self.embedder.encode_documents(
                candidate_texts, show_progress=False
            )  # (n, dim) float32

        reranked = mmr_rerank(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidates=raw_candidates,
            top_k=self.top_k,
            mmr_lambda=self.mmr_lambda,
        )

        results: list[RetrievedChunk] = []
        for i, chunk in enumerate(reranked):
            meta = chunk.metadata
            # Compute MMR score as the position-weighted relevance (simplified)
            mmr_score = chunk.score * (self.mmr_lambda ** i)
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    score=chunk.score,
                    mmr_score=mmr_score,
                    chunk_id=chunk.chunk_id,
                    source_file=meta.get("source_file", ""),
                    page_num=int(meta.get("page_num", 0)),
                    ticker=meta.get("ticker"),
                    doc_type=meta.get("doc_type"),
                    metadata=meta,
                )
            )

        log.info(
            "retriever.retrieved",
            query_len=len(query),
            candidates=len(raw_candidates),
            selected=len(results),
        )
        return results
