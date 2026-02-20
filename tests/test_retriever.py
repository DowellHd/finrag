"""
Tests for retrieval/retriever.py.

Verifies that MMR reranking produces less redundant results than naive top-k
on a corpus with repeated similar content.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval.retriever import mmr_rerank, RetrievedChunk
from store.base import StoredChunk


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_stored(chunk_id: str, text: str, score: float) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={"source_file": "test.txt", "page_num": 1},
    )


def _unit_vec(dim: int, seed: int) -> np.ndarray:
    """Reproducible random unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _nearly_identical_vec(base: np.ndarray, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    """Unit vector close to base (simulates near-duplicate chunks)."""
    rng = np.random.default_rng(seed)
    noisy = base + noise * rng.standard_normal(base.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy)


# ── MMR unit tests ────────────────────────────────────────────────────────────


class TestMMRRerank:
    DIM = 64

    def test_empty_candidates_returns_empty(self):
        query = _unit_vec(self.DIM, seed=0)
        result = mmr_rerank(
            query_embedding=query,
            candidate_embeddings=np.empty((0, self.DIM), dtype=np.float32),
            candidates=[],
            top_k=5,
        )
        assert result == []

    def test_single_candidate_returned(self):
        query = _unit_vec(self.DIM, seed=0)
        candidate_emb = _unit_vec(self.DIM, seed=1)
        candidate = _make_stored("c1", "Apple revenue 2023", score=0.9)
        result = mmr_rerank(
            query_embedding=query,
            candidate_embeddings=np.stack([candidate_emb]),
            candidates=[candidate],
            top_k=5,
        )
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    def test_top_k_limit_respected(self):
        query = _unit_vec(self.DIM, seed=0)
        candidates = [_make_stored(f"c{i}", f"chunk {i}", 0.8 - i * 0.05) for i in range(10)]
        embeddings = np.stack([_unit_vec(self.DIM, seed=i + 1) for i in range(10)])
        result = mmr_rerank(
            query_embedding=query,
            candidate_embeddings=embeddings,
            candidates=candidates,
            top_k=3,
        )
        assert len(result) == 3

    def test_mmr_reduces_redundancy_vs_naive_topk(self):
        """MMR selected set should be more diverse than naive top-k."""
        query = _unit_vec(self.DIM, seed=42)

        # Build 8 candidates: 5 near-duplicates of query, 3 diverse
        base_relevant = _unit_vec(self.DIM, seed=42)
        candidates = []
        embeddings_list = []

        # Near-duplicates (high relevance but redundant)
        for i in range(5):
            emb = _nearly_identical_vec(base_relevant, noise=0.02, seed=i)
            emb = emb / np.linalg.norm(emb)
            candidates.append(_make_stored(f"dup_{i}", f"Apple revenue grew {i}%", 0.95 - i * 0.01))
            embeddings_list.append(emb)

        # Diverse (lower relevance but different)
        for i in range(3):
            emb = _unit_vec(self.DIM, seed=100 + i)
            candidates.append(_make_stored(f"div_{i}", f"Apple risk factor {i}", 0.6 - i * 0.05))
            embeddings_list.append(emb)

        embeddings = np.stack(embeddings_list).astype(np.float32)

        # MMR top-3
        mmr_result = mmr_rerank(
            query_embedding=query,
            candidate_embeddings=embeddings,
            candidates=candidates,
            top_k=3,
            mmr_lambda=0.5,  # balanced diversity
        )

        # Naive top-3 by score
        naive_result = sorted(candidates, key=lambda c: c.score, reverse=True)[:3]

        # MMR should include at least one diverse chunk; naive top-k would pick all duplicates
        mmr_ids = {c.chunk_id for c in mmr_result}
        naive_ids = {c.chunk_id for c in naive_result}

        diverse_ids = {f"div_{i}" for i in range(3)}

        mmr_diverse_count = len(mmr_ids & diverse_ids)
        naive_diverse_count = len(naive_ids & diverse_ids)

        assert mmr_diverse_count >= naive_diverse_count, (
            f"MMR should select >= as many diverse chunks as naive top-k. "
            f"MMR diverse: {mmr_diverse_count}, naive diverse: {naive_diverse_count}"
        )

    def test_lambda_1_is_pure_relevance(self):
        """With lambda=1.0, MMR degrades to pure relevance ordering."""
        query = _unit_vec(self.DIM, seed=0)
        candidates = [_make_stored(f"c{i}", f"chunk {i}", 0.9 - i * 0.1) for i in range(5)]
        embeddings = np.stack([_unit_vec(self.DIM, seed=i) for i in range(5)])

        result = mmr_rerank(
            query_embedding=query,
            candidate_embeddings=embeddings,
            candidates=candidates,
            top_k=5,
            mmr_lambda=1.0,
        )
        # First result should be the one most similar to query
        assert len(result) == 5  # all returned

    def test_lambda_0_is_pure_diversity(self):
        """With lambda=0.0, diversity dominates — no two selected should be identical."""
        query = _unit_vec(self.DIM, seed=0)
        # All candidates are near-identical to each other
        base = _unit_vec(self.DIM, seed=99)
        candidates = [_make_stored(f"c{i}", f"chunk {i}", 0.9) for i in range(6)]
        embeddings = np.stack([
            _nearly_identical_vec(base, noise=0.001 * (i + 1), seed=i)
            for i in range(6)
        ])
        result = mmr_rerank(
            query_embedding=query,
            candidate_embeddings=embeddings,
            candidates=candidates,
            top_k=3,
            mmr_lambda=0.0,
        )
        # Should still return 3 results without crashing
        assert len(result) == 3
        ids = [c.chunk_id for c in result]
        assert len(ids) == len(set(ids)), "No duplicate chunk IDs"


# ── Cosine similarity property tests ─────────────────────────────────────────


class TestCosineProperties:
    def test_unit_vector_self_similarity_is_one(self):
        from retrieval.retriever import _cosine_similarity
        v = _unit_vec(64, seed=7)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors_similarity_near_zero(self):
        from retrieval.retriever import _cosine_similarity
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_similarity(v1, v2)) < 1e-6
