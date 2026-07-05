"""
Tests for pipeline/ephemeral_store.py.

Verifies session create/get round-trips, TTL-based eviction, capacity
enforcement, and thread-safety of the in-memory ephemeral session store.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ingestion.chunker import Chunk
from pipeline.ephemeral_store import (
    EphemeralSessionStore,
    SessionCapacityExceededError,
    SessionNotFoundError,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            text=f"chunk {i}",
            chunk_index=i,
            source_file="test.txt",
            page_num=1,
        )
        for i in range(n)
    ]


def _make_embeddings(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ── Create / get round-trip ───────────────────────────────────────────────────


class TestCreateAndGet:
    def test_create_returns_session_with_id(self):
        store = EphemeralSessionStore(ttl_seconds=60, max_sessions=10)
        session = store.create(
            filename="doc.txt",
            chunks=_make_chunks(3),
            embeddings=_make_embeddings(3),
            truncated=False,
            total_chunks_before_cap=3,
        )
        assert session.session_id
        assert session.filename == "doc.txt"
        assert len(session.chunks) == 3

    def test_get_returns_the_created_session(self):
        store = EphemeralSessionStore(ttl_seconds=60, max_sessions=10)
        created = store.create(
            filename="doc.txt",
            chunks=_make_chunks(2),
            embeddings=_make_embeddings(2),
            truncated=False,
            total_chunks_before_cap=2,
        )
        fetched = store.get(created.session_id)
        assert fetched.session_id == created.session_id
        assert fetched.filename == "doc.txt"

    def test_get_unknown_session_raises(self):
        store = EphemeralSessionStore(ttl_seconds=60, max_sessions=10)
        with pytest.raises(SessionNotFoundError):
            store.get("00000000-0000-0000-0000-000000000000")

    def test_count_reflects_active_sessions(self):
        store = EphemeralSessionStore(ttl_seconds=60, max_sessions=10)
        assert store.count() == 0
        store.create(
            filename="a.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        store.create(
            filename="b.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        assert store.count() == 2


# ── TTL eviction ──────────────────────────────────────────────────────────────


class TestTTLEviction:
    def test_expired_session_not_returned_by_get(self):
        store = EphemeralSessionStore(ttl_seconds=0, max_sessions=10)
        session = store.create(
            filename="doc.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        time.sleep(0.01)  # ensure we're past expires_at (ttl=0 -> expires immediately)
        with pytest.raises(SessionNotFoundError):
            store.get(session.session_id)

    def test_expired_session_evicted_on_next_create(self):
        # Use a real ttl for the store so the *second* session doesn't also
        # expire before we can observe it — force-expire only the first by
        # mutating its expires_at directly (deterministic, no sleep/race).
        store = EphemeralSessionStore(ttl_seconds=3600, max_sessions=10)
        first = store.create(
            filename="doc.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        first.expires_at = time.time() - 1  # force expiry
        store.create(
            filename="doc2.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        # The first session should have been swept away, leaving only the second.
        assert store.count() == 1

    def test_non_expired_session_survives(self):
        store = EphemeralSessionStore(ttl_seconds=3600, max_sessions=10)
        session = store.create(
            filename="doc.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        fetched = store.get(session.session_id)
        assert fetched.session_id == session.session_id


# ── Capacity enforcement ──────────────────────────────────────────────────────


class TestCapacityLimit:
    def test_capacity_exceeded_raises(self):
        store = EphemeralSessionStore(ttl_seconds=3600, max_sessions=2)
        store.create(
            filename="a.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        store.create(
            filename="b.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        with pytest.raises(SessionCapacityExceededError):
            store.create(
                filename="c.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
                truncated=False, total_chunks_before_cap=1,
            )

    def test_capacity_freed_by_expiry_allows_new_session(self):
        # Real ttl so the *replacement* session isn't itself born pre-expired;
        # force-expire only the first session deterministically.
        store = EphemeralSessionStore(ttl_seconds=3600, max_sessions=1)
        first = store.create(
            filename="a.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        first.expires_at = time.time() - 1  # force expiry
        # Should succeed: the first session expired and gets swept before the
        # capacity check runs.
        store.create(
            filename="b.txt", chunks=_make_chunks(1), embeddings=_make_embeddings(1),
            truncated=False, total_chunks_before_cap=1,
        )
        assert store.count() == 1


# ── Concurrency ───────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_creates_respect_capacity_cap(self):
        """Several threads racing create() simultaneously should never exceed
        max_sessions, exercising the threading.Lock around the dict."""
        max_sessions = 5
        store = EphemeralSessionStore(ttl_seconds=3600, max_sessions=max_sessions)
        num_threads = 20
        errors: list[Exception] = []
        successes: list[str] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                session = store.create(
                    filename=f"doc{i}.txt",
                    chunks=_make_chunks(1),
                    embeddings=_make_embeddings(1),
                    truncated=False,
                    total_chunks_before_cap=1,
                )
                with lock:
                    successes.append(session.session_id)
            except SessionCapacityExceededError as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert store.count() == max_sessions
        assert len(successes) == max_sessions
        assert len(errors) == num_threads - max_sessions
