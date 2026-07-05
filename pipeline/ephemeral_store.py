"""
FinRAG ephemeral session store.

Holds session-scoped, in-memory-only document uploads for the "upload your
own document" demo flow. Deliberately does NOT implement
``store.base.AbstractVectorStore`` — there is no persistence semantics here
by design: uploaded documents (which may contain PII such as tax forms or
billing statements) are never written to the permanent ChromaDB corpus and
are discarded on TTL expiry, process restart, or capacity eviction.

Single-instance, in-process, dict-based — sufficient for a single Render
free-tier dyno. No Redis/shared storage needed, and none of this state is
expected (or intended) to survive a restart.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
import structlog

from ingestion.chunker import Chunk

log = structlog.get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class EphemeralSessionError(Exception):
    """Base for ephemeral-session lifecycle errors (not input validation)."""


class SessionNotFoundError(EphemeralSessionError):
    """Raised when a session_id is unknown, expired, or already evicted."""


class SessionCapacityExceededError(EphemeralSessionError):
    """Raised when the in-memory session cap is reached."""


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class EphemeralSession:
    """An uploaded document held in memory for the duration of one session."""

    session_id: str
    filename: str
    created_at: float
    expires_at: float
    chunks: list[Chunk]
    embeddings: np.ndarray  # (n, dim) float32, L2-normalised, aligned with chunks
    truncated: bool
    total_chunks_before_cap: int


# ── Store ─────────────────────────────────────────────────────────────────────


class EphemeralSessionStore:
    """Thread-safe, TTL-evicting, dict-based store for ephemeral upload sessions."""

    def __init__(self, *, ttl_seconds: int, max_sessions: int) -> None:
        self._sessions: dict[str, EphemeralSession] = {}
        # threading.Lock, NOT asyncio.Lock: endpoint handlers call the
        # synchronous OpenAI embedder via starlette's run_in_threadpool,
        # which really does dispatch to OS threads — an asyncio.Lock only
        # protects interleaving within one event-loop thread and would not
        # prevent a race between two concurrent create() calls both passing
        # the capacity check before either inserts.
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions

    def _evict_expired_locked(self) -> None:
        """Must be called while holding self._lock."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if s.expires_at <= now]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            log.info("ephemeral_store.evicted", count=len(expired))

    def create(
        self,
        *,
        filename: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        truncated: bool,
        total_chunks_before_cap: int,
    ) -> EphemeralSession:
        with self._lock:
            self._evict_expired_locked()
            if len(self._sessions) >= self._max_sessions:
                raise SessionCapacityExceededError(
                    f"Ephemeral session capacity ({self._max_sessions}) reached."
                )
            now = time.time()
            session = EphemeralSession(
                session_id=str(uuid.uuid4()),
                filename=filename,
                created_at=now,
                expires_at=now + self._ttl,
                chunks=chunks,
                embeddings=embeddings,
                truncated=truncated,
                total_chunks_before_cap=total_chunks_before_cap,
            )
            self._sessions[session.session_id] = session
            log.info(
                "ephemeral_store.created",
                session_id=session.session_id,
                chunks=len(chunks),
                truncated=truncated,
            )
            return session

    def get(self, session_id: str) -> EphemeralSession:
        with self._lock:
            self._evict_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError("Session not found or expired.")
            return session

    def count(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self._sessions)
