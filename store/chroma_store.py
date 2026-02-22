"""
FinRAG ChromaDB vector store implementation.

Implements ``AbstractVectorStore`` using ChromaDB with persistent local storage.
Supports optional bearer-token auth for remote ChromaDB instances.

Swap this module for a pgvector adapter without touching retrieval or pipeline
code — the ``AbstractVectorStore`` interface is the only dependency.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

from typing import Any

import chromadb
import structlog

from store.base import AbstractVectorStore, StoredChunk

log = structlog.get_logger(__name__)


class ChromaVectorStore(AbstractVectorStore):
    """ChromaDB-backed vector store for FinRAG.

    Args:
        persist_dir: Directory for persistent ChromaDB storage.
        collection_name: ChromaDB collection name.
        auth_token: Optional bearer token (for remote ChromaDB instances;
                    unused in local PersistentClient mode).
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "finrag_docs",
        auth_token: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Local persistent client — no network, no auth required for local dev.
        # For remote ChromaDB with auth, replace with chromadb.HttpClient.
        self._client = chromadb.PersistentClient(path=persist_dir)

        if auth_token:
            log.warning(
                "chroma_store.auth_token_ignored",
                note="auth_token is only used with chromadb.HttpClient (remote). "
                "PersistentClient (local) does not support bearer auth.",
            )

        # Get or create collection — cosine distance for L2-normalised embeddings
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "chroma_store.initialized",
            persist_dir=persist_dir,
            collection=collection_name,
            doc_count=self._collection.count(),
        )

    # ── AbstractVectorStore interface ─────────────────────────────────────────

    def upsert(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update chunks. Existing IDs are overwritten."""
        if not chunk_ids:
            return
        self._collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        log.info("chroma_store.upserted", count=len(chunk_ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[StoredChunk]:
        """Return top-k chunks by cosine similarity."""
        n = min(top_k, max(self._collection.count(), 1))
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        chunks: list[StoredChunk] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        embeddings = result.get("embeddings", [[]])[0] or [None] * len(ids)

        for chunk_id, text, meta, dist, emb in zip(ids, docs, metas, distances, embeddings):
            # ChromaDB cosine distance = 1 − similarity → convert back
            score = float(1.0 - dist)
            chunks.append(
                StoredChunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    score=score,
                    metadata=meta or {},
                    embedding=list(emb) if emb is not None else None,
                )
            )

        return chunks

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        if not chunk_ids:
            return
        self._collection.delete(ids=chunk_ids)
        log.info("chroma_store.deleted", count=len(chunk_ids))

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._collection.count()

    def exists(self, chunk_id: str) -> bool:
        """Return True if chunk_id is already in the collection."""
        result = self._collection.get(ids=[chunk_id], include=[])
        return bool(result.get("ids"))
