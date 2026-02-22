"""
FinRAG abstract vector store interface.

Defines the contract that all vector store backends must satisfy.
ChromaDB is the default implementation; a pgvector adapter can replace it
without touching retrieval or pipeline code.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class StoredChunk:
    """A chunk as returned by a vector store query."""

    chunk_id: str
    text: str
    score: float  # cosine similarity (0–1)
    metadata: dict[str, Any]
    embedding: list[float] | None = None  # stored vector, if returned by the backend


class AbstractVectorStore(ABC):
    """Interface for all FinRAG vector store backends.

    Concrete implementations must be drop-in replaceable — callers depend only
    on this interface.
    """

    @abstractmethod
    def upsert(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update chunks in the store.

        Args:
            chunk_ids: Unique ID per chunk (e.g. SHA256 hash).
            embeddings: Corresponding embedding vectors.
            texts: Raw text for each chunk.
            metadatas: Arbitrary metadata dicts (source, page, ticker, etc.).
        """

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[StoredChunk]:
        """Return the top-k most similar chunks.

        Args:
            query_embedding: L2-normalised query vector.
            top_k: Number of results to return.

        Returns:
            List of ``StoredChunk`` objects, ordered by descending similarity.
        """

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID.

        Args:
            chunk_ids: IDs of chunks to remove.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks stored."""

    @abstractmethod
    def exists(self, chunk_id: str) -> bool:
        """Return True if a chunk with the given ID already exists."""
