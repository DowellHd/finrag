"""
FinRAG BGE embedding wrapper.

Wraps BAAI/bge-small-en-v1.5 from sentence-transformers with:
- L2-normalised output (cosine-ready dot products)
- Batch encoding with tqdm progress bars
- Auto device selection: CUDA → MPS → CPU
- Model caching in ./models/ to avoid re-download

BGE was chosen over MiniLM for stronger finance benchmark scores and better
performance on domain-specific dense retrieval tasks. See README design decisions.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from ingestion.chunker import Chunk

log = structlog.get_logger(__name__)

# BGE models perform best with this query prefix during retrieval
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _select_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class BGEEmbedder:
    """Sentence-transformer BGE embedding wrapper for FinRAG.

    Args:
        model_name: HuggingFace model ID (default: BAAI/bge-small-en-v1.5).
        cache_dir: Local directory to cache model weights.
        device: Compute device (auto-detected if not supplied).
        batch_size: Encoding batch size.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str = "./models",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or _select_device()
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    # ── Lazy loading ──────────────────────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        """Load model on first access (lazy init)."""
        if self._model is None:
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(
                "embedder.loading_model",
                model=self.model_name,
                device=self.device,
                cache_dir=self.cache_dir,
            )
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
            )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_documents(
        self,
        texts: list[str],
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode document passages for storage.

        BGE document embeddings do NOT use the query prefix.

        Args:
            texts: List of passage strings.
            show_progress: Show tqdm progress bar.

        Returns:
            Float32 numpy array of shape ``(len(texts), embedding_dim)``,
            L2-normalised (unit vectors, cosine-ready).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalise → cosine via dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)  # type: ignore[union-attr]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single user query for retrieval.

        Prepends the BGE query prefix for improved retrieval quality.

        Args:
            query: User question string.

        Returns:
            Float32 numpy array of shape ``(embedding_dim,)``, L2-normalised.
        """
        prefixed = BGE_QUERY_PREFIX + query
        embedding = self.model.encode(
            [prefixed],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)  # type: ignore[index]

    # ── Chunk embedding convenience ───────────────────────────────────────────

    def embed_chunks(
        self,
        chunks: list[Chunk],
        *,
        show_progress: bool = True,
    ) -> tuple[list[str], np.ndarray]:
        """Embed a list of ``Chunk`` objects for storage.

        Args:
            chunks: Output from ``ingestion.chunker.chunk_page_docs``.
            show_progress: Show tqdm progress bar.

        Returns:
            Tuple of (texts, embeddings_matrix) where embeddings are L2-normalised.
        """
        texts = [c.text for c in chunks]
        embeddings = self.encode_documents(texts, show_progress=show_progress)
        log.info(
            "embedder.chunks_embedded",
            n_chunks=len(chunks),
            embedding_dim=embeddings.shape[1] if embeddings.ndim == 2 else 0,
            device=self.device,
        )
        return texts, embeddings
