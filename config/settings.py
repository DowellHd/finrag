"""
FinRAG configuration module.

Centralises all runtime settings using pydantic-settings so every value is
validated at startup and secrets are never exposed via logs or repr.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),  # allow model_* field names
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(
        ...,
        description="OpenAI API key — stored as SecretStr, never logged.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model ID.",
    )

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model ID.",
    )
    model_cache_dir: str = Field(
        default="./models",
        description="Local directory for cached model weights.",
    )

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Persistent storage path for ChromaDB.",
    )
    chroma_collection: str = Field(
        default="finrag_docs",
        description="ChromaDB collection name.",
    )
    chroma_auth_token: SecretStr | None = Field(
        default=None,
        description="Optional ChromaDB bearer token.",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=50)
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity.",
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)

    # ── File Ingestion ────────────────────────────────────────────────────────
    max_file_size_bytes: int = Field(
        default=52_428_800,  # 50 MB
        description="Hard upper limit on uploaded file sizes.",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(
        default=["https://dowellstandley.com", "http://localhost:3000", "http://localhost:8080"]
    )
    rate_limit_ingest: int = Field(
        default=10,
        description="Max ingest requests per minute per IP.",
    )
    rate_limit_query: int = Field(
        default=30,
        description="Max query requests per minute per IP.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | list[str]) -> list[str]:
        """Accept comma-separated string or a proper list."""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
