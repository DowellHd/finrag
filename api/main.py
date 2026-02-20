"""
FinRAG FastAPI application.

Exposes three endpoints:
- POST /ingest  — ingest a financial document into the vector store
- POST /query   — ask a financial question and get a grounded answer
- GET  /health  — liveness check (no internal details exposed)

Security layers:
- Rate limiting via slowapi (per-IP)
- Strict Pydantic request models with field constraints
- Path validation delegated to security.validators before any I/O
- Generic error responses — detailed errors go to structured logs only
- CORS restricted to configured origins

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config.settings import settings
from pipeline.rag_pipeline import RAGPipeline, build_pipeline
from security.validators import (
    ValidationError,
    SuspiciousQueryError,
    PathTraversalError,
    FileTooLargeError,
    ExtensionNotAllowedError,
)

log = structlog.get_logger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")


# ── App lifecycle ─────────────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once on startup and reuse across requests."""
    global _pipeline
    log.info("api.startup", host=settings.api_host, port=settings.api_port)
    _pipeline = build_pipeline(settings)
    yield
    log.info("api.shutdown")


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised.")
    return _pipeline


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinRAG",
    description=(
        "Finance-domain RAG system for document Q&A. "
        "Companion project to SSB (Smart Strategies Builder)."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


# ── Request / response models ─────────────────────────────────────────────────


class IngestRequest(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Absolute or relative path to the document to ingest.",
    )
    ticker: str | None = Field(
        default=None,
        max_length=10,
        pattern=r"^[A-Z0-9\.\-]{1,10}$",
        description="Ticker symbol (e.g. AAPL). Optional if encoded in filename.",
    )
    doc_type: str | None = Field(
        default=None,
        max_length=20,
        description="Document type (e.g. 10-K, 10-Q, earnings).",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _upper_ticker(cls, v):
        return v.upper() if isinstance(v, str) else v


class IngestResponse(BaseModel):
    source_file: str
    chunks_stored: int
    chunks_skipped: int
    total_chunks: int
    request_id: str


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Financial question to answer from ingested documents.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve.",
    )


class SourceRefOut(BaseModel):
    source_file: str
    page_num: int
    ticker: str | None
    doc_type: str | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRefOut]
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    doc_count: int


# ── Error handling helpers ────────────────────────────────────────────────────


def _request_id() -> str:
    return str(uuid.uuid4())[:8]


def _log_and_raise(
    exc: Exception,
    *,
    http_status: int,
    log_msg: str,
    request_id: str,
    user_msg: str,
) -> None:
    log.warning(log_msg, request_id=request_id, error_type=type(exc).__name__)
    raise HTTPException(status_code=http_status, detail=f"{user_msg} [req:{request_id}]")


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a financial document",
)
@limiter.limit(f"{settings.rate_limit_ingest}/minute")
async def ingest(
    request: Request,
    body: IngestRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Ingest a PDF, TXT, or MD financial document into the vector store.

    The file path is validated server-side against path traversal and extension
    whitelist before any I/O is performed.
    """
    rid = _request_id()

    try:
        result = pipeline.ingest(
            body.file_path,
            ticker=body.ticker,
            doc_type=body.doc_type,
        )
    except (PathTraversalError, ExtensionNotAllowedError) as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.ingest.invalid_path",
            request_id=rid,
            user_msg="Invalid file path or extension.",
        )
    except FileTooLargeError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            log_msg="api.ingest.file_too_large",
            request_id=rid,
            user_msg="File exceeds the maximum allowed size.",
        )
    except ValidationError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.ingest.validation_error",
            request_id=rid,
            user_msg="Ingestion validation failed.",
        )
    except Exception as exc:  # noqa: BLE001
        log.error("api.ingest.unexpected_error", request_id=rid, error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error. [req:{rid}]",
        ) from None

    return IngestResponse(
        source_file=result.source_file,
        chunks_stored=result.chunks_stored,
        chunks_skipped=result.chunks_skipped,
        total_chunks=result.total_chunks,
        request_id=rid,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a financial question",
)
@limiter.limit(f"{settings.rate_limit_query}/minute")
async def query(
    request: Request,
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Retrieve relevant context from ingested documents and generate a grounded answer."""
    rid = _request_id()

    try:
        result = await pipeline.query(body.question)
    except SuspiciousQueryError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.query.suspicious_query",
            request_id=rid,
            user_msg="Query was rejected by the security filter.",
        )
    except ValidationError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.query.validation_error",
            request_id=rid,
            user_msg="Query validation failed.",
        )
    except Exception as exc:  # noqa: BLE001
        log.error("api.query.unexpected_error", request_id=rid, error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error. [req:{rid}]",
        ) from None

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceRefOut(
                source_file=s.source_file,
                page_num=s.page_num,
                ticker=s.ticker,
                doc_type=s.doc_type,
            )
            for s in result.sources
        ],
        latency_ms=round(result.latency_ms, 1),
        request_id=result.request_id,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
)
async def health(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> HealthResponse:
    """Lightweight health check.

    Returns only operational status and document count.
    Internal paths, model names, DB locations, and versions are NOT exposed.
    """
    try:
        doc_count = pipeline.store.count()
    except Exception:  # noqa: BLE001
        doc_count = -1

    return HealthResponse(status="ok", doc_count=doc_count)
