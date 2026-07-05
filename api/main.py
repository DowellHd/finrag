"""
FinRAG FastAPI application.

Exposes five endpoints:
- POST /ingest           — ingest a financial document into the permanent vector store
- POST /query            — ask a financial question and get a grounded answer
- POST /upload           — upload an ephemeral document for session-scoped Q&A
- POST /ephemeral-query  — ask a question about a previously uploaded ephemeral document
- GET  /health           — liveness check (no internal details exposed)

Security layers:
- Rate limiting via slowapi (per-IP)
- Strict Pydantic request models with field constraints
- Path validation delegated to security.validators before any I/O
- Generic error responses — detailed errors go to structured logs only
- CORS restricted to configured origins

Ephemeral uploads (/upload, /ephemeral-query) are processed in memory only and
are never persisted into the permanent corpus — see pipeline/ephemeral_store.py
and pipeline/ephemeral_pipeline.py.

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from config.settings import settings
from ingestion.chunker import chunk_page_docs
from ingestion.loader import load_document
from pipeline.ephemeral_pipeline import answer_ephemeral_query
from pipeline.ephemeral_store import (
    EphemeralSessionStore,
    SessionCapacityExceededError,
    SessionNotFoundError,
)
from pipeline.rag_pipeline import RAGPipeline, build_pipeline
from security.validators import (
    ValidationError,
    SuspiciousQueryError,
    PathTraversalError,
    FileTooLargeError,
    ExtensionNotAllowedError,
    validate_upload_bytes,
)

log = structlog.get_logger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")


# ── App lifecycle ─────────────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None
_ephemeral_store: EphemeralSessionStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once on startup and reuse across requests."""
    global _pipeline, _ephemeral_store
    log.info("api.startup", host=settings.api_host, port=settings.api_port)
    _pipeline = build_pipeline(settings)
    _ephemeral_store = EphemeralSessionStore(
        ttl_seconds=settings.ephemeral_session_ttl_seconds,
        max_sessions=settings.ephemeral_max_sessions,
    )
    yield
    log.info("api.shutdown")


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised.")
    return _pipeline


def get_ephemeral_store() -> EphemeralSessionStore:
    if _ephemeral_store is None:
        raise RuntimeError("Ephemeral session store not initialised.")
    return _ephemeral_store


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
    ticker: str | None = Field(
        default=None,
        max_length=10,
        pattern=r"^[A-Z0-9\.\-]{1,10}$",
        description="Restrict retrieval to a single company (e.g. AAPL). Omit to search all indexed companies.",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _upper_ticker(cls, v):
        return v.upper() if isinstance(v, str) else v


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


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_indexed: int
    truncated: bool
    expires_in_seconds: int
    request_id: str


class EphemeralQueryRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=32,
        max_length=36,
        description="session_id returned by /upload.",
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question about the uploaded document.",
    )


class EphemeralQueryResponse(BaseModel):
    answer: str
    sources: list[SourceRefOut]
    truncated_notice: str | None
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    doc_count: int


class CompaniesResponse(BaseModel):
    tickers: list[str]


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
        result = await pipeline.query(body.question, ticker=body.ticker)
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


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload an ephemeral document for session-scoped Q&A",
)
@limiter.limit(f"{settings.rate_limit_upload}/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
    session_store: EphemeralSessionStore = Depends(get_ephemeral_store),
) -> UploadResponse:
    """Upload a document (PDF, TXT, MD) for ephemeral, session-scoped Q&A.

    The document is processed in memory only — never written to the
    permanent corpus — and is discarded after
    ``settings.ephemeral_session_ttl_seconds`` or on capacity eviction.
    """
    rid = _request_id()

    # Bounded read regardless of a missing/spoofed Content-Length header.
    raw = await file.read(settings.ephemeral_max_file_size_bytes + 1)

    try:
        safe_name = validate_upload_bytes(
            file.filename or "upload",
            raw,
            max_size_bytes=settings.ephemeral_max_file_size_bytes,
        )
    except FileTooLargeError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            log_msg="api.upload.file_too_large",
            request_id=rid,
            user_msg="Uploaded file exceeds the maximum allowed size.",
        )
    except ExtensionNotAllowedError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.upload.extension_rejected",
            request_id=rid,
            user_msg="Unsupported file type. Allowed: .pdf, .txt, .md",
        )
    except ValidationError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.upload.validation_error",
            request_id=rid,
            user_msg="Upload validation failed.",
        )

    tmp_path: Path | None = None
    try:
        suffix = Path(safe_name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        pages = await run_in_threadpool(
            load_document,
            tmp_path,
            doc_type="uploaded",
            override_filename=safe_name,
        )
    except Exception as exc:  # noqa: BLE001 — malformed/unparseable upload
        log.warning(
            "api.upload.parse_failed", request_id=rid, error_type=type(exc).__name__
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not extract text from the uploaded document. [req:{rid}]",
        ) from None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    chunks = chunk_page_docs(
        pages, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No extractable text found in the uploaded document. [req:{rid}]",
        )

    total_before_cap = len(chunks)
    truncated = total_before_cap > settings.ephemeral_max_chunks_per_doc
    if truncated:
        chunks = chunks[: settings.ephemeral_max_chunks_per_doc]

    _, embeddings = await run_in_threadpool(
        pipeline.embedder.embed_chunks, chunks, show_progress=False
    )

    try:
        session = session_store.create(
            filename=safe_name,
            chunks=chunks,
            embeddings=embeddings,
            truncated=truncated,
            total_chunks_before_cap=total_before_cap,
        )
    except SessionCapacityExceededError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            log_msg="api.upload.capacity_exceeded",
            request_id=rid,
            user_msg="Too many active demo sessions right now — please try again shortly.",
        )

    log.info(
        "api.upload.complete",
        request_id=rid,
        filename=safe_name,
        chunks=len(chunks),
        truncated=truncated,
        size_bytes=len(raw),
    )

    return UploadResponse(
        session_id=session.session_id,
        filename=safe_name,
        chunks_indexed=len(chunks),
        truncated=truncated,
        expires_in_seconds=settings.ephemeral_session_ttl_seconds,
        request_id=rid,
    )


@app.post(
    "/ephemeral-query",
    response_model=EphemeralQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about a previously uploaded ephemeral document",
)
@limiter.limit(f"{settings.rate_limit_ephemeral_query}/minute")
async def ephemeral_query(
    request: Request,
    body: EphemeralQueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    session_store: EphemeralSessionStore = Depends(get_ephemeral_store),
) -> EphemeralQueryResponse:
    """Answer a question about a document uploaded via /upload.

    Returns 404 if the session_id is unknown or has expired — the caller
    should prompt the user to re-upload their document in that case.
    """
    rid = _request_id()

    try:
        session = session_store.get(body.session_id)
    except SessionNotFoundError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_404_NOT_FOUND,
            log_msg="api.ephemeral_query.session_not_found",
            request_id=rid,
            user_msg="Upload session not found or has expired. Please upload your document again.",
        )

    try:
        result = await answer_ephemeral_query(
            session,
            body.question,
            embedder=pipeline.embedder,
            llm=pipeline.llm,
            top_k=settings.top_k,
            mmr_lambda=settings.mmr_lambda,
        )
    except SuspiciousQueryError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.ephemeral_query.suspicious_query",
            request_id=rid,
            user_msg="Query was rejected by the security filter.",
        )
    except ValidationError as exc:
        _log_and_raise(
            exc,
            http_status=status.HTTP_400_BAD_REQUEST,
            log_msg="api.ephemeral_query.validation_error",
            request_id=rid,
            user_msg="Query validation failed.",
        )
    except Exception as exc:  # noqa: BLE001
        log.error(
            "api.ephemeral_query.unexpected_error",
            request_id=rid,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error. [req:{rid}]",
        ) from None

    return EphemeralQueryResponse(
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
        truncated_notice=result.truncated_notice,
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


@app.get(
    "/companies",
    response_model=CompaniesResponse,
    summary="List tickers currently indexed",
)
async def companies(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> CompaniesResponse:
    """Return the distinct tickers currently in the vector store.

    Used by client UIs (e.g. a company picker) to discover which companies
    can be queried without hardcoding the list.
    """
    try:
        tickers = pipeline.store.list_tickers()
    except Exception:  # noqa: BLE001
        tickers = []

    return CompaniesResponse(tickers=tickers)
