"""
FinRAG security validators.

Centralised input validation layer — committed second (after repo init) to
signal that security is architectural, not an afterthought.

Covers:
- Path traversal prevention
- File extension whitelisting
- File size enforcement
- Prompt injection pattern detection

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

# Patterns that suggest prompt injection attempts in user queries.
# We log these for monitoring but do NOT expose them to the LLM as instructions.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Covers "ignore previous/all/prior/above ... instructions/prompts/context"
    # and multi-word variants like "ignore all prior instructions"
    re.compile(r"ignore\s+(?:\w+\s+){0,3}(?:instructions?|prompts?|context)", re.I),
    re.compile(r"system\s*:", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)\b", re.I),
    re.compile(r"disregard\s+(your|all|previous)", re.I),
    re.compile(r"forget\s+(everything|all|prior)", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
    re.compile(r"prompt\s*injection", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\b", re.I),  # "Do Anything Now" jailbreak prefix
]


# ── Exceptions ────────────────────────────────────────────────────────────────


class ValidationError(ValueError):
    """Raised when an input fails a security validation check."""


class PathTraversalError(ValidationError):
    """Raised when a path contains traversal sequences."""


class ExtensionNotAllowedError(ValidationError):
    """Raised when a file extension is not in the whitelist."""


class FileTooLargeError(ValidationError):
    """Raised when a file exceeds the configured size limit."""


class SuspiciousQueryError(ValidationError):
    """Raised when a query matches a known injection pattern."""


# ── Path Validation ───────────────────────────────────────────────────────────


def validate_file_path(
    raw_path: str,
    *,
    max_size_bytes: int,
    allowed_extensions: frozenset[str] = ALLOWED_EXTENSIONS,
) -> Path:
    """Validate a file path before any file I/O is performed.

    Checks (in order):
    1. Resolve to absolute path — detect traversal sequences (../../).
    2. Extension whitelist.
    3. File exists on disk.
    4. File size within allowed limit.

    Args:
        raw_path: The raw string path supplied by the caller.
        max_size_bytes: Maximum allowed file size in bytes.
        allowed_extensions: Set of permitted extensions (default: .pdf, .txt, .md).

    Returns:
        Resolved, validated ``pathlib.Path`` object.

    Raises:
        PathTraversalError: If the resolved path contains traversal artifacts.
        ExtensionNotAllowedError: If the file extension is not whitelisted.
        ValidationError: If the file does not exist.
        FileTooLargeError: If the file exceeds ``max_size_bytes``.
    """
    # --- 1. Traversal check BEFORE any normalization -------------------------
    # Split on both Unix and Windows separators to catch ".." in raw input.
    # os.path.normpath would silently collapse traversal sequences before we see them.
    raw_parts = re.split(r"[/\\]", raw_path)
    if ".." in raw_parts:
        log.warning(
            "security.path_traversal_attempt",
            raw_path=raw_path,
        )
        raise PathTraversalError("Path traversal sequences are not allowed.")

    # Resolve to absolute path for subsequent checks
    try:
        resolved = Path(raw_path).resolve()
    except Exception as exc:  # noqa: BLE001
        raise PathTraversalError(f"Could not resolve path: {exc}") from exc

    # --- 2. Extension whitelist ----------------------------------------------
    suffix = resolved.suffix.lower()
    if suffix not in allowed_extensions:
        log.warning(
            "security.extension_rejected",
            extension=suffix,
            allowed=sorted(allowed_extensions),
        )
        raise ExtensionNotAllowedError(
            f"File extension '{suffix}' is not allowed. "
            f"Permitted: {sorted(allowed_extensions)}"
        )

    # --- 3. Existence check --------------------------------------------------
    if not resolved.is_file():
        raise ValidationError(f"File not found: {resolved}")

    # --- 4. Size check -------------------------------------------------------
    file_size = resolved.stat().st_size
    if file_size > max_size_bytes:
        log.warning(
            "security.file_too_large",
            file_size_bytes=file_size,
            max_size_bytes=max_size_bytes,
        )
        raise FileTooLargeError(
            f"File size {file_size:,} bytes exceeds limit of {max_size_bytes:,} bytes."
        )

    return resolved


# ── Query Sanitisation ────────────────────────────────────────────────────────


def sanitize_query(query: str, *, max_length: int = 500) -> str:
    """Sanitise a user query before it is passed to the LLM pipeline.

    Strips leading/trailing whitespace, enforces length, and checks for
    prompt-injection patterns.  Suspicious queries are *logged* for monitoring
    (never exposed to the LLM as trusted instructions) but are still processed
    — the document trust boundary in the system prompt is the primary defence.

    Args:
        query: Raw user query string.
        max_length: Maximum allowed character count.

    Returns:
        Stripped, validated query string.

    Raises:
        ValidationError: If the query is empty or exceeds ``max_length``.
        SuspiciousQueryError: If a known injection pattern is detected.
    """
    cleaned = query.strip()

    if not cleaned:
        raise ValidationError("Query must not be empty.")

    if len(cleaned) > max_length:
        raise ValidationError(
            f"Query length {len(cleaned)} exceeds maximum of {max_length} characters."
        )

    # Check for injection patterns — log and raise
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            log.warning(
                "security.suspicious_query_detected",
                pattern=pattern.pattern,
                # We intentionally do NOT log the full query text (PII / injection content risk).
            )
            raise SuspiciousQueryError(
                "Query contains patterns that are not allowed. "
                "Please rephrase your question."
            )

    return cleaned


# ── Convenience re-exports ────────────────────────────────────────────────────

__all__ = [
    "validate_file_path",
    "sanitize_query",
    "ValidationError",
    "PathTraversalError",
    "ExtensionNotAllowedError",
    "FileTooLargeError",
    "SuspiciousQueryError",
    "ALLOWED_EXTENSIONS",
]
