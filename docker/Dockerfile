# FinRAG — Multi-stage Dockerfile
# Runs as non-root user 'finrag' with minimal capabilities.
# Secrets are injected at runtime via --env-file or environment variables.
# No secrets are baked into the image.

# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (gcc for some C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into a prefix we can copy over
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: create non-root user
RUN groupadd --system finrag && useradd --system --gid finrag --shell /bin/false finrag

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source — chroma_db/ and models/ are excluded via .dockerignore
COPY --chown=finrag:finrag . .

# Runtime directories — owned by non-root user
RUN mkdir -p /app/chroma_db /app/models /app/data/sample \
    && chown -R finrag:finrag /app

# Switch to non-root user
USER finrag

# Expose API port
EXPOSE 8000

# Health check (matches /health endpoint — no internal details exposed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Default command: start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
