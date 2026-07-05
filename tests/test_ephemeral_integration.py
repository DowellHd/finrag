"""
Integration test for the /upload -> /ephemeral-query round trip.

Uses FastAPI's TestClient against the real app, but patches build_pipeline
so lifespan startup never touches real ChromaDB or the OpenAI API — the
pipeline's embedder/llm are fully faked, matching the existing offline-test
convention (see tests/test_pipeline.py's fake_llm fixture).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from generation.llm import LLMResponse

DIM = 8


class _FakeEmbedder:
    """Deterministic fake embedder — no real OpenAI calls."""

    def embed_chunks(self, chunks, *, show_progress: bool = True):
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((len(chunks), DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        return [c.text for c in chunks], vecs

    def encode_query(self, query: str) -> np.ndarray:
        rng = np.random.default_rng(1)
        v = rng.standard_normal(DIM).astype(np.float32)
        return v / np.linalg.norm(v)


class _FakeLLM:
    async def generate(self, system_prompt: str, user_message: str, *, request_id: str = ""):
        return LLMResponse(
            content="Based on the uploaded document, here is the answer.",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            model="fake-model",
            latency_ms=1.0,
        )


class _FakePipeline:
    """Only the attributes the /upload and /ephemeral-query endpoints touch."""

    def __init__(self) -> None:
        self.embedder = _FakeEmbedder()
        self.llm = _FakeLLM()


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr("api.main.build_pipeline", lambda settings: _FakePipeline())
    from api.main import app

    with TestClient(app) as c:
        yield c


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestUploadAndEphemeralQuery:
    def test_upload_returns_session(self, client):
        files = {"file": ("my_notes.txt", b"Some financial document content.", "text/plain")}
        res = client.post("/upload", files=files)
        assert res.status_code == 200
        body = res.json()
        assert body["session_id"]
        assert body["filename"] == "my_notes.txt"
        assert body["chunks_indexed"] > 0
        assert body["truncated"] is False

    def test_full_round_trip_answers_question(self, client):
        files = {"file": ("statement.txt", b"Total amount due: $482.19. Due date: March 1.", "text/plain")}
        upload_res = client.post("/upload", files=files)
        assert upload_res.status_code == 200
        session_id = upload_res.json()["session_id"]

        query_res = client.post(
            "/ephemeral-query",
            json={"session_id": session_id, "question": "What is the total amount due?"},
        )
        assert query_res.status_code == 200
        body = query_res.json()
        assert body["answer"]
        assert isinstance(body["sources"], list)
        assert len(body["sources"]) > 0
        assert body["sources"][0]["source_file"] == "statement.txt"
        assert body["sources"][0]["ticker"] is None
        assert body["sources"][0]["doc_type"] is None

    def test_bogus_session_id_returns_404(self, client):
        res = client.post(
            "/ephemeral-query",
            json={"session_id": "00000000-0000-0000-0000-000000000000", "question": "Anything?"},
        )
        assert res.status_code == 404

    def test_disallowed_extension_returns_400(self, client):
        files = {"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")}
        res = client.post("/upload", files=files)
        assert res.status_code == 400

    def test_empty_file_returns_400(self, client):
        files = {"file": ("empty.txt", b"", "text/plain")}
        res = client.post("/upload", files=files)
        assert res.status_code == 400
