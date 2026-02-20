"""
FinRAG async OpenAI LLM wrapper.

Features:
- Async calls with exponential backoff retry (3 attempts, jittered)
- SecretStr key access — API key never appears in logs or exceptions
- Token usage logging (prompt + completion)
- Config-swappable model (gpt-4o-mini default)

Part of FinRAG — a standalone finance-domain RAG system built as a companion
to SSB (Smart Strategies Builder).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import structlog
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import SecretStr

log = structlog.get_logger(__name__)

# ── Retry configuration ───────────────────────────────────────────────────────
MAX_RETRIES = 3
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    latency_ms: float


class OpenAILLM:
    """Async OpenAI chat wrapper for FinRAG generation.

    The API key is accessed via ``SecretStr.get_secret_value()`` only at call
    time — it never appears in object repr, logs, or exception messages.

    Args:
        api_key: OpenAI API key as a pydantic SecretStr.
        model: OpenAI model ID (default: gpt-4o-mini).
        temperature: Sampling temperature. 0.0 for deterministic financial Q&A.
        max_tokens: Maximum tokens in the model response.
        timeout_s: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: SecretStr,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1_000,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

        # Build client — key accessed once and not stored on self
        self._client = AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            timeout=timeout_s,
        )

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        request_id: str = "",
    ) -> LLMResponse:
        """Send a chat completion request with retry logic.

        Args:
            system_prompt: System message (finance persona + trust boundary).
            user_message: User message containing context + question.
            request_id: Optional request ID for log correlation.

        Returns:
            ``LLMResponse`` with content and token usage.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency_ms = (time.monotonic() - t0) * 1_000

                usage = response.usage
                content = response.choices[0].message.content or ""

                log.info(
                    "llm.generation_complete",
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=round(latency_ms, 1),
                    request_id=request_id,
                )

                return LLMResponse(
                    content=content,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    model=self.model,
                    latency_ms=latency_ms,
                )

            except RateLimitError as exc:
                last_exc = exc
                wait = _backoff(attempt)
                log.warning(
                    "llm.rate_limit",
                    attempt=attempt,
                    wait_s=wait,
                    request_id=request_id,
                )
                await asyncio.sleep(wait)

            except APITimeoutError as exc:
                last_exc = exc
                wait = _backoff(attempt)
                log.warning(
                    "llm.timeout",
                    attempt=attempt,
                    wait_s=wait,
                    request_id=request_id,
                )
                await asyncio.sleep(wait)

            except APIError as exc:
                last_exc = exc
                log.error(
                    "llm.api_error",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    request_id=request_id,
                    # Intentionally omit exc.message — may contain request content
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(_backoff(attempt))
                else:
                    break

        raise RuntimeError(
            f"LLM generation failed after {MAX_RETRIES} attempts. "
            f"Last error type: {type(last_exc).__name__}"
        ) from last_exc


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter."""
    ceiling = min(BASE_BACKOFF_S * (2 ** (attempt - 1)), MAX_BACKOFF_S)
    return random.uniform(0, ceiling)  # noqa: S311 — not cryptographic
