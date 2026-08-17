"""OpenRouter generation client for the PCES dataset.

Async, bounded-concurrency generation with retries and per-call provenance.
Offline by default in tests; network access happens only through ``generate``.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
MAX_RETRIES = 4


@dataclass(frozen=True)
class GenerationResult:
    """One generated arm text with provenance.

    Attributes:
        text: The generated passage.
        model: The generating model id.
        latency_s: Wall-clock latency of the successful call.
    """

    text: str
    model: str
    latency_s: float


class GenerationError(RuntimeError):
    """A generation call failed after all retries."""


async def _call(
    client: httpx.AsyncClient, prompt: str, model: str, temperature: float, thinking: bool
) -> GenerationResult:
    """Make one chat completion call with retries."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise GenerationError("OPENROUTER_API_KEY is not set")
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "reasoning": {"enabled": thinking},
    }
    start = time.monotonic()
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"].get("content")
                if content and content.strip():
                    return GenerationResult(
                        text=content.strip(), model=model, latency_s=time.monotonic() - start
                    )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise GenerationError("empty completion content after retries")
            if resp.status_code in (429, 500, 502, 503):
                await asyncio.sleep(2**attempt)
                continue
            raise GenerationError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except httpx.HTTPError:
            if attempt == MAX_RETRIES - 1:
                raise GenerationError(f"network error after {MAX_RETRIES} attempts") from None
            await asyncio.sleep(2**attempt)
    raise GenerationError("unreachable")


async def generate_batch(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    concurrency: int = 16,
    thinking: bool = False,
) -> list[GenerationResult]:
    """Generate a batch of prompts with bounded concurrency.

    Args:
        prompts: The prompts to run; output order matches input order.
        model: OpenRouter model id.
        temperature: Sampling temperature.
        concurrency: Maximum in-flight requests.
        thinking: Enable provider reasoning; slower and costlier, better
            rubric adherence for judging.

    Returns:
        One ``GenerationResult`` per prompt, in input order.
    """
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:

        async def one(prompt: str) -> GenerationResult:
            async with sem:
                return await _call(client, prompt, model, temperature, thinking)

        return await asyncio.gather(*(one(p) for p in prompts))
