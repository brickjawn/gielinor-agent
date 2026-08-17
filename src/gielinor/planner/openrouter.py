"""Async OpenRouter client: retries, backoff, model fallback, no leaked key.

Three things about the free tier drive this design, and all three are properties
of the *roster*, not of any one model:

**Free slugs disappear.** The list in SPEC section 1 was verified in Aug 2026
and will rot. So the client takes a *chain* and walks it: a 404 on a slug that
was retired last week costs one request and then it moves on, rather than taking
the agent down. ``scripts/list_models.py`` re-checks the roster at runtime.

**Free capacity is rationed.** 429 is normal, not exceptional. Retries use
exponential backoff with full jitter (uncorrelated retries; synchronised ones
just rebuild the spike) and honour ``Retry-After`` when the server sends it.

**Latency is 700 ms to 6 s.** That is why nothing here is on the tick path, and
why the timeout is generous rather than aggressive: a slow answer is still worth
having, since the runtime revalidates it against fresh state before dispatch.

On the key: it is read from ``OPENROUTER_API_KEY``, held in a private attribute,
and never logged, never put in a repr, never written to a trace. The only place
it appears is the Authorization header. Log lines carry the model, status and
latency; if you need the request body for debugging, log it from the caller
where you can see what you are about to disclose.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable, Sequence

    import numpy as np

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# SPEC section 1, verified Aug 2026. Text-first: perception has usually already
# extracted what the model needs, and the text-only models are much faster.
DEFAULT_TEXT_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
)
DEFAULT_VISION_MODELS: tuple[str, ...] = (
    "google/gemma-4-26b-a4b-it:free",  # best free image reasoning
    "nvidia/nemotron-nano-12b-v2-vl:free",  # smaller, lower latency, multi-image
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",  # slowest; last resort
)
DEFAULT_REASONING_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
)

# Transient: worth retrying the same model.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# Terminal for this model but not for the request: try the next slug.
NEXT_MODEL_STATUS: frozenset[int] = frozenset({400, 402, 403, 404, 413, 422})


class OpenRouterError(RuntimeError):
    """Base class for OpenRouter failures."""


class AuthError(OpenRouterError):
    """401/403 on the key itself. Retrying will not help and may get you limited."""


class AllModelsFailed(OpenRouterError):
    """Every model in the chain failed. Carries the per-model reasons."""

    def __init__(self, attempts: Sequence[str]) -> None:
        self.attempts = list(attempts)
        super().__init__("all models failed: " + "; ".join(attempts))


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    model: str
    attempts: int
    latency_ms: float
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


def build_image_data_url(
    image: np.ndarray | bytes,
    *,
    fmt: str = "jpeg",
    quality: int = 72,
    max_width: int | None = 765,
) -> str:
    """Encode a BGR frame (or raw bytes) as a data URL for the chat API.

    JPEG at moderate quality on purpose: a 765x503 PNG is ~400 KB of base64 and
    the free tier is not generous with request size. Game text survives JPEG at
    q72 well enough for a model to read, and perception — which needs exact
    pixels — never uses this path.
    """
    if isinstance(image, bytes):
        payload = image
    else:
        import cv2

        array = image
        if max_width is not None and array.shape[1] > max_width:
            scale = max_width / array.shape[1]
            array = cv2.resize(
                array, (max_width, int(array.shape[0] * scale)), interpolation=cv2.INTER_AREA
            )
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpeg" else []
        ok, buf = cv2.imencode(f".{fmt}", array, params)
        if not ok:
            raise ValueError(f"could not encode frame as {fmt}")
        payload = buf.tobytes()
    return f"data:image/{fmt};base64,{base64.b64encode(payload).decode('ascii')}"


def user_message(text: str, image_data_url: str | None = None) -> dict[str, Any]:
    """Build a user turn, multimodal only when an image is actually attached."""
    if image_data_url is None:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    }


class OpenRouterClient:
    """Minimal async client for the OpenRouter chat completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        models: Sequence[str] = DEFAULT_TEXT_MODELS,
        base_url: str = BASE_URL,
        timeout_s: float = 20.0,
        connect_timeout_s: float = 5.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
        referer: str = "https://github.com/example/gielinor-agent",
        title: str = "gielinor-agent",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        # Private and underscored so a stray `vars(client)` in a debug print is
        # at least conspicuous; nothing in this package ever serialises it.
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        self.models = tuple(models)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            transport=transport,
            headers={
                # OpenRouter uses these for attribution on the free tier.
                "HTTP-Referer": referer,
                "X-Title": title,
                "Content-Type": "application/json",
            },
        )

    # -- plumbing ----------------------------------------------------------

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def _auth_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise AuthError(
                f"no API key: set {API_KEY_ENV} in the environment or a .env file. "
                "The agent runs fine without one — the reflex planner needs no network — "
                "but escalations will be skipped."
            )
        return {"Authorization": f"Bearer {self._api_key}"}

    def __repr__(self) -> str:
        # Explicitly redacted. An accidental repr in a log or a traceback is one
        # of the more common ways a key escapes.
        state = "set" if self._api_key else "unset"
        return f"OpenRouterClient(models={self.models!r}, api_key=<{state}>)"

    __str__ = __repr__

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- the call ----------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        models: Sequence[str] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        json_mode: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send a chat completion, walking the model chain until one answers.

        Raises ``AuthError`` immediately on a bad key (retrying a 401 is how you
        get rate limited) and ``AllModelsFailed`` when the chain is exhausted.
        """
        # None means "use the configured chain"; an explicitly empty sequence is
        # a caller bug, not a request to fall back to the default.
        chain = self.models if models is None else tuple(models)
        if not chain:
            raise ValueError("model chain is empty")
        headers = self._auth_headers()
        failures: list[str] = []
        attempts = 0
        started = time.monotonic()

        for model in chain:
            payload: dict[str, Any] = {
                "model": model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                # Advisory: several free models ignore it, which is why the
                # parser in llm.py is defensive rather than trusting.
                payload["response_format"] = {"type": "json_object"}
            if extra_body:
                payload.update(extra_body)

            for attempt in range(self.max_retries + 1):
                attempts += 1
                try:
                    response = await self._client.post(
                        "/chat/completions", json=payload, headers=headers
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    failures.append(f"{model}: {type(exc).__name__}")
                    logger.warning(
                        "openrouter transport failure model=%s attempt=%d: %s",
                        model,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    if attempt < self.max_retries:
                        await self._backoff(attempt)
                        continue
                    break

                status = response.status_code
                if status in (401, 403):
                    raise AuthError(
                        f"OpenRouter rejected the key (HTTP {status}). Check {API_KEY_ENV}."
                    )
                if status in RETRYABLE_STATUS:
                    failures.append(f"{model}: HTTP {status}")
                    logger.warning(
                        "openrouter retryable status=%d model=%s attempt=%d",
                        status,
                        model,
                        attempt + 1,
                    )
                    if attempt < self.max_retries:
                        await self._backoff(attempt, retry_after=_retry_after(response))
                        continue
                    break
                if status in NEXT_MODEL_STATUS or status >= 400:
                    failures.append(f"{model}: HTTP {status}")
                    logger.warning("openrouter dropping model=%s status=%d", model, status)
                    break

                text, finish, usage = _extract_completion(response)
                if text is None:
                    failures.append(f"{model}: malformed response body")
                    logger.warning("openrouter malformed body from model=%s", model)
                    break
                latency_ms = (time.monotonic() - started) * 1000.0
                logger.info(
                    "openrouter ok model=%s attempts=%d latency=%.0fms tokens=%s",
                    model,
                    attempts,
                    latency_ms,
                    usage.get("total_tokens", "?"),
                )
                return ChatResponse(
                    text=text,
                    model=model,
                    attempts=attempts,
                    latency_ms=latency_ms,
                    usage=usage,
                    finish_reason=finish,
                )

        raise AllModelsFailed(failures)

    async def _backoff(self, attempt: int, *, retry_after: float | None = None) -> None:
        if retry_after is not None:
            delay = min(retry_after, self.backoff_cap_s)
        else:
            # Full jitter: uniform in [0, base * 2^attempt]. Correlated retries
            # from many clients are what turn a blip into an outage.
            ceiling = min(self.backoff_base_s * (2**attempt), self.backoff_cap_s)
            delay = self._rng.uniform(0.0, ceiling)
        await self._sleep(delay)

    # -- roster ------------------------------------------------------------

    async def list_models(self, *, free_only: bool = True) -> list[dict[str, Any]]:
        """GET /models. The free roster churns; this is how you re-check it."""
        response = await self._client.get("/models", headers=self._auth_headers())
        response.raise_for_status()
        data = response.json().get("data", [])
        if free_only:
            data = [m for m in data if str(m.get("id", "")).endswith(":free")]
        return sorted(data, key=lambda m: str(m.get("id", "")))


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    with contextlib.suppress(ValueError):
        return max(0.0, float(raw))
    return None


def _extract_completion(response: httpx.Response) -> tuple[str | None, str, dict[str, int]]:
    """Pull text out of a completion, tolerating the shapes free models emit."""
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return None, "", {}
    if not isinstance(data, dict):
        return None, "", {}
    # OpenRouter sometimes returns an error object with HTTP 200.
    if "error" in data and "choices" not in data:
        logger.warning("openrouter returned an error object with HTTP 200")
        return None, "", {}
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "", {}
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Multimodal-style content parts; concatenate the text ones.
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not isinstance(content, str):
        return None, "", {}
    usage_raw = data.get("usage") or {}
    usage = {k: int(v) for k, v in usage_raw.items() if isinstance(v, (int, float))}
    return content, str(choices[0].get("finish_reason") or ""), usage


__all__ = [
    "API_KEY_ENV",
    "BASE_URL",
    "DEFAULT_REASONING_MODELS",
    "DEFAULT_TEXT_MODELS",
    "DEFAULT_VISION_MODELS",
    "NEXT_MODEL_STATUS",
    "RETRYABLE_STATUS",
    "AllModelsFailed",
    "AuthError",
    "ChatResponse",
    "OpenRouterClient",
    "OpenRouterError",
    "build_image_data_url",
    "user_message",
]
