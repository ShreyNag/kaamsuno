"""Groq chat completions on the free tier. The limits are read from Groq's own headers, never assumed.

Every response carries x-ratelimit-{limit,remaining,reset}-{requests,tokens}; a 429 adds retry-after.
On our key (2026-10-31, gpt-oss-20b) "requests" is per day (1000) and "tokens" per minute (8000), and
a call costs ~2,000 tokens (~1,600 prompt, most of it cached on repeats). See notes/decisions.md.

`GroqPacer` keeps the last headers and says how long to wait before the next call, so a session
waits for the token bucket instead of hitting 429s. Every limit event goes to the hop log.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger

from app import config

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration(text: str | None) -> float | None:
    """Groq's reset/retry values in seconds: "25m55.199s", "15.4s", "120ms", "7" -> 1555.199, 15.4, 0.12, 7."""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = _DURATION.findall(text)
    if not parts:
        return None
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * scale[u] for n, u in parts)


def _int(x: str | None) -> int | None:
    try:
        return int(float(x)) if x is not None else None
    except ValueError:
        return None


def parse_limits(headers: Mapping[str, str]) -> dict[str, Any]:
    """The rate-limit headers of one response; absent ones are None."""
    h = {k.lower(): v for k, v in headers.items()}
    return {
        "limit_requests": _int(h.get("x-ratelimit-limit-requests")),
        "remaining_requests": _int(h.get("x-ratelimit-remaining-requests")),
        "reset_requests_s": parse_duration(h.get("x-ratelimit-reset-requests")),
        "limit_tokens": _int(h.get("x-ratelimit-limit-tokens")),
        "remaining_tokens": _int(h.get("x-ratelimit-remaining-tokens")),
        "reset_tokens_s": parse_duration(h.get("x-ratelimit-reset-tokens")),
        "retry_after_s": parse_duration(h.get("retry-after")),
    }


class RateLimited(Exception):
    """Groq said 429. The interpreter keeps the segments and retries after `retry_after_s`."""

    def __init__(self, retry_after_s: float, detail: str = ""):
        super().__init__(f"429, retry after {retry_after_s:.1f} s: {detail}")
        self.retry_after_s = retry_after_s


class GroqPacer:
    """How long to wait before the next call, from the last response's headers:
    - after a 429: its retry-after (or LLM_429_DEFAULT_WAIT_S when it has none)
    - requests left today = 0: until that bucket resets
    - tokens left this minute < the last call's tokens x LLM_TOKEN_MARGIN: until the bucket has refilled enough
    """

    def __init__(self, log: Callable[..., None] | None = None, clock: Callable[[], float] = time.monotonic):
        self._log = log or (lambda event, **kw: None)
        self._clock = clock
        self.last: dict[str, Any] = {}  # parse_limits() of the last response
        self._at = 0.0  # when `last` was seen
        self.call_tokens: int | None = None  # total tokens of the last call: the estimate for the next one
        self._backoff_until = 0.0
        self.n_429 = 0

    def observe(self, headers: Mapping[str, str], usage: Any = None) -> None:
        self.last, self._at = parse_limits(headers), self._clock()
        u = {}
        if usage is not None:
            self.call_tokens = getattr(usage, "total_tokens", None) or self.call_tokens
            details = getattr(usage, "prompt_tokens_details", None)
            u = {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                 "cached_tokens": getattr(details, "cached_tokens", None) if details else None,
                 "completion_tokens": getattr(usage, "completion_tokens", None)}
        self._log("groq_limits", **{k: v for k, v in self.last.items() if k != "retry_after_s"}, **u)

    def observe_429(self, headers: Mapping[str, str], detail: str = "") -> float:
        lim = parse_limits(headers)
        wait = lim["retry_after_s"]
        if wait is None:  # Groq's body says "Please try again in 7.66s"
            m = re.search(r"try again in ([\dhms.]+)", detail)
            wait = parse_duration(m[1]) if m else None
        wait = wait if wait is not None else config.LLM_429_DEFAULT_WAIT_S
        self._backoff_until = max(self._backoff_until, self._clock() + wait)
        self.n_429 += 1
        if any(v is not None for k, v in lim.items() if k != "retry_after_s"):
            self.last, self._at = lim, self._clock()
        self._log("groq_429", retry_after_s=round(wait, 2), n=self.n_429, detail=detail[:200],
                  **{k: v for k, v in lim.items() if k != "retry_after_s"})
        logger.warning(f"Groq 429 #{self.n_429}: waiting {wait:.1f} s ({detail[:120]})")
        return wait

    def wait_s(self) -> tuple[float, str | None]:
        """(seconds to wait before the next call, which limit) ; (0, None) when a call may go now."""
        now = self._clock()
        waits = [(self._backoff_until - now, "retry_after")]
        lim, age = self.last, now - self._at
        if lim.get("remaining_requests") == 0 and lim.get("reset_requests_s") is not None:
            waits.append((lim["reset_requests_s"] - age, "requests"))
        rem, cap = lim.get("remaining_tokens"), lim.get("limit_tokens")
        need = self.call_tokens * config.LLM_TOKEN_MARGIN if self.call_tokens else None
        if rem is not None and cap and need and rem < need:
            refill = cap / 60.0  # the bucket refills linearly over a minute
            waits.append(((min(need, cap) - rem) / refill - age, "tokens"))
        wait, why = max(waits)
        return (wait, why) if wait > 0 else (0.0, None)


def groq_complete(model: str = config.LLM_MODEL, max_retries: int = 0, pacer: GroqPacer | None = None):
    """JSON-mode chat completion against Groq's OpenAI-compatible endpoint.

    max_retries=0: the SDK's own retries would hide 429s and hold the session's single call slot;
    the caller backs off instead (`RateLimited`). Headers and usage go to `pacer`."""
    import openai
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL, timeout=config.LLM_TIMEOUT_S,
                         max_retries=max_retries)

    async def complete(messages: list[dict[str, str]]) -> str:
        try:
            raw = await client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                reasoning_effort=config.LLM_REASONING_EFFORT,
            )
        except openai.RateLimitError as e:
            headers = e.response.headers if e.response is not None else {}
            wait = pacer.observe_429(headers, str(e)) if pacer else (
                parse_limits(headers)["retry_after_s"] or config.LLM_429_DEFAULT_WAIT_S)
            raise RateLimited(wait, str(e)[:160]) from e
        r = raw.parse()
        if pacer:
            pacer.observe(raw.headers, r.usage)
        return r.choices[0].message.content or "{}"

    return complete
