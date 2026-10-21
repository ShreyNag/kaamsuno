"""Live interpreter: each final Prisma segment + the current report -> field updates via the LLM.

Calls run concurrently as background tasks (a slow LLM never blocks audio), and
their results are merged strictly in segment order. The merge is a pure
function (`merge_update`) that enforces the report rules, whatever the LLM says.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from loguru import logger

from app import config
from app.interpreter.schema import Schema

PROMPT_PATH = config.RESEARCH_DIR / "interpreter_prompt_v0.md"

Report = dict[str, dict[str, Any]]
Complete = Callable[[list[dict[str, str]]], Awaitable[str]]  # chat messages -> JSON text


def empty_report(schema: Schema) -> Report:
    return {n: {"value": None, "confidence": 0.0, "evidence": None, "history": []} for n in schema.names}


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def _confidence(x: Any) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return 0.5


def merge_update(report: Report, updates: dict[str, Any], transcript: str, schema: Schema) -> Report:
    """Apply LLM `updates` ({field: {value, confidence, evidence}}) to a copy of `report`.

    - unknown fields, null values and values outside a field's enum are ignored (null stays null)
    - evidence must occur in the transcript, else the update is dropped (no invented values)
    - a different value for a filled field is a correction: the old entry moves to `history`
    """
    out = copy.deepcopy(report)
    said = f" {_norm(transcript)} "
    for name, upd in (updates or {}).items():
        if name not in out or not isinstance(upd, dict):
            continue
        value, evidence = upd.get("value"), str(upd.get("evidence") or "")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not _norm(evidence) or f" {_norm(evidence)} " not in said:
            logger.debug(f"interpreter: dropped {name}={value!r}, evidence {evidence!r} not in transcript")
            continue
        allowed = schema.allowed(name)
        if allowed:
            value = str(value).strip().lower()
            if value not in allowed:
                logger.debug(f"interpreter: dropped {name}={value!r}, not in {allowed}")
                continue
        cur = out[name]
        if cur["value"] is not None and str(cur["value"]).lower() != str(value).lower():
            cur["history"].append({k: cur[k] for k in ("value", "confidence", "evidence")})
        cur.update(value=value, confidence=_confidence(upd.get("confidence")), evidence=evidence)
    return out


def values_only(report: Report) -> dict[str, Any]:
    return {k: v["value"] for k, v in report.items()}


def build_system_prompt(schema: Schema, path: Path = PROMPT_PATH) -> str:
    fields = "\n".join(
        f"- {f.name}{' (critical)' if f.critical else ''}: {f.description}" for f in schema.fields
    )
    return path.read_text(encoding="utf-8").format(fields=fields, incident_types=", ".join(schema.incident_types))


def groq_complete(model: str = config.LLM_MODEL) -> Complete:
    """JSON-mode chat completion against Groq's OpenAI-compatible endpoint."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL, timeout=config.LLM_TIMEOUT_S)

    async def complete(messages: list[dict[str, str]]) -> str:
        r = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            reasoning_effort=config.LLM_REASONING_EFFORT,
        )
        return r.choices[0].message.content or "{}"

    return complete


class Interpreter:
    def __init__(
        self,
        schema: Schema,
        complete: Complete,
        *,
        system_prompt: str | None = None,
        on_update: Callable[[Report], Awaitable[None]] | None = None,
        log: Callable[..., None] | None = None,
    ):
        self.schema = schema
        self.report = empty_report(schema)
        self.segments: list[str] = []
        self._complete = complete
        self._system = system_prompt if system_prompt is not None else build_system_prompt(schema)
        self.on_update = on_update
        self._log = log or (lambda event, **kw: None)
        self._results: dict[int, dict[str, Any]] = {}
        self._next_commit = 0
        self._tasks: set[asyncio.Task] = set()

    def submit(self, text: str) -> int:
        """Queue one final segment; returns its sequence number. Never blocks."""
        seq = len(self.segments)
        self.segments.append(text)
        task = asyncio.create_task(self._run(seq, values_only(self.report), self.segments[:seq], text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return seq

    async def _run(self, seq: int, current: dict[str, Any], earlier: list[str], text: str) -> None:
        user = json.dumps({"current_report": current, "earlier_segments": earlier, "new_segment": text},
                          ensure_ascii=False)
        self._log("interp_request", seq=seq)
        t0 = time.time()
        try:
            raw = await self._complete([{"role": "system", "content": self._system}, {"role": "user", "content": user}])
            updates = json.loads(raw).get("updates", {})
            if not isinstance(updates, dict):
                raise ValueError(f"'updates' is {type(updates).__name__}, not an object")
        except Exception as e:  # a failed call must not stall later segments
            logger.error(f"interpreter call {seq} failed: {e}")
            updates = {}
        self._log("interp_response", seq=seq, ms=round((time.time() - t0) * 1000), fields=sorted(updates))
        self._results[seq] = updates
        await self._commit_in_order()

    async def _commit_in_order(self) -> None:
        changed = False
        while self._next_commit in self._results:
            updates = self._results.pop(self._next_commit)
            transcript = " ".join(self.segments[: self._next_commit + 1])
            self.report = merge_update(self.report, updates, transcript, self.schema)
            self._next_commit += 1
            changed = True
        if changed and self.on_update:
            await self.on_update(self.report)

    async def drain(self, timeout: float = config.INTERPRETER_DRAIN_TIMEOUT_S) -> bool:
        """Wait for in-flight calls; False if some were still running at the timeout."""
        if not self._tasks:
            return True
        _, pending = await asyncio.wait(set(self._tasks), timeout=timeout)
        return not pending
