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
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from loguru import logger

from app import config
from app.interpreter.schema import Schema

PROMPT_PATH = config.RESEARCH_DIR / "interpreter_prompt_v0.md"
PROMPT_V1_PATH = config.RESEARCH_DIR / "interpreter_prompt_v1.md"  # Aastha's, used when present

Report = dict[str, dict[str, Any]]
Complete = Callable[[list[dict[str, str]]], Awaitable[str]]  # chat messages -> JSON text


def empty_report(schema: Schema) -> Report:
    """Every schema field, empty; the system fields we know at session start are filled."""
    r = {n: {"value": None, "confidence": 0.0, "evidence": None, "history": [], "conflicts": []} for n in schema.names}
    known = {"site_name": config.SITE_NAME, "status": "draft", "reporter_confirmation": "not_obtained"}
    for name, value in known.items():
        if name in r:
            r[name].update(value=value, confidence=1.0, evidence="system")
    return r


def _norm(text: str) -> str:
    """Lowercase; punctuation and symbols to spaces. Not a regex word class: Python's "word
    characters" exclude Devanagari vowel signs (matras), which turned "मंजिल" into "म ज ल"."""
    t = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in text.lower())
    return " ".join(t.split())


def _confidence(x: Any) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return 0.5


def _enum_value(value: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().lower())



def merge_update(report: Report, updates: dict[str, Any], transcript: str, schema: Schema) -> Report:
    """Apply LLM `updates` ({field: {value, confidence, evidence, correction?}}) to a copy of `report`.
    The rules live here, not in the prompt, so they hold whatever the LLM says:

    - unknown fields, system fields, null values and values outside a field's `allowed` are ignored
    - evidence must occur in the transcript, else the update is dropped (no invented values)
    - a location's evidence must contain a place word (config.LOCATION_CUES): "crane pipe" is not a location
    - urgency is never downgraded from "emergency"
    - CONFLICT_FIELDS (location, quantity, unit, asset...): a different value replaces the old one
      (old -> `history`) only if the LLM marks it a correction or it refines the old value
      ("tower B" -> "tower B 3rd floor"); otherwise both are kept: the new one goes to `conflicts`
      and the dialogue asks which is right. Re-stating the current value clears its conflicts.
    - other fields: a different value replaces the old one (old -> `history`)
    - a quantity whose evidence isn't just the digits keeps the spoken phrase in spoken_number_raw
    """
    out = copy.deepcopy(report)
    said = f" {_norm(transcript)} "
    for name, upd in (updates or {}).items():
        if name not in out or name in config.SYSTEM_FIELDS or not isinstance(upd, dict):
            continue
        value, evidence = upd.get("value"), str(upd.get("evidence") or "")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not _norm(evidence) or f" {_norm(evidence)} " not in said:
            logger.debug(f"interpreter: dropped {name}={value!r}, evidence {evidence!r} not in transcript")
            continue
        if name == "location" and not set(_norm(evidence).split()) & config.LOCATION_CUES:
            logger.info(f"interpreter: dropped location={value!r}, evidence {evidence!r} names no place")
            continue
        allowed = schema.allowed(name)
        if allowed:
            value = _enum_value(value)
            if value not in allowed:
                logger.debug(f"interpreter: dropped {name}={value!r}, not in {allowed}")
                continue
        cur = out[name]
        if name == "urgency" and cur["value"] == "emergency" and value != "emergency":
            logger.info(f"interpreter: kept urgency=emergency, refused downgrade to {value!r}")
            continue
        new = {"value": value, "confidence": _confidence(upd.get("confidence")), "evidence": evidence}
        old_v = cur["value"]
        if old_v is None or _norm(str(old_v)) == _norm(str(value)):
            cur.update(new)
            cur["conflicts"] = []
        elif name in config.CONFLICT_FIELDS and not upd.get("correction") and _norm(str(old_v)) not in _norm(str(value)):
            if all(_norm(str(c["value"])) != _norm(str(value)) for c in cur["conflicts"]):
                cur["conflicts"].append(new)
            continue
        else:
            cur["history"].append({k: cur[k] for k in ("value", "confidence", "evidence")})
            cur.update(new)
            cur["conflicts"] = []
        if name == "quantity" and "spoken_number_raw" in out and _norm(evidence) != _norm(str(value)):
            out["spoken_number_raw"].update(value=evidence, confidence=1.0, evidence=evidence)
    return out


def values_only(report: Report) -> dict[str, Any]:
    return {k: v["value"] for k, v in report.items()}


def build_system_prompt(schema: Schema, path: Path | None = None) -> str:
    """The prompt template with the schema's LLM-fillable fields, allowed values and validation rules."""
    path = path or (PROMPT_V1_PATH if PROMPT_V1_PATH.exists() else PROMPT_PATH)
    lines = []
    for f in schema.llm_fields:
        allowed = f" Allowed: {', '.join(f.enum)}." if f.enum else ""
        lines.append(f"- {f.name}{' (required)' if f.critical else ''}: {f.description}{allowed}")
    rules = "\n".join(f"- {r}" for r in schema.rules) or "- Never invent values."
    return path.read_text(encoding="utf-8").format(
        fields="\n".join(lines), incident_types=", ".join(schema.incident_types), rules=rules
    )


def _default_uncertainty_cue(text: str) -> str | None:
    from app.turn.tail_classifier import load_fillers

    return load_fillers().find(text, "uncertainty")


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
        uncertainty_cue: Callable[[str], str | None] | None = None,
    ):
        self.schema = schema
        # "shayad", "pata nahi", "lagbhag"...: values from such a segment get confirmed (Aastha's fillers.json)
        self._uncertainty_cue = uncertainty_cue or _default_uncertainty_cue
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
        cue = self._uncertainty_cue(text) if updates else None
        if cue:
            cap = config.LOW_CONFIDENCE - 0.1
            for upd in updates.values():
                if isinstance(upd, dict):
                    upd["confidence"] = min(_confidence(upd.get("confidence")), cap)
            self._log("interp_uncertain", seq=seq, cue=cue, fields=sorted(updates))
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
