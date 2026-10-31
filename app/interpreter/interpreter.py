"""Live interpreter: final Prisma segments + the current report -> field updates via the LLM.

Calls run as a background task (a slow LLM never blocks audio), one at a time per session, so results
merge in segment order. The merge is a pure function (`merge_update`) that enforces the report rules,
whatever the LLM says.
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
from app.interpreter.groq import RateLimited, groq_complete  # noqa: F401  (groq_complete re-exported)
from app.interpreter.schema import Schema

PROMPT_PATH = config.RESEARCH_DIR / "interpreter_prompt_v0.md"
PROMPT_V1_PATH = config.RESEARCH_DIR / "interpreter_prompt_v1.md"  # Aastha's, used when present

Report = dict[str, dict[str, Any]]
Complete = Callable[[list[dict[str, str]]], Awaitable[str]]  # chat messages -> JSON text


def empty_report(schema: Schema) -> Report:
    """Every schema field, empty; the system fields we know at session start are filled."""
    r = {n: {"value": None, "confidence": 0.0, "evidence": None, "history": [], "conflicts": [], "inferred": False}
         for n in schema.names}
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


# A value made only of these is a label the worker said ("employee ID ..."), not the value itself
LABEL_WORDS = frozenset("id employee number no num naam name machine asset code worker mera meri".split())


def is_known(entry: dict) -> bool:
    """A field counts as filled only with a real, stated value: not null/empty, not "unknown", not inferred."""
    v = entry.get("value")
    if v is None or (isinstance(v, str) and not v.strip()) or entry.get("inferred"):
        return False
    return not (isinstance(v, str) and v.strip().lower() in config.UNKNOWN_VALUES)


def _inferred(name: str, upd: dict, evidence: str, asked_field: str | None = None) -> bool:
    """The LLM says it guessed ("stated": false), or urgency whose evidence has no urgency word
    (unless it answers "how urgent is it?": then any answer is stated)."""
    if upd.get("stated") is False:
        return True
    return name == "urgency" and asked_field != "urgency" and not set(_norm(evidence).split()) & config.URGENCY_CUES


def merge_update(report: Report, updates: dict[str, Any], transcript: str, schema: Schema,
                 assume_correction: bool = False, asked_field: str | None = None) -> Report:
    """Apply LLM `updates` ({field: {value, confidence, evidence, correction?, stated?}}) to a copy of `report`.
    The rules live here, not in the prompt, so they hold whatever the LLM says:

    - unknown fields, system fields, null values and values outside a field's `allowed` are ignored
    - evidence must occur in the transcript, else the update is dropped (no invented values)
    - a location's evidence must contain a place word (config.LOCATION_CUES): "crane pipe" is not a location,
      unless the bot just asked for the location (`asked_field`): then "bhed do mein" is the answer
    - urgency is never downgraded from "emergency"
    - CONFLICT_FIELDS (location, quantity, unit, asset...): a different value replaces the old one
      (old -> `history`) only if the LLM marks it a correction or it refines the old value
      ("tower B" -> "tower B 3rd floor"); otherwise both are kept: the new one goes to `conflicts`
      and the dialogue asks which is right. Re-stating the current value clears its conflicts.
    - other fields: a different value replaces the old one (old -> `history`)
    - a quantity whose evidence isn't just the digits keeps the spoken phrase in spoken_number_raw
    - a value that is only a label ("employee id") is dropped
    - `inferred` marks a value the worker didn't state (the LLM's "stated": false, or urgency without an
      urgency word in its evidence); it is kept and shown, but the dialogue still asks for it
    - assume_correction (during the read-back): any different value is the worker correcting it
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
        if isinstance(value, str) and set(_norm(value).split()) <= LABEL_WORDS:
            logger.info(f"interpreter: dropped {name}={value!r}, a label, not a value")
            continue
        if name == "location" and asked_field != "location" and not set(_norm(evidence).split()) & config.LOCATION_CUES:
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
        new = {"value": value, "confidence": _confidence(upd.get("confidence")), "evidence": evidence,
               "inferred": _inferred(name, upd, evidence, asked_field)}
        if new["inferred"] and is_known(cur):
            continue  # a guess never overwrites a stated value
        old_v = cur["value"]
        correction = bool(upd.get("correction")) or assume_correction
        if old_v is None or _norm(str(old_v)) == _norm(str(value)):
            cur.update(new)
            cur["conflicts"] = []
        elif name in config.CONFLICT_FIELDS and not correction and _norm(str(old_v)) not in _norm(str(value)):
            if all(_norm(str(c["value"])) != _norm(str(value)) for c in cur["conflicts"]):
                cur["conflicts"].append(new)
            continue
        else:
            cur["history"].append({k: cur.get(k) for k in ("value", "confidence", "evidence")})
            cur.update(new)
            cur["conflicts"] = []
        if name == "quantity" and "spoken_number_raw" in out and _norm(evidence) != _norm(str(value)):
            out["spoken_number_raw"].update(value=evidence, confidence=1.0, evidence=evidence)
    return out


def values_only(report: Report) -> dict[str, Any]:
    return {k: v["value"] for k, v in report.items()}


def stated_values(report: Report) -> dict[str, Any]:
    """What the LLM sees as the current report: guessed or "unknown" values are null. (Live test 3:
    a guessed urgency "high" shown as set made the LLM ignore "abhi turant" and "urgency urgent hai".)"""
    return {k: v["value"] if is_known(v) else None for k, v in report.items()}


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


class Interpreter:
    """At most one LLM call in flight per session (Groq free tier: tokens per minute are the limit).

    Segments that arrive while a call runs wait and go out together in the next call, at least
    LLM_MIN_GAP_S after the previous one ended. drain() at turn end sends what is left at once (no gap)
    and waits up to its timeout; after that the dialogue asks from the report as it is, and the late
    result still merges. A call sends only the report's stated values, the bot's question and the new
    text, never the whole transcript. A 429 keeps the segments and retries after Groq's retry-after;
    `pace` (GroqPacer.wait_s) holds a call back until the token bucket can take it.
    """

    def __init__(
        self,
        schema: Schema,
        complete: Complete,
        *,
        system_prompt: str | None = None,
        on_update: Callable[[Report], Awaitable[None]] | None = None,
        log: Callable[..., None] | None = None,
        uncertainty_cue: Callable[[str], str | None] | None = None,
        min_gap_s: float = config.LLM_MIN_GAP_S,
        pace: Callable[[], tuple[float, str | None]] | None = None,
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
        self.assume_correction = False  # set by the dialogue during the read-back
        # The bot's last move, {field, reason, question}: a short answer ("dedh sau bori") fills that field
        self.asked: dict[str, Any] | None = None
        self.touched: set[str] = set()  # fields updated (even to the same value) since take_touched()
        self._asked_for: dict[int, dict[str, Any] | None] = {}  # seq -> the bot's move when it was said
        self._pending: list[int] = []  # segments not yet sent, oldest first
        self._min_gap_s = min_gap_s
        self._pace = pace or (lambda: (0.0, None))
        self._last_end = float("-inf")  # monotonic end of the last call
        self._flush = False  # turn end: skip the min gap
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self.n_calls = 0

    def submit(self, text: str) -> int:
        """Queue one final segment; returns its sequence number. Never blocks."""
        seq = len(self.segments)
        self.segments.append(text)
        self._asked_for[seq] = self.asked
        self._pending.append(seq)
        self._wake.set()
        if not self.busy:
            self._worker = asyncio.create_task(self._loop())
        return seq

    async def _loop(self) -> None:
        while self._pending:
            gap = 0.0 if self._flush else self._last_end + self._min_gap_s - time.monotonic()
            limit_wait, why = self._pace()
            wait = max(gap, limit_wait)
            if wait > 0:
                if limit_wait > 0 and limit_wait >= gap:
                    self._log("interp_wait", seconds=round(limit_wait, 2), limit=why, pending=len(self._pending))
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=wait)
                except TimeoutError:
                    pass
                continue
            await self._call(self._take_batch())

    def _take_batch(self) -> list[int]:
        """The pending segments said after the same bot question (a short answer is read against it)."""
        first = self._asked_for[self._pending[0]]
        n = 1
        while n < len(self._pending) and self._asked_for[self._pending[n]] is first:
            n += 1
        batch, self._pending = self._pending[:n], self._pending[n:]
        return batch

    async def _call(self, batch: list[int]) -> None:
        text = " ".join(self.segments[i] for i in batch)
        asked = self._asked_for[batch[0]]
        current = {k: v for k, v in stated_values(self.report).items()
                   if v is not None and k not in config.SYSTEM_FIELDS}
        user = json.dumps({"current_report": current, "bot_just_asked": asked, "new_segment": text},
                          ensure_ascii=False)
        self._log("interp_request", seq=batch[-1], segments=len(batch), chars=len(user))
        self.n_calls += 1
        t0, error = time.time(), None
        try:
            raw = await self._complete([{"role": "system", "content": self._system}, {"role": "user", "content": user}])
            updates = json.loads(raw).get("updates", {})
            if not isinstance(updates, dict):
                raise ValueError(f"'updates' is {type(updates).__name__}, not an object")
        except RateLimited as e:  # keep the segments: the next call (after retry-after) includes them
            self._pending[:0] = batch
            self._last_end = time.monotonic()
            self._log("interp_response", seq=batch[-1], ms=round((time.time() - t0) * 1000), fields=[],
                      error="rate_limited", retry_after_s=round(e.retry_after_s, 2))
            return
        except Exception as e:  # a failed call must not stall later segments
            logger.error(f"interpreter call for segments {batch} failed: {e}")
            updates, error = {}, str(e)[:160]
        self._last_end = time.monotonic()
        self._log("interp_response", seq=batch[-1], ms=round((time.time() - t0) * 1000), fields=sorted(updates),
                  **({"error": error} if error else {}))
        cue = self._uncertainty_cue(text) if updates else None
        if cue:
            cap = config.LOW_CONFIDENCE - 0.1
            for upd in updates.values():
                if isinstance(upd, dict):
                    upd["confidence"] = min(_confidence(upd.get("confidence")), cap)
            self._log("interp_uncertain", seq=batch[-1], cue=cue, fields=sorted(updates))
        transcript = " ".join(self.segments[: batch[-1] + 1])
        before = self.report
        self.report = merge_update(self.report, updates, transcript, self.schema, self.assume_correction,
                                   asked_field=asked and asked.get("field"))
        self.touched |= {n for n in updates if n in before and self.report[n] != before[n]}
        if self.on_update:
            await self.on_update(self.report)

    def take_touched(self) -> set[str]:
        touched, self.touched = self.touched, set()
        return touched

    @property
    def pending(self) -> int:
        """Segments not yet sent to the LLM."""
        return len(self._pending)

    @property
    def busy(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def drain(self, timeout: float | None = config.INTERPRETER_DRAIN_TIMEOUT_S) -> bool:
        """Turn end: send what is pending now (no min gap) and wait for it. False if still running at the
        timeout (rate-limit wait or slow call); the caller then asks from the report as it is."""
        if not self.busy:
            return True
        self._flush = True
        self._wake.set()
        try:
            done, _ = await asyncio.wait({self._worker}, timeout=timeout)
            return bool(done)
        finally:
            self._flush = False
