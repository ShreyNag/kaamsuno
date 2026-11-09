"""Pipeline glue for the reporter: segments -> interpreter (silent), turn end -> one dialogue move.

Sits after the UserTurnProcessor. It never generates free text: the only thing
it ever sends to TTS is a template line chosen by dialogue.DialogueManager (or the greeting),
and some turns it says nothing at all.

With `reports_dir`, every update also rewrites <reports_dir>/<incident_id>.json:
each field's value, confidence, evidence and history, plus the full transcript. The first report of a
session has incident_id = session_id; a second issue in the same call gets <session_id>-2, and so on.

Sending (`send_report(values, kind)`, e.g. Telegram) happens when the worker answers "aur kuch?" with
no, or when they start a new issue (the confirmed report goes first).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app import config
from app.interpreter.dialogue import DialogueManager, Question
from app.interpreter.interpreter import Interpreter, Report, empty_report, is_known
from app.interpreter.schema import Schema
from app.interpreter.templates import TemplateBook, detect_language
from app.logging_utils import HopLogger
from app.report_server import HUB, ReportHub
from app.stt import PrismaSTTService
from app.turn.tail_classifier import DialogueContext


class ReportProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        session_id: str,
        schema: Schema,
        templates: dict[str, str] | TemplateBook,
        interpreter: Interpreter,
        stt: PrismaSTTService | None,
        hops: HopLogger,
        hub: ReportHub = HUB,
        set_dialogue_context: Callable[[DialogueContext], None] | None = None,
        reports_dir: Path | None = None,
        send_report: Callable[[dict, str], Awaitable[bool]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sid, self._schema, self._templates = session_id, schema, templates
        self._interp, self._stt, self._hops, self._hub = interpreter, stt, hops, hub
        self._set_context = set_dialogue_context  # tells the turn controller what we just asked
        self._reports_dir = reports_dir
        self._send_report = send_report  # (report values, "report" | "emergency") -> sent?
        self._report_n = 1
        self.incident_id = session_id
        self._seg_offset = 0  # this report's segments start here (a 2nd issue starts a new report)
        self._confirmed_report: Report | None = None  # as the worker approved it ("haan" to the read-back)
        self.sent: list[tuple[str, str, bool]] = []  # (incident_id, kind, ok)
        self._turn_first_segment = 0
        self._last_speech_end = 0.0
        self._question: Question | None = None
        self._status = "waiting for the worker"
        self._turn_task: asyncio.Task | None = None
        interpreter.on_update = self._on_report_update
        self._reset_report(fresh=False)
        self.publish()

    @property
    def _report_path(self) -> Path | None:
        return self._reports_dir / f"{self.incident_id}.json" if self._reports_dir else None

    def _reset_report(self, fresh: bool = True) -> None:
        """Start a report: the dialogue, the questions log and (if `fresh`) the interpreter's report."""
        if fresh:
            self._report_n += 1
            self.incident_id = f"{self._sid}-{self._report_n}"
            self._interp.report = empty_report(self._schema)
            self._interp.asked, self._interp.assume_correction = None, False
            self._interp.take_touched()
            self._seg_offset = len(self._interp.segments)
        self.dialogue = DialogueManager()
        self._alert_task: asyncio.Task | None = None  # this report's emergency alert, sent once
        self._questions: list[dict] = []
        self._known_at_last_turn: dict[str, object] = {}  # stated values when the bot last spoke
        self._confirmed_report = None
        for name, value in (("incident_id", self.incident_id), ("reported_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))):
            if name in self._interp.report:
                self._interp.report[name].update(value=value, confidence=1.0, evidence="system")

    def report_values(self) -> dict[str, object]:
        """Every filled field's value (system fields included): what goes to the supervisor."""
        return {k: v["value"] for k, v in self.report_with_system_fields().items() if v["value"] is not None}

    def report_with_system_fields(self) -> Report:
        """The interpreter's report plus the fields only code fills: transcript, missing_fields,
        uncertainties (asked twice and still not answered; low confidence), conflicts."""
        r = {k: dict(v) for k, v in self._interp.report.items()}
        unanswered = [{"field": n, "reason": "not_answered"} for n in sorted(self.dialogue.unconfirmed)
                      if n in r and not is_known(r[n])]
        doubtful = [{"field": n, "reason": "low_confidence", "value": r[n]["value"], "confidence": r[n]["confidence"]}
                    for n in self._schema.critical_priority if is_known(r[n]) and r[n]["confidence"] < config.LOW_CONFIDENCE]
        derived = {
            "transcript": " ".join(self._interp.segments[self._seg_offset:]) or None,
            "missing_fields": [n for n in self._schema.critical_priority if not is_known(r[n])] or None,
            "uncertainties": unanswered + doubtful or None,
            "conflicts": [{"field": n, "values": [r[n]["value"], *(c["value"] for c in r[n]["conflicts"])]}
                          for n in r if r[n].get("conflicts")] or None,
        }
        for name, value in derived.items():
            if name in r:
                r[name].update(value=value, confidence=1.0 if value is not None else 0.0,
                               evidence="system" if value is not None else None)
        return r

    def snapshot(self) -> dict:
        r = self.report_with_system_fields()
        return {
            "session": self._sid,
            "incident_id": self.incident_id,
            "status": self._status,
            "low_confidence": config.LOW_CONFIDENCE,
            "question": self._question and {"field": self._question.field, "reason": self._question.reason,
                                            "text": self._question.text},
            "fields": [{"name": f.name, "critical": f.critical, **r[f.name]} for f in self._schema.fields],
            "segments": list(self._interp.segments[self._seg_offset:]),
        }

    def publish(self) -> None:
        snap = self.snapshot()
        self._hub.publish(self._sid, snap)
        if self._report_path:
            self._save(snap)

    def _save(self, snap: dict) -> None:
        """Write the whole report atomically (temp file + replace), so a reader never sees half a file."""
        doc = {
            "session": self._sid,
            "incident_id": self.incident_id,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": snap["status"],
            "report": {f["name"]: {k: f.get(k) for k in ("critical", "value", "confidence", "evidence", "history",
                                                         "conflicts")}
                       for f in snap["fields"]},
            "transcript": snap["segments"],
            "questions": self._questions,
        }
        try:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._report_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self._report_path)
        except OSError as e:  # a full disk must not take the call down
            logger.error(f"could not save {self._report_path}: {e}")

    async def _on_report_update(self, report: Report) -> None:
        self._hops.log("report_update", report={k: v["value"] for k, v in report.items() if v["value"] is not None})
        self._maybe_alert()  # the LLM flagged an emergency the trigger words missed
        self.publish()

    def _maybe_alert(self) -> None:
        """urgency=emergency (trigger words or the LLM): alert the supervisor now, once per report, without
        waiting for the turn to end. The safety line at turn end says whether it went."""
        if self._alert_task is None and self._interp.report.get("urgency", {}).get("value") == "emergency":
            self._alert_task = self.create_task(self.send("emergency"), "emergency_alert")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._interp.submit(frame.text.strip())  # update state only; never speak here
            self._maybe_alert()  # "sparks", "chot lagi": urgency was set in submit, before the LLM
            self.publish()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._last_speech_end = time.time() - frame.stop_secs
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._cancel_turn_task()  # the worker kept talking: drop the pending question
            self._status = "listening"
            self.publish()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._cancel_turn_task()
            self._turn_task = self.create_task(self._on_turn_end(), "reporter_turn_end")
        await self.push_frame(frame, direction)

    async def _cancel_turn_task(self) -> None:
        if self._turn_task:
            await self.cancel_task(self._turn_task)
            self._turn_task = None

    async def _on_turn_end(self) -> None:
        # UserStoppedSpeakingFrame is a SystemFrame, so it can overtake a TranscriptionFrame still
        # queued here. Waiting for Prisma to go idle (below) also lets such a segment be submitted.
        self._status = "turn ended: waiting for the last segment"
        self.publish()
        if self._stt:
            p = config.PRISMA
            not_before = self._last_speech_end + (p.min_silence_ms + p.tail_grace_ms) / 1000
            if not await self._stt.wait_idle(not_before, p.tail_timeout_ms / 1000):
                logger.warning("turn end: Prisma segment still pending at tail timeout; asking anyway")
        self._hops.log("stt_tail_done")
        if not await self._interp.drain():
            logger.warning("turn end: interpreter still busy at drain timeout; asking from the report as it is")
            self._hops.log("interp_late", pending=self._interp.pending)
        # reply in the worker's style: the language of what they said since the last question
        turn_segments = self._interp.segments[self._turn_first_segment:]
        said = " ".join(turn_segments)
        self._turn_first_segment = len(self._interp.segments)
        lang = detect_language(said) if said else "hinglish"
        templates = self._templates.for_language(lang) if isinstance(self._templates, TemplateBook) \
            else self._templates
        changed = self._changed_fields()
        await self._settle_alert()
        q = self.dialogue.decide(self._interp.report, self._schema, templates, said, changed,
                                 self._interp.take_touched())
        if q.reason == "confirmed":  # what the worker approved; a new issue must not leak into it
            self._confirmed_report = copy.deepcopy(self._interp.report)
        elif q.reason == "submit":
            q = await self._submit(templates)
        elif q.reason == "new_report":
            q = await self._start_new_report(templates, turn_segments)
        self._question = q
        dm = self.dialogue
        if not q.silent:  # the next answer is read against this question ("dedh sau bori" -> quantity)
            self._interp.asked = {"field": q.field, "reason": q.reason, "question": q.text}
        # During the read-back a different value from the worker is a correction, never a conflict
        self._interp.assume_correction = dm.phase in ("readback", "confirmed")
        self._set_system("reporter_confirmation", dm.confirmation)
        self._set_system("status", {"collecting": "draft", "readback": "awaiting_confirmation",
                                    "confirmed": "awaiting_confirmation", "submitted": "submitted"}[dm.phase])
        self._hops.log("question", field=q.field, reason=q.reason, text=q.text, language=lang, phase=dm.phase,
                       changed=sorted(changed) or None, incident_id=self.incident_id)
        self._questions.append({"field": q.field, "reason": q.reason, "text": q.text})
        self._status = {"collecting": "bot asking", "readback": "read-back: waiting for yes/no",
                        "confirmed": "confirmed: asked 'aur kuch?'", "submitted": "sent to the supervisor"}[dm.phase]
        self.publish()
        if self._set_context:
            self._set_context(DialogueContext(q.field, q.expects_yes_no))
        if not q.silent:
            await self.push_frame(TTSSpeakFrame(q.text))
        self._turn_task = None

    async def send(self, kind: str = "report") -> bool:
        """Send the current report to the supervisor (`send_report`). Never raises; False if not sent."""
        ok, error = False, None
        if self._send_report is None:
            error = "no supervisor channel configured"
            logger.warning(f"report {self.incident_id} not sent: {error}")
        else:
            try:
                ok = bool(await asyncio.wait_for(self._send_report(self.report_values(), kind), timeout=10))
            except Exception as e:  # noqa: BLE001 - a failed send must not take the call down
                error = f"{type(e).__name__}: {e}"[:200]
                logger.error(f"report {self.incident_id} not sent: {error}")
        self.sent.append((self.incident_id, kind, ok))
        self._hops.log("report_sent", incident_id=self.incident_id, kind=kind, ok=ok, **({"error": error} if error else {}))
        return ok

    async def _settle_alert(self, timeout_s: float = 3.0) -> None:
        """Before the safety line: wait (briefly) for the emergency alert, so the line is true."""
        task = self._alert_task
        if task is None or self.dialogue.emergency_announced:
            return
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        self.dialogue.alert_sent = bool(done) and not task.cancelled() and task.result() is True

    async def _submit(self, t) -> Question:
        """"Aur kuch?" -> "nahi": send, then tell the worker whether it went."""
        self._set_system("status", "submitted")
        ok = await self.send("report")
        self.publish()
        return Question(None, "submit" if ok else "send_failed", t["_submitted"] if ok else t["_send_failed"])

    async def _start_new_report(self, t, turn_segments: list[str]) -> Question:
        """A second issue: send the confirmed report (unless already sent), then a new report is built
        from this turn's words and the dialogue asks its first question."""
        prefix = ""
        if self.dialogue.phase != "submitted":
            if self._confirmed_report is not None:  # the read-back's correction mode may have merged the
                self._interp.report = self._confirmed_report  # new issue into it: restore what was approved
            self._set_system("reporter_confirmation", self.dialogue.confirmation)
            self._set_system("status", "submitted")
            ok = await self.send("report")
            self.publish()
            prefix = t["_new_report"] if ok else t["_send_failed"]
        self._reset_report()
        self._hops.log("new_report", incident_id=self.incident_id)
        for text in turn_segments:  # read this turn's words again, now against the empty report
            self._interp.submit(text)
        self._turn_first_segment = len(self._interp.segments)
        self._maybe_alert()
        await self._interp.drain(timeout=2 * config.INTERPRETER_DRAIN_TIMEOUT_S)
        await self._settle_alert()
        q = self.dialogue.decide(self._interp.report, self._schema, t, " ".join(turn_segments),
                                 self._changed_fields(), self._interp.take_touched())
        text = f"{prefix} {q.text}".strip()
        return Question(q.field, "new_report" if q.silent and text else q.reason, text)

    def _changed_fields(self) -> dict[str, object]:
        """Interpreter fields whose stated value is new or different since the bot last spoke."""
        now = {f.name: self._interp.report[f.name]["value"] for f in self._schema.llm_fields
               if is_known(self._interp.report[f.name])}
        changed = {n: v for n, v in now.items() if self._known_at_last_turn.get(n) != v}
        self._known_at_last_turn = now
        return changed

    def _set_system(self, name: str, value) -> None:
        if name in self._interp.report:
            self._interp.report[name].update(value=value, confidence=1.0, evidence="system")
