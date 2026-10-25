"""Pipeline glue for the reporter: segments -> interpreter (silent), turn end -> one question.

Sits after the UserTurnProcessor. It never generates free text: the only thing
it ever sends to TTS is a template question from dialogue.py (or the greeting).

With `reports_dir`, every update also rewrites <reports_dir>/<session_id>.json:
each field's value, confidence, evidence and history, plus the full transcript.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
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
from app.interpreter.dialogue import Question, next_question
from app.interpreter.interpreter import Interpreter, Report
from app.interpreter.schema import Schema
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
        templates: dict[str, str],
        interpreter: Interpreter,
        stt: PrismaSTTService | None,
        hops: HopLogger,
        hub: ReportHub = HUB,
        set_dialogue_context: Callable[[DialogueContext], None] | None = None,
        reports_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sid, self._schema, self._templates = session_id, schema, templates
        self._interp, self._stt, self._hops, self._hub = interpreter, stt, hops, hub
        self._set_context = set_dialogue_context  # tells the turn controller what we just asked
        self._report_path = reports_dir / f"{session_id}.json" if reports_dir else None
        self._questions: list[dict] = []
        self._last_speech_end = 0.0
        self._question: Question | None = None
        self._status = "waiting for the worker"
        self._turn_task: asyncio.Task | None = None
        interpreter.on_update = self._on_report_update
        self.publish()

    def snapshot(self) -> dict:
        r = self._interp.report
        return {
            "session": self._sid,
            "status": self._status,
            "low_confidence": config.LOW_CONFIDENCE,
            "question": self._question and {"field": self._question.field, "reason": self._question.reason,
                                            "text": self._question.text},
            "fields": [{"name": f.name, "critical": f.critical, **r[f.name]} for f in self._schema.fields],
            "segments": list(self._interp.segments),
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
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": snap["status"],
            "report": {f["name"]: {k: f[k] for k in ("critical", "value", "confidence", "evidence", "history")}
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
        self.publish()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._interp.submit(frame.text.strip())  # update state only; never speak here
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
            logger.warning("turn end: interpreter calls still running at drain timeout; asking anyway")
        self._question = next_question(self._interp.report, self._schema, self._templates)
        self._hops.log("question", field=self._question.field, reason=self._question.reason, text=self._question.text)
        self._questions.append({"field": self._question.field, "reason": self._question.reason,
                                "text": self._question.text})
        self._status = "bot asking" if self._question.field else "report complete"
        self.publish()
        if self._set_context:
            self._set_context(DialogueContext(self._question.field, self._question.reason == "low_confidence"))
        await self.push_frame(TTSSpeakFrame(self._question.text))
        self._turn_task = None
