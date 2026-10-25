"""Whole conversations through the ReportProcessor (turn ends, interpreter, dialogue manager, saved
report), with a scripted LLM so they're deterministic and free. The live versions run with RUN_LIVE=1
in tests/test_e2e_replay.py."""

from __future__ import annotations

import asyncio
import json

from pipecat.frames.frames import EndFrame, TranscriptionFrame, TTSSpeakFrame, UserStartedSpeakingFrame, \
    UserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.interpreter.interpreter import Interpreter
from app.interpreter.schema import PLACEHOLDER_SCHEMA, PLACEHOLDER_TEMPLATES, parse_schema
from app.logging_utils import HopLogger
from app.report_server import ReportHub
from app.reporter import ReportProcessor

SCHEMA = parse_schema(PLACEHOLDER_SCHEMA)
T = PLACEHOLDER_TEMPLATES


def u(value, evidence, confidence=0.95, **kw):
    return {"value": value, "confidence": confidence, "evidence": evidence, **kw}


# segment text -> the interpreter's updates (what a well-behaved LLM would return)
SCRIPT = {
    "bay 2 mein crane band ho gaya": {
        "location": u("Bay 2", "bay 2"), "incident_type": u("equipment_failure", "crane band ho gaya"),
        "asset_or_equipment": u("crane", "crane"), "description": u("crane stopped", "crane band ho gaya")},
    "bahut urgent hai": {"urgency": u("high", "bahut urgent hai")},
    "crane number C 7": {"asset_or_equipment": u("crane C-7", "crane number C 7")},
    "bay 2 mein scaffolding hil raha hai bahut urgent": {
        "location": u("Bay 2", "bay 2"), "incident_type": u("other", "scaffolding hil raha hai"),
        "description": u("scaffolding shaking", "scaffolding hil raha hai"), "urgency": u("high", "bahut urgent")},
    "nahi, bay 3": {"location": u("Bay 3", "bay 3")},  # no correction flag: the read-back makes it one
    "haan sahi hai": {},
}


async def scripted_llm(messages):
    seg = json.loads(messages[1]["content"])["new_segment"]
    return json.dumps({"updates": SCRIPT.get(seg, {})})


class Spoken(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.said: list[str] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            self.said.append(frame.text)
        await self.push_frame(frame, direction)


def converse(turns: list[str], tmp_path) -> tuple[list[list[str]], dict, ReportProcessor]:
    """Each turn: the worker says one segment, then the turn ends. Returns what the bot said after each turn."""
    hops = HopLogger("flow", log_dir=tmp_path)
    reporter = ReportProcessor(session_id="flow", schema=SCHEMA, templates=T,
                               interpreter=Interpreter(SCHEMA, scripted_llm, system_prompt="", uncertainty_cue=lambda t: None),
                               stt=None, hops=hops, hub=ReportHub(), reports_dir=tmp_path / "reports")
    spoken = Spoken()
    task = PipelineTask(Pipeline([reporter, spoken]), params=PipelineParams())
    per_turn: list[list[str]] = []

    async def feed():
        await asyncio.sleep(0.3)
        for text in turns:
            before = len(spoken.said)
            await task.queue_frame(UserStartedSpeakingFrame())
            await task.queue_frame(TranscriptionFrame(text, "u", "now"))
            await asyncio.sleep(0.1)  # the segment lands before the SystemFrame turn end
            await task.queue_frame(UserStoppedSpeakingFrame())
            await asyncio.sleep(0.4)
            per_turn.append(spoken.said[before:])
        await task.queue_frame(EndFrame())

    async def main():
        await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), feed())

    asyncio.run(main())
    hops.close()
    doc = json.loads((tmp_path / "reports" / "flow.json").read_text(encoding="utf-8"))
    return per_turn, doc, reporter


def test_equipment_failure_without_an_id_gets_the_id_question(tmp_path):
    said, doc, _ = converse(["bay 2 mein crane band ho gaya", "bahut urgent hai", "crane number C 7"], tmp_path)
    assert said[0] == [T["urgency"]]  # required fields first
    assert said[1] == [T["_asset_id"].format(asset="crane")]  # then the machine's ID
    assert said[2][0].startswith("Main report read-back kar raha hoon: Bay 2, equipment failure, crane C-7")
    assert doc["report"]["asset_or_equipment"]["value"] == "crane C-7"


def test_complete_report_gets_a_readback_not_taiyar_hai(tmp_path):
    said, doc, reporter = converse(["bay 2 mein scaffolding hil raha hai bahut urgent"], tmp_path)
    assert len(said[0]) == 1 and "taiyaar" not in said[0][0] and "taiyar" not in said[0][0]
    assert said[0][0] == "Main report read-back kar raha hoon: Bay 2, other, urgency high. Sahi hai toh confirm kar do."
    assert doc["report"]["status"]["value"] == "awaiting_confirmation"
    assert reporter.dialogue.phase == "readback"


def test_nahi_bay_3_after_the_readback_updates_location_and_reconfirms(tmp_path):
    said, doc, reporter = converse(
        ["bay 2 mein scaffolding hil raha hai bahut urgent", "nahi, bay 3", "haan sahi hai", "haan sahi hai"], tmp_path)
    assert said[1] == ["Theek hai, location Bay 3. Sahi hai?"]  # only the corrected field is re-read
    assert said[2] == [T["_confirmed"]]
    assert said[3] == []  # nothing new: the bot says nothing rather than repeat itself
    loc = doc["report"]["location"]
    assert loc["value"] == "Bay 3" and [h["value"] for h in loc["history"]] == ["Bay 2"] and loc["conflicts"] == []
    assert doc["report"]["reporter_confirmation"]["value"] == "corrected_then_confirmed"


def test_no_line_is_spoken_twice_in_a_row(tmp_path):
    said, _, _ = converse(["bay 2 mein crane band ho gaya", "hmm", "hmm", "hmm", "hmm"], tmp_path)
    lines = [line for turn in said for line in turn]
    assert all(a != b for a, b in zip(lines, lines[1:])), lines
