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
    "nahi, bay teen": {"location": u("Bay 3", "bay teen")},
    "location bay 2 hai": {"location": u("Bay 2", "location bay 2")},
    "haan sahi hai": {},
    "teesri manzil pe cement kam pad gaya hai": {
        "location": u("3rd floor", "teesri manzil"), "incident_type": u("material_shortage", "cement kam pad gaya"),
        "material": u("cement", "cement"), "description": u("cement shortage", "cement kam pad gaya hai"),
        "urgency": u("high", "cement kam pad gaya hai", stated=False)},  # a guess: still asked
    "pump band ho gaya hai": {
        "incident_type": u("equipment_failure", "pump band ho gaya"), "asset_or_equipment": u("pump", "pump"),
        "description": u("pump stopped", "pump band ho gaya hai"),
        "urgency": u("high", "pump band ho gaya hai", stated=False)},
    "urgency urgent hai": {"urgency": u("high", "urgency urgent hai")},
}
# Short answers mean something only next to the question just asked: (asked field, segment) -> updates.
# Without that context the scripted LLM, like the real one in live test 3, extracts nothing.
ANSWERS = {
    ("urgency", "abhi turant"): {"urgency": u("high", "abhi turant")},
    ("quantity", "dedh sau bori"): {"quantity": u(150, "dedh sau bori"), "unit": u("bags", "dedh sau bori")},
    ("location", "bhed do mein"): {"location": u("Bay 2", "bhed do mein")},
    ("location", "bay teen"): {"location": u("Bay 3", "bay teen")},
    ("asset_or_equipment", "P 7"): {"asset_or_equipment": u("pump P-7", "P 7")},
}
REQUESTS: list[dict] = []


async def scripted_llm(messages):
    req = json.loads(messages[1]["content"])
    REQUESTS.append(req)
    seg, asked = req["new_segment"], (req.get("bot_just_asked") or {}).get("field")
    return json.dumps({"updates": ANSWERS.get((asked, seg), SCRIPT.get(seg, {}))})


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
    REQUESTS.clear()
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


# --- live test 3, problem 1: short answers belong to the question just asked ---------------------------
def test_short_answers_fill_the_field_just_asked_without_any_repeat(tmp_path):
    said, doc, _ = converse(["teesri manzil pe cement kam pad gaya hai", "abhi turant", "dedh sau bori"], tmp_path)
    assert said[0] == [T["urgency"]]  # the guessed urgency is still asked
    assert said[1] == [T["quantity"]]
    assert said[2][0].startswith("Main report read-back kar raha hoon: 3rd floor, material shortage, cement, 150 bags")
    assert all(T["_repeat"] not in line for turn in said for line in turn)
    r = doc["report"]
    assert (r["quantity"]["value"], r["unit"]["value"], r["urgency"]["value"]) == (150, "bags", "high")
    # the interpreter was told what was asked
    assert REQUESTS[-1]["bot_just_asked"] == {"field": "quantity", "reason": "missing", "question": T["quantity"]}


def test_place_word_rule_is_waived_for_an_answer_to_the_location_question(tmp_path):
    said, doc, _ = converse(["pump band ho gaya hai", "bhed do mein"], tmp_path)
    assert said[0] == [T["location"]]
    assert doc["report"]["location"]["value"] == "Bay 2"


def test_unhelpful_answer_gets_one_reask_then_the_dialogue_moves_on(tmp_path):
    said, doc, reporter = converse(["pump band ho gaya hai", "machine garam hai", "pata nahi sir", "abhi turant"],
                                   tmp_path)
    assert said[0] == [T["location"]]
    assert said[1] == [T["_reask"].format(field_label="location")]  # words were heard: not "clear nahi hui"
    assert said[2] == [T["urgency"]]  # location asked twice: move on
    assert "location" in reporter.dialogue.unconfirmed
    assert {"field": "location", "reason": "not_answered"} in doc["report"]["uncertainties"]["value"]


def test_repeat_only_when_nothing_usable_was_heard_and_only_once(tmp_path):
    said, doc, reporter = converse(["pump band ho gaya hai", "umm", "umm", "abhi turant"], tmp_path)
    assert said[0] == [T["location"]]
    assert said[1] == [T["_repeat"]]
    assert said[2] == [T["urgency"]]
    lines = [line for turn in said for line in turn]
    assert lines.count(T["_repeat"]) == 1


def test_unanswered_field_is_read_back_as_not_given(tmp_path):
    said, _, _ = converse(["pump band ho gaya hai", "umm", "umm", "abhi turant"], tmp_path)
    assert said[3][0].startswith("Main report read-back kar raha hoon: location nahi pata, equipment failure, pump,")


# --- live test 3, problem 2: corrections at the read-back ----------------------------------------------
READY = "bay 2 mein scaffolding hil raha hai bahut urgent"  # complete: read back at once


def test_correction_in_one_breath_nahi_bay_teen(tmp_path):
    said, doc, reporter = converse([READY, "nahi, bay teen", "haan"], tmp_path)
    assert said[1] == ["Theek hai, location Bay 3. Sahi hai?"]
    assert said[2] == [T["_confirmed"]]
    loc = doc["report"]["location"]
    assert loc["value"] == "Bay 3" and [h["value"] for h in loc["history"]] == ["Bay 2"]
    assert doc["report"]["reporter_confirmation"]["value"] == "corrected_then_confirmed"


def test_correction_in_two_turns_location_galat_hai_then_bay_teen(tmp_path):
    said, doc, _ = converse([READY, "location galat hai", "bay teen", "haan"], tmp_path)
    assert said[1] == [T["_which_value"].format(field_label="location")]
    assert said[2] == ["Theek hai, location Bay 3. Sahi hai?"]  # only the corrected field
    assert said[3] == [T["_confirmed"]]
    assert doc["report"]["location"]["value"] == "Bay 3"


def test_nahi_then_the_corrected_value(tmp_path):
    # live test 3: "nahi" -> "which detail?" -> "urgency urgent hai" was answered with "haan ya nahi"
    said, doc, _ = converse(["pump band ho gaya hai", "bhed do mein", "umm", "umm", "nahi", "urgency urgent hai"],
                            tmp_path)
    assert said[3][0].startswith("Main report read-back kar raha hoon")  # urgency read back as not given
    assert said[4] == [T["_rejection"]]
    assert said[5] == ["Theek hai, urgency high. Sahi hai?"]
    assert doc["report"]["urgency"]["value"] == "high"


def test_restating_a_value_re_reads_it(tmp_path):
    said, _, _ = converse([READY, "location bay 2 hai"], tmp_path)
    assert said[1] == ["Theek hai, location Bay 2. Sahi hai?"]


def test_haan_ya_nahi_only_when_the_reply_has_neither_yes_no_nor_field_content(tmp_path):
    said, _, _ = converse([READY, "kya bola aapne"], tmp_path)
    assert said[1] == [T["_readback_prompt"]]
