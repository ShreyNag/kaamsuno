"""Whole conversations through the ReportProcessor (turn ends, interpreter, dialogue manager, saved
report), with a scripted LLM so they're deterministic and free. The live versions run with RUN_LIVE=1
in tests/test_e2e_replay.py."""

from __future__ import annotations

import asyncio
import json

import pytest

from pipecat.frames.frames import BotStoppedSpeakingFrame, EndFrame, TranscriptionFrame, TTSSpeakFrame, UserStartedSpeakingFrame, \
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
        "location": u("Bay 2", "bay 2"), "incident_type": u("safety_hazard", "scaffolding hil raha hai"),
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


def converse(turns: list[str], tmp_path, send_report=None, silence_send_s=60.0, bot_stops=False,
             wait_after_s=0.0) -> tuple[list[list[str]], dict, ReportProcessor]:
    """Each turn: the worker says one segment, then the turn ends. Returns what the bot said after each turn."""
    REQUESTS.clear()
    hops = HopLogger("flow", log_dir=tmp_path)
    reporter = ReportProcessor(session_id="flow", schema=SCHEMA, templates=T,
                               interpreter=Interpreter(SCHEMA, scripted_llm, system_prompt="", uncertainty_cue=lambda t: None),
                               stt=None, hops=hops, hub=ReportHub(), reports_dir=tmp_path / "reports",
                               send_report=send_report, silence_send_s=silence_send_s)
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
            if bot_stops:  # the transport's "bot finished speaking" after each reply
                await task.queue_frame(BotStoppedSpeakingFrame())
            per_turn.append(spoken.said[before:])
        before = len(spoken.said)
        await asyncio.sleep(wait_after_s)  # the worker stays silent
        if wait_after_s:
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
    assert said[0][0] == "Main report read-back kar raha hoon: Bay 2, safety hazard, urgency high. Sahi hai toh confirm kar do."
    assert doc["report"]["status"]["value"] == "awaiting_confirmation"
    assert reporter.dialogue.phase == "readback"


def test_nahi_bay_3_after_the_readback_updates_location_and_reconfirms(tmp_path):
    said, doc, reporter = converse(
        ["bay 2 mein scaffolding hil raha hai bahut urgent", "nahi, bay 3", "haan sahi hai", "haan sahi hai"], tmp_path)
    assert said[1] == ["Theek hai, location Bay 3. Sahi hai?"]  # only the corrected field is re-read
    assert said[2] == [T["_confirmed"]]  # "... Aur kuch batana hai?"
    assert said[3] == [T["_anything_else_prompt"]]  # "haan sahi hai" doesn't answer "aur kuch?"
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


SENTENCE = "Teesri manzil pe dedh sau bori cement kam pad gaya hai, urgent hai"
# What the LLM returns for it with the current prompt: the category is its own inference ("stated": false)
SCRIPT[SENTENCE] = {
    "location": u("3rd floor", "Teesri manzil"), "material": u("cement", "cement"),
    "quantity": u(150, "dedh sau bori"), "unit": u("bags", "dedh sau bori"),
    "incident_type": u("material_shortage", "cement kam pad gaya hai", 0.9, stated=False),
    "description": u("150 bags of cement short on the 3rd floor", "cement kam pad gaya hai"),
    "urgency": u("high", "urgent hai")}


def test_all_details_in_one_sentence_never_ask_incident_type(tmp_path):
    said, doc, _ = converse([SENTENCE], tmp_path)
    assert "Kya hua hai" not in said[0][0]  # the incident_type question
    assert said[0][0].startswith("Main report read-back kar raha hoon: 3rd floor, material shortage")
    assert doc["report"]["incident_type"]["value"] == "material_shortage"


# --- after the confirmation: "aur kuch?" -> send, more details, or a new issue -------------------------
SCRIPT["location bay 3 hai"] = {"location": u("Bay 3", "location bay 3")}
NEW_ISSUE = "ek aur problem hai teesri manzil pe cement kam pad gaya hai"
SCRIPT[NEW_ISSUE] = {
    "location": u("3rd floor", "teesri manzil"), "incident_type": u("material_shortage", "cement kam pad gaya"),
    "material": u("cement", "cement"), "description": u("cement shortage", "cement kam pad gaya hai")}


class FakeSupervisor:
    def __init__(self, ok=True):
        self.ok, self.got = ok, []

    async def __call__(self, values, kind):
        self.got.append((dict(values), kind))
        return self.ok


def test_nahi_after_aur_kuch_sends_the_report(tmp_path):
    sup = FakeSupervisor()
    said, doc, reporter = converse([READY, "haan", "nahi"], tmp_path, send_report=sup)
    assert said[1] == [T["_confirmed"]] and "Aur kuch" in T["_confirmed"]
    assert said[2] == [T["_submitted"]]
    (values, kind), = sup.got
    assert kind == "report" and values["location"] == "Bay 2" and values["incident_id"] == "flow"
    assert values["status"] == "submitted" and values["reporter_confirmation"] == "confirmed"
    assert doc["report"]["status"]["value"] == "submitted"
    assert reporter.dialogue.phase == "submitted"


def test_bas_also_sends_and_a_failed_send_is_said_honestly(tmp_path):
    sup = FakeSupervisor(ok=False)
    said, doc, _ = converse([READY, "haan", "bas"], tmp_path, send_report=sup)
    assert said[2] == [T["_send_failed"]] and len(sup.got) == 1
    assert doc["report"]["status"]["value"] == "submitted"  # saved locally either way


def test_no_channel_configured_is_not_reported_as_sent(tmp_path):
    said, _, reporter = converse([READY, "haan", "nahi"], tmp_path)
    assert said[2] == [T["_send_failed"]]
    assert reporter.sent == [("flow", "report", False)]


def test_new_details_after_aur_kuch_are_re_read_then_confirmed_again(tmp_path):
    sup = FakeSupervisor()
    said, doc, _ = converse([READY, "haan", "location bay 3 hai", "haan", "nahi"], tmp_path, send_report=sup)
    assert said[2] == ["Theek hai, location Bay 3. Sahi hai?"]
    assert said[3] == [T["_confirmed"]]
    assert said[4] == [T["_submitted"]]
    assert sup.got[0][0]["location"] == "Bay 3"
    assert doc["report"]["reporter_confirmation"]["value"] == "corrected_then_confirmed"


RAMBLE = "pata nahi sir woh kal wala kaam bhi abhi tak pending hai"


def test_a_ramble_after_aur_kuch_is_prompted_once_and_never_sent(tmp_path):
    sup = FakeSupervisor()
    said, _, reporter = converse([READY, "haan", "kya bola", RAMBLE, "hmm"], tmp_path, send_report=sup)
    assert said[2] == [T["_anything_else_prompt"]]  # prompted once more
    assert said[3] == [] and said[4] == []  # then quiet; a ramble with "nahi" in it is not a clear no
    assert sup.got == [] and reporter.dialogue.phase == "confirmed"


def test_silence_after_aur_kuch_sends_the_report(tmp_path):
    sup = FakeSupervisor()
    said, doc, _ = converse([READY, "haan"], tmp_path, send_report=sup, silence_send_s=0.3, bot_stops=True,
                            wait_after_s=0.8)
    assert said[1] == [T["_confirmed"]]
    assert said[2] == [T["_submitted"]] and len(sup.got) == 1
    assert doc["report"]["status"]["value"] == "submitted"


def test_silence_after_a_ramble_sends_too(tmp_path):
    sup = FakeSupervisor()
    said, _, _ = converse([READY, "haan", "kya bola", RAMBLE], tmp_path, send_report=sup, silence_send_s=0.6,
                          wait_after_s=1.0)  # the bot said nothing to the ramble: the timer starts at once
    assert said[3] == []  # not at once
    assert said[4] == [T["_submitted"]] and len(sup.got) == 1


def test_speaking_before_the_silence_timer_runs_out_cancels_it(tmp_path):
    sup = FakeSupervisor()
    said, _, _ = converse([READY, "haan", "kya bola"], tmp_path, send_report=sup, silence_send_s=0.6,
                          bot_stops=True)  # each next turn starts 0.5 s after the bot stopped
    assert sup.got == []


@pytest.mark.parametrize("reply", ["nahi", "bas", "nahi sir", "bas itna hi", "nahi, bhej do", "no that's all"])
def test_clear_short_answers_send(reply, tmp_path):
    sup = FakeSupervisor()
    said, _, _ = converse([READY, "haan", reply], tmp_path, send_report=sup)
    assert said[2] == [T["_submitted"]], reply


def test_a_new_issue_sends_the_confirmed_report_and_starts_a_new_one(tmp_path):
    sup = FakeSupervisor()
    said, doc, reporter = converse([READY, "haan", NEW_ISSUE], tmp_path, send_report=sup)
    # the first report goes out as confirmed: the new issue's words did not leak into it
    (first, kind), = sup.got
    assert (first["location"], first["incident_type"], first["incident_id"]) == ("Bay 2", "safety_hazard", "flow")
    assert first["status"] == "submitted"
    assert doc["report"]["location"]["value"] == "Bay 2"  # flow.json is the first report
    # the new report is built from the same words and asks its first missing field
    assert said[2] == [f"{T['_new_report']} {T['urgency']}"]
    assert reporter.incident_id == "flow-2"
    second = json.loads((tmp_path / "reports" / "flow-2.json").read_text(encoding="utf-8"))
    assert second["report"]["incident_type"]["value"] == "material_shortage"
    assert second["report"]["location"]["value"] == "3rd floor"
    assert second["report"]["incident_id"]["value"] == "flow-2"
    assert second["transcript"] == [NEW_ISSUE]


def test_speaking_again_after_the_report_was_sent_starts_a_new_report(tmp_path):
    sup = FakeSupervisor()
    said, _, reporter = converse([READY, "haan", "nahi", NEW_ISSUE], tmp_path, send_report=sup)
    assert said[3] == [T["urgency"]]  # already sent: no "pehli report bhej di" again
    assert len(sup.got) == 1 and reporter.incident_id == "flow-2"


# --- emergency: trigger words -> alert at once -> safety line -> details -> full report -------------------
SPARKS = "wire se sparks aa rahe hain aur log paas hain"  # DIA014; the scripted LLM extracts nothing from it
SCRIPT["bay 2 mein"] = {"location": u("Bay 2", "bay 2")}


def test_emergency_words_alert_the_supervisor_then_the_bot_collects_details(tmp_path):
    sup = FakeSupervisor()
    said, doc, reporter = converse([SPARKS, "bay 2 mein"], tmp_path, send_report=sup)
    (alert, kind), = sup.got[:1]
    assert kind == "emergency" and alert["urgency"] == "emergency" and alert["incident_id"] == "flow"
    assert said[0] == [f"{T['_emergency']} {T['location']}"]  # safety line, then the first missing field
    assert T["_emergency"] in said[0][0] and "alert bhej diya" in T["_emergency"]
    assert not any(T["_emergency"] in line for line in said[1])  # announced once
    assert doc["report"]["urgency"]["value"] == "emergency" and doc["report"]["location"]["value"] == "Bay 2"
    assert [k for _, k, _ in reporter.sent] == ["emergency"]  # alert once, the full report later


def test_emergency_alert_that_did_not_go_tells_the_worker_to_call(tmp_path):
    said, _, reporter = converse([SPARKS], tmp_path)  # no channel configured
    assert said[0][0].startswith(T["_emergency_unsent"])
    assert reporter.sent == [("flow", "emergency", False)]


def test_emergency_full_report_is_sent_after_the_read_back(tmp_path):
    sup = FakeSupervisor()
    SCRIPT["safety hazard hai abhi turant"] = {
        "incident_type": u("safety_hazard", "safety hazard"), "description": u("sparks from a wire", "safety hazard hai")}
    said, _, _ = converse([SPARKS, "bay 2 mein", "safety hazard hai abhi turant", "haan", "nahi"], tmp_path,
                          send_report=sup)
    assert said[2][0].startswith("Main report read-back kar raha hoon: Bay 2, safety hazard")
    assert said[4] == [T["_submitted"]]
    assert [k for _, k in sup.got] == ["emergency", "report"]
    assert sup.got[1][0]["urgency"] == "emergency"  # never downgraded
