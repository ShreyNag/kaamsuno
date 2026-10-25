import asyncio
import json

import pytest

from app import config
from app.interpreter.dialogue import next_question, top_missing_field
from app.interpreter.interpreter import Interpreter, empty_report, merge_update
from app.interpreter.schema import (
    PLACEHOLDER_SCHEMA,
    PLACEHOLDER_TEMPLATES,
    SCHEMA_PATH,
    SchemaFormatError,
    load_schema,
    parse_schema,
)

SCHEMA = parse_schema(PLACEHOLDER_SCHEMA)  # Aastha's v1.x format and field names, a subset
T = PLACEHOLDER_TEMPLATES
SAID = "umm teesri manzil pe dhai sau bori cement kam hai"


def upd(value, evidence, confidence=0.9, correction=False):
    return {"value": value, "confidence": confidence, "evidence": evidence, "correction": correction}


# --- schema loading ----------------------------------------------------------------
def test_kaamsuno_critical_is_required_minus_readback_and_site():
    assert SCHEMA.critical_priority == ["location", "incident_type", "description", "urgency"]
    assert "site_name" not in [f.name for f in SCHEMA.llm_fields]


@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="research/schema.json not present")
def test_aasthas_schema_file_loads_with_critical_fields():
    s = load_schema()
    assert s.critical_priority == ["location", "incident_type", "description", "urgency"]
    assert "material_shortage" in s.incident_types and s.allowed("urgency")[0] == "emergency"


@pytest.mark.parametrize("raw", [
    {"fields": [{"name": "x", "type": "string"}], "version": 9},  # neither format
    {"fields": []},
    [1, 2, 3],
    {"required_before_submission": ["nope"], "fields": [{"name": "x"}]},  # required names a missing field
    {"required_before_submission": ["reporter_confirmation"],  # nothing left to ask
     "fields": [{"name": "reporter_confirmation", "type": "enum", "allowed": ["confirmed"]}]},
    {"required_before_submission": [], "fields": [{"name": "u", "type": "enum"}]},  # enum without allowed
])
def test_unrecognised_schema_fails_loudly(raw):
    with pytest.raises(SchemaFormatError):
        parse_schema(raw, "bad.json")


def test_unparseable_schema_file_fails_loudly(tmp_path):
    p = tmp_path / "schema.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaFormatError):
        load_schema(p)


def test_system_fields_are_prefilled():
    r = empty_report(SCHEMA)
    assert r["site_name"]["value"] == config.SITE_NAME
    assert r["reporter_confirmation"]["value"] == "not_obtained"


# --- merge ---------------------------------------------------------------------
def test_new_values_are_set():
    r = merge_update(empty_report(SCHEMA), {"quantity": upd(250, "dhai sau"), "unit": upd("bags", "bori")}, SAID, SCHEMA)
    assert r["quantity"]["value"] == 250 and r["quantity"]["evidence"] == "dhai sau"
    assert r["unit"]["value"] == "bags"
    assert r["location"]["value"] is None


def test_normalised_quantity_keeps_the_spoken_phrase():
    r = merge_update(empty_report(SCHEMA), {"quantity": upd(250, "dhai sau")}, SAID, SCHEMA)
    assert r["spoken_number_raw"]["value"] == "dhai sau"
    r = merge_update(empty_report(SCHEMA), {"quantity": upd(250, "250")}, "250 bori", SCHEMA)
    assert r["spoken_number_raw"]["value"] is None  # nothing was normalised


def test_nulls_invented_and_system_values_are_dropped():
    r = merge_update(empty_report(SCHEMA), {
        "location": upd(None, "teesri manzil"),            # null value
        "asset_or_equipment": upd("JCB-4572", "JCB 4572"),  # evidence not in transcript: invented
        "material": upd("cement", ""),                      # no evidence
        "site_name": upd("Other Site", "teesri manzil"),    # system field: never from the LLM
        "nonsense_field": upd("x", "umm"),                  # not in schema
    }, SAID, SCHEMA)
    assert r["location"]["value"] is None and r["asset_or_equipment"]["value"] is None
    assert r["material"]["value"] is None and r["site_name"]["value"] == config.SITE_NAME
    assert "nonsense_field" not in r


def test_enum_fields_are_validated_against_allowed():
    r = merge_update(empty_report(SCHEMA), {
        "incident_type": upd("Material Shortage", "cement kam hai"),
        "urgency": upd("apocalyptic", "kam hai"),
    }, SAID, SCHEMA)
    assert r["incident_type"]["value"] == "material_shortage"
    assert r["urgency"]["value"] is None


def test_emergency_is_never_downgraded():
    said = "aag lagi hai, theek hai normal hai"
    r = merge_update(empty_report(SCHEMA), {"urgency": upd("emergency", "aag lagi hai")}, said, SCHEMA)
    r = merge_update(r, {"urgency": upd("normal", "normal hai", correction=True)}, said, SCHEMA)
    assert r["urgency"]["value"] == "emergency"


def test_correction_overwrites_and_keeps_history():
    r = merge_update(empty_report(SCHEMA), {"location": upd("3rd floor", "teesri manzil")}, SAID, SCHEMA)
    said2 = SAID + " nahi nahi chauthi manzil"
    r2 = merge_update(r, {"location": upd("4th floor", "chauthi manzil", 0.95, correction=True)}, said2, SCHEMA)
    assert r2["location"]["value"] == "4th floor"
    assert r2["location"]["history"] == [{"value": "3rd floor", "confidence": 0.9, "evidence": "teesri manzil"}]
    assert r["location"]["value"] == "3rd floor"  # merge is pure


def test_unclear_different_value_is_kept_as_a_conflict_until_clarified():
    said = "300 bori chahiye ... 250 bori"
    r = merge_update(empty_report(SCHEMA), {"quantity": upd(300, "300 bori")}, said, SCHEMA)
    r = merge_update(r, {"quantity": upd(250, "250 bori")}, said, SCHEMA)  # no correction flag
    assert r["quantity"]["value"] == 300 and [c["value"] for c in r["quantity"]["conflicts"]] == [250]
    rest = filled(location=("Tower B", 0.9), incident_type=("material_shortage", 0.9),
                  description=("cement short", 0.9), urgency=("normal", 0.9), unit=("bags", 0.9),
                  material=("cement", 0.9))
    q = next_question({**rest, "quantity": r["quantity"]}, SCHEMA, T)
    assert (q.field, q.reason) == ("quantity", "conflict") and "300 ya 250" in q.text
    said2 = said + " haan 250 sahi hai"
    r = merge_update(r, {"quantity": upd(250, "250 sahi hai", correction=True)}, said2, SCHEMA)
    assert r["quantity"]["value"] == 250 and r["quantity"]["conflicts"] == []
    assert [h["value"] for h in r["quantity"]["history"]] == [300]


def test_refinement_is_not_a_conflict():
    said = "tower B ... tower B teesri manzil"
    r = merge_update(empty_report(SCHEMA), {"location": upd("Tower B", "tower B")}, said, SCHEMA)
    r = merge_update(r, {"location": upd("Tower B, 3rd floor", "tower B teesri manzil")}, said, SCHEMA)
    assert r["location"]["value"] == "Tower B, 3rd floor" and r["location"]["conflicts"] == []


def test_evidence_match_ignores_case_and_punctuation():
    r = merge_update(empty_report(SCHEMA), {"location": upd("3rd floor", "Teesri  manzil,")}, SAID, SCHEMA)
    assert r["location"]["value"] == "3rd floor"


# --- ordering ------------------------------------------------------------------
def test_results_merge_in_segment_order_even_when_they_return_out_of_order():
    segs = ["teesri manzil pe", "nahi nahi chauthi manzil"]
    answers = {
        segs[0]: (0.3, {"location": upd("3rd floor", "teesri manzil")}),  # slow
        segs[1]: (0.0, {"location": upd("4th floor", "chauthi manzil", correction=True)}),  # fast
    }

    async def complete(messages):
        seg = json.loads(messages[1]["content"])["new_segment"]
        delay, updates = answers[seg]
        await asyncio.sleep(delay)
        return json.dumps({"updates": updates})

    async def main():
        it = Interpreter(SCHEMA, complete, system_prompt="")
        for s in segs:
            it.submit(s)
        assert await it.drain(2.0)
        return it.report

    r = asyncio.run(main())
    assert r["location"]["value"] == "4th floor"
    assert [h["value"] for h in r["location"]["history"]] == ["3rd floor"]


def test_a_failed_call_does_not_stall_later_segments():
    async def complete(messages):
        seg = json.loads(messages[1]["content"])["new_segment"]
        if seg == "bad":
            raise RuntimeError("503")
        return json.dumps({"updates": {"unit": upd("bags", "bori")}})

    async def main():
        it = Interpreter(SCHEMA, complete, system_prompt="")
        it.submit("bad")
        it.submit("dedh sau bori")
        await it.drain(2.0)
        return it.report

    assert asyncio.run(main())["unit"]["value"] == "bags"


# --- dialogue ------------------------------------------------------------------
def filled(**values):
    r = empty_report(SCHEMA)
    for k, (v, c) in values.items():
        r[k].update(value=v, confidence=c, evidence="x")
    return r


ALL_CRITICAL = dict(location=("Tower B", 0.9), incident_type=("other", 0.9),
                    description=("loose scaffolding", 0.9), urgency=("high", 0.9))


def test_nothing_captured_asks_the_opening_question_never_complete():
    q = next_question(empty_report(SCHEMA), SCHEMA, T)
    assert (q.field, q.reason, q.text) == (None, "opening", T["_opening"])


@pytest.mark.parametrize("missing", ["location", "incident_type", "description", "urgency"])
def test_a_required_field_that_is_empty_is_asked_not_read_back(missing):
    vals = {k: v for k, v in ALL_CRITICAL.items() if k != missing}
    q = next_question(filled(**vals), SCHEMA, T)
    assert (q.field, q.reason) == (missing, "missing")


@pytest.mark.parametrize("value", ["unknown", "Unknown", "", "pata nahi"])
def test_unknown_values_count_as_missing(value):
    q = next_question(filled(**{**ALL_CRITICAL, "urgency": (value, 0.9)}), SCHEMA, T)
    assert (q.field, q.reason) == ("urgency", "missing")


def test_an_inferred_value_counts_as_missing():
    r = filled(**ALL_CRITICAL)
    r["urgency"]["inferred"] = True  # e.g. the LLM guessed "high" from "cement kam pad gaya hai"
    assert next_question(r, SCHEMA, T).field == "urgency"


def test_urgency_without_an_urgency_word_is_inferred():
    said = "cement kam pad gaya hai bahut jaldi chahiye"
    r = merge_update(empty_report(SCHEMA), {"urgency": upd("high", "cement kam pad gaya hai")}, said, SCHEMA)
    assert r["urgency"]["inferred"]
    r = merge_update(r, {"urgency": upd("high", "bahut jaldi chahiye")}, said, SCHEMA)
    assert not r["urgency"]["inferred"]


def test_llm_stated_false_marks_inferred_and_never_overwrites_a_stated_value():
    said = "tower B mein"
    r = merge_update(empty_report(SCHEMA), {"location": upd("Tower B", "tower B")}, said, SCHEMA)
    r = merge_update(r, {"location": {**upd("Tower C", "tower B"), "stated": False}}, said, SCHEMA)
    assert r["location"]["value"] == "Tower B" and not r["location"]["inferred"]


def test_label_only_values_are_dropped():
    said = "employee id chaar saat"
    r = merge_update(empty_report(SCHEMA), {"asset_or_equipment": upd("employee id", "employee id")}, said, SCHEMA)
    assert r["asset_or_equipment"]["value"] is None


def test_guard_holds_for_a_critical_field_outside_the_ask_order(monkeypatch):
    monkeypatch.setattr(config, "ASK_FIRST", ("location",))
    monkeypatch.setattr(config, "ASK_LAST", ())  # the ask order forgot the other required fields
    q = next_question(filled(location=("Tower B", 0.9)), SCHEMA, T)
    assert q.reason == "missing" and q.field in ("incident_type", "description", "urgency")


def test_ask_order_required_fields_first():
    assert top_missing_field(filled(description=("crane leak", 0.9)), SCHEMA) == ("location", "missing")
    assert top_missing_field(filled(location=("Tower B", 0.9)), SCHEMA) == ("incident_type", "missing")


@pytest.mark.parametrize("itype, first", [
    ("equipment_failure", "asset_or_equipment"),
    ("material_shortage", "material"),
    ("safety_hazard", "people_at_risk"),
    ("injury_or_near_miss", "people_at_risk"),
    ("other", None),
])
def test_type_dependent_fields_after_the_required_ones(itype, first):
    schema = parse_schema({**PLACEHOLDER_SCHEMA, "fields": PLACEHOLDER_SCHEMA["fields"] + [
        {"name": "people_at_risk", "type": "string", "required": False, "description": "who"}]})
    r = empty_report(schema)
    for k, (v, c) in {**ALL_CRITICAL, "incident_type": (itype, 0.9)}.items():
        r[k].update(value=v, confidence=c, evidence="x")
    q = next_question(r, schema, T)
    assert q.field == first and (q.reason == "missing" if first else q.reason == "readback")


def test_equipment_without_an_id_gets_the_id_question():
    r = filled(**{**ALL_CRITICAL, "incident_type": ("equipment_failure", 0.9)}, asset_or_equipment=("crane", 0.9))
    q = next_question(r, SCHEMA, T)
    assert (q.field, q.reason, q.text) == ("asset_or_equipment", "missing_id", "crane ka ID ya number kya hai?")
    r["asset_or_equipment"]["value"] = "crane C-7"
    assert next_question(r, SCHEMA, T).reason == "readback"


def test_emergency_skips_all_questions():
    q = next_question(filled(urgency=("emergency", 0.95)), SCHEMA, T)
    assert (q.field, q.reason, q.text) == (None, "emergency", T["_emergency"])


def test_missing_beats_low_confidence():
    r = filled(location=("Tower B", 0.3))
    assert top_missing_field(r, SCHEMA) == ("incident_type", "missing")


def test_low_confidence_number_is_confirmed():
    vals = {**ALL_CRITICAL, "urgency": ("high", 0.4)}
    q = next_question(filled(**vals), SCHEMA, T)
    assert (q.field, q.reason) == ("urgency", "low_confidence")
    assert q.text == "Ek baar confirm kijiye: urgency high hai, sahi?"


def test_complete_report_gets_a_readback_not_an_announcement():
    q = next_question(filled(**ALL_CRITICAL), SCHEMA, T)
    assert q.reason == "readback" and "taiyaar" not in q.text
    assert q.text.startswith("Main report read-back kar raha hoon: Tower B, other, urgency high")


def test_question_cap_then_readback():
    from app.interpreter.dialogue import DialogueManager

    dm = DialogueManager(max_questions=1, low_confidence=0.95)
    r = filled(**ALL_CRITICAL)  # everything at 0.9 < 0.95: four confirmations pending
    assert dm.decide(r, SCHEMA, T).reason == "low_confidence"
    assert dm.decide(r, SCHEMA, T).reason == "readback"  # cap of 1 reached


def test_never_the_same_line_twice_in_a_row():
    from app.interpreter.dialogue import DialogueManager

    dm = DialogueManager()
    r = filled(location=("Tower B", 0.9))
    first = dm.decide(r, SCHEMA, T)
    second = dm.decide(r, SCHEMA, T, "")  # the worker said nothing usable
    assert first.field == second.field == "incident_type"
    assert second.text != first.text and second.reason == "repeat"
    third = dm.decide(r, SCHEMA, T, "")  # asked twice: move on
    assert third.text not in (first.text, second.text)


def test_readback_yes_no_and_correction():
    from app.interpreter.dialogue import DialogueManager

    dm = DialogueManager()
    r = filled(**ALL_CRITICAL)
    assert dm.decide(r, SCHEMA, T).reason == "readback"
    assert dm.decide(r, SCHEMA, T, "nahi").reason == "rejection"
    q = dm.decide(r, SCHEMA, T, "nahi, Bay 3", changed={"location": "Bay 3"})
    assert (q.reason, q.field, q.text) == ("reread", "location", "Theek hai, location Bay 3. Sahi hai?")
    q = dm.decide(r, SCHEMA, T, "haan sahi hai")
    assert q.reason == "confirmed" and dm.confirmation == "corrected_then_confirmed"
    assert dm.decide(r, SCHEMA, T, "theek hai").silent  # nothing new after confirming: say nothing
    q = dm.decide(r, SCHEMA, T, "aur do log bhi the", changed={"description": "two people there"})
    assert q.reason == "reread" and dm.phase == "readback"  # new info after confirming: re-read it


def test_location_must_name_a_place():
    said = "the crane hydraulic pipe is leaking oil near gate 2"
    r = merge_update(empty_report(SCHEMA), {"location": upd("crane hydraulic pipe area", "crane hydraulic pipe")},
                     said, SCHEMA)
    assert r["location"]["value"] is None  # invented: no place word in the evidence
    r = merge_update(r, {"location": upd("Gate 2", "near gate 2")}, said, SCHEMA)
    assert r["location"]["value"] == "Gate 2"
    r = merge_update(empty_report(SCHEMA), {"location": upd("3rd floor", "तीसरी मंजिल")}, "तीसरी मंजिल पे", SCHEMA)
    assert r["location"]["value"] == "3rd floor"


def test_uncertain_words_lower_confidence_so_the_value_is_confirmed():
    async def complete(messages):
        return json.dumps({"updates": {"quantity": upd(250, "250", 0.95)}})

    async def main():
        it = Interpreter(SCHEMA, complete, system_prompt="",
                         uncertainty_cue=lambda t: "shayad" if "shayad" in t else None)
        it.submit("shayad 250 bori")
        await it.drain(2.0)
        return it.report

    r = asyncio.run(main())
    assert r["quantity"]["value"] == 250 and r["quantity"]["confidence"] < config.LOW_CONFIDENCE
