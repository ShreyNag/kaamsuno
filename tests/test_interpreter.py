import asyncio
import json

from app.interpreter.dialogue import next_question, top_missing_field
from app.interpreter.interpreter import Interpreter, empty_report, merge_update
from app.interpreter.schema import PLACEHOLDER_SCHEMA, PLACEHOLDER_TEMPLATES, parse_schema

SCHEMA = parse_schema(PLACEHOLDER_SCHEMA)
SAID = "umm teesri manzil pe dhai sau bori cement kam hai"


def upd(value, evidence, confidence=0.9):
    return {"value": value, "confidence": confidence, "evidence": evidence}


# --- merge ---------------------------------------------------------------------
def test_new_values_are_set():
    r = merge_update(empty_report(SCHEMA), {"quantity": upd(250, "dhai sau"), "unit": upd("bori", "bori")}, SAID, SCHEMA)
    assert r["quantity"]["value"] == 250 and r["quantity"]["evidence"] == "dhai sau"
    assert r["unit"]["value"] == "bori"
    assert r["location"]["value"] is None


def test_nulls_and_unsupported_values_stay_null():
    r = merge_update(empty_report(SCHEMA), {
        "location": upd(None, "teesri manzil"),          # null value
        "employee_id": upd("4572", "employee ID 4572"),  # evidence not in transcript: invented
        "time": upd("subah", ""),                         # no evidence
        "nonsense_field": upd("x", "umm"),                # not in schema
    }, SAID, SCHEMA)
    assert all(v["value"] is None for v in r.values())
    assert "nonsense_field" not in r


def test_enum_fields_are_validated():
    r = merge_update(empty_report(SCHEMA), {
        "incident_type": upd("Material_Issue", "cement kam hai"),
        "severity": upd("apocalyptic", "kam hai"),
    }, SAID, SCHEMA)
    assert r["incident_type"]["value"] == "material_issue"
    assert r["severity"]["value"] is None


def test_correction_overwrites_and_keeps_history():
    r = merge_update(empty_report(SCHEMA), {"location": upd("3rd floor", "teesri manzil")}, SAID, SCHEMA)
    said2 = SAID + " nahi nahi chauthi manzil"
    r2 = merge_update(r, {"location": upd("4th floor", "chauthi manzil", 0.95)}, said2, SCHEMA)
    assert r2["location"]["value"] == "4th floor"
    assert r2["location"]["history"] == [{"value": "3rd floor", "confidence": 0.9, "evidence": "teesri manzil"}]
    assert r["location"]["value"] == "3rd floor"  # merge is pure


def test_evidence_match_ignores_case_and_punctuation():
    r = merge_update(empty_report(SCHEMA), {"location": upd("3rd floor", "Teesri  manzil,")}, SAID, SCHEMA)
    assert r["location"]["value"] == "3rd floor"


# --- ordering ------------------------------------------------------------------
def test_results_merge_in_segment_order_even_when_they_return_out_of_order():
    segs = ["teesri manzil pe", "nahi nahi chauthi manzil"]
    answers = {
        segs[0]: (0.3, {"location": upd("3rd floor", "teesri manzil")}),  # slow
        segs[1]: (0.0, {"location": upd("4th floor", "chauthi manzil")}),  # fast
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
        return json.dumps({"updates": {"unit": upd("bori", "bori")}})

    async def main():
        it = Interpreter(SCHEMA, complete, system_prompt="")
        it.submit("bad")
        it.submit("dedh sau bori")
        await it.drain(2.0)
        return it.report

    assert asyncio.run(main())["unit"]["value"] == "bori"


# --- dialogue ------------------------------------------------------------------
def filled(**values):
    r = empty_report(SCHEMA)
    for k, (v, c) in values.items():
        r[k].update(value=v, confidence=c, evidence="x")
    return r


def test_top_missing_follows_schema_priority():
    assert top_missing_field(empty_report(SCHEMA), SCHEMA) == ("incident_type", "missing")
    r = filled(incident_type=("hazard", 0.9))
    assert top_missing_field(r, SCHEMA) == ("location", "missing")


def test_missing_beats_low_confidence():
    r = filled(incident_type=("hazard", 0.3), location=("Tower B", 0.9))
    assert top_missing_field(r, SCHEMA) == ("equipment_or_material", "missing")


def test_low_confidence_asked_once_nothing_is_missing():
    crit = {n: ("v", 0.9) for n in SCHEMA.critical_priority}
    crit["severity"] = ("high", 0.4)
    q = next_question(filled(**crit), SCHEMA, PLACEHOLDER_TEMPLATES)
    assert (q.field, q.reason) == ("severity", "low_confidence")
    assert q.text == "Ek baar confirm kijiye: severity high hai, sahi?"


def test_template_question_and_readback():
    q = next_question(filled(incident_type=("hazard", 0.9)), SCHEMA, PLACEHOLDER_TEMPLATES)
    assert q.text == PLACEHOLDER_TEMPLATES["location"]
    done = filled(**{n: ("v", 0.9) for n in SCHEMA.critical_priority})
    q = next_question(done, SCHEMA, PLACEHOLDER_TEMPLATES)
    assert q.field is None and q.reason == "complete"
