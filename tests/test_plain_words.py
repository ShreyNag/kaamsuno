"""No internal label (snake_case enum, field name, "other"/"unknown") is ever spoken or sent to the supervisor."""

from __future__ import annotations

import re

import pytest

from app.interpreter.dialogue import DialogueManager
from app.interpreter.interpreter import empty_report
from app.interpreter.schema import load_schema
from app.interpreter.templates import load_template_book
from app.notify import format_message

SCHEMA = load_schema()
BOOK = load_template_book()
INTERNAL = re.compile(r"\b[a-z]+_[a-z_]+\b|\bother\b|\bunknown\b")


def report(**values):
    r = empty_report(SCHEMA)
    for name, value in values.items():
        r[name].update(value=value, confidence=0.95, evidence="x", inferred=False)
    return r


BASE = dict(location="Bay 2", description="something broke", urgency="high", incident_type="equipment_failure",
            asset_or_equipment="mixer M-4", material="cement", quantity=150, unit="bags", people_at_risk="2")


@pytest.mark.parametrize("lang", ["hinglish", "hi", "en"])
@pytest.mark.parametrize("field", ["incident_type", "urgency"])
def test_every_enum_value_is_spoken_in_plain_words(field, lang):
    t = BOOK.for_language(lang)
    for value in SCHEMA.allowed(field):
        r = report(**{**BASE, field: value})
        if field == "urgency" and value == "emergency":
            continue  # its own path (safety line)
        dm = DialogueManager()
        lines = [dm.decide(r, SCHEMA, t).text]
        lines.append(dm._reread({field: value, "location": "Bay 3"}, t).text)
        lines.append(dm._render(field, "low_confidence", r, t))
        for line in lines:
            assert not INTERNAL.search(line), (field, value, lang, line)


@pytest.mark.parametrize("field", ["incident_type", "urgency", "reporter_confirmation"])
def test_every_enum_value_reaches_the_supervisor_in_plain_words(field):
    for value in SCHEMA.allowed(field):
        m = format_message({**BASE, "incident_id": "r1", field: value}, "report")
        assert not INTERNAL.search(m.lower().replace("another", "")), (field, value, m)


@pytest.mark.parametrize("field", ["asset_or_equipment", "people_at_risk", "reporter_name_or_id", "incident_type"])
def test_field_names_in_questions_are_plain(field):
    t = BOOK.for_language("hinglish")
    dm = DialogueManager()
    assert "_" not in dm._render(field, "missing", report(), t)
    assert "_" not in t["_reask"].format(field_label=__import__("app.interpreter.dialogue", fromlist=["_label"])._label(field, t))
