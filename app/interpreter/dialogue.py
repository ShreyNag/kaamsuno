"""At turn end: pick the one thing to ask, and render it from the templates. No LLM.

Priority:
    1. emergency (urgency == "emergency")      -> alert line, no questions
    2. nothing captured yet                     -> opening question
    3. a field in config.ASK_ORDER that is empty (conditional ones only for their incident_type)
    4. any other critical field that is empty   (guard: a schema whose critical fields aren't in ASK_ORDER)
    5. a conflict on an asked field             -> "which one is right?"
    6. an asked field below LOW_CONFIDENCE      -> confirm
    7. complete                                 -> read-back placeholder
The bot never says "complete" while a critical field is empty (checked again at the end).
"""

from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.interpreter.interpreter import Report
from app.interpreter.schema import Schema


@dataclass(frozen=True)
class Question:
    field: str | None  # None for emergency / opening / complete
    reason: str  # emergency | opening | missing | conflict | low_confidence | complete
    text: str


def _empty(report: Report, name: str) -> bool:
    return name not in report or report[name]["value"] is None


def asked_fields(report: Report, schema: Schema) -> list[str]:
    """Fields the dialogue may ask about right now, in order: ASK_ORDER entries whose condition
    holds, then any critical field ASK_ORDER doesn't mention."""
    itype = report.get("incident_type", {}).get("value")
    out = [n for n, cond in config.ASK_ORDER if schema.has(n) and (cond is None or itype == cond)]
    return out + [n for n in schema.critical_priority if n not in out]


def captured_anything(report: Report, schema: Schema) -> bool:
    return any(not _empty(report, f.name) for f in schema.llm_fields)


def top_missing_field(report: Report, schema: Schema, low_confidence: float = config.LOW_CONFIDENCE) -> tuple[str, str] | None:
    """(field, reason) for the next field to ask about; None if nothing is missing or doubtful."""
    fields = asked_fields(report, schema)
    for name in fields:
        if _empty(report, name):
            return name, "missing"
    for name in fields:
        if report[name].get("conflicts"):
            return name, "conflict"
    for name in fields:
        if report[name]["confidence"] < low_confidence:
            return name, "low_confidence"
    return None


def _label(name: str) -> str:
    return name.replace("_", " ")


def next_question(report: Report, schema: Schema, templates: dict[str, str],
                  low_confidence: float = config.LOW_CONFIDENCE) -> Question:
    if report.get("urgency", {}).get("value") == "emergency":
        return Question(None, "emergency", templates["_emergency"])
    if not captured_anything(report, schema):
        return Question(None, "opening", templates["_opening"])
    pick = top_missing_field(report, schema, low_confidence)
    if pick is None:
        missing = [n for n in schema.critical_priority if _empty(report, n)]
        if missing:  # unreachable while asked_fields covers every critical field; never say "complete" anyway
            pick = (missing[0], "missing")
        else:
            return Question(None, "complete", config.READBACK_PLACEHOLDER)
    name, reason = pick
    entry = report[name]
    if reason == "missing":
        text = templates.get(name) or templates["_generic"].format(field_label=_label(name))
    elif reason == "conflict":
        values = [entry["value"], *(c["value"] for c in entry["conflicts"])]
        text = templates["_conflict"].format(field_label=_label(name), values=" ya ".join(map(str, values)))
    else:
        text = templates["_low_confidence"].format(field_label=_label(name), value=entry["value"])
    return Question(name, reason, text)
