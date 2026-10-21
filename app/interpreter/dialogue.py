"""At turn end: pick the one thing to ask, and render it from the templates. No LLM."""

from __future__ import annotations

from dataclasses import dataclass

from app import config
from app.interpreter.interpreter import Report
from app.interpreter.schema import Schema


@dataclass(frozen=True)
class Question:
    field: str | None  # None = nothing missing, read-back placeholder
    reason: str  # "missing" | "low_confidence" | "complete"
    text: str


def top_missing_field(report: Report, schema: Schema, low_confidence: float = config.LOW_CONFIDENCE) -> tuple[str, str] | None:
    """(field, reason) for the first critical field, in schema priority order, that is
    missing; failing that, the first one below `low_confidence`. None if all are good."""
    critical = schema.critical_priority
    for name in critical:
        if report[name]["value"] is None:
            return name, "missing"
    for name in critical:
        if report[name]["confidence"] < low_confidence:
            return name, "low_confidence"
    return None


def next_question(report: Report, schema: Schema, templates: dict[str, str],
                  low_confidence: float = config.LOW_CONFIDENCE) -> Question:
    pick = top_missing_field(report, schema, low_confidence)
    if pick is None:
        return Question(None, "complete", config.READBACK_PLACEHOLDER)
    name, reason = pick
    if reason == "low_confidence":
        text = templates["_low_confidence"].format(field_label=name.replace("_", " "), value=report[name]["value"])
    else:
        text = templates[name]
    return Question(name, reason, text)
