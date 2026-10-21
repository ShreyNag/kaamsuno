"""Report schema and question templates: research/ files if present, else the placeholders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.config import RESEARCH_DIR

SCHEMA_PATH = RESEARCH_DIR / "schema.json"
TEMPLATES_PATH = RESEARCH_DIR / "question_templates.json"

# Field order = question priority among critical fields (see notes/decisions.md).
PLACEHOLDER_SCHEMA = {
    "incident_types": ["equipment_fault", "material_issue", "hazard", "near_miss", "emergency"],
    "fields": [
        {"name": "incident_type", "critical": True, "description": "one of incident_types"},
        {"name": "location", "critical": True, "description": "tower / floor / area on site"},
        {"name": "equipment_or_material", "critical": True, "description": "machine or material involved"},
        {"name": "severity", "critical": True, "description": "low | medium | high | critical",
         "enum": ["low", "medium", "high", "critical"]},
        {"name": "employee_id", "critical": True, "description": "reporting worker's employee ID"},
        {"name": "what_happened", "critical": False, "description": "short description of the problem"},
        {"name": "quantity", "critical": False, "description": "a number, as digits"},
        {"name": "unit", "critical": False, "description": "unit of quantity (bori, kg, piece...)"},
        {"name": "time", "critical": False, "description": "when it happened"},
        {"name": "people_affected", "critical": False, "description": "number of people hurt or at risk"},
    ],
}

PLACEHOLDER_TEMPLATES = {
    "incident_type": "Kya hua hai? Machine kharab hai, material ki dikkat hai, ya koi khatra hai?",
    "location": "Yeh kis jagah pe hai? Tower aur manzil bataiye.",
    "equipment_or_material": "Kaunsi machine ya kaunsa material hai?",
    "severity": "Kitna gambhir hai? Kam, medium, zyada, ya bahut zyada?",
    "employee_id": "Apna employee ID bataiye.",
    "what_happened": "Thoda bataiye, hua kya?",
    "quantity": "Kitna? Number bataiye.",
    "unit": "Kis unit mein? Bori, kilo, ya piece?",
    "time": "Yeh kab hua?",
    "people_affected": "Kitne log is se prabhavit hain?",
    "_low_confidence": "Ek baar confirm kijiye: {field_label} {value} hai, sahi?",
}


@dataclass(frozen=True)
class Field:
    name: str
    critical: bool
    description: str
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Schema:
    incident_types: tuple[str, ...]
    fields: tuple[Field, ...]

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def critical_priority(self) -> list[str]:
        return [f.name for f in self.fields if f.critical]

    def field(self, name: str) -> Field:
        return next(f for f in self.fields if f.name == name)

    def allowed(self, name: str) -> tuple[str, ...] | None:
        return self.incident_types if name == "incident_type" else self.field(name).enum


def _load_json(path: Path, placeholder: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    logger.warning(f"{path} missing; using the placeholder")
    return placeholder


def parse_schema(raw: dict) -> Schema:
    fields = tuple(
        Field(f["name"], bool(f.get("critical")), f.get("description", ""), tuple(f["enum"]) if f.get("enum") else None)
        for f in raw["fields"]
    )
    return Schema(tuple(raw.get("incident_types", ())), fields)


def load_schema(path: Path = SCHEMA_PATH) -> Schema:
    return parse_schema(_load_json(path, PLACEHOLDER_SCHEMA))


def load_templates(path: Path = TEMPLATES_PATH) -> dict[str, str]:
    return {**PLACEHOLDER_TEMPLATES, **(_load_json(path, PLACEHOLDER_TEMPLATES))}
