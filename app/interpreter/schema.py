"""Report schema and question templates: research/ files if present, else the placeholders.

Two schema formats are understood; anything else stops the bot at startup (SchemaFormatError):

    KaamSuno v1.x (Aastha, research/schema.json):
        {"schema_version", "required_before_submission": [...], "validation_rules": [...],
         "fields": [{"name", "type", "required", "allowed"?, "description"}]}
        critical = required_before_submission minus config.NOT_ASKED_REQUIRED
    legacy (our Phase 1 placeholder):
        {"incident_types": [...], "fields": [{"name", "critical", "description", "enum"?}]}

The silent fallback this replaces loaded Aastha's v1.1 file as "27 fields, 0 critical",
so the dialogue said "report complete" on every turn (see notes/decisions.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from app import config
from app.config import RESEARCH_DIR

SCHEMA_PATH = RESEARCH_DIR / "schema.json"
TEMPLATES_PATH = RESEARCH_DIR / "question_templates.json"


class SchemaFormatError(ValueError):
    pass


# Placeholder in Aastha's v1.x format and field names, for when research/schema.json is absent.
PLACEHOLDER_SCHEMA = {
    "schema_version": "placeholder",
    "required_before_submission": ["incident_type", "location", "description", "urgency", "reporter_confirmation"],
    "validation_rules": ["Never invent a location, quantity, unit, asset ID, date or cause."],
    "fields": [
        {"name": "incident_id", "type": "string", "required": False, "description": "generated"},
        {"name": "site_name", "type": "string", "required": True, "description": "site / project name"},
        {"name": "location", "type": "string", "required": True, "description": "tower / floor / area on site"},
        {"name": "incident_type", "type": "enum", "required": True,
         "allowed": ["equipment_failure", "safety_hazard", "injury_or_near_miss", "material_shortage", "other"],
         "description": "best-fit category"},
        {"name": "description", "type": "string", "required": True, "description": "short faithful summary"},
        {"name": "asset_or_equipment", "type": "string", "required": False, "description": "machine and asset ID"},
        {"name": "material", "type": "string", "required": False, "description": "material name"},
        {"name": "quantity", "type": "number|string", "required": False, "description": "a number, as digits"},
        {"name": "unit", "type": "string", "required": False, "description": "kg, bags, litres...; never guess"},
        {"name": "spoken_number_raw", "type": "string", "required": False, "description": "original number phrase"},
        {"name": "urgency", "type": "enum", "required": True, "allowed": ["emergency", "high", "normal", "low", "unknown"],
         "description": "never downgrade an apparent emergency"},
        {"name": "missing_fields", "type": "array[string]", "required": False, "description": "still unknown"},
        {"name": "uncertainties", "type": "array[object]", "required": False,
         "description": "asked and not answered, or low confidence"},
        {"name": "conflicts", "type": "array[object]", "required": False, "description": "both values kept"},
        {"name": "transcript", "type": "string", "required": False, "description": "what was said"},
        {"name": "reporter_confirmation", "type": "enum", "required": True,
         "allowed": ["confirmed", "corrected_then_confirmed", "rejected", "not_obtained"], "description": "read-back"},
        {"name": "status", "type": "enum", "required": False, "allowed": ["draft", "awaiting_confirmation", "submitted"],
         "description": "workflow state"},
    ],
}

PLACEHOLDER_TEMPLATES = {
    "location": "Yeh kis jagah pe hai? Tower aur manzil bataiye.",
    "incident_type": "Kya hua hai? Machine kharab hai, material kam hai, ya koi khatra hai?",
    "asset_or_equipment": "Kaunsi machine hai? Uska number bhi bataiye agar pata ho.",
    "description": "Thoda aur bataiye, hua kya?",
    "quantity": "Kitna chahiye? Number bataiye.",
    "unit": "Kis unit mein? Bori, kilo, ya piece?",
    "urgency": "Kitna urgent hai? Abhi turant chahiye, ya aaj ke andar chalega?",
    "material": "Kaunsa material hai?",
    "date_or_time": "Yeh kab hua?",
    "_opening": "Bataiye, kya hua hai? Kahan aur kya dikkat hai?",
    "_low_confidence": "Ek baar confirm kijiye: {field_label} {value} hai, sahi?",
    "_conflict": "{field_label} pe do baatein suni: {values}. Kaunsa sahi hai?",
    "_emergency": "Yeh emergency hai. Main abhi supervisor ko alert kar raha hoon. Aap safe jagah pe rahiye.",
    "_generic": "{field_label} bataiye.",
    "people_at_risk": "Kya koi aur log khatre mein hain? Kitne log?",
    "_asset_id": "{asset} ka ID ya number kya hai?",
    "_repeat": "Yeh detail clear nahi hui, please ek baar repeat kar do.",  # only when nothing usable was heard
    "_reask": "Bas {field_label} bata dijiye.",  # 2nd ask of a field the worker's answer didn't fill
    "_which_value": "Theek hai, sahi {field_label} kya hai?",  # read-back: "location galat hai"
    # read-back (minimal, pulled forward from Phase 3; sending to the supervisor stays in Phase 3)
    "_readback": "Main report read-back kar raha hoon. Sahi hai toh confirm kar do.",
    "_readback_prompt": "Report sahi hai? Haan ya nahi boliye.",
    "_reread": "Theek hai, {changes}. Sahi hai?",
    "_rejection": "Theek hai, kaunsi detail galat hai? Woh bata dijiye.",
    "_confirmed": "Theek hai, report confirm ho gayi. Dhanyavaad.",
    "_not_given": "{field_label} nahi pata",
}


@dataclass(frozen=True)
class Field:
    name: str
    critical: bool
    description: str
    enum: tuple[str, ...] | None = None
    type: str = "string"
    required: bool = False


@dataclass(frozen=True)
class Schema:
    incident_types: tuple[str, ...]
    fields: tuple[Field, ...]
    version: str = "legacy"
    rules: tuple[str, ...] = ()
    required: tuple[str, ...] = field(default=())  # required_before_submission, as given

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def critical_priority(self) -> list[str]:
        """Critical fields in ask order (config.ASK_FIRST + ASK_LAST order first, then schema order)."""
        crit = [f.name for f in self.fields if f.critical]
        ordered = [n for n in (*config.ASK_FIRST, *config.ASK_LAST) if n in crit]
        return ordered + [n for n in crit if n not in ordered]

    @property
    def llm_fields(self) -> list[Field]:
        """Fields the interpreter may fill; the rest are system/code fields."""
        return [f for f in self.fields if f.name not in config.SYSTEM_FIELDS]

    def has(self, name: str) -> bool:
        return any(f.name == name for f in self.fields)

    def field(self, name: str) -> Field:
        return next(f for f in self.fields if f.name == name)

    def allowed(self, name: str) -> tuple[str, ...] | None:
        if name == "incident_type" and self.incident_types:
            return self.incident_types
        return self.field(name).enum


def _parse_kaamsuno(raw: dict) -> Schema:
    req = list(raw.get("required_before_submission") or [])
    names = {f.get("name") for f in raw["fields"]}
    unknown = [r for r in req if r not in names]
    if unknown:
        raise SchemaFormatError(f"required_before_submission names fields that don't exist: {unknown}")
    critical = set(req) - set(config.NOT_ASKED_REQUIRED)
    fields = []
    for f in raw["fields"]:
        if "name" not in f:
            raise SchemaFormatError(f"a field has no 'name': {f}")
        allowed = f.get("allowed")
        if f.get("type") == "enum" and not allowed:
            raise SchemaFormatError(f"enum field {f['name']!r} has no 'allowed' list")
        fields.append(Field(f["name"], f["name"] in critical, f.get("description", ""),
                            tuple(str(a) for a in allowed) if allowed else None, f.get("type", "string"),
                            bool(f.get("required"))))
    it = next((f.enum for f in fields if f.name == "incident_type"), None) or ()
    return Schema(tuple(it), tuple(fields), str(raw.get("schema_version")), tuple(raw.get("validation_rules") or ()),
                  tuple(req))


def _parse_legacy(raw: dict) -> Schema:
    fields = tuple(
        Field(f["name"], bool(f.get("critical")), f.get("description", ""), tuple(f["enum"]) if f.get("enum") else None,
              required=bool(f.get("critical")))
        for f in raw["fields"]
    )
    return Schema(tuple(raw.get("incident_types", ())), fields, "legacy",
                  required=tuple(f.name for f in fields if f.critical))


def parse_schema(raw: dict, source: str = "<dict>") -> Schema:
    """Recognise the format or raise SchemaFormatError with what was found."""
    if not isinstance(raw, dict) or not isinstance(raw.get("fields"), list) or not raw["fields"]:
        raise SchemaFormatError(f"{source}: expected an object with a non-empty 'fields' list")
    if "required_before_submission" in raw:
        schema = _parse_kaamsuno(raw)
    elif any("critical" in f for f in raw["fields"]):
        schema = _parse_legacy(raw)
    else:
        raise SchemaFormatError(
            f"{source}: unrecognised schema format (top-level keys {sorted(raw)}). Expected KaamSuno v1.x "
            "('required_before_submission' + fields[].required/allowed) or legacy (fields[].critical)."
        )
    if not schema.critical_priority:
        raise SchemaFormatError(f"{source}: no critical fields after parsing; the bot would never ask anything")
    return schema


def load_schema(path: Path = SCHEMA_PATH) -> Schema:
    if not path.exists():
        logger.warning(f"{path} missing; using the placeholder schema")
        return parse_schema(PLACEHOLDER_SCHEMA, "placeholder")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SchemaFormatError(f"{path}: not valid JSON ({e})") from e
    schema = parse_schema(raw, str(path))
    logger.info(f"schema {path.name} v{schema.version}: {len(schema.fields)} fields, "
                f"critical (ask order) {schema.critical_priority}")
    return schema


def load_templates(path: Path = TEMPLATES_PATH) -> dict[str, str]:
    if not path.exists():
        return dict(PLACEHOLDER_TEMPLATES)
    return {**PLACEHOLDER_TEMPLATES, **json.loads(path.read_text(encoding="utf-8"))}
