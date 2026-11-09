"""Plain words for field names and enum values: nothing internal (material_shortage, "other", asset_or_equipment)
is ever spoken, shown in the supervisor's message or on the report page.

    field_word("asset_or_equipment", "hinglish") -> "machine"
    say_value("incident_type", "material_shortage", "hinglish") -> "material ki kami"
    say_value("incident_type", "other", ...) -> None   (says nothing: read back as unknown / left out)

Languages: "hinglish" | "hi" (romanised Hindi; same words as Hinglish here) | "en". The supervisor's message
and the page use "en".
"""

from __future__ import annotations

from typing import Any

FIELD_WORDS: dict[str, dict[str, str]] = {
    "location": {"hinglish": "location", "en": "location"},
    "incident_type": {"hinglish": "problem", "en": "type of problem"},
    "asset_or_equipment": {"hinglish": "machine", "en": "machine or equipment"},
    "material": {"hinglish": "material", "en": "material"},
    "quantity": {"hinglish": "kitna", "en": "quantity"},
    "unit": {"hinglish": "unit", "en": "unit"},
    "description": {"hinglish": "detail", "en": "details"},
    "urgency": {"hinglish": "urgency", "en": "urgency"},
    "people_at_risk": {"hinglish": "khatre mein log", "en": "people at risk"},
    "reporter_name_or_id": {"hinglish": "aapka naam ya ID", "en": "your name or ID"},
    "date_or_time": {"hinglish": "time", "en": "time"},
    "requested_action": {"hinglish": "kya karna hai", "en": "action needed"},
    "assigned_team": {"hinglish": "team", "en": "team"},
    "site_name": {"hinglish": "site", "en": "site"},
}

# Enum values. None = say nothing (the value tells the listener nothing).
VALUE_WORDS: dict[str, dict[str, dict[str, str | None]]] = {
    "incident_type": {
        "hinglish": {"equipment_failure": "machine kharab", "safety_hazard": "safety ka khatra",
                     "injury_or_near_miss": "chot ya haadsa", "material_shortage": "material ki kami",
                     "quality_issue": "quality ki dikkat", "utility_outage": "bijli ya paani band",
                     "access_or_logistics": "raasta ya delivery ki dikkat",
                     "environmental_issue": "pollution ki dikkat", "other": None},
        "en": {"equipment_failure": "Equipment breakdown", "safety_hazard": "Safety hazard",
               "injury_or_near_miss": "Injury or near miss", "material_shortage": "Material shortage",
               "quality_issue": "Quality problem", "utility_outage": "Power or water outage",
               "access_or_logistics": "Access or delivery problem", "environmental_issue": "Environmental problem",
               "other": None},
    },
    "urgency": {
        "hinglish": {"emergency": "emergency", "high": "bahut urgent", "normal": "normal", "low": "jaldi nahi",
                     "unknown": None},
        "en": {"emergency": "EMERGENCY", "high": "Urgent", "normal": "Normal", "low": "Not urgent", "unknown": None},
    },
    "assigned_team": {
        "en": {"maintenance": "Maintenance", "electrical": "Electrical", "safety": "Safety",
               "stores_or_materials": "Stores / materials", "civil_or_quality": "Civil / quality",
               "operations": "Operations", "supervisor": "Supervisor", "environmental": "Environmental",
               "unknown": None},
    },
    "reporter_confirmation": {
        "en": {"confirmed": "Confirmed by the worker", "corrected_then_confirmed": "Corrected, then confirmed",
               "rejected": "Rejected by the worker", "not_obtained": "Not yet confirmed"},
    },
    "status": {
        "en": {"draft": "Collecting details", "awaiting_clarification": "Waiting for details",
               "awaiting_confirmation": "Waiting for the worker's OK", "submitted": "Sent to supervisor",
               "assigned": "Assigned", "in_progress": "In progress", "resolved": "Resolved", "closed": "Closed"},
    },
}


def _lang(language: str) -> str:
    return "en" if language == "en" else "hinglish"


def field_word(name: str, language: str = "hinglish") -> str:
    words = FIELD_WORDS.get(name)
    return words[_lang(language)] if words else name.replace("_", " ")


def say_value(name: str, value: Any, language: str = "hinglish") -> str | None:
    """A field's value in plain words, or None when it says nothing. Free-text values are returned as they
    are; an unmapped enum-looking value (snake_case) loses its underscores."""
    if value is None:
        return None
    table = VALUE_WORDS.get(name, {})
    words = table.get(_lang(language)) or table.get("en") or table.get("hinglish")
    if words is not None and str(value) in words:
        return words[str(value)]
    return str(value).replace("_", " ") if isinstance(value, str) else str(value)
