"""Question texts from Aastha's dialogue_scripts.json format, per language, with our replacements."""

from __future__ import annotations

import json

import pytest

from app.interpreter.schema import PLACEHOLDER_TEMPLATES
from app.interpreter.templates import OURS, detect_language, load_template_book


def script(sid, lang, intent, resp):
    return {"script_id": sid, "language": lang, "intent": intent, "worker_utterance": "x", "bot_response": resp,
            "expected_next_action": "x", "risk_level": "low"}


@pytest.fixture
def book(tmp_path):
    p = tmp_path / "dialogue_scripts.json"
    p.write_text(json.dumps({"scripts": [
        script("DIA001", "hi", "missing_location", "Kaunsi location par machine band hui hai?"),  # hand-written
        script("DIA022", "hi", "missing_location", "Yeh kis site aur kis area mein hua?"),
        script("DIA032", "hinglish", "missing_location", "Kaunsi site aur exact area hai?"),
        script("DIA042", "en", "missing_location", "Which site and exact area is this in?"),
        script("DIA030", "hi", "emergency", "Maaf kijiye, yeh detail clear nahi hui."),  # broken in v1.1
        script("DIA061", "pa_hi", "missing_location", "Kithe hoya?"),  # unsupported language
    ]}), encoding="utf-8")
    return load_template_book(p)


def test_generated_scripts_by_language(book):
    assert book.get("location", "hi") == "Yeh kis site aur kis area mein hua?"
    assert book.get("location", "en") == "Which site and exact area is this in?"
    assert "Kithe" not in str(book.texts) and "machine band hui" not in str(book.texts)


def test_broken_emergency_script_is_replaced_by_ours(book):
    assert book.get("_emergency", "hi") == OURS["_emergency"]["hi"][0]
    assert "clear nahi" not in book.get("_emergency", "hi")


def test_fallbacks_hinglish_then_placeholder(book):
    assert book.get("urgency", "en") == OURS["urgency"]["en"][0]
    assert book.get("urgency", "hinglish") == PLACEHOLDER_TEMPLATES["urgency"]  # not the hi text
    assert book.get("_generic", "en") == PLACEHOLDER_TEMPLATES["_generic"]


def test_rotation():
    from app.interpreter.templates import TemplateBook

    b = TemplateBook({"location": {"hi": ["A", "B"]}})
    assert [b.get("location", "hi") for _ in range(3)] == ["A", "B", "A"]


@pytest.mark.parametrize("text, lang", [
    ("The conveyor has stopped.", "en"),
    ("Machine band ho gayi hai.", "hinglish"),
    ("मशीन बंद हो गई", "hi"),
    ("Paanch ya pandrah bag chahiye.", "hinglish"),
    ("Kal hua tha, ya parson shayad.", "hi"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang
