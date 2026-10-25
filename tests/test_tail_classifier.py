"""Table-driven tail classifier tests, Latin and Devanagari, on the placeholder filler lists."""

from __future__ import annotations

import json

import pytest

from app.turn.tail_classifier import (
    PLACEHOLDER_FILLERS,
    DialogueContext,
    Fillers,
    TailClassifier,
    load_fillers,
    normalise,
)

CLF = TailClassifier(Fillers.from_lists(PLACEHOLDER_FILLERS))
NONE = DialogueContext()
ASK_ID = DialogueContext(asked_field="employee_id")
ASK_QTY = DialogueContext(asked_field="quantity")
CONFIRM = DialogueContext(asked_field="quantity", confirmation=True)
CONFIRM_LOC = DialogueContext(asked_field="location", confirmation=True)


@pytest.mark.parametrize("raw, expected", [
    ("Ummm...", "umm"),
    ("teesri manzil pe, umm", "teesri manzil pe umm"),
    ("That's it!", "thats it"),
    ("Hmmmmm", "hmm"),
    ("बस, इतना ही।", "बस इतना ही"),
    ("मतलब...", "मतलब"),
])
def test_normalise(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize("segment, context, expected", [
    # hesitation
    ("teesri manzil pe umm", NONE, "hesitation"),
    ("teesri manzil pe ummmm", NONE, "hesitation"),
    ("cement matlab", NONE, "hesitation"),
    ("woh", NONE, "hesitation"),
    ("crane ka kya bolte hain", NONE, "hesitation"),
    ("Aaaa", NONE, "hesitation"),
    ("तीसरी मंजिल पे उम्म", NONE, "hesitation"),
    ("सीमेंट मतलब", NONE, "hesitation"),
    ("क्रेन का क्या बोलते हैं", NONE, "hesitation"),
    # continuation
    ("mujhe aur chijen chaahiye jaise ki", NONE, "continuation"),
    ("cement aur", NONE, "continuation"),
    ("there is a leak because", NONE, "continuation"),
    ("we need material such as", NONE, "continuation"),
    ("tower B ke", NONE, "continuation"),
    ("mixer band hai kyunki", NONE, "continuation"),
    ("सीमेंट और", NONE, "continuation"),
    ("मिक्सर बंद है क्योंकि", NONE, "continuation"),
    ("टावर बी के", NONE, "continuation"),
    # completion
    ("cement kam hai bas", NONE, "completion"),
    ("bas itna hi", NONE, "completion"),
    ("ho gaya", NONE, "completion"),
    ("That's it.", NONE, "completion"),
    ("सीमेंट कम है बस", NONE, "completion"),
    ("बस इतना ही", NONE, "completion"),
    ("हो गया", NONE, "completion"),
    # number in progress
    ("employee ID chaar saat", NONE, "number_in_progress"),
    ("employee ID 4 7", NONE, "number_in_progress"),
    ("मेरा आईडी चार सात", NONE, "number_in_progress"),
    ("chaar saat", ASK_ID, "number_in_progress"),
    ("dhai sau", ASK_QTY, "number_in_progress"),
    ("mujhe dedh sau", NONE, "other"),  # a number with no ID/quantity context is not "in progress"
    # short answer after a confirmation question
    ("haan sahi hai", CONFIRM_LOC, "short_answer"),
    ("haan", CONFIRM_LOC, "short_answer"),
    ("nahi", CONFIRM_LOC, "short_answer"),
    ("correct", CONFIRM_LOC, "short_answer"),
    ("हाँ सही है", CONFIRM_LOC, "short_answer"),
    ("जी", CONFIRM_LOC, "short_answer"),
    ("haan sahi hai", NONE, "other"),  # no confirmation was asked
    # hesitation/continuation beat a number or a confirmation
    ("employee ID chaar umm", NONE, "hesitation"),
    ("haan matlab", CONFIRM_LOC, "hesitation"),
    # other
    ("chahiye teesri manzil pe", NONE, "other"),
    ("crane ka hydraulic leak ho raha hai", NONE, "other"),
    ("", NONE, "other"),
])
def test_classify(segment, context, expected):
    assert CLF.classify(segment, context).tail_class == expected


def test_turn_text_gives_id_context_across_segments():
    # The ID cue was in an earlier segment of the same turn
    res = CLF.classify("chaar saat", NONE, turn_text="mera employee ID chaar saat")
    assert res.tail_class == "number_in_progress"


def test_short_answer_needs_the_whole_turn_to_be_short():
    res = CLF.classify("sahi hai", CONFIRM_LOC, turn_text="tower B mein crane gir gaya sahi hai")
    assert res.tail_class == "other"


def test_fillers_json_accepts_script_sublists(tmp_path):
    p = tmp_path / "fillers.json"
    p.write_text(json.dumps({
        "hesitation": {"latin": ["arre"], "devanagari": ["अरे"]},
        "continuation": ["phir se"],
        "completion": ["khatam ho gaya"],
    }, ensure_ascii=False), encoding="utf-8")
    clf = TailClassifier(load_fillers(p))
    assert clf.classify("crane arre").tail_class == "hesitation"
    assert clf.classify("क्रेन अरे").tail_class == "hesitation"
    assert clf.classify("phir se").tail_class == "continuation"
    assert clf.classify("kaam khatam ho gaya").tail_class == "completion"
    assert clf.classify("cement umm").tail_class == "other"  # Aastha's lists replace the placeholder
