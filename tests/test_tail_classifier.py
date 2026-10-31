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
ASK_ID = DialogueContext(asked_field="asset_or_equipment")  # asset IDs are read out digit by digit
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
    ("employee ID chaar saat", NONE, "id_in_progress"),
    ("employee ID 4 7", NONE, "id_in_progress"),
    ("मेरा आईडी चार सात", NONE, "id_in_progress"),
    ("chaar saat", ASK_ID, "id_in_progress"),
    ("dhai sau", ASK_QTY, "number_in_progress"),
    ("machine number chaar saal", NONE, "id_in_progress"),  # live: Prisma wrote "saat" as "saal"
    ("das saal", NONE, "other"),  # "saal" (year) with no read-out going on
    ("pichle saal", NONE, "other"),
    ("mujhe dedh sau", NONE, "number_in_progress"),  # a number word at the end: more may come
    ("paani bhijava do", NONE, "other"),  # "do" = give: a number only after a number or an ID cue
    ("chaar do", NONE, "number_in_progress"),
    ("employee id", NONE, "id_in_progress"),  # the cue alone announces an ID
    ("machine C", NONE, "id_in_progress"),
    ("kharab hai machine", NONE, "other"),  # "machine" needs an ID-like word after it
    ("pump P07 band hai", NONE, "other"),
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
    ("chahiye teesri manzil pe", NONE, "postposition"),
    ("tower B ke paas", NONE, "postposition"),
    ("yaar 2nd floor pay", NONE, "postposition"),  # Prisma's "pe" (live cut-in 2026-10-31)
    ("pipe wala", NONE, "postposition"),
    ("टावर के पास", NONE, "postposition"),
    ("तीसरी मंजिल पे", NONE, "postposition"),
    ("ek second", NONE, "hesitation"),
    ("एक सेकंड", NONE, "hesitation"),
    ("leak hai lekin", NONE, "continuation"),
    ("agar", NONE, "continuation"),
    ("mixer kharab hai", NONE, "other"),
    ("crane ka hydraulic leak ho raha hai", NONE, "other"),
    ("", NONE, "other"),
])
def test_classify(segment, context, expected):
    assert CLF.classify(segment, context).tail_class == expected


def test_turn_text_gives_id_context_across_segments():
    # The ID cue was in an earlier segment of the same turn
    res = CLF.classify("chaar saat", NONE, turn_text="mera employee ID chaar saat")
    assert res.tail_class == "id_in_progress"


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
    assert clf.classify("cement umm").tail_class == "hesitation"  # config.TURN_* always apply on top


def test_aasthas_entries_format(tmp_path):
    def e(phrase, function, note="seed phrase; verify", ctx=""):
        return {"phrase": phrase, "function": function, "regional_or_register_note": note, "example_context": ctx}

    main = tmp_path / "fillers.json"
    main.write_text(json.dumps({"entries": [
        e("umm", "hesitation_or_thinking"),
        e("uhm", "variant_or_spelling", "spelling/script variant", "Variant associated with 'umm'."),
        e("uske baad", "continuation_signal"),
        e("nahi nahi", "self_correction"),
        e("bas", "possible_turn_completion"),
        e("haan sahi hai", "confirmation_or_rejection"),
        e("shayad", "uncertainty_or_low_confidence"),
        e("haan pump ke baare mein", "hesitation_or_thinking", "generated candidate"),
        e("arre bhai", "address_or_attention"),
    ]}), encoding="utf-8")
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"entries": [e("ke", "continuation_signal", "ours")]}), encoding="utf-8")
    f = load_fillers(main, (extra,), False)
    assert ("uhm",) in f.hesitation  # a spelling variant takes the class of the seed it names
    assert ("haan", "pump", "ke", "baare", "mein") not in f.hesitation  # generated: off by default
    assert ("nahi", "nahi") in f.continuation and ("ke",) in f.continuation  # self-correction; our extra file
    assert ("bas",) in f.completion and ("haan", "sahi", "hai") in f.confirmation
    assert f.find("shayad 250 bori", "uncertainty") == "shayad"
    assert all(("arre", "bhai") not in getattr(f, k) for k in ("hesitation", "continuation", "completion"))
    assert ("haan", "pump", "ke", "baare", "mein") in load_fillers(main, (extra,), True).hesitation


def test_haan_alone_after_a_confirmation_is_a_short_answer_even_if_listed_as_hesitation():
    f = Fillers.from_lists({**PLACEHOLDER_FILLERS, "hesitation": PLACEHOLDER_FILLERS["hesitation"] + ["haan"]})
    clf = TailClassifier(f)
    assert clf.classify("haan", CONFIRM_LOC).tail_class == "short_answer"
    assert clf.classify("haan", NONE).tail_class == "hesitation"  # no confirmation asked: don't end on "haan"


def test_tail_spans_segments():
    """The last 1-3 words of the turn, not just of the newest segment: "ke" | "paas" split by Prisma."""
    assert CLF.classify("paas", NONE, turn_text="tower B ke paas").tail_class == "postposition"
    assert CLF.classify("manzil pe", NONE, turn_text="cement teesri manzil pe").tail_class == "postposition"
