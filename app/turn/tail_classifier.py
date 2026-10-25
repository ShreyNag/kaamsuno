"""What do the last 1-3 words before a pause say about whether the worker is done?

    hesitation          "umm", "matlab", "kya bolte hain"      -> they're thinking, wait
    continuation        "aur", "jaise", "because", "ke"        -> the sentence isn't finished, wait
    completion          "bas", "itna hi", "that's it"          -> done, reply fast
    number_in_progress  "employee ID chaar saat..."            -> digits still coming, wait
    short_answer        "haan sahi hai" after a confirmation  -> done, reply fast
    other               anything else (Smart Turn decides)

Lists come from research/fillers.json (Aastha) when present, else the placeholder
below. Matching is on normalised text, in Latin and Devanagari script (hi-IN
returns Devanagari).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from loguru import logger

from app import config

TAIL_CLASSES = ("hesitation", "continuation", "completion", "number_in_progress", "short_answer", "other")

# Placeholder until research/fillers.json arrives. Single words and short phrases.
PLACEHOLDER_FILLERS: dict[str, list[str]] = {
    "hesitation": [
        "um", "umm", "uh", "uhh", "hmm", "aa", "aah", "er", "erm", "matlab", "woh", "wo", "kya bolte hain",
        "kya kehte hain", "kya bolte", "yaani", "mtlb",
        "उम", "उम्म", "अं", "आ", "अ", "मतलब", "वो", "वह", "क्या बोलते हैं", "क्या कहते हैं", "यानी",
    ],
    "continuation": [
        "aur", "jaise", "jaise ki", "for example", "such as", "toh", "to", "kyunki", "kyonki", "phir", "ki", "ke",
        "ka", "and", "because", "but", "or", "so", "like", "the", "a", "of", "with", "lekin", "par", "ya", "ki wajah se",
        "और", "जैसे", "जैसे कि", "तो", "क्योंकि", "फिर", "कि", "की", "के", "का", "लेकिन", "या",
    ],
    "completion": [
        "bas", "itna hi", "bas itna hi", "ho gaya", "that's it", "thats it", "that is it", "bas yahi", "done",
        "khatam", "itna hi hai", "bas itna",
        "बस", "इतना ही", "बस इतना ही", "हो गया", "बस यही", "खत्म",
    ],
}
# Yes/no vocabulary for short answers to a confirmation question (Aastha's dialogue_scripts.json overrides on Day 3)
YES_NO_WORDS = {
    "haan", "han", "haa", "ha", "ji", "sahi", "hai", "theek", "thik", "correct", "yes", "yeah", "ok", "okay",
    "nahi", "nahin", "na", "galat", "no", "nope", "bilkul", "right",
    "हाँ", "हां", "हा", "जी", "सही", "है", "ठीक", "नहीं", "ना", "गलत", "बिल्कुल",
}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "double", "triple",
    "hundred", "thousand",
    "shunya", "ek", "do", "teen", "char", "chaar", "panch", "paanch", "chhe", "chhah", "che", "saat", "aath", "nau",
    "das", "sau", "hazaar", "hazar", "lakh", "dedh", "dhai", "dhaai", "saade", "sawa", "paune",
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "पाँच", "छह", "छः", "सात", "आठ", "नौ", "दस", "सौ", "हज़ार", "हजार",
    "लाख", "डेढ़", "ढाई", "साढ़े", "सवा", "पौने",
}
# Words that say "an ID is being read out" earlier in the turn
ID_CUES = {"id", "aaidee", "aidi", "number", "no", "numbar", "आईडी", "नंबर", "नम्बर"}

_REPEATS = re.compile(r"(.)\1{2,}")


def normalise(text: str) -> str:
    """Lowercase, punctuation/symbols to spaces, letters repeated 3+ times collapsed to 2 ("ummm" -> "umm")."""
    t = "".join(" " if unicodedata.category(c)[0] in "PS" and c != "'" else c for c in text.lower())
    t = t.replace("'", "")  # "that's" -> "thats"
    return " ".join(_REPEATS.sub(r"\1\1", t).split())


def tokens(text: str) -> list[str]:
    return normalise(text).split()


def is_number(tok: str) -> bool:
    return tok.isdigit() or tok in NUMBER_WORDS or bool(re.fullmatch(r"\d+(st|nd|rd|th)?", tok))


def _phrases(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens(w)) for w in words if tokens(w)}


@dataclass(frozen=True)
class Fillers:
    hesitation: frozenset[tuple[str, ...]]
    continuation: frozenset[tuple[str, ...]]
    completion: frozenset[tuple[str, ...]]

    @classmethod
    def from_lists(cls, lists: dict[str, list[str]]) -> Fillers:
        return cls(*(frozenset(_phrases(lists.get(k, []))) for k in ("hesitation", "continuation", "completion")))


def _flatten(v) -> list[str]:
    """A class's entry is a list, or {"latin": [...], "devanagari": [...]} (any keys)."""
    if isinstance(v, dict):
        return [w for sub in v.values() for w in _flatten(sub)]
    return [str(w) for w in v] if isinstance(v, list) else []


@lru_cache(maxsize=4)  # warn once, parse once
def load_fillers(path: Path = config.FILLERS_PATH) -> Fillers:
    if not path.exists():
        logger.warning(f"{path} missing; using the placeholder filler lists")
        return Fillers.from_lists(PLACEHOLDER_FILLERS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    lists = {k: _flatten(raw.get(k, [])) for k in ("hesitation", "continuation", "completion")}
    for k, v in lists.items():
        if not v:
            logger.warning(f"{path}: no '{k}' list; using the placeholder for it")
            lists[k] = PLACEHOLDER_FILLERS[k]
    return Fillers.from_lists(lists)


@dataclass(frozen=True)
class DialogueContext:
    """What the bot last asked. Set by the reporter when it speaks."""

    asked_field: str | None = None
    confirmation: bool = False  # a yes/no confirmation question ("... sahi?")


@dataclass(frozen=True)
class TailResult:
    tail_class: str
    tail: str  # the matched phrase or last token(s)


class TailClassifier:
    def __init__(self, fillers: Fillers | None = None, max_tokens: int = config.TURN_POLICY.tail_tokens):
        self.fillers = fillers or load_fillers()
        self.max_tokens = max_tokens

    def _ending(self, toks: list[str]) -> tuple[str, tuple[str, ...]] | None:
        """Longest filler phrase the text ends with; ties go hesitation > continuation > completion."""
        for n in range(min(self.max_tokens, len(toks)), 0, -1):
            end = tuple(toks[-n:])
            for cls in ("hesitation", "continuation", "completion"):
                if end in getattr(self.fillers, cls):
                    return cls, end
        return None

    def classify(self, segment: str, context: DialogueContext = DialogueContext(), turn_text: str = "") -> TailResult:
        """`segment` is the text just before the pause; `turn_text` everything said this turn (defaults to it)."""
        toks = tokens(segment)
        turn = tokens(turn_text) or toks
        if not toks:
            return TailResult("other", "")
        hit = self._ending(toks)
        if hit and hit[0] in ("hesitation", "continuation"):
            return TailResult(hit[0], " ".join(hit[1]))
        last = toks[-1]
        if is_number(last) and self._numeric_context(turn, context):
            return TailResult("number_in_progress", last)
        if context.confirmation and (len(turn) == 1 or all(t in YES_NO_WORDS for t in turn)):
            return TailResult("short_answer", " ".join(turn[-self.max_tokens:]))
        if hit:
            return TailResult(hit[0], " ".join(hit[1]))
        return TailResult("other", " ".join(toks[-self.max_tokens:]))

    @staticmethod
    def _numeric_context(turn: list[str], context: DialogueContext) -> bool:
        if context.asked_field in config.ID_FIELDS + config.QUANTITY_FIELDS:
            return True
        # "employee ID chaar saat": an ID cue followed only by numbers up to the pause
        i = len(turn) - 1
        while i >= 0 and is_number(turn[i]):
            i -= 1
        return i >= 0 and turn[i] in ID_CUES
