"""What do the last 1-3 words before a pause say about whether the worker is done?

    hesitation          "umm", "matlab", "ek second"           -> they're thinking, wait
    continuation        "aur", "kyunki", "because"             -> a connector: the sentence isn't finished, wait
    postposition        "pe", "ke paas", "wala"                -> can't end a sentence, wait
    completion          "bas", "itna hi", "that's it"          -> done, reply fast
    id_in_progress      "employee ID", "machine ID chaar"      -> an ID being read out, wait
    number_in_progress  "chaar saat", "dhai sau"               -> a number word at the end, wait
    short_answer        "haan sahi hai" after a confirmation  -> done, reply fast
    other               anything else (Smart Turn decides)

The tail is the last 1-3 words of the turn's newest text (the words before the pause can span two
Prisma segments: "ke" | "paas"). Lists: config.TURN_* (all classes), unioned with research/fillers.json
(Aastha's entries, mapped by `function`, see FUNCTION_TO_CLASS) and our research/fillers_*extra*.json.
Matching is on normalised text, in Latin and Devanagari script (hi-IN returns Devanagari).

    .venv/Scripts/python -m app.turn.tail_classifier      # class counts and examples
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

TAIL_CLASSES = ("hesitation", "continuation", "postposition", "completion", "id_in_progress", "number_in_progress",
                "short_answer", "other")
INCOMPLETE_CLASSES = ("hesitation", "continuation", "postposition")  # the sentence isn't finished: wait

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
# Words that say "an ID is being read out" earlier in the turn (config.TURN_ID_CUES)
ID_CUES = set(config.TURN_ID_CUES)
# Of those, the ones that announce an ID by themselves ("employee ID [pause]"); "machine" and "no" end
# sentences too ("kharab hai machine"), so they need an ID-like word after them
ID_CUES_ALONE = ID_CUES - set(config.TURN_ID_CUES_NEED_FOLLOWER)

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


CLASSES = ("hesitation", "continuation", "completion", "confirmation", "uncertainty", "postposition")
# config's lists (the keyword rule), added to whatever the files give
CONFIG_LISTS = {"hesitation": config.TURN_HESITATION, "continuation": config.TURN_CONNECTORS,
                "completion": config.TURN_COMPLETION, "postposition": config.TURN_POSTPOSITIONS}

# Aastha's fillers.json `function` -> our class (research_integration_prompt.md). Unlisted functions
# (discourse/reference/regional markers, multi-marker sequences, ...) are ignored for turn-taking.
FUNCTION_TO_CLASS = {
    "hesitation_or_thinking": "hesitation", "pause_request": "hesitation", "hesitation_or_reference": "hesitation",
    "continuation_signal": "continuation", "continuation": "continuation", "contrast_or_continuation": "continuation",
    "self_correction": "continuation", "correction_or_pause": "continuation", "reformulation": "continuation",
    "possible_turn_completion": "completion",
    "confirmation_or_rejection": "confirmation", "possible_acknowledgement": "confirmation",
    "possible_confirmation": "confirmation", "polite_acknowledgement": "confirmation",
    "uncertainty_or_low_confidence": "uncertainty", "uncertainty": "uncertainty",
}
_VARIANT_OF = re.compile(r"[Vv]ariant associated with '([^']+)'")


@dataclass(frozen=True)
class Fillers:
    hesitation: frozenset[tuple[str, ...]]
    continuation: frozenset[tuple[str, ...]]
    completion: frozenset[tuple[str, ...]]
    confirmation: frozenset[tuple[str, ...]] = frozenset()  # short answers, only after a confirmation question
    uncertainty: frozenset[tuple[str, ...]] = frozenset()  # not a turn cue: the value being given is doubtful
    postposition: frozenset[tuple[str, ...]] = frozenset()

    @classmethod
    def from_lists(cls, lists: dict[str, list[str]], with_config: bool = True) -> Fillers:
        """`with_config`: add config.TURN_* lists; False only to test a file's lists on their own."""
        merged = {k: list(lists.get(k, [])) + (list(CONFIG_LISTS.get(k, ())) if with_config else [])
                  for k in CLASSES}
        return cls(**{k: frozenset(_phrases(v)) for k, v in merged.items()})

    def find(self, text: str, cls: str) -> str | None:
        """The first phrase of class `cls` anywhere in `text` (whole tokens), else None."""
        toks = tokens(text)
        for phrase in sorted(getattr(self, cls), key=len, reverse=True):
            n = len(phrase)
            if any(tuple(toks[i : i + n]) == phrase for i in range(len(toks) - n + 1)):
                return " ".join(phrase)
        return None

    def summary(self, examples: int = 10) -> str:
        lines = []
        for k in CLASSES:
            ph = sorted(" ".join(p) for p in getattr(self, k))
            step = max(1, len(ph) // examples)
            lines.append(f"{k:13s} {len(ph):4d}  e.g. {ph[::step][:examples]}")
        return "\n".join(lines)


def _flatten(v) -> list[str]:
    """A class's entry is a list, or {"latin": [...], "devanagari": [...]} (any keys)."""
    if isinstance(v, dict):
        return [w for sub in v.values() for w in _flatten(sub)]
    return [str(w) for w in v] if isinstance(v, list) else []


def _entry_kind(e: dict) -> str:
    note = str(e.get("regional_or_register_note", "")).lower()
    if note.startswith("generated"):
        return "generated"
    if note.startswith("spelling") or e.get("function") == "variant_or_spelling":
        return "variant"
    return "seed"  # "seed phrase", and our own additions


def lists_from_entries(entries: list[dict], include_generated: bool = False) -> dict[str, list[str]]:
    """Aastha's entries format -> {class: [phrases]}. A spelling variant takes the class of the seed it
    names ("Variant associated with 'umm'"); generated candidates only when asked for."""
    by_phrase = {normalise(e["phrase"]): FUNCTION_TO_CLASS.get(e.get("function", "")) for e in entries
                 if _entry_kind(e) == "seed"}
    out: dict[str, list[str]] = {k: [] for k in CLASSES}
    for e in entries:
        kind = _entry_kind(e)
        if kind == "generated" and not include_generated:
            continue
        cls = FUNCTION_TO_CLASS.get(e.get("function", ""))
        if kind == "variant":
            m = _VARIANT_OF.search(str(e.get("example_context", "")))
            cls = by_phrase.get(normalise(m.group(1))) if m else None
        if cls:
            out[cls].append(e["phrase"])
    return out


def _read_lists(path: Path, include_generated: bool) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("entries"), list):
        return lists_from_entries(raw["entries"], include_generated)
    if isinstance(raw, dict) and any(k in raw for k in CLASSES):
        return {k: _flatten(raw.get(k, [])) for k in CLASSES}
    keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
    raise ValueError(f"{path}: unrecognised fillers format (keys {keys})")


@lru_cache(maxsize=8)  # warn once, parse once
def load_fillers(path: Path = config.FILLERS_PATH, extra: tuple[Path, ...] = config.FILLERS_EXTRA_PATHS,
                 include_generated: bool = config.FILLERS_INCLUDE_GENERATED) -> Fillers:
    """fillers.json (Aastha's entries format, or {class: [phrases]}) plus our extra files; the
    placeholder lists when fillers.json is missing. A class nobody supplies falls back to the placeholder."""
    if not path.exists():
        logger.warning(f"{path} missing; using the placeholder filler lists")
        return Fillers.from_lists(PLACEHOLDER_FILLERS)
    lists = _read_lists(path, include_generated)
    for x in extra:
        if x.exists():
            for k, v in _read_lists(x, include_generated).items():
                lists.setdefault(k, []).extend(v)
    for k in ("hesitation", "continuation", "completion"):
        if not lists.get(k):
            logger.warning(f"{path}: nothing maps to '{k}'; using the placeholder for it")
            lists[k] = PLACEHOLDER_FILLERS[k]
    fillers = Fillers.from_lists(lists)
    logger.info(f"fillers: {path.name} + {[x.name for x in extra if x.exists()]}, generated="
                f"{include_generated}: " + ", ".join(f"{k} {len(getattr(fillers, k))}" for k in CLASSES))
    return fillers


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
        """Longest filler phrase the text ends with; ties go hesitation > continuation > postposition > completion."""
        for n in range(min(self.max_tokens, len(toks)), 0, -1):
            end = tuple(toks[-n:])
            for cls in ("hesitation", "continuation", "postposition", "completion"):
                if end in getattr(self.fillers, cls):
                    return cls, end
        return None

    def classify(self, segment: str, context: DialogueContext = DialogueContext(), turn_text: str = "") -> TailResult:
        """`segment` is the newest text before the pause; `turn_text` everything said this turn, ending with it
        (defaults to it). The tail is read from the end of the turn, so a phrase split over two segments counts."""
        turn = tokens(turn_text) or tokens(segment)
        toks = turn
        if not toks:
            return TailResult("other", "")
        hit = self._ending(toks)
        waiting = bool(hit) and hit[0] in INCOMPLETE_CLASSES
        # Checked first: Aastha's data calls "haan"/"achha" hesitation ("don't end a turn on them if the
        # utterance is unfinished"), but a whole turn of just "haan" right after "... sahi?" is finished.
        if context.confirmation and (all(t in YES_NO_WORDS for t in turn) or tuple(turn) in self.fillers.confirmation
                                     or (len(turn) == 1 and not waiting)):
            return TailResult("short_answer", " ".join(turn[-self.max_tokens:]))
        if waiting:
            return TailResult(hit[0], " ".join(hit[1]))
        last = toks[-1]
        number = self._number_tail(last, turn, context)
        if number:
            return TailResult(number, " ".join(toks[-2:]))
        if hit:
            return TailResult(hit[0], " ".join(hit[1]))
        return TailResult("other", " ".join(toks[-self.max_tokens:]))

    @staticmethod
    def _numeric_context(turn: list[str], context: DialogueContext) -> str | None:
        """"id_in_progress" | "number_in_progress" | None: is a number or ID being read out here?"""
        if context.asked_field in config.ID_FIELDS:
            return "id_in_progress"
        if context.asked_field in config.QUANTITY_FIELDS:
            return "number_in_progress"
        # "machine ID chaar saat", "machine C 7": an ID cue followed only by ID-like words up to the pause
        i = len(turn) - 1
        while i >= 0 and _id_like(turn[i]):
            i -= 1
        return "id_in_progress" if 0 <= i < len(turn) - 1 and turn[i] in ID_CUES else None

    def _number_tail(self, last: str, turn: list[str], context: DialogueContext) -> str | None:
        """An ID or number still being said:
        - the last word announces an ID ("employee ID"), or follows an ID cue ("machine ID chaar", "machine C")
        - the last word is a number word ("chaar saat", "dhai sau"); "do"/"ek" only after another number or
          when an ID/quantity was asked ("paani bhijava do" is done)
        - a digit soundalike Prisma wrote ("machine number chaar saal"), only inside an ID/quantity read-out"""
        if last in ID_CUES_ALONE:
            return "id_in_progress"
        ctx = self._numeric_context(turn, context)
        prev_number = len(turn) >= 2 and is_number(turn[-2])
        if is_number(last):
            if last in config.AMBIGUOUS_NUMBER_WORDS and not (prev_number or ctx):
                return None
            return ctx or "number_in_progress"
        if last in config.NUMBER_SOUNDALIKES and ctx:  # "das saal" (ten years) is not a read-out
            return ctx
        return ctx if ctx == "id_in_progress" and _id_like(last) else None


def _id_like(tok: str) -> bool:
    """Can be part of an ID read-out: a number word, a digit soundalike, one letter, or a code ("p07")."""
    return (is_number(tok) or tok in config.NUMBER_SOUNDALIKES or (len(tok) == 1 and tok.isalpha())
            or bool(re.fullmatch(r"[a-z]{1,3}\d+", tok)))

if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(load_fillers().summary())
