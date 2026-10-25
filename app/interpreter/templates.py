"""Question texts per (key, language), rotated: built from Aastha's research/dialogue_scripts.json.

    key        a schema field ("location", "quantity", "asset_or_equipment", ...) or a dialogue
               move ("_opening", "_emergency", "_rejection", "_confirmed", "_repeat", "_low_confidence", ...)
    language   "hi" | "hinglish" | "en"  (her scripts' labels; "hi" is romanised Hindi, which Timbre reads)

Her generated scripts (DIA021-DIA060) give one response per (intent, language). The four intents whose
generated `bot_response` is a copy-paste error (filler_pause, confirmation, rejection, emergency; see
notes/decisions.md) are replaced by OURS below. Keys she has no script for keep the placeholder
templates (Hinglish) in schema.PLACEHOLDER_TEMPLATES.
"""

from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from loguru import logger

from app import config
from app.interpreter.schema import PLACEHOLDER_TEMPLATES

SCRIPTS_PATH = config.RESEARCH_DIR / "dialogue_scripts.json"
LANGUAGES = ("hi", "hinglish", "en")

# her intent -> our template key
INTENT_TO_KEY = {
    "opening": "_opening",
    "missing_location": "location",
    "missing_quantity": "quantity",
    "missing_asset_id": "asset_or_equipment",
    "readback": "_readback",
    "unclear_audio": "_repeat",
}
# Intents whose generated responses are wrong in v1.1, and the scripts that carry the error
BROKEN_INTENTS = ("filler_pause", "confirmation", "rejection", "emergency")
BROKEN_SCRIPT_IDS = tuple(f"DIA{n:03d}" for r in ((27, 31), (37, 41), (47, 51), (57, 61)) for n in range(*r))

# OURS, replacing the broken intents. Emergency follows DIA014's pattern (hazard, keep safe distance,
# supervisor alerted, urgent report). filler_pause has no text: the bot says nothing and keeps listening.
OURS: dict[str, dict[str, list[str]]] = {
    "_emergency": {
        "hinglish": ["Yeh urgent safety issue ho sakta hai. Safe distance rakhein aur supervisor ko abhi alert "
                     "karein. Main urgent report bana raha hoon."],
        "hi": ["Yeh khatre ki baat hai. Surakshit doori rakhiye aur supervisor ko abhi bataiye. "
               "Main turant urgent report bana raha hoon."],
        "en": ["This may be an urgent safety issue. Keep a safe distance and alert your supervisor now. "
               "I am making an urgent report."],
    },
    "_rejection": {
        "hinglish": ["Theek hai, kaunsi detail galat hai? Woh bata dijiye."],
        "hi": ["Achha, kaunsi baat galat hai? Kripya woh bataiye."],
        "en": ["Okay, which detail is wrong? Tell me and I'll fix it."],
    },
    "_confirmed": {  # after the read-back is approved; sending to the supervisor is Phase 3, so not "bhej diya"
        "hinglish": ["Theek hai, report confirm ho gayi. Dhanyavaad."],
        "hi": ["Theek hai, report pakki ho gayi. Dhanyavaad."],
        "en": ["Okay, the report is confirmed. Thank you."],
    },
    # Keys her scripts don't cover: hi/en versions of the Hinglish placeholders (which stay the hinglish text)
    "incident_type": {"hi": ["Kya hua hai? Machine kharab hai, saamaan kam hai, ya koi khatra hai?"],
                      "en": ["What happened? A machine fault, a material shortage, or a hazard?"]},
    "description": {"hi": ["Thoda aur bataiye, hua kya?"], "en": ["Tell me a bit more. What happened?"]},
    "urgency": {"hi": ["Kitna zaroori hai? Abhi turant, ya aaj ke andar?"],
                "en": ["How urgent is it? Right now, or later today?"]},
    "unit": {"hi": ["Kis unit mein? Bori, kilo, ya piece?"], "en": ["In what unit? Bags, kilos, or pieces?"]},
    "_low_confidence": {"hi": ["Ek baar pakka kar dijiye: {field_label} {value} hai, sahi?"],
                        "en": ["Just to confirm: {field_label} is {value}, right?"]},
    "_conflict": {"hi": ["{field_label} ke liye do baatein suni: {values}. Kaunsi sahi hai?"],
                  "en": ["I heard two values for {field_label}: {values}. Which one is right?"]},
    "material": {"hi": ["Kaunsa saamaan hai?"], "en": ["Which material is it?"]},
    "people_at_risk": {"hi": ["Kya koi aur log khatre mein hain? Kitne log?"],
                       "en": ["Is anyone else at risk? How many people?"]},
    "_asset_id": {"hi": ["{asset} ka ID ya number kya hai?"], "en": ["What is the {asset}'s ID or number?"]},
    "_readback_prompt": {"hi": ["Kya report sahi hai? Haan ya nahi boliye."],
                         "en": ["Is the report correct? Please say yes or no."]},
    "_reread": {"hi": ["Achha, {changes}. Sahi hai?"], "en": ["Okay, {changes}. Is that right?"]},
    "_not_given": {"hi": ["{field_label} nahi pata"], "en": ["{field_label} not given"]},
}

# Common English words, to tell English from Hinglish from (romanised) Hindi
_EN = set("""
the a an is are was were be been has have had do does did not no yes it its this that these those there here
i we you he she they me my our your his her their what where when which who how why and or but if then so
of to in on at for with from by up down out off over under near some any all more most one two three four five
machine pump motor crane leak leaking broken stopped working fire oil water cement bags bag need needs please
site area floor gate bay block tower report issue problem stuck urgent help someone hurt injured sir
""".split())


def detect_language(text: str) -> str:
    """'hi' (Devanagari, or romanised with few English words) | 'en' | 'hinglish'. Default hinglish."""
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return "hinglish"
    share = sum(w in _EN for w in words) / len(words)
    return "en" if share >= 0.6 else "hi" if share <= 0.1 else "hinglish"


class TemplateBook:
    """texts[key][language] -> list, rotated per (key, language). Falls back to hinglish, then the placeholder."""

    def __init__(self, texts: dict[str, dict[str, list[str]]]):
        self.texts = texts
        self._cycles: dict[tuple[str, str], itertools.cycle] = {}

    def get(self, key: str, language: str = "hinglish") -> str | None:
        """Requested language, then Hinglish, then the (Hinglish) placeholder, then any language."""
        langs = self.texts.get(key, {})
        for lang in (language, "hinglish"):
            if langs.get(lang):
                return next(self._cycles.setdefault((key, lang), itertools.cycle(langs[lang])))
        if key in PLACEHOLDER_TEMPLATES:
            return PLACEHOLDER_TEMPLATES[key]
        for lang in LANGUAGES:
            if langs.get(lang):
                return next(self._cycles.setdefault((key, lang), itertools.cycle(langs[lang])))
        return None

    def for_language(self, language: str) -> _LanguageView:
        """A mapping for dialogue.next_question; a text is drawn (and its rotation advanced) only when used."""
        return _LanguageView(self, language)


class _LanguageView:
    def __init__(self, book: TemplateBook, language: str):
        self._book, self.language = book, language

    def __getitem__(self, key: str) -> str:
        text = self._book.get(key, self.language)
        if text is None:
            raise KeyError(key)
        return text

    def get(self, key: str, default: str | None = None) -> str | None:
        text = self._book.get(key, self.language)
        return default if text is None else text


def _clean(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("—", ", ").replace("’", "'")).strip()


def load_template_book(path: Path = SCRIPTS_PATH) -> TemplateBook:
    texts: dict[str, dict[str, list[str]]] = defaultdict(dict)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        scripts = raw.get("scripts") if isinstance(raw, dict) else None
        if not isinstance(scripts, list):
            raise ValueError(f"{path}: expected {{'scripts': [...]}}, got keys {sorted(raw) if isinstance(raw, dict) else type(raw)}")
        for s in scripts:
            key = INTENT_TO_KEY.get(s.get("intent"))
            lang = s.get("language")
            sid = str(s.get("script_id", ""))
            hand_written = sid[3:].isdigit() and int(sid[3:]) <= 20  # DIA001-020 answer one specific utterance
            if key is None or lang not in LANGUAGES or sid in BROKEN_SCRIPT_IDS or hand_written:
                continue
            resp = _clean(s.get("bot_response", ""))
            if resp and resp not in texts[key].setdefault(lang, []):
                texts[key][lang].append(resp)
        logger.info(f"dialogue scripts: {path.name}: " + ", ".join(f"{k} {sorted(v)}" for k, v in sorted(texts.items())))
    else:
        logger.warning(f"{path} missing; question texts are the placeholders")
    for key, langs in OURS.items():
        for lang, items in langs.items():
            texts[key].setdefault(lang, list(items))
    return TemplateBook(dict(texts))
