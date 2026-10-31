"""What the bot says at the end of each worker turn. Templates only, no LLM.

DialogueManager keeps the conversation state; one `decide()` per turn end:

    emergency (urgency == "emergency")      -> alert line once, no questions, then the read-back
    collecting:
        nothing captured yet                -> opening question
        next pending field                  -> ask it (order below)
        nothing pending, or the cap reached -> read-back                         (phase "readback")
    readback (every reply goes through the interpreter first, as a possible correction):
        a field changed, or was re-stated   -> re-read only those fields         ("nahi, Bay 3")
        yes                                 -> confirmed line                    (phase "confirmed")
        names a field                       -> "what is the right location?"     ("location galat hai")
        no                                  -> "which detail is wrong?"
        no yes/no and no field content      -> short "is it right? haan ya nahi"
    confirmed:
        the worker changed something        -> re-read the changes (back to "readback")
        otherwise                           -> say nothing

Pending fields, in order: the required ones (config.ASK_FIRST, then ASK_LAST), then the important
fields for the incident type (config.TYPE_FIELDS), then any other critical field (guard). Within
that order: missing (null, "unknown" or inferred) -> an ID for config.NEEDS_ID fields -> conflict ->
low confidence. MAX_QUESTIONS caps everything except missing required fields (the guard: never read
back as done while a required field could still be asked).

A field is asked at most MAX_ASKS_PER_FIELD (2) times: the question, then once more if the answer
didn't fill it: "_repeat" ("not clear, say again") only when nothing usable was heard, otherwise
"_reask" ("just tell me the location"). Then the dialogue moves on and the field is `unconfirmed`
(listed in uncertainties, read back as "not given"). The same line is never spoken twice in a row.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app import config
from app.interpreter.interpreter import Report, is_known
from app.interpreter.schema import Schema
from app.turn.tail_classifier import load_fillers, tokens

YES_WORDS = frozenset("""haan han haa ha ji sahi theek thik correct yes yeah yep ok okay bilkul right pakka confirm
हाँ हां हा जी सही ठीक बिल्कुल""".split())
NO_WORDS = frozenset("nahi nahin nai nhi na galat no nope wrong नहीं नही ना गलत".split())
CONFIRM_REASONS = ("low_confidence", "conflict", "readback", "reread", "readback_prompt")


@dataclass(frozen=True)
class Question:
    field: str | None  # the field asked about; None for opening / emergency / read-back moves
    reason: str  # emergency|opening|missing|missing_id|conflict|low_confidence|repeat|reask|readback|
    #               reread|readback_prompt|which_value|rejection|confirmed|silent
    text: str  # "" when silent

    @property
    def silent(self) -> bool:
        return self.reason == "silent"

    @property
    def expects_yes_no(self) -> bool:
        return self.reason in CONFIRM_REASONS


def _words(text: str) -> set[str]:
    return set(re.findall(r"[\wऀ-ॿ]+", text.lower()))


def is_no(text: str) -> bool:
    return bool(_words(text) & NO_WORDS)


def is_yes(text: str) -> bool:
    return bool(_words(text) & YES_WORDS) and not is_no(text)


def usable(text: str) -> bool:
    """Anything heard besides fillers ("umm", "matlab")? Only an empty answer earns "please repeat"."""
    hesitation = {p[0] for p in load_fillers().hesitation if len(p) == 1}
    return any(t not in hesitation for t in tokens(text))


def named_fields(text: str) -> list[str]:
    """Fields the worker names ("location galat hai" -> ["location"]), in config.FIELD_WORDS order."""
    words = _words(text)
    return [name for name, cues in config.FIELD_WORDS.items() if words & set(cues)]


def _label(name: str) -> str:
    return name.replace("_", " ")


def ask_order(report: Report, schema: Schema) -> list[str]:
    itype = report.get("incident_type", {}).get("value") if is_known(report.get("incident_type", {})) else None
    out = [n for n in (*config.ASK_FIRST, *config.ASK_LAST) if schema.has(n)]
    out += [n for n in config.TYPE_FIELDS.get(itype, ()) if schema.has(n) and n not in out]
    return out + [n for n in schema.critical_priority if n not in out]


def pending(report: Report, schema: Schema, low_confidence: float = config.LOW_CONFIDENCE) -> list[tuple[str, str]]:
    """Every (field, reason) the dialogue could ask about now, in priority order."""
    order = ask_order(report, schema)
    out = [(n, "missing") for n in order if not is_known(report[n])
           or report[n]["value"] in config.VAGUE_VALUES.get(n, ())]  # incident_type "other": ask what happened
    out += [(n, "missing_id") for n in order if n in config.NEEDS_ID and is_known(report[n])
            and not re.search(r"\d", str(report[n]["value"]))]
    out += [(n, "conflict") for n in order if report[n].get("conflicts")]
    out += [(n, "low_confidence") for n in order if is_known(report[n]) and report[n]["confidence"] < low_confidence]
    return out


def captured_anything(report: Report, schema: Schema) -> bool:
    return any(is_known(report[f.name]) for f in schema.llm_fields)


@dataclass
class DialogueManager:
    low_confidence: float = config.LOW_CONFIDENCE
    max_questions: int = config.MAX_QUESTIONS
    phase: str = "collecting"  # collecting | readback | confirmed
    confirmation: str = "not_obtained"  # her reporter_confirmation enum
    asked: Counter = field(default_factory=Counter)  # (field, reason) -> times asked
    questions: int = 0  # field questions asked (the cap counts these)
    unconfirmed: set = field(default_factory=set)  # asked the maximum times and still not filled
    last_text: str = ""
    emergency_announced: bool = False
    _corrected: bool = False

    def decide(self, report: Report, schema: Schema, templates, turn_text: str = "",
               changed: dict[str, object] | None = None, touched: set[str] | None = None) -> Question:
        """`changed`: fields whose stated value is new since the bot last spoke. `touched`: fields the
        worker's words updated this turn, even to the same value (a re-statement)."""
        q = self._decide(report, schema, templates, turn_text, changed or {}, touched or set())
        if q.text and q.text == self.last_text:  # never the same line twice in a row
            q = Question(None, "silent", "")
        if q.text:
            self.last_text = q.text
        return q

    # --- the decision ---------------------------------------------------------------
    def _decide(self, report, schema, t, turn_text, changed, touched) -> Question:
        urgency = report.get("urgency", {})
        if is_known(urgency) and urgency["value"] == "emergency" and not self.emergency_announced:
            self.emergency_announced = True
            return Question(None, "emergency", t["_emergency"])
        if self.phase in ("readback", "confirmed"):
            return self._after_readback(report, schema, t, turn_text, changed, touched)
        if not captured_anything(report, schema):
            return Question(None, "opening", t["_opening"])
        if not self.emergency_announced:
            q = self._next_field_question(report, schema, t, turn_text)
            if q:
                return q
        return self._readback(report, schema, t)

    def _next_field_question(self, report, schema, t, turn_text: str) -> Question | None:
        critical = set(schema.critical_priority)
        for name, reason in pending(report, schema, self.low_confidence):
            if self.asked[(name, reason)] >= config.MAX_ASKS_PER_FIELD:
                if reason in ("missing", "missing_id"):
                    self.unconfirmed.add(name)
                continue
            if self.questions >= self.max_questions and not (reason == "missing" and name in critical):
                continue
            text, ask = self._render(name, reason, report, t), reason
            if self.asked[(name, reason)]:  # the answer didn't fill it: once more, worded for what we heard
                ask = "reask" if usable(turn_text) else "repeat"
                text = t["_reask"].format(field_label=_label(name)) if ask == "reask" else t["_repeat"]
            self.asked[(name, reason)] += 1
            self.questions += 1
            return Question(name, ask, text)
        return None

    def _render(self, name: str, reason: str, report: Report, t) -> str:
        entry = report[name]
        if reason == "missing":
            return t.get(name) or t["_generic"].format(field_label=_label(name))
        if reason == "missing_id":
            return t["_asset_id"].format(asset=entry["value"])
        if reason == "conflict":
            values = [entry["value"], *(c["value"] for c in entry["conflicts"])]
            return t["_conflict"].format(field_label=_label(name), values=" ya ".join(map(str, values)))
        return t["_low_confidence"].format(field_label=_label(name), value=entry["value"])

    # --- read-back ------------------------------------------------------------------
    def _readback(self, report, schema, t) -> Question:
        self.phase = "readback"
        intro, _, ask = t["_readback"].partition(". ")
        body = ", ".join(self._summary(report, schema, t))
        return Question(None, "readback", f"{intro.rstrip('.')}: {body}. {ask or t['_readback_prompt']}".strip())

    def _summary(self, report: Report, schema: Schema, t) -> list[str]:
        parts = []
        unit = report.get("unit", {})
        order = ask_order(report, schema)  # read back: where and what first, then details, urgency last
        order = [n for n in order if n != "urgency"] + [n for n in order if n == "urgency"]
        for name in order:
            e = report[name]
            if name in ("description", "unit"):
                continue  # description is a free-text paraphrase; unit goes with the quantity
            if not is_known(e):
                if name in schema.critical_priority or name in self.unconfirmed:
                    parts.append(t["_not_given"].format(field_label=_label(name)))
                continue
            v = str(e["value"])
            if name == "quantity" and is_known(unit):
                v = f"{v} {unit['value']}"
            elif name == "incident_type":
                v = _label(v)
            elif name == "urgency":
                v = f"urgency {v}"
            elif name == "people_at_risk":
                v = f"{_label(name)}: {v}"
            parts.append(v)
        return parts

    def _after_readback(self, report, schema, t, turn_text, changed, touched) -> Question:
        if changed:
            self.phase, self._corrected = "readback", True
            return self._reread(changed, t)
        if self.phase == "confirmed":
            return Question(None, "silent", "")
        if is_yes(turn_text):
            self.phase = "confirmed"
            self.confirmation = "corrected_then_confirmed" if self._corrected else "confirmed"
            return Question(None, "confirmed", t["_confirmed"])
        restated = {n: report[n]["value"] for n in sorted(touched) if n in report and is_known(report[n])}
        if restated:  # "location Bay 2 hai": the same value, said again
            return self._reread(restated, t)
        named = [n for n in named_fields(turn_text) if schema.has(n)]
        if named:  # "location galat hai": ask for the right value; the answer comes back as a change
            return Question(named[0], "which_value", t["_which_value"].format(field_label=_label(named[0])))
        if is_no(turn_text):
            return Question(None, "rejection", t["_rejection"])
        return Question(None, "readback_prompt", t["_readback_prompt"])

    def _reread(self, fields: dict[str, object], t) -> Question:
        items = ", ".join(f"{_label(n)} {v}" for n, v in fields.items())
        return Question(next(iter(fields)), "reread", t["_reread"].format(changes=items))


def next_question(report: Report, schema: Schema, templates, low_confidence: float = config.LOW_CONFIDENCE) -> Question:
    """The first decision of a fresh conversation (stateless helper)."""
    return DialogueManager(low_confidence=low_confidence).decide(report, schema, templates)


def top_missing_field(report: Report, schema: Schema, low_confidence: float = config.LOW_CONFIDENCE) -> tuple[str, str] | None:
    p = pending(report, schema, low_confidence)
    return p[0] if p else None
