"""Every tunable in one place. Secrets come from .env; anything here can be
overridden by an env var of the same name (upper-case) where noted."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional for offline eval
    pass

ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / "research"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"  # reports/<session_id>.json, rewritten on every update


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# --- Secrets -----------------------------------------------------------------
GNANI_API_KEY = os.getenv("GNANI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# --- Language / speech services ---------------------------------------------
LANGUAGE_CODE = _env("LANGUAGE_CODE", "en-IN")  # en-IN accepts Hinglish, outputs Latin script
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class PrismaSettings:
    min_silence_ms: int = 250  # short, so each phrase arrives as its own segment
    min_speech_ms: int = 100
    vad_threshold: float = 0.5
    output_format: str = "transcribe"  # spoken numbers -> digits
    bias_score: float = 2.0  # one of 0.0, 0.5, 1.0, 1.5, 2.0
    # x-partials=true switches Prisma to partial_transcript/final_transcript messages
    # (eval/prisma_probe.py); app/stt.py reads both protocols. See notes/decisions.md.
    partials: bool = True
    # Each extra header can be switched off on its own (PRISMA_SEND_<NAME>=0), e.g. to bisect a live problem
    send_min_silence: bool = _env("PRISMA_SEND_MIN_SILENCE", "1") == "1"
    send_min_speech: bool = _env("PRISMA_SEND_MIN_SPEECH", "1") == "1"
    send_vad_threshold: bool = _env("PRISMA_SEND_VAD_THRESHOLD", "1") == "1"
    send_partials: bool = _env("PRISMA_SEND_PARTIALS", "1") == "1"
    send_bias: bool = _env("PRISMA_SEND_BIAS", "1") == "1"
    # Startup self-test: stream a fixture to Prisma and refuse to start if no transcript comes back
    self_test: bool = _env("PRISMA_SELF_TEST", "1") == "1"
    self_test_timeout_s: float = 5.0
    # After our turn ends, how long to wait for Prisma's last segment of the turn.
    # smart_turn_fillers already waited for it before ending the turn, so this rarely runs out.
    tail_timeout_ms: int = int(_env("PRISMA_TAIL_TIMEOUT_MS", "800"))
    tail_grace_ms: int = 150  # extra wait past min_silence_ms before checking for an open utterance


PRISMA = PrismaSettings()
BIAS_LIST_PLACEHOLDER = (
    "cement", "sariya", "rait", "gitti", "eent", "bori", "crane", "JCB", "mixer", "scaffolding", "manzil", "tower",
)

# pipecat-gnani 0.5.12 only accepts timbre-v2.0 / timbre-v2.5 (v3.0 raises ValueError), and
# `language` only on v2.5 from {auto, hi-IN, en-IN, ...}; "hi-en" is rejected. See decisions.md.
TTS_MODEL = _env("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = _env("TTS_VOICE", "Poorvi")
TTS_LANGUAGE = _env("TTS_LANGUAGE", "auto")
TTS_SAMPLE_RATE = 16_000

LLM_PROVIDER = _env("LLM_PROVIDER", "groq")
# Groq no longer serves the Llama 3.x chat models to our key (404 model_not_found); see decisions.md
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-oss-20b")
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT", "low")  # gpt-oss: low keeps a call well under 1 s
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "8"))
# Groq FREE tier. Limits are read from Groq's headers at run time (app/interpreter/groq.py), not set here.
# One call in flight per session; segments arriving meanwhile go out together in the next call, at least
# LLM_MIN_GAP_S after the previous call ended. Turn end sends what is left at once (no gap).
LLM_MIN_GAP_S = float(_env("LLM_MIN_GAP_S", "1.5"))
LLM_429_DEFAULT_WAIT_S = float(_env("LLM_429_DEFAULT_WAIT_S", "5"))  # a 429 without retry-after
# The pacer holds a call until remaining tokens >= last call's tokens x this. Groq 429'd with ~1,900 left
# for ~1,600-token calls (it reserves room for the completion), so 1.0 is not enough.
# Hop log keeps each interpreter call's user JSON and raw reply (a few hundred bytes each): lets a bad turn
# be explained afterwards. The system prompt is logged once per session as a hash (build_system_prompt rebuilds it).
LLM_LOG_PAYLOADS = _env("LLM_LOG_PAYLOADS", "1") == "1"
LLM_TOKEN_MARGIN = float(_env("LLM_TOKEN_MARGIN", "1.3"))
LLM_EVAL_MAX_TRIES = int(_env("LLM_EVAL_MAX_TRIES", "8"))  # batch evals: tries per uncached call (429s included)


# --- Interpreter / dialogue ----------------------------------------------------
LOW_CONFIDENCE = 0.6  # a critical field below this is asked about again
# At turn end, wait this long for in-flight interpreter calls, then ask from the report as it is
INTERPRETER_DRAIN_TIMEOUT_S = float(_env("INTERPRETER_DRAIN_TIMEOUT_S", "1.2"))
GREETING = "Namaste. Kya hua hai? Bataiye."
MAX_QUESTIONS = int(_env("MAX_QUESTIONS", "3"))  # field questions per report; then read back what we have
# A field is asked at most this often: the question, then once more ("_reask", or "_repeat" when nothing
# usable was heard). Then the dialogue moves on and the field is listed as unconfirmed.
MAX_ASKS_PER_FIELD = 2
# Words that name a field, so "location galat hai" at the read-back asks for the right location
FIELD_WORDS: dict[str, tuple[str, ...]] = {
    "location": ("location", "jagah", "jagaah", "area", "floor", "manzil", "bay", "tower", "block", "जगह", "मंजिल"),
    "urgency": ("urgency", "urgent", "jaldi", "zaroori", "जल्दी"),
    "quantity": ("quantity", "matra", "kitna", "kitni", "count", "मात्रा"),
    "unit": ("unit",),
    "material": ("material", "saamaan", "saman", "maal", "सामान"),
    "asset_or_equipment": ("machine", "equipment", "asset", "id", "मशीन"),
    "incident_type": ("type", "problem", "dikkat", "samasya"),
    "people_at_risk": ("log", "people", "aadmi", "लोग"),
}

# Report schema (research/schema.json, Aastha's KaamSuno v1.x format; see app/interpreter/schema.py)
SITE_NAME = _env("SITE_NAME", "Demo Site A")  # one site per demo session: recorded, never asked
# Required fields that are not asked as questions: collected at read-back, or filled from config
NOT_ASKED_REQUIRED = ("reporter_confirmation", "site_name")
# Filled by code, never by the interpreter LLM
SYSTEM_FIELDS = ("incident_id", "reported_at", "site_name", "spoken_number_raw", "missing_fields", "uncertainties",
                 "conflicts", "source_language", "transcript", "report_summary_spoken", "reporter_confirmation",
                 "status")
# Question order after the emergency check: location, incident_type, then the important fields for
# that incident type (TYPE_FIELDS), then description and urgency. A filled field is never asked.
ASK_FIRST = ("location", "incident_type")
ASK_LAST = ("description", "urgency")
TYPE_FIELDS: dict[str, tuple[str, ...]] = {
    "equipment_failure": ("asset_or_equipment",),  # and its ID: asked separately if the value has no digits
    "material_shortage": ("material", "quantity", "unit"),
    "safety_hazard": ("people_at_risk",),
    "injury_or_near_miss": ("people_at_risk",),
}
NEEDS_ID = ("asset_or_equipment",)  # a value without any digit gets one "what's its ID?" question
# Values that mean "not known": a field holding one of these counts as missing
UNKNOWN_VALUES = frozenset({"unknown", "not_obtained", "pata nahi", "n/a", "na", "none", "null", "-"})
# Urgency counts as stated only if its evidence has one of these words; otherwise the LLM guessed it
URGENCY_CUES = frozenset("""
urgent urgently jaldi jaldii abhi turant foran fauran immediately asap emergency critical serious priority
khatra khatarnak danger dangerous aag fire chot injury injured hurt bleeding khoon gas leak current shock
collapse gir gira giri girne electrocution spark sparks fatal maut
aaj kal later baad tak hurry koi hafte normal low high
तुरंत जल्दी अभी फौरन इमरजेंसी खतरा आग चोट खून गैस करंट गिर आज कल बाद
""".split())
# A different value for these is a conflict (both kept, confirm) unless the worker clearly corrects it;
# other fields (summaries, categories) are simply replaced
# Categories the LLM infers from what the worker described ("cement kam pad gaya" -> material_shortage):
# a guess ("stated": false) still counts as filled, unlike every other field, where a guess counts as missing
# (location, quantity, unit, asset ID, time, urgency...). They are asked only below LOW_CONFIDENCE or when vague.
# description too: it is the LLM's English summary of what the worker said, which it always flags
# "stated": false, so the bot asked "Thoda aur bataiye" after a complete report (decisions.md, 2026-10-31).
INFERRED_OK_FIELDS = ("incident_type", "assigned_team", "description")
# Values that say nothing yet: the field is asked as if missing
VAGUE_VALUES = {"incident_type": ("other",)}
CONFLICT_FIELDS = ("location", "quantity", "unit", "asset_or_equipment", "material", "date_or_time")
# "Never invent a location" in code: a location update's evidence must contain one of these place words
LOCATION_CUES = frozenset("""
location jagah jagaah manzil manjil floor mala mail level storey storey basement terrace chhat roof tower block wing building gate bay area
zone sector site road gali side paas pass near room shed yard godown store store-room lift staircase seedhi pillar
column grid plot phase unit hall gaddi entrance exit parking corridor platform scaffold scaffolding pit tank station
line section plant shop warehouse office canteen
मंजिल मंज़िल फ्लोर माला टावर ब्लॉक विंग गेट एरिया साइड पास रोड गली कमरा छत बेसमेंट सीढ़ी लिफ्ट गोदाम प्लांट
""".split())


# --- Custom VAD --------------------------------------------------------------
@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = SAMPLE_RATE
    frame_len: int = 400  # 25 ms
    hop: int = 160  # 10 ms
    n_fft: int = 512
    band_edges_hz: tuple[float, ...] = (0, 250, 500, 1000, 2000, 4000, 8000)

    # Running RMS (causal EMA), used for the absolute silence floor and plots
    rms_tau_s: float = 1.0
    abs_floor_dbfs: float = -60.0  # frames quieter than this are never speech

    # Feature A: adaptive energy gate
    gate_window_frames: int = 300  # ~3 s
    gate_floor_pct: float = 10.0
    gate_peak_pct: float = 90.0
    gate_alpha: float = 0.3  # threshold = floor + alpha * (peak - floor)
    gate_min_margin_db: float = 3.0  # threshold is at least this far above the floor

    # Feature B: spectral flatness over a speech band
    flatness_lo_hz: float = 100.0
    flatness_hi_hz: float = 4000.0
    flatness_smooth_bins: int = 5  # smooth the periodogram across frequency first
    flatness_max: float = 0.35  # votes speech when below

    # Feature C: peak concentration (max band PSD / sum of band PSDs)
    peak_conc_min: float = 0.45  # votes speech when above

    # Feature D: zero-crossing rate, combined with full-band flatness
    zcr_voiced_max: float = 0.12  # low ZCR -> voiced
    zcr_fricative_min: float = 0.25  # high ZCR -> candidate fricative...
    fricative_flatness_max: float = 0.5  # ...only if the full band is NOT flat (hiss is)

    # Decision logic (Day 2)
    min_votes: int = 2
    enter_frames: int = 3
    exit_frames: int = 1
    hangover_ms: int = 300


VAD = VADConfig()

# Which VAD drives Pipecat's user-speaking events: silero | custom | composite.
# Silero won the comparison on real recordings (eval/out/vad_results.md); custom and composite stay for the eval.
VAD_MODE = _env("VAD_MODE", "silero")

# Pipecat VADParams for the Silero analyser. stop_secs is how long the VAD waits
# before reporting "stopped"; 0.2 is what Smart Turn and Pipecat's latency tables assume.
# Silero speech-probability threshold (silero and composite). Tuning on F2 picked 0.3, but its labels are
# padded and the objective favours recall; 0.5 (Silero's own default) for live. See decisions.md.
SILERO_VAD_CONFIDENCE = float(_env("SILERO_VAD_CONFIDENCE", "0.5"))
VAD_START_SECS = float(_env("VAD_START_SECS", "0.2"))
VAD_STOP_SECS = float(_env("VAD_STOP_SECS", "0.2"))
# custom VAD (app/vad/pipecat_vad.py): confidence = share of 10 ms frames that are speech candidates.
# min_volume 0: our adaptive gate already handles level, and Pipecat's volume floor would cut soft speakers.
CUSTOM_VAD_CONFIDENCE = float(_env("CUSTOM_VAD_CONFIDENCE", "0.5"))
CUSTOM_VAD_MIN_VOLUME = float(_env("CUSTOM_VAD_MIN_VOLUME", "0.0"))

# --- Turn taking -------------------------------------------------------------
TURN_MODE = _env("TURN_MODE", "smart_turn_fillers")  # fixed_timeout | smart_turn | smart_turn_fillers
FIXED_TIMEOUT_MS = int(_env("FIXED_TIMEOUT_MS", "700"))  # silence after the last speech
SMART_TURN_THRESHOLD = float(_env("SMART_TURN_THRESHOLD", "0.5"))  # smart_turn mode: P(complete) above this ends the turn
SMART_TURN_STOP_SECS = float(_env("SMART_TURN_STOP_SECS", "3.0"))  # smart_turn mode: silence that ends the turn regardless
USER_TURN_STOP_TIMEOUT_S = 5.0  # Pipecat's last-resort turn end if no strategy fires


@dataclass(frozen=True)
class TurnPolicyConfig:
    """smart_turn_fillers: how long to wait after the last speech (app/turn/turn_policy.py).

    A fast reply needs BOTH tone and words to say "finished": Smart Turn P(complete) >= st_complete_prob,
    the pause's text has arrived, and its last 1-3 words aren't an incomplete ending. See decisions.md.
    """

    # Wait for Prisma's segment for this pause, counted from the VAD stop (which is
    # already VAD_STOP_SECS after the last speech). Decide without it after this.
    # 680 = p90 of last speech -> segment (883 ms, n=92, old logs) - VAD stop 200 ms. See decisions.md.
    segment_budget_ms: int = int(_env("SEGMENT_BUDGET_MS", "680"))
    wait_completion_ms: int = int(_env("WAIT_COMPLETION_MS", "400"))  # "bas", "itna hi", "that's it"
    wait_short_answer_ms: int = int(_env("WAIT_SHORT_ANSWER_MS", "400"))  # "haan sahi hai" after "... sahi?"
    wait_hesitation_ms: int = int(_env("WAIT_HESITATION_MS", "3500"))  # "umm", "matlab", "ek second"
    wait_continuation_ms: int = int(_env("WAIT_CONTINUATION_MS", "3500"))  # connectors: "aur", "kyunki", "because"
    wait_postposition_ms: int = int(_env("WAIT_POSTPOSITION_MS", "3500"))  # "pe", "ke paas", "wala"
    wait_number_ms: int = int(_env("WAIT_NUMBER_MS", "3000"))  # a number word at the end ("chaar saat")
    wait_id_ms: int = int(_env("WAIT_ID_MS", "3000"))  # after "ID", "number", "machine" ("machine ID chaar")
    wait_st_complete_ms: int = int(_env("WAIT_ST_COMPLETE_MS", "400"))  # P >= st_complete_prob, text in, tail ok
    st_complete_prob: float = float(_env("ST_COMPLETE_PROB", "0.8"))
    st_unsure_prob: float = float(_env("ST_UNSURE_PROB", "0.5"))  # [unsure, complete): Smart Turn unsure
    wait_st_unsure_ms: int = int(_env("WAIT_ST_UNSURE_MS", "2500"))
    # P < st_unsure_prob: the tone says "not finished" though the words don't. Not in the brief; between
    # "unsure" (2.5 s) and "incomplete ending" (3.5 s).
    wait_st_incomplete_ms: int = int(_env("WAIT_ST_INCOMPLETE_MS", "3000"))
    wait_no_signal_ms: int = int(_env("WAIT_NO_SIGNAL_MS", "2500"))  # no Smart Turn verdict: treated as unsure
    # No final and no partial for the pause by the segment budget: text arriving later re-decides
    wait_no_text_ms: int = int(_env("WAIT_NO_TEXT_MS", "2500"))
    max_wait_ms: int = int(_env("TURN_MAX_WAIT_MS", "4000"))  # hard cap after the last speech
    tail_tokens: int = 3  # the classifier looks at the last 1..tail_tokens words of the turn


# Incomplete endings (the tail classifier, app/turn/tail_classifier.py). Matched on the last 1-3 words of
# the turn's newest text, Latin and Devanagari. These are unioned with Aastha's research/fillers.json.
TURN_HESITATION = (
    "um", "umm", "uh", "uhh", "hmm", "aa", "aah", "er", "erm", "matlab", "mtlb", "woh", "wo", "yaani",
    "kya bolte hain", "kya kehte hain", "kya bolte", "ek second", "ek sec", "ek minute", "ruko", "ruk",
    "उम", "उम्म", "अं", "आ", "मतलब", "वो", "यानी", "क्या बोलते हैं", "क्या कहते हैं", "एक सेकंड", "एक मिनट", "रुको",
)
TURN_CONNECTORS = (
    "aur", "toh", "to", "phir", "fir", "lekin", "magar", "par", "ya", "kyunki", "kyonki", "jab", "agar", "ki",
    "jaise", "jaise ki", "ki wajah se", "isliye", "and", "but", "because", "so", "or", "if", "when", "then",
    "like", "such as", "the", "a", "of", "with",
    "और", "तो", "फिर", "लेकिन", "मगर", "पर", "या", "क्योंकि", "जब", "अगर", "कि", "जैसे", "इसलिए",
)
# Postpositions and particles that can't end a sentence. "pay"/"pai": Prisma's spelling of "pe"
# (live 2026-10-31, "yaar 2nd floor pay" cut in).
TURN_POSTPOSITIONS = (
    "ke", "ka", "ki", "mein", "me", "pe", "pay", "pai", "par", "se", "ko", "tak", "wala", "wali", "wale", "jo",
    "ke paas", "ke pass", "ke upar", "ke neeche", "ke andar", "ke bahar", "ke saamne", "ke peeche", "ke baad",
    "ke liye", "ki taraf", "ki wajah", "of the", "near the", "at the", "in the", "on the", "near",
    "के", "का", "की", "में", "पे", "पर", "से", "को", "तक", "वाला", "वाली", "वाले", "जो",
    "के पास", "के ऊपर", "के नीचे", "के अंदर", "के बाहर", "के सामने", "के पीछे", "के बाद", "के लिए",
)
TURN_COMPLETION = (
    "bas", "itna hi", "bas itna hi", "bas itna", "itna hi hai", "ho gaya", "bas yahi", "khatam", "done",
    "thats it", "that is it", "thats all", "bas itni si baat hai",
    "बस", "इतना ही", "बस इतना ही", "हो गया", "बस यही", "खत्म",
)
# A tail right after one of these (or the word itself) is an ID being read out: "machine ID chaar ..."
TURN_ID_CUES = ("id", "aaidee", "aidi", "number", "numbar", "no", "machine", "mashin", "asset",
                "आईडी", "नंबर", "नम्बर", "मशीन")
TURN_ID_CUES_NEED_FOLLOWER = ("no", "machine", "mashin", "asset", "मशीन")  # these end sentences too
# Number words that are also common words ("do" = give, "ek" = a): a number only after another number or
# an ID cue, or when the bot asked for an ID or quantity ("paani bhijava do" is a finished request)
AMBIGUOUS_NUMBER_WORDS = frozenset({"do", "ek", "दो", "एक"})


TURN_POLICY = TurnPolicyConfig()
FILLERS_PATH = RESEARCH_DIR / "fillers.json"
# Ours, same entry format, loaded on top of fillers.json (marked for Aastha to review)
FILLERS_EXTRA_PATHS = (RESEARCH_DIR / "fillers_extra_ours.json", RESEARCH_DIR / "fillers_devanagari_extra.json")
# Her 776 "generated candidate" phrases are long templated annotation prompts, not tail cues: off by default
FILLERS_INCLUDE_GENERATED = _env("FILLERS_INCLUDE_GENERATED", "0") == "1"
# Fields whose answers are spoken digit by digit or as a number: a numeric tail means "more is coming"
ID_FIELDS = ("asset_or_equipment", "reporter_name_or_id")
# Prisma's spellings of digits it misheard ("saat" -> "saal", "nau" -> "no"): count as numbers, but only
# while a number is being read out (after a number in an ID read-out, or when an ID/quantity was asked)
NUMBER_SOUNDALIKES = frozenset("saal sat saath sath no nao now naw tin aat panj".split())
QUANTITY_FIELDS = ("quantity",)

# Experiment: blank non-speech audio before sending to Prisma (off in v0)
BLANK_NONSPEECH_TO_STT = _env("BLANK_NONSPEECH_TO_STT", "0") == "1"


# --- Eval --------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    snrs_db: tuple[float, ...] = (10.0, 5.0)
    noise_types: tuple[str, ...] = ("drilling", "generator", "traffic", "hammering", "crowd")
    seed: int = 1234
    webrtc_aggressiveness: tuple[int, ...] = (0, 1, 2, 3)
    webrtc_frame_ms: int = 30
    silero_threshold: float = 0.5
    labels_csv: Path = field(default=DATA_DIR / "vad_dev" / "vad_labels.csv")


EVAL = EvalConfig()
