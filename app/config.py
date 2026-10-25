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
    partials: bool = True  # x-partials: under-documented, see notes/decisions.md
    # After our turn ends, how long to wait for Prisma's last segment of the turn.
    # smart_turn_fillers already waited for it before ending the turn, so this rarely runs out.
    tail_timeout_ms: int = int(_env("PRISMA_TAIL_TIMEOUT_MS", "800"))
    tail_grace_ms: int = 150  # extra wait past min_silence_ms for Prisma's "processing" message


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


# --- Interpreter / dialogue ----------------------------------------------------
LOW_CONFIDENCE = 0.6  # a critical field below this is asked about again
# At turn end, wait this long for in-flight interpreter calls, then ask from the report as it is
INTERPRETER_DRAIN_TIMEOUT_S = float(_env("INTERPRETER_DRAIN_TIMEOUT_S", "1.2"))
GREETING = "Namaste. Kya hua hai? Bataiye."
READBACK_PLACEHOLDER = "Shukriya, saari zaroori jaankari mil gayi. Report taiyaar hai."


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

# Which VAD drives Pipecat's user-speaking events: silero | custom | composite
VAD_MODE = _env("VAD_MODE", "composite")

# Pipecat VADParams for the Silero analyser. stop_secs is how long the VAD waits
# before reporting "stopped"; 0.2 is what Smart Turn and Pipecat's latency tables assume.
VAD_CONFIDENCE = float(_env("VAD_CONFIDENCE", "0.7"))
VAD_START_SECS = float(_env("VAD_START_SECS", "0.2"))
VAD_STOP_SECS = float(_env("VAD_STOP_SECS", "0.2"))

# --- Turn taking -------------------------------------------------------------
TURN_MODE = _env("TURN_MODE", "smart_turn_fillers")  # fixed_timeout | smart_turn | smart_turn_fillers
FIXED_TIMEOUT_MS = int(_env("FIXED_TIMEOUT_MS", "700"))  # silence after the last speech
SMART_TURN_THRESHOLD = float(_env("SMART_TURN_THRESHOLD", "0.5"))  # smart_turn mode: P(complete) above this ends the turn
SMART_TURN_STOP_SECS = float(_env("SMART_TURN_STOP_SECS", "3.0"))  # smart_turn mode: silence that ends the turn regardless
USER_TURN_STOP_TIMEOUT_S = 5.0  # Pipecat's last-resort turn end if no strategy fires


@dataclass(frozen=True)
class TurnPolicyConfig:
    """smart_turn_fillers: how long to wait after the last speech, by tail class (app/turn/turn_policy.py)."""

    # Wait for Prisma's segment for this pause, counted from the VAD stop (which is
    # already VAD_STOP_SECS after the last speech). Decide without it after this.
    # 680 = p90 of last speech -> segment (883 ms, n=92, old logs) - VAD stop 200 ms. See decisions.md.
    segment_budget_ms: int = int(_env("SEGMENT_BUDGET_MS", "680"))
    wait_completion_ms: int = 300
    wait_short_answer_ms: int = 300
    wait_hesitation_ms: int = 2000  # regardless of Smart Turn
    wait_continuation_ms: int = 2000  # regardless of Smart Turn
    wait_number_ms: int = 1500
    wait_st_complete_ms: int = 400  # tail "other" and Smart Turn P(complete) >= st_complete_prob
    wait_st_incomplete_ms: int = 1200
    wait_no_signal_ms: int = 800  # tail "other" and no Smart Turn verdict
    st_complete_prob: float = 0.7
    max_wait_ms: int = 2500  # hard cap after the last speech
    tail_tokens: int = 3  # the classifier looks at the last 1..tail_tokens tokens


TURN_POLICY = TurnPolicyConfig()
FILLERS_PATH = RESEARCH_DIR / "fillers.json"
# Fields whose answers are spoken digit by digit or as a number: a numeric tail means "more is coming"
ID_FIELDS = ("employee_id",)
QUANTITY_FIELDS = ("quantity", "people_affected")

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
