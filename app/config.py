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

# --- Language / speech services ---------------------------------------------
LANGUAGE_CODE = _env("LANGUAGE_CODE", "en-IN")  # en-IN accepts Hinglish, outputs Latin script
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class PrismaSettings:
    min_silence_ms: int = 250  # short, so each phrase arrives as its own segment
    min_speech_ms: int = 100
    vad_threshold: float = 0.5
    output_format: str = "transcribe"  # spoken numbers -> digits
    bias_score: float = 2.0
    partials: bool = True  # x-partials: under-documented, see notes/decisions.md


PRISMA = PrismaSettings()

# pipecat-gnani 0.5.12 only accepts timbre-v2.0 / timbre-v2.5 (v3.0 raises ValueError), and
# `language` only on v2.5 from {auto, hi-IN, en-IN, ...}; "hi-en" is rejected. See decisions.md.
TTS_MODEL = _env("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = _env("TTS_VOICE", "Poorvi")
TTS_LANGUAGE = _env("TTS_LANGUAGE", "auto")
TTS_SAMPLE_RATE = 16_000

LLM_PROVIDER = _env("LLM_PROVIDER", "groq")
LLM_MODEL = _env("LLM_MODEL", "llama-3.3-70b-versatile")


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

# --- Turn taking -------------------------------------------------------------
TURN_MODE = _env("TURN_MODE", "smart_turn")  # fixed_timeout | smart_turn
FIXED_TIMEOUT_MS = int(_env("FIXED_TIMEOUT_MS", "700"))

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
