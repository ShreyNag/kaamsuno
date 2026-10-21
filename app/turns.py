"""VAD and turn-taking, built from config, plus the startup report of what is active.

fixed_timeout: the turn ends FIXED_TIMEOUT_MS after the last speech. Pipecat's
    SpeechTimeoutUserTurnStopStrategy starts its timer at the VAD stop event,
    which already comes VAD_STOP_SECS after the speech, so the timer gets the
    difference.
smart_turn: the turn ends only when Smart Turn v3 says P(complete) > threshold,
    or after SMART_TURN_STOP_SECS of silence (Smart Turn's own fallback).

In both modes transcripts neither start nor end a turn (see notes/decisions.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from app import config

TURN_MODES = ("fixed_timeout", "smart_turn")


class ThresholdSmartTurn(LocalSmartTurnAnalyzerV3):
    """Smart Turn v3 with a configurable P(complete) threshold (upstream hard-codes 0.5)."""

    def __init__(self, *, threshold: float = config.SMART_TURN_THRESHOLD, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        result = super()._predict_endpoint(audio_array)
        result["prediction"] = int(result["probability"] > self.threshold)
        return result


def vad_params() -> VADParams:
    return VADParams(confidence=config.VAD_CONFIDENCE, start_secs=config.VAD_START_SECS, stop_secs=config.VAD_STOP_SECS)


def build_vad() -> VADAnalyzer:
    if config.VAD_MODE != "silero":
        logger.warning(
            f"VAD_MODE={config.VAD_MODE!r} requested, but the custom/composite Pipecat wrappers "
            "(app/vad/pipecat_vad.py, Day 3) are not built yet. Using Silero."
        )
    return SileroVADAnalyzer(params=vad_params())


def build_turn_strategies(
    mode: str = config.TURN_MODE,
    *,
    fixed_timeout_ms: int = config.FIXED_TIMEOUT_MS,
    vad_stop_secs: float = config.VAD_STOP_SECS,
    turn_analyzer: LocalSmartTurnAnalyzerV3 | None = None,
) -> UserTurnStrategies:
    """Only VAD starts a turn; only the configured mode ends it."""
    if mode == "fixed_timeout":
        if fixed_timeout_ms / 1000 < vad_stop_secs:
            raise ValueError(f"FIXED_TIMEOUT_MS={fixed_timeout_ms} is shorter than VAD stop_secs={vad_stop_secs}")
        stop = SpeechTimeoutUserTurnStopStrategy(
            user_speech_timeout=fixed_timeout_ms / 1000 - vad_stop_secs, wait_for_transcript=False
        )
    elif mode == "smart_turn":
        analyzer = turn_analyzer or ThresholdSmartTurn(params=SmartTurnParams(stop_secs=config.SMART_TURN_STOP_SECS))
        stop = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer, wait_for_transcript=False)
    else:
        raise ValueError(f"TURN_MODE must be one of {TURN_MODES}, got {mode!r}")
    return UserTurnStrategies(start=[VADUserTurnStartStrategy()], stop=[stop])


def describe(vad: VADAnalyzer, mode: str, strategies: UserTurnStrategies, prisma_headers: dict[str, str]) -> str:
    """Multi-line startup report: what is actually running, read back from the built objects."""
    p = vad.params
    lines = [
        "=== turn-taking config ===",
        f"VAD:        {type(vad).__name__} (VAD_MODE={config.VAD_MODE})",
        f"VADParams:  confidence={p.confidence} start_secs={p.start_secs} stop_secs={p.stop_secs} "
        f"min_volume={p.min_volume}",
        f"turn_mode:  {mode}",
        f"start:      {[type(s).__name__ for s in strategies.start]}",
    ]
    for s in strategies.stop:
        if isinstance(s, SpeechTimeoutUserTurnStopStrategy):
            lines.append(
                f"stop:       SpeechTimeout user_speech_timeout={s._user_speech_timeout:.2f}s "
                f"(+ VAD stop {p.stop_secs}s = {(s._user_speech_timeout + p.stop_secs) * 1000:.0f} ms after last speech), "
                f"wait_for_transcript={s.wait_for_transcript}"
            )
            lines.append("Smart Turn: not used in this mode")
        elif isinstance(s, TurnAnalyzerUserTurnStopStrategy):
            a = s._turn_analyzer
            loaded = getattr(a, "_session", None) is not None
            lines.append(f"stop:       TurnAnalyzer wait_for_transcript={s.wait_for_transcript}")
            lines.append(
                f"Smart Turn: {type(a).__name__} loaded={loaded} threshold={getattr(a, 'threshold', 0.5)} "
                f"silence_fallback={a.params.stop_secs}s"
            )
    lines.append(
        f"Prisma:     format={config.PRISMA.output_format} lang={config.LANGUAGE_CODE} extra headers={prisma_headers}"
    )
    return "\n".join(lines)
