"""VAD and turn-taking, built from config, plus the startup report of what is active.

All three TURN_MODEs run through one stop strategy, app/turn/strategy.py
(PolicyTurnStopStrategy -> app/turn/controller.py), so every mode writes the
same per-pause log and the modes can be compared on the same data:

fixed_timeout:      the turn ends FIXED_TIMEOUT_MS after the last speech.
smart_turn:         ends when Smart Turn v3 says P(complete) > SMART_TURN_THRESHOLD,
                    else SMART_TURN_STOP_SECS after the last speech.
smart_turn_fillers: Smart Turn plus the tail-word rule (app/turn/turn_policy.py).

In every mode transcripts neither start nor end a turn by themselves (see notes/decisions.md).
"""

from __future__ import annotations

from collections.abc import Callable

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from app import config
from app.turn.controller import TURN_MODES, TurnController
from app.turn.smart_turn import ThresholdSmartTurn
from app.turn.strategy import PolicyTurnStopStrategy
from app.turn.tail_classifier import TailClassifier
from app.vad.pipecat_vad import CompositeVADAnalyzer, CustomVADAnalyzer

__all__ = ["TURN_MODES", "VAD_MODES", "ThresholdSmartTurn", "build_turn_strategies", "build_vad", "describe", "turn_strategy"]


VAD_MODES = ("silero", "custom", "composite")


def vad_params(mode: str = config.VAD_MODE) -> VADParams:
    if mode == "custom":
        return VADParams(confidence=config.CUSTOM_VAD_CONFIDENCE, start_secs=config.VAD_START_SECS,
                         stop_secs=config.VAD_STOP_SECS, min_volume=config.CUSTOM_VAD_MIN_VOLUME)
    return VADParams(confidence=config.SILERO_VAD_CONFIDENCE, start_secs=config.VAD_START_SECS, stop_secs=config.VAD_STOP_SECS)


def build_vad(mode: str = config.VAD_MODE) -> VADAnalyzer:
    """silero | custom (our DSP VAD) | composite (Silero AND ours). See app/vad/pipecat_vad.py."""
    if mode == "silero":
        return SileroVADAnalyzer(params=vad_params(mode))
    if mode == "custom":
        return CustomVADAnalyzer(params=vad_params(mode))
    if mode == "composite":
        return CompositeVADAnalyzer(params=vad_params(mode))
    raise ValueError(f"VAD_MODE must be one of {VAD_MODES}, got {mode!r}")


def build_turn_strategies(
    mode: str = config.TURN_MODE,
    *,
    fixed_timeout_ms: int = config.FIXED_TIMEOUT_MS,
    turn_analyzer: ThresholdSmartTurn | None = None,
    classifier: TailClassifier | None = None,
    log: Callable[..., None] | None = None,
) -> UserTurnStrategies:
    """Only VAD starts a turn; the TurnController in the configured mode ends it."""
    ctl = TurnController(mode=mode, fixed_timeout_ms=fixed_timeout_ms, classifier=classifier, log=log)
    if mode != "fixed_timeout" and turn_analyzer is None:
        turn_analyzer = ThresholdSmartTurn(params=SmartTurnParams(stop_secs=config.SMART_TURN_STOP_SECS))
    stop = PolicyTurnStopStrategy(ctl, turn_analyzer=turn_analyzer)
    return UserTurnStrategies(start=[VADUserTurnStartStrategy()], stop=[stop])


def turn_strategy(strategies: UserTurnStrategies) -> PolicyTurnStopStrategy:
    return next(s for s in strategies.stop if isinstance(s, PolicyTurnStopStrategy))


def describe(vad: VADAnalyzer, mode: str, strategies: UserTurnStrategies, prisma_headers: dict[str, str]) -> str:
    """Multi-line startup report: what is actually running, read back from the built objects."""
    p = vad.params
    s = turn_strategy(strategies)
    ctl, a, pol = s.controller, s.turn_analyzer, s.controller.policy
    lines = [
        "=== turn-taking config ===",
        f"VAD:        {type(vad).__name__} (VAD_MODE={config.VAD_MODE})",
        f"VADParams:  confidence={p.confidence} start_secs={p.start_secs} stop_secs={p.stop_secs} "
        f"min_volume={p.min_volume}",
        f"turn_mode:  {ctl.mode}",
        f"start:      {[type(x).__name__ for x in strategies.start]}",
        f"stop:       {type(s).__name__} -> TurnController",
    ]
    if ctl.mode == "fixed_timeout":
        lines.append(f"rule:       turn over {ctl.fixed_timeout_ms} ms after the last speech")
    elif ctl.mode == "smart_turn":
        lines.append(f"rule:       Smart Turn complete at VAD stop, else {ctl.smart_turn_stop_secs * 1000:.0f} ms "
                     "after the last speech")
    else:
        lines.append(
            f"rule:       segment budget {pol.segment_budget_ms} ms after VAD stop; waits completion/short "
            f"{pol.wait_completion_ms}/{pol.wait_short_answer_ms}, hesitation/connector/postposition "
            f"{pol.wait_hesitation_ms}/{pol.wait_continuation_ms}/{pol.wait_postposition_ms}, number/ID "
            f"{pol.wait_number_ms}/{pol.wait_id_ms}, no text {pol.wait_no_text_ms}, other: "
            f"ST>={pol.st_complete_prob} {pol.wait_st_complete_ms} / ST>={pol.st_unsure_prob} {pol.wait_st_unsure_ms} / "
            f"ST lower {pol.wait_st_incomplete_ms} / no ST {pol.wait_no_signal_ms}; cap {pol.max_wait_ms} ms"
        )
    if a is None:
        lines.append("Smart Turn: not used in this mode")
    else:
        loaded = getattr(a, "_session", None) is not None
        lines.append(f"Smart Turn: {type(a).__name__} loaded={loaded} threshold={a.threshold}")
    lines.append(
        f"Prisma:     format={config.PRISMA.output_format} lang={config.LANGUAGE_CODE} extra headers={prisma_headers}"
    )
    return "\n".join(lines)
