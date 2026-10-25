"""How long to wait after the last speech before the turn is over. Table-driven; numbers in config.TURN_POLICY.

    tail class                          wait (ms after the last speech)
    completion / short_answer           300
    hesitation / continuation           2000, whatever Smart Turn says
    number_in_progress                  1500
    other + Smart Turn complete (>=0.7) 400
    other + Smart Turn incomplete       1200
    other + no Smart Turn verdict       800
    hard cap                            2500
"""

from __future__ import annotations

from dataclasses import dataclass

from app import config


@dataclass(frozen=True)
class SmartTurnVerdict:
    complete: bool  # the analyzer's own verdict (its threshold)
    probability: float  # P(complete)
    infer_ms: float = 0.0


@dataclass(frozen=True)
class Wait:
    ms: int  # after the last speech, already capped
    rule: str  # which table row chose it, for the pause log
    capped: bool = False


def choose_wait(tail_class: str | None, st: SmartTurnVerdict | None, p: config.TurnPolicyConfig = config.TURN_POLICY) -> Wait:
    """`tail_class` None = no segment arrived within the budget (treated as "other")."""
    by_class = {
        "completion": p.wait_completion_ms,
        "short_answer": p.wait_short_answer_ms,
        "hesitation": p.wait_hesitation_ms,
        "continuation": p.wait_continuation_ms,
        "number_in_progress": p.wait_number_ms,
    }
    if tail_class in by_class:
        ms, rule = by_class[tail_class], tail_class
    elif st is None:
        ms, rule = p.wait_no_signal_ms, "other+no_smart_turn"
    elif st.probability >= p.st_complete_prob:
        ms, rule = p.wait_st_complete_ms, "other+smart_turn_complete"
    else:
        ms, rule = p.wait_st_incomplete_ms, "other+smart_turn_incomplete"
    return Wait(min(ms, p.max_wait_ms), rule, ms > p.max_wait_ms)
