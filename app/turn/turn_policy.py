"""How long to wait after the last speech before the turn is over. Table-driven; numbers in config.TURN_POLICY.

A fast reply needs BOTH tone and words to say "finished": Smart Turn complete (P >= 0.8), the pause's
text has arrived, and its last 1-3 words aren't an incomplete ending.

    tail class / signal                                    wait (ms after the last speech)
    completion ("bas", "itna hi") / short_answer           400, whatever Smart Turn says
    hesitation / continuation / postposition               3500, whatever Smart Turn says
    number_in_progress / id_in_progress                    3000
    no text for the pause yet                              2500 (text arriving later re-decides)
    other + Smart Turn P >= 0.8                            400  (the only fast path without "bas")
    other + Smart Turn 0.5 <= P < 0.8, or no verdict       2500 (unsure)
    other + Smart Turn P < 0.5                             3000 (tone says not finished)
    hard cap                                               4000

Speech resuming at any point cancels the pause (TurnController.user_started).
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


def choose_wait(tail_class: str | None, st: SmartTurnVerdict | None, p: config.TurnPolicyConfig = config.TURN_POLICY,
                have_text: bool = True) -> Wait:
    """`tail_class` None = treated as "other". `have_text` False = no final or partial for the pause."""
    by_class = {
        "completion": p.wait_completion_ms,
        "short_answer": p.wait_short_answer_ms,
        "hesitation": p.wait_hesitation_ms,
        "continuation": p.wait_continuation_ms,
        "postposition": p.wait_postposition_ms,
        "number_in_progress": p.wait_number_ms,
        "id_in_progress": p.wait_id_ms,
    }
    if tail_class in by_class:
        ms, rule = by_class[tail_class], tail_class
    else:
        if st is None:
            ms, rule = p.wait_no_signal_ms, "other+no_smart_turn"
        elif st.probability >= p.st_complete_prob:
            ms, rule = p.wait_st_complete_ms, "other+smart_turn_complete"
        elif st.probability >= p.st_unsure_prob:
            ms, rule = p.wait_st_unsure_ms, "other+smart_turn_unsure"
        else:
            ms, rule = p.wait_st_incomplete_ms, "other+smart_turn_incomplete"
        # No text: never fast, and the longer of the two when the tone also says "not finished"
        # (a lone "umm": Prisma sends nothing, Smart Turn ~0.01 -> 3000, not 2500)
        if not have_text and p.wait_no_text_ms >= ms:
            ms, rule = p.wait_no_text_ms, "no_text"
    return Wait(min(ms, p.max_wait_ms), rule, ms > p.max_wait_ms)
