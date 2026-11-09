"""Table-driven tests for the wait policy (config defaults): a fast reply needs tone AND words."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app import config
from app.turn.turn_policy import SmartTurnVerdict, choose_wait

ST_DONE = SmartTurnVerdict(True, 0.98)
ST_EDGE = SmartTurnVerdict(True, 0.8)  # exactly the fast-reply threshold
ST_UNSURE = SmartTurnVerdict(True, 0.7)  # the analyzer's 0.5 threshold says complete; the policy says unsure
ST_NOT = SmartTurnVerdict(False, 0.1)


@pytest.mark.parametrize("tail_class, st, ms, rule", [
    ("completion", ST_NOT, 400, "completion"),  # "bas itna hi": fast whatever the tone
    ("short_answer", None, 400, "short_answer"),
    ("hesitation", ST_DONE, 2500, "hesitation"),  # incomplete endings: regardless of Smart Turn
    ("continuation", ST_DONE, 2500, "continuation"),
    ("postposition", ST_DONE, 2500, "postposition"),
    ("hesitation", None, 2500, "hesitation"),
    ("number_in_progress", ST_DONE, 2500, "number_in_progress"),
    ("id_in_progress", ST_DONE, 3000, "id_in_progress"),
    ("other", ST_DONE, 400, "other+smart_turn_complete"),
    ("other", ST_EDGE, 400, "other+smart_turn_complete"),
    ("other", ST_UNSURE, 2500, "other+smart_turn_unsure"),
    ("other", SmartTurnVerdict(False, 0.5), 2500, "other+smart_turn_unsure"),
    ("other", ST_NOT, 3000, "other+smart_turn_incomplete"),
    ("other", None, 2500, "other+no_smart_turn"),
    (None, ST_DONE, 400, "other+smart_turn_complete"),
    (None, None, 2500, "other+no_smart_turn"),
])
def test_wait_table(tail_class, st, ms, rule):
    w = choose_wait(tail_class, st)
    assert (w.ms, w.rule, w.capped) == (ms, rule, False)


@pytest.mark.parametrize("st, ms, rule", [
    (ST_DONE, 2500, "no_text"), (ST_UNSURE, 2500, "no_text"), (None, 2500, "no_text"),
    (ST_NOT, 3000, "other+smart_turn_incomplete"),  # a lone "umm": no text and the tone says not done
])
def test_no_text_for_the_pause_is_never_a_fast_reply(st, ms, rule):
    w = choose_wait(None, st, have_text=False)
    assert (w.ms, w.rule) == (ms, rule)


def test_hard_cap():
    p = replace(config.TURN_POLICY, wait_hesitation_ms=5000)
    w = choose_wait("hesitation", None, p)
    assert (w.ms, w.capped) == (3000, True)


def test_no_default_exceeds_the_cap():
    p = config.TURN_POLICY
    waits = [v for k, v in vars(p).items() if k.startswith("wait_")]
    assert max(waits) <= p.max_wait_ms == 3000
