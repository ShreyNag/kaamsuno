"""Table-driven tests for the wait policy (config defaults)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app import config
from app.turn.turn_policy import SmartTurnVerdict, choose_wait

ST_DONE = SmartTurnVerdict(True, 0.98)
ST_WEAK = SmartTurnVerdict(True, 0.6)  # the analyzer's 0.5 threshold says complete, the policy's 0.7 doesn't
ST_NOT = SmartTurnVerdict(False, 0.1)


@pytest.mark.parametrize("tail_class, st, ms, rule", [
    ("completion", ST_NOT, 300, "completion"),
    ("short_answer", None, 300, "short_answer"),
    ("hesitation", ST_DONE, 3000, "hesitation"),  # regardless of Smart Turn
    ("continuation", ST_DONE, 3000, "continuation"),
    ("hesitation", None, 3000, "hesitation"),
    ("number_in_progress", ST_DONE, 2000, "number_in_progress"),
    ("id_in_progress", ST_DONE, 3000, "id_in_progress"),
    ("other", ST_DONE, 400, "other+smart_turn_complete"),
    ("other", SmartTurnVerdict(True, 0.7), 400, "other+smart_turn_complete"),
    ("other", ST_WEAK, 1200, "other+smart_turn_incomplete"),
    ("other", ST_NOT, 1200, "other+smart_turn_incomplete"),
    ("other", None, 800, "other+no_smart_turn"),
    (None, ST_DONE, 400, "other+smart_turn_complete"),  # no segment in time: Smart Turn decides
    (None, None, 800, "other+no_smart_turn"),
])
def test_wait_table(tail_class, st, ms, rule):
    w = choose_wait(tail_class, st)
    assert (w.ms, w.rule, w.capped) == (ms, rule, False)


@pytest.mark.parametrize("st", [ST_DONE, ST_NOT, None])
def test_no_text_for_the_pause_waits_to_the_cap_whatever_smart_turn_says(st):
    w = choose_wait(None, st, have_text=False)
    assert (w.ms, w.rule) == (config.TURN_POLICY.max_wait_ms, "no_text")


def test_hard_cap():
    p = replace(config.TURN_POLICY, wait_hesitation_ms=4000)
    w = choose_wait("hesitation", None, p)
    assert (w.ms, w.capped) == (3500, True)


def test_no_default_exceeds_the_cap():
    p = config.TURN_POLICY
    waits = [v for k, v in vars(p).items() if k.startswith("wait_")]
    assert max(waits) <= p.max_wait_ms
