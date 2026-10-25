"""TurnController state machine on simulated event timelines (no audio, no Pipecat, no clock).

Timelines use VAD stop_secs = 0.2, so a VAD stop at t means the speech ended at t - 0.2.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app import config
from app.turn.controller import State, TurnController, run_timeline
from app.turn.tail_classifier import PLACEHOLDER_FILLERS, DialogueContext, Fillers, TailClassifier
from app.turn.turn_policy import SmartTurnVerdict

CLF = TailClassifier(Fillers.from_lists(PLACEHOLDER_FILLERS))
ST_DONE = SmartTurnVerdict(True, 0.98)  # what Smart Turn really said on a Hinglish mid-sentence pause
ST_NOT = SmartTurnVerdict(False, 0.1)


def ctl(mode: str, **kw) -> tuple[TurnController, list[dict]]:
    log: list[dict] = []
    c = TurnController(mode=mode, classifier=CLF, log=lambda event, **f: log.append({"event": event, **f}), **kw)
    return c, log


def test_spec_example_hesitation_holds_the_turn_in_fillers_mode_only():
    # speech 0-2.1 s, segment "teesri manzil pe umm" at 2.4 s, speech again at 3.6 s
    events = [
        (0.2, "vad_start"), (2.3, "vad_stop", 0.2, ST_DONE), (2.4, "segment", "teesri manzil pe umm"),
        (3.6, "vad_start"),
    ]
    fillers, log = ctl("smart_turn_fillers")
    assert run_timeline(fillers, events) == []
    assert fillers.state == State.USER_SPEAKING
    assert log[0]["outcome"] == "resumed" and log[0]["tail_class"] == "hesitation" and log[0]["wait_ms"] == 2000

    fixed, _ = ctl("fixed_timeout")
    assert run_timeline(fixed, events) == [pytest.approx(2.8)]

    smart, _ = ctl("smart_turn")
    assert run_timeline(smart, events) == [pytest.approx(2.3)]  # Smart Turn's 0.98 cuts the worker off


def test_completion_replies_as_soon_as_the_segment_arrives():
    c, log = ctl("smart_turn_fillers")
    ends = run_timeline(c, [(0.2, "vad_start"), (2.3, "vad_stop", 0.2, ST_NOT), (2.6, "segment", "bas, itna hi")])
    assert ends == [pytest.approx(2.6)]  # 300 ms after 2.1 is already past when the words arrive
    assert log[-1]["rule"] == "completion" and log[-1]["outcome"] == "turn_over"
    assert c.state == State.BOT_THINKING


def test_completion_without_delay_when_segment_is_early():
    c, _ = ctl("smart_turn_fillers")
    # Prisma's segment arrives before our VAD stop (but after the last speech): it is the pause's tail
    ends = run_timeline(c, [(0.2, "vad_start"), (2.2, "segment", "ho gaya"), (2.3, "vad_stop", 0.2, ST_NOT)])
    assert ends == [pytest.approx(2.4)]  # last speech 2.1 + 300 ms


def test_segment_from_before_the_last_speech_is_not_the_tail():
    c, log = ctl("smart_turn_fillers")
    ends = run_timeline(c, [(0.2, "vad_start"), (1.0, "segment", "bas"), (2.3, "vad_stop", 0.2, None)])
    assert log[-1]["tail_class"] is None and log[-1]["rule"] == "other+no_smart_turn"
    assert ends == [pytest.approx(2.9)]  # 2.1 + 800


def test_no_segment_within_budget_falls_back_to_smart_turn():
    c, log = ctl("smart_turn_fillers")
    ends = run_timeline(c, [(0.2, "vad_start"), (2.3, "vad_stop", 0.2, ST_DONE)])
    budget = config.TURN_POLICY.segment_budget_ms / 1000
    assert ends == [pytest.approx(2.3 + budget)]  # 400 ms after the speech has passed by then
    assert log[-1]["rule"] == "other+smart_turn_complete" and log[-1]["tail_text"] is None


def test_late_segment_shortens_the_wait():
    c, log = ctl("smart_turn_fillers")
    ends = run_timeline(c, [(0.2, "vad_start"), (2.3, "vad_stop", 0.2, ST_NOT), (3.0, "segment", "bas")])
    assert ends == [pytest.approx(3.0)]  # decided 1200 ms at 2.7, then "bas" ends it
    assert log[-1]["revisions"] == 1 and log[-1]["rule"] == "completion"


def test_late_segment_extends_the_wait():
    c, log = ctl("smart_turn_fillers")
    events = [(0.2, "vad_start"), (2.3, "vad_stop", 0.2, None), (2.8, "segment", "cement umm")]
    assert run_timeline(c, events) == [pytest.approx(2.1 + 2.0)]  # 800 ms became 2000 ms
    assert log[-1]["revisions"] == 1 and log[-1]["rule"] == "hesitation"


def test_slow_id_is_not_cut_off():
    events = [
        (0.2, "vad_start"), (2.3, "vad_stop", 0.2, ST_DONE), (2.6, "segment", "employee ID chaar saat"),
        (3.3, "vad_start"),  # 1.2 s pause, then "ek nau"
        (4.3, "vad_stop", 0.2, ST_DONE), (4.6, "segment", "ek nau"),
    ]
    c, log = ctl("smart_turn_fillers")
    # second pause: "ek nau" still numeric with an ID cue in the turn -> 1500 ms
    assert run_timeline(c, events) == [pytest.approx(4.1 + 1.5)]
    assert [r["tail_class"] for r in log] == ["number_in_progress", "number_in_progress"]
    fixed, _ = ctl("fixed_timeout")
    assert run_timeline(fixed, events)[0] == pytest.approx(2.8)


def test_short_answer_after_a_confirmation_question():
    c, log = ctl("smart_turn_fillers")
    events = [
        (0.0, "context", DialogueContext("location", confirmation=True)),
        (0.2, "vad_start"), (1.2, "vad_stop", 0.2, ST_NOT), (1.5, "segment", "haan sahi hai"),
    ]
    assert run_timeline(c, events) == [pytest.approx(1.5)]
    assert log[-1]["tail_class"] == "short_answer"


def test_smart_turn_mode_incomplete_waits_for_the_silence_fallback():
    c, _ = ctl("smart_turn")
    ends = run_timeline(c, [(0.2, "vad_start"), (2.3, "vad_stop", 0.2, ST_NOT), (2.4, "segment", "bas")])
    assert ends == [pytest.approx(2.1 + config.SMART_TURN_STOP_SECS)]  # segments don't matter in this mode


def test_hard_cap():
    c, log = ctl("smart_turn_fillers", policy=replace(config.TURN_POLICY, wait_hesitation_ms=9000))
    ends = run_timeline(c, [(0.2, "vad_start"), (2.3, "vad_stop", 0.2, None), (2.4, "segment", "umm")])
    assert ends == [pytest.approx(2.1 + 2.5)]
    assert log[-1]["capped"]


def test_full_cycle_states_and_one_log_line_per_pause():
    c, log = ctl("smart_turn_fillers")
    assert c.state == State.LISTENING
    c.user_started(0.2)
    assert c.state == State.USER_SPEAKING
    c.user_stopped(1.3, 0.2, ST_NOT)
    assert c.state == State.PAUSED
    c.segment(1.4, "tower B ke")
    assert c.state == State.PAUSED
    c.user_started(2.0)
    assert c.state == State.USER_SPEAKING
    c.user_stopped(3.3, 0.2, ST_NOT)
    assert c.segment(3.5, "paas crane gira bas")
    assert c.state == State.BOT_THINKING
    c.bot_started(4.5)
    assert c.state == State.BOT_SPEAKING
    c.bot_stopped(6.0)
    assert c.state == State.LISTENING
    assert [(r["event"], r["outcome"]) for r in log] == [("pause", "resumed"), ("pause", "turn_over")]
    keys = {"tail_text", "tail_class", "smart_turn", "wait_ms", "outcome", "last_speech", "vad_stop", "outcome_ts"}
    assert keys <= log[0].keys()
    assert c.turn_segments == ["tower B ke", "paas crane gira bas"]


def test_worker_resumes_while_bot_is_thinking():
    c, _ = ctl("fixed_timeout")
    run_timeline(c, [(0.2, "vad_start"), (1.3, "vad_stop", 0.2), (2.1, "vad_start")])
    assert c.state == State.USER_SPEAKING
    assert c.turn_segments == []  # a new turn


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        TurnController(mode="vibes", classifier=CLF)
