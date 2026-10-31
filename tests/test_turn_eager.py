"""Turn taking must not be eager: a fast reply needs BOTH tone (Smart Turn) and words to say "finished".

Synthetic timelines through the real config policy and the real filler lists (config.TURN_* + fillers.json),
in the worst case seen live: Smart Turn says complete (0.98) at a mid-sentence pause. VAD stop_secs = 0.2,
so a VAD stop at t means the speech ended at t - 0.2. Prisma's final arrives ~0.45-0.55 s after the last
speech (live p50 ~0.5 s).
"""

from __future__ import annotations

import pytest

from app.turn.controller import State, TurnController, run_timeline
from app.turn.turn_policy import SmartTurnVerdict

LAST = 2.1  # last speech before the pause
ST_DONE = SmartTurnVerdict(True, 0.98)
ST_UNSURE = SmartTurnVerdict(True, 0.65)
ST_NOT = SmartTurnVerdict(False, 0.05)

MID_SENTENCE = ["cement aur", "pump ke paas", "teesri manzil pe umm", "machine ID chaar saat", "chaar saat",
                "cement teesri manzil pe", "umm",
                "सीमेंट और", "पंप के पास", "तीसरी मंजिल पे"]
FINISHED = [("mixer kharab hai", ST_DONE), ("bas itna hi", ST_NOT), ("bas itna hi", None), ("that's it", ST_UNSURE)]


def ctl() -> tuple[TurnController, list[dict]]:
    log: list[dict] = []
    return TurnController(mode="smart_turn_fillers", log=lambda e, **f: log.append(f)), log


def timeline(text_events, st, pause_s):
    return ([(0.2, "vad_start"), (LAST + 0.2, "vad_stop", 0.2, st)] + text_events
            + [(LAST + pause_s, "vad_start")])


@pytest.mark.parametrize("pause_s", [1.0, 2.0, 2.5])
@pytest.mark.parametrize("st", [ST_DONE, ST_UNSURE, None], ids=["st_0.98", "st_0.65", "no_st"])
@pytest.mark.parametrize("text", MID_SENTENCE)
def test_mid_sentence_pause_never_ends_the_turn(text, st, pause_s):
    c, log = ctl()
    ends = run_timeline(c, timeline([(LAST + 0.5, "segment", text)], st, pause_s))
    assert ends == [], log
    assert c.state == State.USER_SPEAKING and log[-1]["outcome"] == "resumed"


@pytest.mark.parametrize("text", MID_SENTENCE)
def test_mid_sentence_pause_with_only_a_partial(text):
    """No final yet: the partial's last words are the tail."""
    c, log = ctl()
    ends = run_timeline(c, timeline([(LAST - 0.1, "partial", text)], ST_DONE, 2.0))
    assert ends == [], log


@pytest.mark.parametrize("text", MID_SENTENCE)
def test_mid_sentence_pause_with_text_arriving_late(text):
    """Nothing within the segment budget (2500 ms no-text wait), then the final at +1.5 s re-decides."""
    c, log = ctl()
    ends = run_timeline(c, timeline([(LAST + 1.5, "segment", text)], ST_DONE, 2.4))
    assert ends == [], log


def test_tail_split_over_two_segments():
    """"ke" | "paas": Prisma split the words before the pause; the last 1-3 words of the turn decide."""
    c, log = ctl()
    events = [(0.2, "vad_start"), (1.5, "segment", "pump ke"), (LAST + 0.2, "vad_stop", 0.2, ST_DONE),
              (LAST + 0.5, "segment", "paas"), (LAST + 2.0, "vad_start")]
    assert run_timeline(c, events) == [], log
    assert log[-1]["tail_class"] == "postposition"


@pytest.mark.parametrize("text, st", FINISHED)
def test_finished_sentence_ends_within_half_a_second(text, st):
    c, log = ctl()
    ends = run_timeline(c, [(0.2, "vad_start"), (LAST + 0.2, "vad_stop", 0.2, st), (LAST + 0.45, "segment", text)])
    assert len(ends) == 1 and ends[0] - LAST <= 0.5 + 1e-9, (ends, log)


def test_finished_words_without_tone_are_not_fast():
    """"mixer kharab hai" but Smart Turn unsure: not both -> 2.5 s."""
    c, log = ctl()
    ends = run_timeline(c, [(0.2, "vad_start"), (LAST + 0.2, "vad_stop", 0.2, ST_UNSURE),
                            (LAST + 0.45, "segment", "mixer kharab hai")])
    assert ends == [pytest.approx(LAST + 2.5)] and log[-1]["rule"] == "other+smart_turn_unsure"


def test_resuming_after_the_turn_was_decided_cancels_it():
    c, log = ctl()
    events = [(0.2, "vad_start"), (LAST + 0.2, "vad_stop", 0.2, ST_UNSURE), (LAST + 0.45, "segment", "mixer kharab hai"),
              (LAST + 1.8, "vad_start")]
    assert run_timeline(c, events) == [] and log[-1]["outcome"] == "resumed"


def test_incomplete_ending_waits_three_and_a_half_seconds_even_with_low_tone():
    c, log = ctl()
    events = [(0.2, "vad_start"), (LAST + 0.2, "vad_stop", 0.2, ST_NOT), (LAST + 0.5, "segment", "cement aur")]
    assert run_timeline(c, events) == [pytest.approx(LAST + 3.5)]  # 3500 for "aur", under the 4000 cap
