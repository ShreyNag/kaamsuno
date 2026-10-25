"""The turn controller: an explicit state machine over timestamped events. Pure, no Pipecat, no clock.

    LISTENING -> USER_SPEAKING -> PAUSED -> (USER_SPEAKING again | TURN_OVER) -> BOT_THINKING
              -> BOT_SPEAKING -> LISTENING

Inputs (all with a wall-clock time `now`, seconds): VAD start/stop, Prisma
segments, the Smart Turn verdict (delivered with the VAD stop, which is when
it's computed), bot audio start/stop, and the dialogue context. The owner calls
`tick(now)` at `next_wakeup()`. Every method returns True when the turn is over.

What happens in PAUSED depends on the mode:
    fixed_timeout       turn over FIXED_TIMEOUT_MS after the last speech
    smart_turn          over at the VAD stop if Smart Turn says complete, else after its silence fallback
    smart_turn_fillers  wait for the pause's Prisma segment (up to a budget), classify its tail,
                        then wait as turn_policy says. A segment arriving later re-decides.

Each pause is logged once, when it resolves (`resumed` or `turn_over`): the
turn-taking evaluation data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app import config
from app.turn.tail_classifier import DialogueContext, TailClassifier
from app.turn.turn_policy import SmartTurnVerdict, Wait, choose_wait

TURN_MODES = ("fixed_timeout", "smart_turn", "smart_turn_fillers")


class State(str, Enum):
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PAUSED = "PAUSED"
    TURN_OVER = "TURN_OVER"
    BOT_THINKING = "BOT_THINKING"
    BOT_SPEAKING = "BOT_SPEAKING"


@dataclass
class Pause:
    last_speech: float
    vad_stop: float
    smart_turn: SmartTurnVerdict | None
    segment: str | None = None
    segment_ts: float | None = None
    segment_deadline: float | None = None  # fillers mode: decide without the segment after this
    tail_class: str | None = None
    tail: str | None = None
    wait: Wait | None = None
    deadline: float | None = None  # turn over at this time unless speech resumes
    decided_at: float | None = None
    revisions: int = 0  # decisions after the first (a late segment)

    def record(self, mode: str, outcome: str, now: float) -> dict[str, Any]:
        r = lambda t: None if t is None else round(t, 3)  # noqa: E731
        ms = lambda t: None if t is None else round((t - self.last_speech) * 1000)  # noqa: E731
        st = self.smart_turn
        return {
            "mode": mode, "outcome": outcome,
            "last_speech": r(self.last_speech), "vad_stop": r(self.vad_stop), "outcome_ts": r(now),
            "silence_ms": ms(now), "segment_lag_ms": ms(self.segment_ts), "decided_ms": ms(self.decided_at),
            "tail_text": self.segment, "tail": self.tail, "tail_class": self.tail_class,
            "smart_turn": st and {"complete": st.complete, "p": round(st.probability, 3)},
            "wait_ms": self.wait and self.wait.ms, "rule": self.wait and self.wait.rule,
            "capped": bool(self.wait and self.wait.capped), "revisions": self.revisions,
        }


@dataclass
class TurnController:
    mode: str = config.TURN_MODE
    classifier: TailClassifier | None = None
    policy: config.TurnPolicyConfig = config.TURN_POLICY
    fixed_timeout_ms: int = config.FIXED_TIMEOUT_MS
    smart_turn_stop_secs: float = config.SMART_TURN_STOP_SECS
    log: Callable[..., None] | None = None  # log(event, **fields), e.g. HopLogger.log
    context: DialogueContext = field(default_factory=DialogueContext)

    state: State = State.LISTENING
    pause: Pause | None = None
    turn_segments: list[str] = field(default_factory=list)
    _last_segment: tuple[float, str] | None = None

    def __post_init__(self):
        if self.mode not in TURN_MODES:
            raise ValueError(f"TURN_MODE must be one of {TURN_MODES}, got {self.mode!r}")
        if self.classifier is None:
            self.classifier = TailClassifier()  # every mode logs the tail class, so the modes can be compared

    # --- inputs -----------------------------------------------------------------
    def user_started(self, now: float) -> bool:
        if self.state == State.PAUSED:
            self._close_pause("resumed", now)
        elif self.state != State.USER_SPEAKING:
            self.turn_segments = []  # a new turn (also a barge-in over the bot; Day 4 tunes that)
        self.state = State.USER_SPEAKING
        return False

    def user_stopped(self, now: float, stop_secs: float, smart_turn: SmartTurnVerdict | None = None) -> bool:
        if self.state != State.USER_SPEAKING:
            return False
        self.state = State.PAUSED
        p = self.pause = Pause(last_speech=now - stop_secs, vad_stop=now, smart_turn=smart_turn)
        if self._last_segment and self._last_segment[0] >= p.last_speech:
            # Prisma closed the segment before our VAD did: it already holds the words before the pause
            p.segment_ts, p.segment = self._last_segment
        if self.mode == "smart_turn_fillers" and p.segment is None:
            p.segment_deadline = now + self.policy.segment_budget_ms / 1000
            return False
        return self._decide(now)

    def segment(self, now: float, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        self._last_segment = (now, text)
        if self.state in (State.USER_SPEAKING, State.PAUSED):
            self.turn_segments.append(text)
        if self.state != State.PAUSED:
            return False
        p = self.pause
        p.segment, p.segment_ts = text, now
        if self.mode != "smart_turn_fillers":
            self._classify_for_log()
            return False
        if p.decided_at is not None:
            p.revisions += 1
        return self._decide(now)

    def tick(self, now: float) -> bool:
        if self.state != State.PAUSED:
            return False
        p = self.pause
        if p.decided_at is None and p.segment_deadline is not None and now >= p.segment_deadline:
            return self._decide(now)  # no segment within the budget: decide without it
        if p.deadline is not None and now >= p.deadline:
            return self._turn_over(now)
        return False

    def next_wakeup(self) -> float | None:
        if self.state != State.PAUSED:
            return None
        p = self.pause
        return p.segment_deadline if p.decided_at is None else p.deadline

    def bot_started(self, now: float) -> None:
        if self.state in (State.LISTENING, State.TURN_OVER, State.BOT_THINKING):
            self.state = State.BOT_SPEAKING

    def bot_stopped(self, now: float) -> None:
        if self.state == State.BOT_SPEAKING:
            self.state = State.LISTENING

    def turn_stopped_externally(self, now: float) -> None:
        """Pipecat ended the turn without us (e.g. its stop watchdog)."""
        if self.state == State.PAUSED:
            self._close_pause("turn_over_external", now)
        if self.state in (State.USER_SPEAKING, State.PAUSED):
            self.state = State.BOT_THINKING

    # --- decision -----------------------------------------------------------------
    def _classify_for_log(self) -> None:
        if self.classifier and self.pause.segment is not None:
            res = self.classifier.classify(self.pause.segment, self.context, " ".join(self.turn_segments))
            self.pause.tail_class, self.pause.tail = res.tail_class, res.tail

    def _decide(self, now: float) -> bool:
        p = self.pause
        self._classify_for_log()  # tail_class stays None without a segment
        if self.mode == "fixed_timeout":
            wait = Wait(self.fixed_timeout_ms, "fixed_timeout")
        elif self.mode == "smart_turn":
            if p.smart_turn is not None and p.smart_turn.complete:
                wait = Wait(round((p.vad_stop - p.last_speech) * 1000), "smart_turn_complete")
            else:
                wait = Wait(round(self.smart_turn_stop_secs * 1000), "smart_turn_silence_fallback")
        else:
            wait = choose_wait(p.tail_class, p.smart_turn, self.policy)
        p.wait, p.decided_at = wait, now
        p.deadline = p.last_speech + wait.ms / 1000
        return self.tick(now)

    def _turn_over(self, now: float) -> bool:
        self._close_pause("turn_over", now)
        self.state = State.TURN_OVER
        self.state = State.BOT_THINKING  # TURN_OVER is instantaneous: the reporter is now choosing a question
        return True

    def _close_pause(self, outcome: str, now: float) -> None:
        if self.pause and self.log:
            self.log("pause", **self.pause.record(self.mode, outcome, now))
        self.pause = None


def run_timeline(ctl: TurnController, events: list[tuple]) -> list[float]:
    """Drive `ctl` through a simulated timeline; returns the turn-end times.

    events: (t, "vad_start") | (t, "vad_stop", stop_secs[, SmartTurnVerdict]) | (t, "segment", text)
            | (t, "bot_start") | (t, "bot_stop") | (t, "context", DialogueContext), in time order.
    Deadlines falling between events fire at their own time, as the live timer would.
    """
    ends: list[float] = []

    def advance(until: float) -> None:
        while (w := ctl.next_wakeup()) is not None and w <= until:
            if ctl.tick(w):
                ends.append(w)

    for ev in sorted(events, key=lambda e: e[0]):
        t, kind, *args = ev
        advance(t)
        over = False
        if kind == "vad_start":
            over = ctl.user_started(t)
        elif kind == "vad_stop":
            over = ctl.user_stopped(t, args[0], args[1] if len(args) > 1 else None)
        elif kind == "segment":
            over = ctl.segment(t, args[0])
        elif kind == "bot_start":
            ctl.bot_started(t)
        elif kind == "bot_stop":
            ctl.bot_stopped(t)
        elif kind == "context":
            ctl.context = args[0]
        else:
            raise ValueError(kind)
        if over:
            ends.append(t)
    advance(float("inf"))
    return ends
