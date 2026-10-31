"""Replay a live session's pauses through the current turn controller and compare with what happened.

    .venv/Scripts/python -m eval.replay_pauses logs/<session>.jsonl [...]

Rebuilds the event timeline from the hop log (VAD start/stop with Smart Turn's verdict, Prisma
partials and finals, the bot's questions as dialogue context), runs it through a fresh
smart_turn_fillers TurnController, and prints one row per pause: what text had arrived by the
decision, the tail class, Smart Turn, the wait, and the outcome then (live) and now (replayed).
A live "turn_over" followed by the worker speaking again within a few seconds is marked as a cut-in.

The worker's later speech is taken as given: if the live bot cut in, the worker may have talked
differently afterwards, so rows after a changed outcome are indicative only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import config
from app.interpreter.dialogue import CONFIRM_REASONS
from app.turn.controller import TurnController
from app.turn.tail_classifier import DialogueContext
from app.turn.turn_policy import SmartTurnVerdict

CUT_IN_S = 3.0  # speech this soon after a live turn end = the worker wasn't done


def timeline(events: list[dict]) -> list[tuple]:
    out: list[tuple] = []
    pending_stop = None
    for r in events:
        e, ts = r["event"], r["ts"]
        if e == "speech_start":
            out.append((ts, "vad_start"))
        elif e == "speech_end":
            pending_stop = [ts, "vad_stop", r.get("stop_secs", 0.2), None]
            out.append(pending_stop)
        elif e == "turn_verdict" and pending_stop is not None:
            pending_stop[3] = SmartTurnVerdict(r["complete"], r["p_complete"], r.get("infer_ms", 0.0))
            pending_stop = None
        elif e == "stt_segment":
            out.append((ts, "segment", r["text"]))
        elif e == "stt_partial":
            out.append((ts, "partial", r["text"]))
        elif e == "question" and r.get("text"):
            out.append((ts, "context", DialogueContext(r.get("field"), r.get("reason") in CONFIRM_REASONS)))
    return [tuple(x) for x in out]


def replay(events: list[dict]) -> list[dict]:
    log: list[dict] = []
    c = TurnController(mode="smart_turn_fillers", log=lambda event, **f: log.append(f))
    for t, kind, *args in sorted(timeline(events), key=lambda e: e[0]):
        while (w := c.next_wakeup()) is not None and w <= t:
            if c.tick(w):
                c.state = c.state.LISTENING  # the bot "spoke"; the worker's next words start a new turn
        over = False
        if kind == "vad_start":
            over = c.user_started(t)
        elif kind == "vad_stop":
            over = c.user_stopped(t, args[0], args[1])
        elif kind == "segment":
            over = c.segment(t, args[0])
        elif kind == "partial":
            over = c.partial(t, args[0])
        elif kind == "context":
            c.context = args[0]
        if over:
            c.state = c.state.LISTENING
    while (w := c.next_wakeup()) is not None:
        c.tick(w)
    return log


def compare(path: Path) -> list[str]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    t0 = events[0]["ts"]
    starts = [r["ts"] for r in events if r["event"] == "speech_start"]
    live = [r for r in events if r["event"] == "pause"]
    replayed = replay(events)
    rows = [f"### {path.stem}", "",
            "| t (s) | words before the pause | text by decision | tail class | Smart Turn | wait then -> now | "
            "outcome then -> now |", "|---:|---|---|---|---:|---|---|"]
    for p in live:
        n = min(replayed, key=lambda q: abs(q["last_speech"] - p["last_speech"]), default={})
        n = n if n and abs(n["last_speech"] - p["last_speech"]) < 0.2 else {}
        nxt = next((s for s in starts if s > p["outcome_ts"]), None)
        cut = p["outcome"] == "turn_over" and nxt is not None and nxt - p["last_speech"] < CUT_IN_S
        gap = f" (worker resumed after {nxt - p['last_speech']:.1f} s)" if cut else ""
        src = p.get("tail_source") or ("final" if p.get("segment_lag_ms") is not None and p.get("decided_ms") is not None
                                       and p["segment_lag_ms"] <= p["decided_ms"] else None)
        text_then = f"final @{p['segment_lag_ms']} ms" if src == "final" else "none"
        text_now = {"final": "final", "partial": "partial", None: "none"}[n.get("tail_source")]
        st = p["smart_turn"] and f"{p['smart_turn']['p']:.2f}"
        rows.append(
            f"| {p['last_speech'] - t0:.1f} | {p['tail_text'] or n.get('partial_text') or '(nothing)'} | "
            f"{text_then} -> {text_now} | {p['tail_class']} -> {n.get('tail_class')} | {st} | "
            f"{p['wait_ms']} -> {n.get('wait_ms')} ({n.get('rule')}) | "
            f"{'**CUT IN** ' if cut else ''}{p['outcome']}{gap} -> {n.get('outcome', '?')} |")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    p = config.TURN_POLICY
    print(f"policy now: fast {p.wait_st_complete_ms} ms only if ST >= {p.st_complete_prob} + text + no incomplete "
          f"ending; hesitation/connector/postposition {p.wait_hesitation_ms}/{p.wait_continuation_ms}/"
          f"{p.wait_postposition_ms}, number/ID {p.wait_number_ms}/{p.wait_id_ms}, ST unsure {p.wait_st_unsure_ms}, "
          f"no text {p.wait_no_text_ms}, cap {p.max_wait_ms} ms\n")
    for path in args.logs:
        print("\n".join(compare(path)) + "\n")


if __name__ == "__main__":
    main()
