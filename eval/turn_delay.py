"""End-of-turn delay and cut-ins: the current turn policy replayed on logged sessions vs what happened live.

    .venv/Scripts/python -m eval.turn_delay                      # all logs/*.jsonl
    .venv/Scripts/python -m eval.turn_delay logs/2026103*.jsonl

Each session's pauses go through a fresh smart_turn_fillers TurnController (eval.replay_pauses.replay).
- delay: last speech -> turn end, over pauses the controller ended (turn_over)
- cut-in: a turn end followed by the worker speaking again within CUT_IN_S of the last speech
  (live: the bot cut in; replayed: the policy would have)
- held: a live cut-in that the policy now waits through (outcome "resumed")

Replayed outcomes after a live cut-in are indicative only: the worker may have talked differently once the
bot had spoken. Writes eval/out/turn_delay.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from app import config
from eval.replay_pauses import CUT_IN_S, replay

OUT = config.ROOT / "eval" / "out" / "turn_delay.md"


def pct(xs: list[float], q: float) -> str:
    return f"{np.percentile(xs, q):.0f}" if xs else "-"


def analyse(path: Path) -> dict:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [r["ts"] for r in events if r["event"] == "speech_start"]
    live = [r for r in events if r["event"] == "pause"]
    if not any(r["event"] == "turn_verdict" for r in events):
        return {"log": path.stem, "pauses": 0}  # no Smart Turn verdicts logged: fixed_timeout runs, or older logs
    now = replay(events)

    def cut(p: dict) -> bool:
        if p["outcome"] != "turn_over":
            return False
        nxt = next((s for s in starts if s > p["outcome_ts"]), None)
        return nxt is not None and nxt - p["last_speech"] < CUT_IN_S

    live_cuts = [p for p in live if cut(p)]
    now_by_ts = {round(p["last_speech"], 2): p for p in now}
    held = [p for p in live_cuts if (n := now_by_ts.get(round(p["last_speech"], 2))) and n["outcome"] == "resumed"]
    return {
        "log": path.stem, "pauses": len(now),
        "live_delay": [p["silence_ms"] for p in live if p["outcome"] == "turn_over" and not cut(p)],
        "now_delay": [p["silence_ms"] for p in now if p["outcome"] == "turn_over" and not cut(p)],
        "live_cuts": len(live_cuts), "now_cuts": sum(cut(p) for p in now), "held": len(held),
        "now_rules": [p["rule"] for p in now if p["outcome"] == "turn_over"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    paths = args.logs or sorted((config.ROOT / "logs").glob("*.jsonl"))
    rows = [r for r in (analyse(p) for p in paths) if r["pauses"]]
    p = config.TURN_POLICY
    lines = ["# End-of-turn delay: current policy replayed on logged sessions", "",
             f"Policy: fast {p.wait_st_complete_ms} ms only if Smart Turn >= {p.st_complete_prob} AND text arrived AND "
             f"no incomplete ending; completion {p.wait_completion_ms}; hesitation/connector/postposition "
             f"{p.wait_hesitation_ms}/{p.wait_continuation_ms}/{p.wait_postposition_ms}; number/ID "
             f"{p.wait_number_ms}/{p.wait_id_ms}; unsure or no text {p.wait_st_unsure_ms}/{p.wait_no_text_ms}; "
             f"tone < {p.st_unsure_prob} {p.wait_st_incomplete_ms}; cap {p.max_wait_ms} ms.", "",
             "Delay = last speech -> turn end, on turns that really ended (not cut-ins). "
             "\"Then\" is what each session's live policy did (older sessions ran older policies). Sessions "
             "without Smart Turn verdicts in the log (fixed_timeout runs, logs before 2026-10-25) are skipped.", "",
             "| session | pauses | turn ends then / now | delay p50 then -> now (ms) | delay p95 then -> now (ms) | "
             "cut-ins then -> now | live cut-ins now held |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['log']} | {r['pauses']} | {len(r['live_delay'])} / {len(r['now_delay'])} | "
                     f"{pct(r['live_delay'], 50)} -> {pct(r['now_delay'], 50)} | "
                     f"{pct(r['live_delay'], 95)} -> {pct(r['now_delay'], 95)} | "
                     f"{r['live_cuts']} -> {r['now_cuts']} | {r['held']}/{r['live_cuts']} |")
    lv = [x for r in rows for x in r["live_delay"]]
    nw = [x for r in rows for x in r["now_delay"]]
    rules: dict[str, int] = {}
    for r in rows:
        for k in r["now_rules"]:
            rules[k] = rules.get(k, 0) + 1
    lines += ["", f"**All: delay p50 {pct(lv, 50)} -> {pct(nw, 50)} ms, p95 {pct(lv, 95)} -> {pct(nw, 95)} ms "
              f"(n {len(lv)} -> {len(nw)}); cut-ins {sum(r['live_cuts'] for r in rows)} -> "
              f"{sum(r['now_cuts'] for r in rows)}; live cut-ins now held "
              f"{sum(r['held'] for r in rows)}/{sum(r['live_cuts'] for r in rows)}.**", "",
              "Turn ends now, by rule: " + ", ".join(f"{k} {v}" for k, v in sorted(rules.items(), key=lambda kv: -kv[1]))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
