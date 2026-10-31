"""Groq free-tier budget: calls per turn under the live policy, replayed on real session logs.

    .venv/Scripts/python -m eval.groq_budget                       # all logs/*.jsonl
    .venv/Scripts/python -m eval.groq_budget --tokens 1600 --tpm 8000 --gap 1.5

For each logged session, the real Prisma segment times and turn ends go through the Interpreter's
policy (one call in flight, segments arriving meanwhile merged into the next call, LLM_MIN_GAP_S
between calls, turn end sends what is left at once). Call latency is that session's measured median.
Reports calls per turn and the worst rolling 60 s of tokens, against the limits Groq's headers gave
(--tpm, --rpd; defaults are what our key returned on 2026-10-31). Writes eval/out/groq_budget.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from app import config

OUT = config.ROOT / "eval" / "out" / "groq_budget.md"


def simulate(segments: list[float], turn_ends: list[float], latency: float, gap: float) -> list[tuple[float, float]]:
    """(start, end) of every call. Segments queue; a call starts when none is in flight and either the gap
    since the last call has passed or a turn end has flushed."""
    events = sorted([(t, "seg") for t in segments] + [(t, "end") for t in turn_ends])
    calls: list[tuple[float, float]] = []
    pending, busy_until, last_end = 0, float("-inf"), float("-inf")
    flush_at: float | None = None
    i = 0
    t = events[0][0] if events else 0
    while i < len(events) or pending:
        # next moment a call could start
        start = max(busy_until, t)
        if pending:
            ready = start if (flush_at is not None and flush_at <= start) else max(start, last_end + gap)
            nxt = events[i][0] if i < len(events) else float("inf")
            if ready <= nxt:
                calls.append((ready, ready + latency))
                busy_until = last_end = ready + latency
                pending, t = 0, ready
                if flush_at is not None and flush_at <= ready:
                    flush_at = None
                continue
        if i >= len(events):
            break
        t, kind = events[i]
        i += 1
        if kind == "seg":
            pending += 1
        else:
            flush_at = t if pending or busy_until > t else None
    return calls


def peak_window(starts: list[float], window: float = 60.0) -> int:
    best, j = 0, 0
    for k, s in enumerate(starts):
        while starts[j] < s - window:
            j += 1
        best = max(best, k - j + 1)
    return best


def bucket_delays(starts: list[float], need: int, cap: int) -> list[float]:
    """How long the pacer holds each call: a token bucket of `cap` per minute, refilled linearly, full at
    the session start. Calls are sequential, so a held call pushes the later ones."""
    refill, level, t, delays = cap / 60.0, float(cap), None, []
    for s in starts:
        s = max(s, t) if t is not None else s
        if t is not None:
            level = min(cap, level + (s - t) * refill)
        d = max(0.0, (need - level) / refill)
        level += d * refill - need
        t = s + d
        delays.append(d)
    return delays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(config.ROOT / "logs"))
    ap.add_argument("--tokens", type=int, default=1600, help="tokens per call (measured: 1435 prompt + ~170 out)")
    ap.add_argument("--tpm", type=int, default=8000)
    ap.add_argument("--rpd", type=int, default=1000)
    ap.add_argument("--gap", type=float, default=config.LLM_MIN_GAP_S)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for f in sorted(Path(args.logs).glob("*.jsonl")):
        ev = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        segs = [e["ts"] for e in ev if e["event"] == "stt_segment"]
        ends = [e["ts"] for e in ev if e["event"] == "turn_end"]
        worker_turns = len({e["turn"] for e in ev if e["event"] == "stt_segment"})  # turns with words in them
        ms = [e["ms"] for e in ev if e["event"] == "interp_response" and e.get("ms")]
        if not segs or not ends:
            continue
        lat = statistics.median(ms) / 1000 if ms else 0.8
        calls = simulate(segs, ends, lat, args.gap)
        old = len(segs)  # before: one call per segment
        starts = [s for s, _ in calls]
        dur = max(ends + segs) - min(segs)
        delays = bucket_delays(starts, args.tokens, args.tpm)
        rows.append({"log": f.stem, "turns": worker_turns, "held": sum(d > 0 for d in delays), "max_hold": max(delays), "segments": len(segs), "old_calls": old, "calls": len(calls),
                     "per_turn": len(calls) / worker_turns, "peak_60s": peak_window(starts), "dur_s": dur,
                     "lat_ms": round(lat * 1000)})
    per_call_cap = args.tpm // args.tokens
    tot_turns = sum(r["turns"] for r in rows)
    tot_calls = sum(r["calls"] for r in rows)
    tot_old = sum(r["old_calls"] for r in rows)
    cpt = tot_calls / tot_turns if tot_turns else 0
    lines = [
        "# Groq free-tier budget (live interpreter)", "",
        f"Limits from Groq's headers: {args.tpm} tokens/min, {args.rpd} requests/day. "
        f"A call costs ~{args.tokens} tokens, so at most {per_call_cap} calls in any minute "
        f"(plus whatever is left in the bucket).", "",
        f"Policy: one call in flight, merge segments, min gap {args.gap} s, flush at turn end. "
        f"Replayed on {len(rows)} logged sessions.", "",
        "Turns = worker turns with words in them (turn ends on noise alone make no call). Held = calls the "
        "pacer would hold for the token bucket (full at the start of the session).", "",
        "| session | turns | segments | calls before (1/segment) | calls now | calls/turn | peak calls in 60 s | held | max hold s | min | interp p50 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        flag = " ⚠" if r["peak_60s"] > per_call_cap else ""
        lines.append(f"| {r['log']} | {r['turns']} | {r['segments']} | {r['old_calls']} | {r['calls']} | "
                     f"{r['per_turn']:.2f} | {r['peak_60s']}{flag} | {r['held']} | {r['max_hold']:.1f} | {r['dur_s'] / 60:.1f} | {r['lat_ms']} |")
    ten = cpt * 10
    # a 10-turn session at a given turn period: how many calls the pacer holds, and for how long at most
    sweep = []
    for period in (6, 9, 12, 15):
        d = bucket_delays([i * period for i in range(round(ten))], args.tokens, args.tpm)
        sweep.append(f"| {period} s | {sum(x > 0 for x in d)} | {max(d):.1f} |")
    lines += ["", f"**All sessions: {tot_calls} calls for {tot_turns} turns = {cpt:.2f} calls/turn** "
              f"(was {tot_old / tot_turns:.2f} with one call per segment).", "",
              f"10-turn session: ~{ten:.0f} calls = ~{ten * args.tokens:,.0f} tokens; requests/day allow "
              f"~{args.rpd // max(ten, 1):.0f} such sessions. ⚠ = a minute over {per_call_cap} calls: the pacer "
              "holds the next call until the bucket refills, and the dialogue asks from the report as it is.", "",
              f"10 turns, {round(ten)} calls, one call every N seconds (the bucket starts full):", "",
              "| a call every | calls held | longest hold s |", "|---:|---:|---:|"] + sweep
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
