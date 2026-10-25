"""p50/p95 per hop from the hop logs, plus a per-turn table of the last N turns.

    .venv/Scripts/python -m eval.latency_report               # all logs/*.jsonl
    .venv/Scripts/python -m eval.latency_report --last 10 logs/<session>.jsonl

"Last speech" = VAD speech_end minus the VAD's stop_secs (when the worker actually
went quiet). Rows are grouped by turn mode (from the controller's `pause` events;
"unknown" for logs written before Phase 2). Logs written before stop_secs was logged assume 0.2 s, and their
doubled events (pre-fix broadcast duplicates) are collapsed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.config import LOG_DIR

LEGACY_STOP_SECS = 0.2
HOPS = [
    ("speech_end -> turn_end", "speech_end", "turn_end"),
    ("last speech -> turn_end", "last_speech", "turn_end"),
    ("last speech -> last segment", "last_speech", "segment"),
    ("turn_end -> question", "turn_end", "question"),
    ("turn_end -> tts_first_byte", "turn_end", "tts_first_byte"),
    ("question -> tts_first_byte (Timbre)", "question", "tts_first_byte"),
    ("last speech -> tts_first_byte", "last_speech", "tts_first_byte"),
    ("last speech -> playback_start", "last_speech", "playback_start"),
]


def read_events(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if out and out[-1]["event"] == r["event"] and r["ts"] - out[-1]["ts"] < 0.01:
            continue  # duplicate copy of a broadcast frame (logs before the observer fix)
        out.append(r)
    return out


def turns_from_events(events: list[dict], session: str) -> list[dict]:
    """One row per turn_end with the timestamps of the hops around it."""
    rows: list[dict] = []
    speech_end = stop = None
    segment = None
    mode = "unknown"
    for r in events:
        e, ts = r["event"], r["ts"]
        if e == "pause":
            mode = r.get("mode", mode)
        elif e == "speech_start":
            speech_end = None
        elif e == "speech_end":
            speech_end, stop = ts, r.get("stop_secs", LEGACY_STOP_SECS)
            segment = None
        elif e == "stt_segment":
            if rows and "question" not in rows[-1] and rows[-1]["turn_end"] <= ts:
                rows[-1].setdefault("segment", ts)  # tail segment arriving after turn end
            segment = (ts, r.get("text", ""))
        elif e == "turn_end" and speech_end is not None:
            # the controller logs the pause that ended the turn right before Pipecat's turn_end
            rows.append({"session": session, "speech_end": speech_end, "last_speech": speech_end - stop,
                         "turn_end": ts, "text": segment[1] if segment else "", "mode": mode})
            if segment:
                rows[-1]["segment"] = segment[0]
            speech_end = segment = None
        elif e in ("question", "tts_first_byte", "playback_start") and rows and e not in rows[-1]:
            rows[-1][e] = ts
            if e == "question":
                rows[-1]["asked"] = r.get("field")
        elif e == "tts_ttfb" and rows and "question" in rows[-1] and "tts_ttfb_ms" not in rows[-1]:
            rows[-1]["tts_ttfb_ms"] = r["ms"]  # Pipecat's own TTS TTFB metric
    return rows


def hop_ms(row: dict, a: str, b: str) -> float | None:
    return (row[b] - row[a]) * 1000 if a in row and b in row else None


def barge_ins(events: list[dict]) -> list[dict]:
    """User speech starting while the bot is talking -> when the bot stopped (and whether an
    interruption was broadcast in between). One row per barge-in."""
    out: list[dict] = []
    speaking, pending = False, None
    for r in events:
        e, ts = r["event"], r["ts"]
        if e == "bot_started":
            speaking = True
        elif e == "speech_start" and speaking and pending is None:
            pending = {"speech_start": ts, "interruption": None}
        elif e == "interruption" and pending is not None and pending["interruption"] is None:
            pending["interruption"] = ts
        elif e == "bot_stopped":
            speaking = False
            if pending is not None:
                out.append({**pending, "bot_stopped": ts, "reaction_ms": (ts - pending["speech_start"]) * 1000})
                pending = None
    return out


def _row(name: str, v: list[float]) -> str:
    if not v:
        return f"| {name} | 0 | - | - |"
    return f"| {name} | {len(v)} | {np.percentile(v, 50):.0f} | {np.percentile(v, 95):.0f} |"


def hop_table(rows: list[dict]) -> list[str]:
    out = ["| hop | n | p50 ms | p95 ms |", "|---|---:|---:|---:|"]
    for name, a, b in HOPS:
        out.append(_row(name, [x for x in (hop_ms(r, a, b) for r in rows) if x is not None]))
    out.append(_row("TTS TTFB (Pipecat metric)", [r["tts_ttfb_ms"] for r in rows if "tts_ttfb_ms" in r]))
    return out


def report(rows: list[dict], last: int, barges: list[dict] | None = None) -> str:
    out: list[str] = []
    for mode in sorted({r["mode"] for r in rows}):
        out += [f"## turn mode: {mode}", ""] + hop_table([r for r in rows if r["mode"] == mode]) + [""]
    if barges:
        v = [b["reaction_ms"] for b in barges]
        missed = sum(b["interruption"] is None for b in barges)
        out += ["## Barge-in (user speech start during bot audio -> bot audio stops)", "",
                "| n | p50 ms | p95 ms | max ms | without an interruption frame |", "|---:|---:|---:|---:|---:|",
                f"| {len(v)} | {np.percentile(v, 50):.0f} | {np.percentile(v, 95):.0f} | {max(v):.0f} | {missed} |", ""]
    out += [f"Last {min(last, len(rows))} turns:", "",
            "| session | mode | speech_end->turn_end ms | last speech->turn_end ms | text |", "|---|---|---:|---:|---|"]
    for r in rows[-last:]:
        out.append(f"| {r['session']} | {r['mode']} | {hop_ms(r, 'speech_end', 'turn_end'):.0f} | "
                   f"{hop_ms(r, 'last_speech', 'turn_end'):.0f} | {r['text'][:50]} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--last", type=int, default=10)
    args = ap.parse_args()
    paths = args.logs or sorted(LOG_DIR.glob("*.jsonl"))
    events = {p: read_events(p) for p in paths}
    rows = [row for p in paths for row in turns_from_events(events[p], p.stem)]
    print(report(rows, args.last, [b for p in paths for b in barge_ins(events[p])]))


if __name__ == "__main__":
    main()
