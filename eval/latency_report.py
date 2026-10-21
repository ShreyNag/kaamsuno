"""p50/p95 per hop from the hop logs, plus a per-turn table of the last N turns.

    .venv/Scripts/python -m eval.latency_report               # all logs/*.jsonl
    .venv/Scripts/python -m eval.latency_report --last 10 logs/<session>.jsonl

"Last speech" = VAD speech_end minus the VAD's stop_secs (when the worker actually
went quiet). Logs written before stop_secs was logged assume 0.2 s, and their
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
    ("last speech -> first bot audio", "last_speech", "tts_first_byte"),
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
    for r in events:
        e, ts = r["event"], r["ts"]
        if e == "speech_start":
            speech_end = None
        elif e == "speech_end":
            speech_end, stop = ts, r.get("stop_secs", LEGACY_STOP_SECS)
            segment = None
        elif e == "stt_segment":
            if rows and "question" not in rows[-1] and rows[-1]["turn_end"] <= ts:
                rows[-1].setdefault("segment", ts)  # tail segment arriving after turn end
            segment = (ts, r.get("text", ""))
        elif e == "turn_end" and speech_end is not None:
            rows.append({"session": session, "speech_end": speech_end, "last_speech": speech_end - stop,
                         "turn_end": ts, "text": segment[1] if segment else ""})
            if segment:
                rows[-1]["segment"] = segment[0]
            speech_end = segment = None
        elif e in ("question", "tts_first_byte") and rows and e not in rows[-1]:
            rows[-1][e] = ts
            if e == "question":
                rows[-1]["asked"] = r.get("field")
    return rows


def hop_ms(row: dict, a: str, b: str) -> float | None:
    return (row[b] - row[a]) * 1000 if a in row and b in row else None


def report(rows: list[dict], last: int) -> str:
    out = ["| hop | n | p50 ms | p95 ms |", "|---|---:|---:|---:|"]
    for name, a, b in HOPS:
        v = [x for x in (hop_ms(r, a, b) for r in rows) if x is not None]
        if v:
            out.append(f"| {name} | {len(v)} | {np.percentile(v, 50):.0f} | {np.percentile(v, 95):.0f} |")
        else:
            out.append(f"| {name} | 0 | - | - |")
    out += ["", f"Last {min(last, len(rows))} turns:", "",
            "| session | speech_end->turn_end ms | last speech->turn_end ms | text |", "|---|---:|---:|---|"]
    for r in rows[-last:]:
        out.append(f"| {r['session']} | {hop_ms(r, 'speech_end', 'turn_end'):.0f} | "
                   f"{hop_ms(r, 'last_speech', 'turn_end'):.0f} | {r['text'][:50]} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--last", type=int, default=10)
    args = ap.parse_args()
    paths = args.logs or sorted(LOG_DIR.glob("*.jsonl"))
    rows = [row for p in paths for row in turns_from_events(read_events(p), p.stem)]
    print(report(rows, args.last))


if __name__ == "__main__":
    main()
