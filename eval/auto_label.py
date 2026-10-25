"""Speech labels from CLEAN recordings, by energy, with generous padding. A stand-in for hand labels.

    .venv/Scripts/python -m eval.auto_label data/vad_dev_tmp/clean            # writes vad_labels.csv + plots
    .venv/Scripts/python -m eval.auto_label data/vad_dev_tmp/clean --force    # overwrite existing labels

Per file: 10 ms frame energy (dBFS); threshold = noise floor (10th percentile) +
max(MIN_MARGIN_DB, ALPHA * (peak - floor)), peak = 95th percentile, and never below
peak - PEAK_REL_DB (a gated recorder puts the floor at digital silence, ~-97 dBFS, which
would otherwise make clicks and breaths at -60..-80 dBFS "speech"). Frames above it are
speech; gaps shorter than MERGE_GAP_S are bridged (word gaps), segments shorter than
MIN_SEG_S dropped, then each segment padded by PAD_BEFORE_S / PAD_AFTER_S, but inside a
gap by at most a quarter of the gap per side: every pause of MERGE_GAP_S (300 ms) or more
stays non-speech (at least half of it).

Writes <dir>/vad_labels.csv (Aastha's format: file,start_s,end_s) and one
overlay plot per file in eval/out/labels/<dir name>/, so a human can check them.
It refuses to overwrite an existing vad_labels.csv without --force (hand labels win).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from app.audio_io import load_wav
from app.config import ROOT, SAMPLE_RATE
from app.vad import features as F

HOP_S = 0.01
FRAME_LEN = 400  # 25 ms
ALPHA = 0.2
MIN_MARGIN_DB = 10.0
PEAK_REL_DB = 30.0
MERGE_GAP_S = 0.3
MIN_SEG_S = 0.1
PAD_BEFORE_S = 0.15
PAD_AFTER_S = 0.2
PLOT_DIR = ROOT / "eval" / "out" / "labels"


def label_clip(x: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[list[tuple[float, float]], dict]:
    hop = int(HOP_S * sr)
    e = F.frame_energy_db(F.frame_signal(x, FRAME_LEN, hop))
    floor, peak = np.percentile(e, (10, 95))
    thr = max(floor + max(MIN_MARGIN_DB, ALPHA * (peak - floor)), peak - PEAK_REL_DB)
    t = (np.arange(len(e)) * hop + FRAME_LEN / 2) / sr
    on = e > thr
    segs: list[list[float]] = []
    for i in np.flatnonzero(on):
        s, end = t[i] - HOP_S / 2, t[i] + HOP_S / 2
        if segs and s - segs[-1][1] < MERGE_GAP_S:
            segs[-1][1] = end
        else:
            segs.append([s, end])
    dur = len(x) / sr
    segs = [s for s in segs if s[1] - s[0] >= MIN_SEG_S]
    out: list[tuple[float, float]] = []
    for i, (s, end) in enumerate(segs):
        gap_before = s - segs[i - 1][1] if i else None
        gap_after = segs[i + 1][0] - end if i + 1 < len(segs) else None
        s -= PAD_BEFORE_S if gap_before is None else min(PAD_BEFORE_S, gap_before / 4)
        end += PAD_AFTER_S if gap_after is None else min(PAD_AFTER_S, gap_after / 4)
        s, end = max(0.0, s), min(dur, end)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], end)
        else:
            out.append((s, end))
    return [(round(s, 3), round(e_, 3)) for s, e_ in out], {"t": t, "energy_db": e, "thr": thr, "floor": floor}


def plot(x: np.ndarray, segs, info: dict, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
    tt = np.arange(len(x)) / SAMPLE_RATE
    a.plot(tt, x, lw=0.4, color="0.3")
    b.plot(info["t"], info["energy_db"], lw=0.8, color="tab:blue", label="frame energy")
    b.axhline(info["thr"], color="tab:red", lw=0.8, ls="--", label="threshold")
    b.axhline(info["floor"], color="0.5", lw=0.8, ls=":", label="noise floor")
    for ax in (a, b):
        for s, e in segs:
            ax.axvspan(s, e, color="tab:green", alpha=0.18)
    a.set_title(f"{title}: green = labelled speech ({len(segs)} segments)")
    b.set_ylabel("dBFS")
    b.set_xlabel("s")
    b.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", type=Path, help="folder of clean WAVs")
    ap.add_argument("--force", action="store_true", help="overwrite an existing vad_labels.csv")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--plot-dir", type=Path, default=None, help=f"default {PLOT_DIR}/<clips folder name>")
    a = ap.parse_args()
    out_csv = a.clips / "vad_labels.csv"
    if out_csv.exists() and not a.force:
        raise SystemExit(f"{out_csv} exists (hand labels?). Use --force to overwrite.")
    wavs = sorted(a.clips.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files in {a.clips}")
    plot_dir = a.plot_dir or PLOT_DIR / a.clips.name
    rows = []
    for w in wavs:
        x = load_wav(w)
        segs, info = label_clip(x)
        rows += [(w.name, s, e) for s, e in segs]
        speech = sum(e - s for s, e in segs)
        gaps = [f"{segs[i][1]:.2f}-{segs[i + 1][0]:.2f}" for i in range(len(segs) - 1)]
        print(f"{w.name}: {len(segs)} segments, {speech:.1f} of {len(x) / SAMPLE_RATE:.1f} s speech; "
              f"non-speech gaps {gaps or '-'}")
        if not a.no_plots:
            plot(x, segs, info, w.name, plot_dir / f"{w.stem}.png")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "start_s", "end_s"])
        wr.writerows(rows)
    print(f"wrote {out_csv}" + ("" if a.no_plots else f" and plots in {plot_dir}"))


if __name__ == "__main__":
    main()
