"""Speech labels from CLEAN recordings, by energy, with generous padding. A stand-in for hand labels.

    .venv/Scripts/python -m eval.auto_label data/vad_dev_tmp/clean            # writes vad_labels.csv + plots
    .venv/Scripts/python -m eval.auto_label data/vad_dev_tmp/clean --force    # overwrite existing labels

Per file: 10 ms frame energy (dBFS); threshold = noise floor (10th percentile) +
max(MIN_MARGIN_DB, ALPHA * (peak - floor)), peak = 95th percentile. Frames above
it are speech; gaps shorter than MERGE_GAP_S are bridged (word gaps), segments
shorter than MIN_SEG_S dropped, then each segment padded by PAD_BEFORE_S /
PAD_AFTER_S. Real mid-sentence pauses (>= ~0.6 s) survive as non-speech.

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
MERGE_GAP_S = 0.3
MIN_SEG_S = 0.1
PAD_BEFORE_S = 0.15
PAD_AFTER_S = 0.2
PLOT_DIR = ROOT / "eval" / "out" / "labels"


def label_clip(x: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[list[tuple[float, float]], dict]:
    hop = int(HOP_S * sr)
    e = F.frame_energy_db(F.frame_signal(x, FRAME_LEN, hop))
    floor, peak = np.percentile(e, (10, 95))
    thr = floor + max(MIN_MARGIN_DB, ALPHA * (peak - floor))
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
    out: list[tuple[float, float]] = []
    for s, end in segs:
        if end - s < MIN_SEG_S:
            continue
        s, end = max(0.0, s - PAD_BEFORE_S), min(dur, end + PAD_AFTER_S)
        if out and s <= out[-1][1]:  # padding made them touch
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
    a = ap.parse_args()
    out_csv = a.clips / "vad_labels.csv"
    if out_csv.exists() and not a.force:
        raise SystemExit(f"{out_csv} exists (hand labels?). Use --force to overwrite.")
    wavs = sorted(a.clips.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files in {a.clips}")
    rows = []
    for w in wavs:
        x = load_wav(w)
        segs, info = label_clip(x)
        rows += [(w.name, s, e) for s, e in segs]
        speech = sum(e - s for s, e in segs)
        print(f"{w.name}: {len(segs)} segments, {speech:.1f} of {len(x) / SAMPLE_RATE:.1f} s speech")
        if not a.no_plots:
            plot(x, segs, info, w.name, PLOT_DIR / a.clips.name / f"{w.stem}.png")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "start_s", "end_s"])
        wr.writerows(rows)
    print(f"wrote {out_csv}" + ("" if a.no_plots else f" and plots in {PLOT_DIR / a.clips.name}"))


if __name__ == "__main__":
    main()
