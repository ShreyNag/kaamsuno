"""Diagnostic plots: waveform, RMS + energy gate, spectrogram, each feature with
its threshold, per-feature votes / decisions, and ground truth, on one time axis.

    python -m eval.plots                      # all dev clips, clean + drilling/generator @ 5 dB
    python -m eval.plots --noise all --snr 10 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.audio_io import dev_dirs, load_labels, load_wav, segments_to_frames
from app.config import EVAL, ROOT, VAD
from app.vad.features import compute_features, feature_votes
from eval.mix_noise import condition_rng, mix_at_snr

OUT = ROOT / "eval" / "out" / "plots"


def _raster(ax, t: np.ndarray, rows: dict[str, np.ndarray], title: str) -> None:
    names = list(rows)
    img = np.vstack([rows[n].astype(float) for n in names])
    ax.imshow(img, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1,
              extent=(t[0], t[-1], len(names) - 0.5, -0.5))
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_ylabel(title, fontsize=8)


def plot_clip(x: np.ndarray, labels: list[tuple[float, float]], title: str, out: Path,
              decisions: dict[str, np.ndarray] | None = None) -> None:
    f = compute_features(x)
    v = feature_votes(f)
    t = f["t"]
    truth = segments_to_frames(labels, t)

    fig, axes = plt.subplots(8, 1, figsize=(14, 15), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 1.5, 1, 1, 1, 1.1, 0.4]})
    ts = np.arange(len(x)) / VAD.sample_rate

    def shade(ax):
        for s, e in labels:
            ax.axvspan(s, e, color="tab:green", alpha=0.12, lw=0)

    ax = axes[0]
    ax.plot(ts, x, lw=0.4, color="k")
    shade(ax)
    ax.set_ylabel("wave", fontsize=8)

    ax = axes[1]
    ax.plot(t, f["energy_db"], lw=0.6, label="frame energy")
    ax.plot(t, f["rms_db"], lw=1.0, label=f"running RMS (tau {VAD.rms_tau_s}s)")
    ax.plot(t, f["gate_floor_db"], lw=0.8, ls=":", c="grey", label="floor p10")
    ax.plot(t, f["gate_peak_db"], lw=0.8, ls=":", c="grey", label="peak p90")
    ax.plot(t, f["gate_thr_db"], lw=1.2, c="tab:red", label="A: gate threshold")
    ax.axhline(VAD.abs_floor_dbfs, lw=0.8, ls="--", c="tab:purple", label="abs floor")
    shade(ax)
    ax.set_ylabel("dBFS", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=3)

    ax = axes[2]
    ax.specgram(x + 1e-9, NFFT=VAD.n_fft, Fs=VAD.sample_rate, noverlap=VAD.n_fft - VAD.hop, cmap="magma")
    for edge in VAD.band_edges_hz[1:-1]:
        ax.axhline(edge, c="w", lw=0.4, ls=":")
    ax.set_ylabel("Hz", fontsize=8)

    for ax, key, thr, name in [
        (axes[3], "flatness", VAD.flatness_max, "B: flatness (<)"),
        (axes[4], "peak_conc", VAD.peak_conc_min, "C: peak conc (>)"),
    ]:
        ax.plot(t, f[key], lw=0.6)
        ax.axhline(thr, c="tab:red", lw=1)
        shade(ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel(name, fontsize=8)

    ax = axes[5]
    ax.plot(t, f["zcr"], lw=0.6, label="ZCR")
    ax.plot(t, f["flatness_full"], lw=0.6, alpha=0.7, label="full-band flatness")
    ax.axhline(VAD.zcr_voiced_max, c="tab:red", lw=1, label="voiced max")
    ax.axhline(VAD.zcr_fricative_min, c="tab:orange", lw=1, label="fricative min")
    ax.axhline(VAD.fricative_flatness_max, c="tab:orange", lw=0.8, ls="--", label="hiss flatness")
    shade(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("D: ZCR rule", fontsize=8)
    ax.legend(fontsize=6, loc="upper right", ncol=5)

    rows = {"A gate": f["gate"], "B flat": v["flatness"], "C peak": v["peak_conc"], "D zcr": v["zcr"]}
    for name, d in (decisions or {}).items():
        rows[name] = d[: len(t)]
    _raster(axes[6], t, rows, "votes")
    _raster(axes[7], t, {"truth": truth}, "")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--noise", nargs="*", default=["drilling", "generator"], help="noise types, or 'all'")
    p.add_argument("--snr", nargs="*", type=float, default=[5.0])
    p.add_argument("--clips", nargs="*", default=None, help="subset of clip filenames")
    a = p.parse_args()

    clips_dir, noise_dir = dev_dirs()
    labels = load_labels(clips_dir / "vad_labels.csv")
    noise_types = list(EVAL.noise_types) if a.noise == ["all"] else a.noise
    noises = {n: load_wav(noise_dir / f"{n}.wav") for n in noise_types}

    for clip in a.clips or sorted(labels):
        x = load_wav(clips_dir / clip)
        stem = Path(clip).stem
        plot_clip(x, labels[clip], f"{clip} | clean", OUT / f"{stem}__clean.png")
        for n, noise in noises.items():
            for snr in a.snr:
                y = mix_at_snr(x, noise, snr, condition_rng(clip, n, snr), labels[clip])
                plot_clip(y, labels[clip], f"{clip} | {n} @ {snr:g} dB", OUT / f"{stem}__{n}_{snr:g}dB.png")
        print("plotted", clip)


if __name__ == "__main__":
    main()
