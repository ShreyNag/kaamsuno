"""VAD comparison: custom vs Silero vs WebRTC vs composite, in noise, leave-one-clip-out.

    .venv/Scripts/python -m eval.vad_eval --clips data/vad_dev_tmp/clean
    .venv/Scripts/python -m eval.vad_eval --clips data/vad_dev --noise data/noise      # Aastha's real clips later
    .venv/Scripts/python -m eval.vad_eval --clips data/vad_dev_synth --quick           # smoke test

--clips: a folder of clean WAVs plus vad_labels.csv (file,start_s,end_s; eval.auto_label makes one).
--noise: real noise WAVs (any names, e.g. drilling.wav); white, pink and 50 Hz hum are generated too.

Each clean clip is mixed with each noise at each SNR (eval/mix_noise.py, seeded, SNR measured
on the labelled speech), plus the clean clip itself. Every VAD is scored per 10 ms frame:
precision, recall, F1, F2 (recall weighted 2x, the tuning objective) and the false-alarm
rate on noise-only frames.

Two decision views, from the same confidences:
    raw       the VAD's per-chunk decision (confidence >= threshold) at its native chunk size
    deployed  that decision run through Pipecat's start/stop counters (VAD_START_SECS / VAD_STOP_SECS),
              i.e. what the bot's turn logic actually sees (STOPPING still counts as speech)

Tuning: each VAD has one knob (custom: candidate share; Silero: confidence; WebRTC:
aggressiveness; composite: Silero confidence x custom share). Leave-one-clip-out: for every
clip, pick the knob with the best mean raw F2 on the OTHER clips (all conditions), then score
the held-out clip with it. Only held-out scores are reported.

Writes eval/out/vad_results.md and eval/out/vad_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio_io import load_labels, load_wav, segments_to_frames
from app.config import DATA_DIR, EVAL, ROOT, SAMPLE_RATE, VAD_START_SECS, VAD_STOP_SECS
from app.vad import custom_vad
from eval.mix_noise import condition_rng, mix_at_snr

SR = SAMPLE_RATE
HOP_S = 0.01  # scoring resolution
OUT_DIR = ROOT / "eval" / "out"

GRIDS: dict[str, list] = {
    "custom": [0.5, 1.0],  # share of 10 ms candidate frames in a 20 ms chunk
    "silero": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "webrtc": [0, 1, 2, 3],  # aggressiveness
    "composite": list(itertools.product([0.3, 0.5, 0.7], [0.34, 0.67, 1.0])),  # (silero conf, custom share)
}
CHUNK_S = {"custom": 0.02, "silero": 0.032, "webrtc": 0.03, "composite": 0.032}


# --- noise -----------------------------------------------------------------------------------
def synthetic_noises(n: int, seed: int = EVAL.seed) -> dict[str, np.ndarray]:
    rng = np.random.default_rng([seed, 99])
    white = rng.normal(size=n)
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    pink = np.fft.irfft(spec / np.sqrt(f), n)
    t = np.arange(n) / SR
    hum = sum(np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 8))
    hum = hum + 0.02 * rng.normal(size=n)  # mains hum with harmonics and a little hiss
    return {k: (v / np.std(v)).astype(np.float32) for k, v in {"white": white, "pink": pink, "hum50": hum}.items()}


def load_noises(noise_dir: Path | None, synthetic: bool) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if noise_dir and noise_dir.exists():
        for w in sorted(noise_dir.glob("*.wav")):
            out[w.stem] = load_wav(w)
    if not out:
        print(f"warning: no noise WAVs in {noise_dir}; synthetic noise only")
    if synthetic:
        out.update(synthetic_noises(SR * 60))
    return out


# --- per-VAD confidences -------------------------------------------------------------------
def chunk_starts(n_samples: int, chunk_s: float) -> np.ndarray:
    c = int(round(chunk_s * SR))
    return np.arange(0, n_samples - c + 1, c)


def custom_frames(x: np.ndarray) -> np.ndarray:
    """Per-10 ms-frame speech candidates (identical to the streaming/live wrapper)."""
    _, per = custom_vad.process_file(x)
    return per["candidate"]


def frame_share(cand: np.ndarray, x_len: int, chunk_s: float) -> np.ndarray:
    """Share of 10 ms candidate frames per chunk, exactly as the live wrapper computes it: a frame
    counts in the chunk where its last sample arrives (StreamingVAD can't finish it earlier), and a
    chunk that completes no frame repeats the previous share."""
    c = int(round(chunk_s * SR))
    n_chunks = len(chunk_starts(x_len, chunk_s))
    ends = np.arange(len(cand)) * custom_vad.VAD.hop + custom_vad.VAD.frame_len  # sample count when frame is done
    idx = -(-ends // c) - 1  # ceil(end / c) - 1
    keep = idx < n_chunks
    share, cnt = np.zeros(n_chunks), np.zeros(n_chunks)
    np.add.at(share, idx[keep], cand[keep].astype(float))
    np.add.at(cnt, idx[keep], 1)
    out, last = np.zeros(n_chunks), 0.0
    for i in range(n_chunks):
        last = share[i] / cnt[i] if cnt[i] else last
        out[i] = last
    return out


_silero = None


def silero_conf(x: np.ndarray) -> np.ndarray:
    global _silero
    if _silero is None:
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        _silero = SileroVADAnalyzer()
        _silero.set_sample_rate(SR)
    _silero._model.reset_states()
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    c = 512
    conf = [_silero.voice_confidence(pcm[s : s + c].tobytes()) for s in chunk_starts(len(x), CHUNK_S["silero"])]
    return np.array(conf, dtype=float).reshape(-1)  # the model returns 1-element arrays


def webrtc_dec(x: np.ndarray, aggr: int) -> np.ndarray:
    import webrtcvad

    v = webrtcvad.Vad(aggr)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    c = int(CHUNK_S["webrtc"] * SR)
    return np.array([v.is_speech(pcm[s : s + c].tobytes(), SR) for s in chunk_starts(len(x), CHUNK_S["webrtc"])])


@dataclass
class ClipConf:
    """Everything needed to score any knob value without re-running a model."""

    n: int
    custom_share20: np.ndarray
    custom_share32: np.ndarray
    silero: np.ndarray
    webrtc: dict[int, np.ndarray]


def confidences(x: np.ndarray) -> ClipConf:
    cand = custom_frames(x)
    return ClipConf(
        n=len(x),
        custom_share20=frame_share(cand, len(x), CHUNK_S["custom"]),
        custom_share32=frame_share(cand, len(x), CHUNK_S["composite"]),
        silero=silero_conf(x),
        webrtc={a: webrtc_dec(x, a) for a in GRIDS["webrtc"]},
    )


def chunk_decisions(cc: ClipConf, vad: str, knob) -> np.ndarray:
    if vad == "custom":
        return cc.custom_share20 >= knob
    if vad == "silero":
        return cc.silero >= knob
    if vad == "webrtc":
        return cc.webrtc[knob]
    conf, share = knob
    n = min(len(cc.silero), len(cc.custom_share32))
    return (cc.silero[:n] >= conf) & (cc.custom_share32[:n] >= share)


# --- Pipecat's start/stop counters ------------------------------------------------------------
def pipecat_counter(dec: np.ndarray, chunk_s: float, start_secs: float = VAD_START_SECS,
                    stop_secs: float = VAD_STOP_SECS) -> np.ndarray:
    """Replays VADAnalyzer._run_analyzer (one chunk per call): speaking = SPEAKING or STOPPING."""
    start_n, stop_n = round(start_secs / chunk_s), round(stop_secs / chunk_s)
    out = np.zeros(len(dec), dtype=bool)
    state, n_start, n_stop = "QUIET", 0, 0
    for i, sp in enumerate(dec):
        if sp:
            if state == "QUIET":
                state, n_start = "STARTING", 1
            elif state == "STARTING":
                n_start += 1
            elif state == "STOPPING":
                state, n_stop = "SPEAKING", 0
        else:
            if state == "STARTING":
                state, n_start = "QUIET", 0
            elif state == "SPEAKING":
                state, n_stop = "STOPPING", 1
            elif state == "STOPPING":
                n_stop += 1
        if state == "STARTING" and n_start >= start_n:
            state, n_start = "SPEAKING", 0
        if state == "STOPPING" and n_stop >= stop_n:
            state, n_stop = "QUIET", 0
        out[i] = state in ("SPEAKING", "STOPPING")
    return out


def to_frames(dec: np.ndarray, chunk_s: float, n_frames: int) -> np.ndarray:
    """Chunk decisions -> 10 ms frames (a frame takes the decision of the chunk containing its centre).
    Frames past the last full chunk are non-speech."""
    centres = (np.arange(n_frames) + 0.5) * HOP_S
    idx = (centres / chunk_s).astype(int)
    ok = idx < len(dec)
    out = np.zeros(n_frames, dtype=bool)
    out[ok] = dec[idx[ok]]
    return out


# --- scoring --------------------------------------------------------------------------------
def counts(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """[tp, fp, fn, tn]"""
    return np.array([np.sum(pred & truth), np.sum(pred & ~truth), np.sum(~pred & truth), np.sum(~pred & ~truth)])


def metrics(c: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = c.astype(float)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = lambda b: (1 + b * b) * p * r / (b * b * p + r) if p + r else 0.0  # noqa: E731
    return {"precision": p, "recall": r, "f1": f(1), "f2": f(2), "false_alarm": fp / (fp + tn) if fp + tn else 0.0}


def score(cc: ClipConf, vad: str, knob, truth: np.ndarray, deployed: bool) -> np.ndarray:
    dec = chunk_decisions(cc, vad, knob)
    if deployed:
        dec = pipecat_counter(dec, CHUNK_S[vad])
    return counts(to_frames(dec, CHUNK_S[vad], len(truth)), truth)


# --- main -----------------------------------------------------------------------------------
def run(clips_dir: Path, noise_dir: Path | None, snrs: list[float], synthetic: bool, quick: bool) -> list[dict]:
    labels = load_labels(clips_dir / "vad_labels.csv")
    names = sorted(n for n in labels if (clips_dir / n).exists())
    missing = sorted(set(labels) - set(names))
    if missing:
        print(f"warning: labelled but missing WAVs: {missing}")
    noises = load_noises(noise_dir, synthetic)
    if quick:
        names, noises, snrs = names[:3], dict(list(noises.items())[:2]), snrs[:1]
    if len(names) < 2:
        raise SystemExit("leave-one-clip-out needs at least 2 labelled clips")
    conditions = [("clean", None)] + [(nz, snr) for nz in noises for snr in snrs]
    print(f"{len(names)} clips x {len(conditions)} conditions; noises: {list(noises)}; SNRs: {snrs}")

    # confidences for every (clip, condition), computed once
    data: dict[tuple[str, str], tuple[ClipConf, np.ndarray]] = {}
    t0 = time.time()
    for name in names:
        clean = load_wav(clips_dir / name)
        n_frames = int(len(clean) / SR / HOP_S)
        truth = segments_to_frames(labels[name], (np.arange(n_frames) + 0.5) * HOP_S)
        for nz, snr in conditions:
            x = clean if nz == "clean" else mix_at_snr(clean, noises[nz], snr, condition_rng(name, nz, snr), labels[name])
            data[(name, f"{nz}" if snr is None else f"{nz}@{snr:g}dB")] = (confidences(x), truth)
        print(f"  {name} done ({time.time() - t0:.0f} s)")

    cond_names = sorted({c for _, c in data}, key=lambda c: (c != "clean", c))
    rows: list[dict] = []
    for vad, grid in GRIDS.items():
        # raw counts for every (clip, knob), summed over conditions, for tuning
        per_clip = {n: {k: sum(score(*data[(n, c)][:1], vad, k, data[(n, c)][1], False) for c in cond_names)
                        for k in grid} for n in names}
        for held in names:
            train = [n for n in names if n != held]
            best = max(grid, key=lambda k: np.mean([metrics(per_clip[n][k])["f2"] for n in train]))
            for c in cond_names:
                cc, truth = data[(held, c)]
                for view in ("raw", "deployed"):
                    tp, fp, fn, tn = score(cc, vad, best, truth, view == "deployed")
                    m = metrics(np.array([tp, fp, fn, tn]))
                    rows.append({"vad": vad, "view": view, "held_out": held, "condition": c, "knob": str(best),
                                 "tp": tp, "fp": fp, "fn": fn, "tn": tn, **{k: round(v, 4) for k, v in m.items()}})
    return rows


def agg(rows: list[dict], **where) -> dict[str, float]:
    sel = [r for r in rows if all(r[k] == v for k, v in where.items())]
    return metrics(np.sum([[r["tp"], r["fp"], r["fn"], r["tn"]] for r in sel], axis=0)) if sel else {}


def write_report(rows: list[dict], clips_dir: Path, args) -> str:
    vads = list(GRIDS)
    conds = sorted({r["condition"] for r in rows}, key=lambda c: (c != "clean", c))
    noises = sorted({c.split("@")[0] for c in conds if c != "clean"})
    snrs = sorted({c.split("@")[1] for c in conds if c != "clean"})
    lines = [
        "# VAD comparison (held-out clips only)", "",
        f"Data: `{clips_dir}` ({len({r['held_out'] for r in rows})} clips). Noises: {', '.join(noises)} "
        f"(from `{args.noise}`{' + generated' if not args.no_synthetic_noise else ''}); SNRs {', '.join(snrs)}. "
        f"Frames of {HOP_S * 1000:.0f} ms; metrics pooled over frames (micro). Tuning objective: F2 on the other clips.",
        "", "## Overall", "",
        "| VAD | view | precision | recall | F1 | F2 | false alarm (noise-only) | knob chosen (per fold) |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for vad in vads:
        knobs = sorted({r["knob"] for r in rows if r["vad"] == vad})
        for view in ("raw", "deployed"):
            m = agg(rows, vad=vad, view=view)
            lines.append(f"| {vad} | {view} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                         f"{m['f2']:.3f} | {m['false_alarm']:.3f} | {', '.join(knobs)} |")
    for metric, label in (("recall", "Recall"), ("false_alarm", "False-alarm rate on noise-only frames"), ("f1", "F1")):
        lines += ["", f"## {label} by condition (raw view)", "", "| condition | " + " | ".join(vads) + " |",
                  "|---|" + "---:|" * len(vads)]
        for c in conds:
            lines.append(f"| {c} | " + " | ".join(f"{agg(rows, vad=v, view='raw', condition=c)[metric]:.3f}"
                                                   for v in vads) + " |")
    lines += ["", "Raw = per-chunk decision; deployed = after Pipecat's start/stop counters "
              f"({VAD_START_SECS}/{VAD_STOP_SECS} s), what the turn logic sees. Labels from "
              "`eval/auto_label.py` are padded generously, so precision/false alarm near segment edges is pessimistic."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=DATA_DIR / "vad_dev_tmp" / "clean")
    ap.add_argument("--noise", type=Path, default=DATA_DIR / "noise")
    ap.add_argument("--snr", type=float, nargs="+", default=list(EVAL.snrs_db))
    ap.add_argument("--no-synthetic-noise", action="store_true", help="skip generated white/pink/hum50")
    ap.add_argument("--quick", action="store_true", help="3 clips, 2 noises, 1 SNR (smoke test)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    rows = run(a.clips, a.noise, a.snr, not a.no_synthetic_noise, a.quick)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "vad_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = write_report(rows, a.clips, a)
    (a.out / "vad_results.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote {a.out / 'vad_results.md'} and {a.out / 'vad_results.csv'}")


if __name__ == "__main__":
    main()
