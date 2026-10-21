"""Audio loading and label helpers shared by eval and tests."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.config import DATA_DIR, SAMPLE_RATE

log = logging.getLogger(__name__)


def load_wav(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read any soundfile-supported file as float32 mono at `sr` (resampled if needed)."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def load_labels(csv_path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """vad_labels.csv (file,start_s,end_s; one speech segment per row) -> {file: [(start, end)]}."""
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["file"]].append((float(row["start_s"]), float(row["end_s"])))
    return {k: sorted(v) for k, v in out.items()}


def segments_to_frames(segments: list[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    """Boolean per-frame labels: a frame is speech if its centre time is inside a segment."""
    y = np.zeros(len(times), dtype=bool)
    for s, e in segments:
        y |= (times >= s) & (times < e)
    return y


def dev_dirs() -> tuple[Path, Path]:
    """(clips dir, noise dir): Aastha's real data if present, else the synthetic stand-ins."""
    clips = DATA_DIR / "vad_dev"
    if not (clips / "vad_labels.csv").exists():
        log.warning("data/vad_dev/vad_labels.csv missing; using synthetic data/vad_dev_synth")
        clips = DATA_DIR / "vad_dev_synth"
    noise = DATA_DIR / "noise"
    if not any(noise.glob("*.wav")):
        log.warning("data/noise/ empty; using synthetic data/noise_synth")
        noise = DATA_DIR / "noise_synth"
    return clips, noise
