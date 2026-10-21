"""Mix a clean clip with noise at a target SNR (seeded, reproducible).

SNR is measured on speech-active samples only when labels are given (otherwise
on the whole clip), because long pauses would otherwise deflate the speech RMS
and make the mix look easier than the stated SNR.
"""

from __future__ import annotations

import argparse
import zlib

import numpy as np
import soundfile as sf

from app.audio_io import load_wav
from app.config import EVAL, SAMPLE_RATE


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def speech_mask(n: int, segments: list[tuple[float, float]] | None, sr: int = SAMPLE_RATE) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in segments or []:
        m[int(s * sr) : int(e * sr)] = True
    return m if m.any() else np.ones(n, dtype=bool)


def noise_excerpt(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """A random length-n excerpt of `noise`, tiled if the noise file is too short."""
    if len(noise) < n:
        noise = np.tile(noise, n // len(noise) + 1)
    start = rng.integers(0, len(noise) - n + 1)
    return noise[start : start + n]


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    segments: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """clean + g*noise with 20*log10(rms(speech) / rms(g*noise)) == snr_db.
    Peak-normalised to 0.99 only if it would clip (scaling both keeps the SNR)."""
    nz = noise_excerpt(noise, len(clean), rng)
    s_rms = rms(clean[speech_mask(len(clean), segments)])
    g = s_rms / (rms(nz) * 10 ** (snr_db / 20))
    y = clean + g * nz
    peak = np.abs(y).max()
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def condition_rng(clip: str, noise_type: str, snr_db: float, seed: int = EVAL.seed) -> np.random.Generator:
    """Deterministic per-(clip, noise, snr) generator, independent of iteration order."""
    key = zlib.crc32(f"{clip}|{noise_type}|{snr_db}".encode())
    return np.random.default_rng([seed, key])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clean")
    p.add_argument("noise")
    p.add_argument("snr_db", type=float)
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=EVAL.seed)
    a = p.parse_args()
    y = mix_at_snr(load_wav(a.clean), load_wav(a.noise), a.snr_db, np.random.default_rng(a.seed))
    sf.write(a.out, y, SAMPLE_RATE, subtype="PCM_16")


if __name__ == "__main__":
    main()
