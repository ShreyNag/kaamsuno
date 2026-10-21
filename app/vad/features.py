"""Pure numpy VAD features. Every function is causal (uses only the current and
past frames) so the same code serves offline evaluation and live streaming.

Shapes: `frames` is (n_frames, frame_len); `power` is (n_frames, n_fft//2 + 1).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.config import VAD, VADConfig

EPS = 1e-12


# --- Framing -----------------------------------------------------------------
def n_frames(n_samples: int, frame_len: int = VAD.frame_len, hop: int = VAD.hop) -> int:
    """Number of full frames; a trailing partial frame is dropped (it would need future audio)."""
    if n_samples < frame_len:
        return 0
    return 1 + (n_samples - frame_len) // hop


def frame_signal(x: np.ndarray, frame_len: int = VAD.frame_len, hop: int = VAD.hop) -> np.ndarray:
    """Split a 1-D signal into overlapping frames (a strided view, no copy)."""
    n = n_frames(len(x), frame_len, hop)
    if n == 0:
        return np.zeros((0, frame_len), dtype=np.float32)
    return sliding_window_view(x, frame_len)[::hop][:n]


def frame_times(n: int, cfg: VADConfig = VAD) -> np.ndarray:
    """Centre time (s) of each frame."""
    return (np.arange(n) * cfg.hop + cfg.frame_len / 2) / cfg.sample_rate


# --- Energy ------------------------------------------------------------------
def frame_energy_db(frames: np.ndarray) -> np.ndarray:
    """Mean-square energy of each (unwindowed) frame in dBFS (full scale = 1.0)."""
    return 10.0 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPS)


def running_rms_db(energy_db: np.ndarray, cfg: VADConfig = VAD, init_db: float | None = None) -> np.ndarray:
    """Causal EMA of frame RMS (in the power domain), returned in dB.

    `init_db` lets a streaming caller continue from a previous state.
    """
    alpha = 1.0 - np.exp(-cfg.hop / (cfg.rms_tau_s * cfg.sample_rate))
    p = 10.0 ** (energy_db / 10.0)
    out = np.empty_like(p)
    state = p[0] if init_db is None and len(p) else 10.0 ** ((init_db or 0.0) / 10.0)
    for i, v in enumerate(p):  # first-order IIR; n is small (100 frames/s)
        state += alpha * (v - state)
        out[i] = state
    return 10.0 * np.log10(out + EPS)


def adaptive_threshold_db(energy_db: np.ndarray, cfg: VADConfig = VAD) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature A. Rolling (past `gate_window_frames`, current included) floor/peak
    percentiles of log energy and the resulting threshold. Returns (threshold, floor, peak).

    During warm-up the window is simply shorter (expanding window).
    """
    w = cfg.gate_window_frames
    padded = np.concatenate([np.full(w - 1, np.nan), energy_db])
    win = sliding_window_view(padded, w)
    floor = np.nanpercentile(win, cfg.gate_floor_pct, axis=1)
    peak = np.nanpercentile(win, cfg.gate_peak_pct, axis=1)
    thr = floor + np.maximum(cfg.gate_alpha * (peak - floor), cfg.gate_min_margin_db)
    return thr, floor, peak


def energy_gate(energy_db: np.ndarray, cfg: VADConfig = VAD) -> np.ndarray:
    """Hard gate: above the adaptive threshold AND above the absolute silence floor."""
    thr, _, _ = adaptive_threshold_db(energy_db, cfg)
    return (energy_db > thr) & (energy_db > cfg.abs_floor_dbfs)


# --- Spectrum ----------------------------------------------------------------
def power_spectrum(frames: np.ndarray, n_fft: int = VAD.n_fft) -> np.ndarray:
    """Hamming-windowed periodogram, (n_frames, n_fft//2 + 1)."""
    win = np.hamming(frames.shape[1])
    spec = np.fft.rfft(frames * win, n=n_fft, axis=1)
    return (spec.real**2 + spec.imag**2) / np.sum(win**2)


def fft_freqs(cfg: VADConfig = VAD) -> np.ndarray:
    return np.fft.rfftfreq(cfg.n_fft, 1.0 / cfg.sample_rate)


def band_energies(power: np.ndarray, cfg: VADConfig = VAD) -> np.ndarray:
    """Mean power per band (a density, so wide bands aren't favoured), (n_frames, n_bands)."""
    f = fft_freqs(cfg)
    edges = cfg.band_edges_hz
    out = np.empty((power.shape[0], len(edges) - 1))
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (f >= lo) & (f < hi) if b < len(edges) - 2 else (f >= lo) & (f <= hi)
        out[:, b] = power[:, sel].mean(axis=1)
    return out


def _smooth_freq(power: np.ndarray, n_bins: int) -> np.ndarray:
    """Centred moving average across frequency (zero-padded at the edges), vectorised over frames."""
    if n_bins <= 1:
        return power
    left = n_bins // 2
    c = np.cumsum(np.pad(power, ((0, 0), (left + 1, n_bins - 1 - left))), axis=1)
    return (c[:, n_bins:] - c[:, :-n_bins]) / n_bins


def spectral_flatness(power: np.ndarray, lo_hz: float, hi_hz: float, cfg: VADConfig = VAD) -> np.ndarray:
    """Feature B. Geometric / arithmetic mean of the (frequency-smoothed) power
    spectrum between lo_hz and hi_hz. ~1 = flat noise, ~0 = tonal / structured."""
    f = fft_freqs(cfg)
    p = _smooth_freq(power, cfg.flatness_smooth_bins)[:, (f >= lo_hz) & (f <= hi_hz)] + EPS
    return np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)


def peak_concentration(bands: np.ndarray) -> np.ndarray:
    """Feature C. Largest band PSD / sum of band PSDs (1/n_bands for white noise, ~1 for a tone)."""
    return bands.max(axis=1) / (bands.sum(axis=1) + EPS)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Feature D. Fraction of adjacent sample pairs whose sign differs."""
    s = np.signbit(frames)
    return np.mean(s[:, 1:] != s[:, :-1], axis=1)


# --- All features at once ----------------------------------------------------
def compute_features(x: np.ndarray, cfg: VADConfig = VAD) -> dict[str, np.ndarray]:
    """Every per-frame feature and threshold for one clip (float32 in [-1, 1], 16 kHz)."""
    frames = frame_signal(x, cfg.frame_len, cfg.hop)
    e_db = frame_energy_db(frames)
    thr, floor, peak = adaptive_threshold_db(e_db, cfg)
    power = power_spectrum(frames, cfg.n_fft)
    bands = band_energies(power, cfg)
    return {
        "t": frame_times(len(frames), cfg),
        "energy_db": e_db,
        "rms_db": running_rms_db(e_db, cfg) if len(e_db) else e_db,
        "gate_thr_db": thr,
        "gate_floor_db": floor,
        "gate_peak_db": peak,
        "gate": (e_db > thr) & (e_db > cfg.abs_floor_dbfs),
        "flatness": spectral_flatness(power, cfg.flatness_lo_hz, cfg.flatness_hi_hz, cfg),
        "flatness_full": spectral_flatness(power, 0.0, cfg.sample_rate / 2, cfg),
        "peak_conc": peak_concentration(bands),
        "zcr": zero_crossing_rate(frames),
        "bands": bands,
    }


def feature_votes(feats: dict[str, np.ndarray], cfg: VADConfig = VAD) -> dict[str, np.ndarray]:
    """The three per-feature speech votes (the ZCR rule is documented in notes/decisions.md)."""
    voiced = feats["zcr"] < cfg.zcr_voiced_max
    fricative = (feats["zcr"] > cfg.zcr_fricative_min) & (feats["flatness_full"] < cfg.fricative_flatness_max)
    return {
        "flatness": feats["flatness"] < cfg.flatness_max,
        "peak_conc": feats["peak_conc"] > cfg.peak_conc_min,
        "zcr": voiced | fricative,
    }
