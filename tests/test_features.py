import numpy as np
import pytest

from app.config import VAD
from app.vad.features import (
    adaptive_threshold_db,
    band_energies,
    compute_features,
    energy_gate,
    frame_energy_db,
    frame_signal,
    n_frames,
    peak_concentration,
    power_spectrum,
    spectral_flatness,
    zero_crossing_rate,
)

SR = VAD.sample_rate


def sine(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * np.arange(int(seconds * SR)) / SR)).astype(np.float32)


def white(seconds: float = 1.0, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    return (amp * np.random.default_rng(seed).normal(size=int(seconds * SR))).astype(np.float32)


@pytest.mark.parametrize("n,expected", [(0, 0), (399, 0), (400, 1), (559, 1), (560, 2), (16000, 98)])
def test_frame_count(n, expected):
    assert n_frames(n) == expected
    assert frame_signal(np.zeros(n, dtype=np.float32)).shape == (expected, VAD.frame_len)


def test_frame_hop_contents():
    x = np.arange(1000, dtype=np.float32)
    f = frame_signal(x)
    assert f[0, 0] == 0 and f[1, 0] == VAD.hop and f[2, -1] == 2 * VAD.hop + VAD.frame_len - 1


def test_sine_is_tonal():
    p = power_spectrum(frame_signal(sine(700)))
    assert np.median(spectral_flatness(p, VAD.flatness_lo_hz, VAD.flatness_hi_hz)) < 0.1
    assert np.median(peak_concentration(band_energies(p))) > 0.9


def test_white_noise_is_flat():
    p = power_spectrum(frame_signal(white()))
    assert np.median(spectral_flatness(p, VAD.flatness_lo_hz, VAD.flatness_hi_hz)) > 0.8
    assert np.median(spectral_flatness(p, 0, SR / 2)) > 0.8
    # band PSDs are roughly equal -> concentration near 1 / n_bands
    assert np.median(peak_concentration(band_energies(p))) < 0.3


def test_digital_silence_fails_gate():
    e = frame_energy_db(frame_signal(np.zeros(SR, dtype=np.float32)))
    assert not energy_gate(e).any()


def test_gate_opens_on_onset_after_noise_floor():
    x = np.concatenate([white(2.0, amp=0.001), sine(300, 0.5, amp=0.3)])
    f = compute_features(x)
    onset = f["t"] > 2.05
    assert f["gate"][onset].mean() > 0.9
    assert f["gate"][f["t"] < 1.9].mean() < 0.3


def test_threshold_is_causal():
    e = np.random.default_rng(1).normal(-40, 5, 600)
    thr_full, _, _ = adaptive_threshold_db(e)
    thr_prefix, _, _ = adaptive_threshold_db(e[:350])
    np.testing.assert_allclose(thr_full[:350], thr_prefix)


def test_zcr_extremes():
    assert np.median(zero_crossing_rate(frame_signal(sine(100)))) < 0.05
    assert np.median(zero_crossing_rate(frame_signal(white()))) > 0.4
