import numpy as np
import pytest

from app.config import VAD
from app.vad.custom_vad import Hysteresis, StreamingVAD, frames_to_segments, process_file

SR = VAD.sample_rate


def speechlike(seed: int = 0) -> np.ndarray:
    """1 s low noise, 1 s harmonic 'vowel', 0.15 s gap, 0.8 s vowel, 1 s low noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(SR) / SR
    vowel = sum(np.sin(2 * np.pi * 150 * h * t) / h for h in range(1, 12)) * 0.05
    quiet = lambda s: rng.normal(0, 1e-3, int(s * SR))  # noqa: E731
    return np.concatenate([quiet(1), vowel, quiet(0.15), vowel[: int(0.8 * SR)], quiet(1)]).astype(np.float32)


def test_hysteresis_enter_and_hangover():
    h = Hysteresis()
    hang = h.hangover_frames
    cand = np.array([1, 1, 0, 1, 1, 1] + [0] * (hang - 1) + [1] + [0] * hang + [0], dtype=bool)
    out = h.run_all(cand)
    assert not out[:5].any()  # a 2-frame burst doesn't enter; 3rd consecutive frame does
    assert out[5]
    assert out[5 : 6 + hang].all()  # gap shorter than hangover is bridged
    assert not out[-1]


def test_frames_to_segments():
    speech = np.zeros(100, dtype=bool)
    speech[10:20] = speech[50:60] = True
    segs = frames_to_segments(speech)
    assert len(segs) == 2
    assert segs[0] == pytest.approx((0.1075, 0.2075), abs=1e-3)  # 12.5 ms centre offset, ±5 ms


def test_offline_finds_speech_and_bridges_short_gap():
    segs, pf = process_file(speechlike())
    assert len(segs) == 1, segs
    s, e = segs[0]
    assert 0.95 < s < 1.1 and 2.9 < e < 3.3


@pytest.mark.parametrize("chunk", [7, 37, 160, 320, 512, 4000])
def test_streaming_equals_offline(chunk):
    x = speechlike(seed=3)
    _, pf = process_file(x)
    vad = StreamingVAD()
    res = []
    for i in range(0, len(x), chunk):
        res += vad.push(x[i : i + chunk])
    assert len(res) == len(pf["speech"])
    np.testing.assert_array_equal([r.candidate for r in res], pf["candidate"])
    np.testing.assert_array_equal([r.speech for r in res], pf["speech"])
    np.testing.assert_allclose([r.confidence for r in res], pf["confidence"])


def test_streaming_accepts_int16_bytes():
    x = speechlike()
    pcm = (x * 32767).astype(np.int16).tobytes()
    res = StreamingVAD().push(pcm)
    assert any(r.speech for r in res)
