"""CustomVAD: features -> per-frame candidate -> hysteresis + hangover -> segments.

Two interfaces with identical per-frame output:
  * `process_file(wav_or_array)` for offline eval
  * `StreamingVAD.push(pcm)` for live audio in arbitrary-length chunks

Everything is causal: the decision for frame i uses only samples up to the end
of frame i, so streaming never needs to revise a past decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import VAD, VADConfig
from app.vad import features as F

Segment = tuple[float, float]


# --- Pure decision functions -------------------------------------------------
def candidates(feats: dict[str, np.ndarray], cfg: VADConfig = VAD) -> tuple[np.ndarray, np.ndarray]:
    """(speech candidate, confidence in [0, 1]) per frame.
    candidate = gate AND >= min_votes of 3; confidence = gate * votes / 3."""
    n_votes = sum(v.astype(int) for v in F.feature_votes(feats, cfg).values())
    gate = feats["gate"]
    return gate & (n_votes >= cfg.min_votes), gate * n_votes / 3.0


@dataclass
class Hysteresis:
    """Enter speech after `enter_frames` consecutive candidates; once in speech,
    stay until `hangover` consecutive non-candidates. Stateful so it can stream."""

    cfg: VADConfig = VAD
    in_speech: bool = False
    run: int = 0  # consecutive candidates while out of speech
    silence: int = 0  # consecutive non-candidates while in speech

    @property
    def hangover_frames(self) -> int:
        return max(self.cfg.exit_frames, round(self.cfg.hangover_ms * self.cfg.sample_rate / 1000 / self.cfg.hop))

    def step(self, cand: bool) -> bool:
        if self.in_speech:
            self.silence = 0 if cand else self.silence + 1
            if self.silence >= self.hangover_frames:
                self.in_speech, self.silence = False, 0
        else:
            self.run = self.run + 1 if cand else 0
            if self.run >= self.cfg.enter_frames:
                self.in_speech, self.run = True, 0
        return self.in_speech

    def run_all(self, cand: np.ndarray) -> np.ndarray:
        return np.fromiter((self.step(bool(c)) for c in cand), dtype=bool, count=len(cand))


def frames_to_segments(speech: np.ndarray, cfg: VADConfig = VAD) -> list[Segment]:
    """Boolean per-frame array -> [(start_s, end_s)], each frame covering centre ± hop/2."""
    if not len(speech):
        return []
    d = np.diff(np.concatenate([[0], speech.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    centre0 = cfg.frame_len / 2 / cfg.sample_rate
    hop_s = cfg.hop / cfg.sample_rate
    return [(round(centre0 + (s - 0.5) * hop_s, 3), round(centre0 + (e - 0.5) * hop_s, 3)) for s, e in zip(starts, ends)]


# --- Offline -----------------------------------------------------------------
def process_file(wav: str | Path | np.ndarray, cfg: VADConfig = VAD) -> tuple[list[Segment], dict[str, np.ndarray]]:
    """Offline VAD. Returns (segments, per-frame dict with features, candidate, confidence, speech)."""
    if isinstance(wav, (str, Path)):
        from app.audio_io import load_wav

        wav = load_wav(wav, cfg.sample_rate)
    feats = F.compute_features(wav, cfg)
    cand, conf = candidates(feats, cfg)
    speech = Hysteresis(cfg).run_all(cand)
    per_frame = {**feats, "candidate": cand, "confidence": conf, "speech": speech}
    return frames_to_segments(speech, cfg), per_frame


# --- Streaming ---------------------------------------------------------------
@dataclass
class FrameResult:
    index: int
    t: float  # frame centre, seconds since stream start
    candidate: bool
    confidence: float
    speech: bool


@dataclass
class StreamingVAD:
    """Push PCM in any chunk size; get one FrameResult per completed 10 ms hop.

    Keeps the last frame_len - hop samples (frame overlap) and the last
    gate_window_frames energies, so per-frame output equals `process_file`.
    """

    cfg: VADConfig = VAD
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _energy_hist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _n_done: int = 0
    _hyst: Hysteresis | None = None

    def __post_init__(self) -> None:
        self._hyst = Hysteresis(self.cfg)

    def reset(self) -> None:
        self.__init__(self.cfg)

    @property
    def in_speech(self) -> bool:
        return self._hyst.in_speech

    def push(self, pcm: bytes | np.ndarray) -> list[FrameResult]:
        if isinstance(pcm, (bytes, bytearray, memoryview)):
            pcm = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, np.asarray(pcm, dtype=np.float32)])
        frames = F.frame_signal(self._buf, self.cfg.frame_len, self.cfg.hop)
        if not len(frames):
            return []
        n = len(frames)
        self._buf = self._buf[n * self.cfg.hop :]

        cfg = self.cfg
        e_db = F.frame_energy_db(frames)
        hist = np.concatenate([self._energy_hist, e_db])
        thr = F.adaptive_threshold_db(hist, cfg)[0][-n:]
        self._energy_hist = hist[-(cfg.gate_window_frames - 1) :]
        power = F.power_spectrum(frames, cfg.n_fft)
        feats = {
            "gate": (e_db > thr) & (e_db > cfg.abs_floor_dbfs),
            "flatness": F.spectral_flatness(power, cfg.flatness_lo_hz, cfg.flatness_hi_hz, cfg),
            "flatness_full": F.spectral_flatness(power, 0.0, cfg.sample_rate / 2, cfg),
            "peak_conc": F.peak_concentration(F.band_energies(power, cfg)),
            "zcr": F.zero_crossing_rate(frames),
        }
        cand, conf = candidates(feats, cfg)
        idx = np.arange(self._n_done, self._n_done + n)
        times = (idx * cfg.hop + cfg.frame_len / 2) / cfg.sample_rate
        self._n_done += n
        return [
            FrameResult(int(i), float(t), bool(c), float(p), self._hyst.step(bool(c)))
            for i, t, c, p in zip(idx, times, cand, conf)
        ]
