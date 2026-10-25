"""Our DSP VAD as Pipecat VADAnalyzers, so VAD_MODE=custom|composite runs live.

Pipecat's VADAnalyzer calls `voice_confidence(buffer)` on fixed-size chunks and
applies its own start_secs / stop_secs counters and `confidence` threshold.
We return the share of 10 ms frames in the chunk that our features call a speech
candidate (gate AND >= 2 of 3 votes, `custom_vad.candidates`), and let
Pipecat's counters play the part of our offline Hysteresis (enter/hangover).
That way the VADUserStoppedSpeakingFrame's `stop_secs` is the real stop delay,
which the turn controller needs to know when the worker actually went quiet.

    custom     candidate share >= CUSTOM_VAD_CONFIDENCE
    composite  Silero confidence, but zeroed unless our VAD also calls the chunk speech
               (custom AND Silero: noise Silero mistakes for speech is usually flat/broadband)
"""

from __future__ import annotations

import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer

from app.config import VAD, VADConfig
from app.vad.custom_vad import StreamingVAD


class _CustomStream:
    def __init__(self, cfg: VADConfig):
        self.vad = StreamingVAD(cfg)
        self.last = 0.0

    def share(self, buffer: bytes) -> float:
        """Share of the frames completed by this chunk that are speech candidates.
        A chunk too short to complete a frame repeats the previous answer."""
        res = self.vad.push(buffer)
        if res:
            self.last = float(np.mean([r.candidate for r in res]))
        return self.last


class CustomVADAnalyzer(VADAnalyzer):
    def __init__(self, *, cfg: VADConfig = VAD, chunk_ms: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._cfg, self._chunk_ms = cfg, chunk_ms
        self._stream = _CustomStream(cfg)

    def set_sample_rate(self, sample_rate: int):
        if sample_rate != self._cfg.sample_rate:
            raise ValueError(f"custom VAD is configured for {self._cfg.sample_rate} Hz, input is {sample_rate} Hz")
        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        return self._cfg.sample_rate * self._chunk_ms // 1000

    def voice_confidence(self, buffer: bytes) -> float:
        return self._stream.share(buffer)


class CompositeVADAnalyzer(SileroVADAnalyzer):
    """Silero's 512-sample chunks; speech only where our VAD agrees."""

    def __init__(self, *, cfg: VADConfig = VAD, min_share: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._stream = _CustomStream(cfg)
        self._min_share = min_share
        self.last_custom_share = 0.0  # for debugging / logs

    def voice_confidence(self, buffer) -> float:
        silero = super().voice_confidence(buffer)
        self.last_custom_share = self._stream.share(buffer)
        return silero if self.last_custom_share >= self._min_share else 0.0
