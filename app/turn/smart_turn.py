"""Smart Turn v3 as a verdict source for the TurnController."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.metrics.metrics import MetricsData, TurnMetricsData

from app import config
from app.turn.turn_policy import SmartTurnVerdict


class ThresholdSmartTurn(LocalSmartTurnAnalyzerV3):
    """Smart Turn v3 with a configurable P(complete) threshold (upstream hard-codes 0.5).

    `verdict()` predicts on the turn's audio so far WITHOUT clearing the buffer
    (upstream `analyze_end_of_turn` clears on COMPLETE). We may keep waiting after
    a COMPLETE (a filler), and if the worker resumes, the next prediction should
    still see the whole turn. The buffer is cleared when the turn really ends.
    """

    def __init__(self, *, threshold: float = config.SMART_TURN_THRESHOLD, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        state = super().append_audio(buffer, is_speech)
        # Upstream only trims between turns, and we no longer clear on COMPLETE: keep what the
        # model can use (max_duration_secs, plus the pre-speech margin) so a long turn stays cheap.
        keep_s = self._params.max_duration_secs + self._params.pre_speech_ms / 1000 + self._vad_start_secs
        oldest = time.monotonic() - keep_s
        buf = self._audio_buffer
        i = 0
        while i < len(buf) and buf[i][0] < oldest:
            i += 1
        if i:
            del buf[:i]
        return state

    def _predict_endpoint(self, audio_array: np.ndarray) -> dict[str, Any]:
        result = super()._predict_endpoint(audio_array)
        result["prediction"] = int(result["probability"] > self.threshold)
        return result

    async def verdict(self) -> tuple[SmartTurnVerdict | None, MetricsData | None]:
        loop = asyncio.get_running_loop()
        state, metrics = await loop.run_in_executor(self._executor, self._process_speech_segment, list(self._audio_buffer))
        if not isinstance(metrics, TurnMetricsData):
            return None, metrics
        return SmartTurnVerdict(state == EndOfTurnState.COMPLETE, metrics.probability, metrics.e2e_processing_time_ms), metrics
