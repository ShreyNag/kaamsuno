"""Where does the turn end on a sentence with a 1.5 s mid-sentence pause?

Streams tests/fixtures/mid_sentence_pause.wav in real time through the same
VAD + UserTurnProcessor the bot uses, and records when UserStoppedSpeakingFrame
comes out. Real time because Pipecat's turn timers are asyncio sleeps. ~10 s per case.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    MetricsFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_processor import UserTurnProcessor

from app.audio_io import load_wav
from app.turns import ThresholdSmartTurn, build_turn_strategies, build_vad

FIX = Path(__file__).parent / "fixtures"
META = json.loads((FIX / "mid_sentence_pause.json").read_text())
PAUSE_START, PAUSE_END = META["pause"]
SPEECH_END = META["speech"][-1][1]
SR, CHUNK = 16_000, 320  # 20 ms chunks, like WebRTC input
TOL = 0.3  # VAD onset/offset jitter + scheduling


class Recorder(FrameProcessor):
    def __init__(self, t0_ref: list[float]):
        super().__init__()
        self.t0 = t0_ref
        self.turn_ends: list[float] = []
        self.vad_stops: list[float] = []
        self.verdicts: list[tuple[float, bool, float]] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.monotonic() - self.t0[0]
        if isinstance(frame, UserStoppedSpeakingFrame):
            self.turn_ends.append(now)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.vad_stops.append(now)
        elif isinstance(frame, MetricsFrame):
            self.verdicts += [(now, m.is_complete, m.probability) for m in frame.data if isinstance(m, TurnMetricsData)]
        await self.push_frame(frame, direction)


def run_wav(strategies, t0: list[float] | None = None) -> Recorder:
    """t0[0] is set to the monotonic time of the WAV's first sample."""
    pcm = (np.clip(load_wav(FIX / "mid_sentence_pause.wav"), -1, 1) * 32767).astype(np.int16).tobytes()
    t0 = t0 if t0 is not None else [0.0]
    rec = Recorder(t0)
    task = PipelineTask(
        Pipeline([VADProcessor(vad_analyzer=build_vad()), UserTurnProcessor(user_turn_strategies=strategies), rec]),
        params=PipelineParams(audio_in_sample_rate=SR),
    )

    async def feed():
        await asyncio.sleep(0.5)  # let StartFrame reach every processor
        t0[0] = time.monotonic()
        for i, off in enumerate(range(0, len(pcm), CHUNK * 2)):
            await asyncio.sleep(max(0.0, t0[0] + i * CHUNK / SR - time.monotonic()))
            await task.queue_frame(InputAudioRawFrame(audio=pcm[off : off + CHUNK * 2], sample_rate=SR, num_channels=1))
        await task.queue_frame(EndFrame())

    async def main():
        await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), feed())

    asyncio.run(main())
    return rec


def test_fixed_timeout_700ms_ends_turn_inside_the_pause():
    rec = run_wav(build_turn_strategies("fixed_timeout", fixed_timeout_ms=700))
    assert rec.turn_ends, "no turn end at all"
    first = rec.turn_ends[0]
    # 1.5 s pause > 700 ms: a fixed timeout cuts the worker off mid-sentence...
    assert first == pytest.approx(PAUSE_START + 0.7, abs=TOL), rec.turn_ends
    assert first < PAUSE_END
    # ...and the second half is a new turn ending 700 ms after the speech.
    assert rec.turn_ends[-1] == pytest.approx(SPEECH_END + 0.7, abs=TOL), rec.turn_ends


def test_fixed_timeout_2000ms_waits_through_the_pause():
    rec = run_wav(build_turn_strategies("fixed_timeout", fixed_timeout_ms=2000))
    assert len(rec.turn_ends) == 1, rec.turn_ends
    assert rec.turn_ends[0] == pytest.approx(SPEECH_END + 2.0, abs=TOL)


class ScriptedSmartTurn(ThresholdSmartTurn):
    """Smart Turn with the model replaced: 'complete' only once the sentence is really over."""

    def __init__(self, t0_ref: list[float], **kw):
        super().__init__(**kw)
        self._t0 = t0_ref

    def _predict_endpoint(self, audio_array):
        done = time.monotonic() - self._t0[0] > SPEECH_END
        return {"prediction": int(done), "probability": 0.95 if done else 0.05}


def test_smart_turn_ends_only_when_analyzer_says_complete():
    t0 = [0.0]
    analyzer = ScriptedSmartTurn(t0, params=SmartTurnParams(stop_secs=3.0))
    rec = run_wav(build_turn_strategies("smart_turn", turn_analyzer=analyzer), t0)
    assert [v for v in rec.verdicts if not v[1]], "analyzer never consulted during the pause"
    assert len(rec.turn_ends) == 1, rec.turn_ends  # pause did NOT end the turn
    assert rec.turn_ends[0] > PAUSE_END
    # VAD stop (0.2 s) + inference, no extra STT wait
    assert rec.turn_ends[0] == pytest.approx(SPEECH_END + 0.2, abs=TOL), rec.turn_ends


def test_smart_turn_real_model_turn_end_matches_a_complete_verdict():
    """Real Smart Turn v3 on the fixture. Whatever it decides about the pause, a turn
    may only end on a COMPLETE verdict or on the 3 s silence fallback."""
    rec = run_wav(build_turn_strategies("smart_turn"))
    print("verdicts (t, complete, p):", [(round(t, 2), c, round(p, 3)) for t, c, p in rec.verdicts])
    print("turn ends:", [round(t, 2) for t in rec.turn_ends])
    assert rec.turn_ends
    complete_at = [t for t, c, _ in rec.verdicts if c]
    for t in rec.turn_ends:
        on_verdict = any(abs(t - c) < 0.1 for c in complete_at)
        on_fallback = any(t - s >= 3.0 - TOL for s in rec.vad_stops if s < t)
        assert on_verdict or on_fallback, (t, rec.verdicts, rec.vad_stops)
