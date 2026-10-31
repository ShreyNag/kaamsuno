"""Where does the turn end on a sentence with a 1.5 s mid-sentence pause?

Streams a fixture WAV in real time through the same VAD + UserTurnProcessor
(+ our TurnController strategy) the bot uses, and records when
UserStoppedSpeakingFrame comes out. Prisma isn't called: where a test needs
segments it injects TranscriptionFrames at the time Prisma would send them.
Real time because the turn timers are asyncio sleeps. ~10 s per case.
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
    TranscriptionFrame,
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


def run_wav(strategies, t0: list[float] | None = None, wav: str = "mid_sentence_pause",
            segments: list[tuple[float, str]] = ()) -> Recorder:
    """t0[0] is set to the monotonic time of the WAV's first sample. `segments`
    are (t, text) TranscriptionFrames injected at WAV time t, standing in for Prisma."""
    pcm = (np.clip(load_wav(FIX / f"{wav}.wav"), -1, 1) * 32767).astype(np.int16).tobytes()
    pending = sorted(segments)
    t0 = t0 if t0 is not None else [0.0]
    rec = Recorder(t0)
    task = PipelineTask(
        Pipeline([VADProcessor(vad_analyzer=build_vad("silero")), UserTurnProcessor(user_turn_strategies=strategies), rec]),
        params=PipelineParams(audio_in_sample_rate=SR),
    )

    async def feed():
        await asyncio.sleep(0.5)  # let StartFrame reach every processor
        t0[0] = time.monotonic()
        for i, off in enumerate(range(0, len(pcm), CHUNK * 2)):
            await asyncio.sleep(max(0.0, t0[0] + i * CHUNK / SR - time.monotonic()))
            while pending and pending[0][0] <= i * CHUNK / SR:
                await task.queue_frame(TranscriptionFrame(text=pending.pop(0)[1], user_id="", timestamp=""))
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


# --- the filler rule on a real WAV: 1.5 s pause after "matlab" ------------------------------------
MATLAB = json.loads((FIX / "matlab_pause.json").read_text())
M_PAUSE_START, M_PAUSE_END = MATLAB["pause"]
M_SPEECH_END = MATLAB["speech"][-1][1]
PRISMA_LAG = 0.45  # last speech -> segment: 250 ms min silence + processing
M_SEGMENTS = [(M_PAUSE_START + PRISMA_LAG, MATLAB["text"][0]), (M_SPEECH_END + PRISMA_LAG, MATLAB["text"][1])]


def _matlab(mode: str, analyzer=None) -> Recorder:
    t0 = [0.0]
    if isinstance(analyzer, type):
        analyzer = analyzer(t0, params=SmartTurnParams(stop_secs=3.0))
    rec = run_wav(build_turn_strategies(mode, turn_analyzer=analyzer), t0, wav="matlab_pause", segments=M_SEGMENTS)
    print(mode, "verdicts:", [(round(t, 2), c, round(p, 3)) for t, c, p in rec.verdicts],
          "turn ends:", [round(t, 2) for t in rec.turn_ends])
    return rec


class AlwaysCompleteSmartTurn(ScriptedSmartTurn):
    """The worst case seen live: Smart Turn says complete (0.98) at a mid-sentence pause."""

    def _predict_endpoint(self, audio_array):
        return {"prediction": 1, "probability": 0.98}


def test_matlab_fixed_timeout_cuts_in():
    rec = _matlab("fixed_timeout")
    assert rec.turn_ends[0] == pytest.approx(M_PAUSE_START + 0.7, abs=TOL)


def test_matlab_smart_turn_alone_cuts_in_when_it_says_complete():
    rec = _matlab("smart_turn", AlwaysCompleteSmartTurn)
    assert rec.turn_ends[0] < M_PAUSE_END


@pytest.mark.parametrize("analyzer", [AlwaysCompleteSmartTurn, None], ids=["st_says_complete", "real_smart_turn"])
def test_matlab_fillers_waits_through_the_pause_and_replies_fast_after_bas(analyzer):
    rec = _matlab("smart_turn_fillers", analyzer)
    assert len(rec.turn_ends) == 1, rec.turn_ends  # no cut-in after "matlab"
    # "...bas": completion -> turn over as soon as the segment arrives
    assert rec.turn_ends[0] == pytest.approx(M_SPEECH_END + PRISMA_LAG, abs=TOL)


# --- the new policy on the same recording: tone AND words before a fast reply ----------------------
@pytest.mark.parametrize("before", ["teesri manzil pe cement aur", "pump ke paas", "teesri manzil pe umm",
                                    "machine ID chaar saat", "cement teesri manzil pe"])
def test_matlab_audio_mid_sentence_tails_hold_the_turn(before):
    """The 1.5 s pause in matlab_pause.wav with each incomplete tail as Prisma's text, Smart Turn forced to
    0.98 (the worst case seen live): no turn end in the pause."""
    t0 = [0.0]
    segs = [(M_PAUSE_START + PRISMA_LAG, before), (M_SPEECH_END + PRISMA_LAG, MATLAB["text"][1])]
    rec = run_wav(build_turn_strategies("smart_turn_fillers", turn_analyzer=AlwaysCompleteSmartTurn(
        t0, params=SmartTurnParams(stop_secs=3.0))), t0, wav="matlab_pause", segments=segs)
    assert len(rec.turn_ends) == 1 and rec.turn_ends[0] > M_PAUSE_END, rec.turn_ends


@pytest.mark.parametrize("last, analyzer", [("mixer kharab hai", AlwaysCompleteSmartTurn), ("bas itna hi", None)],
                         ids=["mixer_kharab_hai_st_complete", "bas_itna_hi_real_smart_turn"])
def test_matlab_audio_finished_sentence_ends_within_half_a_second(last, analyzer):
    """After the second phrase: "mixer kharab hai" (tone complete + words fine) and "bas itna hi" (completion
    words) end the turn ~0.5 s after the speech ends (the segment arrives at +0.45 s)."""
    t0 = [0.0]
    an = analyzer(t0, params=SmartTurnParams(stop_secs=3.0)) if analyzer else None
    segs = [(M_PAUSE_START + PRISMA_LAG, MATLAB["text"][0]), (M_SPEECH_END + PRISMA_LAG, last)]
    rec = run_wav(build_turn_strategies("smart_turn_fillers", turn_analyzer=an), t0, wav="matlab_pause", segments=segs)
    assert len(rec.turn_ends) == 1, rec.turn_ends
    assert rec.turn_ends[0] - M_SPEECH_END <= 0.5 + TOL, rec.turn_ends
