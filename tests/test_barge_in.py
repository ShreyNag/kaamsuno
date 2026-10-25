"""Barge-in: the worker starts talking while the bot is speaking -> InterruptionFrame at the bot's
audio output within 400 ms of speech onset, through the real VAD, our turn strategy and the reporter.

The output is a stand-in (Pipecat's real output transport stops on InterruptionFrame by itself; what
we own, and what could break, is everything upstream of it). Real time, ~5 s.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_processor import UserTurnProcessor

from app.audio_io import load_wav
from app.interpreter.interpreter import Interpreter
from app.interpreter.schema import PLACEHOLDER_SCHEMA, PLACEHOLDER_TEMPLATES, parse_schema
from app.logging_utils import HopLogger
from app.report_server import ReportHub
from app.reporter import ReportProcessor
from app.turns import build_turn_strategies, build_vad, turn_strategy

FIX = Path(__file__).parent / "fixtures"
META = json.loads((FIX / "mid_sentence_pause.json").read_text())
ONSET = META["speech"][0][0]
SR, CHUNK = 16_000, 320


class FakeBotOutput(FrameProcessor):
    """Says it is speaking from 0.2 s into the WAV until an InterruptionFrame arrives."""

    def __init__(self, t0: list[float]):
        super().__init__()
        self.t0 = t0
        self.interrupted_at: float | None = None
        self.speaking = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self.create_task(self._start_talking(), "fake_bot")
            return
        if isinstance(frame, InterruptionFrame) and self.speaking and self.interrupted_at is None:
            self.interrupted_at = time.monotonic() - self.t0[0]
            self.speaking = False
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)

    async def _start_talking(self):
        while not self.t0[0]:
            await asyncio.sleep(0.01)
        await asyncio.sleep(max(0.0, self.t0[0] + 0.2 - time.monotonic()))
        self.speaking = True
        await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)


@pytest.mark.parametrize("mode", ["fixed_timeout", "smart_turn", "smart_turn_fillers"])
def test_user_speech_during_bot_audio_interrupts_within_400ms(tmp_path, mode):
    async def llm(messages):
        return json.dumps({"updates": {}})

    schema = parse_schema(PLACEHOLDER_SCHEMA)
    t0 = [0.0]
    hops = HopLogger("barge", log_dir=tmp_path)
    strategies = build_turn_strategies(mode)
    reporter = ReportProcessor(session_id="barge", schema=schema, templates=PLACEHOLDER_TEMPLATES,
                               interpreter=Interpreter(schema, llm, system_prompt="", uncertainty_cue=lambda t: None),
                               stt=None, hops=hops, hub=ReportHub(),
                               set_dialogue_context=turn_strategy(strategies).set_dialogue_context)
    bot = FakeBotOutput(t0)
    task = PipelineTask(
        Pipeline([VADProcessor(vad_analyzer=build_vad("silero")), UserTurnProcessor(user_turn_strategies=strategies),
                  reporter, bot]),
        params=PipelineParams(audio_in_sample_rate=SR),
    )
    pcm = (np.clip(load_wav(FIX / "mid_sentence_pause.wav")[: int(3 * SR)], -1, 1) * 32767).astype(np.int16).tobytes()

    async def feed():
        await asyncio.sleep(0.5)
        t0[0] = time.monotonic()
        for i, off in enumerate(range(0, len(pcm), CHUNK * 2)):
            await asyncio.sleep(max(0.0, t0[0] + i * CHUNK / SR - time.monotonic()))
            await task.queue_frame(InputAudioRawFrame(audio=pcm[off : off + CHUNK * 2], sample_rate=SR, num_channels=1))
        await task.queue_frame(EndFrame())

    async def main():
        await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), feed())

    asyncio.run(main())
    hops.close()
    assert bot.interrupted_at is not None, "the bot was never interrupted"
    reaction = bot.interrupted_at - ONSET
    print(f"{mode}: speech onset {ONSET:.2f} s, interruption at the bot output {bot.interrupted_at:.2f} s "
          f"-> {reaction * 1000:.0f} ms")
    assert 0 < reaction <= 0.4
