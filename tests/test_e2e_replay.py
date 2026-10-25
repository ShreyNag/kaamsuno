"""End to end with the real services and Aastha's schema: WAV -> VAD -> real Prisma -> turn controller ->
real Groq interpreter -> dialogue -> reports/<session>.json. No TTS (the question text is captured).

Costs ~10 s of Prisma audio and 2-4 Groq calls, so it only runs with RUN_LIVE=1:

    $env:RUN_LIVE="1"; .venv\\Scripts\\python -m pytest -s tests\\test_e2e_replay.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from pipecat.frames.frames import EndFrame, InputAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_turn_processor import UserTurnProcessor

from app import config
from app.audio_io import load_wav
from app.interpreter.schema import SCHEMA_PATH, load_schema
from app.interpreter.templates import load_template_book
from app.logging_utils import HopLogger
from app.reporter import ReportProcessor
from app.stt import prisma_headers
from app.turns import build_turn_strategies, build_vad, turn_strategy

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="live services; set RUN_LIVE=1")
FIX = Path(__file__).parent / "fixtures"
SR, CHUNK = 16_000, 320


class Spoken(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.said: list[str] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            self.said.append(frame.text)
        await self.push_frame(frame, direction)


@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="needs Aastha's research/schema.json")
def test_missing_location_gets_a_where_question_and_the_report_is_saved(tmp_path):
    from app.bot import build_interpreter, build_stt

    schema, templates = load_schema(), load_template_book()
    assert schema.version.startswith("1.")  # Aastha's file, not the placeholder
    hops = HopLogger(log_dir=tmp_path)
    strategies = build_turn_strategies("smart_turn_fillers", log=hops.log)
    stt = build_stt(prisma_headers(), hops)
    reporter = ReportProcessor(session_id=hops.session_id, schema=schema, templates=templates,
                               interpreter=build_interpreter(schema, hops), stt=stt, hops=hops,
                               set_dialogue_context=turn_strategy(strategies).set_dialogue_context,
                               reports_dir=tmp_path / "reports")
    spoken = Spoken()
    task = PipelineTask(
        Pipeline([VADProcessor(vad_analyzer=build_vad("silero")), stt, UserTurnProcessor(user_turn_strategies=strategies),
                  reporter, spoken]),
        params=PipelineParams(audio_in_sample_rate=SR),
    )
    x = load_wav(FIX / "no_location.wav")
    pcm = (np.clip(np.concatenate([x, np.zeros(SR * 4, np.float32)]), -1, 1) * 32767).astype(np.int16).tobytes()

    async def feed():
        await asyncio.sleep(1.0)  # StartFrame everywhere, Prisma connected
        t0 = time.monotonic()
        for i, off in enumerate(range(0, len(pcm), CHUNK * 2)):  # real time, then 4 s of mic silence
            await asyncio.sleep(max(0.0, t0 + i * CHUNK / SR - time.monotonic()))
            await task.queue_frame(InputAudioRawFrame(audio=pcm[off : off + CHUNK * 2], sample_rate=SR, num_channels=1))
        await task.queue_frame(EndFrame())

    async def main():
        await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), feed())

    asyncio.run(main())
    hops.close()

    events = [json.loads(line) for line in hops.path.read_text(encoding="utf-8").splitlines()]
    questions = [(e["field"], e["reason"], e["text"]) for e in events if e["event"] == "question"]
    doc = json.loads((tmp_path / "reports" / f"{hops.session_id}.json").read_text(encoding="utf-8"))
    segments = doc["transcript"]  # what reached the interpreter
    filled = {k: v["value"] for k, v in doc["report"].items() if v["value"] is not None and v["evidence"] != "system"}
    print("\nsegments:", segments, "\nquestions:", questions, "\nfilled:", filled, "\nsaid:", spoken.said)

    assert segments, "Prisma transcripts never reached the pipeline"
    assert questions, "no turn ended"
    assert questions[-1][:2] == ("location", "missing"), questions  # the "where" question
    assert spoken.said[-1] == templates.get("location", "en")  # English speech gets the English question
    assert filled, "nothing the worker said reached reports/<session>.json"
    assert all(q[1] != "complete" for q in questions)
    assert config.SITE_NAME == doc["report"]["site_name"]["value"]
