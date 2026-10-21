"""ReportProcessor: segments only update the report; one template question per turn end."""

import asyncio
import json

from pipecat.frames.frames import (
    EndFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.interpreter.interpreter import Interpreter
from app.interpreter.schema import PLACEHOLDER_SCHEMA, PLACEHOLDER_TEMPLATES, parse_schema
from app.logging_utils import HopLogger
from app.report_server import ReportHub
from app.reporter import ReportProcessor

SCHEMA = parse_schema(PLACEHOLDER_SCHEMA)


class Spoken(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.said: list[tuple[str, int]] = []  # (text, frames seen before it)
        self.n = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.n += 1
        if isinstance(frame, TTSSpeakFrame):
            self.said.append((frame.text, self.n))
        await self.push_frame(frame, direction)


async def fake_llm(messages):
    seg = json.loads(messages[1]["content"])["new_segment"]
    await asyncio.sleep(0.2)
    if "manzil" in seg:
        return json.dumps({"updates": {"location": {"value": "3rd floor", "confidence": 0.9, "evidence": "teesri manzil"}}})
    return json.dumps({"updates": {"incident_type": {"value": "material_issue", "confidence": 0.9,
                                                      "evidence": "cement kam hai"}}})


def test_segments_are_silent_and_turn_end_asks_top_missing_field(tmp_path):
    hub, hops = ReportHub(), HopLogger("t", log_dir=tmp_path)
    reporter = ReportProcessor(session_id="t", schema=SCHEMA, templates=PLACEHOLDER_TEMPLATES,
                               interpreter=Interpreter(SCHEMA, fake_llm, system_prompt=""), stt=None,
                               hops=hops, hub=hub)
    spoken = Spoken()
    task = PipelineTask(Pipeline([reporter, spoken]), params=PipelineParams())

    async def feed():
        await asyncio.sleep(0.3)
        await task.queue_frame(UserStartedSpeakingFrame())
        await task.queue_frame(TranscriptionFrame("dedh sau bori cement kam hai", "u", "now"))
        await task.queue_frame(TranscriptionFrame("teesri manzil pe", "u", "now"))
        await asyncio.sleep(0.1)
        assert spoken.said == []  # nothing spoken mid-turn
        await task.queue_frame(UserStoppedSpeakingFrame())
        await asyncio.sleep(0.8)
        await task.queue_frame(EndFrame())

    async def main():
        await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), feed())

    asyncio.run(main())
    hops.close()
    # incident_type and location came from the segments, so the next critical field is asked
    assert [t for t, _ in spoken.said] == [PLACEHOLDER_TEMPLATES["equipment_or_material"]]
    snap = hub.snapshots["t"]
    assert snap["question"]["field"] == "equipment_or_material"
    assert snap["segments"] == ["dedh sau bori cement kam hai", "teesri manzil pe"]
    assert {f["name"]: f["value"] for f in snap["fields"]}["location"] == "3rd floor"


def test_resuming_speech_cancels_the_pending_question(tmp_path):
    async def slow_llm(messages):
        await asyncio.sleep(0.5)
        return json.dumps({"updates": {}})

    hops = HopLogger("t2", log_dir=tmp_path)
    reporter = ReportProcessor(session_id="t2", schema=SCHEMA, templates=PLACEHOLDER_TEMPLATES,
                               interpreter=Interpreter(SCHEMA, slow_llm, system_prompt=""), stt=None,
                               hops=hops, hub=ReportHub())
    spoken = Spoken()
    task = PipelineTask(Pipeline([reporter, spoken]), params=PipelineParams())

    async def feed():
        await asyncio.sleep(0.3)
        await task.queue_frame(TranscriptionFrame("umm", "u", "now"))
        await asyncio.sleep(0.05)  # system frames overtake queued data frames; live, Prisma is never this close
        await task.queue_frame(UserStoppedSpeakingFrame())
        await asyncio.sleep(0.1)  # question still waiting on the interpreter...
        await task.queue_frame(UserStartedSpeakingFrame())  # ...and the worker carries on
        await asyncio.sleep(0.8)
        await task.queue_frame(EndFrame())

    async def main():
        await asyncio.gather(PipelineRunner(handle_sigint=False).run(task), feed())

    asyncio.run(main())
    hops.close()
    assert spoken.said == []
