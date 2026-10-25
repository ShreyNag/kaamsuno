"""Pipecat pipeline entry point: the voice incident reporter.

    .venv/Scripts/python -m app.bot -t webrtc      then open http://localhost:7860/client
                                                    and the report at http://localhost:7860/report/latest

Pipeline: mic -> VAD -> Prisma STT -> turn controller (app/turn/) -> reporter -> Timbre TTS -> speaker.
There is no chat LLM in the loop: every Prisma segment only updates the report
(via the interpreter LLM, JSON out), and at turn end the reporter speaks one
template question from dialogue.py.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat_gnani import GnaniTTSService

from app import config
from app.interpreter.interpreter import Interpreter, groq_complete
from app.interpreter.schema import load_schema, load_templates
from app.logging_utils import HopLogger, HopObserver
from app.reporter import ReportProcessor
from app.stt import PrismaSTTService, prisma_headers
from app.turns import build_turn_strategies, build_vad, describe, turn_strategy

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def _require(name: str, value: str) -> str:
    if not value or value.startswith("paste-your-"):
        raise RuntimeError(f"{name} is not set. Put your key in .env (replace the placeholder).")
    return value


def build_stt(extra_headers: dict[str, str]) -> PrismaSTTService:
    return PrismaSTTService(
        api_key=_require("GNANI_API_KEY", config.GNANI_API_KEY),
        sample_rate=config.SAMPLE_RATE,
        extra_headers=extra_headers,
        settings=PrismaSTTService.Settings(
            language=Language(config.LANGUAGE_CODE),
            format=config.PRISMA.output_format,
        ),
    )


def build_tts() -> GnaniTTSService:
    return GnaniTTSService(
        api_key=_require("GNANI_API_KEY", config.GNANI_API_KEY),
        sample_rate=config.TTS_SAMPLE_RATE,
        settings=GnaniTTSService.Settings(
            voice=config.TTS_VOICE,
            model=config.TTS_MODEL,
            language=config.TTS_LANGUAGE if config.TTS_MODEL == "timbre-v2.5" else None,
        ),
    )


def build_interpreter(schema, hops: HopLogger) -> Interpreter:
    if config.LLM_PROVIDER != "groq":
        raise NotImplementedError(f"LLM_PROVIDER={config.LLM_PROVIDER!r}; only groq is wired up")
    _require("GROQ_API_KEY", config.GROQ_API_KEY)
    return Interpreter(schema, groq_complete(config.LLM_MODEL), log=hops.log)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    hops = HopLogger()
    logger.info(f"hop log: {hops.path}")
    logger.info(f"live report: http://localhost:7860/report/{hops.session_id}  (or /report/latest)")

    headers = prisma_headers()
    vad, strategies = build_vad(), build_turn_strategies(log=hops.log)
    logger.info("\n" + describe(vad, config.TURN_MODE, strategies, headers))

    schema, templates = load_schema(), load_templates()
    stt = build_stt(headers)
    reporter = ReportProcessor(
        session_id=hops.session_id, schema=schema, templates=templates,
        interpreter=build_interpreter(schema, hops), stt=stt, hops=hops,
        set_dialogue_context=turn_strategy(strategies).set_dialogue_context,
    )
    turns = UserTurnProcessor(user_turn_strategies=strategies, user_turn_stop_timeout=config.USER_TURN_STOP_TIMEOUT_S)

    pipeline = Pipeline(
        [transport.input(), VADProcessor(vad_analyzer=vad), stt, turns, reporter, build_tts(), transport.output()]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[HopObserver(hops)],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([TTSSpeakFrame(config.GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)
    finally:
        hops.close()


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import sys

    from pipecat.runner.run import app, main

    from app.report_server import register_routes

    # The runner's banner uses box-drawing chars; Windows consoles default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    register_routes(app)
    main()
