"""Pipecat pipeline entry point (Phase 0: Gnani's foundational example, driven by config).

    .venv/Scripts/python -m app.bot -t webrtc      then open http://localhost:7860/client
"""

from __future__ import annotations

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat_gnani import GnaniSTTService, GnaniTTSService

from app import config
from app.logging_utils import HopLogger, HopObserver

SYSTEM_PROMPT = (
    "You are a site-safety assistant. Keep answers to one or two short sentences, "
    "in the same mix of Hindi and English the worker uses. No formatting."
)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


def _require(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def build_stt() -> GnaniSTTService:
    return GnaniSTTService(
        api_key=_require("GNANI_API_KEY", config.GNANI_API_KEY),
        sample_rate=config.SAMPLE_RATE,
        settings=GnaniSTTService.Settings(
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


def build_llm() -> GroqLLMService:
    if config.LLM_PROVIDER != "groq":
        raise NotImplementedError(f"LLM_PROVIDER={config.LLM_PROVIDER!r}; only groq is wired up")
    return GroqLLMService(
        api_key=_require("GROQ_API_KEY", config.GROQ_API_KEY),
        settings=GroqLLMService.Settings(model=config.LLM_MODEL, system_instruction=SYSTEM_PROMPT),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    hops = HopLogger()
    logger.info(f"hop log: {hops.path}")

    stt, tts, llm = build_stt(), build_tts(), build_llm()
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [transport.input(), stt, aggregators.user(), llm, tts, transport.output(), aggregators.assistant()]
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
        await task.queue_frames(
            [LLMMessagesAppendFrame(messages=[{"role": "user", "content": "Namaste"}], run_llm=True)]
        )

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

    from pipecat.runner.run import main

    # The runner's banner uses box-drawing chars; Windows consoles default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
