"""Pipecat glue: the TurnController as a user-turn stop strategy (Pipecat 1.7 `turns` API).

The UserTurnProcessor hands every frame, in both directions, to its stop
strategies, so this one sees VAD start/stop, input audio (for Smart Turn),
Prisma segments, and the bot's BotStarted/StoppedSpeaking. It translates them
into TurnController calls, runs the controller's timer, and calls
`trigger_user_turn_stopped()` when the controller says the turn is over.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import BaseUserTurnStopStrategy

from app.turn.controller import TurnController
from app.turn.smart_turn import ThresholdSmartTurn
from app.turn.tail_classifier import DialogueContext


class PolicyTurnStopStrategy(BaseUserTurnStopStrategy):
    def __init__(self, controller: TurnController, *, turn_analyzer: ThresholdSmartTurn | None = None, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self._analyzer = turn_analyzer if controller.mode != "fixed_timeout" else None
        self._vad_speaking = False
        self._timer: asyncio.Task | None = None

    @property
    def turn_analyzer(self) -> ThresholdSmartTurn | None:
        return self._analyzer

    def set_dialogue_context(self, context: DialogueContext) -> None:
        self.controller.context = context

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        await super().process_frame(frame)
        c, over = self.controller, False
        if isinstance(frame, StartFrame):
            if self._analyzer:
                self._analyzer.set_sample_rate(frame.audio_in_sample_rate)
        elif isinstance(frame, InputAudioRawFrame):
            if self._analyzer:
                self._analyzer.append_audio(frame.audio, self._vad_speaking)  # its own silence timeout is unused
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_speaking = True
            if self._analyzer:
                self._analyzer.update_vad_start_secs(frame.start_secs)
            over = c.user_started(time.time())
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
            verdict = None
            if self._analyzer:
                verdict, metrics = await self._analyzer.verdict()
                if metrics:
                    await self.push_frame(MetricsFrame(data=[metrics]))  # HopObserver logs it as turn_verdict
            over = c.user_stopped(frame.timestamp, frame.stop_secs, verdict)
        elif isinstance(frame, TranscriptionFrame):
            over = c.segment(time.time(), frame.text)
        elif isinstance(frame, InterimTranscriptionFrame):
            over = c.partial(time.time(), frame.text)
        elif isinstance(frame, BotStartedSpeakingFrame):
            c.bot_started(time.time())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            c.bot_stopped(time.time())
        await self._after(over)
        return ProcessFrameResult.CONTINUE

    async def _after(self, over: bool) -> None:
        """Turn over now, or (re)arm the timer for the controller's next deadline."""
        await self._cancel_timer()
        if over:
            await self.trigger_user_turn_stopped()
            return
        wake = self.controller.next_wakeup()
        if wake is not None:
            self._timer = self.task_manager.create_task(self._sleep_until(wake), f"{self}::turn_timer")

    async def _sleep_until(self, wake: float) -> None:
        try:
            await asyncio.sleep(max(0.0, wake - time.time()))
        except asyncio.CancelledError:
            return
        self._timer = None
        await self._after(self.controller.tick(time.time()))

    async def _cancel_timer(self) -> None:
        t, self._timer = self._timer, None
        if t and t is not asyncio.current_task():
            await self.task_manager.cancel_task(t)

    async def handle_user_turn_stopped(self):
        self.controller.turn_stopped_externally(time.time())  # no-op when we ended it
        await self._cancel_timer()
        if self._analyzer:
            self._analyzer.clear()

    async def cleanup(self):
        await super().cleanup()
        await self._cancel_timer()
        if self._analyzer:
            await self._analyzer.cleanup()
        logger.debug(f"{self}: cleaned up")
