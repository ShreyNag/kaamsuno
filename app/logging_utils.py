"""Per-hop timestamps as JSON lines: logs/<session_id>.jsonl, one event per line.

Every event carries wall-clock `ts` (epoch seconds), `session`, `turn`, and
`event`. `eval/latency_report.py` turns these into p50/p95 per hop.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from app.config import LOG_DIR


class HopLogger:
    """Append-only JSON-lines event log for one session."""

    def __init__(self, session_id: str | None = None, log_dir: Path = LOG_DIR):
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.turn = 0
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{self.session_id}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def log(self, event: str, **extra: Any) -> None:
        rec = {"ts": time.time(), "session": self.session_id, "turn": self.turn, "event": event, **extra}
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def next_turn(self) -> None:
        self.turn += 1

    def close(self) -> None:
        self._f.close()


class HopObserver(BaseObserver):
    """Maps pipeline frames to hop events, logging each frame once (on its first push;
    Pipecat 1.7 has no `observe_every_push=False`, so we dedupe by frame id).

    Turn numbering: a turn = one user turn plus the bot's reply. It advances when
    the bot starts speaking after the user's turn ended.
    """

    _TRACKED = (
        VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame,
        InterimTranscriptionFrame, TranscriptionFrame, TTSAudioRawFrame, BotStartedSpeakingFrame,
    )

    def __init__(self, hops: HopLogger):
        super().__init__()
        self._hops = hops
        self._awaiting_tts = False
        self._seen: deque[int] = deque(maxlen=512)

    async def on_push_frame(self, data: FramePushed):
        f, h = data.frame, self._hops
        if not isinstance(f, self._TRACKED) or f.id in self._seen:
            return
        self._seen.append(f.id)
        if isinstance(f, VADUserStartedSpeakingFrame):
            h.log("speech_start")
        elif isinstance(f, VADUserStoppedSpeakingFrame):
            h.log("speech_end")
        elif isinstance(f, UserStoppedSpeakingFrame):
            h.log("turn_end")
            self._awaiting_tts = True
        elif isinstance(f, InterimTranscriptionFrame):
            h.log("stt_partial", text=f.text)
        elif isinstance(f, TranscriptionFrame):
            h.log("stt_segment", text=f.text)
        elif isinstance(f, TTSAudioRawFrame) and self._awaiting_tts:
            h.log("tts_first_byte")
            self._awaiting_tts = False
        elif isinstance(f, BotStartedSpeakingFrame):
            h.log("playback_start")
            h.next_turn()
