"""PrismaSTTService reads both Prisma protocols (recorded from the live API by eval/prisma_probe.py)."""

from __future__ import annotations

import asyncio

import pytest

from app import config
from app.stt import PrismaSTTService, prisma_headers

# Real message shapes, trimmed (eval/prisma_probe.py, 2026-10-25)
LEGACY = [
    {"type": "speech_start", "utterance_id": "u1"},
    {"type": "transcript", "text": "t sree manzil pe cement matlab", "utterance_id": "u1", "is_final": True},
]
PARTIALS = [
    {"type": "speech_start", "utterance_id": "u2"},
    {"type": "partial_transcript", "utterance_id": "u2", "seq": 1, "is_final": False, "text": "t three manzil p",
     "stable_text": "", "volatile_text": "t three manzil p"},
    {"type": "final_transcript", "text": "t sree manzil pe cement matlab", "utterance_id": "u2", "is_final": True,
     "output": {"literal": "t sree manzil pe cement matlab"}},
]


class Rec(PrismaSTTService):
    def __init__(self, **kw):
        self.events: list[tuple] = []
        super().__init__(api_key="test", sample_rate=16000, extra_headers={},
                         log=lambda event, **f: self.events.append((event, f)), **kw)
        self.pushed: list[tuple[str, bool]] = []
        self.errors: list[str] = []

    async def _push_transcript(self, text, *, is_final, language=None, result=None):
        self.pushed.append((text, is_final))

    async def push_error(self, error_msg="", exception=None, **kw):
        self.errors.append(error_msg)


def run(svc: Rec, messages: list[dict]) -> None:
    async def go():
        for m in messages:
            await svc.handle_message(m)

    asyncio.run(go())


def test_legacy_protocol_final_transcript():
    s = Rec()
    run(s, LEGACY)
    assert s.pushed == [("t sree manzil pe cement matlab", True)]


def test_partials_protocol_interim_then_final():
    s = Rec()
    run(s, PARTIALS)
    assert s.pushed == [("t three manzil p", False), ("t sree manzil pe cement matlab", True)]


def test_open_utterance_tracking_for_wait_idle():
    s = Rec()
    run(s, PARTIALS[:2])
    assert s.pending_segments == 1 and not s._idle.is_set()  # speech_start seen, final not yet
    run(s, PARTIALS[2:])
    assert s.pending_segments == 0 and s._idle.is_set()


def test_unknown_types_are_logged_not_dropped_silently():
    s = Rec()
    run(s, [{"type": "brand_new_thing", "text": "hello"}])
    assert s.pushed == []
    assert ("stt_unknown_type", {"type": "brand_new_thing", "data": '{"type": "brand_new_thing", "text": "hello"}'}) \
        in s.events


def test_errors_are_logged_in_full():
    s = Rec()
    run(s, [{"type": "error", "message": "idle timeout", "timeout_reason": "no_audio"}])
    assert s.errors == ["Gnani STT: idle timeout"]
    assert ("stt_error", {"message": "idle timeout", "timeout_reason": "no_audio"}) in s.events


@pytest.mark.parametrize("flag, header", [
    ("send_min_silence", "x-min-silence-ms"), ("send_min_speech", "x-min-speech-ms"),
    ("send_vad_threshold", "x-vad-threshold"), ("send_partials", "x-partials"), ("send_bias", "x-bias-list"),
])
def test_every_header_is_switchable(flag, header):
    from dataclasses import replace

    assert header in prisma_headers(config.PRISMA, bias=["cement"])
    assert header not in prisma_headers(replace(config.PRISMA, **{flag: False}), bias=["cement"])
