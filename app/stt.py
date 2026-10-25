"""Prisma STT: pipecat-gnani's WebSocket service, plus the headers and message types it doesn't handle.

pipecat-gnani 0.5.12 builds its connection headers inline in `_connect` and only
sends language / format / ITN. `PrismaSTTService._connect` is a copy of the
upstream method with our Prisma settings added (see notes/decisions.md).

Message types (found with eval/prisma_probe.py, 2026-10-25):
    without x-partials:  speech_start, transcript
    with x-partials:     speech_start, partial_transcript (~every 450 ms), final_transcript
The plugin only reads `transcript`, so with partials on it dropped every word. Our
`_receive_messages` reads both protocols: transcript / final_transcript ->
TranscriptionFrame, partial_transcript -> InterimTranscriptionFrame. Any other type
is logged (`stt_unknown_type`), never silently dropped; errors are logged in full.

Segment lifecycle: `speech_start` opens an utterance and its final closes it. The
dialogue layer waits on `wait_idle` so it never asks a question while Prisma still
owes the turn's last words.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from pipecat.services.settings import NOT_GIVEN
from pipecat_gnani import GnaniSTTService
from pipecat_gnani._common import GNANI_STT_WS_URL, get_language_string, settings_language, stt_language_to_gnani
from pipecat_gnani._sdk import _generate_request_id, sdk_headers, ws_header_kwargs
from websockets.asyncio.client import connect as websocket_connect

from app import config

BIAS_LIST_PATH = config.RESEARCH_DIR / "bias_list.txt"
FINAL_TYPES = ("transcript", "final_transcript")
PARTIAL_TYPES = ("partial_transcript",)
KNOWN_TYPES = (*FINAL_TYPES, *PARTIAL_TYPES, "speech_start", "processing", "speech_end", "vad_start", "vad_end",
               "error", "connected")


def load_bias_list(path: Path = BIAS_LIST_PATH) -> list[str]:
    """research/bias_list.txt (comma or newline separated), else the placeholder."""
    if not path.exists():
        logger.warning(f"{path} missing; using the placeholder bias list")
        return list(config.BIAS_LIST_PLACEHOLDER)
    words = [w.strip() for w in path.read_text(encoding="utf-8").replace("\n", ",").split(",")]
    return [w for w in words if w]


def prisma_headers(s: config.PrismaSettings = config.PRISMA, bias: list[str] | None = None) -> dict[str, str]:
    """The extra Prisma headers, from config; each switchable. Values per docs.gnani.ai/api/STT/stt-websocket."""
    h: dict[str, str] = {}
    if s.send_min_silence:
        h["x-min-silence-ms"] = str(s.min_silence_ms)
    if s.send_min_speech:
        h["x-min-speech-ms"] = str(s.min_speech_ms)
    if s.send_vad_threshold:
        h["x-vad-threshold"] = str(s.vad_threshold)
    if s.send_partials:
        h["x-partials"] = str(s.partials).lower()
    if s.send_bias and s.bias_score > 0:
        bias = load_bias_list() if bias is None else bias
        # Prisma takes single letter-only words; drops multi-word entries silently, so we do it loudly
        single = [w for w in bias if w.isalpha()]
        if len(single) != len(bias):
            logger.warning(f"bias list: dropped non single-word entries {sorted(set(bias) - set(single))}")
        if single:
            h["x-bias-list"] = json.dumps(single[:100], ensure_ascii=False)
            h["x-bias-score"] = str(s.bias_score)
    return h


SELF_TEST_WAV = config.ROOT / "tests" / "fixtures" / "matlab_pause.wav"  # first phrase: 0.6-2.9 s


async def self_test(extra_headers: dict[str, str], timeout_s: float = config.PRISMA.self_test_timeout_s,
                    wav: Path = SELF_TEST_WAV) -> str:
    """Stream ~2.5 s of a fixture to Prisma with the bot's exact headers, in real time, and
    return the first final transcript. Raises RuntimeError if none arrives within `timeout_s`
    after the audio ends; this is the check that would have caught the x-partials breakage."""
    import numpy as np

    from app.audio_io import load_wav

    x = load_wav(wav)[int(0.4 * config.SAMPLE_RATE) : int(2.9 * config.SAMPLE_RATE)]
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    # then silence for the whole wait, like a live mic: Prisma only closes a segment on silence it receives
    pcm += bytes(int(config.SAMPLE_RATE * 2 * timeout_s))
    headers = {
        "x-api-key-id": config.GNANI_API_KEY, "lang_code": config.LANGUAGE_CODE,
        "x-sample-rate": str(config.SAMPLE_RATE), "x-api-request-id": _generate_request_id(),
        "x-format": config.PRISMA.output_format, **sdk_headers(), **extra_headers,
    }
    seen: Counter[str] = Counter()
    chunk = 640  # 20 ms
    async with websocket_connect(GNANI_STT_WS_URL, **ws_header_kwargs(headers), close_timeout=5) as ws:
        await asyncio.wait_for(ws.recv(), 10)  # "connected"

        async def send():
            t0 = time.monotonic()
            for i, off in enumerate(range(0, len(pcm), chunk)):
                await asyncio.sleep(max(0.0, t0 + i * 0.02 - time.monotonic()))
                await ws.send(pcm[off : off + chunk])

        async def first_final() -> str:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                d = json.loads(msg)
                seen[d.get("type", "")] += 1
                if d.get("type") in FINAL_TYPES and (d.get("text") or "").strip():
                    return d["text"].strip()
                if d.get("type") == "error":
                    raise RuntimeError(f"Prisma self-test: error message {d}")
            raise RuntimeError("Prisma self-test: socket closed before a transcript")

        waiter = asyncio.create_task(first_final())
        sender = asyncio.create_task(send())
        try:
            # 2.5 s of speech, then up to timeout_s for the transcript
            return await asyncio.wait_for(waiter, 2.5 + timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Prisma self-test: no final transcript {timeout_s:.0f} s after 2.5 s of speech. Message types "
                f"received: {dict(seen) or 'none'}. Headers sent: {sorted(extra_headers)}. If only partial/unknown "
                "types arrived, app/stt.py doesn't understand Prisma's protocol; bisect with "
                "`python -m eval.prisma_probe` or PRISMA_SEND_<HEADER>=0."
            ) from None
        finally:
            sender.cancel()


class PrismaSTTService(GnaniSTTService):
    def __init__(self, *, extra_headers: dict[str, str], log: Callable[..., None] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.extra_headers = extra_headers
        self._log = log or (lambda event, **kw: None)
        self.msg_types: Counter[str] = Counter()
        self._open: set[str] = set()  # utterance_ids with a speech_start but no final yet
        self.last_message_ts = 0.0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def supports_ttfs(self) -> bool:
        # Pipecat's turn strategies would otherwise add a 1.0 s "STT p99" wait after
        # every VAD stop. Our dialogue layer waits for Prisma explicitly instead.
        return False

    @property
    def pending_segments(self) -> int:
        return len(self._open)

    def _track(self, kind: str, data: dict) -> None:
        """Message-type census and the open-utterance set behind wait_idle."""
        self.last_message_ts = time.time()
        if kind not in self.msg_types:
            logger.info(f"[STT WS] first '{kind}' message: keys={sorted(data)}")
            self._log("stt_msg_type", type=kind, keys=sorted(data), is_final=data.get("is_final"))
        self.msg_types[kind] += 1
        uid = data.get("utterance_id")
        if kind == "speech_start" and uid:
            self._open.add(uid)
        elif kind in FINAL_TYPES or kind == "error":
            if uid:
                self._open.discard(uid)
            else:
                self._open.clear()
        if self._open:
            self._idle.clear()
        else:
            self._idle.set()

    async def wait_idle(self, not_before: float, timeout: float) -> bool:
        """Wait until wall-clock `not_before` (so Prisma has had time to close the
        segment), then until no utterance is open. False if `timeout` ran out."""
        deadline = time.time() + timeout
        await asyncio.sleep(max(0.0, not_before - time.time()))
        try:
            await asyncio.wait_for(self._idle.wait(), max(0.0, deadline - time.time()))
            return True
        except asyncio.TimeoutError:
            return False

    async def handle_message(self, data: dict) -> None:
        """One parsed Prisma message. Separate from the socket loop so tests can drive it."""
        kind = data.get("type", "")
        self._track(kind, data)
        if kind in FINAL_TYPES or kind in PARTIAL_TYPES:
            text = (data.get("text") or "").strip()
            if text:
                await self._push_transcript(
                    text, is_final=kind in FINAL_TYPES, language=settings_language(self._settings), result=data
                )
        elif kind == "error":
            logger.error(f"[STT WS] stream error | request_id={self._request_id} | {data}")
            self._log("stt_error", **{k: v for k, v in data.items() if k != "type"})
            self._ws = None
            await self.push_error(error_msg=f"Gnani STT: {data.get('message', 'Unknown error')}")
        elif kind not in KNOWN_TYPES:
            raw = json.dumps(data, ensure_ascii=False)
            logger.warning(f"[STT WS] unknown message type {kind!r}: {raw[:300]}")
            self._log("stt_unknown_type", type=kind, data=raw[:500])

    async def _receive_messages(self):
        # Replaces GnaniSTTService._receive_messages (pipecat-gnani 0.5.12), which only knows `transcript`.
        ws = self._ws
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                try:
                    data = json.loads(msg)
                except ValueError:
                    logger.warning(f"[STT WS] non-JSON message: {msg[:200]!r}")
                    continue
                await self.handle_message(data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            code = getattr(ws, "close_code", None)
            logger.error(f"[STT WS] receive error | request_id={self._request_id} | close_code={code} | error={e}")
            self._log("stt_error", message=str(e), close_code=code)
            self._ws = None
            await self.push_error(error_msg=f"Gnani STT receive error: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _connect(self):
        # Copy of GnaniSTTService._connect (pipecat-gnani 0.5.12) + extra_headers.
        request_id = _generate_request_id()
        self._request_id = request_id
        self._open.clear()
        self._idle.set()
        try:
            lang = get_language_string(self._settings, stt_language_to_gnani)
            headers = {
                "x-api-key-id": self._api_key,
                "lang_code": lang or "en-IN",
                "x-sample-rate": str(self._resolve_sample_rate()),
                "x-api-request-id": request_id,
                **sdk_headers(),
                **self.extra_headers,
            }
            fmt = getattr(self._settings, "format", None)
            if fmt and fmt not in (NOT_GIVEN, None):
                headers["x-format"] = fmt
            itn = getattr(self._settings, "itn_native_numerals", None)
            if itn and itn not in (NOT_GIVEN, None):
                headers["itn_native_numerals"] = str(itn).lower()

            ws = await websocket_connect(
                GNANI_STT_WS_URL, **ws_header_kwargs(headers), ping_interval=20, ping_timeout=20, close_timeout=10
            )
            connected = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if connected.get("type") == "connected":
                logger.info(f"[STT WS] connected | request_id={request_id} | config={connected.get('config')}")
            else:
                logger.warning(f"[STT WS] unexpected first message | request_id={request_id} | data={connected}")
            self._ws = ws
            self._receive_task = asyncio.create_task(self._receive_messages(), name="gnani-stt-recv")
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"[STT WS] connect failed | request_id={request_id} | error={e}")
            self._ws = None
            await self.push_error(error_msg=f"Failed to connect to Gnani STT: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))
