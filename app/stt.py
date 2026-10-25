"""Prisma STT: pipecat-gnani's WebSocket service plus the headers it doesn't expose.

pipecat-gnani 0.5.12 builds its connection headers inline in `_connect` and only
sends language / format / ITN. `PrismaSTTService._connect` is a copy of the
upstream method with our Prisma settings added (see notes/decisions.md).

Every message type seen on the wire is logged once (`stt_msg_type`), with its
keys: the plugin silently drops types it doesn't know, so this is how we find
out whether Prisma sends partials at all (see notes/decisions.md).

It also counts Prisma's segment lifecycle: the server sends `processing` when
its VAD closes a segment and `transcript` when the text is ready. The dialogue
layer waits on `wait_idle` so it never asks a question before the last
segment of the turn has been interpreted.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from pipecat_gnani import GnaniSTTService
from pipecat_gnani._common import GNANI_STT_WS_URL, get_language_string, stt_language_to_gnani
from pipecat_gnani._sdk import _generate_request_id, sdk_headers, ws_header_kwargs
from pipecat.services.settings import NOT_GIVEN
from websockets.asyncio.client import connect as websocket_connect

from app import config

BIAS_LIST_PATH = config.RESEARCH_DIR / "bias_list.txt"


def load_bias_list(path: Path = BIAS_LIST_PATH) -> list[str]:
    """research/bias_list.txt (comma or newline separated), else the placeholder."""
    if not path.exists():
        logger.warning(f"{path} missing; using the placeholder bias list")
        return list(config.BIAS_LIST_PLACEHOLDER)
    words = [w.strip() for w in path.read_text(encoding="utf-8").replace("\n", ",").split(",")]
    return [w for w in words if w]


def prisma_headers(s: config.PrismaSettings = config.PRISMA, bias: list[str] | None = None) -> dict[str, str]:
    """The extra Prisma headers, from config. Values per docs.gnani.ai/api/STT/stt-websocket."""
    bias = load_bias_list() if bias is None else bias
    h = {
        "x-min-silence-ms": str(s.min_silence_ms),
        "x-min-speech-ms": str(s.min_speech_ms),
        "x-vad-threshold": str(s.vad_threshold),
        "x-partials": str(s.partials).lower(),
    }
    if bias and s.bias_score > 0:
        # Prisma takes single letter-only words; drops multi-word entries silently, so we do it loudly
        single = [w for w in bias if w.isalpha()]
        if len(single) != len(bias):
            logger.warning(f"bias list: dropped non single-word entries {sorted(set(bias) - set(single))}")
        h["x-bias-list"] = json.dumps(single[:100], ensure_ascii=False)
        h["x-bias-score"] = str(s.bias_score)
    return h


class _SegmentCountingSocket:
    """Wraps the WebSocket so every server message passes through `on_message` first."""

    def __init__(self, ws, on_message):
        self._ws, self._on_message = ws, on_message

    def __getattr__(self, name):
        return getattr(self._ws, name)

    async def __aiter__(self):
        async for msg in self._ws:
            if isinstance(msg, str):
                self._on_message(msg)
            yield msg


class PrismaSTTService(GnaniSTTService):
    def __init__(self, *, extra_headers: dict[str, str], log: Callable[..., None] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.extra_headers = extra_headers
        self._log = log or (lambda event, **kw: None)
        self.msg_types: Counter[str] = Counter()
        self.pending_segments = 0  # `processing` seen, `transcript` not yet
        self.last_message_ts = 0.0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def supports_ttfs(self) -> bool:
        # Pipecat's turn strategies would otherwise add a 1.0 s "STT p99" wait after
        # every VAD stop. Our dialogue layer waits for Prisma explicitly instead.
        return False

    def _on_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
            kind = data.get("type", "")
        except (ValueError, AttributeError):
            return
        self.last_message_ts = time.time()
        if kind not in self.msg_types:
            logger.info(f"[STT WS] first '{kind}' message: keys={sorted(data)}")
            self._log("stt_msg_type", type=kind, keys=sorted(data), is_final=data.get("is_final"))
        self.msg_types[kind] += 1
        if kind == "transcript" and data.get("is_final") is False:
            self._log("stt_msg_type", type="transcript_not_final", text=data.get("text", "")[:80])
        if kind == "processing":
            self.pending_segments += 1
            self._idle.clear()
        elif kind in ("transcript", "error"):
            self.pending_segments = max(0, self.pending_segments - 1)
            if self.pending_segments == 0:
                self._idle.set()

    async def wait_idle(self, not_before: float, timeout: float) -> bool:
        """Wait until wall-clock `not_before` (so Prisma has had time to close the
        segment), then until no segment is in flight. False if `timeout` ran out."""
        deadline = time.time() + timeout
        await asyncio.sleep(max(0.0, not_before - time.time()))
        try:
            await asyncio.wait_for(self._idle.wait(), max(0.0, deadline - time.time()))
            return True
        except asyncio.TimeoutError:
            return False

    async def _connect(self):
        # Copy of GnaniSTTService._connect (pipecat-gnani 0.5.12) + extra_headers + socket wrapper.
        request_id = _generate_request_id()
        self._request_id = request_id
        self.pending_segments = 0
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
            self._ws = _SegmentCountingSocket(ws, self._on_message)
            self._receive_task = asyncio.create_task(self._receive_messages(), name="gnani-stt-recv")
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"[STT WS] connect failed | request_id={request_id} | error={e}")
            self._ws = None
            await self.push_error(error_msg=f"Failed to connect to Gnani STT: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))
