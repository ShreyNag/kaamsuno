"""Live report panel: GET /report/{session_id} (page) and /report/{session_id}/events (SSE).

`/report/latest` redirects to the newest session. State lives in-process in `HUB`;
the bot publishes a full snapshot on every change.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

PAGE = Path(__file__).parent / "ui" / "report_page.html"
KEEPALIVE_S = 15.0


class ReportHub:
    def __init__(self):
        self.snapshots: dict[str, dict[str, Any]] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self.latest: str | None = None

    def publish(self, session_id: str, snapshot: dict[str, Any]) -> None:
        self.snapshots[session_id] = snapshot
        self.latest = session_id
        for q in self._subscribers.get(session_id, ()):
            q.put_nowait(snapshot)

    async def stream(self, session_id: str):
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(session_id, set()).add(q)
        try:
            if session_id in self.snapshots:
                yield _sse(self.snapshots[session_id])
            while True:
                try:
                    yield _sse(await asyncio.wait_for(q.get(), KEEPALIVE_S))
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            self._subscribers[session_id].discard(q)


def _sse(snapshot: dict[str, Any]) -> str:
    return f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"


HUB = ReportHub()


def register_routes(app: FastAPI, hub: ReportHub = HUB) -> None:
    @app.get("/report/latest", include_in_schema=False)
    async def report_latest():
        if hub.latest is None:
            raise HTTPException(404, "no session yet; connect to /client first")
        return RedirectResponse(f"/report/{hub.latest}")

    @app.get("/report/{session_id}", response_class=HTMLResponse)
    async def report_page(session_id: str):
        return PAGE.read_text(encoding="utf-8")

    @app.get("/report/{session_id}/events")
    async def report_events(session_id: str):
        return StreamingResponse(
            hub.stream(session_id), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
