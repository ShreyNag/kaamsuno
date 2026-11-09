"""Sending a report to the supervisor: a readable Telegram message.

    format_message(values, kind)   report values -> plain text (no markup, so no escaping problems)
    TelegramSender(token, chat_id) async callable (values, kind) -> bool, for ReportProcessor(send_report=...)
    build_sender()                 from .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID); None if not set

Check the setup without running the bot:

    .venv/Scripts/python -m app.notify          sends a sample report to TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

from app import config

KIND_HEADER = {
    "emergency": "🚨 EMERGENCY ALERT (KaamSuno)",
    "report": "📋 Incident report (KaamSuno)",
}
URGENCY_MARK = {"emergency": "🚨", "high": "🔴", "normal": "🟡", "low": "🟢"}
CONFIRMATION_TEXT = {"confirmed": "confirmed by the worker", "corrected_then_confirmed": "corrected, then confirmed",
                     "rejected": "rejected by the worker", "not_obtained": "not yet confirmed"}


def _label(v: Any) -> str:
    return str(v).replace("_", " ")


def format_message(values: dict[str, Any], kind: str = "report") -> str:
    """Urgency, type, location, item/asset, quantity + unit, description, report ID; then the rest."""
    v = values.get
    lines = [KIND_HEADER.get(kind, KIND_HEADER["report"]), ""]
    urgency = v("urgency")
    lines.append(f"Urgency: {URGENCY_MARK.get(urgency, '')} {_label(urgency).upper()}".replace(":  ", ": ")
                 if urgency else "Urgency: not given yet")
    lines.append(f"Type: {_label(v('incident_type')) if v('incident_type') else 'not given yet'}")
    lines.append(f"Location: {v('location') or 'not given yet'}")
    item = v("asset_or_equipment") or v("material")
    if item:
        lines.append(f"Item/asset: {item}")
    if v("quantity") is not None:
        lines.append(f"Quantity: {v('quantity')} {v('unit') or ''}".rstrip())
    if v("people_at_risk"):
        lines.append(f"People at risk: {v('people_at_risk')}")
    if v("description"):
        lines.append(f"Description: {v('description')}")
    lines += ["", f"Report ID: {v('incident_id', '?')}"]
    meta = [str(x) for x in (v("site_name"), v("reported_at")) if x]
    if meta:
        lines.append("Site: " + " · ".join(meta))
    if kind == "report":
        lines.append(f"Read-back: {CONFIRMATION_TEXT.get(v('reporter_confirmation'), 'not yet confirmed')}")
    if v("uncertainties"):
        lines.append("Unclear: " + ", ".join(_label(u.get("field", u)) if isinstance(u, dict) else _label(u)
                                             for u in v("uncertainties")))
    if kind == "emergency":
        lines += ["", "Details are still being collected; the full report follows."]
    if v("transcript"):
        lines += ["", f"Worker said: “{str(v('transcript'))[:600]}”"]
    return "\n".join(lines)


class TelegramSender:
    """POST sendMessage. True only when Telegram says ok; errors are logged, never raised."""

    def __init__(self, token: str, chat_id: str, timeout_s: float = 8.0):
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id, self._timeout = chat_id, timeout_s

    async def __call__(self, values: dict[str, Any], kind: str = "report") -> bool:
        payload = {"chat_id": self.chat_id, "text": format_message(values, kind)[:4000],
                   "disable_web_page_preview": True}
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, json=payload)
            body = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.error(f"telegram: send failed: {type(e).__name__}: {e}")  # the URL (with the token) isn't logged
            return False
        if not body.get("ok"):
            logger.error(f"telegram: {resp.status_code} {body.get('description')}")
            return False
        logger.info(f"telegram: sent {kind} {values.get('incident_id')} in {round((time.time() - t0) * 1000)} ms")
        return True


def build_sender() -> TelegramSender | None:
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set: reports are saved but not sent")
        return None
    return TelegramSender(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)


SAMPLE = {
    "incident_id": "test-" + time.strftime("%H%M%S"), "site_name": config.SITE_NAME,
    "reported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "urgency": "high", "incident_type": "material_shortage",
    "location": "3rd floor", "material": "cement", "quantity": 150, "unit": "bags",
    "description": "Cement ran short on the 3rd floor (test message)", "reporter_confirmation": "confirmed",
}

if __name__ == "__main__":
    import asyncio
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(format_message(SAMPLE), "\n")
    sender = build_sender()
    if sender is None:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first.")
    raise SystemExit(0 if asyncio.run(sender(SAMPLE, "report")) else "Telegram did not accept the message (see log).")
