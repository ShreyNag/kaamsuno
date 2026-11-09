"""Sending a report to the supervisor: a readable Telegram (or WhatsApp) message.

    format_message(values, kind)   report values -> plain text (no markup, so no escaping problems)
    TelegramSender(token, chat_id) async callable (values, kind) -> bool, for ReportProcessor(send_report=...)
    WhatsAppSender(sid, token, from, to)   the same over Twilio's WhatsApp API (sandbox)
    build_sender()                 NOTIFY_CHANNEL=telegram|whatsapp (default telegram) and its keys from .env;
                                   None if not set

Check the setup without running the bot:

    .venv/Scripts/python -m app.notify                      sends a sample report (NOTIFY_CHANNEL)
    .venv/Scripts/python -m app.notify --channel whatsapp   ... over WhatsApp
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

from app import config
from app.plain import field_word, say_value

KIND_HEADER = {
    "emergency": "🚨 EMERGENCY ALERT (KaamSuno)",
    "report": "📋 Incident report (KaamSuno)",
}
URGENCY_MARK = {"emergency": "🚨", "high": "🔴", "normal": "🟡", "low": "🟢"}


def _en(name: str, value: Any) -> str | None:
    """Plain English for the supervisor; None for values that say nothing ("other", "unknown")."""
    return say_value(name, value, "en")


def format_message(values: dict[str, Any], kind: str = "report") -> str:
    """Urgency, type, location, item/asset, quantity + unit, description, report ID; then the rest."""
    v = values.get
    lines = [KIND_HEADER.get(kind, KIND_HEADER["report"]), ""]
    urgency, itype = v("urgency"), v("incident_type")
    urgency_text = _en("urgency", urgency)
    lines.append(f"Urgency: {URGENCY_MARK.get(urgency, '')} {urgency_text}".replace(":  ", ": ")
                 if urgency_text else "Urgency: not given yet")
    lines.append(f"Type: {_en('incident_type', itype) or ('see description' if itype else 'not given yet')}")
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
        lines.append(f"Read-back: {_en('reporter_confirmation', v('reporter_confirmation') or 'not_obtained')}")
    if v("uncertainties"):
        names = [u.get("field") if isinstance(u, dict) else u for u in v("uncertainties")]
        lines.append("Unclear: " + ", ".join(field_word(str(n), "en") for n in dict.fromkeys(names)))
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


class WhatsAppSender:
    """Twilio's WhatsApp API (the sandbox works): POST Messages.json, same text as Telegram.
    True only when Twilio accepts the message (2xx with a message sid); errors are logged, never raised.
    Sandbox: the supervisor's phone must have sent "join <code>" to the sandbox number first."""

    MAX_CHARS = 1600  # Twilio's WhatsApp body limit

    def __init__(self, account_sid: str, auth_token: str, from_: str, to: str, timeout_s: float = 8.0):
        self._url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        self._auth = (account_sid, auth_token)
        self.from_, self.to, self._timeout = _wa(from_), _wa(to), timeout_s

    async def __call__(self, values: dict[str, Any], kind: str = "report") -> bool:
        text = format_message(values, kind)
        if len(text) > self.MAX_CHARS:
            text = text[: self.MAX_CHARS - 1] + "…"
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, auth=self._auth, data={"From": self.from_, "To": self.to, "Body": text})
            body = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.error(f"whatsapp: send failed: {type(e).__name__}: {e}")
            return False
        if not (200 <= resp.status_code < 300 and body.get("sid")):
            logger.error(f"whatsapp: {resp.status_code} twilio code {body.get('code')}: {body.get('message')}")
            return False
        logger.info(f"whatsapp: sent {kind} {values.get('incident_id')} in {round((time.time() - t0) * 1000)} ms "
                    f"(status {body.get('status')})")
        return True


def _wa(number: str) -> str:
    number = number.strip()
    return number if number.startswith("whatsapp:") else f"whatsapp:{number}"


def build_sender(channel: str | None = None) -> TelegramSender | WhatsAppSender | None:
    """NOTIFY_CHANNEL (telegram | whatsapp, default telegram). None if that channel's keys are not set."""
    channel = (channel or config.NOTIFY_CHANNEL).lower()
    if channel == "whatsapp":
        keys = (config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN, config.TWILIO_WHATSAPP_FROM,
                config.SUPERVISOR_WHATSAPP_TO)
        if not all(keys):
            logger.warning("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_WHATSAPP_FROM / SUPERVISOR_WHATSAPP_TO "
                           "not all set: reports are saved but not sent")
            return None
        return WhatsAppSender(*keys)
    if channel != "telegram":
        logger.warning(f"NOTIFY_CHANNEL={channel!r} is not telegram or whatsapp: reports are saved but not sent")
        return None
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
    import argparse
    import asyncio
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Send a sample report to the supervisor channel.")
    ap.add_argument("--channel", choices=("telegram", "whatsapp"), default=None,
                    help="default: NOTIFY_CHANNEL from .env (telegram)")
    ch = ap.parse_args().channel or config.NOTIFY_CHANNEL
    print(format_message(SAMPLE), "\n")
    sender = build_sender(ch)
    if sender is None:
        raise SystemExit(f"The {ch} keys are not set in .env (see .env.example).")
    raise SystemExit(0 if asyncio.run(sender(SAMPLE, "report")) else f"{ch} did not accept the message (see log).")
