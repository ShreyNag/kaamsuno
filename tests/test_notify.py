"""The supervisor's Telegram message: readable, and a failed send never raises."""

from __future__ import annotations

import asyncio
import json

import httpx

from app.notify import TelegramSender, format_message

FULL = {"incident_id": "20261110-100000-abc", "site_name": "Demo Site A", "reported_at": "2026-11-10T10:00:00",
        "urgency": "high", "incident_type": "material_shortage", "location": "3rd floor", "material": "cement",
        "quantity": 150, "unit": "bags", "description": "Cement short on the 3rd floor",
        "reporter_confirmation": "corrected_then_confirmed", "transcript": "teesri manzil pe cement kam hai"}


def test_report_message_has_every_field_the_supervisor_needs():
    m = format_message(FULL, "report")
    for line in ("Incident report", "Urgency: 🔴 Urgent", "Type: Material shortage", "Location: 3rd floor",
                 "Item/asset: cement", "Quantity: 150 bags", "Description: Cement short on the 3rd floor",
                 "Report ID: 20261110-100000-abc", "Read-back: Corrected, then confirmed"):
        assert line in m, (line, m)
    assert m.index("Urgency") < m.index("Type") < m.index("Location") < m.index("Item/asset") < m.index("Quantity")


def test_emergency_alert_with_few_details_says_what_is_missing():
    m = format_message({"incident_id": "x", "urgency": "emergency", "incident_type": "injury_or_near_miss"},
                       "emergency")
    assert m.startswith("🚨 EMERGENCY ALERT") and "Location: not given yet" in m
    assert "full report follows" in m and "Read-back" not in m
    assert "Item/asset" not in m and "Quantity" not in m  # empty optional fields are left out


def test_sender_returns_false_instead_of_raising(monkeypatch):
    class Boom(httpx.AsyncClient):
        async def post(self, *a, **k):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr("app.notify.httpx.AsyncClient", Boom)
    assert asyncio.run(TelegramSender("t", "1")(FULL)) is False


def test_sender_true_only_when_telegram_says_ok(monkeypatch):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"ok": request.url.path.endswith("/sendMessage")})

    real = httpx.AsyncClient
    monkeypatch.setattr("app.notify.httpx.AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(TelegramSender("TOKEN", "-100")(FULL, "report")) is True
    body = json.loads(sent[0].read())
    assert body["chat_id"] == "-100" and "Quantity: 150 bags" in body["text"]


def test_other_and_unknown_never_reach_the_supervisor():
    m = format_message({**FULL, "incident_type": "other", "urgency": "unknown",
                        "uncertainties": [{"field": "asset_or_equipment", "reason": "not_answered"}]})
    assert "Type: see description" in m and "Urgency: not given yet" in m
    assert "Unclear: machine or equipment" in m
    assert "other" not in m.lower().replace("another", "") and "_" not in m.replace(FULL["incident_id"], "")


# --- WhatsApp (Twilio) ------------------------------------------------------------------------------------
from urllib.parse import parse_qs  # noqa: E402

from app import config  # noqa: E402
from app.notify import TelegramSender as _TG, WhatsAppSender, build_sender  # noqa: E402


def mock_twilio(monkeypatch, status, body):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(status, json=body)

    real = httpx.AsyncClient
    monkeypatch.setattr("app.notify.httpx.AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return sent


def test_whatsapp_sends_the_same_text_with_basic_auth(monkeypatch):
    sent = mock_twilio(monkeypatch, 201, {"sid": "SM123", "status": "queued"})
    wa = WhatsAppSender("AC1", "secret", "+14155238886", "whatsapp:+919999999999")
    assert asyncio.run(wa(FULL, "report")) is True
    req = sent[0]
    assert req.url.path == "/2010-04-01/Accounts/AC1/Messages.json" and req.headers["authorization"].startswith("Basic ")
    form = parse_qs(req.read().decode())
    assert form["From"] == ["whatsapp:+14155238886"] and form["To"] == ["whatsapp:+919999999999"]
    assert form["Body"] == [format_message(FULL, "report")]


def test_whatsapp_rejected_is_false(monkeypatch):
    mock_twilio(monkeypatch, 400, {"code": 63007, "message": "not joined the sandbox"})
    assert asyncio.run(WhatsAppSender("AC1", "s", "+1", "+91")(FULL)) is False


def test_whatsapp_offline_is_false_not_an_exception(monkeypatch):
    class Boom(httpx.AsyncClient):
        async def post(self, *a, **k):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr("app.notify.httpx.AsyncClient", Boom)
    assert asyncio.run(WhatsAppSender("AC1", "s", "+1", "+91")(FULL)) is False


def test_whatsapp_long_message_is_cut_to_twilio_limit(monkeypatch):
    sent = mock_twilio(monkeypatch, 201, {"sid": "SM1"})
    asyncio.run(WhatsAppSender("AC1", "s", "+1", "+91")({**FULL, "transcript": "x " * 3000}))
    assert len(parse_qs(sent[0].read().decode())["Body"][0]) <= WhatsAppSender.MAX_CHARS


def test_channel_switch(monkeypatch):
    for k, v in dict(TELEGRAM_BOT_TOKEN="t", TELEGRAM_CHAT_ID="1", TWILIO_ACCOUNT_SID="AC", TWILIO_AUTH_TOKEN="s",
                     TWILIO_WHATSAPP_FROM="+1", SUPERVISOR_WHATSAPP_TO="+91").items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.setattr(config, "NOTIFY_CHANNEL", "telegram")
    assert isinstance(build_sender(), _TG)  # the default
    assert isinstance(build_sender("whatsapp"), WhatsAppSender)
    monkeypatch.setattr(config, "NOTIFY_CHANNEL", "whatsapp")
    assert isinstance(build_sender(), WhatsAppSender)
    monkeypatch.setattr(config, "SUPERVISOR_WHATSAPP_TO", "")
    assert build_sender() is None  # a missing key: saved, not sent
    assert build_sender("sms") is None
