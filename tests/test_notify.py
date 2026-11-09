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
    for line in ("Incident report", "Urgency: 🔴 HIGH", "Type: material shortage", "Location: 3rd floor",
                 "Item/asset: cement", "Quantity: 150 bags", "Description: Cement short on the 3rd floor",
                 "Report ID: 20261110-100000-abc", "Read-back: corrected, then confirmed"):
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
