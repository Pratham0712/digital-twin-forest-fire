"""
providers.py - email / SMS providers for the emergency-alert workflow.

Configuration (environment variables only - never hard-coded, never logged):
  ALERT_PROVIDER_MODE   "live" (default) or "mock". mock: nothing leaves the
                        machine; every attempt is recorded with status "mock".
  Email (SMTP, reuses the project's existing SMTP settings):
    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM (optional)
  SMS (Twilio Programmable Messaging REST API):
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER (E.164)

Honest statuses: SMTP acceptance = "accepted" (the server took the message;
that is not proof of delivery to the inbox). Twilio returns "queued" /
"accepted" / "sent"; real delivery is only known from a later status callback,
so nothing here ever reports "delivered". Unconfigured providers return
"not_configured"; errors return "failed" with a short reason (no secrets).
"""
from __future__ import annotations

import os
import smtplib
import ssl
import uuid
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Optional

import requests

TIMEOUT_S = 15


@dataclass
class SendResult:
    status: str                       # accepted | queued | sent | failed | mock | not_configured
    provider: str
    message_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in ("accepted", "queued", "sent")


def mock_mode() -> bool:
    return os.getenv("ALERT_PROVIDER_MODE", "live").strip().lower() == "mock"


class EmailProvider:
    name = "smtp"

    def __init__(self):
        self.host = os.getenv("SMTP_HOST", "")
        self.port = int(os.getenv("SMTP_PORT", "587") or 587)
        self.user = os.getenv("SMTP_USER", "")
        self.password = os.getenv("SMTP_PASSWORD", "")
        self.sender = os.getenv("SMTP_FROM", "") or self.user

    @property
    def configured(self) -> bool:
        placeholder = self.user.startswith("your_") or self.password.startswith("your_")
        return bool(self.host and self.user and self.password and not placeholder)

    def missing(self) -> str:
        return "SMTP_HOST, SMTP_USER, SMTP_PASSWORD (and optionally SMTP_PORT, SMTP_FROM)"

    def send(self, to: str, subject: str, body: str) -> SendResult:
        if mock_mode():
            return SendResult("mock", "mock-email", f"mock-{uuid.uuid4().hex[:10]}")
        if not self.configured:
            return SendResult("not_configured", self.name, error=f"email provider not configured ({self.missing()})")
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
            mid = f"<{uuid.uuid4().hex}@forest-fire-digital-twin>"
            msg["Message-ID"] = mid
            with smtplib.SMTP(self.host, self.port, timeout=TIMEOUT_S) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(self.user, self.password)
                refused = s.sendmail(self.sender, [to], msg.as_string())
            if refused:
                return SendResult("failed", self.name, mid, "recipient refused by the SMTP server")
            return SendResult("accepted", self.name, mid)
        except (smtplib.SMTPException, OSError) as exc:
            return SendResult("failed", self.name, error=type(exc).__name__ + ": " + str(exc)[:160])


class SmsProvider:
    name = "twilio"

    def __init__(self):
        self.sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "")

    @property
    def configured(self) -> bool:
        return bool(self.sid and self.token and self.from_number)

    def missing(self) -> str:
        return "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER"

    def send(self, to: str, body: str) -> SendResult:
        if mock_mode():
            return SendResult("mock", "mock-sms", f"mock-{uuid.uuid4().hex[:10]}")
        if not self.configured:
            return SendResult("not_configured", self.name, error=f"SMS provider not configured ({self.missing()})")
        try:
            r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json",
                              data={"To": to, "From": self.from_number, "Body": body[:1500]},
                              auth=(self.sid, self.token), timeout=TIMEOUT_S)
            if r.status_code >= 400:
                try:
                    detail = r.json().get("message", "")
                except ValueError:
                    detail = ""
                return SendResult("failed", self.name, error=f"HTTP {r.status_code} {detail[:160]}")
            j = r.json()
            status = j.get("status", "queued")
            return SendResult(status if status in ("queued", "accepted", "sent") else "unknown", self.name, j.get("sid"))
        except requests.RequestException as exc:
            return SendResult("failed", self.name, error=type(exc).__name__)


def provider_status() -> dict:
    """What is configured (no values, only presence) - shown in the UI."""
    e, s = EmailProvider(), SmsProvider()
    return {"mode": "mock" if mock_mode() else "live",
            "email": "configured" if e.configured else f"not configured - set {e.missing()}",
            "sms": "configured" if s.configured else f"not configured - set {s.missing()}"}
