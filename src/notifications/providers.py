"""
providers.py - email / SMS providers for the alert workflow.

Configuration comes from src/notifications/config.py (Admin page for this
session, then environment / .env, then Streamlit secrets) - never hard-coded,
never logged:
  ALERT_PROVIDER_MODE   "live" (default) or "mock". mock: nothing leaves the
                        machine; every attempt is recorded with status "mock".
  Email (SMTP): SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM (optional),
                SMTP_SECURITY (starttls | ssl | none; default starttls, ssl on port 465), SMTP_TIMEOUT (s)
  SMS gateway, chosen by SMS_PROVIDER (textbee | twilio; see config.sms_provider_name):
    textbee.dev Android SMS gateway (the user's own Android phone + SIM sends the SMS):
                TEXTBEE_API_KEY, TEXTBEE_DEVICE_ID (optional), TEXTBEE_BASE_URL
                (default https://api.textbee.dev/api/v1), TEXTBEE_TIMEOUT (s)
    Twilio Programmable Messaging REST API:
                TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER (E.164)

Honest statuses: SMTP acceptance = "accepted" (the server took the message;
that is not proof of inbox delivery). A successful SMTP login without sending
is "connected". Twilio returns queued / accepted / sending / sent; "delivered"
or "undelivered" is only reported when Twilio's message resource says so
(`SmsProvider.status()`), never from the submission alone. textbee: a 2xx answer means the request
was accepted by the gateway ("accepted"); the phone then reports dispatched / sent / delivered / failed,
read back with `TextbeeSmsProvider.status()` - "delivered" only when textbee's message record says so. Unconfigured
providers return "not_configured"; errors return "failed" with a short,
redacted reason.
"""
from __future__ import annotations

import smtplib
import socket
import ssl
import uuid
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from typing import Optional

import requests

from src.notifications import config as cfg

TIMEOUT_S = 15
TWILIO_STATUSES = {"queued": "queued", "accepted": "accepted", "scheduled": "queued", "sending": "queued",
                   "sent": "sent", "delivered": "delivered", "undelivered": "failed", "failed": "failed",
                   "receiving": "unknown", "received": "unknown", "canceled": "failed", "read": "delivered"}


@dataclass
class SendResult:
    status: str                       # accepted | queued | sent | delivered | connected | failed | mock | not_configured | unknown
    provider: str
    message_id: Optional[str] = None
    error: Optional[str] = None
    detail: Optional[str] = None      # extra, non-error information (e.g. which phone / last heartbeat)

    @property
    def ok(self) -> bool:
        return self.status in ("accepted", "queued", "sent", "delivered", "connected")


def mock_mode() -> bool:
    return cfg.mock_mode()


class EmailProvider:
    name = "smtp"

    def __init__(self):
        s = cfg.smtp_settings()
        self.host, self.port, self.user, self.password = s["host"], s["port"], s["user"], s["password"]
        self.sender, self.security, self.timeout = s["sender"], s["security"], s["timeout"] or TIMEOUT_S
        self._s = s

    @property
    def configured(self) -> bool:
        return not cfg.smtp_problems(self._s)

    def missing(self) -> str:
        return "; ".join(cfg.smtp_problems(self._s)) or "-"

    def _connect(self):
        if self.security == "ssl":
            s = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=ssl.create_default_context())
        else:
            s = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
            if self.security == "starttls":
                s.starttls(context=ssl.create_default_context())
        s.login(self.user, self.password)
        return s

    def _fail(self, exc) -> SendResult:
        kind = ("authentication failed - check the username and app password"
                if isinstance(exc, smtplib.SMTPAuthenticationError) else
                "timed out" if isinstance(exc, (socket.timeout, TimeoutError)) else type(exc).__name__)
        return SendResult("failed", self.name, error=cfg.redact(f"{kind}: {str(exc)[:160]}"))

    def test_connection(self) -> SendResult:
        """Connects, negotiates TLS and logs in - sends nothing."""
        if mock_mode():
            return SendResult("mock", "mock-email")
        if not self.configured:
            return SendResult("not_configured", self.name, error=self.missing())
        try:
            s = self._connect()
            s.noop()
            s.quit()
            return SendResult("connected", self.name)
        except (smtplib.SMTPException, OSError) as exc:
            return self._fail(exc)

    def send(self, to: str, subject: str, body: str, html: Optional[str] = None) -> SendResult:
        if mock_mode():
            return SendResult("mock", "mock-email", f"mock-{uuid.uuid4().hex[:10]}")
        if not self.configured:
            return SendResult("not_configured", self.name, error=f"email provider not configured ({self.missing()})")
        if not cfg.valid_email(to):
            return SendResult("failed", self.name, error="invalid recipient email address")
        try:
            msg = MIMEMultipart("alternative")
            msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
            msg["Date"] = formatdate(localtime=False)
            mid = f"<{uuid.uuid4().hex}@forest-fire-digital-twin>"
            msg["Message-ID"] = mid
            msg.attach(MIMEText(body, "plain", "utf-8"))
            if html:
                msg.attach(MIMEText(html, "html", "utf-8"))
            s = self._connect()
            try:
                refused = s.sendmail(self.sender, [to], msg.as_string())
            finally:
                try:
                    s.quit()
                except Exception:
                    pass
            if refused:
                return SendResult("failed", self.name, mid, "recipient refused by the SMTP server")
            return SendResult("accepted", self.name, mid)
        except (smtplib.SMTPException, OSError) as exc:
            return self._fail(exc)


class SmsProvider:
    name = "twilio"

    def __init__(self):
        s = cfg.sms_settings()
        self.sid, self.token, self.from_number = s["sid"], s["token"], s["from_number"]
        self._s = s

    @property
    def configured(self) -> bool:
        return not cfg.sms_problems(self._s)

    def missing(self) -> str:
        return "; ".join(cfg.sms_problems(self._s)) or "-"

    def _err(self, r) -> str:
        try:
            j = r.json()
            detail = f"{j.get('code', '')} {j.get('message', '')}".strip()
        except ValueError:
            detail = ""
        return cfg.redact(f"HTTP {r.status_code} {detail[:160]}")

    def test_connection(self) -> SendResult:
        """Checks the credentials against the account resource - sends nothing."""
        if mock_mode():
            return SendResult("mock", "mock-sms")
        if not self.configured:
            return SendResult("not_configured", self.name, error=self.missing())
        try:
            r = requests.get(f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}.json",
                             auth=(self.sid, self.token), timeout=TIMEOUT_S)
            if r.status_code >= 400:
                return SendResult("failed", self.name, error=self._err(r))
            return SendResult("connected", self.name, error=None)
        except requests.RequestException as exc:
            return SendResult("failed", self.name, error=type(exc).__name__)

    def send(self, to: str, body: str) -> SendResult:
        if mock_mode():
            return SendResult("mock", "mock-sms", f"mock-{uuid.uuid4().hex[:10]}")
        if not self.configured:
            return SendResult("not_configured", self.name, error=f"SMS provider not configured ({self.missing()})")
        if not cfg.valid_phone(to):
            return SendResult("failed", self.name, error="recipient number must be in E.164 format, e.g. +919812345678")
        try:
            r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json",
                              data={"To": to, "From": self.from_number, "Body": body[:1500]},
                              auth=(self.sid, self.token), timeout=TIMEOUT_S)
            if r.status_code >= 400:
                return SendResult("failed", self.name, error=self._err(r))
            j = r.json()
            return SendResult(TWILIO_STATUSES.get(j.get("status", "queued"), "unknown"), self.name, j.get("sid"),
                              j.get("error_message"))
        except requests.RequestException as exc:
            return SendResult("failed", self.name, error=type(exc).__name__)

    def status(self, message_sid: str) -> SendResult:
        """Current provider status of a sent message (the only source of 'delivered')."""
        if mock_mode() or not message_sid or message_sid.startswith("mock-"):
            return SendResult("mock", "mock-sms", message_sid)
        if not self.configured:
            return SendResult("not_configured", self.name, message_sid, self.missing())
        try:
            r = requests.get(f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages/{message_sid}.json",
                             auth=(self.sid, self.token), timeout=TIMEOUT_S)
            if r.status_code >= 400:
                return SendResult("unknown", self.name, message_sid, self._err(r))
            j = r.json()
            err = j.get("error_message") or (f"error code {j['error_code']}" if j.get("error_code") else None)
            return SendResult(TWILIO_STATUSES.get(j.get("status"), "unknown"), self.name, message_sid, err)
        except requests.RequestException as exc:
            return SendResult("unknown", self.name, message_sid, type(exc).__name__)


def provider_status() -> dict:
    """What is configured (presence / problems only, never values) - shown in the UI."""
    e, s = EmailProvider(), sms_provider()
    return {"mode": "mock" if mock_mode() else "live",
            "email": "configured" if e.configured else f"not configured - {e.missing()}",
            "sms": "configured" if s.configured else f"not configured - {s.missing()}"}


# ── textbee.dev Android SMS gateway ─────────────────────────────────────────
# API (https://textbee.dev/docs/api-reference, openapi https://textbee.dev/openapi.json):
#   auth     header  x-api-key: <key>
#   send     POST {base}/gateway/send-sms   {"recipients": ["+91..."], "message": "...", "deviceId": "..."}
#            -> {"data": {"success", "message", "smsBatchId", "recipientCount", "successCount",
#                         "failureCount", "warning"}}
#   devices  GET  {base}/gateway/devices    -> {"data": [{"_id", "enabled", "isDefault", "name", "brand",
#                                                         "model", "lastHeartbeat", ...}]}
#   status   GET  {base}/gateway/devices/{id}/sms-batch/{smsBatchId}
#            -> {"data": {"batch": {"status", ...}, "messages": [{"status", ...}]}}
#   message status: pending | dispatched | sent | delivered | failed | unknown | received
#   401 missing / invalid / revoked key; 429 daily, monthly or per-batch plan limit used up.

TEXTBEE_STATUSES = {"pending": "accepted",       # accepted by the gateway, waiting for the phone
                    "dispatched": "queued",      # handed to the phone
                    "sent": "sent",              # the phone sent it to the mobile network
                    "delivered": "delivered",    # delivery report received by the phone
                    "failed": "failed", "unknown": "unknown", "received": "unknown",
                    # batch-level values
                    "processing": "queued", "completed": "sent", "partial_success": "unknown"}
HEARTBEAT_STALE_MIN = 45                  # textbee: heartbeat every 30 min; < 45 min is healthy
_DEVICE_CACHE: dict = {}


def _age_minutes(iso: Optional[str]) -> Optional[float]:
    from datetime import datetime, timezone
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds() / 60.0)
    except ValueError:
        return None


def _not_sent(exc: Exception) -> bool:
    """True only when the request certainly never reached the gateway (safe to retry a POST)."""
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    text = repr(exc)
    return isinstance(exc, requests.exceptions.ConnectionError) and any(
        k in text for k in ("NewConnectionError", "NameResolutionError", "Failed to establish a new connection",
                            "getaddrinfo failed", "Name or service not known"))


class TextbeeSmsProvider:
    """SMS through the textbee.dev Android gateway: the user's Android phone (textbee app, signed in,
    SMS permission granted, online) sends each message from its own SIM."""
    name = "textbee"
    RETRIES = 2                      # extra attempts, ONLY when the request never left this machine

    def __init__(self, session=None, sleep=None):
        s = cfg.textbee_settings()
        self.api_key, self.device_id, self.base_url = s["api_key"], s["device_id"], s["base_url"]
        self.timeout = s["timeout"] if s["timeout"] > 0 else 15.0
        self._s = s
        self.http = session or requests
        import time
        self._sleep = sleep or time.sleep

    @property
    def configured(self) -> bool:
        return not cfg.textbee_problems(self._s)

    def missing(self) -> str:
        return "; ".join(cfg.textbee_problems(self._s)) or "-"

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "Content-Type": "application/json", "Accept": "application/json"}

    def _err(self, r) -> str:
        try:
            j = r.json()
            msg = j.get("message") or j.get("error") or (j.get("data") or {}).get("message") or ""
            if isinstance(msg, list):
                msg = "; ".join(map(str, msg))
        except ValueError:
            msg = ""
        kind = {400: "rejected (check the recipient number and message)", 401: "authentication failed - API key "
                "missing, invalid or revoked", 403: "forbidden for this API key / device", 404: "not found (check "
                "TEXTBEE_DEVICE_ID / TEXTBEE_BASE_URL)", 422: "rejected (invalid request)",
                429: "plan limit reached (daily, monthly or per-batch SMS limit)"}.get(
            r.status_code, "gateway server error" if r.status_code >= 500 else "request failed")
        return cfg.redact(f"HTTP {r.status_code} {kind}" + (f": {str(msg)[:160]}" if msg else ""))

    def _get(self, path: str):
        return self.http.get(f"{self.base_url}{path}", headers=self._headers(), timeout=(5, self.timeout))

    def devices(self):
        """(devices list, error SendResult or None)."""
        try:
            r = self._get("/gateway/devices")
        except requests.RequestException as exc:
            return [], SendResult("failed", self.name, error=cfg.redact(f"connection failed: {type(exc).__name__}"))
        if r.status_code >= 400:
            return [], SendResult("failed", self.name, error=self._err(r))
        try:
            data = r.json().get("data") or []
        except ValueError:
            return [], SendResult("failed", self.name, error="unexpected answer from the gateway (not JSON)")
        return (data if isinstance(data, list) else []), None

    def _pick_device(self, devices: list):
        if self.device_id:
            return next((d for d in devices if str(d.get("_id")) == self.device_id), None)
        enabled = [d for d in devices if d.get("enabled")]
        default = [d for d in enabled if d.get("isDefault")]
        if default:
            return default[0]
        return sorted(enabled, key=lambda d: str(d.get("lastHeartbeat") or ""), reverse=True)[0] if enabled else None

    @staticmethod
    def _describe(d: dict) -> str:
        label = d.get("name") or " ".join(x for x in (d.get("brand"), d.get("model")) if x) or "Android device"
        age = _age_minutes(d.get("lastHeartbeat"))
        seen = "no heartbeat recorded" if age is None else (
            f"last heartbeat {age:.0f} min ago" if age < 120 else f"last heartbeat {age / 60:.1f} h ago")
        return f"{label} ({'enabled' if d.get('enabled') else 'DISABLED'}, {seen})"

    def test_connection(self) -> SendResult:
        """Checks the API key and the registered Android phone - sends nothing."""
        if mock_mode():
            return SendResult("mock", "mock-sms")
        if not self.configured:
            return SendResult("not_configured", self.name, error=self.missing())
        devices, err = self.devices()
        if err:
            return err
        if not devices:
            return SendResult("failed", self.name, error="no Android device registered: install the textbee app on "
                                                         "the phone and sign in / scan the QR code from the dashboard")
        d = self._pick_device(devices)
        if d is None:
            return SendResult("failed", self.name, error=("TEXTBEE_DEVICE_ID does not match any registered device"
                                                          if self.device_id else "no enabled device - enable the "
                                                          "phone in the textbee app / dashboard"))
        if not d.get("enabled"):
            return SendResult("failed", self.name, error="the selected device is disabled: " + self._describe(d))
        age = _age_minutes(d.get("lastHeartbeat"))
        note = self._describe(d)
        if age is None or age > HEARTBEAT_STALE_MIN:
            note += " - the phone may be offline or asleep; messages will wait until it reconnects"
        _DEVICE_CACHE[self.api_key[-6:]] = str(d.get("_id"))
        return SendResult("connected", self.name, detail=note)

    def _device_for_send(self) -> Optional[str]:
        """Explicit device id (configured, else the gateway's own choice resolved once) so the SMS status
        can be looked up later. None if it cannot be resolved (textbee then picks the device itself)."""
        if self.device_id:
            return self.device_id
        key = self.api_key[-6:]
        if key not in _DEVICE_CACHE:
            devices, err = self.devices()
            d = None if err else self._pick_device(devices)
            if d is not None:
                _DEVICE_CACHE[key] = str(d.get("_id"))
        return _DEVICE_CACHE.get(key)

    def send(self, to: str, body: str) -> SendResult:
        if mock_mode():
            return SendResult("mock", "mock-sms", f"mock-{uuid.uuid4().hex[:10]}")
        if not self.configured:
            return SendResult("not_configured", self.name, error=f"SMS gateway not configured ({self.missing()})")
        if not cfg.valid_phone(to):
            return SendResult("failed", self.name, error="recipient number must be in international E.164 format, "
                                                         "e.g. +919812345678")
        device = self._device_for_send()
        payload = {"recipients": [to.strip()], "message": body[:1500]}
        if device:
            payload["deviceId"] = device
        for attempt in range(self.RETRIES + 1):
            try:
                r = self.http.post(f"{self.base_url}/gateway/send-sms", json=payload, headers=self._headers(),
                                   timeout=(5, self.timeout))
                break
            except requests.exceptions.ReadTimeout:
                return SendResult("unknown", self.name, error=f"the gateway did not answer within {self.timeout:.0f} s; "
                                  "the SMS may or may not have been queued - not retried, to avoid a duplicate")
            except requests.RequestException as exc:
                if _not_sent(exc) and attempt < self.RETRIES:
                    self._sleep(1.5 * (attempt + 1))
                    continue
                if _not_sent(exc):
                    return SendResult("failed", self.name, error=f"connection failed: cannot reach {self.base_url} "
                                      f"({type(exc).__name__}); nothing was sent")
                return SendResult("unknown", self.name, error=f"connection interrupted ({type(exc).__name__}); the SMS "
                                  "may or may not have been queued - not retried, to avoid a duplicate")
        if r.status_code >= 400:
            return SendResult("failed", self.name, error=self._err(r))
        try:
            data = r.json().get("data") or {}
        except ValueError:
            return SendResult("unknown", self.name, error="the gateway answered without JSON; check the textbee dashboard")
        if data.get("success") is False or (data.get("successCount") == 0 and (data.get("failureCount") or 0) > 0):
            return SendResult("failed", self.name, error=cfg.redact(str(data.get("message") or "rejected by the gateway")[:200]))
        batch = data.get("smsBatchId")
        mid = f"{device}:{batch}" if (device and batch) else (batch or None)
        warn = data.get("warning") or {}
        return SendResult("accepted", self.name, mid,
                          detail=(f"gateway warning: {warn.get('message')}" if warn.get("message") else
                                  "accepted by the textbee gateway; the phone sends it next - delivery not yet confirmed"))

    def status(self, message_id: str) -> SendResult:
        """Current status of a submitted SMS from textbee's own records (the only source of 'delivered')."""
        if mock_mode() or not message_id or str(message_id).startswith("mock-"):
            return SendResult("mock", "mock-sms", message_id)
        if not self.configured:
            return SendResult("not_configured", self.name, message_id, self.missing())
        if ":" not in str(message_id):
            return SendResult("unknown", self.name, message_id, "no device id stored for this SMS; check the textbee dashboard")
        device, batch = str(message_id).split(":", 1)
        try:
            r = self._get(f"/gateway/devices/{device}/sms-batch/{batch}")
        except requests.RequestException as exc:
            return SendResult("unknown", self.name, message_id, f"status check failed: {type(exc).__name__}")
        if r.status_code >= 400:
            return SendResult("unknown", self.name, message_id, self._err(r))
        try:
            data = r.json().get("data") or {}
        except ValueError:
            return SendResult("unknown", self.name, message_id, "unexpected answer from the gateway")
        msgs = data.get("messages") or []
        raw = (msgs[0].get("status") if msgs and isinstance(msgs[0], dict) else None) or (data.get("batch") or {}).get("status")
        err = (msgs[0].get("errorMessage") or msgs[0].get("error")) if msgs and isinstance(msgs[0], dict) else None
        return SendResult(TEXTBEE_STATUSES.get(str(raw or "").lower(), "unknown"), self.name, message_id,
                          cfg.redact(str(err)[:160]) if err else None)


def sms_provider():
    """The configured SMS gateway (SMS_PROVIDER): TextbeeSmsProvider or the Twilio SmsProvider."""
    return SmsProvider() if cfg.sms_provider_name() == "twilio" else TextbeeSmsProvider()
