"""
config.py - notification settings (email / SMS) from one place.

Resolution order for every setting:
  1. a value entered by an administrator on the Admin page for THIS running app
     (kept in process memory only - never written to disk, the database, logs
     or reports; cleared on restart),
  2. the environment / .env (local development),
  3. Streamlit secrets (hosted deployments).

Secrets (SMTP_PASSWORD, TWILIO_AUTH_TOKEN) are never returned by `public_view()`
and `redact()` removes them from any error text before it is shown or logged.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Dict, Optional

SECRET_KEYS = ("SMTP_PASSWORD", "TWILIO_AUTH_TOKEN", "TEXTBEE_API_KEY")
KEYS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_SECURITY", "SMTP_TIMEOUT",
        "ALERT_RECIPIENT_EMAILS", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
        "ALERT_PROVIDER_MODE", "DASHBOARD_URL",
        "SMS_PROVIDER", "TEXTBEE_API_KEY", "TEXTBEE_DEVICE_ID", "TEXTBEE_BASE_URL", "TEXTBEE_TIMEOUT",
        "ALERT_SMS_RECIPIENTS")
SMS_PROVIDERS = ("textbee", "twilio")
TEXTBEE_DEFAULT_BASE = "https://api.textbee.dev/api/v1"
SECURITY_MODES = ("starttls", "ssl", "none")
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")
PLACEHOLDER_RE = re.compile(r"^(your[_-]|changeme|example|xxx)", re.I)

_LOCK = threading.Lock()
_RUNTIME: Dict[str, str] = {}          # admin-entered values for this process only


def _secret(key: str) -> Optional[str]:
    try:
        import streamlit as st
        return str(st.secrets[key]) if key in st.secrets else None
    except Exception:
        return None


def get(key: str, default: str = "") -> str:
    with _LOCK:
        if key in _RUNTIME:
            return _RUNTIME[key]
    v = os.getenv(key)
    if v is not None and v != "":
        return v
    s = _secret(key)
    return s if s not in (None, "") else default


def source_of(key: str) -> str:
    with _LOCK:
        if key in _RUNTIME:
            return "admin (this session)"
    if os.getenv(key):
        return "environment"
    if _secret(key):
        return "secrets"
    return "not set"


def set_runtime(values: Dict[str, Optional[str]]) -> None:
    """Admin-entered values for the running app. Empty secret fields keep the current value."""
    with _LOCK:
        for k, v in values.items():
            if k not in KEYS:
                continue
            if v is None or (k in SECRET_KEYS and v == ""):
                continue
            _RUNTIME[k] = str(v).strip()


def clear_runtime(prefix: str = "") -> None:
    with _LOCK:
        for k in [k for k in _RUNTIME if k.startswith(prefix)]:
            del _RUNTIME[k]


def is_placeholder(v: str) -> bool:
    return bool(v) and bool(PLACEHOLDER_RE.match(v))


def valid_email(v: str) -> bool:
    return bool(v) and bool(EMAIL_RE.match(v.strip()))


def normalise_phone(raw: str) -> Optional[str]:
    """Indian mobile numbers in any common form -> '+91XXXXXXXXXX' (10 digits starting 6-9); other numbers must
    already be E.164 ('+<country><number>'). None when invalid."""
    v = re.sub(r"[\s\-().]", "", str(raw or ""))
    if not v:
        return None
    if v.startswith("00"):
        v = "+" + v[2:]
    digits = v[1:] if v.startswith("+") else v
    if not digits.isdigit():
        return None
    if not v.startswith("+"):
        if len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) != 10:
            return None
        v = "+91" + digits
    if v.startswith("+91") and not re.fullmatch(r"\+91[6-9]\d{9}", v):
        return None
    return v if E164_RE.match(v) else None


def valid_phone(v: str) -> bool:
    return bool(v) and bool(E164_RE.match(v.strip()))


def smtp_settings() -> dict:
    port_raw = get("SMTP_PORT", "587")
    try:
        port = int(port_raw)
    except ValueError:
        port = -1
    sec = (get("SMTP_SECURITY") or ("ssl" if port == 465 else "starttls")).lower()
    try:
        timeout = float(get("SMTP_TIMEOUT", "15"))
    except ValueError:
        timeout = 15.0
    user = get("SMTP_USER")
    return {"host": get("SMTP_HOST"), "port": port, "user": user, "password": get("SMTP_PASSWORD"),
            "sender": get("SMTP_FROM") or user, "security": sec, "timeout": timeout,
            "recipients": [r.strip() for r in get("ALERT_RECIPIENT_EMAILS").split(",") if r.strip()]}


def smtp_problems(s: Optional[dict] = None) -> list:
    """What is missing or invalid (empty list = complete). Never includes secret values."""
    s = s or smtp_settings()
    out = []
    if not s["host"]:
        out.append("SMTP host is not set")
    if not (0 < s["port"] < 65536):
        out.append("SMTP port must be a number between 1 and 65535")
    if s["security"] not in SECURITY_MODES:
        out.append(f"SMTP security must be one of {', '.join(SECURITY_MODES)}")
    if not s["user"] or is_placeholder(s["user"]):
        out.append("SMTP username is not set")
    if not s["password"] or is_placeholder(s["password"]):
        out.append("SMTP password / app password is not set")
    if s["sender"] and not valid_email(s["sender"]):
        out.append("sender address is not a valid email address")
    bad = [r for r in s["recipients"] if not valid_email(r)]
    if bad:
        out.append(f"{len(bad)} recipient address(es) are not valid email addresses")
    return out


def sms_settings() -> dict:
    return {"provider": "twilio", "sid": get("TWILIO_ACCOUNT_SID"), "token": get("TWILIO_AUTH_TOKEN"),
            "from_number": get("TWILIO_FROM_NUMBER")}


def sms_problems(s: Optional[dict] = None) -> list:
    s = s or sms_settings()
    out = []
    if not s["sid"] or is_placeholder(s["sid"]):
        out.append("Twilio Account SID is not set")
    elif not re.match(r"^AC[0-9a-fA-F]{32}$", s["sid"]):
        out.append("Twilio Account SID should look like AC followed by 32 hex characters")
    if not s["token"] or is_placeholder(s["token"]):
        out.append("Twilio auth token is not set")
    if not s["from_number"]:
        out.append("Twilio sender number is not set")
    elif not valid_phone(s["from_number"]):
        out.append("Twilio sender number must be in E.164 format, e.g. +14155550100")
    return out


def sms_provider_name() -> str:
    """Configured SMS gateway: SMS_PROVIDER if set, else textbee when its API key is present,
    else twilio when Twilio credentials are present, else textbee (shown as not configured)."""
    v = get("SMS_PROVIDER").strip().lower()
    if v in SMS_PROVIDERS:
        return v
    if get("TEXTBEE_API_KEY"):
        return "textbee"
    if get("TWILIO_ACCOUNT_SID") or get("TWILIO_AUTH_TOKEN"):
        return "twilio"
    return "textbee"


def textbee_settings() -> dict:
    try:
        timeout = float(get("TEXTBEE_TIMEOUT") or "15")
    except ValueError:
        timeout = -1.0
    return {"provider": "textbee", "api_key": get("TEXTBEE_API_KEY"), "device_id": get("TEXTBEE_DEVICE_ID").strip(),
            "base_url": (get("TEXTBEE_BASE_URL") or TEXTBEE_DEFAULT_BASE).strip().rstrip("/"), "timeout": timeout}


def textbee_problems(s: Optional[dict] = None) -> list:
    """What is missing or invalid (empty list = complete). Never includes the API key."""
    s = s or textbee_settings()
    out = []
    if not s["api_key"] or is_placeholder(s["api_key"]):
        out.append("textbee API key (TEXTBEE_API_KEY) is not set")
    elif len(s["api_key"]) < 16 or any(ch.isspace() for ch in s["api_key"]):
        out.append("textbee API key looks malformed (too short or contains spaces)")
    if not s["base_url"].startswith("https://"):
        out.append("TEXTBEE_BASE_URL must start with https://")
    if s["device_id"] and not re.match(r"^[A-Za-z0-9_-]{6,64}$", s["device_id"]):
        out.append("TEXTBEE_DEVICE_ID looks malformed (copy the device id from the textbee dashboard)")
    if not (1 <= s["timeout"] <= 120):
        out.append("TEXTBEE_TIMEOUT must be a number of seconds between 1 and 120")
    return out


def sms_recipients() -> list:
    """Phone numbers (E.164) for AUTOMATIC fire alerts (ALERT_SMS_RECIPIENTS, comma separated)."""
    return [r.strip() for r in get("ALERT_SMS_RECIPIENTS").split(",") if r.strip()]


def mock_mode() -> bool:
    return get("ALERT_PROVIDER_MODE", "live").strip().lower() == "mock"


def redact(text: str) -> str:
    """Removes secret values (and the SMTP user's password-like strings) from text."""
    t = str(text or "")
    for k in SECRET_KEYS:
        v = get(k)
        if v and len(v) >= 4:
            t = t.replace(v, "•••")
    return t


def public_view() -> dict:
    """Settings for display: secrets reduced to 'set' / 'not set'."""
    out = {}
    for k in KEYS:
        v = get(k)
        out[k] = ("set" if v else "not set") if k in SECRET_KEYS else v
        out[k + "__source"] = source_of(k)
    return out
