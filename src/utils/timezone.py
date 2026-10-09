"""
timezone.py - the ONE place where timestamps are converted for display.

The data layer stays UTC (API responses, caches, database rows, simulation
state). The dashboard calls these helpers only when it renders a timestamp, so
the user sees India Standard Time (Asia/Kolkata, UTC+05:30):

    2026-10-07T17:06:00Z   ->   07 Oct 2026, 10:36 PM IST

Accepted inputs: ISO 8601 strings (with Z, +00:00 or any other offset, with or
without seconds/microseconds, "T" or space), strings ending in " UTC" / " IST",
datetime / pandas Timestamp (aware, or naive = UTC as stored by this project),
and Unix epoch seconds (or milliseconds). Values that already carry an offset
(e.g. an IST time) are converted, never shifted twice. A date without a time is
shown as a date (no shift). Missing or unparsable values give `missing`.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        IST = ZoneInfo("Asia/Kolkata")
    except ZoneInfoNotFoundError:                       # Windows without the tzdata package
        IST = timezone(timedelta(hours=5, minutes=30), "IST")
except ImportError:                                     # pragma: no cover - Python < 3.9
    IST = timezone(timedelta(hours=5, minutes=30), "IST")

UTC = timezone.utc
FORMATS = {
    "full": "%d %b %Y, %I:%M %p IST",                  # 07 Oct 2026, 10:36 PM IST
    "dot": "%d %b %Y · %I:%M %p IST",                  # 07 Oct 2026 · 10:36 PM IST
    "compact": "%I:%M %p IST",                         # 10:36 PM IST
    "seconds": "%d %b %Y, %I:%M:%S %p IST",            # 07 Oct 2026, 10:36:05 PM IST
}
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SUFFIX = re.compile(r"\s*(UTC|GMT|IST)$", re.IGNORECASE)


def to_utc(value) -> Optional[datetime]:
    """Aware UTC datetime for any supported input, or None."""
    if value is None:
        return None
    try:
        import pandas as pd
        if value is pd.NaT or (isinstance(value, float) and value != value):
            return None
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
    except ImportError:                                 # pragma: no cover
        pass
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        secs = float(value) / (1000.0 if abs(value) > 1e11 else 1.0)   # epoch ms or s
        try:
            return datetime.fromtimestamp(secs, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, datetime):
        return (value.replace(tzinfo=UTC) if value.tzinfo is None else value).astimezone(UTC)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or _DATE_ONLY.match(s):
        return None
    m = _SUFFIX.search(s)
    tz = UTC
    if m:
        tz = IST if m.group(1).upper() == "IST" else UTC
        s = s[:m.start()].strip()
    s = s.replace(" ", "T", 1) if "T" not in s else s
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            import pandas as pd
            ts = pd.to_datetime(s, errors="coerce")
            if ts is pd.NaT or ts is None:
                return None
            dt = ts.to_pydatetime()
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)                      # naive = the stated zone, else UTC (project convention)
    return dt.astimezone(UTC)


def utc_to_ist(value) -> Optional[datetime]:
    """The same instant in Asia/Kolkata (aware datetime), or None."""
    dt = to_utc(value)
    return dt.astimezone(IST) if dt is not None else None


def format_ist(value, style: str = "full", missing: str = "-") -> str:
    """Display string in IST: full '07 Oct 2026, 10:36 PM IST', dot
    '07 Oct 2026 · 10:36 PM IST', compact '10:36 PM IST', seconds
    '07 Oct 2026, 10:36:05 PM IST'. A bare date is shown as '07 Oct 2026'."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.strftime("%d %b %Y")
    if isinstance(value, str) and _DATE_ONLY.match(value.strip()):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").strftime("%d %b %Y")
        except ValueError:
            return missing
    dt = utc_to_ist(value)
    return dt.strftime(FORMATS.get(style, FORMATS["full"])) if dt is not None else missing


# alias matching the requested API name
format_ist_timestamp = format_ist


def format_ist_series(series, style: str = "full", missing: str = "-"):
    """pandas Series of timestamps -> Series of IST display strings."""
    return series.map(lambda v: format_ist(v, style, missing))


_EMBEDDED = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2}|\s?UTC)?")


def localize_text(text: Optional[str], style: str = "full") -> Optional[str]:
    """Replace UTC timestamps embedded in a status message (e.g. 'cached from
    2026-10-07T17:06:00+00:00') with their IST display form."""
    if not text:
        return text
    return _EMBEDDED.sub(lambda m: format_ist(m.group(0), style, m.group(0)), str(text))
