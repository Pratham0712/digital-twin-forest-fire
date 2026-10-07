"""
Activity Log - live, newest-first feed of everything the system and its users
do: refreshes (with per-stage timing), new EXTREME detections, alert emails,
simulations, scenario changes, report downloads, sign-ins and admin changes.
Read straight from the activity_log table; the feed polls every few seconds.
"""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from src.dashboard.dashboard_common import set_page, build_sidebar
from src.auth.auth_gate import require_login, render_user_badge_in_sidebar
from src.storage import database as db

set_page("Activity Log")
require_login()
build_sidebar()
render_user_badge_in_sidebar()

EVENT_LABELS = {
    "refresh": "Refresh",
    "new_extreme": "New EXTREME zone",
    "email_sent": "Email sent",
    "email_failed": "Email failed",
    "simulation": "Spread simulation",
    "scenario": "Scenario",
    "report": "Report",
    "login": "Sign in",
    "login_failed": "Failed sign-in",
    "logout": "Sign out",
    "admin": "Admin",
    "error": "Error",
}
from src.utils.timezone import IST as LOCAL_TZ, format_ist_series  # noqa: E402 - one shared converter

st.markdown("""
<div class="hero">
  <div class="eyebrow">Forest Fire Digital Twin</div>
  <h1>Activity Log</h1>
  <div class="sub">Every refresh, detection, alert and user action, as it happens.</div>
</div>
""", unsafe_allow_html=True)

db.init_db()

if "activity_filter" not in st.session_state:
    st.session_state["activity_filter"] = []
if "activity_limit" not in st.session_state:
    st.session_state["activity_limit"] = 100

c1, c2 = st.columns([3, 1])
with c1:
    st.multiselect("Show only", options=list(EVENT_LABELS.keys()),
                   format_func=lambda k: EVENT_LABELS.get(k, k), key="activity_filter",
                   placeholder="All events")
with c2:
    st.selectbox("Rows", [50, 100, 250, 500], key="activity_limit")


def _to_local(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="coerce", format="ISO8601")
    return t.dt.tz_convert(LOCAL_TZ)


@st.fragment(run_every=timedelta(seconds=10))
def live_feed():
    # One query per poll; the filter and row limit are applied in memory.
    fetched = db.get_activity(limit=max(500, st.session_state["activity_limit"]))
    recent = pd.DataFrame(fetched)
    wanted = st.session_state["activity_filter"]
    rows = [r for r in fetched if not wanted or r["event_type"] in wanted][:st.session_state["activity_limit"]]

    # headline numbers for the last 24 hours
    k1, k2, k3, k4 = st.columns(4)
    if not recent.empty:
        recent["t"] = _to_local(recent["timestamp_utc"])
        day = recent[recent["t"] >= pd.Timestamp.now(tz=LOCAL_TZ) - pd.Timedelta(hours=24)]
        refreshes = day[day["event_type"] == "refresh"]
        secs = refreshes["detail"].str.extract(r"· ([\d.]+)s \(")[0].astype(float)
        k1.metric("Events (24h)", f"{len(day):,}")
        k2.metric("Refreshes (24h)", f"{len(refreshes):,}")
        k3.metric("Median refresh time", f"{secs.median():.1f}s" if secs.notna().any() else "—")
        k4.metric("New EXTREME alerts (24h)", f"{int((day['event_type'] == 'new_extreme').sum()):,}")
    else:
        for k, label in zip((k1, k2, k3, k4), ("Events (24h)", "Refreshes (24h)",
                                                "Median refresh time", "New EXTREME alerts (24h)")):
            k.metric(label, "—")

    if not rows:
        st.info("No activity yet.")
        return

    df = pd.DataFrame(rows)
    df["Time"] = format_ist_series(df["timestamp_utc"], "seconds")
    df["Event"] = df["event_type"].map(lambda e: EVENT_LABELS.get(e, e))
    df["User"] = df["actor"]
    df["Region"] = df["region_name"].fillna("—")
    df["Details"] = df["detail"].fillna("")
    st.dataframe(
        df[["Time", "Event", "User", "Region", "Details"]],
        use_container_width=True, hide_index=True, height=560,
        column_config={
            "Time": st.column_config.TextColumn(width="medium"),
            "Event": st.column_config.TextColumn(width="medium"),
            "User": st.column_config.TextColumn(width="small"),
            "Region": st.column_config.TextColumn(width="medium"),
            "Details": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"Stored in {db.backend_name()} · updates every 10 seconds")


live_feed()
