"""
Offline UI timing (no API keys, no network): dashboard imports, regional twin
build (demo data), and the What-If Simulator / Spread Simulation pages run
through Streamlit's AppTest (first run and a rerun). Real API calls are
impossible here: the data-API keys are blanked and every HTTP request is
refused. Usage: python scripts/benchmark_ui_offline.py
"""
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

import requests  # noqa: E402


def _refuse(*a, **k):
    raise requests.ConnectionError("benchmark: network disabled")


requests.Session.send = _refuse
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "bench.db")   # never the real DB
os.environ.pop("FIRMS_MAP_KEY", None)
os.environ.pop("OWM_API_KEY", None)
os.environ["GOOGLE_MAPS_API_KEY"] = "BENCHMARK-NOT-A-KEY"       # map payloads are built; nothing is sent

from config.config import API  # noqa: E402

API.firms_map_key = ""
API.owm_api_key = ""


def t(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"{label:<58}{time.perf_counter() - t0:8.2f} s", flush=True)
    return out


def main():
    t("import dashboard modules", lambda: __import__("src.dashboard.dashboard_common"))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "src" / "dashboard" / "app.py"), default_timeout=240)
    at.session_state["auth_user"] = "admin"
    at.session_state["auth_role"] = "admin"
    at.session_state["_pref_offline_mode"] = True
    t("app start: Command Center first run (demo twin build)", at.run)
    for page in ("pages/1_What_If_Simulator.py", "pages/2_Spread_Simulation.py"):
        at.switch_page(page)
        t(f"{Path(page).stem}: first run", at.run)
        t(f"{Path(page).stem}: rerun (what a map event triggers)", at.run)
        if at.exception:
            print("  exception:", at.exception[0].message[:200])


if __name__ == "__main__":
    main()
