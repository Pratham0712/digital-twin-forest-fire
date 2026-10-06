"""Where does the time go?  python scripts/profile_startup.py
Prints how long each start-up stage takes on THIS machine (database round
trip, imports, model load, offline and live data refresh)."""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

def stage(label, fn):
    t = time.time()
    try:
        fn(); note = ""
    except Exception as exc:
        note = f"   FAILED: {type(exc).__name__}"
    print(f"{label:<46}{time.time() - t:7.2f}s{note}", flush=True)

from src.storage import database as db
print("Database:", db.backend_name(), "\n")
stage("connect + 1 query (first, includes TLS)", lambda: db.get_table_counts())
stage("1 more query (steady-state round trip)", lambda: db.get_table_counts())
stage("init_db (table check)", db.init_db)
stage("import dashboard modules", lambda: __import__("src.dashboard.dashboard_common"))
from src.dashboard import dashboard_common as dc
from src.regions import REGION_PRESETS
region = REGION_PRESETS[dc.DEFAULT_PRESET]
holder = {}
def build(offline):
    def _b():
        holder["t"] = dc.get_twin(offline, None, region); holder["t"].refresh()
    return _b
stage("build twin + refresh, OFFLINE", build(True))
if os.getenv("FIRMS_MAP_KEY"):
    stage("build twin + refresh, LIVE (satellite + weather)", build(False))
else:
    print("LIVE refresh skipped (no FIRMS_MAP_KEY in .env)")
