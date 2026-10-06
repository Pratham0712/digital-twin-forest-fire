"""
scheduler.py - background refresh scheduler (cross-question 10: "there's a
scheduling algorithm behind the OS task manager, we should use something
similar - not exactly that, but similar").

This runs as its OWN process, separate from the Streamlit dashboard, for the
same reason a real production system separates "keep the data fresh" from
"render the UI": the dashboard should stay responsive even if a refresh
cycle is slow (API calls, model inference), and the data should keep
updating even if nobody has the dashboard open in a browser right now.

It uses APScheduler's BlockingScheduler to run DigitalTwin.refresh() on a
fixed interval (SYSTEM.min_refresh_interval_minutes, default 15 - see
config/config.py) for every configured region, and on each cycle:
  1. reads which zones were EXTREME in the previous snapshot for that region
     (database.get_previously_extreme_zone_ids)
  2. refreshes the twin
  3. saves the new snapshot + alerts (database.save_snapshot)
  4. diffs current EXTREME zones against the previous set -> genuinely NEW
     extreme detections only
  5. emails those new detections (email_notifier.send_new_extreme_alert)

Run it:
    python -m src.scheduler.scheduler                 # runs forever, cron-like
    python -m src.scheduler.scheduler --once           # single cycle then exit
                                                        # (use this from an OS
                                                        # task scheduler / cron
                                                        # if you'd rather let
                                                        # the OS own the timing)
    python -m src.scheduler.scheduler --regions "Karnataka Western Ghats,California"
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.config import REGION, SYSTEM, API, MODELS_DIR
from src.digital_twin.twin_state import DigitalTwin, persist_snapshot_and_notify
from src.storage import database as db
from src.notifications.email_notifier import EmailNotifier
from src.regions import REGION_PRESETS

logger = logging.getLogger(__name__)


def load_ml_model(region=None):
    """Plain (non-Streamlit-cached) model loader for this headless process."""
    from src.ml_models.model_registry import choose_for_region, load_xgb
    return load_xgb(choose_for_region(region).model_file)


def run_one_cycle(region_name: str, offline: bool, notifier: EmailNotifier):
    region = REGION_PRESETS.get(region_name, REGION)
    logger.info("[%s] Starting refresh cycle...", region.name)

    twin = DigitalTwin(ml_model=load_ml_model(region), offline=offline, region=region)
    snapshot = twin.refresh()
    # Same persistence + alert-email + activity-log path the dashboard uses.
    result = persist_snapshot_and_notify(twin, notifier=notifier, actor="scheduler",
                                         trigger="Background scheduler")
    logger.info("[%s] %d alerts, %d newly extreme, emailed=%s, %.1fs",
                region.name, len(snapshot.alerts), result["new_extreme_count"],
                result["emailed"], twin.last_timings.get("total_s", 0.0))


def main():
    parser = argparse.ArgumentParser(description="Digital Twin background refresh scheduler")
    parser.add_argument("--once", action="store_true",
                         help="Run a single refresh cycle for each region, then exit "
                              "(use this if you'd rather schedule it with cron / Windows "
                              "Task Scheduler instead of leaving this process running).")
    parser.add_argument("--regions", type=str, default=None,
                         help="Comma-separated region names from REGION_PRESETS "
                              "(default: just the validated Karnataka region).")
    parser.add_argument("--interval-minutes", type=int, default=None,
                         help="Override SYSTEM.min_refresh_interval_minutes.")
    parser.add_argument("--offline", action="store_true", default=None,
                         help="Force offline/demo mode regardless of API keys.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    db.init_db()
    db.ensure_default_admin()

    region_names = (
        [r.strip() for r in args.regions.split(",")] if args.regions
        else [REGION.name]
    )
    offline = args.offline if args.offline is not None else not bool(API.firms_map_key)
    interval_minutes = args.interval_minutes or SYSTEM.min_refresh_interval_minutes
    notifier = EmailNotifier()

    if not notifier.is_configured:
        logger.warning(
            "Email notifications are not configured (set SMTP_HOST, SMTP_USER, "
            "SMTP_PASSWORD, ALERT_RECIPIENT_EMAILS in .env). The scheduler will "
            "still run and log new detections, it just won't email them."
        )

    if args.once:
        for name in region_names:
            run_one_cycle(name, offline, notifier)
        return

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        logger.warning(
            "APScheduler not installed (pip install apscheduler) - falling back "
            "to a plain sleep loop with the same %d-minute interval.", interval_minutes
        )
        while True:
            for name in region_names:
                run_one_cycle(name, offline, notifier)
            time.sleep(interval_minutes * 60)
        return

    scheduler = BlockingScheduler()
    for name in region_names:
        scheduler.add_job(
            run_one_cycle, "interval", minutes=interval_minutes,
            args=[name, offline, notifier], next_run_time=None,
            id=f"refresh_{name}", name=f"Refresh: {name}",
        )
    logger.info(
        "Scheduler started - refreshing %s every %d minute(s). Press Ctrl+C to stop.",
        ", ".join(region_names), interval_minutes,
    )
    # Run one cycle immediately on startup, then let the interval jobs take over.
    for name in region_names:
        run_one_cycle(name, offline, notifier)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
