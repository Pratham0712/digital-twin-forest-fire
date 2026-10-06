"""
DigitalTwin - Layer 5 core (Report Ch.6.1.1): maintains the live state of the
Karnataka/Western Ghats region as a single object that ties together:
  - the latest ingested + processed grid (Layers 1-2)
  - ML risk scores per zone (Layer 3)
  - the most recent CA spread projection (Layer 4)
  - alert state, per SRS 5.1 Alert System (threshold-based, SYSTEM.alert_threshold_pct)

This is the object the Streamlit dashboard reads from directly - it never
talks to the lower layers itself, matching the layered architecture in the
report (Ch.6.1.1: 'the Digital Twin/Dashboard layer visualises state
maintained by the layers beneath it').
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.config import SYSTEM, REGION
from src.data_ingestion.ingestion_module import DataIngestionModule
from src.data_processing.feature_engineering import DataProcessor
from src.simulation.cellular_automata import FireSpreadSimulator, CellState

logger = logging.getLogger(__name__)


@dataclass
class ZoneAlert:
    zone_id: str
    latitude: float
    longitude: float
    risk_score: float
    severity: str  # "LOW", "MODERATE", "HIGH", "EXTREME"
    reason: str
    triggered_at: str


@dataclass
class TwinSnapshot:
    """Immutable snapshot of the entire twin state at one point in time."""
    timestamp: str
    processed_grid: pd.DataFrame
    risk_scores: np.ndarray
    n_rows: int
    n_cols: int
    ca_history: Optional[list] = None
    alerts: List[ZoneAlert] = field(default_factory=list)


class AlertEngine:
    """
    Threshold-based alert classifier per SRS 5.1: 'system issues alerts when
    fire risk probability exceeds 70%'. Severity bands are documented so the
    dashboard and report use identical language.
    """
    SEVERITY_BANDS = [
        (0.60, "EXTREME"),
        (0.40, "HIGH"),
        (0.20, "MODERATE"),
        (0.0,  "LOW"),
    ]

    def __init__(self, threshold_pct: float = SYSTEM.alert_threshold_pct):
        self.threshold = threshold_pct / 100.0

    def classify(self, risk_score: float) -> str:
        for cutoff, label in self.SEVERITY_BANDS:
            if risk_score >= cutoff:
                return label
        return "LOW"

    def generate_alerts(self, processed: pd.DataFrame, risk_scores: np.ndarray) -> List[ZoneAlert]:
        alerts = []
        now = datetime.now(timezone.utc).isoformat()
        for i, row in processed.reset_index(drop=True).iterrows():
            score = float(risk_scores[i])
            if score < self.threshold:
                continue
            severity = self.classify(score)
            reason_parts = []
            if row.get("active_fire_nearby", False):
                reason_parts.append("active satellite hotspot nearby")
            if row.get("fwi", 0) > 60:
                reason_parts.append(f"extreme FWI ({row['fwi']:.1f})")
            if row.get("wx_wind_speed_ms", 0) > 8:
                reason_parts.append(f"high wind ({row['wx_wind_speed_ms']:.1f} m/s)")
            reason = "; ".join(reason_parts) if reason_parts else "elevated model risk score"

            alerts.append(ZoneAlert(
                zone_id=row["zone_id"], latitude=row["latitude"], longitude=row["longitude"],
                risk_score=score, severity=severity, reason=reason, triggered_at=now,
            ))
        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        logger.info("AlertEngine: %d zones above threshold (%.0f%%)", len(alerts), self.threshold * 100)
        return alerts


class DigitalTwin:
    """
    Orchestrates one full refresh cycle: ingest -> process -> predict ->
    (optionally) simulate spread -> generate alerts -> store as the current
    snapshot. Matches Scenario 1 (SRS 6.2.4) end-to-end.
    """

    def __init__(self, ml_model=None, offline: bool = True, use_real_model: bool = True,
                 scenario: Optional[dict] = None, region=None):
        """
        ml_model: any fitted model exposing .predict(X) -> (labels, probs),
        e.g. model_trainer.RandomForestModel or XGBoostModel. If None, the
        twin falls back to using FWI-normalised risk (still principled, just
        not learned) so the twin is usable before model training is run.
        use_real_model: True (default) if ml_model was loaded from a
        *_real.* file (e.g. xgboost_real.json) - builds the matching
        10-column feature matrix instead of the old 12-column demo one.
        Set False only if you deliberately pass a demo-mode model
        (models/xgboost.json etc.).
        """
        self.offline = offline
        self.region = region or REGION
        self.ingestion = DataIngestionModule(offline=offline, scenario=scenario, region=self.region)
        self.processor = DataProcessor()
        self.alert_engine = AlertEngine()
        self.ml_model = ml_model
        self.use_real_model = use_real_model
        self.current_snapshot: Optional[TwinSnapshot] = None
        self._elevation_grid_cache: Optional[np.ndarray] = None  # lazily built, terrain is static

    def _compute_risk_scores(self, processed: pd.DataFrame) -> np.ndarray:
        if self.ml_model is not None:
            X, _ = self.processor.get_feature_matrix(processed, real=self.use_real_model)
            _, probs = self.ml_model.predict(X)
            return probs
        # Fallback: normalise FWI into a pseudo-probability so the twin is
        # still functional before/without a trained model.
        fwi = processed["fwi"].values
        return np.clip(fwi / 100.0, 0, 1)

    def refresh(self) -> TwinSnapshot:
        """Full pipeline refresh - call this on the SYSTEM.min_refresh_interval_minutes cadence."""
        unified = self.ingestion.build_unified_frame()
        processed = self.processor.transform(unified)

        # Inject previous FWI as fwi_lag1 so the real model gets a genuine
        # lag feature instead of a fwi copy. On the very first refresh there
        # is no previous snapshot, so fwi_lag1 falls back to fwi (handled
        # inside DataProcessor.transform via df.get("_prev_fwi", df["fwi"])).
        if self.current_snapshot is not None:
            prev_fwi = (
                self.current_snapshot.processed_grid
                .set_index("zone_id")["fwi"]
                .reindex(processed["zone_id"])
                .fillna(processed["fwi"])
                .values
            )
            processed["_prev_fwi"] = prev_fwi
            processed["fwi_lag1"] = prev_fwi

        risk_scores = self._compute_risk_scores(processed)
        alerts = self.alert_engine.generate_alerts(processed, risk_scores)

        n_rows = int(processed["row"].max()) + 1
        n_cols = int(processed["col"].max()) + 1

        self.current_snapshot = TwinSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            processed_grid=processed, risk_scores=risk_scores,
            n_rows=n_rows, n_cols=n_cols, alerts=alerts,
        )
        logger.info(
            "DigitalTwin refreshed: %d zones, %d alerts, max risk %.2f",
            len(processed), len(alerts), risk_scores.max() if len(risk_scores) else 0,
        )
        return self.current_snapshot

    def simulate_spread_from_alerts(self, horizon_minutes: Optional[int] = None) -> list:
        """
        Runs the CA simulator seeded from current HIGH/EXTREME alert zones
        (Scenario 1: 'system projects fire spread over the next 2 hours').
        Must call refresh() first. Returns [] (no seed zones) whenever the
        current risk state has nothing in HIGH/EXTREME - see
        simulate_spread_from_top_n() for a deliberate fallback that ignites
        regardless of severity, for a demo that always has something to show.
        """
        if self.current_snapshot is None:
            raise RuntimeError("Call refresh() before simulate_spread_from_alerts().")
        snap = self.current_snapshot
        zone_ids = {a.zone_id for a in snap.alerts if a.severity in ("HIGH", "EXTREME")}
        return self._simulate_spread(zone_ids, horizon_minutes)

    def simulate_spread_from_top_n(self, n: int = 5, horizon_minutes: Optional[int] = None) -> list:
        """
        Ignites the N highest-risk-score zones this cycle, REGARDLESS of
        whether they cross the HIGH/EXTREME severity threshold. Deliberate
        fallback for when today's real/synthetic conditions genuinely have
        no HIGH/EXTREME zones (e.g. monsoon-season live data, or a quiet
        offline draw) - lets the Spread Simulation page still produce a
        projection to look at instead of a dead end.
        """
        if self.current_snapshot is None:
            raise RuntimeError("Call refresh() before simulate_spread_from_top_n().")
        snap = self.current_snapshot
        if len(snap.risk_scores) == 0:
            return self._simulate_spread(set(), horizon_minutes)
        order = np.argsort(snap.risk_scores)[::-1][:n]
        zone_ids = set(snap.processed_grid.reset_index(drop=True).iloc[order]["zone_id"])
        return self._simulate_spread(zone_ids, horizon_minutes)

    def _simulate_spread(self, ignition_zone_ids: set, horizon_minutes: Optional[int] = None) -> list:
        snap = self.current_snapshot
        processed = snap.processed_grid
        horizon = horizon_minutes or SYSTEM.fire_spread_horizon_hours * 60

        ignition = np.zeros((snap.n_rows, snap.n_cols), dtype=bool)
        dryness = np.zeros((snap.n_rows, snap.n_cols))
        fuel = np.zeros((snap.n_rows, snap.n_cols))
        buildup = np.ones((snap.n_rows, snap.n_cols))
        non_fuel = np.zeros((snap.n_rows, snap.n_cols), dtype=bool)

        rows = processed["row"].values
        cols = processed["col"].values
        dryness[rows, cols] = (processed["ffmc"] / 101.0).clip(0, 1).values
        fuel[rows, cols] = processed["ndvi"].clip(0, 1).values
        non_fuel[rows, cols] = processed["ndvi"].values < 0.15
        # BUI (Buildup Index) -> fuel-availability multiplier. See
        # cellular_automata.grids_from_processed for the same normalisation,
        # kept consistent here since this method builds grids independently.
        if "bui" in processed.columns:
            buildup[rows, cols] = np.clip(processed["bui"].values / 30.0, 0.4, 2.0)
        is_ignition = processed["zone_id"].isin(ignition_zone_ids).values
        ignition[rows[is_ignition], cols[is_ignition]] = True

        if ignition.sum() == 0:
            logger.info("No zones to seed CA simulation - skipping.")
            snap.ca_history = []
            return []

        elevation = self._get_elevation_grid(processed, snap.n_rows, snap.n_cols)

        sim = FireSpreadSimulator(snap.n_rows, snap.n_cols)
        history = sim.run(
            ignition_mask=ignition, dryness_grid=dryness, fuel_load_grid=fuel,
            non_fuel_mask=non_fuel,
            wind_speed_ms=float(processed["wx_wind_speed_ms"].mean()),
            wind_from_deg=float(processed["wx_wind_deg"].mean()),
            horizon_minutes=horizon,
            elevation_grid=elevation, fuel_buildup_grid=buildup,
        )
        snap.ca_history = history
        return history

    def _get_elevation_grid(self, processed, n_rows: int, n_cols: int) -> np.ndarray:
        """
        Terrain is static, so elevation is fetched/generated once per twin
        instance and cached - no need to re-fetch on every refresh. Live
        fetch is only attempted when NOT in offline mode, consistent with
        every other data source in this project (FIRMS, weather).
        """
        if self._elevation_grid_cache is not None:
            return self._elevation_grid_cache

        from src.data_ingestion.elevation_client import get_elevation_grid
        elevations, is_real = get_elevation_grid(processed[["latitude", "longitude"]], use_live=not self.offline)
        logger.info("Elevation grid ready (%s)", "real API" if is_real else "synthetic terrain")

        grid = np.zeros((n_rows, n_cols))
        rows = processed["row"].values
        cols = processed["col"].values
        grid[rows, cols] = elevations
        self._elevation_grid_cache = grid
        return grid

    def get_summary(self) -> dict:
        """Compact dict for API responses / dashboard header cards."""
        if self.current_snapshot is None:
            return {"status": "not_initialised"}
        snap = self.current_snapshot
        severity_counts = {}
        for a in snap.alerts:
            severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1
        return {
            "status": "ok",
            "timestamp": snap.timestamp,
            "region": self.region.name,
            "total_zones": len(snap.processed_grid),
            "total_alerts": len(snap.alerts),
            "severity_breakdown": severity_counts,
            "max_risk_score": float(snap.risk_scores.max()) if len(snap.risk_scores) else 0.0,
            "mean_risk_score": float(snap.risk_scores.mean()) if len(snap.risk_scores) else 0.0,
        }


def persist_snapshot_and_notify(twin: "DigitalTwin", notifier=None) -> dict:
    """
    Shared hook used by BOTH the dashboard (on every manual/auto refresh)
    and the background scheduler (src/scheduler/scheduler.py), so "save to
    the database" and "email on a new EXTREME detection" happen exactly the
    same way no matter which one triggered the refresh.

    Must be called AFTER twin.refresh(). Returns a small dict describing
    what happened, useful for a dashboard toast/log line.

    - Reads which zones were EXTREME in the region's previous snapshot
      (i.e. before this refresh) from the database.
    - Saves this refresh as a new snapshot row + one alerts_log row per
      alert (cross-question 6/11: "where is the data being stored").
    - Diffs current EXTREME zones against the previous set. Only zones
      that are NEWLY extreme trigger an email - a zone that was already
      extreme last cycle does not re-notify every 15 minutes.
    - Never raises: a DB or SMTP failure must not break a dashboard page
      render or a scheduler cycle. Failures are logged and reflected in
      the returned dict instead.
    """
    result = {"persisted": False, "new_extreme_count": 0, "emailed": False}
    if twin.current_snapshot is None:
        return result

    try:
        from src.storage import database as db
    except Exception:
        logger.warning("Storage module unavailable - snapshot not persisted.")
        return result

    try:
        db.init_db()
    except Exception as exc:
        logger.warning("Could not initialise database (%s) - skipping persistence.", exc)
        return result

    region_name = twin.region.name
    snapshot = twin.current_snapshot
    summary = twin.get_summary()

    try:
        previously_extreme = db.get_previously_extreme_zone_ids(region_name)
    except Exception as exc:
        logger.warning("Could not read previous snapshot (%s) - treating as none.", exc)
        previously_extreme = set()

    try:
        db.save_snapshot(region_name, summary, snapshot.alerts, twin.offline)
        result["persisted"] = True
    except Exception as exc:
        logger.warning("Could not save snapshot (%s).", exc)
        return result

    currently_extreme = {a.zone_id: a for a in snapshot.alerts if a.severity == "EXTREME"}
    new_zone_ids = set(currently_extreme.keys()) - previously_extreme
    new_alerts = [currently_extreme[z] for z in new_zone_ids]
    result["new_extreme_count"] = len(new_alerts)

    if new_alerts:
        if notifier is None:
            from src.notifications.email_notifier import EmailNotifier
            notifier = EmailNotifier()
        try:
            result["emailed"] = notifier.send_new_extreme_alert(region_name, new_alerts)
        except Exception as exc:
            logger.warning("Email notification raised unexpectedly (%s) - ignored.", exc)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from config.config import API
    use_offline = not bool(API.firms_map_key)
    if use_offline:
        logger.info("No FIRMS_MAP_KEY found - running in offline/synthetic mode.")
    else:
        logger.info("FIRMS_MAP_KEY found - fetching real satellite data.")

    twin = DigitalTwin(offline=use_offline)
    snapshot = twin.refresh()

    print("\n=== TWIN SUMMARY ===")
    for k, v in twin.get_summary().items():
        print(f"{k}: {v}")

    print("\n=== TOP 5 ALERTS ===")
    for a in snapshot.alerts[:5]:
        print(f"{a.zone_id}  [{a.severity}]  risk={a.risk_score:.2f}  {a.reason}")

    print("\n=== RUNNING CA SIMULATION FROM HIGH-RISK ZONES ===")
    history = twin.simulate_spread_from_alerts()
    if history:
        final = history[-1]
        print(f"Final step: {final.minutes_elapsed} min, burned={final.n_burned}, burning={final.n_burning}")
    else:
        print("No high-risk zones to simulate from this cycle.")
