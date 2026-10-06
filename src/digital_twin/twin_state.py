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

from config.config import API, SYSTEM, REGION, MODELS_DIR
from src.data_ingestion.ingestion_module import DataIngestionModule
from src.data_ingestion.terrain import TERRAIN_COLUMNS, TerrainTable
from src.data_ingestion.wind import interpolate_schedule, mean_wind
from src.data_processing.feature_engineering import DataProcessor
from src.ml_models.fire_history import ZoneClimatology, live_fire_history_features
from src.simulation.cellular_automata import FireSpreadSimulator, CellState

logger = logging.getLogger(__name__)

# Terrain never changes, so the elevation grid is shared by every twin in the
# process (a fresh twin per region switch or session no longer re-reads it or,
# for regions without a terrain table, re-calls the elevation API).
_ELEVATION_CACHE: dict = {}


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
    # Wind that drove the latest spread simulation:
    # {"source": "forecast" | "current", "speed_ms", "from_deg", "n_points"}
    ca_wind: Optional[dict] = None


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
                 scenario: Optional[dict] = None, region=None,
                 climatology_file: Optional[str] = None):
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
        from src.ml_models.model_registry import choose_for_region
        self.model_choice = choose_for_region(self.region)
        self.climatology = ZoneClimatology.load(
            MODELS_DIR, climatology_file or self.model_choice.climatology_file)
        from src.ml_models.risk_index import RiskIndex
        self.risk_index = RiskIndex.load(MODELS_DIR, self.model_choice.risk_index_file)
        self.last_timings: dict = {}
        self.terrain = TerrainTable.load(MODELS_DIR, self.region.name)
        self._forecast_cache: dict = {}

    def _model_columns(self) -> Optional[list]:
        """Feature names the loaded model was trained with (so an older
        weather-only model file keeps working with the newer pipeline)."""
        try:
            names = self.ml_model.model.get_booster().feature_names
            return list(names) if names else None
        except Exception:
            return None

    def _add_model_v2_features(self, processed: pd.DataFrame) -> pd.DataFrame:
        """Satellite fire history (previous 10 days) + zone climatology -
        computed by the same code used to build the training set."""
        hist = live_fire_history_features(
            processed, getattr(self.ingestion, "last_hotspots", None),
            self.region.grid_resolution_deg)
        clim = self.climatology.transform(processed)
        terrain = self.terrain.transform(processed)   # NaN columns if no terrain table was built
        processed = processed.drop(columns=[c for c in hist.columns if c != "zone_id"] +
                                   ["zone_clim", "nbr_clim"] + TERRAIN_COLUMNS, errors="ignore")
        return (processed.merge(hist, on="zone_id", how="left")
                         .merge(clim, on="zone_id", how="left")
                         .merge(terrain, on="zone_id", how="left"))

    def _compute_risk_scores(self, processed: pd.DataFrame) -> np.ndarray:
        if self.ml_model is not None:
            X, _ = self.processor.get_feature_matrix(processed, real=self.use_real_model,
                                                     columns=self._model_columns())
            _, probs = self.ml_model.predict(X)
            processed["fire_probability"] = probs
            # Dashboard risk = validated risk index (see risk_index.py); the
            # raw calibrated probability stays available as fire_probability.
            return self.risk_index.transform(probs) if self.risk_index is not None else probs
        # Fallback: normalise FWI into a pseudo-probability so the twin is
        # still functional before/without a trained model.
        fwi = processed["fwi"].values
        return np.clip(fwi / 100.0, 0, 1)

    def refresh(self) -> TwinSnapshot:
        """Full pipeline refresh - call this on the SYSTEM.min_refresh_interval_minutes cadence."""
        import time as _time
        t0 = _time.perf_counter()
        unified = self.ingestion.build_unified_frame()
        t_ingest = _time.perf_counter()
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

        processed = self._add_model_v2_features(processed)
        t_features = _time.perf_counter()

        risk_scores = self._compute_risk_scores(processed)
        alerts = self.alert_engine.generate_alerts(processed, risk_scores)
        t_model = _time.perf_counter()
        # Per-stage latency, surfaced in the activity log and the Admin page.
        self.last_timings = {
            "ingest_s": t_ingest - t0, "features_s": t_features - t_ingest,
            "model_s": t_model - t_features, "total_s": t_model - t0,
        }

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
        schedule, wind = self._spread_wind(processed, is_ignition, horizon, sim.minutes_per_step)
        snap.ca_wind = wind
        history = sim.run(
            ignition_mask=ignition, dryness_grid=dryness, fuel_load_grid=fuel,
            non_fuel_mask=non_fuel,
            wind_speed_ms=wind["speed_ms"], wind_from_deg=wind["from_deg"],
            horizon_minutes=horizon,
            elevation_grid=elevation, fuel_buildup_grid=buildup,
            wind_schedule=schedule,
        )
        snap.ca_history = history
        return history

    def _spread_wind(self, processed: pd.DataFrame, is_ignition: np.ndarray, horizon_min: int,
                     step_min: int):
        """Wind for the spread simulation. Uses the OpenWeatherMap FORECAST
        near the ignition zones (a per-step schedule over the horizon) when a
        key is configured and the twin is live; otherwise the current wind at
        the ignition zones. Direction is always averaged as a vector."""
        near = processed.loc[is_ignition]
        cur_speed, cur_deg = mean_wind(near["wx_wind_speed_ms"].values, near["wx_wind_deg"].values)
        current = {"source": "current", "speed_ms": cur_speed, "from_deg": cur_deg, "n_points": 0}

        if self.offline or not API.owm_api_key:
            return None, current
        try:
            slots_by_point = self._forecast_slots(near, horizon_min)
        except Exception as exc:                       # never let the forecast break the simulation
            logger.warning("Forecast wind unavailable (%s); using current wind.", exc)
            return None, current
        if not slots_by_point:
            logger.warning("Forecast wind unavailable; using current wind.")
            return None, current

        import time as _time
        now_s = _time.time()
        per_point = [interpolate_schedule([x["dt"] for x in sl], [x["wind_speed_ms"] for x in sl],
                                          [x["wind_deg"] for x in sl], now_s, horizon_min, step_min)
                     for sl in slots_by_point]
        schedule = [mean_wind([p[k][0] for p in per_point], [p[k][1] for p in per_point])
                    for k in range(len(per_point[0]))]
        speed, deg = mean_wind([w[0] for w in schedule], [w[1] for w in schedule])
        logger.info("Spread driven by forecast wind from %d point(s): %.1f m/s from %.0f deg",
                    len(per_point), speed, deg)
        return schedule, {"source": "forecast", "speed_ms": speed, "from_deg": deg,
                          "n_points": len(per_point)}

    def _forecast_slots(self, near: pd.DataFrame, horizon_min: int, max_points: int = 4,
                        ttl_s: int = 1800) -> list:
        """Forecast for the (at most `max_points`) weather-grid points nearest
        to the ignition zones - a handful of calls, only when a simulation is
        run, cached for `ttl_s` so repeated runs don't spend API quota."""
        import time as _time
        from scipy.spatial import cKDTree
        wg = self.ingestion.weather_grid.reset_index(drop=True)
        tree = cKDTree(wg[["latitude", "longitude"]].values)
        _, idx = tree.query(near[["latitude", "longitude"]].values, k=1)
        counts = pd.Series(idx).value_counts()
        chosen = wg.iloc[counts.index[:max_points]]
        key = tuple(sorted(zip(chosen["latitude"].round(3), chosen["longitude"].round(3)))) + (horizon_min,)
        hit = self._forecast_cache.get(key)
        if hit and _time.time() - hit[0] < ttl_s:
            return hit[1]
        slots = self.ingestion.weather.fetch_forecast_wind(
            chosen[["latitude", "longitude"]].to_dict("records"), horizon_hours=horizon_min / 60.0)
        if slots:
            self._forecast_cache[key] = (_time.time(), slots)
        return slots

    def _get_elevation_grid(self, processed, n_rows: int, n_cols: int) -> np.ndarray:
        """
        Terrain is static, so elevation is fetched/generated once per twin
        instance and cached - no need to re-fetch on every refresh. Live
        fetch is only attempted when NOT in offline mode, consistent with
        every other data source in this project (FIRMS, weather).
        """
        if self._elevation_grid_cache is not None:
            return self._elevation_grid_cache
        ekey = (self.region.name, n_rows, n_cols)
        if ekey in _ELEVATION_CACHE:              # shared by every twin in this process
            self._elevation_grid_cache = _ELEVATION_CACHE[ekey]
            return self._elevation_grid_cache

        # Real terrain built by scripts/build_terrain.py (also feeds the model).
        if "elev_m" in processed.columns and processed["elev_m"].notna().all():
            grid = np.zeros((n_rows, n_cols))
            grid[processed["row"].values, processed["col"].values] = processed["elev_m"].values
            self._elevation_grid_cache = _ELEVATION_CACHE[ekey] = grid
            logger.info("Elevation grid ready (real terrain table)")
            return grid

        from src.data_ingestion.elevation_client import get_elevation_grid
        elevations, is_real = get_elevation_grid(processed[["latitude", "longitude"]], use_live=not self.offline)
        logger.info("Elevation grid ready (%s)", "real API" if is_real else "synthetic terrain")

        grid = np.zeros((n_rows, n_cols))
        rows = processed["row"].values
        cols = processed["col"].values
        grid[rows, cols] = elevations
        self._elevation_grid_cache = grid
        if is_real:                                # never pin a synthetic fallback for the whole process
            _ELEVATION_CACHE[ekey] = grid
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


def persist_snapshot_and_notify(twin: "DigitalTwin", notifier=None,
                                actor: str = "system", trigger: str = "") -> dict:
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
        db.log_activity("error", f"Snapshot could not be saved: {exc}", actor, region_name)
        return result

    t = twin.last_timings or {}
    sev = summary.get("severity_breakdown", {})
    detail = (f"{summary.get('total_zones', 0):,} zones scored · {len(snapshot.alerts)} alerts "
              f"({sev.get('EXTREME', 0)} extreme, {sev.get('HIGH', 0)} high) · "
              f"peak risk {summary.get('max_risk_score', 0):.0%} · "
              f"{'offline' if twin.offline else 'live'} data")
    if t:
        detail += (f" · {t['total_s']:.1f}s (data {t['ingest_s']:.1f}s, "
                   f"features {t['features_s']:.2f}s, model {t['model_s']:.2f}s)")
    if trigger:
        detail = f"{trigger} — {detail}"
    db.log_activity("refresh", detail, actor, region_name)

    currently_extreme = {a.zone_id: a for a in snapshot.alerts if a.severity == "EXTREME"}
    new_zone_ids = set(currently_extreme.keys()) - previously_extreme
    new_alerts = [currently_extreme[z] for z in new_zone_ids]
    result["new_extreme_count"] = len(new_alerts)

    if new_alerts:
        top = sorted(new_alerts, key=lambda a: a.risk_score, reverse=True)
        names = ", ".join(f"{a.zone_id} ({a.risk_score:.0%})" for a in top[:5])
        more = f" and {len(top) - 5} more" if len(top) > 5 else ""
        db.log_activity("new_extreme", f"{len(top)} zone(s) newly EXTREME: {names}{more}",
                        actor, region_name)
        if notifier is None:
            from src.notifications.email_notifier import EmailNotifier
            notifier = EmailNotifier()
        try:
            result["emailed"] = notifier.send_new_extreme_alert(region_name, new_alerts)
        except Exception as exc:
            logger.warning("Email notification raised unexpectedly (%s) - ignored.", exc)
        if getattr(notifier, "is_configured", False):
            if result["emailed"]:
                db.log_activity("email_sent", f"Alert email for {len(top)} zone(s) sent to "
                                f"{len(notifier.recipients)} recipient(s)", actor, region_name)
            else:
                db.log_activity("email_failed", "Alert email could not be sent", actor, region_name)

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
