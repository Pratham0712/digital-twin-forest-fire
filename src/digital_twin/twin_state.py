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
        (0.90, "EXTREME"),
        (0.70, "HIGH"),
        (0.40, "MODERATE"),
        (0.0, "LOW"),
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

    def __init__(self, ml_model=None, offline: bool = True):
        """
        ml_model: any fitted model exposing .predict(X) -> (labels, probs),
        e.g. model_trainer.RandomForestModel or XGBoostModel. If None, the
        twin falls back to using FWI-normalised risk (still principled, just
        not learned) so the twin is usable before model training is run.
        """
        self.offline = offline
        self.ingestion = DataIngestionModule(offline=offline)
        self.processor = DataProcessor()
        self.alert_engine = AlertEngine()
        self.ml_model = ml_model
        self.current_snapshot: Optional[TwinSnapshot] = None

    def _compute_risk_scores(self, processed: pd.DataFrame) -> np.ndarray:
        if self.ml_model is not None:
            X, _ = self.processor.get_feature_matrix(processed)
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
        Must call refresh() first.
        """
        if self.current_snapshot is None:
            raise RuntimeError("Call refresh() before simulate_spread_from_alerts().")

        snap = self.current_snapshot
        processed = snap.processed_grid
        horizon = horizon_minutes or SYSTEM.fire_spread_horizon_hours * 60

        high_risk_zone_ids = {a.zone_id for a in snap.alerts if a.severity in ("HIGH", "EXTREME")}
        ignition = np.zeros((snap.n_rows, snap.n_cols), dtype=bool)
        dryness = np.zeros((snap.n_rows, snap.n_cols))
        fuel = np.zeros((snap.n_rows, snap.n_cols))
        non_fuel = np.zeros((snap.n_rows, snap.n_cols), dtype=bool)

        rows = processed["row"].values
        cols = processed["col"].values
        dryness[rows, cols] = (processed["ffmc"] / 101.0).clip(0, 1).values
        fuel[rows, cols] = processed["ndvi"].clip(0, 1).values
        non_fuel[rows, cols] = processed["ndvi"].values < 0.15
        is_high_risk = processed["zone_id"].isin(high_risk_zone_ids).values
        ignition[rows[is_high_risk], cols[is_high_risk]] = True

        if ignition.sum() == 0:
            logger.info("No HIGH/EXTREME zones to seed CA simulation - skipping.")
            snap.ca_history = []
            return []

        sim = FireSpreadSimulator(snap.n_rows, snap.n_cols)
        history = sim.run(
            ignition_mask=ignition, dryness_grid=dryness, fuel_load_grid=fuel,
            non_fuel_mask=non_fuel,
            wind_speed_ms=float(processed["wx_wind_speed_ms"].mean()),
            wind_from_deg=float(processed["wx_wind_deg"].mean()),
            horizon_minutes=horizon,
        )
        snap.ca_history = history
        return history

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
            "region": REGION.name,
            "total_zones": len(snap.processed_grid),
            "total_alerts": len(snap.alerts),
            "severity_breakdown": severity_counts,
            "max_risk_score": float(snap.risk_scores.max()) if len(snap.risk_scores) else 0.0,
            "mean_risk_score": float(snap.risk_scores.mean()) if len(snap.risk_scores) else 0.0,
        }


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
