"""
demo_dynamic_predictions.py - shows that changing the prediction inputs changes the
model output and every dependent dashboard value (no hard-coded percentages).

Runs the real pipeline (DigitalTwin -> shared prediction service -> risk index ->
alerts / summary / zone conditions) offline on three What-If weather inputs at the
Bandipur preset. No network, no API quota.

    python scripts/demo_dynamic_predictions.py
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)

from src.dashboard.live_modes import zone_risk                       # noqa: E402
from src.digital_twin.twin_state import DigitalTwin                 # noqa: E402
from src.ml_models.model_registry import load_xgb                   # noqa: E402

LAT, LON = 11.6560, 76.5800
SCENARIOS = {"cool, humid, rain": (22, 90, 1, 5.0), "mild": (28, 60, 3, 0.0), "hot, dry, windy": (40, 12, 12, 0.0)}


def main():
    model = load_xgb("xgboost_real.json")
    print(f"{'inputs (T °C, RH %, wind m/s, rain mm)':40s} {'FFMC':>6s} {'FWI':>6s} {'model P(fire)':>13s} "
          f"{'risk index':>10s} {'category':>9s} {'region peak':>11s} {'region mean':>11s} {'alerts':>6s}  model")
    for name, (t, h, w, r) in SCENARIOS.items():
        pw = {"temperature_c": t, "humidity_pct": h, "wind_speed_ms": w, "wind_deg": 225.0, "pressure_hpa": 1013.0,
              "precipitation_mm": r, "clouds_pct": 0, "weather_main": "Clear"}
        twin = DigitalTwin(ml_model=model, offline=True, scenario={"point_weather": pw, "weather_mode": "scenario"})
        twin.refresh()
        z, s = zone_risk(twin, LAT, LON), twin.get_summary()
        print(f"{name + f' ({t}, {h}, {w}, {r})':40s} {z['ffmc']:6.1f} {z['fwi']:6.1f} {z['fire_probability']:13.4%} "
              f"{z['risk_score']:10.1%} {z['severity']:>9s} {s['max_risk_score']:11.1%} {s['mean_risk_score']:11.1%} "
              f"{s['total_alerts']:6d}  {s['model_version']} @ {s['predicted_utc'][:19]}Z")


if __name__ == "__main__":
    main()
