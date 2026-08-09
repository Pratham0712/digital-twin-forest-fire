"""
main.py - single entry point that runs the full Digital Twin pipeline once,
end-to-end, and prints a summary. Useful for demos, viva, and as the
reference implementation of Scenario 1 in the SRS.

Run:
    python main.py              # offline/demo mode if no API keys
    python main.py --train      # also (re)trains the ML models first
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config.config import API, REGION, MODELS_DIR
from src.digital_twin.twin_state import DigitalTwin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trained_model():
    model_path = MODELS_DIR / "xgboost.json"
    if not model_path.exists():
        logger.warning("No trained model found at %s - twin will use FWI-based risk fallback. "
                        "Run 'python src/ml_models/train.py' first for ML-based predictions.", model_path)
        return None
    from src.ml_models.model_trainer import XGBoostModel
    wrapper = XGBoostModel()
    wrapper.model.load_model(str(model_path))
    logger.info("Loaded trained XGBoost model from %s", model_path)
    return wrapper


def run_pipeline(retrain: bool = False):
    print("=" * 70)
    print("  DIGITAL TWIN FRAMEWORK FOR FOREST FIRE PREDICTION")
    print(f"  Region: {REGION.name}")
    print("=" * 70)

    if retrain:
        logger.info("Retraining models before running the pipeline...")
        from src.ml_models.train import main as train_main
        train_main(offline=not bool(API.firms_map_key))

    model = load_trained_model()
    use_offline = not bool(API.firms_map_key)
    logger.info("Mode: %s", "OFFLINE (synthetic data)" if use_offline else "LIVE (real FIRMS/OWM data)")

    twin = DigitalTwin(ml_model=model, offline=use_offline)

    logger.info("Step 1-3: Ingesting data, engineering features, scoring risk...")
    snapshot = twin.refresh()

    print(f"\n{'='*70}\n  TWIN STATE SUMMARY\n{'='*70}")
    for k, v in twin.get_summary().items():
        print(f"  {k}: {v}")

    print(f"\n{'='*70}\n  TOP 10 ALERTS\n{'='*70}")
    if snapshot.alerts:
        for a in snapshot.alerts[:10]:
            print(f"  [{a.severity:>8}]  {a.zone_id:<8}  risk={a.risk_score:.2f}  {a.reason}")
    else:
        print("  No zones currently above the alert threshold.")

    logger.info("Step 4: Running Cellular Automata spread simulation from HIGH/EXTREME zones...")
    history = twin.simulate_spread_from_alerts()

    print(f"\n{'='*70}\n  FIRE SPREAD PROJECTION\n{'='*70}")
    if history:
        final = history[-1]
        print(f"  Horizon: {final.minutes_elapsed} minutes")
        print(f"  Cells burning: {final.n_burning}")
        print(f"  Cells burned: {final.n_burned}")
    else:
        print("  No HIGH/EXTREME zones this cycle - nothing simulated.")

    print(f"\n{'='*70}")
    print("  Pipeline run complete. Launch the full interactive view with:")
    print("  streamlit run src/dashboard/app.py")
    print(f"{'='*70}\n")

    return twin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Digital Twin pipeline end-to-end.")
    parser.add_argument("--train", action="store_true", help="Retrain ML models before running")
    args = parser.parse_args()
    run_pipeline(retrain=args.train)
