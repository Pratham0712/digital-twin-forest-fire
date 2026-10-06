"""
run_evaluation_extras.py - lift, precision-recall, persistence baseline and
grid-comparable metrics for the SAVED Karnataka model on each held-out season
(2025, and 2026 when that season has been added with scripts/add_season.py).

    python scripts/run_evaluation_extras.py

Writes data/processed/eval_*.csv, which the Model Insights page displays.
No retraining: it scores the model already saved in models/xgboost_real.json.
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config.config import DATA_PROCESSED_DIR, MODELS_DIR, REGION  # noqa: E402
from src.ml_models.evaluation_extras import season_report  # noqa: E402
from src.ml_models.model_registry import KARNATAKA_MODEL, load_xgb  # noqa: E402
from src.ml_models.train_real import (  # noqa: E402
    SECOND_TEST_START, TEST_SEASON_START, add_terrain_training, load_real_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    wrapper = load_xgb(KARNATAKA_MODEL)
    if wrapper is None:
        sys.exit(f"{KARNATAKA_MODEL} not found - run python src/ml_models/train_real.py first.")
    cols = list(wrapper.model.get_booster().feature_names)

    df = load_real_dataset()
    if "elev_m" in cols:
        df = add_terrain_training(df)
    seasons = {
        "2025": df[(df["date"] >= TEST_SEASON_START) & (df["date"] < SECOND_TEST_START)],
        "2026": df[df["date"] >= SECOND_TEST_START],
    }
    meta_path = MODELS_DIR / "training_metadata_real.json"
    alert_thr = json.loads(meta_path.read_text()).get("xgboost_alert_threshold") if meta_path.exists() else None

    parts = {k: [] for k in ("lift", "pr", "persistence", "comparable", "summary")}
    for season, test_df in seasons.items():
        if test_df.empty:
            logger.info("Season %s: no data - skipped (add it with scripts/add_season.py)", season)
            continue
        _, prob = wrapper.predict(test_df[cols])
        logger.info("Season %s: %d zone-days, %.2f%% fire days", season, len(test_df),
                    100 * test_df["fire_risk_label"].mean())
        rep = season_report(test_df, prob, season, REGION.name, alert_thr)
        for k, v in rep.items():
            parts[k].append(v)

    names = {"lift": "eval_lift_table.csv", "pr": "eval_pr_curve.csv", "persistence": "eval_persistence.csv",
             "comparable": "eval_comparable.csv", "summary": "eval_summary.csv"}
    for k, fname in names.items():
        frames = [f for f in parts[k] if len(f)]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(DATA_PROCESSED_DIR / fname, index=False)
            logger.info("Wrote %s", DATA_PROCESSED_DIR / fname)

    if parts["summary"]:
        print("\n=== Held-out summary (base-rate normalised) ===")
        print(pd.concat(parts["summary"]).to_string(index=False))
        print("\n=== Lift: flag the riskiest X% of zone-days ===")
        print(pd.concat(parts["lift"]).drop(columns=["region"]).to_string(index=False))
    if parts["persistence"]:
        print("\n=== Model vs simple 'burned recently' rules (same flagged share) ===")
        print(pd.concat(parts["persistence"]).drop(columns=["region"]).to_string(index=False))


if __name__ == "__main__":
    main()
