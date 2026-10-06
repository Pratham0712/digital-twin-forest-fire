"""
model_registry.py - which trained model serves which region.

  * Karnataka / Western Ghats (0.1 deg grid): models/xgboost_real.json
    + models/zone_climatology.csv         (src/ml_models/train_real.py)
  * the other curated Indian states (0.5 deg grid): models/xgboost_india.json
    + models/zone_climatology_india.csv   (scripts/build_india_regions_dataset.py --train)

If the India model hasn't been trained yet, other regions fall back to the
Karnataka model, exactly as before.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.config import MODELS_DIR, REGION, DATA_PROCESSED_DIR

KARNATAKA_MODEL = "xgboost_real.json"
KARNATAKA_CLIMATOLOGY = "zone_climatology.csv"
INDIA_MODEL = "xgboost_india.json"
INDIA_CLIMATOLOGY = "zone_climatology_india.csv"
INDIA_METRICS = "india_model_by_state.csv"
KARNATAKA_RISK_INDEX = "risk_index.json"
INDIA_RISK_INDEX = "risk_index_india.json"


@dataclass
class ModelChoice:
    model_file: str
    climatology_file: str
    risk_index_file: str
    label: str            # short description for the UI


def choose_for_region(region, models_dir: Path = MODELS_DIR) -> ModelChoice:
    if region is None or region.name == REGION.name:
        return ModelChoice(KARNATAKA_MODEL, KARNATAKA_CLIMATOLOGY, KARNATAKA_RISK_INDEX,
                           "Karnataka model (0.1°)")
    if (Path(models_dir) / INDIA_MODEL).exists():
        return ModelChoice(INDIA_MODEL, INDIA_CLIMATOLOGY, INDIA_RISK_INDEX, "Pan-India model (0.5°)")
    return ModelChoice(KARNATAKA_MODEL, KARNATAKA_CLIMATOLOGY, KARNATAKA_RISK_INDEX,
                       "Karnataka model (not trained on this region)")


def load_xgb(model_file: str, models_dir: Path = MODELS_DIR):
    path = Path(models_dir) / model_file
    if not path.exists():
        return None
    from src.ml_models.model_trainer import XGBoostModel
    w = XGBoostModel()
    w.model.load_model(str(path))
    return w


def india_state_metrics(region_name: str) -> Optional[dict]:
    """Held-out-2025 metrics for one state from the India model's evaluation,
    or None if that model hasn't been trained."""
    path = Path(DATA_PROCESSED_DIR) / INDIA_METRICS
    if not path.exists():
        return None
    import pandas as pd
    df = pd.read_csv(path)
    row = df[df["region"] == region_name]
    return None if row.empty else row.iloc[0].to_dict()
