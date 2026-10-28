"""
model_selection.py - controlled, reproducible evaluation / retraining / promotion
of the fire-occurrence model that serves the dashboard (Karnataka model).

Nothing here runs from the dashboard. A new model is only ever produced by an
operator running this script on the validated training dataset, and it only
replaces the serving model when it passes the promotion gate on the held-out
season. The serving model is never retrained on live inputs.

    python -m src.ml_models.model_selection evaluate     # baseline vs candidate(s), writes the report
    python -m src.ml_models.model_selection promote      # evaluate + promote ONLY if the gate passes
    python -m src.ml_models.model_selection rollback --to v3.0
    python -m src.ml_models.model_selection status       # active version + history

Validation protocol (leakage-safe)
  * hold-out test = the whole 2025 season, never used for fitting, tuning or
    threshold selection (same split as train_real.py).
  * candidate hyper-parameters and decision thresholds are chosen with
    SEASON-WISE cross-validation on 2023/2024 only (fit on one season, predict
    the other), i.e. on out-of-season predictions. The original pipeline picked
    thresholds on a random 15 % split of the same seasons, which mixes days of
    the same fire season into fit and validation.
  * feature engineering is the shared code in fire_history.py / train_real.py
    (climatology leave-one-season-out on training rows, fire history t-10..t-1).

Promotion gate (all must hold on the held-out season, against the serving model):
  PR-AUC  >= baseline + GATE_MIN_PR_AUC_GAIN      (ranking of fire days: primary metric, rare event)
  ROC-AUC >= baseline - GATE_MAX_ROC_AUC_LOSS
  log loss <= baseline                            (probabilities no worse)
  early-warning recall >= baseline - GATE_MAX_RECALL_LOSS   (safety: do not miss more fires)
Otherwise the candidate is REJECTED and the serving model stays active.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from config.config import DATA_PROCESSED_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

SEED = 42
TEST_START, TEST_END = "2025-01-01", "2026-01-01"
LABEL = "fire_risk_label"
SERVING_MODEL = "xgboost_real.json"
SERVING_META = "training_metadata_real.json"
SERVING_INDEX = "risk_index.json"
REGISTRY_FILE = "model_registry.json"
VERSIONS_DIR = "versions"
REPORT_JSON = "model_evaluation_report.json"
REPORT_CSV = "model_evaluation_metrics.csv"

GATE_MIN_PR_AUC_GAIN = 0.005
GATE_MAX_ROC_AUC_LOSS = 0.002
GATE_MAX_RECALL_LOSS = 0.01

# Candidate configurations. Chosen by season-wise CV on 2023/2024 (see module doc):
# a shallower, regularised booster generalises better to an unseen season than the
# original random-split-tuned model, and dropping raw latitude/longitude (zone
# identity is already carried by the leave-one-season-out climatology) gave the
# best out-of-season PR-AUC.
CANDIDATES = {
    "xgb_regularised_no_latlon": {
        "drop": ["latitude", "longitude"],
        "params": dict(n_estimators=200, max_depth=4, learning_rate=0.03, min_child_weight=20, subsample=0.8,
                       colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist", random_state=SEED),
    },
    "xgb_regularised_all_features": {
        "drop": [],
        "params": dict(n_estimators=200, max_depth=4, learning_rate=0.03, min_child_weight=20, subsample=0.8,
                       colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist", random_state=SEED),
    },
}


# ── metrics ───────────────────────────────────────────────────────────────── #

def metrics_at(y, p, threshold: float) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)
    y = np.asarray(y).astype(int)
    yp = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=[0, 1], zero_division=0)
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, yp)),
        "precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1[1]),
        "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, yp, average="weighted", zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class": {"no_fire": {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f1[0]), "support": int(sup[0])},
                      "fire": {"precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1[1]), "support": int(sup[1])}},
    }


def calibration_table(y, p, n_bins: int = 10) -> List[dict]:
    """Reliability by predicted-probability decile: mean predicted vs observed fire rate."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": float(p[m].mean()),
                         "observed_rate": float(y[m].mean())})
    return rows


def full_metrics(y, p, t_early: float, t_alert: float) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    cal = calibration_table(y, p)
    ece = float(sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in cal) / max(len(y), 1))
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)),
        "expected_calibration_error": ece, "calibration": cal,
        "early_warning": metrics_at(y, p, t_early),
        "alert": metrics_at(y, p, t_alert),
    }


# ── data / thresholds ─────────────────────────────────────────────────────── #

def load_features() -> pd.DataFrame:
    """The real dataset with every engineered feature (same code as train_real.py)."""
    from src.ml_models import train_real as tr
    csv = DATA_PROCESSED_DIR / "real_training_data.csv"
    gz = DATA_PROCESSED_DIR / "real_training_data.csv.gz"
    if not csv.exists() and gz.exists():
        import gzip
        with gzip.open(gz, "rb") as fi, open(csv, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    df = tr.load_real_dataset()
    return tr.add_terrain_training(df)


def split(df: pd.DataFrame):
    train = df[df["date"] < TEST_START]
    test = df[(df["date"] >= TEST_START) & (df["date"] < TEST_END)]
    return train, test


def thresholds_from(y, p):
    """Same two operating-point rules as model_trainer (>= 80 % recall; >= 92 % accuracy),
    applied to OUT-OF-SEASON predictions, on a finer grid."""
    from src.ml_models.model_trainer import _pick_accuracy_threshold, _pick_best_threshold
    return (float(_pick_best_threshold(y, p, grid=np.arange(0.005, 0.5, 0.0025))),
            float(_pick_accuracy_threshold(y, p)))


def season_oof(train: pd.DataFrame, cols: List[str], params: dict) -> np.ndarray:
    import xgboost as xgb
    seasons = train["date"].dt.year.values
    oof = np.full(len(train), np.nan)
    for s in np.unique(seasons):
        fit, val = seasons != s, seasons == s
        m = xgb.XGBClassifier(**params).fit(train.loc[fit, cols], train.loc[fit, LABEL])
        oof[val] = m.predict_proba(train.loc[val, cols])[:, 1]
    return oof


# ── registry / versions ───────────────────────────────────────────────────── #

def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_registry(models_dir: Path = MODELS_DIR) -> dict:
    p = Path(models_dir) / REGISTRY_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {"active": None, "versions": []}


def save_registry(reg: dict, models_dir: Path = MODELS_DIR):
    (Path(models_dir) / REGISTRY_FILE).write_text(json.dumps(reg, indent=2))


def ensure_baseline_registered(models_dir: Path = MODELS_DIR) -> dict:
    """Registers the currently serving model as a version (copy of its three files)
    so it can always be rolled back to."""
    models_dir = Path(models_dir)
    reg = load_registry(models_dir)
    sha = _sha(models_dir / SERVING_MODEL)
    for v in reg["versions"]:
        if v["sha256"] == sha:
            if reg["active"] is None:
                reg["active"] = v["version"]
                save_registry(reg, models_dir)
            return reg
    meta = json.loads((models_dir / SERVING_META).read_text())
    version = f"v{meta.get('feature_set_version', 0)}.{len(reg['versions'])}"
    vdir = models_dir / VERSIONS_DIR / version
    vdir.mkdir(parents=True, exist_ok=True)
    for f in (SERVING_MODEL, SERVING_META, SERVING_INDEX):
        if (models_dir / f).exists():
            shutil.copy2(models_dir / f, vdir / f)
    reg["versions"].append({"version": version, "sha256": sha, "registered_utc": datetime.now(timezone.utc).isoformat(),
                            "source": "serving model at registration", "feature_columns": meta.get("feature_columns")})
    reg["active"] = reg["active"] or version
    save_registry(reg, models_dir)
    return reg


def activate(version: str, models_dir: Path = MODELS_DIR) -> dict:
    """Copies a registered version's files into the serving slot (promotion or rollback)."""
    models_dir = Path(models_dir)
    reg = load_registry(models_dir)
    if version not in {v["version"] for v in reg["versions"]}:
        raise ValueError(f"unknown model version {version!r}; registered: {[v['version'] for v in reg['versions']]}")
    vdir = models_dir / VERSIONS_DIR / version
    for f in (SERVING_MODEL, SERVING_META, SERVING_INDEX):
        if (vdir / f).exists():
            shutil.copy2(vdir / f, models_dir / f)
    reg["active"] = version
    reg.setdefault("history", []).append({"activated": version, "utc": datetime.now(timezone.utc).isoformat()})
    save_registry(reg, models_dir)
    return reg


# ── gate ──────────────────────────────────────────────────────────────────── #

def gate(baseline: dict, candidate: dict) -> dict:
    checks = {
        "pr_auc_gain": (candidate["pr_auc"] - baseline["pr_auc"], candidate["pr_auc"] >= baseline["pr_auc"] + GATE_MIN_PR_AUC_GAIN),
        "roc_auc_change": (candidate["roc_auc"] - baseline["roc_auc"],
                           candidate["roc_auc"] >= baseline["roc_auc"] - GATE_MAX_ROC_AUC_LOSS),
        "log_loss_change": (candidate["log_loss"] - baseline["log_loss"], candidate["log_loss"] <= baseline["log_loss"]),
        "early_warning_recall_change": (candidate["early_warning"]["recall"] - baseline["early_warning"]["recall"],
                                        candidate["early_warning"]["recall"]
                                        >= baseline["early_warning"]["recall"] - GATE_MAX_RECALL_LOSS),
    }
    return {"passed": all(ok for _, ok in checks.values()),
            "checks": {k: {"delta": float(d), "ok": bool(ok)} for k, (d, ok) in checks.items()}}


# ── main workflow ─────────────────────────────────────────────────────────── #

def evaluate(df: Optional[pd.DataFrame] = None, candidates: Dict[str, dict] = None,
             models_dir: Path = MODELS_DIR, out_dir: Path = DATA_PROCESSED_DIR, n_jobs: int = -1) -> dict:
    import xgboost as xgb
    from src.ml_models.risk_index import RiskIndex
    candidates = candidates or CANDIDATES
    models_dir, out_dir = Path(models_dir), Path(out_dir)
    df = load_features() if df is None else df
    train, test = split(df)
    y_te = test[LABEL].values
    meta = json.loads((models_dir / SERVING_META).read_text())
    base_cols = list(meta["feature_columns"])

    serving = xgb.XGBClassifier()
    serving.load_model(str(models_dir / SERVING_MODEL))
    p_base = serving.predict_proba(test[base_cols])[:, 1]
    baseline = {"name": "serving model (baseline)", "file": SERVING_MODEL, "sha256": _sha(models_dir / SERVING_MODEL),
                "feature_columns": base_cols,
                "threshold_selection": "original pipeline: random 15 % split of the training seasons",
                "metrics": full_metrics(y_te, p_base, meta["xgboost_decision_threshold"], meta["xgboost_alert_threshold"])}

    results, fitted = [], {}
    for name, spec in candidates.items():
        cols = [c for c in base_cols if c not in spec.get("drop", [])]
        params = {**spec["params"], "n_jobs": n_jobs}
        oof = season_oof(train, cols, params)
        t_early, t_alert = thresholds_from(train[LABEL].values, oof)
        model = xgb.XGBClassifier(**params).fit(train[cols], train[LABEL])
        p = model.predict_proba(test[cols])[:, 1]
        m = full_metrics(y_te, p, t_early, t_alert)
        from sklearn.metrics import average_precision_score
        res = {"name": name, "feature_columns": cols, "params": spec["params"],
               "threshold_selection": "season-wise out-of-season predictions on 2023/2024",
               "season_cv_pr_auc": float(average_precision_score(train[LABEL].values, oof)),
               "metrics": m, "gate": gate(baseline["metrics"], m)}
        results.append(res)
        fitted[name] = (model, cols, t_early, t_alert, oof)
        logger.info("%s: PR-AUC %.4f (baseline %.4f) gate %s", name, m["pr_auc"], baseline["metrics"]["pr_auc"],
                    "PASS" if res["gate"]["passed"] else "FAIL")

    passing = [r for r in results if r["gate"]["passed"]]
    best = max(passing, key=lambda r: r["metrics"]["pr_auc"]) if passing else None
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": "binary: satellite-detected fire in a 0.1-degree zone on a given day",
        "validation": {"train": [str(train["date"].min().date()), str(train["date"].max().date())],
                       "test_held_out": [str(test["date"].min().date()), str(test["date"].max().date())],
                       "train_rows": int(len(train)), "test_rows": int(len(test)),
                       "method": "temporal hold-out (2025 season) + season-wise CV on 2023/2024 for candidates",
                       "seed": SEED},
        "primary_metric": "PR-AUC (fires are ~4 % of zone-days) with early-warning recall as the safety constraint",
        "gate_rules": {"min_pr_auc_gain": GATE_MIN_PR_AUC_GAIN, "max_roc_auc_loss": GATE_MAX_ROC_AUC_LOSS,
                       "log_loss": "must not increase", "max_early_warning_recall_loss": GATE_MAX_RECALL_LOSS},
        "baseline": baseline, "candidates": results,
        "decision": (f"PROMOTE {best['name']}" if best else
                     "KEEP serving model: no candidate passed the promotion gate on the held-out season"),
        "selected": best["name"] if best else baseline["name"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_JSON).write_text(json.dumps(report, indent=2))
    rows = [{"model": baseline["name"], **_flat(baseline["metrics"])}] + \
           [{"model": r["name"], **_flat(r["metrics"]), "gate_passed": r["gate"]["passed"]} for r in results]
    pd.DataFrame(rows).to_csv(out_dir / REPORT_CSV, index=False)
    report["_fitted"] = fitted
    report["_train"] = train
    return report


def _flat(m: dict) -> dict:
    out = {k: round(m[k], 4) for k in ("roc_auc", "pr_auc", "brier", "log_loss", "expected_calibration_error")}
    for op in ("early_warning", "alert"):
        for k in ("threshold", "accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
                  "false_positive_rate", "false_negative_rate"):
            out[f"{op}_{k}"] = round(m[op][k], 4)
    return out


def promote(report: dict, models_dir: Path = MODELS_DIR) -> Optional[str]:
    """Saves and activates the gate-passing candidate as a new version; returns the
    version, or None (serving model untouched) when nothing passed."""
    from src.ml_models.risk_index import RiskIndex
    models_dir = Path(models_dir)
    ensure_baseline_registered(models_dir)
    best = next((r for r in report["candidates"] if r["name"] == report["selected"]), None)
    if best is None or not best["gate"]["passed"]:
        logger.info("No promotion: %s", report["decision"])
        return None
    model, cols, t_early, t_alert, oof = report["_fitted"][best["name"]]
    train = report["_train"]
    reg = load_registry(models_dir)
    version = f"v4.{len(reg['versions'])}"
    vdir = models_dir / VERSIONS_DIR / version
    vdir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(vdir / SERVING_MODEL))
    RiskIndex.fit(train[LABEL].values, oof, t_early, high_threshold=t_alert).save(vdir, SERVING_INDEX)
    (vdir / SERVING_META).write_text(json.dumps({
        "model_version": version, "train_rows": int(len(train)),
        "train_date_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_date_range": report["validation"]["test_held_out"],
        "feature_columns": cols, "feature_set_version": 4,
        "xgboost_decision_threshold": t_early, "xgboost_alert_threshold": t_alert,
        "params": best["params"], "held_out_metrics": _flat(best["metrics"]),
        "split_method": report["validation"]["method"],
    }, indent=2))
    reg["versions"].append({"version": version, "sha256": _sha(vdir / SERVING_MODEL),
                            "registered_utc": datetime.now(timezone.utc).isoformat(),
                            "source": f"candidate {best['name']}", "feature_columns": cols})
    save_registry(reg, models_dir)
    activate(version, models_dir)
    return version


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    sub.add_parser("promote")
    rb = sub.add_parser("rollback")
    rb.add_argument("--to", required=True)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(load_registry(), indent=2))
        return
    if a.cmd == "rollback":
        ensure_baseline_registered()
        print(json.dumps(activate(a.to), indent=2))
        return
    ensure_baseline_registered()
    report = evaluate()
    print(pd.read_csv(DATA_PROCESSED_DIR / REPORT_CSV).set_index("model").T.to_string())
    print("\nDECISION:", report["decision"])
    if a.cmd == "promote":
        v = promote(report)
        print("promoted:", v or "nothing (serving model unchanged)")


if __name__ == "__main__":
    main()
