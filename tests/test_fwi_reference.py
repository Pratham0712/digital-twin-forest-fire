import numpy as np
import pandas as pd

from src.ml_models import fwi_reference as fr


def test_percentile_from_quantile_file(tmp_path):
    fr._cache.clear()
    q = np.quantile(np.arange(0, 101, dtype=float), np.linspace(0, 1, 101))
    pd.DataFrame({"percentile": np.arange(101), "fwi": q}).to_csv(tmp_path / fr.QUANTILE_FILE, index=False)
    assert abs(fr.fwi_percentile(50, tmp_path) - 50) < 1
    assert fr.fwi_percentile(-5, tmp_path) == 0.0
    assert fr.fwi_percentile(500, tmp_path) == 100.0
    assert fr.fwi_percentile(float("nan"), tmp_path) is None


def test_build_from_csv_then_reuse(tmp_path):
    fr._cache.clear()
    src = tmp_path / "train.csv"
    pd.DataFrame({"fwi": np.arange(0, 200, dtype=float)}).to_csv(src, index=False)
    q = fr.build_quantiles(src, tmp_path)
    assert q is not None and (tmp_path / fr.QUANTILE_FILE).exists()
    assert abs(fr.fwi_percentile(100, tmp_path) - 50) < 2


def test_missing_everything_returns_none(tmp_path, monkeypatch):
    fr._cache.clear()
    monkeypatch.setattr(fr, "DATA_PROCESSED_DIR", tmp_path)
    assert fr.fwi_percentile(10, tmp_path) is None
