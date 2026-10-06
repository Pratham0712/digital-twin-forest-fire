import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml_models.explain import contributions, sigmoid


def test_contributions_sum_to_model_logit():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = (X["a"] + 0.5 * X["b"] + rng.normal(scale=0.3, size=300) > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=30, max_depth=3).fit(X, y)
    c, bias = contributions(m, X)
    p = m.predict_proba(X)[:, 1]
    assert np.allclose(sigmoid(c.sum(axis=1).to_numpy() + bias), p, atol=1e-4)
    assert c.abs().mean().idxmax() in ("a", "b")
