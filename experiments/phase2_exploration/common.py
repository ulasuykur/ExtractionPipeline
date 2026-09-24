"""Shared helpers for the Phase 2 exploration scripts (see README.md in this folder)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# The repo's frozen meta-feature table (two folders up from this file).
DATA = Path(__file__).resolve().parents[2] / "data" / "metafeatures.csv"
F = ["log_n", "log_p", "skew_mean", "skew_sd", "kurt_min", "kurt_q3", "cor_mean", "conc_mean"]
SLOG = ["skew_mean", "skew_sd", "kurt_min", "kurt_q3"]

d = pd.read_csv(DATA)
X = d[F].copy()
y = d["k_true"].to_numpy().astype(float)
groups = d["group"].to_numpy()
# 5 k-brackets of the pool (D10): 2 | 3 | 4-5 | 6-9 | 10-50
bracket = pd.cut(y, [1, 2, 3, 5, 9, 50], labels=list("ABCDE")).astype(str).to_numpy()


def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


def pre(scale=True):
    ct = ColumnTransformer(
        [("slog", FunctionTransformer(signed_log), SLOG)], remainder="passthrough"
    )
    steps = [("slog", ct)]
    if scale:
        steps.append(("sc", StandardScaler()))
    return steps


class Median(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.m_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


def models_default(rf_trees=200):
    return {
        "median": Median(),
        "ridge": Pipeline(pre() + [("m", Ridge(alpha=1.0))]),
        "rf": Pipeline(pre(False) + [("m", RandomForestRegressor(n_estimators=rf_trees, random_state=0, n_jobs=2))]),
        "knn": Pipeline(pre() + [("m", KNeighborsRegressor(n_neighbors=5))]),
    }


GRIDS = {
    "ridge": {"m__alpha": np.logspace(-3, 3, 13)},
    "rf": {"m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": [1.0, 0.5, 0.33]},
    "knn": {"m__n_neighbors": [1, 3, 5, 7, 10, 15], "m__weights": ["uniform", "distance"]},
}


def clip(p):
    return np.maximum(p, 2.0)


def mae(yt, p):
    return float(np.mean(np.abs(yt - clip(p))))


def within1(yt, p):
    return float(np.mean(np.abs(np.round(clip(p)) - yt) <= 1))
