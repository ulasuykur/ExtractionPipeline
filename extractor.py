"""
Static meta-feature extractor (Phase 1).

Builds the static part of the meta-dataset: one row per OpenML dataset,
8 unsupervised meta-features + bookkeeping columns.

Feature choice and reasoning: Thesis Writing Notes M3, M5-M9;
Design Decisions Log D2-D4, D8, O3.

    Simple (M5)              log_n, log_p
    Statistical (M6)         skew_mean, skew_sd, kurt_min, kurt_q3, cor_mean
    Information-theor. (M7)  conc_mean

Pipeline:  load (OpenML)  ->  preprocess (drop constant columns)
           ->  extract (pymfe, no labels)  ->  table (CSV)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from pymfe.mfe import MFE

warnings.filterwarnings("ignore")  # pymfe is chatty about NaNs/precision

RANDOM_STATE = 0  # fixed seed: pymfe samples 12 columns for concentration (M7)
OUTPUT_PATH = Path("outputs") / "metafeatures.csv"  # outputs/ is git-ignored

# ---------------------------------------------------------------------
# 1. Test datasets: OpenML id -> name
#    10 from the first prototype + 1 real image dataset (runtime and
#    constant-pixel test). The final ~100-dataset pool comes later (M4).
# ---------------------------------------------------------------------

DATASETS: dict[int, str] = {
    61: "iris",
    187: "wine",
    1499: "seeds",
    41: "glass",
    39: "ecoli",
    54: "vehicle",
    36: "segment",
    28: "optdigits",
    32: "pendigits",
    1462: "banknote-auth",
    554: "mnist_784",  # 70,000 x 784 image dataset
}

# Output column order
COLUMNS = [
    "dataset_id",
    "name",
    "k_true",
    "n",
    "p_raw",
    "p_used",
    "log_n",
    "log_p",
    "skew_mean",
    "skew_sd",
    "kurt_min",
    "kurt_q3",
    "cor_mean",
    "conc_mean",
    "runtime_s",
]


# ---------------------------------------------------------------------
# 2. Loading: OpenML dataset -> numeric matrix X and labels y
# ---------------------------------------------------------------------


def load_openml_numeric(dataset_id: int) -> tuple[np.ndarray, pd.Series]:
    """Fetch a dataset from OpenML; return numeric features X and labels y.

    Labels are used ONLY for the ground truth k (D4), never for the
    meta-features (D3). Non-numeric columns are dropped; rows with NaNs
    are removed from X and y together so that k is counted on the same rows.
    """
    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X_df, y, _, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )
    X_num = X_df.select_dtypes(include=[np.number])
    keep = X_num.notna().all(axis=1) & y.notna()
    return X_num[keep].to_numpy(dtype=float), y[keep]


# ---------------------------------------------------------------------
# 3. Preprocessing (M9 / D8): drop constant columns, no scaling
# ---------------------------------------------------------------------


def drop_constant_columns(X: np.ndarray) -> np.ndarray:
    """Remove zero-variance columns (skewness, kurtosis, correlation are
    undefined for them; D2). No scaling: all 8 features are scale-invariant."""
    return X[:, np.ptp(X, axis=0) > 0]


# ---------------------------------------------------------------------
# 4. Extraction: X -> 8 meta-features (pymfe, unsupervised)
# ---------------------------------------------------------------------


def _summary(values: np.ndarray, func) -> float:
    """NaN-safe summary; NaN only if there is no valid value at all."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(func(values))


def extract_metafeatures(X: np.ndarray) -> dict[str, float]:
    """Compute the 8 static meta-features on the preprocessed matrix X.

    pymfe is called once WITHOUT summaries (summary=None) so that it returns
    the raw per-column / per-pair values; our own NaN-safe summaries are then
    applied, because each feature needs a different summary (M6, M7).
    No y is passed to fit (D3).
    """
    n, p = X.shape

    mfe = MFE(
        features=["skewness", "kurtosis", "cor", "attr_conc"],
        summary=None,
        random_state=RANDOM_STATE,
    )
    mfe.fit(X)
    names, values = mfe.extract(suppress_warnings=True)
    raw = dict(zip(names, values))

    return {
        "log_n": float(np.log10(n)),
        "log_p": float(np.log10(p)),
        "skew_mean": _summary(raw["skewness"], np.nanmean),
        "skew_sd": _summary(raw["skewness"], np.nanstd),
        "kurt_min": _summary(raw["kurtosis"], np.nanmin),
        # q3 instead of max: max is driven by near-constant columns (~n-3)
        "kurt_q3": _summary(raw["kurtosis"], lambda v: np.nanpercentile(v, 75)),
        "cor_mean": _summary(raw["cor"], np.nanmean),  # pymfe: |correlation|
        "conc_mean": _summary(raw["attr_conc"], np.nanmean),
    }


# ---------------------------------------------------------------------
# 5. Pipeline: datasets -> meta-feature table
# ---------------------------------------------------------------------


def process_dataset(dataset_id: int, name: str) -> dict:
    """Load, preprocess and extract one dataset; returns one table row."""
    start = time.perf_counter()
    X, y = load_openml_numeric(dataset_id)
    p_raw = X.shape[1]
    X = drop_constant_columns(X)
    if X.shape[1] < 2:
        raise ValueError(f"only {X.shape[1]} non-constant column(s) left")

    row = {
        "dataset_id": dataset_id,
        "name": name,
        "k_true": int(y.nunique()),  # D4: classes actually present
        "n": X.shape[0],
        "p_raw": p_raw,
        "p_used": X.shape[1],  # log_p is based on this (M9)
    }
    row.update(extract_metafeatures(X))
    row["runtime_s"] = round(time.perf_counter() - start, 2)
    return row


def build_metafeature_table(datasets: dict[int, str]) -> pd.DataFrame:
    rows = []
    for did, name in datasets.items():
        print(f"[{did:>5}] {name}: ...", end=" ", flush=True)
        try:
            row = process_dataset(did, name)
            rows.append(row)
            print(
                f"done (n={row['n']}, p={row['p_raw']}->{row['p_used']}, "
                f"{row['runtime_s']} s)"
            )
        except Exception as exc:  # failure isolation: one bad dataset != crash
            print(f"FAILED ({exc})")
            rows.append({"dataset_id": did, "name": name})

    return pd.DataFrame(rows).reindex(columns=COLUMNS)


if __name__ == "__main__":
    table = build_metafeature_table(DATASETS)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== Meta-feature table ===\n")
    print(table.to_string(index=False))

    n_nan = int(table.drop(columns=["runtime_s"]).isna().sum().sum())
    print(f"\nMissing values in table: {n_nan}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
