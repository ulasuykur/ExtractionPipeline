"""
screen_t1.py - T1: classical k-estimators as meta-features, screened on the REAL pool
(branch feature/proposed-metafeatures, 2026-09-25; idea T1 in the project doc "Ideas - To Be Tested").

Idea: clustering methods that must be GIVEN a number of clusters (k-means, Gaussian mixture)
are run for every k of a fixed grid. A validity index rates each resulting partition, and the
candidate meta-feature is "the k that the index rates best". No labels are used anywhere (D3):
the grid is fixed in advance for all datasets and was derived from the pool filter
2 <= k <= 50 (D10), not from the true k of our datasets.

Candidates (4):
  t1_best_k_sil   k-means partitions, silhouette (max)            pymfe MFEClustering.ft_sil
  t1_best_k_ch    k-means partitions, Calinski-Harabasz (max)     pymfe MFEClustering.ft_ch
  t1_best_k_db    k-means partitions, Davies-Bouldin (min)        pymfe MFEClustering.ft_vdb
  t1_best_k_bic   Gaussian mixture (diagonal covariances), BIC (min)   scikit-learn
The pymfe functions are called directly (as for t4 in the pymfe screening): they wrap the
scikit-learn metrics; going through MFE(groups=["clustering"]) would add precomputations
that are not needed here.

Fixed settings (decided before the run):
  GRID           k in {2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50} (roughly log-spaced)
  MIN_PER_GROUP  grid values with k > n_rows / 5 are skipped (at least 5 points per group on
                 average); depends only on n, never on the true k
  rows           at most 5,000 rows per dataset (fixed seed); silhouette on 2,000 of them
  scaling        columns standardized (k-means and the mixture depend on scale)
  k-means        n_init = 3, random_state = 0;  mixture: diagonal covariances, random_state = 0

Pass criterion: identical to the pymfe screening (|rho beyond size| >= 0.30 on all datasets,
>= 0.20 on the tabular ones, same sign). In addition the script reports each estimator used
directly as a prediction of k (MAE, share within +-1) - a preview of the "best single
classical estimator" baseline, which is evaluated formally with the protocol afterwards.

Usage (from the repo root, conda env ai-meta-k; OpenML must be reachable):
    python experiments/t1_classical/screen_t1.py            # extract (resumes) + analyse
    python experiments/t1_classical/screen_t1.py --analyse  # analyse existing results only
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymfe.clustering import MFEClustering
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from extractor import (  # noqa: E402  (reuse the exact Phase 1 loading steps)
    drop_constant_columns,
    load_openml_numeric,
    take_class_subset,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 0
GRID = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50]
MIN_PER_GROUP = 5
MAX_ROWS = 5_000
MAX_ROWS_SIL = 2_000
POOL = ROOT / "data" / "pool.csv"
METAFEATURES = ROOT / "data" / "metafeatures.csv"  # for k_true, log_n, log_p, source
OUT_DIR = ROOT / "outputs" / "t1_classical"
CANDIDATES_CSV = OUT_DIR / "candidates.csv"  # one row per dataset: the 4 best-k values
CURVES_CSV = OUT_DIR / "curves.csv"  # every index value for every k (for later analyses)
RESULT_CSV = OUT_DIR / "screen_result.csv"
ESTIMATORS_CSV = OUT_DIR / "estimators_direct.csv"

CANDIDATES = ["t1_best_k_sil", "t1_best_k_ch", "t1_best_k_db", "t1_best_k_bic"]
PHASE2_FEATURES = ["skew_mean", "skew_sd", "kurt_min", "kurt_q3", "cor_mean", "conc_mean"]

# Pass criterion (same as the pymfe screening, fixed before the run)
MIN_RHO_BEYOND_SIZE = 0.30
MIN_RHO_TABULAR = 0.20


# ----------------------------------------------------------------------------------------
# 1. Index curves and best k for one dataset
# ----------------------------------------------------------------------------------------
def index_curves(X):
    """X: numeric matrix after constant columns were removed (as in extractor.py).
    Returns one row per grid value k with the four index values."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = X.shape[0]
    rows = np.sort(rng.choice(n, size=MAX_ROWS, replace=False)) if n > MAX_ROWS else np.arange(n)
    Z = StandardScaler().fit_transform(X[rows])
    Z = Z[:, np.isfinite(Z).all(axis=0) & (Z.std(axis=0) > 0)]  # constant within the subsample
    m = Z.shape[0]
    sil_frac = min(1.0, MAX_ROWS_SIL / m)

    out = []
    for k in GRID:
        if k > m / MIN_PER_GROUP:
            continue  # too few points per group on average (rule depends on n only)
        row = {"k": k}
        try:
            labels = KMeans(n_clusters=k, n_init=3, random_state=RANDOM_STATE).fit_predict(Z)
            if len(np.unique(labels)) >= 2:
                row["sil"] = float(MFEClustering.ft_sil(Z, labels, sample_frac=sil_frac,
                                                        random_state=RANDOM_STATE))
                row["ch"] = float(MFEClustering.ft_ch(Z, labels))
                row["db"] = float(MFEClustering.ft_vdb(Z, labels))
        except Exception:  # a failing k leaves NaN; the other k values still count
            pass
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                  random_state=RANDOM_STATE).fit(Z)
            row["bic"] = float(gmm.bic(Z))
        except Exception:
            pass
        out.append(row)
    return pd.DataFrame(out).reindex(columns=["k", "sil", "ch", "db", "bic"])


def best_k(curves):
    """The k each index rates best (NaN values are ignored)."""
    def pick(col, fn):
        c = curves.dropna(subset=[col])
        return float(c["k"].iloc[fn(c[col].to_numpy())]) if len(c) else np.nan
    return {
        "t1_best_k_sil": pick("sil", np.argmax),
        "t1_best_k_ch": pick("ch", np.argmax),
        "t1_best_k_db": pick("db", np.argmin),
        "t1_best_k_bic": pick("bic", np.argmin),
        "t1_k_max_used": float(curves["k"].max()),
    }


# ----------------------------------------------------------------------------------------
# 2. Extraction over the pool (resumable: finished datasets are skipped)
# ----------------------------------------------------------------------------------------
def extract():
    pool = pd.read_csv(POOL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(CANDIDATES_CSV) if CANDIDATES_CSV.exists() else pd.DataFrame(columns=["name"])
    all_curves = pd.read_csv(CURVES_CSV) if CURVES_CSV.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for spec in pool.to_dict("records"):
        if spec["name"] in set(done["name"]):
            continue
        start = time.perf_counter()
        print(f"[{int(spec['dataset_id']):>5}] {spec['name']}: ...", end=" ", flush=True)
        try:
            X, y = load_openml_numeric(int(spec["dataset_id"]))
            if pd.notna(spec.get("subset_k")):
                X, y, _ = take_class_subset(
                    X, y, int(spec["dataset_id"]), int(spec["subset_k"]), int(spec["n_max"])
                )
            X = drop_constant_columns(X)
            curves = index_curves(X)
            row = {"name": spec["name"], **best_k(curves)}
            curves.insert(0, "name", spec["name"])
            all_curves = pd.concat([all_curves, curves], ignore_index=True)
            row["runtime_s"] = round(time.perf_counter() - start, 1)
            print(f"done ({row['runtime_s']} s)")
        except Exception as exc:  # one failing dataset must not stop the run
            row = {"name": spec["name"]}
            print(f"FAILED ({exc})")
        rows.append(row)
        pd.DataFrame(rows).to_csv(CANDIDATES_CSV, index=False)  # save after every dataset
        all_curves.to_csv(CURVES_CSV, index=False)


# ----------------------------------------------------------------------------------------
# 3. Analysis
# ----------------------------------------------------------------------------------------
def partial_spearman(df, feature, controls=("log_n", "log_p")):
    """Spearman correlation of feature and k_true after removing (linearly, on ranks) what
    the controls explain: does the feature tell us about k BEYOND the dataset size?"""
    r = df[[feature, "k_true", *controls]].rank()
    C = np.column_stack([np.ones(len(r))] + [r[c] for c in controls])
    resid = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]  # noqa: E731
    rho, pval = stats.pearsonr(resid(r[feature].to_numpy()), resid(r["k_true"].to_numpy()))
    return float(rho), float(pval)


def analyse():
    mf = pd.read_csv(METAFEATURES)
    cand = pd.read_csv(CANDIDATES_CSV)
    df = mf.merge(cand.drop(columns=["runtime_s"], errors="ignore"), on="name", how="left")
    df["tabular"] = ~df["source"].str.startswith("image")
    missing = df[CANDIDATES].isna().any(axis=1).sum()
    print(f"{len(df)} datasets, {missing} with missing candidate values")
    capped = int((df["k_true"] > df["t1_k_max_used"]).sum())
    print(f"{capped} datasets have a true k above the largest grid value used for them\n")

    rows = []
    for f in CANDIDATES + PHASE2_FEATURES:
        d = df.dropna(subset=[f])
        t = d[d["tabular"]]
        rho_all, p_all = stats.spearmanr(d[f], d["k_true"])
        rho_part, p_part = partial_spearman(d, f)
        rho_tab, p_tab = partial_spearman(t, f)
        passed = (abs(rho_part) >= MIN_RHO_BEYOND_SIZE and abs(rho_tab) >= MIN_RHO_TABULAR
                  and np.sign(rho_part) == np.sign(rho_tab))
        rows.append({
            "feature": f,
            "role": "candidate" if f in CANDIDATES else "Phase 2 reference",
            "rho_all": rho_all, "p_value_all": p_all,
            "rho_beyond_size": rho_part, "p_value_beyond_size": p_part,
            "rho_beyond_size_tabular": rho_tab, "p_value_tabular": p_tab,
            "n": len(d),
            "PASS": bool(passed) if f in CANDIDATES else None,
        })
    res = pd.DataFrame(rows)
    res.to_csv(RESULT_CSV, index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"\nPass criterion: |rho beyond size| >= {MIN_RHO_BEYOND_SIZE} (all datasets) and "
          f">= {MIN_RHO_TABULAR} on tabular only, same sign.")

    # Preview: each estimator used directly as the prediction of k (no training involved)
    base = float(np.median(df["k_true"]))
    est = []
    for f in CANDIDATES:
        d = df.dropna(subset=[f])
        err = (d[f] - d["k_true"]).abs()
        est.append({"estimator": f, "MAE": err.mean(),
                    "within1": float((err <= 1).mean()), "n": len(d)})
    err = (df["k_true"] - base).abs()
    est.append({"estimator": f"constant median k ({base:g}), for orientation", "MAE": err.mean(),
                "within1": float((err <= 1).mean()), "n": len(df)})
    est = pd.DataFrame(est)
    est.to_csv(ESTIMATORS_CSV, index=False, float_format="%.4f")
    print("\nPreview - estimators used directly as the prediction of k (all datasets, no CV;")
    print("the formal 'best single classical estimator' baseline follows with the protocol):")
    print(est.round(3).to_string(index=False))
    print(f"\nSaved {RESULT_CSV.name}, {ESTIMATORS_CSV.name} in {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T1 screening: classical k-estimators.")
    parser.add_argument("--analyse", action="store_true", help="only analyse existing results")
    args = parser.parse_args()
    if not args.analyse:
        extract()
    analyse()
