"""
screen_pool.py - decisive screening of label-free pymfe candidates on the REAL pool
(branch feature/structure-metafeatures, 2026-09-24).

Background (docs/metalearner_methodology.md, History): on artificial data, 6 candidate
features passed a controlled test, but that test also let the Phase 2 feature cor_mean
pass, which failed on the real pool. So only the real pool can decide.

What it does:
  1. Extraction: for every dataset in data/pool.csv, load it exactly as extractor.py does
     (same loader, same class subsets for the image variants, same constant-column
     removal) and compute the 6 candidates below. Labels are only used for k_true (D3/D4).
  2. Analysis: correlation of every candidate with k_true - on all 100 datasets, on the
     83 tabular ones only, and "beyond size" (after removing what log_n and log_p explain).
     The Phase 2 features are analysed the same way as a reference.

Candidates (definitions identical to the sandbox screening):
  A_eff_rank_ratio   eigenvalue spectrum (pymfe 'eigenvalues', statistical group), standardized
                     data: effective number of dimensions exp(entropy of eigenvalue shares) / p
  A_top1_share       share of the largest eigenvalue in the total variance
  A_n90_ratio        number of components needed for 90% of the variance / p
  B_t4               pymfe complexity 't4' (PCA dimensions for 95% variance / p), standardized data
  C_wg_dist_mean     pymfe concept 'wg_dist' (weighted distance / density), [0,1]-scaled, mean
  D_nr_cor_attr      pymfe statistical 'nr_cor_attr' (share of highly correlated column pairs)

t4 is computed with pymfe's MFEComplexity.ft_t4 directly (see compute_candidates).

Subsampling (screening only, deviates from M9): A, B, D use at most 20,000 rows and C uses
1,000 rows (C needs all pairwise distances), drawn with a fixed seed.

Usage (from the repo root, conda env ai-meta-k; OpenML must be reachable):
    python experiments/pymfe_screen/screen_pool.py            # extract (resumes) + analyse
    python experiments/pymfe_screen/screen_pool.py --analyse  # analyse existing file only
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymfe.complexity import MFEComplexity
from pymfe.mfe import MFE
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from extractor import (  # noqa: E402  (reuse the exact Phase 1 loading steps)
    drop_constant_columns,
    load_openml_numeric,
    take_class_subset,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 0
MAX_ROWS = 20_000  # A, B, D
MAX_ROWS_DIST = 1_000  # C (pairwise distances)
POOL = ROOT / "data" / "pool.csv"
METAFEATURES = ROOT / "data" / "metafeatures.csv"  # for k_true, log_n, log_p, source
OUT_DIR = ROOT / "outputs" / "pymfe_screen"
CANDIDATES_CSV = OUT_DIR / "candidates.csv"
RESULT_CSV = OUT_DIR / "screen_result.csv"

CANDIDATES = [
    "A_eff_rank_ratio",
    "A_top1_share",
    "A_n90_ratio",
    "B_t4",
    "C_wg_dist_mean",
    "D_nr_cor_attr",
]
PHASE2_FEATURES = ["skew_mean", "skew_sd", "kurt_min", "kurt_q3", "cor_mean", "conc_mean"]

# Pass criterion (fixed BEFORE the run, decided with the user):
MIN_RHO_BEYOND_SIZE = 0.30  # |partial rho| on all datasets, after removing log_n and log_p
MIN_RHO_TABULAR = 0.20  # |partial rho| on tabular datasets only, same sign


# ----------------------------------------------------------------------------------------
# 1. Candidate features for one dataset
# ----------------------------------------------------------------------------------------
def _pymfe(X, groups, features):
    """Raw pymfe values (summary=None), fitted WITHOUT labels (D3)."""
    m = MFE(groups=groups, features=features, summary=None, random_state=RANDOM_STATE)
    m.fit(X, suppress_warnings=True)
    names, values = m.extract(suppress_warnings=True)
    return dict(zip(names, values))


def _rows(n, max_rows, rng):
    return np.sort(rng.choice(n, size=max_rows, replace=False)) if n > max_rows else np.arange(n)


def compute_candidates(X):
    """X: numeric matrix after constant columns were removed (as in extractor.py)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n, p = X.shape
    Xs = X[_rows(n, MAX_ROWS, rng)]
    Z = StandardScaler().fit_transform(Xs)  # eigenvalues/PCA depend on scale -> standardize
    Z = Z[:, np.isfinite(Z).all(axis=0) & (Z.std(axis=0) > 0)]  # columns constant in the subsample
    out = {}

    # A. eigenvalue spectrum of the standardized data (= correlation matrix)
    ev = np.asarray(_pymfe(Z, ["statistical"], ["eigenvalues"])["eigenvalues"], float)
    ev = np.clip(np.sort(ev)[::-1], 0, None)
    share = ev / ev.sum()
    nz = share[share > 0]
    out["A_eff_rank_ratio"] = float(np.exp(-(nz * np.log(nz)).sum())) / Z.shape[1]
    out["A_top1_share"] = float(share[0])
    out["A_n90_ratio"] = float(np.searchsorted(np.cumsum(share), 0.90) + 1) / Z.shape[1]

    # B. PCA dimension ratio. pymfe's own t4 function is called directly: going through
    # MFE(groups=["complexity"]) would first build a pairwise distance table of ALL rows for
    # the whole complexity group (hours for e.g. dilbert, 10,000 x 2,000), although t4 does
    # not use it. Same definition, identical value (checked 2026-09-24).
    out["B_t4"] = float(MFEComplexity.ft_t4(N=Z, random_state=RANDOM_STATE))

    # C. density; pymfe assumes columns scaled to [0, 1]
    M = MinMaxScaler().fit_transform(X[_rows(n, MAX_ROWS_DIST, rng)])
    wg = np.asarray(_pymfe(M, ["concept"], ["wg_dist"])["wg_dist"], float)
    out["C_wg_dist_mean"] = float(np.nanmean(wg))

    # D. share of highly correlated column pairs
    out["D_nr_cor_attr"] = float(_pymfe(Xs, ["statistical"], ["nr_cor_attr"])["nr_cor_attr"])
    return out


# ----------------------------------------------------------------------------------------
# 2. Extraction over the pool (resumable: already finished datasets are skipped)
# ----------------------------------------------------------------------------------------
def extract():
    pool = pd.read_csv(POOL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(CANDIDATES_CSV) if CANDIDATES_CSV.exists() else pd.DataFrame(columns=["name"])
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
            row = {"name": spec["name"], **compute_candidates(X)}
            row["runtime_s"] = round(time.perf_counter() - start, 1)
            print(f"done ({row['runtime_s']} s)")
        except Exception as exc:  # one failing dataset must not stop the run
            row = {"name": spec["name"]}
            print(f"FAILED ({exc})")
        rows.append(row)
        pd.DataFrame(rows).to_csv(CANDIDATES_CSV, index=False)  # save after every dataset


# ----------------------------------------------------------------------------------------
# 3. Analysis
# ----------------------------------------------------------------------------------------
def partial_spearman(df, feature, controls=("log_n", "log_p")):
    """Spearman correlation of feature and k_true after removing (linearly, on ranks) what
    the controls explain. Answers: does the feature tell us about k BEYOND the size?"""
    r = df[[feature, "k_true", *controls]].rank()
    C = np.column_stack([np.ones(len(r))] + [r[c] for c in controls])
    resid = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    rho, pval = stats.pearsonr(resid(r[feature].to_numpy()), resid(r["k_true"].to_numpy()))
    return float(rho), float(pval)


def analyse():
    mf = pd.read_csv(METAFEATURES)
    cand = pd.read_csv(CANDIDATES_CSV)
    df = mf.merge(cand.drop(columns=["runtime_s"], errors="ignore"), on="name", how="left")
    df["tabular"] = ~df["source"].str.startswith("image")
    missing = df[CANDIDATES].isna().any(axis=1).sum()
    print(f"{len(df)} datasets, {missing} with missing candidate values\n")

    rows = []
    for f in CANDIDATES + PHASE2_FEATURES:
        d = df.dropna(subset=[f])
        t = d[d["tabular"]]
        rho_all, p_all = stats.spearmanr(d[f], d["k_true"])
        rho_part, p_part = partial_spearman(d, f)
        rho_tab, p_tab = partial_spearman(t, f)
        passed = (
            abs(rho_part) >= MIN_RHO_BEYOND_SIZE
            and abs(rho_tab) >= MIN_RHO_TABULAR
            and np.sign(rho_part) == np.sign(rho_tab)
        )
        rows.append(
            {
                "feature": f,
                "role": "candidate" if f in CANDIDATES else "Phase 2 reference",
                "rho_all": rho_all,
                "p_value_all": p_all,
                "rho_beyond_size": rho_part,
                "p_value_beyond_size": p_part,
                "rho_beyond_size_tabular": rho_tab,
                "p_value_tabular": p_tab,
                "n": len(d),
                "PASS": bool(passed) if f in CANDIDATES else None,
            }
        )
    res = pd.DataFrame(rows)
    res.to_csv(RESULT_CSV, index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))
    print(f"\nPass criterion: |rho beyond size| >= {MIN_RHO_BEYOND_SIZE} (all datasets) and "
          f"|rho beyond size| >= {MIN_RHO_TABULAR} on tabular only, same sign.")
    print(f"Saved {RESULT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pool screening of pymfe candidates.")
    parser.add_argument("--analyse", action="store_true", help="only analyse existing results")
    args = parser.parse_args()
    if not args.analyse:
        extract()
    analyse()
