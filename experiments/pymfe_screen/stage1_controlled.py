"""
Stage 1 of the pymfe screening (sandbox, 2026-09-24): controlled artificial series.
EXPLORATORY - see docs/metalearner_methodology.md, section 10.1. The decisive stage is
screen_pool.py (real pool). Needs no OpenML data.

Screening of pymfe candidate meta-features (families A-D, approved 2026-09-24).
Two tests, both without OpenML:
  1. synthetic Gaussian groups: k in 2..50, p in {5, 20, 60}, two separations, n = 1000 fixed
  2. sklearn digits class subsets: k in {2, 3, 5, 10}, n = 340 fixed, balanced, 10 draws each
Criterion: |Spearman rho| >= 0.5 with k (synthetic: mean over strata where only k varies)
and the same sign with |rho| >= 0.5 on the digits subsets.
"""
import warnings, time, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.datasets import make_blobs, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymfe.mfe import MFE

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)


def drop_constant(X):
    return X[:, X.std(axis=0) > 0]


def pymfe_raw(X, groups, features):
    m = MFE(groups=groups, features=features, summary=None, random_state=0)
    m.fit(X, suppress_warnings=True)
    names, vals = m.extract(suppress_warnings=True)
    return dict(zip(names, vals))


def mean_(v):
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def candidates(X):
    X = drop_constant(np.asarray(X, dtype=float))
    n, p = X.shape
    Z = StandardScaler().fit_transform(X)                     # A, B: scale-free version
    out = {}
    # A. eigenvalue spectrum (pymfe 'eigenvalues', statistical group) on standardized data
    ev = np.sort(np.asarray(pymfe_raw(Z, ["statistical"], ["eigenvalues"])["eigenvalues"], float))[::-1]
    ev = np.clip(ev, 0, None); share = ev / ev.sum()
    out["A_eff_rank"] = float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum()))
    out["A_eff_rank_ratio"] = out["A_eff_rank"] / p
    out["A_top1_share"] = float(share[0])
    out["A_n90"] = int(np.searchsorted(np.cumsum(share), 0.90) + 1)
    out["A_n90_ratio"] = out["A_n90"] / p
    m = min(p - 1, 60)
    out["A_gap_pos"] = int(np.argmax(ev[:m] / np.maximum(ev[1:m + 1], 1e-12)) + 1) if m >= 1 else 1
    # B. PCA dimension ratios (complexity group) on standardized data
    c = pymfe_raw(Z, ["complexity"], ["t3", "t4"])
    out["B_t3"], out["B_t4"] = float(c["t3"]), float(c["t4"])
    # C. density (concept group); pymfe assumes [0, 1]-scaled columns; subsample for O(n^2)
    idx = RNG.choice(n, size=min(n, 1000), replace=False)
    M = MinMaxScaler().fit_transform(X[idx])
    cc = pymfe_raw(M, ["concept"], ["wg_dist", "cohesiveness"])
    out["C_wg_dist_mean"] = mean_(cc["wg_dist"]); out["C_cohesiveness_mean"] = mean_(cc["cohesiveness"])
    # D. others
    d = pymfe_raw(X, ["statistical", "itemset"], ["nr_norm", "nr_cor_attr", "sparsity", "one_itemset", "two_itemset"])
    out["D_nr_norm_ratio"] = float(d["nr_norm"]) / p
    out["D_nr_cor_attr"] = float(d["nr_cor_attr"])
    out["D_sparsity_mean"] = mean_(d["sparsity"])
    out["D_one_itemset_mean"] = mean_(d["one_itemset"]); out["D_two_itemset_mean"] = mean_(d["two_itemset"])
    # reference: Phase 2 style features (should FAIL, validates the screen)
    out["ref_skew_mean"] = float(np.mean(stats.skew(X, axis=0)))
    out["ref_kurt_min"] = float(np.min(stats.kurtosis(X, axis=0)))
    cm = np.abs(np.corrcoef(X, rowvar=False)); out["ref_cor_mean"] = float(cm[np.triu_indices(p, 1)].mean()) if p > 1 else np.nan
    return out


def synthetic():
    rows = []
    ks = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50]
    for p in [5, 20, 60]:
        for sep_name, std in [("separated", 1.0), ("overlapping", 3.0)]:
            for seed in range(3):
                for k in ks:
                    X, _ = make_blobs(n_samples=1000, centers=k, n_features=p, cluster_std=std,
                                      center_box=(-10, 10), random_state=1000 * seed + k + p)
                    rows.append({"p": p, "sep": sep_name, "seed": seed, "k": k, **candidates(X)})
        print(f"  synthetic p={p} done", flush=True)
    return pd.DataFrame(rows)


def digits():
    X, y = load_digits(return_X_y=True)
    rows = []
    for k in [2, 3, 5, 10]:
        for draw in range(10):
            cls = RNG.choice(10, size=k, replace=False)
            per = 340 // k
            idx = np.concatenate([RNG.choice(np.where(y == c)[0], size=per, replace=False) for c in cls])
            rows.append({"k": k, "draw": draw, **candidates(X[idx])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    syn = synthetic(); dig = digits()
    out = Path(__file__).resolve().parents[2] / "outputs" / "pymfe_screen"
    out.mkdir(parents=True, exist_ok=True)
    syn.to_csv(out / "stage1_synthetic.csv", index=False); dig.to_csv(out / "stage1_digits.csv", index=False)
    feats = [c for c in syn.columns if c[:2] in ("A_", "B_", "C_", "D_") or c.startswith("ref_")]
    res = []
    for f in feats:
        # synthetic: Spearman within each stratum (p, separation, seed): only k varies there
        strata = [stats.spearmanr(g["k"], g[f])[0] for _, g in syn.groupby(["p", "sep", "seed"]) if g[f].nunique() > 1]
        rho_syn = float(np.nanmean(strata)) if strata else np.nan
        rho_sep = float(np.nanmean([stats.spearmanr(g["k"], g[f])[0] for _, g in syn[syn.sep == "separated"].groupby(["p", "seed"]) if g[f].nunique() > 1] or [np.nan]))
        rho_ovl = float(np.nanmean([stats.spearmanr(g["k"], g[f])[0] for _, g in syn[syn.sep == "overlapping"].groupby(["p", "seed"]) if g[f].nunique() > 1] or [np.nan]))
        rho_pool = float(stats.spearmanr(syn["k"], syn[f])[0])
        rho_dig = float(stats.spearmanr(dig["k"], dig[f])[0]) if dig[f].nunique() > 1 else np.nan
        passed = (abs(rho_syn) >= 0.5) and (abs(rho_dig) >= 0.5) and (np.sign(rho_syn) == np.sign(rho_dig))
        res.append({"feature": f, "rho_synthetic": rho_syn, "rho_syn_separated": rho_sep, "rho_syn_overlapping": rho_ovl,
                    "rho_syn_pooled": rho_pool, "rho_digits": rho_dig, "PASS": passed})
    r = pd.DataFrame(res); r.to_csv(out / "stage1_result.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(2).to_string(index=False))
    print("\ndigits: mean per k"); print(dig.groupby("k")[feats].mean().T.round(3).to_string())
    print(f"\nruntime {time.time() - t0:.0f} s")
