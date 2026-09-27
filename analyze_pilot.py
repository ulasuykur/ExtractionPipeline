"""
analyze_pilot.py - candidate meta-features and decision rules R1-R3 of the O2 pilot (Phase 3).

Everything here was fixed BEFORE the pilot results existed (methodology §11.3, D12); this script only
applies those rules. It reads the raw probe outputs of run_probes.py and writes:

    results/phase3_pilot/
        candidates.csv   one row per dataset: the 16 candidate values (8 candidates x DEC/IDEC),
                         averaged over seeds (G4 is computed across seeds), plus k_true, n, p
        per_seed.csv     the same per seed (for R2c)
        r1_runtime.csv   measured times per dataset and the extrapolation to the pool
        r2a_mnist.csv    R2a: Spearman rho of each feature with k on the MNIST family (deciding)
        r2b_all.csv      R2b: rho with k_true on all 12 datasets; error of the k* candidates (report only)
        r2c_seeds.csv    R2c: agreement between seeds (rho across datasets for each pair of seeds)
        decision.json    R1-R3 outcome in words and numbers
        run_info.json    input fingerprints, versions

Candidates (per dataset, seed and algorithm; "last" = the last checkpoint, step 500):
    G1  G = M1/M2 at the last step for every grid k; k* = grid value just BEFORE the largest drop in G
        (Xie et al. 2016, §5.5: sharp drop from 9 to 10 -> 9)
    G2  grid k with the highest silhouette M10
    G3  largest grid k with M8 = k at the last step (all k clusters hold >= 1 % of the points)
    G4  grid k with the highest agreement between seeds (mean adjusted Rand index of the final labels
        over all pairs of seeds, M12); computed across seeds, so there is one value per dataset
    L1  grid k with the largest relative drop of M1 over the probe, (M1[0] - M1[last]) / M1[0]
    O1  M8 at the last step for the overcomplete K
    O2  effective number of clusters at the last step for K: exp(entropy of cluster sizes) = K ** M6
    O3  M8[0] - M8[last] for K
Ties in an arg-max are broken towards the smaller k (fixed rule, stated in advance).

Usage:  python analyze_pilot.py                        (reads outputs/phase3/pilot)
        python analyze_pilot.py --input <folder> --out <folder>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from run_probes import ALGORITHMS, GRID, PILOT

IN_DIR = Path("outputs") / "phase3" / "pilot"
OUT_DIR = Path("results") / "phase3_pilot"
METAFEATURES = (
    Path("data") / "metafeatures.csv"
)  # n and p of every pool dataset (for R1)

MNIST_FAMILY = ["mnist_784_k2", "mnist_784_k3", "mnist_784_k5", "mnist_784"]
GRID_CANDIDATES = ["G1", "G2", "G3", "G4", "L1"]
OVER_CANDIDATES = ["O1", "O2", "O3"]
FALLBACK_GRID = [2, 3, 5, 8, 13, 20, 30, 50]

# decision rules (methodology §11.3)
R1_MAX_EXTRA_HOURS = (
    3.0  # the grid may add at most 3 h of wall-clock time over overcomplete
)
R1_GPUS = 6  # wall-clock = total GPU time / 6 (at most 6 jobs at once)
R1_GPU_NAME = "2080 Ti"  # timings must come from this GPU type
R2A_MIN_RHO = 0.8  # |Spearman rho| with k on the MNIST family
R2C_MIN_RHO = 0.9  # "seeds agree very closely"
MAX_ROWS, HELDOUT_SHARE, MIN_POINTS = (
    5000,
    0.10,
    5,
)  # to predict n_train and the k values per pool dataset


# ---------------------------------------------------------------------
# 1. Reading the raw outputs
# ---------------------------------------------------------------------


def read_dataset(folder: Path) -> dict:
    info = json.loads((folder / "run_info.json").read_text())
    return {
        "info": info,
        "final": pd.read_csv(folder / "final.csv"),
        "ckpt": pd.read_csv(folder / "checkpoints.csv"),
        "pretrain": pd.read_csv(folder / "pretrain.csv"),
        "labels": {p.stem: np.load(p) for p in (folder / "labels").glob("*.npy")},
    }


def check_complete(name: str, d: dict) -> list[str]:
    """All seeds x algorithms x k of the design must be present; returns a list of problems."""
    design = d["info"]["design"]
    all_k = sorted(set(design["grid"]) | {design["overcomplete_k"]})
    expected = {
        (s, a, k) for s in design["seeds"] for a in design["algorithms"] for k in all_k
    }
    found = set(zip(d["final"].seed, d["final"].algorithm, d["final"].k))
    problems = [
        f"{name}: missing run seed={s} {a} k={k}"
        for s, a, k in sorted(expected - found)
    ]
    if "finished" not in d["info"]:
        problems.append(f"{name}: run_info.json has no 'finished' time")
    return problems


# ---------------------------------------------------------------------
# 2. Candidate meta-features
# ---------------------------------------------------------------------


def argmax_k(values: pd.Series) -> float:
    """k with the largest value (ties -> smaller k); NaN values are ignored."""
    values = values.dropna()
    if values.empty:
        return np.nan
    best = values.max()
    return float(min(k for k, v in values.items() if v == best))


def candidates_one_seed(d: dict, seed: int, alg: str) -> dict:
    design = d["info"]["design"]
    grid_k, K = design["grid"], design["overcomplete_k"]
    ck = d["ckpt"][(d["ckpt"].seed == seed) & (d["ckpt"].algorithm == alg)]
    last_step = ck.step.max()
    first = ck[ck.step == 0].set_index("k")
    last = ck[ck.step == last_step].set_index("k")
    fin = d["final"][
        (d["final"].seed == seed) & (d["final"].algorithm == alg)
    ].set_index("k")

    g = (last.M1_loss_train / last.M2_loss_heldout).loc[grid_k]  # G ratio per grid k
    drops = pd.Series(
        g.values[:-1] - g.values[1:], index=grid_k[:-1]
    )  # G(k_i) - G(k_i+1)
    fully_used = [k for k in grid_k if last.loc[k, "M8_clusters_used"] == k]
    rel_drop = ((first.M1_loss_train - last.M1_loss_train) / first.M1_loss_train).loc[
        grid_k
    ]
    return {
        "G1": argmax_k(drops) if len(grid_k) > 1 else np.nan,
        "G2": argmax_k(fin.M10_silhouette.loc[grid_k]),
        "G3": float(max(fully_used)) if fully_used else np.nan,
        "L1": argmax_k(rel_drop),
        "O1": float(last.loc[K, "M8_clusters_used"]),
        "O2": float(K ** last.loc[K, "M6_size_entropy"]),
        "O3": float(first.loc[K, "M8_clusters_used"] - last.loc[K, "M8_clusters_used"]),
    }


def stability_k(d: dict, alg: str) -> float:
    """G4: grid k whose final clusterings agree most between seeds (mean pairwise ARI, M12)."""
    design = d["info"]["design"]
    seeds, agreement = design["seeds"], {}
    for k in design["grid"]:
        labels = [d["labels"][f"seed{s}_{alg}_k{k}"] for s in seeds]
        agreement[k] = np.mean(
            [adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)]
        )
    return argmax_k(pd.Series(agreement))


def all_candidates(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (seed-averaged table, per-seed table). Column names like 'G1_DEC'."""
    rows, per_seed = [], []
    for name, d in data.items():
        info = d["info"]
        base = {
            "name": name,
            "k_true": info["k_true"],
            "n_train": info["n_train"],
            "p_used": info["p_used"],
        }
        row = dict(base)
        for alg in ALGORITHMS:
            seed_values = []
            for seed in info["design"]["seeds"]:
                values = candidates_one_seed(d, seed, alg)
                seed_values.append(values)
                per_seed.append({**base, "seed": seed, "algorithm": alg, **values})
            for cand in ["G1", "G2", "G3", "L1", *OVER_CANDIDATES]:
                row[f"{cand}_{alg}"] = np.nanmean([v[cand] for v in seed_values])
            row[f"G4_{alg}"] = stability_k(d, alg)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


def feature_columns(protocol: str) -> list[str]:
    names = GRID_CANDIDATES if protocol == "grid" else OVER_CANDIDATES
    return [f"{c}_{a}" for c in names for a in ALGORITHMS]


# ---------------------------------------------------------------------
# 3. Rules
# ---------------------------------------------------------------------


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    # too few values, or one side constant -> no correlation defined
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def rule_r1(data: dict) -> tuple[pd.DataFrame, dict]:
    """Runtime: fit time = a + b*p for one probe run and for one pretraining (2080 Ti only),
    predict both for every pool dataset, sum up per protocol, divide by the number of GPUs.
    """
    rows = []
    for name, d in data.items():
        info, fin = d["info"], d["final"]
        K = info["design"]["overcomplete_k"]
        grid_extra = fin[fin.k.isin(info["design"]["grid"]) & (fin.k != K)]
        rows.append(
            {
                "name": name,
                "gpu": info["gpu"],
                "p_used": info["p_used"],
                "run_s_mean": fin.M11_runtime_s.mean(),
                "pretrain_s_mean": d["pretrain"].pretrain_s.mean(),
                "grid_extra_s": grid_extra.M11_runtime_s.sum(),
                "overcomplete_s": d["pretrain"].groupby("seed").pretrain_s.first().sum()
                + fin[fin.k == K].M11_runtime_s.sum(),
                "job_total_s": info["total_s"],
            }
        )
    measured = pd.DataFrame(rows)
    usable = measured[measured.gpu.fillna("").str.contains(R1_GPU_NAME)]
    if len(usable) < 2:
        return measured, {"error": f"fewer than 2 datasets timed on {R1_GPU_NAME}"}
    b_run, a_run = np.polyfit(usable.p_used, usable.run_s_mean, 1)
    b_pre, a_pre = np.polyfit(usable.p_used, usable.pretrain_s_mean, 1)

    pool = pd.read_csv(METAFEATURES)
    n_seeds, n_alg = len(next(iter(data.values()))["info"]["design"]["seeds"]), len(
        ALGORITHMS
    )
    totals = {"overcomplete": 0.0, "grid_extra": 0.0, "fallback_extra": 0.0}
    for n, p in zip(pool.n, pool.p_used):
        n_train = min(n, MAX_ROWS) - int(round(HELDOUT_SHARE * min(n, MAX_ROWS)))
        k_max = n_train // MIN_POINTS
        K = min(50, k_max)
        t_run, t_pre = a_run + b_run * p, a_pre + b_pre * p
        totals["overcomplete"] += n_seeds * (t_pre + n_alg * t_run)
        totals["grid_extra"] += (
            n_seeds * n_alg * t_run * len([k for k in GRID if k <= k_max and k != K])
        )
        totals["fallback_extra"] += (
            n_seeds
            * n_alg
            * t_run
            * len([k for k in FALLBACK_GRID if k <= k_max and k != K])
        )
    hours = {key: val / 3600 / R1_GPUS for key, val in totals.items()}
    result = {
        "timed_on": list(usable.name),
        "fit_run_s": {"a": a_run, "b_per_column": b_run},
        "fit_pretrain_s": {"a": a_pre, "b_per_column": b_pre},
        "wallclock_hours_on_6_gpus": hours,
        "full_grid_passes": bool(hours["grid_extra"] <= R1_MAX_EXTRA_HOURS),
        "fallback_grid_passes": bool(hours["fallback_extra"] <= R1_MAX_EXTRA_HOURS),
    }
    return measured, result


def rule_r2(
    cands: pd.DataFrame, per_seed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = feature_columns("grid") + feature_columns("overcomplete")
    mnist = cands.set_index("name").loc[MNIST_FAMILY]
    r2a = pd.DataFrame(
        [
            {
                "feature": f,
                "protocol": "grid" if f in feature_columns("grid") else "overcomplete",
                **{f"value_{n}": mnist.loc[n, f] for n in MNIST_FAMILY},
                "rho_with_k": spearman(mnist[f], mnist.k_true),
            }
            for f in features
        ]
    )
    # tiny tolerance: with 4 datasets, rho = 0.8 exactly can come out as 0.79999...
    r2a["passes"] = r2a.rho_with_k.abs() >= R2A_MIN_RHO - 1e-9  # NaN -> False

    k_star = {f for f in features if f.split("_")[0] in GRID_CANDIDATES}
    r2b = pd.DataFrame(
        [
            {
                "feature": f,
                "rho_with_k_true": spearman(cands[f], cands.k_true),
                "mae_as_estimate": (
                    float(np.nanmean(np.abs(cands[f] - cands.k_true)))
                    if f in k_star
                    else np.nan
                ),
            }
            for f in features
        ]
    )

    rows = []
    for f in features:
        cand, alg = f.split("_")
        if cand == "G4":  # computed across seeds: no per-seed value
            continue
        wide = per_seed[per_seed.algorithm == alg].pivot(
            index="name", columns="seed", values=cand
        )
        for s1, s2 in itertools.combinations(wide.columns, 2):
            rows.append(
                {
                    "feature": f,
                    "seeds": f"{s1}-{s2}",
                    "rho": spearman(wide[s1], wide[s2]),
                }
            )
    r2c = pd.DataFrame(rows)
    return r2a, r2b, r2c


def rule_r3(r1: dict, r2a: pd.DataFrame, r2c: pd.DataFrame) -> dict:
    grid_signal = bool(r2a[(r2a.protocol == "grid")].passes.any())
    over_signal = bool(r2a[(r2a.protocol == "overcomplete")].passes.any())
    grid_choice = (
        "full grid"
        if r1.get("full_grid_passes")
        else "fallback grid" if r1.get("fallback_grid_passes") else None
    )
    passed = list(r2a[r2a.passes].feature)
    seeds_close = bool(
        len(passed)
        and not r2c.empty
        and (r2c[r2c.feature.isin(passed)].rho >= R2C_MIN_RHO - 1e-9).all()
    )
    if "error" in r1:
        outcome = f"no decision: R1 could not be evaluated ({r1['error']})"
    elif not grid_signal and not over_signal:
        outcome = "STOP: no feature passes R2a - discuss before any full run"
    elif grid_signal and grid_choice:
        outcome = f"full run: overcomplete + {grid_choice}"
    elif grid_signal:
        outcome = "full run: overcomplete only (no grid passes R1) - discuss"
    else:
        outcome = "full run: overcomplete only (no grid feature passes R2a)"
    return {
        "grid_has_signal": grid_signal,
        "overcomplete_has_signal": over_signal,
        "features_passing_r2a": passed,
        "grid_choice_r1": grid_choice,
        "fewer_seeds_allowed": (grid_choice is None) and seeds_close,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=IN_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    missing = [n for n in PILOT if not (args.input / n / "final.csv").exists()]
    if missing:
        raise SystemExit(f"no results yet for: {missing}")
    data = {n: read_dataset(args.input / n) for n in PILOT}
    problems = [p for n, d in data.items() for p in check_complete(n, d)]
    if problems:
        raise SystemExit("incomplete pilot:\n  " + "\n  ".join(problems))

    cands, per_seed = all_candidates(data)
    measured, r1 = rule_r1(data)
    r2a, r2b, r2c = rule_r2(cands, per_seed)
    r3 = rule_r3(r1, r2a, r2c)

    args.out.mkdir(parents=True, exist_ok=True)
    cands.to_csv(args.out / "candidates.csv", index=False)
    per_seed.to_csv(args.out / "per_seed.csv", index=False)
    measured.to_csv(args.out / "r1_runtime.csv", index=False)
    r2a.to_csv(args.out / "r2a_mnist.csv", index=False)
    r2b.to_csv(args.out / "r2b_all.csv", index=False)
    r2c.to_csv(args.out / "r2c_seeds.csv", index=False)
    decision = {"R1": r1, "R3": r3}
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, default=float)
    )
    inputs = sorted(args.input.glob("*/final.csv")) + sorted(
        args.input.glob("*/checkpoints.csv")
    )
    (args.out / "run_info.json").write_text(
        json.dumps(
            {
                "input_sha256": {
                    str(p.relative_to(args.input)): sha256(p) for p in inputs
                },
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
            indent=2,
        )
    )

    pd.set_option("display.width", 200)
    print(
        "== R1 runtime (hours of wall-clock on 6 GPUs):",
        {
            k: round(float(v), 2)
            for k, v in r1.get("wallclock_hours_on_6_gpus", {}).items()
        },
        "| error:" if "error" in r1 else "",
        r1.get("error", ""),
    )
    print("== R2a MNIST family (deciding):")
    print(
        r2a[["feature", "protocol", "rho_with_k", "passes"]]
        .round(2)
        .to_string(index=False)
    )
    print("== R3:", r3["outcome"])
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
