"""
t3_measures.py - additional error measures for the Phase 4 main run (methodology §12.7; idea T3).

Reads the saved predictions of the main run (outputs/phase4/predictions.csv) and computes, for every
configuration and every test part:
  RMSE, mean absolute log error, MAPE, MRAE, RRMSE, median absolute error, exact hits, Spearman rho.
They are averaged as in §2.6 (test parts -> repeat -> mean ± sd over the 10 repeats).

MAPE, MRAE and RRMSE follow Pimentel & de Carvalho (2020, §5.5, Eqs. 7-9). MRAE and RRMSE compare the
error with that of the "average" baseline; here that is the mean k of the TRAINING part (no test
information is used).

Constant baselines that are optimal for a measure are added, each computed on the training part:
  mean k (best constant for RMSE/RRMSE), 1/k-weighted median (best constant for MAPE),
  most frequent k (best constant for exact hits; ties -> smaller k). The median (best constant for
  MAE, median absolute error and absolute log error) is already in the predictions.

Reported only: no decision depends on these numbers; MAE stays the primary measure (§2.6).
Split by source (tabular / image; added on the user's request 2026-10-01, reported only): the image
test parts are tiny (0-5 image datasets per fold), so per source the measures are computed on the
100 test predictions of each repeat (every dataset once) and then averaged over the repeats.
Built-in check: the MAE recomputed from the predictions must equal results/phase4/summary.csv.

Usage:  python t3_measures.py
"""

import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import evaluation as ev  # frozen protocol (unchanged)
import train_static as ts  # sha256 helper (unchanged)

ROOT = Path(__file__).resolve().parent
PREDICTIONS = (
    ROOT / "outputs" / "phase4" / "predictions.csv"
)  # written by train_phase4.py
MAIN_SUMMARY = ROOT / "results" / "phase4" / "summary.csv"  # for the MAE check
OUT_DIR = ROOT / "results" / "phase4_t3"

MEASURES = [
    "mae",
    "rmse",
    "male",
    "mape",
    "mrae",
    "rrmse",
    "medae",
    "exact",
    "spearman",
]
LABELS = {
    "mae": "MAE (primary, for the check)",
    "rmse": "RMSE",
    "male": "mean absolute log error",
    "mape": "MAPE",
    "mrae": "MRAE (vs. training mean)",
    "rrmse": "RRMSE (vs. training mean)",
    "medae": "median absolute error",
    "exact": "exact hits (rounded)",
    "spearman": "Spearman rho (pred vs. true)",
}


# ----------------------------------------------------------------------------------------
# 1. Constant baselines (computed on the training part only)
# ----------------------------------------------------------------------------------------
def weighted_median(values, weights):
    """Smallest value whose cumulative weight reaches half of the total weight. With weights 1/k
    this constant minimises MAPE = mean(|k - c| / k)."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])


def mode(values):
    """Most frequent value; ties -> the smaller value (np.unique sorts)."""
    vals, counts = np.unique(values, return_counts=True)
    return float(vals[np.argmax(counts)])


def constant_predictions(df, folds):
    """Test predictions of the three extra constant baselines + the training mean per split."""
    y = df[ev.TARGET].to_numpy(dtype=float)
    rows, train_means = [], []
    for repeat, fold, tr, te in ev.iter_splits(df, folds):
        consts = {
            "const_mean": float(np.mean(y[tr])),
            "const_weighted_median": weighted_median(y[tr], 1.0 / y[tr]),
            "const_mode": mode(y[tr]),
        }
        train_means.append(
            {"repeat": repeat, "fold": fold, "train_mean": consts["const_mean"]}
        )
        for config, c in consts.items():
            for i in te:
                rows.append(
                    {
                        ev.KEY: df.at[i, ev.KEY],
                        "repeat": repeat,
                        "fold": fold,
                        "config": config,
                        "k_true": y[i],
                        "k_pred": float(ev.clip_k([c])[0]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(train_means)


# ----------------------------------------------------------------------------------------
# 2. Measures for one test part
# ----------------------------------------------------------------------------------------
def measures(k, p, a_bar):
    """All measures for true k, (clipped) predictions p and the training mean a_bar."""
    err = np.abs(k - p)
    base_err = np.abs(k - a_bar)
    ok = base_err > 1e-12  # MRAE is undefined where k equals the training mean exactly
    if np.unique(p).size > 1:
        rho = stats.spearmanr(p, k).statistic
    else:
        rho = np.nan  # constant predictions in this test part: correlation undefined
    return {
        "mae": err.mean(),
        "rmse": np.sqrt(np.mean(err**2)),
        "male": np.mean(np.abs(np.log(p) - np.log(k))),
        "mape": np.mean(err / k),
        "mrae": np.mean(err[ok] / base_err[ok]),
        "mrae_excluded": int((~ok).sum()),
        "rrmse": np.sqrt(np.sum(err**2) / np.sum(base_err**2)),
        "medae": np.median(err),
        "exact": np.mean(np.round(p) == k),
        "spearman": rho,
    }


def fold_measures(preds, train_means):
    """One row per (config, repeat, fold) with all measures."""
    a_bar = train_means.set_index(["repeat", "fold"])["train_mean"]
    rows = []
    for (config, repeat, fold), part in preds.groupby(
        ["config", "repeat", "fold"], sort=False
    ):
        m = measures(
            part["k_true"].to_numpy(float),
            part["k_pred"].to_numpy(float),
            a_bar.at[(repeat, fold)],
        )
        rows.append({"config": config, "repeat": repeat, "fold": fold, **m})
    return pd.DataFrame(rows)


def summarize(fold_scores):
    """Mean ± sd over the repeats of the per-repeat averages (§2.6), per configuration."""
    rows = []
    for config, part in fold_scores.groupby("config", sort=False):
        row = {"config": config}
        for m in MEASURES:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", RuntimeWarning
                )  # all-NaN Spearman (constants)
                mean, sd = ev.summarize(part, m)
            row[f"{m}_mean"], row[f"{m}_sd"] = mean, sd
        row["mrae_excluded_total"] = int(part["mrae_excluded"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def by_source(preds, train_means, df):
    """Measures per source (tabular / image): pooled over the test predictions of each repeat,
    then mean ± sd over the repeats. Each prediction keeps the training mean of its own split.
    """
    source = np.where(df["source"].str.startswith("image"), "image", "tabular")
    p = preds.merge(pd.DataFrame({ev.KEY: df[ev.KEY], "source": source}), on=ev.KEY)
    p = p.merge(train_means, on=["repeat", "fold"])
    rows = []
    for (config, src, repeat), part in p.groupby(
        ["config", "source", "repeat"], sort=False
    ):
        m = measures(
            part["k_true"].to_numpy(float),
            part["k_pred"].to_numpy(float),
            part["train_mean"].to_numpy(float),
        )
        if config == "median_baseline" or config.startswith("const_"):
            # constant within a fold, but the constant differs slightly between folds:
            # a correlation over the pooled predictions would be an artefact -> left empty
            m["spearman"] = np.nan
        rows.append({"config": config, "source": src, "repeat": repeat, **m})
    per_repeat = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # sd of a single repeat (test runs)
        out = per_repeat.groupby(["config", "source"], sort=False)[MEASURES].agg(
            ["mean", "std"]
        )
    out.columns = [f"{m}_{s.replace('std', 'sd')}" for m, s in out.columns]
    return out.reset_index()


def main():
    df = ev.load_metafeatures()
    folds = ev.load_folds(df)
    preds = pd.read_csv(PREDICTIONS)
    consts, train_means = constant_predictions(df, folds)
    all_preds = pd.concat([preds, consts], ignore_index=True)

    scores = fold_measures(all_preds, train_means)
    summary = summarize(scores)

    # Check: the recomputed MAE must equal the main run's summary (saved with 4 decimals).
    main = pd.read_csv(MAIN_SUMMARY).set_index("config")["mae_mean"]
    ours = summary.set_index("config")["mae_mean"].loc[main.index]
    worst = float((ours - main).abs().max())
    if worst > 1e-4:
        raise SystemExit(
            f"predictions do not match results/phase4/summary.csv (max {worst:.2g})"
        )
    print(
        f"MAE check against results/phase4/summary.csv: OK (max difference {worst:.1e})"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "t3_fold_scores.csv", index=False, float_format="%.6f")
    summary.to_csv(OUT_DIR / "t3_summary.csv", index=False, float_format="%.4f")
    split = by_source(all_preds, train_means, df)
    split.to_csv(OUT_DIR / "t3_by_source.csv", index=False, float_format="%.4f")
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "design": "docs/metalearner_methodology.md §12.7 (T3); reported only",
        "predictions_sha256": ts.sha256(PREDICTIONS),
        "main_summary_sha256": ts.sha256(MAIN_SUMMARY),
        "mae_check_max_diff": worst,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved {OUT_DIR}")

    key = [
        "median_baseline",
        "const_mean",
        "const_weighted_median",
        "const_mode",
        "knn__B__logk",
        "rf__A__logk",
        "rf__D__logk",
        "rf__E__logk",
        "rf__F__logk",
        "direct_best_probe",
        "direct_best_t1",
    ]
    table = summary.set_index("config").loc[key, [f"{m}_mean" for m in MEASURES]]
    table.columns = MEASURES
    pd.set_option("display.width", 200)
    print(
        "\nAdditional measures (mean over repeats; lower is better except exact and spearman):"
    )
    print(table.round(3).to_string())

    for src in ["tabular", "image"]:
        part = split[split["source"] == src].set_index("config")
        tab = part.loc[
            [c for c in key if c in part.index], [f"{m}_mean" for m in MEASURES]
        ]
        tab.columns = MEASURES
        print(f"\n{src} datasets only (pooled per repeat, mean over repeats):")
        print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
