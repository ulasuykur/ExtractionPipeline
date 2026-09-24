"""
train_static.py - Phase 2: static-only meta-learner (baseline regressors).

What it does:
  1. Reads the frozen meta-feature table (data/metafeatures.csv) and the frozen splits
     (data/cv_folds.csv) through evaluation.py.
  2. Trains and tests 13 configurations on the same 5 x 10 splits:
       - median baseline (always predicts the median k of the training part)
       - Ridge, k-NN, Random Forest
           x target: raw k  or  log(k)
           x inputs: all 8 static meta-features  or  only log_n + log_p
             ("size/dimension heuristic" from the supervisor's proposal)
     Hyperparameters are tuned INSIDE each training part (nested CV, protocol P4);
     the test part is never used for any choice.
  3. Writes all scores, predictions and chosen hyperparameters, a summary table with the
     corrected t-test against the baseline, a breakdown per k-bin and per source,
     and two figures.

Outputs:
  outputs/phase2/   everything (git-ignored scratch)
  results/phase2/   the small files for the thesis (committed) - only after a FULL run

Usage:
    python train_static.py                # full run (10 repeats)
    python train_static.py --repeats 1    # quick test run (results/ is not touched)
"""

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files, no window needed

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import evaluation as ev

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "phase2"
RESULTS_DIR = ROOT / "results" / "phase2"

RANDOM_STATE = 0  # same seed convention as extractor.py

# ----------------------------------------------------------------------------------------
# 1. Inputs, transforms and models
# ----------------------------------------------------------------------------------------
# Two input sets: all 8 static meta-features, and the size-only heuristic.
FEATURE_SETS = {
    "static8": ev.FEATURES,
    "size2": ["log_n", "log_p"],
}

# Columns with a very uneven range and negative values (kurt_q3 up to 34,775; kurt_min,
# skew_mean and kurt_q3 partly negative). They get the signed log: sign(x) * log(1 + |x|).
# It keeps the sign, is 0 at 0, and compresses large values. Stateless -> no leakage.
SIGNED_LOG_COLS = ["skew_mean", "skew_sd", "kurt_min", "kurt_q3"]


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Trivial baseline: ignores the inputs and predicts the median k of the training part.
    (For MAE, the median is the best possible constant prediction.)"""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


# Model definitions: the estimator, whether inputs are standardized, and the tuning grid.
# Grids are wider than in the exploration, because the chosen values sat at the edges.
MODELS = {
    "ridge": {
        "estimator": lambda: Ridge(),
        "scale": True,  # Ridge penalizes coefficient size -> inputs must be comparable
        "grid": {"alpha": np.logspace(-3, 5, 17)},
    },
    "knn": {
        "estimator": lambda: KNeighborsRegressor(),
        "scale": True,  # distances -> every input must count equally
        "grid": {
            "n_neighbors": [1, 3, 5, 7, 10, 15, 20, 30],
            "weights": ["uniform", "distance"],
        },
    },
    "rf": {
        # n_jobs=1 per forest; the parallelism happens across the tuning grid instead
        "estimator": lambda: RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
        ),
        "scale": False,  # trees only compare values within a column -> no scaling needed
        "grid": {
            "min_samples_leaf": [1, 2, 5, 10, 20],
            "max_features": [0.33, 0.5, 1.0],
        },
    },
}
MODEL_LABELS = {"ridge": "Ridge", "knn": "k-NN", "rf": "Random Forest"}
FEATURE_LABELS = {"static8": "8 meta-features", "size2": "log n + log p only"}
TARGET_LABELS = {"k": "raw k", "logk": "log k"}


def build_pipeline(model, features):
    """Preprocessing + model for one configuration (fitted on the training part only)."""
    slog = [c for c in features if c in SIGNED_LOG_COLS]
    rest = [c for c in features if c not in SIGNED_LOG_COLS]
    transformers = [("passthrough", "passthrough", rest)]
    if slog:
        transformers.insert(0, ("signed_log", FunctionTransformer(signed_log), slog))
    steps = [("prep", ColumnTransformer(transformers))]
    if MODELS[model]["scale"]:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", MODELS[model]["estimator"]()))
    return Pipeline(steps)


def build_configs():
    """All 13 configurations, baseline first."""
    configs = [
        {
            "name": "median_baseline",
            "model": "median",
            "features": "static8",
            "target": "k",
        }
    ]
    for features in FEATURE_SETS:
        for model in MODELS:
            for target in ["k", "logk"]:
                configs.append(
                    {
                        "name": f"{model}__{features}__{target}",
                        "model": model,
                        "features": features,
                        "target": target,
                    }
                )
    return configs


def make_estimator(config):
    """
    The full estimator for one configuration.
    - log target: TransformedTargetRegressor trains on log(k) and converts predictions back
      with exp(), so everything outside (tuning score, metrics) stays on the k scale.
    - tuning: GridSearchCV with GroupKFold inside the training part (groups kept together),
      scored with exactly the protocol's MAE.
    """
    if config["model"] == "median":
        return MedianBaseline()

    pipe = build_pipeline(config["model"], FEATURE_SETS[config["features"]])
    prefix = "model__"
    if config["target"] == "logk":
        pipe = TransformedTargetRegressor(
            regressor=pipe, func=np.log, inverse_func=np.exp
        )
        prefix = "regressor__model__"
    grid = {prefix + k: v for k, v in MODELS[config["model"]]["grid"].items()}
    return GridSearchCV(
        pipe,
        grid,
        cv=GroupKFold(n_splits=ev.INNER_SPLITS),
        scoring=ev.mae_scorer,
        n_jobs=-1 if config["model"] == "rf" else 1,
    )


# ----------------------------------------------------------------------------------------
# 2. Running the protocol
# ----------------------------------------------------------------------------------------
def run(df, folds, configs, max_repeats):
    """Train/test every configuration on every (repeat, fold). Returns three tables."""
    X = df[ev.FEATURES]
    y = df[ev.TARGET].to_numpy(dtype=float)
    groups = df[ev.GROUP].to_numpy()

    pred_rows, score_rows, param_rows = [], [], []
    start = time.time()
    for repeat, fold, tr, te in ev.iter_splits(df, folds):
        if repeat >= max_repeats:
            break
        for config in configs:
            est = make_estimator(config)
            if isinstance(est, GridSearchCV):
                est.fit(X.iloc[tr], y[tr], groups=groups[tr])
                chosen = {k.split("__")[-1]: v for k, v in est.best_params_.items()}
            else:
                est.fit(X.iloc[tr], y[tr])
                chosen = {}
            pred = ev.clip_k(est.predict(X.iloc[te]))

            score_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "config": config["name"],
                    "mae": ev.mae(y[te], pred),
                    "within1": ev.within_one(y[te], pred),
                    "n_train": len(tr),
                    "n_test": len(te),
                }
            )
            param_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "config": config["name"],
                    "params": json.dumps(chosen, default=float),
                }
            )
            for i, p in zip(te, pred):
                pred_rows.append(
                    {
                        ev.KEY: df.at[i, ev.KEY],
                        "repeat": repeat,
                        "fold": fold,
                        "config": config["name"],
                        "k_true": y[i],
                        "k_pred": p,
                    }
                )
        if fold == ev.N_SPLITS - 1:
            print(f"repeat {repeat} done ({time.time() - start:.0f} s)", flush=True)
    return pd.DataFrame(pred_rows), pd.DataFrame(score_rows), pd.DataFrame(param_rows)


# ----------------------------------------------------------------------------------------
# 3. Summaries
# ----------------------------------------------------------------------------------------
def make_summary(scores, configs):
    """
    One row per configuration: mean +- sd over repeats (P5) and the corrected t-test of
    its MAE against the median baseline on the identical folds (P6).
    mae_diff < 0 means the model is better than the baseline.
    """
    base = scores[scores["config"] == "median_baseline"].set_index(["repeat", "fold"])
    rows = []
    for c in configs:
        s = scores[scores["config"] == c["name"]]
        mae_mean, mae_sd = ev.summarize(s, "mae")
        w1_mean, w1_sd = ev.summarize(s, "within1")
        row = {
            "config": c["name"],
            "model": c["model"],
            "features": c["features"],
            "target": c["target"],
            "mae_mean": mae_mean,
            "mae_sd": mae_sd,
            "within1_mean": w1_mean,
            "within1_sd": w1_sd,
        }
        if c["name"] != "median_baseline":
            s = s.set_index(["repeat", "fold"])
            diffs = s["mae"] - base.loc[s.index, "mae"]
            test = ev.corrected_ttest(
                diffs, n_train=s["n_train"].mean(), n_test=s["n_test"].mean()
            )
            row.update({f"vs_baseline_{k}": v for k, v in test.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def make_breakdown(preds, df):
    """
    MAE and within-1 per k-bin and per source (P5, secondary view), pooled over all
    repeats. Shows where the errors come from (e.g. the k >= 20 tail).
    """
    info = df[[ev.KEY, "source"]].copy()
    info["source"] = np.where(
        info["source"].str.startswith("image"), "image", "tabular"
    )
    p = preds.merge(info, on=ev.KEY)
    p["k_bin"] = ev.k_bin(p["k_true"])
    p["abs_err"] = (p["k_true"] - p["k_pred"]).abs()
    p["hit1"] = (np.round(p["k_pred"]) - p["k_true"]).abs() <= 1

    rows = []
    for col in ["k_bin", "source"]:
        g = p.groupby(["config", col])
        part = g.agg(
            mae=("abs_err", "mean"),
            within1=("hit1", "mean"),
            n_datasets=(ev.KEY, "nunique"),
        ).reset_index()
        part = part.rename(columns={col: "subgroup"})
        part.insert(1, "split_by", col)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------------------------
# Colors: validated categorical palette (colorblind-safe pair). Blue = 8 meta-features,
# orange = size-only. Filled marker = log k target, open marker = raw k target, so the
# meaning never depends on color alone.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FEATURE_COLORS = {"static8": BLUE, "size2": ORANGE}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scores(summary, path):
    """
    Figure 1: MAE (left) and share within +-1 (right) of every regressor, with the median
    baseline as a dashed reference line. Dots = mean over the 10 repeats, bars = sd.
    """
    base = summary[summary["config"] == "median_baseline"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    # The baseline value is written into the panel title (not next to the line), so the
    # label can never collide with a data point.
    panels = [
        (
            "mae",
            f"MAE on the k scale (lower is better; baseline {base['mae_mean']:.2f})",
            base["mae_mean"],
        ),
        (
            "within1",
            f"Share within ±1 (higher is better; baseline {base['within1_mean']:.0%})",
            base["within1_mean"],
        ),
    ]
    variants = [(f, t) for f in FEATURE_SETS for t in ["k", "logk"]]
    offsets = np.linspace(-0.27, 0.27, len(variants))
    models = list(MODELS)

    for ax, (metric, title, base_value) in zip(axes, panels):
        _style(ax)
        for (feat, target), dx in zip(variants, offsets):
            for xi, model in enumerate(models):
                r = summary[summary["config"] == f"{model}__{feat}__{target}"].iloc[0]
                color = FEATURE_COLORS[feat]
                ax.errorbar(
                    xi + dx,
                    r[f"{metric}_mean"],
                    yerr=r[f"{metric}_sd"],
                    fmt="o",
                    markersize=8,
                    color=color,
                    markerfacecolor=color if target == "logk" else "white",
                    markeredgewidth=2,
                    elinewidth=1.5,
                    capsize=0,
                )
        ax.axhline(base_value, color=INK_2, linestyle="--", linewidth=1.5)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], color=INK)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        if metric == "within1":
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(1.0, decimals=0)
            )

    # One legend explaining both encodings (color = inputs, fill = target) + the baseline.
    handles = [
        plt.Line2D(
            [], [], color=INK_2, linestyle="--", linewidth=1.5, label="median baseline"
        )
    ]
    for feat, target in variants:
        c = FEATURE_COLORS[feat]
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=8,
                color=c,
                markeredgewidth=2,
                markerfacecolor=c if target == "logk" else "white",
                label=f"{FEATURE_LABELS[feat]}, {TARGET_LABELS[target]}",
            )
        )
    fig.legend(
        handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=9
    )
    fig.suptitle(
        "Static-only meta-learner vs. median baseline (5-fold grouped CV x 10 repeats; mean ± sd over repeats)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pred_vs_true(preds, df, summary, path):
    """
    Figure 2: predicted vs. true k for each dataset (its test predictions averaged over the
    repeats). Columns = the three models. Rows = 8 meta-features with raw k target,
    8 meta-features with log k target, and the size-only inputs with log k target.
    Diagonal = perfect prediction; dashed line = the baseline's average prediction.
    Both axes are logarithmic, so small and large k are equally readable.
    """
    rows_spec = [("static8", "k"), ("static8", "logk"), ("size2", "logk")]
    src = df.set_index(ev.KEY)["source"].str.startswith("image")
    base_pred = preds.loc[preds["config"] == "median_baseline", "k_pred"].mean()
    ticks = [2, 3, 5, 10, 20, 50]
    rng = np.random.default_rng(RANDOM_STATE)  # small horizontal jitter, same every run
    fig, axes = plt.subplots(
        3, 3, figsize=(11, 10.4), sharex=True, sharey=True, constrained_layout=True
    )
    for row, (feats, target) in enumerate(rows_spec):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            _style(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            name = f"{model}__{feats}__{target}"
            mean_pred = (
                preds[preds["config"] == name]
                .groupby(ev.KEY)
                .agg(k_true=("k_true", "first"), k_pred=("k_pred", "mean"))
            )
            is_img = src.loc[mean_pred.index].to_numpy()
            jitter = np.exp(rng.uniform(-0.05, 0.05, len(mean_pred)))
            ax.plot(
                [1.8, 55],
                [1.8, 55],
                color=INK_2,
                linewidth=1,
                label="perfect prediction",
            )
            ax.axhline(
                base_pred,
                color=INK_2,
                linestyle="--",
                linewidth=1.5,
                label=f"median baseline (≈{base_pred:.1f})",
            )
            for mask, color, label in [
                (~is_img, BLUE, "tabular"),
                (is_img, ORANGE, "image"),
            ]:
                ax.scatter(
                    mean_pred["k_true"].to_numpy()[mask] * jitter[mask],
                    mean_pred["k_pred"].to_numpy()[mask],
                    s=36,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.9,
                    label=label,
                    zorder=3,
                )
            mae_val = summary.loc[summary["config"] == name, "mae_mean"].iloc[0]
            ax.set_title(
                f"{MODEL_LABELS[model]}, {TARGET_LABELS[target]}  (MAE {mae_val:.2f})\n"
                f"{FEATURE_LABELS[feats]}",
                loc="left",
                fontsize=10,
                color=INK,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.set_yticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
            ax.set_xlim(1.8, 55)
            ax.set_ylim(1.8, 55)
            if row == len(rows_spec) - 1:
                ax.set_xlabel("true k", color=INK_2)
            if col == 0:
                ax.set_ylabel("predicted k (mean over repeats)", color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9
    )
    fig.suptitle(
        "Predicted vs. true k per dataset",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------------------
def sha256(path):
    """Fingerprint of a file's content. Line endings are normalized (Windows CRLF -> LF),
    so the same content gives the same fingerprint on Windows, Linux and the cluster."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Phase 2: static-only meta-learner.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=ev.N_REPEATS,
        help="number of repeats to run (default: all 10; fewer = test run)",
    )
    args = parser.parse_args()
    full_run = args.repeats >= ev.N_REPEATS

    df = ev.load_metafeatures()
    folds = ev.load_folds(df)
    configs = build_configs()
    print(
        f"{len(df)} datasets, {len(configs)} configurations, {args.repeats} repeat(s)"
    )

    t0 = time.time()
    preds, scores, params = run(df, folds, configs, args.repeats)
    runtime = time.time() - t0
    summary = make_summary(scores, configs)
    breakdown = make_breakdown(preds, df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    scores.to_csv(OUT_DIR / "fold_scores.csv", index=False)
    params.to_csv(OUT_DIR / "hyperparameters.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False, float_format="%.4f")
    breakdown.to_csv(OUT_DIR / "breakdown.csv", index=False, float_format="%.4f")
    plot_scores(summary, OUT_DIR / "fig_scores.png")
    plot_pred_vs_true(preds, df, summary, OUT_DIR / "fig_pred_vs_true.png")

    # Everything needed to trace the numbers back to exact inputs and versions.
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "full_run": full_run,
        "repeats": args.repeats,
        "runtime_s": round(runtime),
        "metafeatures_sha256": sha256(ev.METAFEATURES_CSV),
        "cv_folds_sha256": sha256(ev.FOLDS_CSV),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2))

    # Only a complete run goes into the committed results folder.
    if full_run:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for f in [
            "summary.csv",
            "breakdown.csv",
            "fold_scores.csv",
            "hyperparameters.csv",
            "fig_scores.png",
            "fig_pred_vs_true.png",
            "run_info.json",
        ]:
            shutil.copy(OUT_DIR / f, RESULTS_DIR / f)
        print(f"Copied results to {RESULTS_DIR}")
    else:
        print("Test run: results/phase2 was NOT updated.")

    cols = [
        "config",
        "mae_mean",
        "mae_sd",
        "within1_mean",
        "vs_baseline_mean_diff",
        "vs_baseline_p",
    ]
    print(
        f"\nFinished in {runtime / 60:.1f} min. Summary (mae_diff < 0 = better than baseline):"
    )
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
