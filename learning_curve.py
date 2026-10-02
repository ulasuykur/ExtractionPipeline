"""
learning_curve.py - Phase 4 learning curve (O1) for the main meta-learner (methodology §12.8, design §6).

Question: would more datasets help the main meta-learner (Random Forest, feature set D)?

Design (fixed in §6 on 2026-09-24 and in §12.8 on 2026-10-01, before any Phase 4 result):
  - test parts exactly as in the frozen protocol (data/cv_folds.csv, 5 folds x 10 repeats);
  - only the TRAINING part is subsampled to 25/50/75/100%: it is split into 4 quarters that keep
    groups together and are stratified by k-bin (as in experiments/phase2_exploration/e6_lc_noise.py),
    and the first 1, 2, 3 or 4 quarters are used (nested subsets); 5 random draws per fold;
  - on every subsample the screening (§12.2) and the tuning (§2.5) are redone;
  - plateau rule (§6): the curve has leveled off if the MAE improves by less than 5% from 75% to 100%
    AND the corrected t-test of that step (§2.7) is not significant;
  - additional condition (§6): a flat curve means "100 datasets are enough" only if the model at 100%
    beats the median baseline (pre-registered test of §12.6, results/phase4/hypotheses.csv); otherwise
    it means "the features lack signal", which more datasets would not fix.

Built-in check: at 100% the subsample is the full training part, so the scores must equal the main
run (results/phase4/fold_scores.csv) up to 1e-9; otherwise results/phase4_lc/ is not written.

Usage:
    python learning_curve.py               # full run (10 repeats)
    python learning_curve.py --repeats 1   # quick test run (results/ is not touched)
"""

import argparse
import json
import platform
import shutil
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files, no window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

import evaluation as ev  # frozen protocol (unchanged)
import train_phase4 as p4  # Phase 4 inputs, screening and models (unchanged)
import train_static as ts  # figure style (unchanged)

ROOT = Path(__file__).resolve().parent
RUN_NAME = "phase4_lc"  # output folders outputs/phase4_lc/ and results/phase4_lc/
MAIN_SCORES = (
    ROOT / "results" / "phase4" / "fold_scores.csv"
)  # main run, for the 100% check
MAIN_TESTS = (
    ROOT / "results" / "phase4" / "hypotheses.csv"
)  # for the §6 additional condition

MODEL, FEATURE_SET = "rf", "D"  # the main meta-learner (§12.6)
CONFIG = f"{MODEL}__{FEATURE_SET}__{p4.TARGET}"  # rf__D__logk
QUARTERS = [1, 2, 3, 4]  # 25 / 50 / 75 / 100% of the training part
N_DRAWS = 5  # random draws per fold (§6)
PLATEAU_GAIN = 0.05  # "improves by less than 5% from 75% to 100%" (§6)


def nested_subsets(tr, bins, groups, seed):
    """
    Split the training part `tr` (row positions) into 4 quarters that keep groups together and
    are stratified by k-bin, and return the nested subsets of 1, 2, 3 and 4 quarters.
    Each subset is sorted, so the 4-quarter subset is exactly `tr` (same rows, same order as in
    the main run; needed for the 100% check).
    """
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # the k >= 20 bin has only ~5 datasets per training part, so sklearn may warn that a bin
        # has few members; it is still spread over the quarters as evenly as possible
        warnings.simplefilter("ignore", UserWarning)
        quarters = [tr[q] for _, q in splitter.split(tr, bins[tr], groups[tr])]
    return [np.sort(np.concatenate(quarters[:j])) for j in QUARTERS]


def run(df, folds, max_repeats):
    """For every (repeat, fold, draw, fraction): screen + tune + fit RF D on the subsample,
    and fit the median baseline on the same subsample; score both on the full test part.
    """
    X = df[
        p4.STATIC8 + p4.PROBE16
    ]  # model inputs only; bookkeeping never reaches a model
    y = df[ev.TARGET].to_numpy(dtype=float)
    groups = df[ev.GROUP].to_numpy()
    bins = ev.k_bin(df[ev.TARGET]).to_numpy()  # the protocol's 6 k-bins (§2.2)
    spec = p4.FEATURE_SETS[FEATURE_SET]
    rows = []
    start = time.time()

    for repeat, fold, tr, te in ev.iter_splits(df, folds):
        if repeat >= max_repeats:
            break
        for draw in range(N_DRAWS):
            subsets = nested_subsets(
                tr, bins, groups, seed=1000 * repeat + 10 * fold + draw
            )
            for quarters, sub in zip(QUARTERS, subsets):
                if quarters == 4 and draw > 0:
                    continue  # 100% = the full training part: identical for every draw
                passed, _ = p4.screen(
                    df.iloc[sub], spec["screened"]
                )  # training data only
                features = spec["base"] + passed
                est = p4.make_estimator(MODEL, features)
                est.fit(X.iloc[sub], y[sub], groups=groups[sub])
                base = ts.MedianBaseline().fit(X.iloc[sub], y[sub])
                for config, pred, used in [
                    (CONFIG, est.predict(X.iloc[te]), features),
                    ("median_baseline", base.predict(X.iloc[te]), []),
                ]:
                    pred = ev.clip_k(pred)
                    rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "draw": draw,
                            "fraction": quarters / 4,
                            "config": config,
                            "mae": ev.mae(y[te], pred),
                            "within1": ev.within_one(y[te], pred),
                            "n_train": len(sub),
                            "n_test": len(te),
                            "features": json.dumps(used),
                        }
                    )
        if fold == ev.N_SPLITS - 1:
            print(f"repeat {repeat} done ({time.time() - start:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def summarize_curve(scores):
    """Mean ± sd over the repeats per fraction (draws, then folds averaged first, as in §2.6)."""
    per_fold = scores.groupby(["config", "fraction", "repeat", "fold"], as_index=False)[
        ["mae", "within1", "n_train", "n_test"]
    ].mean()  # average over the draws
    rows = []
    for (config, fraction), part in per_fold.groupby(["config", "fraction"]):
        mae_mean, mae_sd = ev.summarize(part, "mae")
        w1_mean, w1_sd = ev.summarize(part, "within1")
        rows.append(
            {
                "config": config,
                "fraction": fraction,
                "n_train_mean": part["n_train"].mean(),
                "mae_mean": mae_mean,
                "mae_sd": mae_sd,
                "within1_mean": w1_mean,
                "within1_sd": w1_sd,
            }
        )
    return pd.DataFrame(rows), per_fold


def decide(per_fold, summary):
    """Apply the plateau rule and the additional condition of §6 to the main meta-learner."""
    rf = per_fold[per_fold["config"] == CONFIG].set_index(["repeat", "fold"])
    m75 = rf[rf["fraction"] == 0.75]["mae"]
    m100 = rf[rf["fraction"] == 1.0]["mae"].loc[m75.index]
    full = rf[rf["fraction"] == 1.0]
    test = (
        ev.corrected_ttest(  # same correction as §2.7, with the full training-part size
            m100 - m75, n_train=full["n_train"].mean(), n_test=full["n_test"].mean()
        )
    )
    s = summary[summary["config"] == CONFIG].set_index("fraction")
    gain = (s.at[0.75, "mae_mean"] - s.at[1.0, "mae_mean"]) / s.at[0.75, "mae_mean"]
    plateau = bool(gain < PLATEAU_GAIN and test["p"] >= 0.05)

    hyp = pd.read_csv(MAIN_TESTS).set_index("test")
    beats_median = bool(hyp.at["H2: vs median baseline", "significant_improvement"])
    if not plateau:
        conclusion = (
            "not leveled off: more datasets may still improve the main meta-learner"
        )
    elif beats_median:
        conclusion = (
            "leveled off and the model beats the median: 100 datasets are enough (O1)"
        )
    else:
        conclusion = (
            "leveled off, but the model does not beat the median baseline: the features lack "
            "signal, which more datasets would not fix (§6, additional condition)"
        )
    return {
        "config": CONFIG,
        "mae_75": s.at[0.75, "mae_mean"],
        "mae_100": s.at[1.0, "mae_mean"],
        "relative_gain_75_to_100": gain,
        "step_test_75_to_100": test,  # mean_diff < 0 = 100% better than 75%
        "plateau": plateau,
        "beats_median_at_100": beats_median,
        "conclusion": conclusion,
    }


def check_main_run(scores):
    """At 100% the subsample is the full training part: RF D and the median baseline must
    reproduce the main run's fold scores (results/phase4/fold_scores.csv) up to 1e-9."""
    ref = pd.read_csv(MAIN_SCORES).set_index(["config", "repeat", "fold"])["mae"]
    full = scores[scores["fraction"] == 1.0]
    rows = []
    for config in [CONFIG, "median_baseline"]:
        ours = full[full["config"] == config].set_index(["repeat", "fold"])["mae"]
        theirs = ref.loc[config].loc[ours.index]
        rows.append(
            {
                "config": config,
                "folds": len(ours),
                "max_abs_diff": float((ours - theirs).abs().max()),
            }
        )
    table = pd.DataFrame(rows)
    table["ok"] = table["max_abs_diff"] <= 1e-9
    return table


def plot_curve(summary, path, n_repeats):
    """MAE vs. number of training datasets for RF D and the median baseline (mean ± sd over repeats).
    Identity by color, marker AND a direct label at the line end (not color alone)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ts._style(ax)
    styles = {
        CONFIG: (ts.AQUA, "o", "-", "Random Forest, set D (main meta-learner)"),
        "median_baseline": (ts.INK_2, "s", "--", "median baseline"),
    }
    for config, (color, marker, line, label) in styles.items():
        s = summary[summary["config"] == config].sort_values("fraction")
        ax.errorbar(
            s["n_train_mean"],
            s["mae_mean"],
            yerr=s["mae_sd"],
            fmt=marker + line,
            color=color,
            markersize=8,
            linewidth=2,
            elinewidth=1.5,
            capsize=0,
        )
        last = s.iloc[-1]
        ax.annotate(
            label,
            (last["n_train_mean"], last["mae_mean"]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=ts.INK,
        )
    ax.set_xlabel(
        "training datasets (25 / 50 / 75 / 100% of a training part)", color=ts.INK_2
    )
    ax.set_ylabel("MAE on the k scale", color=ts.INK_2)
    xmax = summary["n_train_mean"].max()
    ax.set_xlim(0, xmax * 1.9)  # room for the direct labels
    ax.set_xticks([20, 40, 60, 80])  # the four subsample sizes (≈ training datasets)
    ax.set_title(
        f"Learning curve (fixed test parts; 5 draws per fold; {n_repeats} repeat(s); mean ± sd)",
        loc="left",
        fontsize=11,
        color=ts.INK,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


RESULT_FILES = [
    "lc_scores.csv",
    "lc_summary.csv",
    "lc_decision.json",
    "lc_check.csv",
    "fig_learning_curve.png",
    "run_info.json",
]


def main():
    parser = argparse.ArgumentParser(description="Phase 4 learning curve (O1).")
    parser.add_argument("--repeats", type=int, default=ev.N_REPEATS)
    args = parser.parse_args()
    full_run = args.repeats >= ev.N_REPEATS
    out_dir = ROOT / "outputs" / RUN_NAME
    results_dir = ROOT / "results" / RUN_NAME

    df = p4.load_tables()
    folds = ev.load_folds(df)
    print(f"Learning curve for {CONFIG}: {len(df)} datasets, {args.repeats} repeat(s)")

    t0 = time.time()
    scores = run(df, folds, args.repeats)
    runtime = time.time() - t0
    summary, per_fold = summarize_curve(scores)
    check = check_main_run(scores)
    valid = bool(check["ok"].all())
    decision = decide(per_fold, summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "lc_scores.csv", index=False)
    summary.to_csv(out_dir / "lc_summary.csv", index=False, float_format="%.4f")
    check.to_csv(out_dir / "lc_check.csv", index=False)
    (out_dir / "lc_decision.json").write_text(json.dumps(decision, indent=2))
    plot_curve(summary, out_dir / "fig_learning_curve.png", args.repeats)
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "design": "docs/metalearner_methodology.md §6 and §12.8",
        "full_run": full_run,
        "valid_main_run_check": valid,
        "repeats": args.repeats,
        "draws": N_DRAWS,
        "runtime_s": round(runtime),
        "metafeatures_sha256": ts.sha256(ev.METAFEATURES_CSV),
        "probe_features_sha256": ts.sha256(p4.PROBE_CSV),
        "cv_folds_sha256": ts.sha256(ev.FOLDS_CSV),
        "main_fold_scores_sha256": ts.sha256(MAIN_SCORES),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    if full_run and valid:
        results_dir.mkdir(parents=True, exist_ok=True)
        for f in RESULT_FILES:
            shutil.copy(out_dir / f, results_dir / f)
        print(f"Copied results to {results_dir}")
    elif full_run:
        print(
            f"INVALID RUN: 100% does not reproduce the main run - results/{RUN_NAME} NOT written."
        )
    else:
        print(f"Test run: results/{RUN_NAME} was NOT updated.")

    pd.set_option("display.width", 200)
    print(
        f"\nFinished in {runtime / 60:.1f} min.\n\nCheck against the main run (100%):"
    )
    print(check.to_string(index=False))
    print("\nLearning curve (mean over repeats):")
    print(summary.round(3).to_string(index=False))
    print("\nDecision (§6):")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
