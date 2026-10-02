"""
posthoc_k10.py - POST-HOC variant: the Phase 4 meta-learner on the datasets with k <= 10
(methodology §13.1, branch feature/posthoc-variants).

POST-HOC: designed on 2026-10-02, AFTER the Phase 4 results (§12.10-12.13) were known. The results
never replace the pre-registered Phase 4 results; they answer a "what if" question: does the
picture change if the meta-learner only has to handle datasets with at most 10 classes?

Why k <= 10: the 6 datasets with k >= 20 cause 43% of the median baseline's total error (§1), and
the probe k-grid is dense only for small k. Limiting the range removes this tail.

What it does (no new modelling code; everything is train_phase4.py, unchanged):
  1. Loads the Phase 4 inputs (train_phase4.load_tables) and keeps the 90 datasets with
     k_true <= 10 (75 tabular, 15 image). Choosing the datasets by their true k defines the task
     ("the user knows that k is at most 10"); the models still never see k of a test dataset.
  2. Uses the frozen folds of data/cv_folds.csv for these 90 datasets (the mechanism of rule P7:
     every dataset keeps its fold, the 10 others are skipped; test parts have 16-19 datasets).
  3. Runs train_phase4.run unchanged: screening inside each training part (§12.2), sets A-F x
     Ridge / k-NN / RF on log k with nested tuning, median baseline, direct estimators.
  4. Applies the Phase 4 tests unchanged (§12.6: Q1 and Q2, Holm within each). Reading, fixed
     before the run: the Phase 4 conclusion "the probe features do not improve the meta-learner"
     also holds for k <= 10 unless H1a (RF D vs RF C) is a significant improvement after Holm.
  5. Reference "trained on all 100": the saved Phase 4 main-run predictions, scored on exactly the
     same test datasets (k <= 10) in the same folds, compared with this run (corrected t-test,
     uncorrected p, reported only). It separates two effects: removing the large-k datasets from
     the TEST (lower MAE for every model) and from the TRAINING (the question of item 4).

Checks (the script stops if one fails): 90 / 75 / 15 datasets; the copied screening rule
reproduces §10.1 (on all 100); the saved main-run predictions reproduce
results/phase4/fold_scores.csv (max difference <= 1e-9); the reference uses exactly the test
datasets of this run. Results are copied to results/posthoc_k10/ only after a full run.

Usage:
    python posthoc_k10.py               # full run (10 repeats; ~25 min on the PC)
    python posthoc_k10.py --repeats 1   # quick test run (results/ is not touched)
"""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

import evaluation as ev  # frozen protocol (unchanged)
import train_phase4 as p4  # Phase 4 inputs, screening, models, tests (unchanged)
import train_static as ts  # summary, breakdown and sha256 helpers (unchanged)

ROOT = Path(__file__).resolve().parent
RUN_NAME = "posthoc_k10"  # output folders outputs/posthoc_k10/ and results/posthoc_k10/
MAX_K = 10  # §13.1: keep the datasets with k_true <= 10
EXPECTED = {"all": 90, "tabular": 75, "image": 15}  # counts in data/metafeatures.csv

# Phase 4 main run (§12.10): predictions are git-ignored (on the PC), fold scores are committed.
MAIN_PREDICTIONS = ROOT / "outputs" / "phase4" / "predictions.csv"
MAIN_SCORES = ROOT / "results" / "phase4" / "fold_scores.csv"

# Configurations compared "trained on k <= 10" vs "trained on all 100" (step 5).
REFERENCE_CONFIGS = [
    "median_baseline",
    "knn__B__logk",
    "rf__B__logk",
    "rf__C__logk",
    "rf__D__logk",
]


# ----------------------------------------------------------------------------------------
# 1. The subset
# ----------------------------------------------------------------------------------------
def select_subset(df):
    """The datasets with k_true <= MAX_K. The index is reset to 0..n-1 because
    train_phase4.run takes row POSITIONS from evaluation.iter_splits and also looks rows up by
    index label (df.at[i, name]); both must mean the same row."""
    sub = df[df[ev.TARGET] <= MAX_K].reset_index(drop=True)
    counts = {
        "all": len(sub),
        "tabular": int(sub["tabular"].sum()),
        "image": int((~sub["tabular"]).sum()),
    }
    if counts != EXPECTED:
        raise SystemExit(
            f"subset has {counts}, expected {EXPECTED} - did an input change? Stop and discuss."
        )
    return sub


# ----------------------------------------------------------------------------------------
# 2. Reference: the Phase 4 main run, scored on the k <= 10 test datasets
# ----------------------------------------------------------------------------------------
def fold_scores_from_predictions(preds):
    """MAE and within ±1 per (config, repeat, fold), recomputed from saved test predictions
    with the protocol's metric functions (§2.6)."""
    rows = []
    for (config, repeat, fold), part in preds.groupby(
        ["config", "repeat", "fold"], sort=False
    ):
        rows.append(
            {
                "config": config,
                "repeat": repeat,
                "fold": fold,
                "mae": ev.mae(part["k_true"], part["k_pred"]),
                "within1": ev.within_one(part["k_true"], part["k_pred"]),
                "n_test": len(part),
            }
        )
    return pd.DataFrame(rows)


def load_main_predictions(repeats):
    """Saved Phase 4 main-run predictions for the given repeats. Check: recomputed on all 100
    datasets they must reproduce the committed fold scores (otherwise the reference would come
    from a different run). Done BEFORE the long run, so a missing file stops it early.
    """
    if not MAIN_PREDICTIONS.exists():
        raise SystemExit(
            f"{MAIN_PREDICTIONS} not found - it is written by train_phase4.py"
        )
    preds = pd.read_csv(MAIN_PREDICTIONS)
    preds = preds[preds["repeat"].isin(repeats)]
    ours = fold_scores_from_predictions(preds).set_index(["config", "repeat", "fold"])
    ref = pd.read_csv(MAIN_SCORES).set_index(["config", "repeat", "fold"])
    ref = ref[ref.index.get_level_values("repeat").isin(repeats)]
    if set(ours.index) != set(ref.index):
        raise SystemExit("main-run predictions and fold scores cover different folds")
    worst = float((ours["mae"] - ref.loc[ours.index, "mae"]).abs().max())
    if worst > 1e-9:
        raise SystemExit(
            f"main-run predictions do not reproduce {MAIN_SCORES} (max diff {worst:.2g})"
        )
    print(f"Main-run predictions check: OK (max difference {worst:.1e})")
    return preds, worst


def reference_full_pool(scores, preds, main_preds):
    """
    Each configuration in REFERENCE_CONFIGS twice, on the SAME test datasets and folds:
    trained on k <= 10 (this run) and trained on all 100 (Phase 4 main run).
    mean_diff = MAE(k <= 10) - MAE(all 100); < 0 means limiting the training range helps.
    Corrected t-test (§2.7) with the sizes of this run; uncorrected p, reported only.
    """
    # Check: the main run's test predictions on k <= 10 datasets are exactly this run's test rows.
    key = [ev.KEY, "repeat", "fold"]
    ours_rows = preds[preds["config"] == "median_baseline"][key]
    main_rows = main_preds[
        (main_preds["config"] == "median_baseline")
        & (main_preds[ev.KEY].isin(set(ours_rows[ev.KEY])))
    ][key]
    if set(map(tuple, ours_rows.to_numpy())) != set(map(tuple, main_rows.to_numpy())):
        raise SystemExit("reference: test datasets differ from this run")

    main_k10 = fold_scores_from_predictions(
        main_preds[main_preds[ev.KEY].isin(set(ours_rows[ev.KEY]))]
    )
    rows = []
    for c in REFERENCE_CONFIGS:
        a = scores[scores["config"] == c].set_index(["repeat", "fold"])
        b = main_k10[main_k10["config"] == c].set_index(["repeat", "fold"]).loc[a.index]
        test = ev.corrected_ttest(
            a["mae"] - b["mae"], n_train=a["n_train"].mean(), n_test=a["n_test"].mean()
        )
        rows.append(
            {
                "config": c,
                "mae_trained_k10": ev.summarize(a.reset_index(), "mae")[0],
                "mae_trained_all100": ev.summarize(b.reset_index(), "mae")[0],
                "within1_trained_k10": ev.summarize(a.reset_index(), "within1")[0],
                "within1_trained_all100": ev.summarize(b.reset_index(), "within1")[0],
                **test,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 3. Reading (fixed before the run, §13.1)
# ----------------------------------------------------------------------------------------
def reading(tests):
    """The Phase 4 probe claim rule (§12.6) applied to k <= 10."""
    h1a = tests.set_index("test").loc["H1a: probe effect (D vs C)"]
    if h1a["significant_improvement"]:
        return (
            "PROBE EFFECT for k <= 10: RF D significantly better than RF C after Holm "
            "(post-hoc finding, to be confirmed on new data)"
        )
    return "Phase 4 conclusion HOLDS for k <= 10: no significant probe effect (H1a)"


# ----------------------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------------------
RESULT_FILES = [
    "summary.csv",
    "hypotheses.csv",
    "secondary_tests.csv",
    "reference_full_pool.csv",
    "fold_scores.csv",
    "hyperparameters.csv",
    "screening.csv",
    "selection_frequency.csv",
    "screen_all90.csv",
    "breakdown.csv",
    "gain_share.csv",
    "run_info.json",
]


def main():
    parser = argparse.ArgumentParser(
        description="POST-HOC: Phase 4 meta-learner on k <= 10."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=ev.N_REPEATS,
        help="number of repeats to run (default: all 10; fewer = test run)",
    )
    args = parser.parse_args()
    full_run = args.repeats >= ev.N_REPEATS
    out_dir = ROOT / "outputs" / RUN_NAME
    results_dir = ROOT / "results" / RUN_NAME

    df = p4.load_tables()  # Phase 4 inputs, checked as in Phase 4
    p4.check_screening_copy(df)  # on all 100: inputs are the committed ones
    sub = select_subset(df)
    folds = ev.load_folds(sub)  # frozen folds; prints the P7 note (10 skipped)
    main_preds, main_check = load_main_predictions(list(range(args.repeats)))
    print(
        f"POST-HOC k <= {MAX_K}: {len(sub)} datasets ({EXPECTED['tabular']} tabular, "
        f"{EXPECTED['image']} image), {args.repeats} repeat(s)"
    )

    t0 = time.time()
    preds, scores, params, screening = p4.run(sub, folds, args.repeats)
    runtime = time.time() - t0

    summary = ts.make_summary(scores, p4.config_list())  # + corrected t-test vs median
    hypotheses = p4.test_hypotheses(scores)  # Q1 / Q2 with Holm, as §12.6
    secondary = p4.secondary_tests(scores)
    reference = reference_full_pool(scores, preds, main_preds)
    _, all90 = p4.screen(sub, p4.STATIC6 + p4.PROBE16)  # descriptive only, never used
    all90 = all90.rename(columns={"passed": "would_pass_on_all_90"})

    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    scores.to_csv(out_dir / "fold_scores.csv", index=False)
    params.to_csv(out_dir / "hyperparameters.csv", index=False)
    screening.to_csv(out_dir / "screening.csv", index=False, float_format="%.4f")
    p4.selection_frequency(screening).to_csv(
        out_dir / "selection_frequency.csv", index=False, float_format="%.4f"
    )
    all90.to_csv(out_dir / "screen_all90.csv", index=False, float_format="%.4f")
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    hypotheses.to_csv(out_dir / "hypotheses.csv", index=False, float_format="%.4f")
    secondary.to_csv(out_dir / "secondary_tests.csv", index=False, float_format="%.4f")
    reference.to_csv(
        out_dir / "reference_full_pool.csv", index=False, float_format="%.4f"
    )
    ts.make_breakdown(preds, sub).to_csv(
        out_dir / "breakdown.csv", index=False, float_format="%.4f"
    )
    p4.gain_share(preds, sub).to_csv(
        out_dir / "gain_share.csv", index=False, float_format="%.4f"
    )

    verdict = reading(hypotheses)
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "status": "POST-HOC (designed after the Phase 4 results)",
        "design": "docs/metalearner_methodology.md §13.1",
        "max_k": MAX_K,
        "datasets": EXPECTED,
        "full_run": full_run,
        "repeats": args.repeats,
        "runtime_s": round(runtime),
        "main_predictions_check_max_diff": main_check,
        "reading": verdict,
        "metafeatures_sha256": ts.sha256(ev.METAFEATURES_CSV),
        "probe_features_sha256": ts.sha256(p4.PROBE_CSV),
        "t1_candidates_sha256": ts.sha256(p4.T1_CSV),
        "cv_folds_sha256": ts.sha256(ev.FOLDS_CSV),
        "main_predictions_sha256": ts.sha256(MAIN_PREDICTIONS),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    # Only a complete run goes into the committed results folder (all checks above stop the
    # script if they fail, so reaching this point means they passed).
    if full_run:
        results_dir.mkdir(parents=True, exist_ok=True)
        for f in RESULT_FILES:
            shutil.copy(out_dir / f, results_dir / f)
        print(f"Copied results to {results_dir}")
    else:
        print(f"Test run: results/{RUN_NAME} was NOT updated.")

    pd.set_option("display.width", 200)
    print(f"\nFinished in {runtime / 60:.1f} min.  POST-HOC, k <= {MAX_K}.")
    print("\nPhase 4 tests on k <= 10 (§12.6 rules; mean_diff < 0 = RF D better):")
    cols = ["family", "test", "mean_diff", "ci_low", "ci_high", "p", "p_holm"]
    print(
        hypotheses[cols + ["significant_improvement"]].round(3).to_string(index=False)
    )
    print(f"\nReading: {verdict}")
    print(
        "\nTrained on k <= 10 vs trained on all 100, same test datasets "
        "(mean_diff < 0 = limiting the training range helps; uncorrected):"
    )
    print(reference.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
