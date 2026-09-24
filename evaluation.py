"""
evaluation.py - the frozen evaluation protocol of the meta-learner (Phase 2, reused in Phase 4).

Everything that must be IDENTICAL between "static-only" (Phase 2) and "static + probe"
(Phase 4) lives in this one file:
  - which columns are model inputs, which column is the target, which are bookkeeping
  - how the datasets are split into training and test parts (grouped, stratified, 5 x 10)
  - the metrics (MAE on the k scale, share within +-1) and how they are averaged
  - the statistical test for "is model A really better than model B?"

The splits themselves are saved ONCE to data/cv_folds.csv and committed to git. Every later
run (also Phase 4) reads that file instead of re-creating the splits, so the splits can never
change silently (e.g. after a scikit-learn update). Details and reasoning:
docs/metalearner_methodology.md, section 2.

Usage:
    python evaluation.py --make-folds     # create data/cv_folds.csv (only once!)
    python evaluation.py --check          # check the fold file against the meta-feature table
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

# ----------------------------------------------------------------------------------------
# 1. Files and columns
# ----------------------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
METAFEATURES_CSV = DATA_DIR / "metafeatures.csv"  # frozen Phase 1 output (D10)
FOLDS_CSV = DATA_DIR / "cv_folds.csv"  # frozen splits (created by --make-folds)

# The ONLY columns the model may see (O3: the final static set of 8 meta-features).
FEATURES = [
    "log_n",
    "log_p",
    "skew_mean",
    "skew_sd",
    "kurt_min",
    "kurt_q3",
    "cor_mean",
    "conc_mean",
]
TARGET = "k_true"  # ground truth = number of classes present (D4)
KEY = "name"  # row key: unique per dataset (image variants share a dataset_id)
GROUP = "group"  # all variants of one image source share a group (M4)

# Bookkeeping columns: kept in the table for traceability, NEVER model input.
BOOKKEEPING = [
    "dataset_id",
    "name",
    "source",
    "group",
    "classes",
    "n",
    "p_raw",
    "p_used",
    "runtime_s",
]

# ----------------------------------------------------------------------------------------
# 2. Split settings (protocol decisions P1 and P2)
# ----------------------------------------------------------------------------------------
N_SPLITS = 5  # 5 parts: train on 4 (~80 datasets), test on 1 (~20 datasets)
N_REPEATS = 10  # the whole 5-part split is repeated 10 times with different shuffles
SEEDS = list(range(N_REPEATS))  # repeat r uses random_state = r (0..9)
INNER_SPLITS = 5  # parts used INSIDE a training part to tune hyperparameters (P4)

# Stratification bins (P1): every test part should contain a similar mix of small and
# large k. Same brackets as the pool (D10), except that k >= 20 gets its own bin, so
# that each test part holds 1-2 of the six datasets with very large k.
# pd.cut uses right-closed intervals: (1,2] (2,3] (3,5] (5,9] (9,19] (19,50]
STRAT_EDGES = [1, 2, 3, 5, 9, 19, 50]
STRAT_LABELS = ["k2", "k3", "k4-5", "k6-9", "k10-19", "k20-50"]

# Metrics (P5): predictions below 2 make no sense (k >= 2 everywhere in the pool)
MIN_K = 2.0


# ----------------------------------------------------------------------------------------
# 3. Loading the meta-feature table
# ----------------------------------------------------------------------------------------
def load_metafeatures(path=METAFEATURES_CSV):
    """Read the meta-feature table and check that it is usable as model input."""
    df = pd.read_csv(path)

    missing_cols = [c for c in FEATURES + [TARGET, KEY, GROUP] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path}: missing columns {missing_cols}")
    if df[KEY].duplicated().any():
        raise ValueError(f"{path}: column '{KEY}' must be unique (it is the row key)")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(f"{path}: missing values in the feature or target columns")
    if (df[TARGET] < MIN_K).any():
        raise ValueError(f"{path}: found k < {MIN_K}")

    # Safety net: a feature column must never also be a bookkeeping column.
    overlap = set(FEATURES) & set(BOOKKEEPING + [TARGET])
    assert not overlap, f"columns used as feature AND bookkeeping/target: {overlap}"
    return df.reset_index(drop=True)


def k_bin(k):
    """Stratification bin of each k value (see STRAT_EDGES)."""
    return pd.cut(k, bins=STRAT_EDGES, labels=STRAT_LABELS).astype(str)


# ----------------------------------------------------------------------------------------
# 4. Creating, saving and loading the splits
# ----------------------------------------------------------------------------------------
def make_folds(df):
    """
    Create the 5 x 10 split assignment.

    StratifiedGroupKFold does two things at once:
      - "Group": all rows with the same `group` land in the same part, so e.g. the four
        MNIST versions (k = 2, 3, 5, 10) are never split between training and test.
      - "Stratified": each part gets a similar mix of the k bins above.
    shuffle=True + random_state=seed gives a different (but reproducible) split per repeat.

    Returns a table with one row per (dataset, repeat): name, group, repeat, fold.
    """
    bins = k_bin(df[TARGET])
    rows = []
    for repeat, seed in enumerate(SEEDS):
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=seed
        )
        for fold, (_, test_idx) in enumerate(
            splitter.split(df, bins, groups=df[GROUP])
        ):
            for i in test_idx:
                rows.append(
                    {
                        KEY: df.at[i, KEY],
                        GROUP: df.at[i, GROUP],
                        "repeat": repeat,
                        "fold": fold,
                    }
                )
    folds = pd.DataFrame(rows).sort_values(["repeat", "fold", KEY])
    return folds.reset_index(drop=True)


def check_folds(folds, df):
    """
    Check that the fold file fits the meta-feature table. Raises an error if not.

    - every dataset in the table must be in the fold file
    - in every repeat, every dataset is tested exactly once
    - a group is never split across parts (no leakage between training and test)
    The table may contain FEWER datasets than the fold file (rule P7 for Phase 4: if probe
    runs fail on some datasets, those rows are dropped and the remaining ones keep their
    folds). That case only prints a note.
    """
    table_names = set(df[KEY])
    fold_names = set(folds[KEY])
    not_in_folds = table_names - fold_names
    if not_in_folds:
        raise ValueError(
            f"{len(not_in_folds)} datasets have no fold: {sorted(not_in_folds)[:5]} ..."
        )
    dropped = fold_names - table_names
    if dropped:
        print(
            f"Note: {len(dropped)} datasets of the fold file are not in the table "
            f"(evaluated on the remaining {len(table_names)})."
        )

    for repeat, part in folds.groupby("repeat"):
        if part[KEY].duplicated().any() or set(part[KEY]) != fold_names:
            raise ValueError(f"repeat {repeat}: every dataset must appear exactly once")
        folds_per_group = part.groupby(GROUP)["fold"].nunique()
        if (folds_per_group > 1).any():
            raise ValueError(f"repeat {repeat}: a group is split across folds")
    if sorted(folds["repeat"].unique()) != list(range(N_REPEATS)):
        raise ValueError("fold file does not contain the expected repeats 0..9")


def load_folds(df, path=FOLDS_CSV):
    """Read data/cv_folds.csv and check it against the table `df`."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found - create it once with: python evaluation.py --make-folds"
        )
    folds = pd.read_csv(path)
    check_folds(folds, df)
    return folds


def iter_splits(df, folds):
    """
    Loop over all 50 (repeat, fold) combinations.

    Yields (repeat, fold, train_idx, test_idx), where the indices are row positions in
    `df`. Datasets that are in the fold file but not in `df` are simply skipped (P7).
    """
    position = pd.Series(np.arange(len(df)), index=df[KEY])
    folds = folds[folds[KEY].isin(position.index)]
    for repeat in sorted(folds["repeat"].unique()):
        in_repeat = folds[folds["repeat"] == repeat]
        for fold in sorted(in_repeat["fold"].unique()):
            is_test = in_repeat["fold"] == fold
            test_idx = position[in_repeat.loc[is_test, KEY]].to_numpy()
            train_idx = position[in_repeat.loc[~is_test, KEY]].to_numpy()
            yield repeat, fold, np.sort(train_idx), np.sort(test_idx)


# ----------------------------------------------------------------------------------------
# 5. Metrics (P5)
# ----------------------------------------------------------------------------------------
def clip_k(pred):
    """Predictions are clipped at k = 2 (a k below 2 is meaningless)."""
    return np.maximum(np.asarray(pred, dtype=float), MIN_K)


def mae(k_true, pred):
    """Mean absolute error on the ORIGINAL k scale, on the continuous (clipped) prediction."""
    return float(np.mean(np.abs(np.asarray(k_true, dtype=float) - clip_k(pred))))


def within_one(k_true, pred):
    """Share of datasets whose ROUNDED (clipped) prediction is at most 1 away from k."""
    rounded = np.round(clip_k(pred))
    return float(np.mean(np.abs(rounded - np.asarray(k_true, dtype=float)) <= 1))


def mae_scorer(estimator, X, y):
    """Same MAE as above, in the form GridSearchCV expects (higher = better -> negative)."""
    return -mae(y, estimator.predict(X))


def summarize(fold_scores, metric):
    """
    Aggregate per-fold scores the way the protocol prescribes:
    first average the 5 folds of each repeat, then mean and sd over the 10 repeat averages.
    (The sd over single folds would mostly show which datasets happened to be in a fold.)
    """
    per_repeat = fold_scores.groupby("repeat")[metric].mean()
    return float(per_repeat.mean()), float(per_repeat.std(ddof=1))


# ----------------------------------------------------------------------------------------
# 6. Statistical comparison of two models (P6)
# ----------------------------------------------------------------------------------------
def corrected_ttest(diffs, n_train, n_test, alpha=0.05):
    """
    Corrected resampled t-test (Nadeau & Bengio, 2003) on paired per-fold differences.

    `diffs` = score(model A) - score(model B) for each of the 50 (repeat, fold) pairs,
    computed on IDENTICAL folds. A plain t-test would assume the 50 differences are
    independent, but the training parts overlap heavily, so it finds "significant"
    differences far too often (51% instead of 5% in our simulation). The correction
    enlarges the variance by the factor (1/J + n_test/n_train).

    Returns the mean difference, its corrected 95% interval, t and the two-sided p-value.
    """
    diffs = np.asarray(diffs, dtype=float)
    J = len(diffs)
    mean_diff = diffs.mean()
    var_corrected = (1.0 / J + n_test / n_train) * diffs.var(ddof=1)
    se = np.sqrt(var_corrected)
    t = mean_diff / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    half_width = stats.t.ppf(1 - alpha / 2, df=J - 1) * se
    return {
        "mean_diff": float(mean_diff),
        "ci_low": float(mean_diff - half_width),
        "ci_high": float(mean_diff + half_width),
        "t": float(t),
        "p": float(p),
    }


# ----------------------------------------------------------------------------------------
# 7. Command line
# ----------------------------------------------------------------------------------------
def describe(folds, df):
    """Print how balanced the test parts are (for a quick visual check)."""
    merged = folds.merge(df[[KEY, TARGET]], on=KEY)
    merged["bin"] = k_bin(merged[TARGET])
    sizes = merged.groupby(["repeat", "fold"]).size()
    big = merged[merged[TARGET] >= 20].groupby(["repeat", "fold"]).size()
    big = big.reindex(sizes.index, fill_value=0)
    print(f"{merged['repeat'].nunique()} repeats x {merged['fold'].nunique()} folds")
    print(f"test-part size: {sizes.min()}-{sizes.max()} datasets")
    print(f"datasets with k >= 20 per test part: {big.min()}-{big.max()}")
    per_bin = merged.groupby(["repeat", "fold", "bin"]).size().unstack(fill_value=0)
    print("datasets per bin in a test part (min-max):")
    for b in STRAT_LABELS:
        if b in per_bin:
            print(f"  {b:7s} {per_bin[b].min()}-{per_bin[b].max()}")


def main():
    parser = argparse.ArgumentParser(description="Frozen evaluation protocol.")
    parser.add_argument(
        "--make-folds", action="store_true", help="create data/cv_folds.csv (once)"
    )
    parser.add_argument("--check", action="store_true", help="check data/cv_folds.csv")
    parser.add_argument(
        "--force", action="store_true", help="allow overwriting an existing fold file"
    )
    args = parser.parse_args()

    df = load_metafeatures()
    if args.make_folds:
        if FOLDS_CSV.exists() and not args.force:
            raise SystemExit(
                f"{FOLDS_CSV} already exists. The splits are frozen - they must not "
                "change once results were produced. Use --force only if you are sure."
            )
        folds = make_folds(df)
        check_folds(folds, df)
        folds.to_csv(FOLDS_CSV, index=False)
        print(f"Saved {FOLDS_CSV} ({len(folds)} rows).")
        describe(folds, df)
    elif args.check:
        folds = load_folds(df)
        print(f"{FOLDS_CSV} is consistent with {METAFEATURES_CSV.name}.")
        describe(folds, df)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
