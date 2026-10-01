"""
train_phase4.py - Phase 4: static + probe meta-learner (methodology §12, decision D13).

Everything below was fixed before any Phase 4 result (docs/metalearner_methodology.md §12).

What it does:
  1. Reads the frozen static table (data/metafeatures.csv), the probe table
     (data/probe_features.csv), the classical T1 estimators (results/t1_classical/candidates.csv)
     and the frozen splits (data/cv_folds.csv), through evaluation.py (unchanged).
  2. For each of the 50 (repeat, fold) splits:
       - screens the candidate features on the TRAINING part only (§12.2),
       - trains 6 feature sets (A-F) x 3 models (Ridge, k-NN, Random Forest) on log k, with the
         Phase 2 grids and nested tuning (§12.3),
       - scores the median baseline and the direct estimators (§12.5).
  3. Writes scores, predictions, chosen hyperparameters and selected features, a summary with the
     corrected t-test against the baseline, the pre-registered tests with Holm correction (§12.6),
     the secondary analyses (§12.7) and two figures.
  4. Checks that sets A and B reproduce the Phase 2 fold scores (§12.3). Only a FULL and VALID run
     is copied to results/phase4/.

Usage:
    python train_phase4.py               # full run (10 repeats)
    python train_phase4.py --repeats 1   # quick test run (results/ is not touched)

Not in this script (separate steps after the main run, §12.7-12.8): the additional error
measures T3 (computed from outputs/phase4/predictions.csv) and the learning curve.
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
import matplotlib.ticker
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import evaluation as ev  # frozen protocol (unchanged)
import train_static as ts  # Phase 2 models, grids, transforms and helpers (unchanged)

ROOT = Path(__file__).resolve().parent
RUN_NAME = "phase4"  # output folders outputs/phase4/ and results/phase4/

# ----------------------------------------------------------------------------------------
# 1. Input files
# ----------------------------------------------------------------------------------------
PROBE_CSV = (
    ROOT / "data" / "probe_features.csv"
)  # Phase 3 output: name + 16 probe features
T1_CSV = ROOT / "results" / "t1_classical" / "candidates.csv"  # §10.2: best k per index
PHASE2_SCORES = (
    ROOT / "results" / "phase2" / "fold_scores.csv"
)  # for the A/B check (§12.3)
SCREEN_REF = (
    ROOT / "results" / "pymfe_screen" / "screen_result.csv"
)  # §10.1, for the copy check

# ----------------------------------------------------------------------------------------
# 2. Feature lists and feature sets (§12.3)
# ----------------------------------------------------------------------------------------
SIZE2 = ["log_n", "log_p"]
STATIC8 = list(ev.FEATURES)  # the frozen static set (O3)
STATIC6 = [c for c in STATIC8 if c not in SIZE2]  # the 6 distribution features

ALGORITHMS = ["DEC", "IDEC"]  # as in run_probes.py
PROBE_CANDIDATES = ["G1", "G2", "G3", "G4", "L1", "O1", "O2", "O3"]  # §11.3
PROBE16 = [f"{c}_{a}" for c in PROBE_CANDIDATES for a in ALGORITHMS]  # file order

# §12.4: probe values on the k scale (>= 1 by construction) get log; O3 (number of emptied
# clusters, can be negative) gets the signed log. Both are parameter-free (no leakage).
PROBE_LOG_COLS = [f for f in PROBE16 if not f.startswith("O3")]
PROBE_SIGNED_LOG_COLS = [f for f in PROBE16 if f.startswith("O3")]

# §12.5: the probe candidates that ARE estimates of k (O3 is not) and the 4 T1 estimators.
DIRECT_PROBE = [f for f in PROBE16 if not f.startswith("O3")]  # 14, order = tie order
DIRECT_T1 = ["t1_best_k_sil", "t1_best_k_ch", "t1_best_k_db", "t1_best_k_bic"]

# "base" columns are always used; "screened" columns must pass the §10.1 rule in the training part.
FEATURE_SETS = {
    "A": {"base": STATIC8, "screened": []},  # Phase 2 reference
    "B": {"base": SIZE2, "screened": []},  # Phase 2 size-only model
    "C": {"base": SIZE2, "screened": STATIC6},  # static-only with the screening rule
    "D": {"base": SIZE2, "screened": STATIC6 + PROBE16},  # MAIN META-LEARNER
    "E": {"base": STATIC8 + PROBE16, "screened": []},  # robustness: no screening
    "F": {"base": PROBE16, "screened": []},  # probe features alone
}
SET_LABELS = {
    "A": "A: static 8",
    "B": "B: log n + log p",
    "C": "C: size + screened static",
    "D": "D: size + screened static & probe",
    "E": "E: static 8 + probe 16",
    "F": "F: probe 16 only",
}
PHASE2_EQUIVALENT = {
    "A": "static8",
    "B": "size2",
}  # same inputs in Phase 2 (§12.3 check)
MODELS = list(ts.MODELS)  # ridge, knn, rf - with the Phase 2 grids
TARGET = "logk"  # §12.3: log k only

# ----------------------------------------------------------------------------------------
# 3. Pre-registered tests (§12.6)
# ----------------------------------------------------------------------------------------
PRIMARY = "rf__D__logk"  # primary model (RF) with the main feature set (D)
ALPHA = 0.05
Q1_TESTS = [  # Holm over these 2
    ("H1a: probe effect (D vs C)", PRIMARY, "rf__C__logk"),
    ("H1b: vs Phase 2 reference (D vs A)", PRIMARY, "rf__A__logk"),
]
Q2_TESTS = [  # Holm over these 4
    ("H2: vs median baseline", PRIMARY, "median_baseline"),
    ("H2: vs size-only k-NN", PRIMARY, "knn__B__logk"),
    ("H2: vs best direct probe estimator", PRIMARY, "direct_best_probe"),
    ("H2: vs best classical estimator (T1)", PRIMARY, "direct_best_t1"),
]
# Secondary, uncorrected (§12.6, "all other comparisons"): the same contrasts for Ridge and
# k-NN, and the robustness check E vs A for all three models.
SECONDARY_TESTS = (
    [(f"{m}: D vs C", f"{m}__D__logk", f"{m}__C__logk") for m in ["ridge", "knn"]]
    + [(f"{m}: D vs A", f"{m}__D__logk", f"{m}__A__logk") for m in ["ridge", "knn"]]
    + [(f"{m}: E vs A", f"{m}__E__logk", f"{m}__A__logk") for m in MODELS]
)

# ----------------------------------------------------------------------------------------
# 4. Screening rule (§10.1, unchanged; applied inside each training part, §12.2)
# ----------------------------------------------------------------------------------------
# The two thresholds and partial_spearman() are VERBATIM copies from
# experiments/pymfe_screen/screen_pool.py. That script cannot be imported here (it needs pymfe
# and the OpenML loader, and switches off all warnings). check_screening_copy() verifies that
# the copy reproduces the §10.1 values exactly.
MIN_RHO_BEYOND_SIZE = (
    0.30  # |partial rho| on all datasets, after removing log_n and log_p
)
MIN_RHO_TABULAR = 0.20  # |partial rho| on tabular datasets only, same sign


def partial_spearman(df, feature, controls=("log_n", "log_p")):
    """Spearman correlation of feature and k_true after removing (linearly, on ranks) what
    the controls explain. Answers: does the feature tell us about k BEYOND the size?"""
    r = df[[feature, "k_true", *controls]].rank()
    C = np.column_stack([np.ones(len(r))] + [r[c] for c in controls])
    resid = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    rho, pval = stats.pearsonr(
        resid(r[feature].to_numpy()), resid(r["k_true"].to_numpy())
    )
    return float(rho), float(pval)


def screen(data, candidates):
    """
    Apply the §10.1 rule to `candidates` on the datasets in `data` (a training part).
    A candidate passes if |rho beyond size| >= 0.30 on all datasets AND >= 0.20 on the tabular
    ones, with the same sign. Returns the passing candidates (in candidate order) and a table
    with all rho values.

    Guard (added in Phase 4; the §10 candidates were never constant): a feature that is constant
    in `data` or in its tabular part gets rho = NaN and fails. Without the guard, its rank
    residuals are pure floating-point noise and partial_spearman returns a random rho.
    """
    tab = data[data["tabular"]]
    rows = []
    for f in candidates:
        if data[f].nunique() <= 1 or tab[f].nunique() <= 1:
            rho_all = rho_tab = float("nan")  # NaN -> fails the rule below
        else:
            with warnings.catch_warnings():
                # exactly constant residuals -> pearsonr warns and returns NaN -> fails
                warnings.simplefilter("ignore", stats.ConstantInputWarning)
                rho_all, _ = partial_spearman(data, f)
                rho_tab, _ = partial_spearman(tab, f)
        passed = (
            abs(rho_all) >= MIN_RHO_BEYOND_SIZE
            and abs(rho_tab) >= MIN_RHO_TABULAR
            and np.sign(rho_all) == np.sign(rho_tab)
        )
        rows.append(
            {
                "feature": f,
                "rho_beyond_size": rho_all,
                "rho_beyond_size_tabular": rho_tab,
                "n": len(data),
                "n_tabular": len(tab),
                "passed": bool(passed),
            }
        )
    return [r["feature"] for r in rows if r["passed"]], pd.DataFrame(rows)


def check_screening_copy(df):
    """The copied rule must reproduce the §10.1 values of the six static features on all 100
    datasets (results/pymfe_screen/screen_result.csv, saved with 4 decimals). Stops otherwise.
    """
    ref = pd.read_csv(SCREEN_REF).set_index("feature")
    tab = df[df["tabular"]]
    worst = 0.0
    for f in STATIC6:
        ours = (partial_spearman(df, f)[0], partial_spearman(tab, f)[0])
        theirs = (ref.at[f, "rho_beyond_size"], ref.at[f, "rho_beyond_size_tabular"])
        worst = max(worst, *(abs(a - b) for a, b in zip(ours, theirs)))
    if worst > 1e-4:  # 4 saved decimals -> rounding error <= 5e-5
        raise SystemExit(
            f"screening copy does not reproduce §10.1 (max diff {worst:.2g})"
        )
    print(f"Screening rule check vs §10.1: OK (max difference {worst:.1e})")
    return worst


# ----------------------------------------------------------------------------------------
# 5. Loading
# ----------------------------------------------------------------------------------------
def load_tables():
    """Static table (checked by evaluation.py) + probe table + T1 estimators, joined by name.
    The row order of data/metafeatures.csv is kept (needed to reproduce Phase 2 exactly).
    """
    df = ev.load_metafeatures()
    probe = pd.read_csv(PROBE_CSV)
    if list(probe.columns) != [ev.KEY] + PROBE16:
        raise ValueError(f"{PROBE_CSV}: expected columns name + {PROBE16}")
    t1 = pd.read_csv(T1_CSV)[[ev.KEY] + DIRECT_T1]
    df = df.merge(probe, on=ev.KEY, how="left", validate="one_to_one")
    df = df.merge(t1, on=ev.KEY, how="left", validate="one_to_one")

    added = PROBE16 + DIRECT_T1
    if df[added].isna().any().any():  # would trigger P7; not expected (100/100, §12.1)
        raise ValueError(
            "missing probe or T1 values - rule P7 would apply, stop and discuss"
        )
    if (df[PROBE_LOG_COLS] < 1).any().any():  # assumption of the log transform (§12.4)
        raise ValueError("a probe column meant for log() has values < 1")
    overlap = set(added) & set(ev.BOOKKEEPING + [ev.TARGET])
    assert not overlap, f"columns used as feature AND bookkeeping/target: {overlap}"

    # Label-free flag, used ONLY by the screening rule (tabular criterion), never a model input.
    df["tabular"] = ~df["source"].str.startswith("image")
    return df


# ----------------------------------------------------------------------------------------
# 6. Models
# ----------------------------------------------------------------------------------------
def build_pipeline(model, features):
    """
    Preprocessing + model for one feature list (fitted on the training part only).
    For the static columns this is exactly the Phase 2 pipeline (same transformers in the same
    order), so sets A and B reproduce Phase 2. Probe columns are appended after them.
    """
    slog = [c for c in features if c in ts.SIGNED_LOG_COLS]  # static skew/kurt (§3.1)
    plog = [c for c in features if c in PROBE_LOG_COLS]  # §12.4
    pslog = [c for c in features if c in PROBE_SIGNED_LOG_COLS]  # §12.4 (O3)
    rest = [c for c in features if c not in slog + plog + pslog]
    transformers = []
    if slog:
        transformers.append(("signed_log", FunctionTransformer(ts.signed_log), slog))
    transformers.append(("passthrough", "passthrough", rest))
    if plog:
        transformers.append(("log", FunctionTransformer(np.log), plog))
    if pslog:
        transformers.append(
            ("signed_log_probe", FunctionTransformer(ts.signed_log), pslog)
        )
    steps = [("prep", ColumnTransformer(transformers))]
    if ts.MODELS[model]["scale"]:
        steps.append(("scale", StandardScaler()))  # Ridge, k-NN (§3.2)
    steps.append(("model", ts.MODELS[model]["estimator"]()))
    return Pipeline(steps)


def make_estimator(model, features):
    """As train_static.make_estimator for the log-k target, but for an explicit feature list
    (the screened sets have their own list in every fold). Tuning: GridSearchCV with an inner
    GroupKFold on the training part, scored with the protocol's MAE (§2.5)."""
    pipe = TransformedTargetRegressor(
        regressor=build_pipeline(model, features), func=np.log, inverse_func=np.exp
    )
    grid = {"regressor__model__" + k: v for k, v in ts.MODELS[model]["grid"].items()}
    return GridSearchCV(
        pipe,
        grid,
        cv=GroupKFold(n_splits=ev.INNER_SPLITS),
        scoring=ev.mae_scorer,
        n_jobs=-1 if model == "rf" else 1,
    )


def config_list():
    """All configurations in reporting order, in the format train_static.make_summary expects."""
    configs = [
        {"name": "median_baseline", "model": "median", "features": "-", "target": "k"}
    ]
    for s in FEATURE_SETS:
        for m in MODELS:
            configs.append(
                {
                    "name": f"{m}__{s}__{TARGET}",
                    "model": m,
                    "features": s,
                    "target": TARGET,
                }
            )
    for family in ["probe", "t1"]:
        configs.append(
            {
                "name": f"direct_best_{family}",
                "model": "direct",
                "features": f"best {family}",
                "target": "-",
            }
        )
    for c in DIRECT_PROBE + DIRECT_T1:
        configs.append(
            {"name": f"direct__{c}", "model": "direct", "features": c, "target": "-"}
        )
    return configs


# ----------------------------------------------------------------------------------------
# 7. Running the protocol
# ----------------------------------------------------------------------------------------
def run(df, folds, max_repeats):
    """Train/test every configuration on every (repeat, fold). Returns four tables."""
    X = df[
        STATIC8 + PROBE16
    ]  # every column a model may use; bookkeeping never reaches a model
    y = df[ev.TARGET].to_numpy(dtype=float)
    groups = df[ev.GROUP].to_numpy()
    pred_rows, score_rows, param_rows, screen_tables = [], [], [], []
    start = time.time()

    for repeat, fold, tr, te in ev.iter_splits(df, folds):
        if repeat >= max_repeats:
            break

        def record(config, pred, params):
            """Store the clipped test predictions, the fold scores and the chosen settings."""
            pred = ev.clip_k(pred)
            score_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "config": config,
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
                    "config": config,
                    "params": json.dumps(params, default=float),
                }
            )
            for i, p in zip(te, pred):
                pred_rows.append(
                    {
                        ev.KEY: df.at[i, ev.KEY],
                        "repeat": repeat,
                        "fold": fold,
                        "config": config,
                        "k_true": y[i],
                        "k_pred": p,
                    }
                )

        # (a) median baseline: median k of the training part (§2.6)
        base = ts.MedianBaseline().fit(X.iloc[tr], y[tr])
        record("median_baseline", base.predict(X.iloc[te]), {"median": base.median_})

        # (b) screening on the TRAINING part only (§12.2); test datasets are never used
        features = {}
        for s, spec in FEATURE_SETS.items():
            passed = []
            if spec["screened"]:
                passed, table = screen(df.iloc[tr], spec["screened"])
                table.insert(0, "set", s)
                table.insert(0, "fold", fold)
                table.insert(0, "repeat", repeat)
                screen_tables.append(table)
            features[s] = (
                spec["base"] + passed
            )  # nothing passes -> base (log n + log p) only

        # (c) 6 feature sets x 3 models, log k, nested tuning (§12.3)
        for s in FEATURE_SETS:
            for m in MODELS:
                est = make_estimator(m, features[s])
                est.fit(X.iloc[tr], y[tr], groups=groups[tr])
                params = {k.split("__")[-1]: v for k, v in est.best_params_.items()}
                if FEATURE_SETS[s]["screened"]:
                    params["features"] = features[s]  # the columns used in this fold
                record(f"{m}__{s}__{TARGET}", est.predict(X.iloc[te]), params)

        # (d) direct estimators: no training; the "best" one is chosen by its MAE on the
        #     training part (ties: first in list order) and applied to the test part (§12.5)
        for family, cols in [("probe", DIRECT_PROBE), ("t1", DIRECT_T1)]:
            values = {c: df[c].to_numpy(dtype=float) for c in cols}
            for c in cols:
                record(f"direct__{c}", values[c][te], {})
            train_mae = {c: ev.mae(y[tr], values[c][tr]) for c in cols}
            best = min(
                cols, key=lambda c: train_mae[c]
            )  # min() keeps the first on ties
            record(
                f"direct_best_{family}",
                values[best][te],
                {"estimator": best, "train_mae": train_mae[best]},
            )

        if fold == ev.N_SPLITS - 1:
            print(f"repeat {repeat} done ({time.time() - start:.0f} s)", flush=True)

    return (
        pd.DataFrame(pred_rows),
        pd.DataFrame(score_rows),
        pd.DataFrame(param_rows),
        pd.concat(screen_tables, ignore_index=True),
    )


# ----------------------------------------------------------------------------------------
# 8. Tests and checks
# ----------------------------------------------------------------------------------------
def paired_test(scores, a, b):
    """Corrected resampled t-test (§2.7) of config a vs config b on identical folds.
    mean_diff < 0 means a has the lower (better) MAE."""
    table = scores.set_index(["config", "repeat", "fold"])
    sa = table.loc[a]
    sb = table.loc[b].loc[sa.index]  # KeyError if the folds differ
    return ev.corrected_ttest(
        sa["mae"] - sb["mae"], n_train=sa["n_train"].mean(), n_test=sa["n_test"].mean()
    )


def holm(pvalues):
    """Holm (1979) step-down adjusted p-values, returned in the input order: the smallest p is
    multiplied by m, the next by m-1, ..., kept monotone and capped at 1."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def test_hypotheses(scores):
    """The pre-registered tests of §12.6: two families (Q1: 2 tests, Q2: 4 tests), Holm within
    each family. An improvement is claimed only if p_holm < 0.05 and mean_diff < 0."""
    rows = []
    for family, tests in [("Q1", Q1_TESTS), ("Q2", Q2_TESTS)]:
        fam = [
            {"family": family, "test": name, "config": a, "compared_with": b}
            | paired_test(scores, a, b)
            for name, a, b in tests
        ]
        for r, p_adj in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = p_adj
            r["significant_improvement"] = bool(p_adj < ALPHA and r["mean_diff"] < 0)
        rows += fam
    return pd.DataFrame(rows)


def secondary_tests(scores):
    """Secondary comparisons (§12.6): uncorrected p-values, reported only."""
    return pd.DataFrame(
        [
            {"test": name, "config": a, "compared_with": b} | paired_test(scores, a, b)
            for name, a, b in SECONDARY_TESTS
        ]
    )


def check_phase2(scores):
    """§12.3 built-in check: sets A and B (and the median baseline) must reproduce the Phase 2
    fold scores. ok = largest absolute MAE difference over the folds <= 1e-9."""
    ref = pd.read_csv(PHASE2_SCORES).set_index(["config", "repeat", "fold"])["mae"]
    pairs = [("median_baseline", "median_baseline")] + [
        (f"{m}__{s}__{TARGET}", f"{m}__{p2}__{TARGET}")
        for s, p2 in PHASE2_EQUIVALENT.items()
        for m in MODELS
    ]
    rows = []
    for ours_name, ref_name in pairs:
        ours = scores[scores["config"] == ours_name].set_index(["repeat", "fold"])[
            "mae"
        ]
        theirs = ref.loc[ref_name].loc[ours.index]
        rows.append(
            {
                "config": ours_name,
                "phase2_config": ref_name,
                "folds": len(ours),
                "max_abs_diff": float((ours - theirs).abs().max()),
            }
        )
    table = pd.DataFrame(rows)
    table["ok"] = table["max_abs_diff"] <= 1e-9
    return table


# ----------------------------------------------------------------------------------------
# 9. Secondary analyses (§12.7)
# ----------------------------------------------------------------------------------------
def gain_share(preds, df):
    """
    Where does each configuration's gain over the median baseline come from?
    gain = baseline error - model error per test prediction, summed over the datasets and
    averaged over the repeats (as in Phase 2, §5.2: 87% of the size-only gain came from images).
    share_image = gain_image / gain_total; it is left empty when the total gain is not positive,
    and it is above 1 when the model is worse than the baseline on the tabular datasets.
    """
    is_image = df.set_index(ev.KEY)["source"].str.startswith("image")
    p = preds.assign(abs_err=(preds["k_true"] - preds["k_pred"]).abs())
    n_repeats = p["repeat"].nunique()
    base = p[p["config"] == "median_baseline"].set_index([ev.KEY, "repeat"])["abs_err"]
    rows = []
    for config, part in p[p["config"] != "median_baseline"].groupby(
        "config", sort=False
    ):
        part = part.set_index([ev.KEY, "repeat"])
        gain = base.loc[part.index].to_numpy() - part["abs_err"].to_numpy()
        img = is_image.loc[part.index.get_level_values(ev.KEY)].to_numpy()
        total, g_img = gain.sum() / n_repeats, gain[img].sum() / n_repeats
        rows.append(
            {
                "config": config,
                "gain_total": total,
                "gain_image": g_img,
                "gain_tabular": total - g_img,
                "share_image": g_img / total if total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def selection_frequency(screening):
    """How often each candidate passed the screening over the training parts (sets C and D)."""
    return (
        screening.groupby(["set", "feature"], sort=False)
        .agg(
            training_parts=("passed", "size"),
            selected=("passed", "sum"),
            share_selected=("passed", "mean"),
            median_rho_beyond_size=("rho_beyond_size", "median"),
        )
        .reset_index()
    )


# ----------------------------------------------------------------------------------------
# 10. Figures
# ----------------------------------------------------------------------------------------
# Model identity = color AND marker shape (the 3rd palette slot has low contrast on white, so
# color is never the only cue; exact numbers are in summary.csv). Colors: the validated
# categorical palette of Phase 2 (train_static.py), in fixed order.
MODEL_STYLE = {"ridge": (ts.BLUE, "s"), "knn": (ts.ORANGE, "^"), "rf": (ts.AQUA, "o")}
SET_TICKS = {
    "A": "A\nstatic 8",
    "B": "B\nsize",
    "C": "C\nsize +\nscreened static",
    "D": "D (main)\nsize + screened\nstatic & probe",
    "E": "E\nstatic 8 +\nprobe 16",
    "F": "F\nprobe 16",
}


def plot_scores(summary, path, n_repeats):
    """MAE (left) and share within ±1 (right) per feature set and model; dashed = median
    baseline. Dots = mean over the repeats, bars = sd over the repeats."""
    s = summary.set_index("config")
    base = s.loc["median_baseline"]
    sets = list(FEATURE_SETS)
    offsets = {"ridge": -0.22, "knn": 0.0, "rf": 0.22}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    panels = [
        (
            "mae",
            f"MAE on the k scale (lower is better; baseline {base['mae_mean']:.2f})",
        ),
        (
            "within1",
            f"Share within ±1 (higher is better; baseline {base['within1_mean']:.0%})",
        ),
    ]
    for ax, (metric, title) in zip(axes, panels):
        ts._style(ax)
        for m in MODELS:
            color, marker = MODEL_STYLE[m]
            r = s.loc[[f"{m}__{x}__{TARGET}" for x in sets]]
            ax.errorbar(
                np.arange(len(sets)) + offsets[m],
                r[f"{metric}_mean"],
                yerr=r[f"{metric}_sd"],
                fmt=marker,
                markersize=8,
                color=color,
                elinewidth=1.5,
                capsize=0,
                label=ts.MODEL_LABELS[m],
            )
        ax.axhline(
            base[f"{metric}_mean"],
            color=ts.INK_2,
            linestyle="--",
            linewidth=1.5,
            label="median baseline",
        )
        ax.set_xticks(
            range(len(sets)), [SET_TICKS[x] for x in sets], color=ts.INK, fontsize=8
        )
        ax.set_title(title, loc="left", fontsize=11, color=ts.INK)
        if metric == "within1":
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(1.0, decimals=0)
            )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9
    )
    fig.suptitle(
        "Phase 4: static + probe meta-learner vs. median baseline (log k target; "
        f"5-fold grouped CV x {n_repeats} repeat(s); mean ± sd over repeats)",
        fontsize=11,
        color=ts.INK,
        x=0.01,
        ha="left",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# 11. Main
# ----------------------------------------------------------------------------------------
RESULT_FILES = [
    "summary.csv",
    "hypotheses.csv",
    "secondary_tests.csv",
    "phase2_check.csv",
    "fold_scores.csv",
    "hyperparameters.csv",
    "screening.csv",
    "selection_frequency.csv",
    "screen_all100.csv",
    "breakdown.csv",
    "gain_share.csv",
    "fig_scores.png",
    "fig_pred_vs_true.png",
    "run_info.json",
]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: static + probe meta-learner."
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

    df = load_tables()
    check_screening_copy(df)
    folds = ev.load_folds(df)
    configs = config_list()
    print(
        f"{len(df)} datasets, {len(configs)} configurations, {args.repeats} repeat(s)"
    )

    t0 = time.time()
    preds, scores, params, screening = run(df, folds, args.repeats)
    runtime = time.time() - t0

    summary = ts.make_summary(
        scores, configs
    )  # mean ± sd + corrected t-test vs. baseline
    hypotheses = test_hypotheses(scores)
    secondary = secondary_tests(scores)
    phase2 = check_phase2(scores)
    valid = bool(phase2["ok"].all())
    _, all100 = screen(
        df, STATIC6 + PROBE16
    )  # descriptive only (§12.7), not used for anything
    all100 = all100.rename(columns={"passed": "would_pass_on_all_100"})

    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    scores.to_csv(out_dir / "fold_scores.csv", index=False)
    params.to_csv(out_dir / "hyperparameters.csv", index=False)
    screening.to_csv(out_dir / "screening.csv", index=False, float_format="%.4f")
    selection_frequency(screening).to_csv(
        out_dir / "selection_frequency.csv", index=False, float_format="%.4f"
    )
    all100.to_csv(out_dir / "screen_all100.csv", index=False, float_format="%.4f")
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    hypotheses.to_csv(out_dir / "hypotheses.csv", index=False, float_format="%.4f")
    secondary.to_csv(out_dir / "secondary_tests.csv", index=False, float_format="%.4f")
    phase2.to_csv(out_dir / "phase2_check.csv", index=False)
    ts.make_breakdown(preds, df).to_csv(
        out_dir / "breakdown.csv", index=False, float_format="%.4f"
    )
    gain_share(preds, df).to_csv(
        out_dir / "gain_share.csv", index=False, float_format="%.4f"
    )

    plot_scores(summary, out_dir / "fig_scores.png", scores["repeat"].nunique())
    ts.FEATURE_LABELS.update(
        SET_LABELS
    )  # labels for the reused Phase 2 figure function
    ts.plot_pred_vs_true(
        preds,
        df,
        summary,
        out_dir / "fig_pred_vs_true.png",
        rows_spec=[("A", TARGET), ("C", TARGET), ("D", TARGET)],
    )

    # Everything needed to trace the numbers back to exact inputs and versions.
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "design": "docs/metalearner_methodology.md §12, Design Decisions Log D13",
        "full_run": full_run,
        "valid_phase2_check": valid,
        "repeats": args.repeats,
        "runtime_s": round(runtime),
        "metafeatures_sha256": ts.sha256(ev.METAFEATURES_CSV),
        "probe_features_sha256": ts.sha256(PROBE_CSV),
        "t1_candidates_sha256": ts.sha256(T1_CSV),
        "cv_folds_sha256": ts.sha256(ev.FOLDS_CSV),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    # Only a complete AND valid run goes into the committed results folder.
    if full_run and valid:
        results_dir.mkdir(parents=True, exist_ok=True)
        for f in RESULT_FILES:
            shutil.copy(out_dir / f, results_dir / f)
        print(f"Copied results to {results_dir}")
    elif full_run:
        print(
            "INVALID RUN: sets A/B do not reproduce Phase 2 (phase2_check.csv) - "
            f"results/{RUN_NAME} was NOT written."
        )
    else:
        print(f"Test run: results/{RUN_NAME} was NOT updated.")

    pd.set_option("display.width", 200)
    print(f"\nFinished in {runtime / 60:.1f} min.")
    print("\nPhase 2 reproduction check (§12.3):")
    print(phase2.to_string(index=False))
    print("\nPre-registered tests (§12.6; mean_diff < 0 = primary model better):")
    cols = [
        "family",
        "test",
        "mean_diff",
        "ci_low",
        "ci_high",
        "p",
        "p_holm",
        "significant_improvement",
    ]
    print(hypotheses[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
