"""
posthoc_probes.py - POST-HOC analysis: why do the probe features fail? (methodology §12.13)

POST-HOC: designed on 2026-10-01, AFTER the Phase 4 results (§12.10-12.12) were known. Descriptive only:
no model is trained, no claim of the thesis rests on these numbers, and nothing here may be used as
evidence that the probe features work.

(a) Where do the probe features carry information about k?
    For each of the 16 probe features and, as references, the 8 static features:
      - Spearman rho with the true k on all 100, the 83 tabular and the 17 image datasets;
      - the same "beyond size" (partial Spearman after removing log n and log p, the measure of the
        §10.1 screening rule);
      - rho within each image family whose variants differ only in k (MNIST, Fashion-MNIST, KMNIST:
        k = 2, 3, 5, 10; the pilot's R2a check, now on all three families) and the mean over them;
      - rho with log n and with log p (does the feature mostly measure dataset size?);
      - rho with k on the datasets whose probe k range was NOT capped (n_train >= 250; below that the
        probes could only use k <= n_train // 5 < 50, run_probes.py).
(b) How noisy are the probe features across the 3 seeds?
      - Spearman rho between each pair of seeds across the 100 datasets (as R2c in the pilot);
      - rho with k for each single seed (smallest and largest of the 3);
      - noise share = mean over datasets of the sd across seeds / sd across datasets of the
        seed-averaged value (0 = no seed noise; around 1 = the seed noise is as large as the
        differences between datasets). G4 is computed across seeds and has no per-seed value.

Outputs: results/phase4_posthoc/ (feature_signal.csv, seed_agreement.csv, fig_signal_by_source.png,
run_info.json).
Usage:  python posthoc_probes.py      (seconds; no GPU)
"""

import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files, no window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

import evaluation as ev  # frozen protocol (unchanged)
import train_phase4 as p4  # inputs and the copied §10.1 partial_spearman (unchanged)
import train_static as ts  # figure style and sha256 (unchanged)

ROOT = Path(__file__).resolve().parent
PER_SEED = (
    ROOT / "results" / "phase3_full" / "per_seed.csv"
)  # Phase 3: candidates per seed
OUT_DIR = ROOT / "results" / "phase4_posthoc"
CAP_N_TRAIN = (
    250  # n_train // 5 < 50 below this: the probe k range was capped (run_probes.py)
)
PER_SEED_CANDIDATES = [
    "G1",
    "G2",
    "G3",
    "L1",
    "O1",
    "O2",
    "O3",
]  # G4 exists only across seeds


def rho(x, y):
    """Spearman rho; NaN if either input is constant (correlation undefined)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def partial_rho(data, feature):
    """The §10.1 measure ('beyond size'); NaN for a feature that is constant in `data`."""
    if data[feature].nunique() < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        return p4.partial_spearman(data, feature)[0]


# ----------------------------------------------------------------------------------------
# (a) Where do the features carry information?
# ----------------------------------------------------------------------------------------
def feature_signal(df, n_train):
    """One row per feature with the rho values described in the module docstring."""
    k = ev.TARGET
    tab, img = df[df["tabular"]], df[~df["tabular"]]
    # image families = groups with several datasets (variants that differ only in k)
    sizes = df.groupby(ev.GROUP).size()
    families = list(sizes[sizes > 1].index)
    uncapped = df[df[ev.KEY].map(n_train) >= CAP_N_TRAIN]

    rows = []
    for f in p4.PROBE16 + p4.STATIC8:
        fam = {
            fam_name: rho(part[f], part[k])
            for fam_name, part in df[df[ev.GROUP].isin(families)].groupby(ev.GROUP)
        }
        rows.append(
            {
                "feature": f,
                "kind": "probe" if f in p4.PROBE16 else "static",
                "rho_all": rho(df[f], df[k]),
                "rho_tabular": rho(tab[f], tab[k]),
                "rho_image": rho(img[f], img[k]),
                "rho_beyond_size_all": (
                    partial_rho(df, f) if f not in p4.SIZE2 else np.nan
                ),
                "rho_beyond_size_tabular": (
                    partial_rho(tab, f) if f not in p4.SIZE2 else np.nan
                ),
                "rho_beyond_size_image": (
                    partial_rho(img, f) if f not in p4.SIZE2 else np.nan
                ),
                **{f"rho_family_{name}": v for name, v in fam.items()},
                "rho_family_mean": np.nanmean(list(fam.values())) if fam else np.nan,
                "rho_with_log_n": rho(df[f], df["log_n"]),
                "rho_with_log_p": rho(df[f], df["log_p"]),
                "rho_uncapped": rho(uncapped[f], uncapped[k]),
                "n_uncapped": len(uncapped),
            }
        )
    return pd.DataFrame(rows), families


# ----------------------------------------------------------------------------------------
# (b) Seed noise
# ----------------------------------------------------------------------------------------
def seed_agreement(per_seed):
    """Agreement between the 3 seeds for every per-seed candidate and algorithm."""
    rows = []
    for alg in p4.ALGORITHMS:
        part = per_seed[per_seed["algorithm"] == alg]
        k_true = part.groupby(ev.KEY)["k_true"].first()
        for cand in PER_SEED_CANDIDATES:
            wide = part.pivot(
                index=ev.KEY, columns="seed", values=cand
            )  # datasets x seeds
            seeds = list(wide.columns)
            pair_rhos = [
                rho(wide[a], wide[b])
                for i, a in enumerate(seeds)
                for b in seeds[i + 1 :]
            ]
            seed_rhos_k = [rho(wide[s], k_true.loc[wide.index]) for s in seeds]
            between = wide.mean(axis=1).std(
                ddof=1
            )  # spread of the seed average across datasets
            within = wide.std(axis=1, ddof=1).mean()  # average spread across seeds
            rows.append(
                {
                    "feature": f"{cand}_{alg}",
                    "seed_pair_rho_min": np.nanmin(pair_rhos),
                    "seed_pair_rho_mean": np.nanmean(pair_rhos),
                    "rho_k_single_seed_min": np.nanmin(seed_rhos_k),
                    "rho_k_single_seed_max": np.nanmax(seed_rhos_k),
                    "noise_share": within / between if between > 0 else np.nan,
                    "datasets": len(wide),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------
def plot_signal(signal, path):
    """Spearman rho with k per feature: all datasets, tabular, image, and the mean within the
    image families. Identity by color AND marker; the legend names every series."""
    order = p4.SIZE2 + [f for f in p4.PROBE16]
    s = signal.set_index("feature").loc[order]
    y = np.arange(len(order))[::-1]
    series = [
        ("rho_all", "all 100 datasets", ts.INK_2, "o", "none"),
        ("rho_tabular", "tabular (83)", ts.BLUE, "s", None),
        ("rho_image", "image (17)", ts.ORANGE, "^", None),
        (
            "rho_family_mean",
            "within MNIST / F-MNIST / KMNIST (mean)",
            ts.AQUA,
            "D",
            None,
        ),
    ]
    fig, ax = plt.subplots(figsize=(8, 7.5), constrained_layout=True)
    ts._style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=ts.GRID, linewidth=0.8)
    ax.axvline(0, color=ts.INK_2, linewidth=1)
    for col, label, color, marker, face in series:
        ax.plot(
            s[col],
            y,
            linestyle="none",
            marker=marker,
            markersize=7,
            color=color,
            markerfacecolor=face if face else color,
            markeredgewidth=1.5,
            label=label,
        )
    ax.set_yticks(y, order, fontsize=9, color=ts.INK)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Spearman ρ with the true k", color=ts.INK_2)
    ax.set_title(
        "POST-HOC: where do the features carry information about k?",
        loc="left",
        fontsize=11,
        color=ts.INK,
    )
    fig.legend(loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    df = (
        p4.load_tables()
    )  # static + probe features, k, source, group (checked as in Phase 4)
    per_seed = pd.read_csv(PER_SEED)
    n_train = per_seed.groupby(ev.KEY)["n_train"].first()
    if set(n_train.index) != set(df[ev.KEY]):
        raise SystemExit("per_seed.csv does not cover exactly the 100 pool datasets")

    signal, families = feature_signal(df, n_train)
    agreement = seed_agreement(per_seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signal.to_csv(OUT_DIR / "feature_signal.csv", index=False, float_format="%.4f")
    agreement.to_csv(OUT_DIR / "seed_agreement.csv", index=False, float_format="%.4f")
    plot_signal(signal, OUT_DIR / "fig_signal_by_source.png")
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "status": "POST-HOC (designed after the Phase 4 results); descriptive only",
        "design": "docs/metalearner_methodology.md §12.13",
        "image_families": families,
        "datasets_uncapped": int((n_train >= CAP_N_TRAIN).sum()),
        "metafeatures_sha256": ts.sha256(ev.METAFEATURES_CSV),
        "probe_features_sha256": ts.sha256(p4.PROBE_CSV),
        "per_seed_sha256": ts.sha256(PER_SEED),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"POST-HOC analysis saved to {OUT_DIR}")

    pd.set_option("display.width", 220)
    cols = [
        "feature",
        "rho_all",
        "rho_tabular",
        "rho_image",
        "rho_beyond_size_all",
        "rho_family_mean",
        "rho_with_log_n",
        "rho_uncapped",
    ]
    print("\n(a) Spearman rho with the true k (probe features, then static features):")
    print(signal[cols].round(2).to_string(index=False))
    print("\n(b) Seed agreement (per-seed candidates):")
    print(agreement.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
