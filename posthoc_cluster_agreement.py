"""
posthoc_cluster_agreement.py - POST-HOC part (c): do DEC/IDEC recover the classes as clusters?
(methodology §12.13)

POST-HOC: designed on 2026-10-02, AFTER the Phase 4 results and post-hoc parts (a)+(b) were known.
DIAGNOSIS ONLY: this script uses the true class LABELS, which the meta-learner must never see (D3).
Nothing here is a meta-feature, and no claim that the probes work may rest on it.

Question. Parts (a)+(b) showed that the probe features carry information about k on the image datasets but
not on the tabular ones. Hypothesis: the probes track k where DEC/IDEC find the classes as clusters, and fail
where they do not. This script measures, for every pool dataset, how well the clusters of the Phase 3 probe
runs agree with the classes.

No new DC runs. Phase 3 saved the final cluster labels of every probe run
(outputs/phase3/full/<name>/labels/seed{s}_{DEC|IDEC}_k{k}.npy, 500 probe steps). The probe rows were drawn
with a fixed, label-free random draw (probes.split_and_scale), so the same rows - and their true labels -
are reconstructed here with the same code.

Per dataset:
  - k_ref = the grid k closest to the true k on the log scale (the true k itself is often not on the grid);
  - AMI and ARI (both corrected for chance; 0 = chance, 1 = perfect) between the probe clusters at k_ref and
    the classes, for DEC and IDEC, mean over the 3 seeds;
  - the best AMI over all grid k (seed mean) and the k where it occurs (upper bound: "if the right k were
    known");
  - reference without deep clustering: k-means (n_init = 3, random_state = 0) on the same scaled rows,
    at k_ref and at the true k.
Summary: AMI by source (tabular / image) and per image family; Spearman rho across datasets between the AMI
(DEC and IDEC at k_ref) and the error |log(feature) - log(k)| of the three probe features that carried
information on the image datasets (G3_DEC, O1_DEC, L1_IDEC); negative rho = the probe is right where the
classes are clusters.

Reading, fixed before the run (descriptive, post-hoc): the hypothesis is SUPPORTED if
  (1) the median AMI at k_ref on the image datasets exceeds that on the tabular datasets by at least 0.20,
      for DEC and for IDEC, and
  (2) rho(AMI, probe error) <= -0.30 across all 100 datasets for at least 2 of the 3 probe features
      (with the AMI of the matching algorithm);
NOT SUPPORTED if neither holds; PARTLY SUPPORTED otherwise.

Built-in checks: every dataset's reconstructed training rows must have the same number of points as the
saved cluster labels and as results/phase3_full/per_seed.csv, and the number of classes must equal k_true.

Runs on the cluster (the saved labels and the OpenML cache are there), CPU only:
    sbatch slurm/run_posthoc_c.sh
Outputs: results/phase4_posthoc_c/ (agreement_per_dataset.csv, agreement_per_run.csv, summary.json,
fig_cluster_agreement.png, run_info.json).
"""

import argparse
import json
import platform
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files, no window needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

import train_static as ts  # figure style and sha256 (unchanged)

ROOT = Path(__file__).resolve().parent
POOL = ROOT / "data" / "pool.csv"
METAFEATURES = ROOT / "data" / "metafeatures.csv"
PROBE_TABLE = ROOT / "data" / "probe_features.csv"
PER_SEED = ROOT / "results" / "phase3_full" / "per_seed.csv"
PROBE_OUT = ROOT / "outputs" / "phase3" / "full"  # raw Phase 3 outputs (cluster only)
OUT_DIR = ROOT / "results" / "phase4_posthoc_c"

LABEL_FILE = re.compile(r"seed(\d+)_(DEC|IDEC)_k(\d+)\.npy")
OVERCOMPLETE_K = 50  # the overcomplete run (not a grid k) is excluded from the grid
FEATURES_TO_CHECK = {
    "G3_DEC": "DEC",
    "O1_DEC": "DEC",
    "L1_IDEC": "IDEC",
}  # (a): signal on images
MIN_IMAGE_GAP = 0.20  # reading rule (1)
MAX_RHO = -0.30  # reading rule (2)


# ----------------------------------------------------------------------------------------
# 1. The probe rows and their true labels
# ----------------------------------------------------------------------------------------
def load_with_labels(spec):
    """Same steps as probes.prepare_data (same functions, same order, same random draw), but the
    true labels of the training rows are kept. Returns (X_train, y_train)."""
    import probes  # imported here: needs torch/clustpy, only available in the project env
    from extractor import drop_constant_columns, load_openml_numeric, take_class_subset

    dataset_id = int(spec["dataset_id"])
    X, y = load_openml_numeric(dataset_id)
    subset_k = spec.get("subset_k")
    has_subset = subset_k is not None and pd.notna(subset_k)
    if has_subset:
        X, y, _ = take_class_subset(X, y, dataset_id, int(subset_k), int(spec["n_max"]))
    X = drop_constant_columns(X)
    is_image = spec["source"] in probes.IMAGE_SOURCES
    rng_key = [dataset_id, int(subset_k) if has_subset else 0]
    data = probes.split_and_scale(
        X, is_image, rng_key
    )  # the identical row draw and scaling
    y_train = np.asarray(y)[data["rows_train"]]
    return data["X_train"], y_train


def saved_runs(name):
    """{(seed, algorithm, k): cluster labels} of all saved probe runs of one dataset."""
    runs = {}
    for f in sorted((PROBE_OUT / name / "labels").glob("*.npy")):
        m = LABEL_FILE.fullmatch(f.name)
        if m:
            runs[(int(m.group(1)), m.group(2), int(m.group(3)))] = np.load(f)
    return runs


def nearest_on_log_scale(candidates, k_true):
    """Grid k with the smallest |log k - log k_true| (ties: the smaller k)."""
    c = np.array(sorted(candidates))
    return int(c[np.argmin(np.abs(np.log(c) - np.log(k_true)))])


# ----------------------------------------------------------------------------------------
# 2. Agreement for one dataset
# ----------------------------------------------------------------------------------------
def agreement_one(spec, k_true, n_train_expected):
    name = spec["name"]
    X_train, y = load_with_labels(spec)
    runs = saved_runs(name)
    if not runs:
        raise SystemExit(f"{name}: no saved labels in {PROBE_OUT / name / 'labels'}")
    # built-in checks: same rows as in Phase 3, same number of classes as the ground truth
    lengths = {len(v) for v in runs.values()}
    if lengths != {len(y)} or len(y) != n_train_expected:
        raise SystemExit(
            f"{name}: rows {len(y)}, saved labels {lengths}, per_seed {n_train_expected}"
        )
    if len(np.unique(y)) != k_true:
        raise SystemExit(
            f"{name}: {len(np.unique(y))} classes in the probe rows, k_true {k_true}"
        )

    run_rows = []
    for (seed, alg, k), labels in runs.items():
        run_rows.append(
            {
                "name": name,
                "seed": seed,
                "algorithm": alg,
                "k": k,
                "ami": adjusted_mutual_info_score(y, labels),
                "ari": adjusted_rand_score(y, labels),
                "clusters_used": int(np.unique(labels).size),
            }
        )
    per_run = pd.DataFrame(run_rows)
    grid = per_run[
        per_run["k"] != OVERCOMPLETE_K
    ]  # grid runs only (K = 50 is the O run)
    k_ref = nearest_on_log_scale(grid["k"].unique(), k_true)
    seed_mean = grid.groupby(["algorithm", "k"])[["ami", "ari"]].mean()

    row = {"name": name, "k_true": k_true, "n_train": len(y), "k_ref": k_ref}
    for alg in ["DEC", "IDEC"]:
        at_ref = grid[(grid["algorithm"] == alg) & (grid["k"] == k_ref)]
        best = seed_mean.loc[alg]["ami"]
        row.update(
            {
                f"ami_{alg}": at_ref["ami"].mean(),
                f"ami_{alg}_seed_min": at_ref["ami"].min(),
                f"ami_{alg}_seed_max": at_ref["ami"].max(),
                f"ari_{alg}": at_ref["ari"].mean(),
                f"ami_{alg}_best_over_grid": best.max(),
                f"k_best_ami_{alg}": int(best.idxmax()),
            }
        )
    for label, k in [("kref", k_ref), ("ktrue", k_true)]:
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(X_train)
        row[f"ami_kmeans_raw_{label}"] = adjusted_mutual_info_score(y, km)
        row[f"ari_kmeans_raw_{label}"] = adjusted_rand_score(y, km)
    return row, per_run


# ----------------------------------------------------------------------------------------
# 3. Summary and figure
# ----------------------------------------------------------------------------------------
def summarize(per_dataset, meta, probe):
    d = per_dataset.merge(meta[["name", "source", "group"]], on="name").merge(
        probe, on="name"
    )
    d["type"] = np.where(d["source"].str.startswith("image"), "image", "tabular")
    out = {"by_type": {}, "families": {}, "rho_ami_vs_probe_error": {}}
    cols = [
        "ami_DEC",
        "ami_IDEC",
        "ami_kmeans_raw_kref",
        "ami_DEC_best_over_grid",
        "ami_IDEC_best_over_grid",
    ]
    for t, part in d.groupby("type"):
        out["by_type"][t] = {
            c: {
                "median": float(part[c].median()),
                "q1": float(part[c].quantile(0.25)),
                "q3": float(part[c].quantile(0.75)),
            }
            for c in cols
        } | {"datasets": int(len(part))}
    sizes = d.groupby("group").size()
    for g in sizes[sizes > 1].index:
        part = d[d["group"] == g].sort_values("k_true")
        out["families"][g] = (
            part[["name", "k_true", "ami_DEC", "ami_IDEC", "ami_kmeans_raw_kref"]]
            .round(3)
            .to_dict("records")
        )
    passed = 0
    for feat, alg in FEATURES_TO_CHECK.items():
        err = np.abs(np.log(d[feat]) - np.log(d["k_true"]))
        r_all = float(stats.spearmanr(d[f"ami_{alg}"], err).statistic)
        tab = d["type"] == "tabular"
        r_tab = float(stats.spearmanr(d.loc[tab, f"ami_{alg}"], err[tab]).statistic)
        out["rho_ami_vs_probe_error"][feat] = {
            "algorithm": alg,
            "all": r_all,
            "tabular": r_tab,
        }
        passed += r_all <= MAX_RHO

    types = set(out["by_type"])
    if types != {"image", "tabular"}:  # only in test runs with --limit
        out["reading"] = {
            "verdict": "INCOMPLETE (test run: not both data types present)"
        }
        return out, d
    gap = {
        alg: out["by_type"]["image"][f"ami_{alg}"]["median"]
        - out["by_type"]["tabular"][f"ami_{alg}"]["median"]
        for alg in ["DEC", "IDEC"]
    }
    rule1 = all(g >= MIN_IMAGE_GAP for g in gap.values())
    rule2 = passed >= 2
    if rule1 and rule2:
        verdict = "SUPPORTED"
    elif rule1 or rule2:
        verdict = "PARTLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    out["reading"] = {
        "median_gap_image_minus_tabular": gap,
        "rule1_gap_at_least_0.20": rule1,
        "rule2_features_with_rho_at_most_-0.30": passed,
        "verdict": verdict,
    }
    return out, d


def plot(d, path):
    """Left: AMI per dataset at k_ref for k-means (raw), DEC and IDEC, by data type.
    Right: AMI (DEC) vs. the error of the probe feature G3_DEC. Type = color AND marker.
    """
    style = {"tabular": (ts.BLUE, "s"), "image": (ts.ORANGE, "^")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    rng = np.random.default_rng(0)  # small horizontal jitter, same every run
    methods = [
        ("ami_kmeans_raw_kref", "k-means (raw)"),
        ("ami_DEC", "DEC"),
        ("ami_IDEC", "IDEC"),
    ]
    for ax in (ax1, ax2):
        ts._style(ax)
    for i, (col, label) in enumerate(methods):
        for j, (t, (color, marker)) in enumerate(style.items()):
            part = d[d["type"] == t]
            x = i + (j - 0.5) * 0.35 + rng.uniform(-0.08, 0.08, len(part))
            ax1.plot(
                x,
                part[col],
                linestyle="none",
                marker=marker,
                color=color,
                markersize=5,
                alpha=0.8,
                label=f"{t} ({len(part)})" if i == 0 else None,
            )
            ax1.plot(
                [i + (j - 0.5) * 0.35 - 0.14, i + (j - 0.5) * 0.35 + 0.14],
                [part[col].median()] * 2,
                color=ts.INK,
                linewidth=2,
            )
    ax1.set_xticks(range(len(methods)), [m[1] for m in methods], color=ts.INK)
    ax1.set_ylabel("AMI with the classes at k_ref (0 = chance)", color=ts.INK_2)
    ax1.set_title(
        "Do the clusters match the classes? (black bar = median)",
        loc="left",
        fontsize=11,
        color=ts.INK,
    )
    ax1.legend(frameon=False, fontsize=9)
    err = np.abs(np.log(d["G3_DEC"]) - np.log(d["k_true"]))
    for t, (color, marker) in style.items():
        m = d["type"] == t
        ax2.plot(
            d.loc[m, "ami_DEC"],
            err[m],
            linestyle="none",
            marker=marker,
            color=color,
            markersize=6,
            alpha=0.85,
            label=t,
        )
    ax2.set_xlabel("AMI of DEC with the classes at k_ref", color=ts.INK_2)
    ax2.set_ylabel("|log G3_DEC - log k| (0 = probe exactly right)", color=ts.INK_2)
    ax2.set_title(
        "Is the probe right where the classes are clusters?",
        loc="left",
        fontsize=11,
        color=ts.INK,
    )
    ax2.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "POST-HOC (c), diagnosis with the true labels - not a meta-feature",
        x=0.01,
        ha="left",
        fontsize=11,
        color=ts.INK,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="POST-HOC (c): cluster-class agreement."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N pool datasets (test)"
    )
    args = parser.parse_args(argv)

    pool = pd.read_csv(POOL)
    meta = pd.read_csv(METAFEATURES)
    probe = pd.read_csv(PROBE_TABLE)
    n_train = pd.read_csv(PER_SEED).groupby("name")["n_train"].first()
    k_true = meta.set_index("name")["k_true"]
    specs = pool.to_dict("records")[: args.limit]

    rows, runs, t0 = [], [], time.time()
    for i, spec in enumerate(specs):
        row, per_run = agreement_one(
            spec, int(k_true[spec["name"]]), int(n_train[spec["name"]])
        )
        rows.append(row)
        runs.append(per_run)
        print(
            f"{i + 1:3d}/{len(specs)} {spec['name']}: AMI DEC {row['ami_DEC']:.2f}, IDEC {row['ami_IDEC']:.2f}, "
            f"k-means {row['ami_kmeans_raw_kref']:.2f} ({time.time() - t0:.0f} s)",
            flush=True,
        )
    per_dataset = pd.DataFrame(rows)
    per_run = pd.concat(runs, ignore_index=True)
    summary, merged = summarize(per_dataset, meta, probe)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_dataset.to_csv(
        OUT_DIR / "agreement_per_dataset.csv", index=False, float_format="%.4f"
    )
    per_run.to_csv(OUT_DIR / "agreement_per_run.csv", index=False, float_format="%.4f")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(merged, OUT_DIR / "fig_cluster_agreement.png")
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "status": "POST-HOC (designed after the Phase 4 results); diagnosis with the true labels",
        "design": "docs/metalearner_methodology.md §12.13 (c)",
        "datasets": len(per_dataset),
        "complete": args.limit is None,
        "runtime_s": round(time.time() - t0),
        "pool_sha256": ts.sha256(POOL),
        "probe_features_sha256": ts.sha256(PROBE_TABLE),
        "per_seed_sha256": ts.sha256(PER_SEED),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (OUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved {OUT_DIR}\n")
    print(
        json.dumps(
            {k: summary[k] for k in ["by_type", "rho_ami_vs_probe_error", "reading"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
