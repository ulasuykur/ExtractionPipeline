"""
run_probes.py - run all probes for ONE dataset and save the raw results (Phase 3).

Design: docs/metalearner_methodology.md, Section 11 (D12). For one dataset:
    prepare the data once  ->  for each seed: pretrain once  ->  one probe per (algorithm, k)
    for every k of the T1 grid (k <= n/5) and for the overcomplete K = min(50, n/5).
    If K is also a grid value, the same run serves both protocols (same seed -> identical run).

Usage (from the repo root):
    python run_probes.py --pilot --index 3            # pilot dataset no. 3 (0-11), full design
    python run_probes.py --pilot --download-only      # only fetch the 12 pilot datasets (CPU is enough)
    python run_probes.py --name Iris --seeds 0 --grid 2 3 --pretrain-steps 100 --probe-steps 50
                                                      # quick test with reduced settings (-> outputs/phase3/test)

OpenML cache: set the environment variable OPENML_CACHE_DIR (the Slurm script points it to /nfs/data8).

Output folder (git-ignored), one per dataset:
    checkpoints.csv  M1-M9 per seed, algorithm, k and checkpoint step
    final.csv        M10 (silhouette), M11 (runtime) per seed, algorithm, k
    pretrain.csv     pretraining time and final reconstruction error per seed
    labels/          final cluster labels of the training points per run (for M12)
    run_info.json    settings, dataset sizes, versions, GPU, git commit, timings
A restarted job skips the runs that are already in final.csv (resume).
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import probes

POOL_PATH = Path("data") / "pool.csv"
OUT_ROOT = Path("outputs") / "phase3"

# ---------------------------------------------------------------------
# Design (methodology §11.3). Changing any of these changes the experiment.
# ---------------------------------------------------------------------
PILOT = [  # array index 0-11 -> dataset (reason per dataset: §11.3)
    "mnist_784_k2",
    "mnist_784_k3",
    "mnist_784_k5",
    "mnist_784",
    "Iris",
    "banknote-authentication",
    "segment",
    "satimage",
    "yeast",
    "isolet",
    "binary_alpha_digits",
    "CIFAR_10",
]
GRID = [
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    13,
    16,
    20,
    25,
    30,
    40,
    50,
]  # T1 grid, fixed without looking at true k
OVERCOMPLETE_K = 50  # pool maximum; capped at n/5 like the grid
MIN_POINTS_PER_CLUSTER = 5  # k is skipped if k > n_train / 5
SEEDS = [0, 1, 2]
ALGORITHMS = ["DEC", "IDEC"]


def probe_seed(seed: int, k: int) -> int:
    """Seed of one probe run. DEC and IDEC get the SAME seed for the same (seed, k), so they start
    from the same k-means solution and see the same batches: a paired comparison, only the loss differs.
    """
    return seed * 1000 + k


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # git missing or not a repository: not fatal, only bookkeeping
        return None


def versions() -> dict:
    import clustpy
    import sklearn

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "clustpy": clustpy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


# ---------------------------------------------------------------------
# Saving and resuming
# ---------------------------------------------------------------------


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV file (header only when the file is new)."""
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def done_runs(out_dir: Path) -> set[tuple[int, str, int]]:
    """Runs already finished, as (seed, algorithm, k). A run counts as finished once its row is in
    final.csv (written last). Checkpoint rows of unfinished runs (crash mid-run) are removed.
    """
    final_path, ckpt_path = out_dir / "final.csv", out_dir / "checkpoints.csv"
    if not final_path.exists():
        for path in (ckpt_path, out_dir / "pretrain.csv"):
            path.unlink(missing_ok=True)
        return set()
    final = pd.read_csv(final_path)
    done = set(zip(final.seed, final.algorithm, final.k))
    if ckpt_path.exists():
        ckpt = pd.read_csv(ckpt_path)
        keep = [(s, a, k) in done for s, a, k in zip(ckpt.seed, ckpt.algorithm, ckpt.k)]
        ckpt[keep].to_csv(ckpt_path, index=False)
    return done


def check_same_design(out_dir: Path, design: dict) -> None:
    """Refuse to resume into a folder that was started with other settings (no mixed results)."""
    info_path = out_dir / "run_info.json"
    if info_path.exists():
        old = json.loads(info_path.read_text())["design"]
        if old != json.loads(
            json.dumps(design)
        ):  # compare in saved (JSON) form: tuples -> lists
            raise SystemExit(
                f"{out_dir} was started with other settings - use another --out folder"
            )


# ---------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------


def run_dataset(spec: dict, seeds: list[int], grid: list[int], out_dir: Path) -> None:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print(f"== {spec['name']} on {gpu or 'CPU'}", flush=True)

    data = probes.prepare_data(spec)
    n_train = data["n_train"]
    k_max = n_train // MIN_POINTS_PER_CLUSTER
    grid_k = [k for k in grid if k <= k_max]
    overcomplete_k = min(OVERCOMPLETE_K, k_max)
    all_k = sorted(set(grid_k) | {overcomplete_k})
    print(
        f"   n_train={n_train}, n_heldout={data['n_heldout']}, p={data['p_used']}, "
        f"grid={grid_k}, overcomplete K={overcomplete_k}",
        flush=True,
    )

    design = {
        "settings": probes.SETTINGS,
        "seeds": seeds,
        "grid": grid_k,
        "overcomplete_k": overcomplete_k,
        "algorithms": ALGORITHMS,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    check_same_design(out_dir, design)
    info = {
        "dataset": spec["name"],
        "spec": {key: (None if pd.isna(val) else val) for key, val in spec.items()},
        "n_loaded": data["n_loaded"],
        "n_train": n_train,
        "n_heldout": data["n_heldout"],
        "p_used": data["p_used"],
        "k_true": data["k_true"],  # bookkeeping only; not used by any probe
        "design": design,
        "versions": versions(),
        "gpu": gpu,
        "host": socket.gethostname(),
        "git_commit": git_commit(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    (out_dir / "labels").mkdir(exist_ok=True)

    done = done_runs(out_dir)
    for seed in seeds:
        todo = [
            (alg, k) for alg in ALGORITHMS for k in all_k if (seed, alg, k) not in done
        ]
        if not todo:
            print(f"   seed {seed}: all runs already done", flush=True)
            continue
        t0 = time.perf_counter()
        ae = probes.pretrain(data["X_train"], seed, device)
        with torch.no_grad():  # final reconstruction error of the pretrained network (sanity check)
            X_t = torch.from_numpy(data["X_train"]).to(device)
            recon = torch.nn.functional.mse_loss(ae.decode(ae.encode(X_t)), X_t).item()
        pretrain_s = time.perf_counter() - t0
        append_rows(
            out_dir / "pretrain.csv",
            [{"seed": seed, "pretrain_s": round(pretrain_s, 2), "recon_mse": recon}],
        )
        print(
            f"   seed {seed}: pretraining {pretrain_s:.1f} s (reconstruction error {recon:.4f}), "
            f"{len(todo)} probe runs",
            flush=True,
        )

        for alg, k in todo:
            result = probes.run_probe(ae, data, k, alg, probe_seed(seed, k), device)
            key = {"seed": seed, "algorithm": alg, "k": k}
            np.save(out_dir / "labels" / f"seed{seed}_{alg}_k{k}.npy", result["labels"])
            append_rows(
                out_dir / "checkpoints.csv",
                [{**key, **row} for row in result["checkpoints"]],
            )
            append_rows(
                out_dir / "final.csv",
                [
                    {  # written last: marks the run as finished
                        **key,
                        "M10_silhouette": result["M10_silhouette"],
                        "M11_runtime_s": round(result["M11_runtime_s"], 3),
                    }
                ],
            )
            print(f"      {alg} k={k}: {result['M11_runtime_s']:.1f} s", flush=True)

    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["total_s"] = round(time.time() - started, 1)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"== done: {spec['name']} in {info['total_s']} s -> {out_dir}", flush=True)


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--pilot", action="store_true", help="use the 12 pilot datasets (with --index)"
    )
    which.add_argument("--name", help="any dataset name from data/pool.csv")
    parser.add_argument(
        "--index", type=int, help="pilot dataset number 0-11 (Slurm array index)"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="only fill the OpenML cache"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--grid", type=int, nargs="+", default=GRID)
    parser.add_argument("--pretrain-steps", type=int, help="override (tests only)")
    parser.add_argument("--probe-steps", type=int, help="override (tests only)")
    parser.add_argument(
        "--out", type=Path, help="output root (default outputs/phase3/pilot)"
    )
    args = parser.parse_args(argv)

    pool = pd.read_csv(POOL_PATH).set_index("name", drop=False)
    names = PILOT if args.pilot else [args.name]
    missing = [n for n in names if n not in pool.index]
    if missing:
        raise SystemExit(f"not in {POOL_PATH}: {missing}")

    if args.download_only:
        from extractor import load_openml_numeric

        for dataset_id in dict.fromkeys(int(pool.loc[n, "dataset_id"]) for n in names):
            X, _ = load_openml_numeric(dataset_id)
            print(f"cached OpenML {dataset_id}: {X.shape}", flush=True)
        return

    if args.pilot:
        if args.index is None or not 0 <= args.index < len(PILOT):
            raise SystemExit(f"--pilot needs --index 0-{len(PILOT) - 1}")
        names = [PILOT[args.index]]

    # Reduced settings are only for tests: they go to their own folder unless --out is given
    is_test = (
        any(v is not None for v in (args.pretrain_steps, args.probe_steps))
        or args.seeds != SEEDS
        or args.grid != GRID
    )
    if args.pretrain_steps is not None:
        probes.SETTINGS["pretrain_steps"] = args.pretrain_steps
    if args.probe_steps is not None:
        probes.SETTINGS["probe_steps"] = args.probe_steps
    out_root = args.out or (OUT_ROOT / ("test" if is_test else "pilot"))

    spec = pool.loc[names[0]].to_dict()
    run_dataset(spec, args.seeds, sorted(args.grid), out_root / spec["name"])


if __name__ == "__main__":
    main()
