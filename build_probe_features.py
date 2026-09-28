"""
build_probe_features.py - turn the raw full-run probe outputs into the probe meta-feature table (Phase 3).

For every dataset of data/pool.csv it reads outputs/phase3/full/<name>/ (written by run_probes.py) and
computes the 16 candidate meta-features (G1-G4, L1 from the k-grid; O1-O3 from overcomplete K; each for
DEC and IDEC, averaged over seeds; G4 across seeds). The computation is the SAME code as the pilot
analysis (analyze_pilot.py, rules fixed before any result, methodology §11.3), imported, not copied.

Output:
    data/probe_features.csv        column `name` + the 16 features, one row per COMPLETE dataset.
                                   No k_true or other bookkeeping (the true k must never be an input).
                                   Missing values (e.g. G3 when no grid k keeps all clusters in use)
                                   are left missing; how Phase 4 handles them is decided there.
    results/phase3_full/
        per_seed.csv               the candidates per seed and algorithm (incl. k_true, for analysis only)
        run_summary.csv            per dataset: status, GPU, host, sizes, number of runs, job time
        run_info.json              input fingerprints, versions, list of incomplete datasets

Usage:  python build_probe_features.py                    (refuses if any dataset is incomplete)
        python build_probe_features.py --allow-incomplete (table for the complete datasets only;
                                                           Phase 4 rule P7: evaluate on the same subset)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_pilot import all_candidates, check_complete, read_dataset
from run_probes import ALGORITHMS

POOL_PATH = Path("data") / "pool.csv"
IN_DIR = Path("outputs") / "phase3" / "full"
TABLE_PATH = Path("data") / "probe_features.csv"
RESULTS_DIR = Path("results") / "phase3_full"
FEATURES = [
    f"{c}_{a}"
    for c in ["G1", "G2", "G3", "G4", "L1", "O1", "O2", "O3"]
    for a in ALGORITHMS
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=IN_DIR)
    parser.add_argument("--table", type=Path, default=TABLE_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    names = list(pd.read_csv(POOL_PATH).name)
    data, summary, problems = {}, [], []
    for name in names:
        folder = args.input / name
        if not (folder / "final.csv").exists():
            problems.append(f"{name}: no results")
            summary.append({"name": name, "status": "missing"})
            continue
        d = read_dataset(folder)
        issues = check_complete(name, d)
        info = d["info"]
        summary.append(
            {
                "name": name,
                "status": "incomplete" if issues else "complete",
                "gpu": info.get("gpu"),
                "host": info.get("host"),
                "n_train": info.get("n_train"),
                "p_used": info.get("p_used"),
                "runs_done": len(d["final"]),
                "job_total_s": info.get("total_s"),
            }
        )
        if issues:
            problems.extend(issues)
        else:
            data[name] = d

    print(f"{len(data)} of {len(names)} datasets complete")
    if problems:
        print(
            "Problems:\n  "
            + "\n  ".join(problems[:30])
            + ("\n  ..." if len(problems) > 30 else "")
        )
        if not args.allow_incomplete:
            raise SystemExit(
                "incomplete full run - rerun the missing tasks, or use --allow-incomplete"
            )

    cands, per_seed = all_candidates(data)
    table = cands[
        ["name", *FEATURES]
    ]  # no k_true / bookkeeping columns: they must never be inputs
    print(
        "Missing values per feature:",
        {f: int(n) for f, n in table[FEATURES].isna().sum().items() if n},
    )

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table, index=False)
    per_seed.to_csv(args.results / "per_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(args.results / "run_summary.csv", index=False)
    inputs = sorted(args.input.glob("*/final.csv")) + sorted(
        args.input.glob("*/checkpoints.csv")
    )
    (args.results / "run_info.json").write_text(
        json.dumps(
            {
                "datasets_complete": len(data),
                "datasets_total": len(names),
                "problems": problems,
                "table": str(args.table),
                "table_sha256": sha256(args.table),
                "input_sha256": {
                    str(p.relative_to(args.input)): sha256(p) for p in inputs
                },
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            indent=2,
        )
    )
    print(
        f"Saved {args.table} ({len(table)} rows x {len(FEATURES)} features) and {args.results}/"
    )


if __name__ == "__main__":
    main()
