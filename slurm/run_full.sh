#!/usr/bin/env bash
#
# run_full.sh - full probe run (methodology §11; O2 decided by the pilot: overcomplete + full grid):
# one array task per pool dataset (row 0-99 of data/pool.csv), at most 6 at the same time, any free GPU.
# Run slurm/download_pool.sh first. Submit from the repo root on madeira:
#   sbatch slurm/run_full.sh
# A single dataset again (e.g. row 17 after a failure; finished runs are skipped):
#   sbatch --array=17 slurm/run_full.sh
#
#SBATCH --job-name=full
#SBATCH --partition=minor
#SBATCH --gres=gpu:1
#SBATCH --array=0-99%6
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -eo pipefail

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

export OPENML_CACHE_DIR=/nfs/data8/$USER/openml-cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic GPU matrix products (also set in probes.py)

# dataset name = row SLURM_ARRAY_TASK_ID of data/pool.csv
NAME=$(python -c "import pandas as pd; print(pd.read_csv('data/pool.csv').name.iloc[${SLURM_ARRAY_TASK_ID}])")
echo "== task ${SLURM_ARRAY_TASK_ID}: ${NAME} on $(hostname), commit $(git log -1 --format=%h)"
python run_probes.py --name "$NAME" --out outputs/phase3/full
echo "Done: $(date)"
