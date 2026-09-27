#!/usr/bin/env bash
#
# run_pilot.sh - O2 pilot (methodology §11.3): one array task per pilot dataset (0-11),
# at most 6 at the same time, all on one GPU type (comparable timings, rule R1): RTX 2080
# (deviation from §11.3, 2026-09-28, before any pilot run: no RTX 2080 Ti free).
# Run slurm/download_pilot.sh first. Submit from the repo root on madeira:
#   sbatch slurm/run_pilot.sh
# A single dataset again (e.g. no. 3 after a failure; finished runs are skipped):
#   sbatch --array=3 slurm/run_pilot.sh
#
#SBATCH --job-name=pilot
#SBATCH --partition=minor
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080:1
#SBATCH --array=0-11%6
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -eo pipefail

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

export OPENML_CACHE_DIR=/nfs/data8/$USER/openml-cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # deterministic GPU matrix products (also set in probes.py)

echo "== task ${SLURM_ARRAY_TASK_ID} on $(hostname), commit $(git log -1 --format=%h)"
python run_probes.py --pilot --index "${SLURM_ARRAY_TASK_ID}"
echo "Done: $(date)"
