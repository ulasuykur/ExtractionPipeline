#!/usr/bin/env bash
#
# run_posthoc_c.sh - POST-HOC part (c) (methodology §12.13): agreement between the clusters of the saved
# Phase 3 probe runs and the true classes. CPU only - no new DC runs. Needs the raw probe outputs
# (outputs/phase3/full/, only on the cluster) and the OpenML cache. Submit from the repo root on madeira
# (after git pull):
#   sbatch slurm/run_posthoc_c.sh
#
#SBATCH --job-name=posthoc-c
#SBATCH --partition=minor
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -eo pipefail   # stop at the first failing command

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

export OPENML_CACHE_DIR=/nfs/data8/$USER/openml-cache

echo "== POST-HOC (c) on $(hostname), commit $(git log -1 --format=%h)"
python posthoc_cluster_agreement.py
echo "Done: $(date)"
