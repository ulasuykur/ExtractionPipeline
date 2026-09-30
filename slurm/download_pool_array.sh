#!/usr/bin/env bash
#
# download_pool_array.sh - download the pool datasets in parallel: one CPU-only array task per
# OpenML dataset id (91 ids for 100 rows; image variants share their source and are fetched once).
# Datasets already in the cache are skipped quickly. Submit from the repo root on madeira:
#   sbatch slurm/download_pool_array.sh
#
#SBATCH --job-name=dl-pool
#SBATCH --partition=minor
#SBATCH --array=0-90%8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00

set -eo pipefail

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

export OPENML_CACHE_DIR=/nfs/data8/$USER/openml-cache

# task number -> the first pool row of the n-th distinct OpenML dataset id
NAME=$(python -c "import pandas as pd; p = pd.read_csv('data/pool.csv').drop_duplicates('dataset_id'); print(p.name.iloc[${SLURM_ARRAY_TASK_ID}])")
echo "== task ${SLURM_ARRAY_TASK_ID}: ${NAME}"
python run_probes.py --name "$NAME" --download-only
echo "Done: $(date)"
