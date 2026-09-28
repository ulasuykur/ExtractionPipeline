#!/usr/bin/env bash
#
# download_pool.sh - fetch all 100 pool datasets (data/pool.csv) into the OpenML cache on /nfs/data8,
# so that the GPU jobs of the full run do not wait for downloads. No GPU needed.
# Submit from the repo root on madeira:   sbatch slurm/download_pool.sh
#
#SBATCH --job-name=download-pool
#SBATCH --partition=minor
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -eo pipefail

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

export OPENML_CACHE_DIR=/nfs/data8/$USER/openml-cache

# one dataset at a time; a failed download is reported but does not stop the others
python -c "import pandas as pd; print('\n'.join(pd.read_csv('data/pool.csv').name))" | while read -r NAME; do
    python run_probes.py --name "$NAME" --download-only || echo "DOWNLOAD FAILED: $NAME"
done
echo "Done: $(date)"
