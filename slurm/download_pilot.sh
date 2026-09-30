#!/usr/bin/env bash
#
# download_pilot.sh - fetch the 12 pilot datasets from OpenML into the cache on /nfs/data8,
# so that the GPU jobs do not wait for downloads. No GPU needed.
# Submit from the repo root on madeira:   sbatch slurm/download_pilot.sh
#
#SBATCH --job-name=download-pilot
#SBATCH --partition=minor
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -eo pipefail

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

export OPENML_CACHE_DIR=/nfs/data8/$USER/openml-cache   # large; not in the home directory

python run_probes.py --pilot --download-only
echo "Done: $(date)"
