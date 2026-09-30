#!/usr/bin/env bash
#
# check_env.sh - environment test on one GPU: driver, torch, clustpy (runs check_env.py).
# Submit from the repo root on madeira, choosing the GPU type on the command line:
#   sbatch --gres=gpu:nvidia_geforce_rtx_2080_ti:1 slurm/check_env.sh
#   sbatch --gres=gpu:<GTX 1080 Ti type name>:1   slurm/check_env.sh
# (--gres on the command line overrides the default below.)
#
#SBATCH --job-name=check-env
#SBATCH --partition=minor
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

set -eo pipefail

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

echo "== nvidia-smi (driver version and max. CUDA version)"
nvidia-smi
python check_env.py
