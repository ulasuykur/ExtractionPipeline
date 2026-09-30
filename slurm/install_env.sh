#!/usr/bin/env bash
#
# install_env.sh - install/update the Python packages of conda env ai-meta-k on the cluster.
# Submit from the repo root on madeira:   sbatch slurm/install_env.sh
# No GPU needed (downloading + compiling clustpy). Long time limit because /nfs is slow.
#
#SBATCH --job-name=install-env
#SBATCH --partition=minor
#SBATCH --output=slurm-%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -eo pipefail   # stop at the first failing command

cd /nfs/data8/$USER/ExtractionPipeline
source /nfs/data8/$USER/miniforge3/etc/profile.d/conda.sh
conda activate ai-meta-k

# pip's download cache (several GB for torch) goes to /nfs/data8, not the small home directory
export PIP_CACHE_DIR=/nfs/data8/$USER/pip-cache

echo "== System facts"
hostname
# (sed -n 1p prints the first line but still reads all output, so the
#  command is never cut off -- head -1 caused a "broken pipe" failure)
ldd --version | sed -n 1p                            # glibc: torch wheels need >= 2.28
if command -v gcc > /dev/null; then                  # C compiler, needed to compile clustpy
    gcc --version | sed -n 1p
else
    echo "gcc NOT FOUND"
fi
python --version
git log -1 --oneline --decorate || true              # which code version is installed

echo "== Install (same command as on the local PC)"
pip install -r requirements-torch.txt -r requirements.txt -c env/constraints.txt

echo "== Snapshot of the installed versions"
pip freeze > env/cluster_pip_freeze.txt
echo "Done: $(date)"
