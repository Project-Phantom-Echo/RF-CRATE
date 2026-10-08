#!/bin/bash
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 000
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/rfcrate-mpl
RFCRATE_ROOT="${RFCRATE_ROOT:-/mnt/weka/fgeikyan/rf-perception-papers/rf-crate}"
RFCRATE_PYTHON="${RFCRATE_PYTHON:-$RFCRATE_ROOT/local/env-rfcrate/bin/python}"
exec "$RFCRATE_PYTHON" -u "$RFCRATE_ROOT/reproduction/run_hupr.py" "$@"
