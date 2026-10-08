#!/bin/bash
#SBATCH --partition=research
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RFCRATE_ROOT="${RFCRATE_ROOT:-/mnt/weka/fgeikyan/rf-perception-papers/rf-crate}"
RFCRATE_PYTHON="${RFCRATE_PYTHON:-$RFCRATE_ROOT/local/env-rfcrate/bin/python}"
exec "$RFCRATE_PYTHON" -u "$RFCRATE_ROOT/reproduction/build_hupr_cache.py" "$@"
