#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/TempoRAC
W="$R/baselines/racnet-author-assets20261112"
test "$#" -eq 2
REVIEW_FILE="$1"
REVIEW_SHA256="$2"
test -f "$REVIEW_FILE"
cd "$W"
export CUDA_VISIBLE_DEVICES=
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$R/cache/racnet-author-assets"
export TMPDIR="$R/tmp/racnet-author-assets"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
exec taskset -c 26 nice -n 10 "$R/env/bin/python" -B acquire_assets.py --control "$W/CONTROL.json" --run --review-file "$REVIEW_FILE" --review-sha256 "$REVIEW_SHA256"
