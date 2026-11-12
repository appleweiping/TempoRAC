#!/usr/bin/env bash
# Prepared CPU production entry; root supplies one reviewed release file.
set -eu
B=/root/autodl-tmp/TempoRAC/baselines/detrc_native_v1
S=$B/driver
RELEASE=${1:?root-owned release JSON required}
CPUS=$(/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/env/bin/python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
exec nice -n 10 ionice -c 2 -n 7 taskset -c "$CPUS" env \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=$B/candidate:/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/vendor/mmaction2-0.24.1:$S \
  CUDA_HOME=/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/cuda-12.4 \
  TMPDIR=$B/tmp /root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/env/bin/python \
  "$S/run.py" rawframes --work "$B" --release "$RELEASE"