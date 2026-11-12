#!/bin/bash
set -euo pipefail
R=/root/autodl-tmp/TempoRAC
W="$R/baselines/transrac_native_v1"
export TMPDIR="$R/tmp/transrac-native"
export XDG_CACHE_HOME="$R/cache/transrac-xdg" XDG_CONFIG_HOME="$R/cache/transrac-config"
export TORCH_HOME="$R/cache/transrac-torch" CUDA_CACHE_PATH="$R/cache/transrac-cuda"
export HF_HOME="$R/cache/transrac-hf" MPLCONFIGDIR="$R/cache/transrac-mpl" PIP_CACHE_DIR="$R/cache/transrac-pip"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_REQUIRE_VIRTUALENV=true CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$W"
"$R/envs/transrac-native310/bin/python" - <<'PY'
import os,sys,tempfile,json
from run import HERE,atomic_json
names=['TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','TORCH_HOME','CUDA_CACHE_PATH','HF_HOME','MPLCONFIGDIR','PIP_CACHE_DIR',
       'PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','PIP_REQUIRE_VIRTUALENV','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
       'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']
atomic_json(HERE/'checks/RUNTIME_SCOPED_ENV.json',{'scoped':{n:os.environ.get(n) for n in names},
            'actual_tempfile_root':tempfile.gettempdir(),'original_HOME_unchanged':True,
            'prefix':sys.prefix,'base_prefix':sys.base_prefix})
assert tempfile.gettempdir()=='/root/autodl-tmp/TempoRAC/tmp/transrac-native'
PY
"$R/envs/transrac-native310/bin/python" -u run.py --contract RUN_CONTRACT.json --phase cpu-preflight
echo CPU_PREFLIGHT_COMPLETE
