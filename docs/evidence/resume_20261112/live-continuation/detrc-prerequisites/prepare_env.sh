set -eu
B=/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites
export CUDA_VISIBLE_DEVICES=
export TMPDIR=$B/tmp
export PIP_CACHE_DIR=$B/pip-cache
export TORCH_EXTENSIONS_DIR=$B/torch-extensions
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$B/wheelhouse"
/root/autodl-tmp/TempoRAC/env/bin/python -m venv --system-site-packages "$B/env"
"$B/env/bin/python" -m pip download --no-deps -d "$B/wheelhouse" setuptools==79.0.1 wheel==0.48.0 mmcv==1.7.0 addict==2.4.0 yapf==0.40.1 h5py==3.11.0 terminaltables==3.1.10
"$B/env/bin/python" -m pip install --no-index --find-links "$B/wheelhouse" --no-deps setuptools==79.0.1 wheel==0.48.0 mmcv==1.7.0 addict==2.4.0 yapf==0.40.1 h5py==3.11.0 terminaltables==3.1.10
"$B/env/bin/python" -m pip list --format=json > "$B/ENV_RESOLVED.json"