set -eu
B=/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites
export CUDA_VISIBLE_DEVICES=
export TMPDIR=$B/tmp TORCH_EXTENSIONS_DIR=$B/torch-extensions
export CUDA_HOME=$B/cuda-12.4
export PATH=$CUDA_HOME/bin:$B/env/bin:/usr/bin:/bin
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.9
export CPATH=$(find /root/autodl-tmp/TempoRAC/env/lib/python3.10/site-packages/nvidia -type d -name include -printf '%p:')
mkdir -p "$B/cuda-12.4/lib64"
ln -sf /root/autodl-tmp/TempoRAC/env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 "$B/cuda-12.4/lib64/libcudart.so"
cd "$B/candidate/DeTRC/model"
"$B/env/bin/python" setup.py build_ext --inplace