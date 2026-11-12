# Prepared invocation only. DO NOT RUN until P2 rawframe identity receipts and
# P6 resource/80epoch contract are released; current jobs own both GPUs.
set -eu
B=/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites
: "${DETRC_RELEASED_GPU:?P6 must supply an actually allocated GPU}"
: "${DETRC_NATIVE_RAWFRAME_RECEIPT:?P2 must supply complete JPEG/frame identity evidence}"
test -f "$DETRC_NATIVE_RAWFRAME_RECEIPT"
export CUDA_VISIBLE_DEVICES=$DETRC_RELEASED_GPU
export CUDA_HOME=$B/cuda-12.4
export PYTHONPATH=$B/candidate:$B/vendor/mmaction2-0.24.1
export TMPDIR=$B/tmp TORCH_EXTENSIONS_DIR=$B/torch-extensions
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CKPT=$B/assets/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth
printf '%s  %s\n' cc665e2a86857b4ff4873747ae4e98f9cd6de3f72111cf3379954f551cc63c49 "$CKPT" | sha256sum -c -
for POP in rc_train rc_valid rc_test ucf_test test; do
  "$B/env/bin/python" "$B/candidate/tools/data/activitynet/tsn_feature_extraction.py" \
    --data-list "$B/annotations/extract_$POP.txt" --data-prefix '' \
    --output-prefix "$B/feature-pkl/$POP" --frame-interval 1 --modality RGB \
    --ckpt "$CKPT" --part 0 --total 1
  "$B/env/bin/python" "$B/combine_feature_h5.py" --plan "$B/EXTRACTION_INPUT_PLAN.json" \
    --population "$POP" --pickle-dir "$B/feature-pkl/$POP" --output "$B/features/$POP.h5"
done