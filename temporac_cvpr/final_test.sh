#!/bin/bash
# Score every final run ONCE on the test splits (after all training has finished).
# usage: final_test.sh LLM_GROUP_DIR   (e.g. /root/autodl-tmp/TempoRAC/runs/llm_frozen)
# - full model: TempoRAC-LLM runs <group>/s{0..4}_count  (eval_llm: density, 2x view, two-view avg, text)
# - QUVA zero-shot: RepCount-selected checkpoint with the RepCount dataset token
# - decoder-level rows (no LLM): adapted base, decoder ablations, per-dataset training (eval_saved)
set -u
R=/root/autodl-tmp/TempoRAC; F=$R/features; G=${1:?llm group dir}; U=$R/data/UNION/manifest_dev.json
cd $R/code
export PYTHONPATH=$R/third_party/hf457:. OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-0}
for d in $G/s*_count; do
  [ -f $d/result.json ] || continue
  $R/env/bin/python -m temporac_cvpr.eval_llm $d --splits rc_valid,cx_val,ucf_dev --test rc_test,cx_test,ucf_test >> $G/final_test.log 2>&1
  $R/env/bin/python -m temporac_cvpr.eval_llm $d --splits rc_valid --test test --skip-val --tag quva \
    --manifest $R/data/QUVA/manifest.json --features $F/QUVA_stmaeA_b8_p4 --aux $F/QUVA_stmaeA_p4 \
    --desc-roots $F/QUVA_desc_bge --domain RepCountA >> $G/final_test.log 2>&1
done
# decoder-level runs (concat head): one evaluation per run and target
ev() {  # run_dir ckpt split tag [domain]
  local run=$1 ck=$2 split=$3 tag=$4 dom=${5:-}
  local feat=$($R/env/bin/python -c "import json;print(json.load(open('$run/config.json'))['args']['features'])")
  $R/env/bin/python -m temporac_cvpr.eval_saved --run $run --manifest $U --features $feat --ckpt $ck --split $split \
    --tag $tag ${dom:+--domain $dom} >> $R/runs/final_test_decoder.log 2>&1
}
for run in $R/runs/adapt_dev/wo_llm_adapted_s* $R/runs/adapt_abl/*_s*; do
  [ -f $run/result.json ] || continue
  for t in rc:rc_valid:rc_test ucf:ucf_dev:ucf_test cx:cx_val:cx_test; do
    IFS=: read short val test <<< "$t"
    [ -f $run/eval_test_$short.json ] || ev $run best_$val.pt $test test_$short
  done
done
for run in $R/runs/adapt_abl_single/wo_joint_*_s*; do
  [ -f $run/result.json ] || continue
  short=$(basename $run | sed -E 's/wo_joint_([a-z]+)_s[0-9]+/\1/')
  case $short in rc) test=rc_test;; ucf) test=ucf_test;; cx) test=cx_test;; esac
  [ -f $run/eval_test_$short.json ] || ev $run best.pt $test test_$short
done
echo FINAL_TEST_DONE >> $G/final_test.log
