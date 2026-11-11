#!/bin/bash
# Score every final model ONCE on the test splits (after all training and all validation-based decisions).
# usage: final_test.sh
# - stage-1 decoders (adapt_dev/wo_llm_adapted_s*): test predictions of both tempo views on RepCount-A,
#   UCFRep, Countix (checkpoint selected on each benchmark's validation split) and QUVA zero-shot
#   (RepCount-A checkpoint and dataset token); the language tempo selector (tempo_select.py apply) then
#   weights the two views of these predictions.
# - decoder ablations (adapt_abl, adapt_abl_single): same evaluation.
# - alternative LLM design (llm_frozen: video-token LLM refining a frozen decoder, 2 seeds): eval_llm.
set -u
R=/root/autodl-tmp/TempoRAC; F=$R/features; U=$R/data/UNION/manifest_dev.json
cd $R/code
export PYTHONPATH=$R/third_party/hf457:. OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=$R/runs/final_test_decoder.log
ev() {  # run_dir ckpt split tag [extra eval_saved args]
  local run=$1 ck=$2 split=$3 tag=$4; shift 4
  local feat=$($R/env/bin/python -c "import json;print(json.load(open('$run/config.json'))['args']['features'])")
  $R/env/bin/python -m temporac_cvpr.eval_saved --run $run --manifest $U --features $feat --ckpt $ck --split $split \
    --tag $tag "$@" >> $LOG 2>&1
}
for run in $R/runs/adapt_dev/wo_llm_adapted_s* $R/runs/adapt_abl/*_s*; do
  [ -f $run/result.json ] || continue
  for t in rc:rc_valid:rc_test ucf:ucf_dev:ucf_test cx:cx_val:cx_test; do
    IFS=: read short val test <<< "$t"
    [ -f $run/eval_test_$short.json ] || ev $run best_$val.pt $test test_$short
  done
done
for run in $R/runs/adapt_dev/wo_llm_adapted_s*; do  # QUVA zero-shot for the full model
  [ -f $run/eval_quva.json ] && continue
  $R/env/bin/python -m temporac_cvpr.eval_saved --run $run --manifest $R/data/QUVA/manifest.json \
    --features $F/QUVA_stmaeA_b8_p4 --aux $F/QUVA_stmaeA_p4 --ckpt best_rc_valid.pt --split test --tag quva \
    --domain RepCountA >> $LOG 2>&1
done
for run in $R/runs/adapt_abl_single/wo_joint_*_s*; do
  [ -f $run/result.json ] || continue
  short=$(basename $run | sed -E 's/wo_joint_([a-z]+)_s[0-9]+/\1/')
  case $short in rc) test=rc_test;; ucf) test=ucf_test;; cx) test=cx_test;; esac
  [ -f $run/eval_test_$short.json ] || ev $run best.pt $test test_$short
done
for d in $R/runs/llm_frozen/s*_count; do
  [ -f $d/result.json ] || continue
  [ -f $d/eval_llm_rc_test.json ] && continue
  $R/env/bin/python -m temporac_cvpr.eval_llm $d --splits rc_valid,cx_val,ucf_dev --test rc_test,cx_test,ucf_test >> $R/runs/llm_frozen/final_test.log 2>&1
done
echo FINAL_TEST_DONE >> $LOG
