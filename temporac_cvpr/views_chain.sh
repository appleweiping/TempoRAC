#!/bin/bash
# Per-view predictions (0.5x, 1x, 2x) for the language tempo selector -- training data and validation only:
#  - out-of-fold: each fold decoder on its held-out training videos (checkpoint selected on validation)
#  - validation: the five stage-1 decoders on the three validation splits
# Waits for the fold queue and the rate-1 extraction.  No test split is touched here.
set -u
R=/root/autodl-tmp/TempoRAC; F=$R/features; FE=$F/UNION_stmaeA_b8_p4
cd $R/code
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=$R/logs/views.log
until grep -q R1_DONE $R/logs/extract_r1.done 2>/dev/null && grep -q QUEUE_DONE $R/logs/queue_oof5.log 2>/dev/null; do sleep 120; done
for f in 0 1 2 3 4; do
  run=$R/runs/oof5/fold${f}_s0
  for t in rc:rc_valid:rc_oof ucf:ucf_dev:ucf_oof cx:cx_val:cx_oof; do
    IFS=: read short val split <<< "$t"
    [ -f $run/eval_oof_$short.json ] || $R/env/bin/python -m temporac_cvpr.eval_saved --run $run \
      --manifest $R/data/UNION/folds/manifest_fold$f.json --features $FE --ckpt best_$val.pt --split $split \
      --tag oof_$short --slow >> $LOG 2>&1
  done
done
MAIN=$(ls -d $R/runs/adapt_dev/wo_llm_adapted_s[0-9])
for t in rc:rc_valid ucf:ucf_dev cx:cx_val; do
  IFS=: read short val <<< "$t"
  $R/env/bin/python -m temporac_cvpr.eval_saved --run $MAIN --manifest $R/data/UNION/manifest_dev.json \
    --features $FE --ckpt best_$val.pt --split $val --tag val_$short --slow >> $LOG 2>&1
done
echo VIEWS_DONE >> $LOG
