#!/bin/bash
# Frozen framework, final step: after both frozen_a queues finish (each run scored its test splits once at the
# end of training), QUVA zero-shot for the five main models, then the tables.
set -u
R=/root/autodl-tmp/TempoRAC; F=$R/features
cd $R/code
export PYTHONPATH=. OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
PY=$R/env/bin/python; LOG=$R/logs/final_frozen_chain.log
until grep -q QUEUE_DONE $R/logs/queue_frozen_a_g0.log 2>/dev/null && grep -q QUEUE_DONE $R/logs/queue_frozen_a_g1.log 2>/dev/null; do sleep 120; done
echo "$(date) queues done" >> $LOG
for run in $R/runs/final_v2/wo_llm_s0 $R/runs/final_v2/wo_llm_s1 $R/runs/final_v2/wo_llm_s2 $R/runs/frozen_a/full_s3 $R/runs/frozen_a/full_s4; do
  [ -f $run/eval_quva.json ] && continue
  $PY -m temporac_cvpr.eval_saved --run $run --manifest $R/data/QUVA/manifest.json --features $F/QUVA_stmae_b8_p4 \
    --aux $F/QUVA_stmae_p4 --ckpt best_rc_valid.pt --split test --tag quva --domain RepCountA >> $LOG 2>&1
done
$PY -m temporac_cvpr.final_tables_frozen $R/runs/final_tables_frozen > $R/runs/final_tables_frozen.log 2>&1
echo "$(date) FINAL_FROZEN_DONE" >> $LOG
