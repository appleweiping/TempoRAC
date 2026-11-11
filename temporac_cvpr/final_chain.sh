#!/bin/bash
# Final, single test evaluation (all design decisions were made on validation; nothing below changes them):
# 1) wait for the last decoder ablations, 2) score every frozen model on the test splits once (final_test.sh),
# 3) add the test/QUVA rows to the selector table, recompute the selectors' weights for them (their settings
#    are chosen by 5-fold CV on validation only, so the choice is unchanged), 4) apply, 5) tables.
set -u
R=/root/autodl-tmp/TempoRAC; O=$R/runs/tsel; T=$R/runs/final_tables_20261111
cd $R/code
export PYTHONPATH=$R/third_party/hf457:. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
PY=$R/env/bin/python; LOG=$R/logs/final_chain.log
until grep -q QUEUE_DONE $R/logs/queue_adapt_abl.log 2>/dev/null; do sleep 120; done
echo "$(date) final test start" >> $LOG
GPU=0 bash $R/code/temporac_cvpr/final_test.sh
echo "$(date) final test done" >> $LOG
$PY -m temporac_cvpr.tempo_select build $O > $O/build_final.log 2>&1
$PY -m temporac_cvpr.tempo_select knncv $O --views 1x,2x --val-only > $O/final_knncv_desc.log 2>&1
$PY -m temporac_cvpr.tempo_select knncv $O --views 1x,2x --val-only --fields vis > $O/final_knncv_vis.log 2>&1
$PY -m temporac_cvpr.tempo_select dsprior $O --views 1x,2x > $O/final_dsprior.log 2>&1
$PY -m temporac_cvpr.tempo_select knn $O --views 1x,2x > $O/final_knn_oof.log 2>&1
for n in knncv_1x-2x_valonly_desc_h_k32 knncv_1x-2x_valonly_vis_k64 dsprior_1x-2x knn_1x-2x_desc_h_k32 \
         llmcv_1x-2x_desc_valonly_allseeds_ds_s0; do
  $PY -m temporac_cvpr.tempo_select apply $O $n >> $O/final_apply.log 2>&1
done
$PY -m temporac_cvpr.final_tables $T > $T.log 2>&1
echo "$(date) FINAL_CHAIN_DONE" >> $LOG
