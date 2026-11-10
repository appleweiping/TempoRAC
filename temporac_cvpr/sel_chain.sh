#!/bin/bash
# Language tempo selector experiments (training: out-of-fold rows; selection: validation rows; no test).
# Phase 1 (two views, 1x/2x) as soon as the fold decoders finish; phase 2 (three views, + exact 0.5x)
# once views_chain.sh has written the per-view predictions.
set -u
R=/root/autodl-tmp/TempoRAC; O=$R/runs/tsel
cd $R/code
export PYTHONPATH=$R/third_party/hf457:. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$R/env/bin/python; mkdir -p $O
until grep -q QUEUE_DONE $R/logs/queue_oof5.log 2>/dev/null; do sleep 120; done
$PY -m temporac_cvpr.tempo_select build $O > $O/build1.log 2>&1
$PY -m temporac_cvpr.tempo_select knn $O --views 1x,2x > $O/knn2.log 2>&1
$PY -m temporac_cvpr.tempo_select mlp $O --views 1x,2x > $O/mlp2.log 2>&1
$PY -m temporac_cvpr.tempo_select llm $O --views 1x,2x --epochs 4 > $O/llm2_desc.log 2>&1
$PY -m temporac_cvpr.tempo_select llm $O --views 1x,2x --epochs 4 --no-desc > $O/llm2_nodesc.log 2>&1
echo PHASE1_DONE >> $O/chain.log
until grep -q VIEWS_DONE $R/logs/views.log 2>/dev/null; do sleep 120; done
$PY -m temporac_cvpr.tempo_select build $O > $O/build2.log 2>&1
$PY -m temporac_cvpr.tempo_select knn $O --views half,1x,2x > $O/knn3.log 2>&1
$PY -m temporac_cvpr.tempo_select mlp $O --views half,1x,2x > $O/mlp3.log 2>&1
$PY -m temporac_cvpr.tempo_select llm $O --views half,1x,2x --epochs 4 > $O/llm3_desc.log 2>&1
echo PHASE2_DONE >> $O/chain.log
