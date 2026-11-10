#!/bin/bash
# Language tempo selector, blending protocol: the selector learns from the stage-1 decoders' held-out
# validation predictions; every validation number is 5-fold cross-validated over validation videos
# (grouped by source video); out-of-fold training rows are a biased source (Countix train/val shift) and
# are only used in the comparison runs.  No test split is touched.
set -u
R=/root/autodl-tmp/TempoRAC; O=$R/runs/tsel
cd $R/code
export PYTHONPATH=$R/third_party/hf457:. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$R/env/bin/python
until grep -q VIEWS_DONE $R/logs/views.log 2>/dev/null; do sleep 60; done
$PY -m temporac_cvpr.tempo_select build $O > $O/build2.log 2>&1
for V in 1x,2x half,1x,2x; do
  $PY -m temporac_cvpr.tempo_select knncv $O --views $V --val-only > $O/knncv_valonly_$V.log 2>&1
  $PY -m temporac_cvpr.tempo_select knncv $O --views $V > $O/knncv_oof_$V.log 2>&1
done
$PY -m temporac_cvpr.tempo_select llmcv $O --views 1x,2x --epochs 4 --val-only > $O/llmcv_2_desc.log 2>&1
$PY -m temporac_cvpr.tempo_select llmcv $O --views half,1x,2x --epochs 4 --val-only > $O/llmcv_3_desc.log 2>&1
echo CV_PHASE1_DONE >> $O/chain2.log
