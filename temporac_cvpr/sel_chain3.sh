#!/bin/bash
# after the cross-validated two-view selector (pid given) finishes: the same selector without the description
R=/root/autodl-tmp/TempoRAC; O=$R/runs/tsel
cd $R/code
export PYTHONPATH=$R/third_party/hf457:. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 $1 2>/dev/null; do sleep 60; done
$R/env/bin/python -m temporac_cvpr.tempo_select llmcv $O --views 1x,2x --epochs 4 --val-only --no-desc > $O/llmcv_2_nodesc.log 2>&1
echo CV_NODESC_DONE >> $O/chain3.log
