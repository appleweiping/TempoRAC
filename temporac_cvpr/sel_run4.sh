#!/bin/bash
# one bounded retry of the LLM selector against its two diagnosed weaknesses (pre-registered decision:
# better 5-fold CV mean relative change vs the plain average wins; ties go to the description kNN)
R=/root/autodl-tmp/TempoRAC; O=$R/runs/tsel
cd $R/code
export PYTHONPATH=$R/third_party/hf457:. OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$R/env/bin/python -m temporac_cvpr.tempo_select llmcv $O --views 1x,2x --epochs 2 --val-only --all-seeds --with-ds > $O/llmcv_2_desc_allseeds_ds.log 2>&1
echo LLMCV4_DONE >> $O/chain4.log
