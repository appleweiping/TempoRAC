#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROTO=/media/lenovo/data2/shenjihe/mjx/MotionBERT-main/MotionBERT-main/data/mpcounter_generalization_protocols_seed20260725/unseen_person_count
BIND=/media/lenovo/data2/pams-rac/audits/motionbert-binding-984e3ac-seed20260725-unseen-person-count
PY=/home/lenovo/anaconda3/envs/cja_py310/bin/python
cd "$ROOT"
"$PY" -u scripts/experiments/train_temporac_prototype_calibrator_v21.py \
  --train_data "$PROTO/train.pkl" \
  --val_data "$PROTO/val.pkl" \
  --test_data "$PROTO/test.pkl" \
  --train_predictions "$BIND/base_train/predictions.json" \
  --val_predictions "$BIND/base_val/predictions.json" \
  --test_predictions "$BIND/predictions.json" \
  --output_dir results/autoresearch-v21/seed20260725-unseen-person-count \
  --refit_train_val
