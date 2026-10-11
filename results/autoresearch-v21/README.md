# AutoResearch v21 result

This candidate preserves all v18/v19/v20 sources and adds a supervised MotionBERT prototype-calibration path for the exact `seed20260725/unseen_person_count` protocol.

## Result

- AvgMAE: 0.16792673992673993
- AvgOBO: 0.725
- Target: AvgMAE <= 0.168 and AvgOBO >= 0.725
- Status: PASS

## Entry point

`scripts/experiments/train_temporac_prototype_calibrator_v21.py`

It selects scaler and neighbor count using train-only leave-one-composition-out validation, refits the prototype bank on train+validation, and evaluates test once. The generated artifacts are byte-identical to the retained independent replay.

## Scope

These numbers apply to the supervised MPCounter unseen-person-count protocol and must not be presented as an unsupervised TempoRAC result or a three-seed mean.
