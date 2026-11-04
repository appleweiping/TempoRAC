# MultiRep candidate protocol audit

The complete candidate's 794 training and 183 validation JSON files were inspected. They contain 1,878 and 486 person objects respectively. Every inspected object's count equals its number of period entries, and its bbox sequence length equals the declared video length. These are annotation consistency checks, not decoded-media validation or empirical performance.

The metadata contains a composite video name but no independent original-source ID/URL/segment mapping. Filename components indicate 36 train–test and 42 validation–test source-name intersections, with none between train and validation. Canonical source identity still needs verification. No test annotation was opened. See `DATA_READINESS_20261104.md` and the split recovery note.

The available third-party source snapshot needs protocol reconciliation before use:

- `tools/eval_multirep.py:32–45` invokes bbox matching, accumulation and PeriodAP.
- `mmdet/datasets/multirep_eval_api.py:438–462` accumulates matched GT/detection instances. Unmatched instances do not enter that PeriodAP construction.
- Its `action_ap`, lines 534–564, processes the first configuration; it is not a demonstrated average over all instance-IoU configurations.
- Temporal IoU, lines 650–677, uses continuous `end-start` interval length; AP uses an interpolated precision envelope over temporal thresholds 0.50 through 0.95.
- `tools/prepare_multirep_data.py:116–142` uses inclusive endpoints and produces `periods`/`periodicity_binary`, while the evaluator expects a GT `periodicity` field. The available converter/evaluator also differ in the predicted interval field name. They are not verified as an end-to-end compatible stack.
- A bounded search did not locate the official AvgMAE/AvgOBO calculation. The local TempoRAC metric is normalized absolute error, with persons averaged within videos, videos within each seed, and three seeds averaged. That definition must not be called officially matched until verified.

Retain the available original implementation as a versioned reference. Establish a tested metric/interval contract and a separately identified corrected scorer where necessary, with all methods evaluated consistently. Never silently repair one method's evaluator or drop unmatched/failed samples to improve its score. Exact baseline source hashes and executable parity remain prerequisites for formal comparison.

The original detailed report, full intersection witnesses and source locations are preserved in the private execution workspace. No model, test evaluation, GPU computation or shared dataset mutation occurred in this audit.
