# Incremental frozen ablation closure

New completions: wo_shift_s2 and wo_token_s0; total6/21. Both completed50epochs and full RC152/Countix2403/UCF105; no training/inference/cache rerun.

Only wo_shift is a closed three-seed contrast. NMAE is lower better; delta=removal minus full, positive favors full. Mean and sampleSD(ddof=1) use matched seeds0/1/2. Fixed original/shift arithmetic-mean predictions before NumPy rounding.

| Dataset | Full NMAE mean +/- SD | wo_shift NMAE mean +/- SD | Paired delta mean +/- SD | Removal relative degradation | Source-cluster95% CI | Bonferroni18 CI |
|---|---:|---:|---:|---:|---|---|
| rc_test | 0.141996913 +/- 0.016629308 | 0.169563887 +/- 0.012970674 | 0.027566975 +/- 0.014645342 | 19.414% | [0.009379996229567587, 0.048249209808864345] | [0.0010339605714800744, 0.060436590719088246] |
| cx_test | 0.243386496 +/- 0.007086311 | 0.252201503 +/- 0.003544772 | 0.008815006 +/- 0.010573788 | 3.622% | [0.0026464071685128425, 0.01505378483196636] | [-0.0006924083355985944, 0.01821326153613453] |
| ucf_test | 0.194626165 +/- 0.012612571 | 0.215744358 +/- 0.005155106 | 0.021118193 +/- 0.017767271 | 10.851% | [0.0009067678624645808, 0.041447808521795665] | [-0.009656390230245795, 0.0526717523316198] |

Raw seed deltas, all views, full/all/zero metrics and factorial diagnostics are retained in WO_SHIFT_PAIRED3_SAFE.json; paired video rows in WO_SHIFT_PAIRED3_PER_VIDEO.jsonl. Existing full3 summary/statistics and prior arm receipts are reused, not recomputed.

10000 draws yield about14 samples per Bonferroni18 extreme tail; no p-values or confirmatory significance claims.

This is saved-output collection and statistical closure, not independent scientific acceptance. Inherited label ambiguity/test-informed method selection remain. No effectiveness/ineffectiveness or equivalence verdict; remaining groups incomplete. Requested Sol, actual identity unattested, same-family provisional.

Observed raw limitations: Countix seed1 removal-minus-full NMAE is -0.002832460220274099 (the removal performs better for that seed). Countix and UCF Bonferroni18 intervals span zero; RepCount-A adjusted interval is above zero conditional on these fitted models/source units. The broad three-component effectiveness requirement remains unmet.

New completion timestamps:

- wo_shift_s2: 2026-11-12T19:42:05.165783+00:00
- wo_token_s0: 2026-11-12T21:33:05.994447+00:00

Runtime observed 2026-11-12T22:48:09.644024+00:00: GPU1 worker435833/trainer531570, wo_token_s1 36/50 committed epochs; GPU0 original held10 residual remains. No signals, HOLD/source/status changes, new experiment, inference, cache build or worker launch.
