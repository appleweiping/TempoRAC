# Completed external performance inventory (private, aggregate only)

Corrected full3 is inherited unchanged; DeTRC completed at 2026-11-13T01:21:15.482200+00:00. Raw numbers precede interpretation. All rates are fractions; six decimals. TempoRAC is mean +/- sample SD, not an ensemble. NA identifies fields absent from existing aggregate evidence.

| Method | Dataset | All/positive N | Primary NMAE positive | NMAE all | Relative error /gt positive |
|---|---|---:|---:|---:|---:|
| TempoRAC corrected full3 | RepCount-A | 152/151 | 0.141997 +/- 0.016629 | 0.426150 +/- 0.025388 | 0.145644 +/- 0.016839 |
| TempoRAC corrected full3 | Countix | 2403/2403 | 0.243386 +/- 0.007086 | 0.243386 +/- 0.007086 | 0.250028 +/- 0.007554 |
| TempoRAC corrected full3 | UCFRep | 105/105 | 0.194626 +/- 0.012613 | 0.194626 +/- 0.012613 | 0.199746 +/- 0.012960 |
| TempoRAC corrected full3 | QUVA | 100/100 | 0.190627 +/- 0.008051 | 0.190627 +/- 0.008051 | 0.192830 +/- 0.008127 |
| ESCounts released RC weights | RepCount-A | 152/151 | 0.215098 | 0.279472 | 0.220586 |
| ESCounts released RC weights | UCFRep | 105/105 | 0.341636 | 0.341636 | 0.348808 |
| ESCounts released RC weights | QUVA | 100/100 | 0.299129 | 0.299129 | 0.302587 |
| RepNet released updated2025 ckpt70 | RepCount-A | 152/151 | 0.287127 | 0.482606 | 0.295070 |
| RepNet released updated2025 ckpt70 | UCFRep | 105/105 | 0.311727 | 0.311727 | 0.319339 |
| RepNet released updated2025 ckpt70 | QUVA | 100/100 | 0.240458 | 0.240458 | 0.243724 |
| DeTRC TSN/R50 full80 seed0 | RepCount-A | 152/151 | 0.655132 | 0.913980 | 0.671104 |
| DeTRC TSN/R50 full80 seed0 | UCFRep | 105/105 | 0.479613 | 0.479613 | 0.489473 |
| DeTRC TSN/R50 full80 seed0 | QUVA | 100/100 | 0.422834 | 0.422834 | 0.426923 |

| Method | Dataset | OBO positive/all | OBZ positive/all | RMSE positive/all | Continuous NMAE all |
|---|---|---:|---:|---:|---:|
| TempoRAC corrected full3 | RepCount-A | 0.693157 +/- 0.061176 / 0.688596 +/- 0.060774 | 0.337748 +/- 0.065224 / 0.335526 +/- 0.064795 | 4.772495 +/- 0.084417 / 4.769911 +/- 0.082372 | 0.424684 +/- 0.027215 |
| TempoRAC corrected full3 | Countix | 0.695242 +/- 0.004106 / 0.695242 +/- 0.004106 | 0.364406 +/- 0.005236 / 0.364406 +/- 0.005236 | 3.784330 +/- 0.070402 / 3.784330 +/- 0.070402 | 0.256141 +/- 0.006586 |
| TempoRAC corrected full3 | UCFRep | 0.704762 +/- 0.025198 / 0.704762 +/- 0.025198 | 0.431746 +/- 0.023968 / 0.431746 +/- 0.023968 | 2.349931 +/- 0.084938 / 2.349931 +/- 0.084938 | 0.209302 +/- 0.008288 |
| TempoRAC corrected full3 | QUVA | 0.530000 +/- 0.060828 / 0.530000 +/- 0.060828 | 0.196667 +/- 0.035119 / 0.196667 +/- 0.035119 | 3.996144 +/- 0.127490 / 3.996144 +/- 0.127490 | 0.197039 +/- 0.003136 |
| ESCounts released RC weights | RepCount-A | 0.556291 / 0.559211 | NA / 0.250000 | NA / 5.051055 | 0.302096 |
| ESCounts released RC weights | UCFRep | 0.466667 / 0.466667 | 0.219048 / 0.219048 | 4.320494 / 4.320494 | 0.346810 |
| ESCounts released RC weights | QUVA | 0.300000 / 0.300000 | 0.150000 / 0.150000 | 5.730620 / 5.730620 | 0.305032 |
| RepNet released updated2025 ckpt70 | RepCount-A | 0.410596 / 0.407895 | 0.112583 / 0.111842 | 5.198064 / 5.186648 | 0.485385 |
| RepNet released updated2025 ckpt70 | UCFRep | 0.571429 / 0.571429 | 0.400000 / 0.400000 | 3.707136 / 3.707136 | 0.327108 |
| RepNet released updated2025 ckpt70 | QUVA | 0.520000 / 0.520000 | 0.190000 / 0.190000 | 4.312772 / 4.312772 | 0.240930 |
| DeTRC TSN/R50 full80 seed0 | RepCount-A | 0.211921 / 0.210526 | 0.099338 / 0.098684 | 15.167061 / 15.120568 | 0.913980 |
| DeTRC TSN/R50 full80 seed0 | UCFRep | 0.447619 / 0.447619 | 0.085714 / 0.085714 | 6.447591 / 6.447591 | 0.479613 |
| DeTRC TSN/R50 full80 seed0 | QUVA | 0.280000 / 0.280000 | 0.120000 / 0.120000 | 11.480418 / 11.480418 | 0.422834 |

ESCounts RC positive OBZ/RMSE are unavailable in the inherited aggregate files; all-population OBZ/RMSE remain listed. No predictions or labels were reopened to fill them.

| DeTRC pre-SoftNMS common_ori | All/positive N | Primary NMAE | All NMAE | OBO pos/all | OBZ pos/all | RMSE pos/all |
|---|---:|---:|---:|---:|---:|---:|
| rc_valid | 131/130 | 2.078668 | 3.055167 | 0.084615/0.083969 | 0.023077/0.022901 | 16.942209/16.915596 |
| rc_test | 152/151 | 2.117342 | 2.892885 | 0.033113/0.032895 | 0.006623/0.006579 | 17.465766/17.435407 |
| ucf_test | 105/105 | 1.335507 | 1.335507 | 0.180952/0.180952 | 0.085714/0.085714 | 6.672616/6.672616 |
| test | 100/100 | 0.992728 | 0.992728 | 0.110000/0.110000 | 0.050000/0.050000 | 11.585336/11.585336 |

| DeTRC native author cycles | N | MAE (/gt+1e-6) | OBO | ori_MAE | ori_OBO | mAP (%) |
|---|---:|---:|---:|---:|---:|---:|
| rc_valid | 130 | 0.594492 | 0.292308 | 2.143688 | 0.084615 | 20.400000 |
| rc_test | 151 | 0.670755 | 0.211921 | 2.184168 | 0.033113 | 20.100000 |

Native short/medium/long MAE/OBO, IoU decay, common131 diagnostic and full pre-NMS fields are preserved in COMPARISON_SAFE.json. Native cycles differ from scalar labels on one retained test record; native/common numbers are not interchangeable.

The primary NMAE mean is numerically lower for TempoRAC than all three exact completed baseline settings on RC/UCF/QUVA. RC all-population NMAE remains worse than ESCounts. No external significance or SOTA conclusion follows. Countix has no completed external baseline in this scope. TransRAC/RACnet remain pending performance, not reproduced rows.

## Provenance and protocol

- TempoRAC: {"training": "our corrected known-source-disjoint RC/Countix/UCF joint training;3seeds0/1/2,50epochs each", "training_commit": "489dc77b91a21d21215784ff49961edb20dab4fb", "data_manifest_sha256": "96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2", "checkpoint_selection": "per-target original-view validation (mae_p,-obo_p); QUVA uses RC checkpoint", "checkpoint_sha256_by_run": {"full_s0": {"best_rc_valid.pt": "c5a38ea576e03f13998e99f3ef3f5cab4432522ed6faff7d1a13a94acebfc539", "best_cx_val.pt": "746bd952c5e4e5d1847d806f486e3ad2108164125e4ad8bd10e576b2146437c7", "best_ucf_dev.pt": "4430e62a623ab5cb3a7122b26e97bbf935d93d3618b2a127c8569e73b378a635"}, "full_s1": {"best_rc_valid.pt": "7760f72e05674c3d643a1b79d1be5a255b34e0277c6af9d23d47c64921ddeec5", "best_cx_val.pt": "fc211e50b56b8151bf8f0c5a5f0d86f2698825c85b6afa1cb974ae3eb5fdee67", "best_ucf_dev.pt": "32abbb023ec4d9138e318f72b6a0842a83350d986e63702206dfda03c04293d8"}, "full_s2": {"best_rc_valid.pt": "556b88902c3d775f8246961d202869892c4c35014813ca64dc890ceb8bfa4c8e", "best_cx_val.pt": "c10a39ff27e803cbf5e0d1caa4753492063f047da4e6155f33aa5326a82d394d", "best_ucf_dev.pt": "798adda29a3a03489eec136849b9d825a4e189d02be85115224e433dc8e2acfc"}}, "inference": "fixed mean of original/shift continuous counts before rounding; mean of3 per-seed metrics, sample SD ddof1; no seed ensemble"}
- ESCounts: {"weights": "author released RC task checkpoint; evaluated by us; no own retraining", "task_weight_recorded_sha256": {"path": "pretrained/escounts_repcount.pyth", "bytes": 239106669, "sha256": "297fc53000417ac6ffa4e08d6876e033df89800c97632fa09d2f3945ba225559"}, "backbone_recorded_sha256": {"path": "pretrained/VIT_B_16x4_MAE_PT.pyth", "bytes": 1207498009, "sha256": "b6d1d0b539dbdc992c3a3544a9bc5bbb0591179b615dc728f951285875d824e8"}, "historical_binding_limitation": "Current asset hashes verified; historical per-evaluation hash receipt not located. ESCounts identity inferred from code/launch paths and existing records, not proven by directory name alone.", "transfer": "UCF/QUVA are RC-weight transfer, not native target-trained strongest settings", "implementation": "verified ST-MAE encoder reimplementation + official decoder/two-offset exemplar-free decoding; retain adapter qualification"}
- RepNet: {"weights": "author updated2025 ckpt70 evaluated by us, not original2020 ckpt88 and no own retraining", "checkpoint_sha256": {"checkpoint": "bb7639ce0ba1adbbc8a3f400e43f9c09a1a89989d8e65ae236648641f8f3d680", "ckpt-70.index": "8b42d3dc13646b1ffe8f6bfa08c93e921566899329123109a30547147ba23967", "ckpt-70.data-00000-of-00001": "6caf32eb28e089987ebc3d9ddf5e1e41c45d1c051eed6fd9635f97845956183d"}, "notebook_commit": "60d48fc1b71b6126462136cad441dd76da7460d0", "source_sha256": {"evaluate_repnet.py": "3cb7aeac5ed62e30050c4a6fa75202c63d7460e2210b72a0d5dcf73250737bc7", "repnet_official_core.py": "af46180e5f7671efbd2e95b73630592802ec75752382800f4ba8ad903abc6475", "frozen_metrics.py": "fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0", "env-spec.json": "3a85c459413a5bd0f282a99fc3acd458096c8175c5198f334cd4404fd8746b39", "setup_repnet_env.py": "95f6be9a024e79b05294925b2c5a752bb5708eb11f2c2978a6e117e70e45b49b", "run_official_repnet.py": "d852fe4dd52676ff692cbe0ef64523859c903e73d8ed9b9e2caa1fa9104541fd"}, "config": "full357, fixed notebook parameters,196 adopted+161 newly evaluated predictions"}
- DeTRC: {"weights": "own full80 seed0 counting-model retraining from official TSN/R50 backbone; no acquired author counting task checkpoint", "original_commit": "538abfcd24e0a81b383ca81862ff4360a2389c97", "backbone_sha256": "cc665e2a86857b4ff4873747ae4e98f9cd6de3f72111cf3379954f551cc63c49", "selected_checkpoint_sha256": "6e5d122cce843262bf2335956d3f3089864467674a5cadec731f1d68b98e7f64", "selected_epoch": 53, "selected_criterion": "post-SoftNMS native130 validation OBO strict maximum; earliest tie; common131 diagnostic never selects", "training": "755 supervised RC videos after frozen2 known-source exclusions and native missing-count omission; batch64,80epochs,FP32,52native batches/epoch", "variant": "official released TSN/ResNet50 implementation; not paper strongest ViT or exact paper-number reproduction", "single_weight": "one RC-selected checkpoint for all357 final videos"}
- pending: {"TransRAC": "not included as a completed/reproduced performance row", "RACnet": "not included as a completed/reproduced performance row"}

## Metric definitions

- rounding: p=np.round(pred), NumPy ties-to-even; err=abs(p-gt)
- primary_mae_p: mean(err/(gt+0.1)) on gt>0; lower better
- all_mae: mean(err/(gt+0.1)) on all valid scalar labels including zeros; lower better
- mae_pos: mean(err/gt) on gt>0; distinct from primary and not raw count MAE
- obo_and_obo_p: fraction err<=1 on all/positive population; higher better
- obz_and_obz_p: fraction err==0 on all/positive population; higher better
- rmse_and_rmse_p: sqrt(mean((p-gt)^2)) on all/positive population; lower better
- mae_raw_cont: mean(abs(pred-gt)/(gt+0.1)) on all; normalized continuous diagnostic, not raw count MAE
- DeTRC_common: count positive-class proposals with final score>0.2 after official SoftNMS sigma0.1/top_k30/threshold0.0001; frozen scalar labels
- DeTRC_common_ori: same score>0.2 count before SoftNMS; retains pre-NMS diagnostics, not an alternative test-selected method
- DeTRC_native_RC: author evaluator: valid native cycles only130val/151test; denominator gt+1e-6 and author cycle GT; native mAP percentage, not harmonized counting score
- native_label_disagreement: one retained RC test scalar/cycle-count disagreement; safe packet records count only, private original evidence preserves identities and labels
- RepNet_derived_native: continuous positive relative error /gt and continuous OBO, derived by prior collector; author notebook provides counts without a dataset metric

## Qualifications and retained negative evidence

- Frozen common UCF inclusive temporal ROIs; RC/QUVA full inputs as frozen. ROI availability is a disclosed input adaptation, not independent unknown-boundary evaluation.
- Shared full evaluation populations do not equal matched training, supervision, pretraining, budget, modality or checkpoint-selection protocol.
- Known-source correction is training-only. Historical test-informed design, inherited conflicting evaluation labels, unknown overlap/pretraining exposure remain unresolved.
- TempoRAC uses joint RC/Countix/UCF data and target-selected checkpoints; DeTRC uses native RC755/130 selection and a single RC-selected checkpoint; ESCounts transfers released RC weights; RepNet uses released synthetic/pretrained model.
- RC all-population TempoRAC NMAE0.426150442 exceeds ESCounts0.279472368; positive-only numerical gains do not establish universal superiority or zero-event accuracy.
- Original-only UCF TempoRAC NMAE0.190705453 is lower than fixed-two-view0.194626165; retain fixed inference and prior negative diagnostic.
- Three completed external families across three common domains, plus TempoRAC Countix-only row, do not satisfy6datasets x7convincing external frameworks. DeTRC journal/variant qualification remains; not every completed family meets requested top-conference qualification.
- Inherited stronger literature rows and protocol caveats remain in group-end-astra/DECISION.md: no new literature review or global ranking here. User preference not to place non-SOTA domains in the eventual main table is unresolved; this private inventory preserves all results.
- No external paired uncertainty analysis, equal-budget superiority, statistical significance, global SOTA, scientific acceptance or submission readiness claimed.
- Source small receipts/aggregate/source files hashed only. Model/video/feature/prediction/label payloads not read or rehashed. No rescoring, inference, training, GPU calls, signals, restarts, queue/code/config/shared-status or Git mutation.

## Acceptance

| Requirement | Outcome | Evidence |
|---|---|---|
| DeTRC_full80_full357 | PASS | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |
| metric_definition_alignment | PASS | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |
| existing_aggregate_metric_inventory | PASS_WITH_EXPLICIT_ESCOUNTS_RC_POSITIVE_OBZ_RMSE_UNAVAILABLE | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |
| negative_results_retained | PASS | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |
| weight_provenance | PARTIAL_ESCOUNTS_HISTORICAL_EVALUATION_HASH_BINDING_GAP | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |
| six_datasets_seven_external_frameworks | UNMET | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |
| six_domain_SOTA | UNMET | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |
| external_significance | NOT_ESTABLISHED | COMPARISON_SAFE.json / SOURCE_BINDINGS.json |

This text/JSON inventory is not a Word/PDF deliverable; no paper or pagination was changed. Requested executor gpt-6.1-sol; actual backend identity unattested.
