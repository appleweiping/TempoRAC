# Completed external performance inventory with TransRAC (private)

All values are inherited aggregate scalars. TempoRAC is mean +/- sample SD over three seeds; external retraining rows have one seed. Rates are fractions. NA means unavailable; no new score calculation was performed.

| Method | Dataset | All/positive N | NMAE positive | NMAE all | /gt positive | OBO positive/all | OBZ positive/all | RMSE positive/all | Continuous NMAE all |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TempoRAC corrected full3 | RepCount-A | 152/151 | 0.141997 +/- 0.016629 | 0.426150 +/- 0.025388 | 0.145644 +/- 0.016839 | 0.693157 +/- 0.061176 / 0.688596 +/- 0.060774 | 0.337748 +/- 0.065224 / 0.335526 +/- 0.064795 | 4.772495 +/- 0.084417 / 4.769911 +/- 0.082372 | 0.424684 +/- 0.027215 |
| TempoRAC corrected full3 | Countix | 2403/2403 | 0.243386 +/- 0.007086 | 0.243386 +/- 0.007086 | 0.250028 +/- 0.007554 | 0.695242 +/- 0.004106 / 0.695242 +/- 0.004106 | 0.364406 +/- 0.005236 / 0.364406 +/- 0.005236 | 3.784330 +/- 0.070402 / 3.784330 +/- 0.070402 | 0.256141 +/- 0.006586 |
| TempoRAC corrected full3 | UCFRep | 105/105 | 0.194626 +/- 0.012613 | 0.194626 +/- 0.012613 | 0.199746 +/- 0.012960 | 0.704762 +/- 0.025198 / 0.704762 +/- 0.025198 | 0.431746 +/- 0.023968 / 0.431746 +/- 0.023968 | 2.349931 +/- 0.084938 / 2.349931 +/- 0.084938 | 0.209302 +/- 0.008288 |
| TempoRAC corrected full3 | QUVA | 100/100 | 0.190627 +/- 0.008051 | 0.190627 +/- 0.008051 | 0.192830 +/- 0.008127 | 0.530000 +/- 0.060828 / 0.530000 +/- 0.060828 | 0.196667 +/- 0.035119 / 0.196667 +/- 0.035119 | 3.996144 +/- 0.127490 / 3.996144 +/- 0.127490 | 0.197039 +/- 0.003136 |
| ESCounts released RC weights | RepCount-A | 152/151 | 0.215098 | 0.279472 | 0.220586 | 0.556291 / 0.559211 | NA / 0.250000 | NA / 5.051055 | 0.302096 |
| ESCounts released RC weights | UCFRep | 105/105 | 0.341636 | 0.341636 | 0.348808 | 0.466667 / 0.466667 | 0.219048 / 0.219048 | 4.320494 / 4.320494 | 0.346810 |
| ESCounts released RC weights | QUVA | 100/100 | 0.299129 | 0.299129 | 0.302587 | 0.300000 / 0.300000 | 0.150000 / 0.150000 | 5.730620 / 5.730620 | 0.305032 |
| RepNet released updated2025 ckpt70 | RepCount-A | 152/151 | 0.287127 | 0.482606 | 0.295070 | 0.410596 / 0.407895 | 0.112583 / 0.111842 | 5.198064 / 5.186648 | 0.485385 |
| RepNet released updated2025 ckpt70 | UCFRep | 105/105 | 0.311727 | 0.311727 | 0.319339 | 0.571429 / 0.571429 | 0.400000 / 0.400000 | 3.707136 / 3.707136 | 0.327108 |
| RepNet released updated2025 ckpt70 | QUVA | 100/100 | 0.240458 | 0.240458 | 0.243724 | 0.520000 / 0.520000 | 0.190000 / 0.190000 | 4.312772 / 4.312772 | 0.240930 |
| DeTRC TSN/R50 full80 seed0 | RepCount-A | 152/151 | 0.655132 | 0.913980 | 0.671104 | 0.211921 / 0.210526 | 0.099338 / 0.098684 | 15.167061 / 15.120568 | 0.913980 |
| DeTRC TSN/R50 full80 seed0 | UCFRep | 105/105 | 0.479613 | 0.479613 | 0.489473 | 0.447619 / 0.447619 | 0.085714 / 0.085714 | 6.447591 / 6.447591 | 0.479613 |
| DeTRC TSN/R50 full80 seed0 | QUVA | 100/100 | 0.422834 | 0.422834 | 0.426923 | 0.280000 / 0.280000 | 0.120000 / 0.120000 | 11.480418 / 11.480418 | 0.422834 |
| TransRAC official-code full200 seed1 | RepCount-A | 152/151 | 0.539804 | 0.733621 | 0.550759 | 0.231788 / 0.230263 | 0.099338 / 0.098684 | 14.149158 / 14.104637 | 0.753310 |
| TransRAC official-code full200 seed1 | UCFRep | 105/105 | 0.763908 | 0.763908 | 0.784849 | 0.304762 / 0.304762 | 0.114286 / 0.114286 | 6.799160 / 6.799160 | 0.760221 |
| TransRAC official-code full200 seed1 | QUVA | 100/100 | 0.540076 | 0.540076 | 0.546264 | 0.190000 / 0.190000 | 0.060000 / 0.060000 | 11.740954 / 11.740954 | 0.543623 |

| TransRAC native evaluator | N | Native MAE | Native OBO | Native OBZ | Native RMSE | Meaning |
|---|---:|---:|---:|---:|---:|---|
| RepCount-A | 152 | 0.695672 | 0.223684 | 0.098684 | 7.611885 | RC original density-derived GT; UCF/QUVA original scalar GT; native tensor rounded predictions |
| UCFRep | 105 | 0.763908 | 0.304762 | 0.114286 | 6.799160 | RC original density-derived GT; UCF/QUVA original scalar GT; native tensor rounded predictions |
| QUVA | 100 | 0.540076 | 0.190000 | 0.060000 | 11.740954 | RC original density-derived GT; UCF/QUVA original scalar GT; native tensor rounded predictions |

TransRAC selected native epoch index: 63 (zero-based; never relabelled as completed epoch count). Training loop wall / training compute / validation seconds, as already recorded: 42326.654046339914 / 37159.577751457924 / 5145.595045809401. Recorded training loop wall durations only. GPU0 co-residency with unrelated project observed from01:27UTC (PID572227 approx21GiB), including01:40UTC source_sync ec384f9; at03:17UTC unrelated PID588883 approx19GiB. Timings span shared GPU resources, not isolated GPU throughput; no standalone or matched end-to-end efficiency comparison

Official/native metrics are separated from common scalar scoring. The full original per-population safe metric objects, original final-metric SHA map, checkpoint SHA, zero-count receipt and inherited negative results remain in COMPARISON_SAFE.json. Missing ESCounts RC positive OBZ/RMSE remain NA.

- Inherited aggregate and its 13 common rows are preserved exactly as historical evidence; historical pending-TransRAC and three-family statements inside it describe that earlier packet only.
- Appended full final populations are RC152 (151 positive), UCF105, QUVA100. RC131 validation is checkpoint selection only and never a final performance row.
- TransRAC is own official-code full200 seed1 counting-model retraining with official K400 backbone, corrected RC756 training, and a single common-validation-selected checkpoint; not author released counting-task weights or exact paper-number reproduction.
- Native selected epoch is zero-based. Native evaluator uses density-derived RC GT and scalar UCF/QUVA GT with native tensor rounding; harmonized scalar-label metrics remain separate.
- Existing zero-event, unfavorable, missing-metric and protocol qualifications remain in inherited_aggregate. No old result was recomputed, overwritten or omitted.
- Training, supervision, pretraining, input adaptation, checkpoint selection and budgets differ. No matched-budget, significance, universal superiority, global SOTA, scientific acceptance or submission-readiness claim.
- Four completed external families on three shared domains do not fulfill six datasets by seven convincing top-conference frameworks or six-domain SOTA. Countix still lacks a completed external comparator in this packet.
- Model/backend identity is unattested; same-family assurance is provisional. This is deterministic safe-metadata integration with no additional semantic review.
- No prediction/label/model/feature/checkpoint payloads opened or rehashed; no metric recomputation, experiment, inference, preprocessing, environment change, source/status/Git mutation or other process control.

Inherited protocol and unresolved limitations:

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
