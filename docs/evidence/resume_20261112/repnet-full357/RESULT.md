# Official RepNet full357 result

Completed 2026-11-12 08:11:13 UTC using released updated-2025 checkpoint ckpt-70 and the literal official notebook core. This is our evaluation of the released method, not an author-reported paper number or training on these datasets. Frozen runtime source: `8facb655a897e01b421bb68b4214f65178f7b009`.

| Frozen population | Evaluated | Positive MAE (+0.1 denominator) | Positive OBO |
|---|---:|---:|---:|
| RepCountA rc_test | 152/152 (151 positive) | 0.287127 | 0.410596 |
| UCFRep ucf_test | 105/105 | 0.311727 | 0.571429 |
| QUVA test | 100/100 | 0.240458 | 0.520000 |

These harmonized metrics use NumPy ties-to-even rounding, `abs(round(pred)-gt)/(gt+0.1)` and OBO error <=1. RepCountA all152 MAE is **0.482606** and OBO **0.407895**. The retained zero-count video train143 has native continuous prediction 2.992647 and rounded prediction 3. Both disputed frozen labels are retained: stu4_5 gt51, stu4_27 gt56.

Positive MAE with denominator gt (rounded predictions) is 0.295070 / 0.319339 / 0.243724 for RepCountA/UCFRep/QUVA. Derived continuous-output positive MAE over gt is 0.298686 / 0.335254 / 0.244218; derived continuous OBO is 0.298013 / 0.476190 / 0.360000. The official notebook returns continuous counts and supplies no dataset scorer; these derived metrics are not claimed as official paper metrics.

| Acceptance item | Status | Evidence |
|---|---|---|
| Released weights and literal official method | Pass | Full226 variables restored exactly; witness-v2 receipt SHA a4b106b4fc38b858d4ddc693fb938202d0c5d7dc40f9b070032fe0b82629f38f; immutable notebook/core provenance in earlier reviewed receipts |
| Full frozen357 coverage and finite predictions | Pass | runtime-v4/completion.json; predictions.jsonl; exact dataset/split/video/media-SHA/gt identity check against frozen manifests |
| No rerun of196 saved predictions | Pass | Adoption196 receipt; original/new record and trace hashes matched V3_ADOPTION_MANIFEST; only161 remaining inferred |
| Decoder/ROI contract | Pass with disclosed bounded exceptions | Exactly2 pinned UCF files have an extra frozen decoder tail duplicate outside the annotated ROI; native ROI pixels bitwise identical. Media/meta/ROI SHAs pinned in config and DECODE_FAULT_DECISION.md. No QUVA exception. Historical v2/v3 failures remain intact |
| Frozen metrics and zero/disputed rows retained | Pass | All population metrics recomputed from final357 predictions and matched; metrics.json includes individual zero/disputed records |
| Isolated server execution and GPU release | Pass | Env/cache under server project root; worker412971/evaluator412972 absent and GPU1 compute empty at08:12:12Z. GPU0 training untouched |
| Clean published source/runtime byte match | Pass | runtime-v4/V4_SOURCE_RUNTIME_VERIFICATION.json: all7 runtime files equal clean8fac snapshot |
| Broader top-conference/SOTA research quality | Not established by this result | This receipt establishes one external-method evaluation on3 populations; it does not certify the full research matrix, significance, SOTA or publication readiness |

Final evidence is mirrored locally in `E:/ICASSP_TempoRAC/aris-work/resume-20261112/baselines/runtime-v4/` and published beside this report. The original complete per-video JSON and NPZ traces remain on the server under `/root/autodl-tmp/TempoRAC/external-baselines-20261112/runs/repnet-full357-v4`.

`repnet-full357-v4-evidence.zip` additionally preserves all357 records,357 prediction traces and19 receipts/logs/provenance files (733 files,1,034,173 archive bytes). The archive SHA256 is `686db1be7df56a2dc6eb7c9e4892892a8048235a48952b7ff1d0094266ded6d3`. Every member was restored on E: and matched against the server inventory by size and SHA256. See `V4_EVIDENCE_TRANSFER_MANIFEST.json` and `V4_CROSS_LOCATION_PRESERVATION_RECEIPT.json`. No video, feature cache or checkpoint is packaged in this result archive.

- FULL357_RECEIPT.json SHA256: `86675021a558c60693cda33d4590a94d0abd2d81f5be033ba626002ed58b9357`
- RAW_PREDICTION_MANIFEST.json SHA256: `cf0d940f3c64cb5faea3f5afabce276bb14f3acec651be06028939e630589cee`
- predictions.jsonl SHA256: `e48ac04cf47d1f42b81b75fc2e8a372f2741095d46ffadbf5b914deb55703d93`
- metrics.json SHA256: `154ccf020b3e037f37dfd23e5ea8118c47a50eb2c8a2d9ae31cae9aa4f2dfa6a`
- completion.json SHA256: `d92ced6fd3e57fbf3ce8d50ad5d6716f1b11003deb35e05661b42c2ae9d1483f`

Only postcompletion integrity verification was run after the finite job. No retraining, extra inference witness, full-video recache, split/label/metric change, scheduled timer or recurring task was added.
