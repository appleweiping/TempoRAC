# Data readiness — new server, 2026-11-04

The new server inventory found an existing larger MultiRep candidate with **794 train / 183 validation / 180 test video directories**, with corresponding MP4 and JSON counts. This is 1,157 videos in total. The inventory recorded filenames and sizes, without opening test annotations.

This is a **candidate release**, not yet a certified protocol. Matching the paper's aggregate video count does not prove release identity, a correct split, or source-video disjointness. The observed split proportions also require reconciliation with the paper's stated split description. Different composites can reuse source clips while having different video hashes.

A complete filename-only audit split composite directory names on `+`. All components matched the observed source-name patterns. It found 174 / 85 / 121 unique source-name proxies in train / validation / test: train–validation intersection 0, train–test intersection 36, validation–test intersection 42. This is strong evidence that the supplied split is not source-name-disjoint; exact-byte source identity is now audited below; recording identity and synthesis recipes remain unresolved. No test JSON or label value was opened for this check. Preserve the original benchmark and report this limitation; do not silently delete overlapping or difficult test cases to improve a score.

Two other directories named `data/MultiRep` contain only **14 / 13 / 14** videos. They are not complete main-experiment inputs. A `MultiRep_lite` path and dummy-feature generator were excluded. These distinctions prevent accidentally treating a convenient small directory as the full benchmark.

RepCount candidates contain **758 train / 131 validation / 152 test videos**. Train and validation annotation row counts were consistent with these inventories. Test annotation contents were not opened. Pose caches found nearby are incomplete or belong to other settings; their existence is not evidence that all inputs for TempoRAC are ready.

The full read-only train/validation MP4 byte audit completed **977/977**, **4,621,403,934 bytes**, zero read/header failures, in 1,251.508 seconds. It found **four exact-byte duplicate pairs within train** and no train/val exact-byte duplicate. It does not establish full decoding, recording independence or model efficacy. Sanitized completion receipt: `docs/evidence/media_integrity_20261104.json`. The private original inventory JSON was copied to the isolated TempoRAC server root and its SHA-256 matched the E-disk copy (`b5a866abd8e35f144f4fe7f0e8b64db7e92272ed1e5cbe0955b840ba7a57bd7a`).

Before training, reconcile the official release/split, extract source-video provenance and verify that no source crosses an independent evaluation boundary. Then build complete pose/track inputs while retaining physical frame clocks and observation support. Do not copy existing test labels into the learner or tune on this test split. There is no formal train/evaluate result in this readiness report.

Raw paths, complete file lists and process IDs remain in the private execution directory. Existing datasets and environments were inspected read-only. No STaR-Body process, other user's process or shared dataset was changed.

## Full source audit and label conflicts — verified continuation

The source audit completed all **906** distinct relevant source files (8,405,836,827 bytes), yielding **302 tokens / 301 canonical byte identities**. `stu3_27` and `stu4_27` have identical original bytes across all three source trees. Full A180 remains preserved: **78 B candidates / 102 overlap / 0 byte-path unknown**. The full graph has 45 components; B candidates touch only **9**, so 78 is not an independent-cluster sample size. Historical exposure and recording-level/synthesis identity remain unresolved. See `SOURCE_INVENTORY_20261104.md` and `evidence/source_identity_20261104.json`.

A targeted train-only comparison of the eight annotations for the four exact duplicate pairs confirmed count/period disagreement for the aliased person (**58 vs 48**), with equal media metadata, bboxes and periodicity; other people agree. No correctness is inferred for either label. All official records and both conflicting values are retained. This does not block the bbox-only frontend projection, whose used fields agree. Any count-supervised baseline/evaluation interpretation must retain and disclose the conflict. Evidence: `evidence/duplicate_annotations_20261104.json`. No test annotation has been read.

The complete **977-video CPU decode/clock/dimension audit is running**, PID 1543699, nice19/ionice2:7, one decoder thread, GPU disabled. The latest public receipt is a timestamped snapshot in `evidence/decode_progress_20261104.json`; current server status and private `current_status.json` take precedence. A failed/partial decode cannot count as successful after the fresh Sol review correction (16 CPU tests passed). Raw PTS/DTS and best-effort ticks remain separate. The reused earlier media SHA lacks stat metadata, so same-size changes between hash and decode are not excluded.

The existing offline native COCO17 ViTPose snapshot was inventoried read-only. Astra fixed the dynamic-N raw cache / bbox-only projection contract in `NATIVE_POSE_CONTRACT_20261104.md`; extraction implementation and its independent technical review are in progress. No full pose extraction, learner bridge, natural target qualification or formal training has completed. Raw cache generation does not wait for the old320 bridge; formal training requires both input and scientific protocol gates.


## Native projection / extractor handoff

Full bbox-only projection completed:977 videos/2364 fixed identities,695625 native annotation frames,1758034 person-frames. Source JSON count/period values do not enter the40MB projection. All supplied records and byte aliases remain. Sanitized hash/census receipt: `evidence/native_projection_20261104.json`. Fresh Sol/xhigh found two bounded resume defects; both fixed and independently rechecked (12CPU tests, Ruff pass). Initial findings/reproductions remain in `NATIVE_POSE_REVIEW_20261104.md`. This is same-family provisional engineering assurance; the real model/device integration and complete raw generation are not yet attested at this code checkpoint.

Astra reconciled exact official evaluator source and paper formulas in `METRIC_CONTRACT_20261104.md`: conditional-only PeriodAP is a documented limitation; official count implementation and conversion chain remain unresolved. Named full-population reference metrics are proposals, not an official/main metric freeze. A separate CPU reference implementation is being prepared without changing existing metrics.
