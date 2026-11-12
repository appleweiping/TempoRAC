# DeTRC prelaunch F1 followup

**PROVISIONAL_PASS: F1 is resolved. Full CPU JPEG production may start under the new frozen identity**, after root provides its matching release and the existing current-capacity gate passes. No idle GPU or repeat prior CPU check is needed for this CPU stage.

Reviewed 2026-11-12 18:49:43 UTC in the original session `/root/detrc_native_prelaunch_review_20261112`. Requested `gpt-6.1-sol` / `xhigh`; actual backend model/effort remain unattested. Same-family acceptance remains provisional. This closes engineering scope only; no scientific outcome is accepted.

New contract: `e41f69c549ec57762f0876595fd8aed282b87a9ebd2687cd6239a4431be52c47`.
Changed `produce_features.py`: `dfd6ad344c1792b3f6e322b20642e163280e7fef740e56cf4374e526f2eeb802`.
Both existing digests were independently read. The old/new raw contract comparison shows only the changed driver identity; method, inputs, resources and scorer remain fixed. RUN_CONTRACT, SAFE_MANIFEST, PUBLIC_ALLOWLIST and the targeted receipt agree. Initial review and prior CPU evidence remain separately retained.

| F1 closure requirement | Verdict | Primary evidence |
|---|---|---|
| Recover final H5 with absent sidecar | Pass | `h5_committed` returns false and `assemble_population` replays validated pickles through `.h5.pending`; repeated interruption remains replayable. |
| Use one criterion for reuse and reservation | Pass | `missing_h5_bytes` and assembly both call `h5_committed`. Existing H5 bytes remain counted while new pending assembly is reserved. |
| Preserve pickles and exact values | Pass, engineering fixture | Raw fixture/log verifies two synthetic FP32 arrays, correct5,283,840-byte reservation, replay, exact arrays, committed reuse and unchanged pickle/receipt bytes. |
| Preserve freeze and scope | Pass, static | Only assembly/persistence factoring changes. New contract differs only in the one driver identity. |

The fix requires a valid completion receipt before H5 reuse. It checks contract, population, row/frame totals, byte size and existing H5 digest. A final H5 without a sidecar is treated as uncommitted and rebuilt from its committed pickles. This directly fixes the original failed-resume path without changing numerical features or the scientific recipe. No additional finding, silent semantic choice or stale ledger row was identified within this affected scope.

The raw targeted fixture uses only two synthetic arrays of3 and2 rows×2048. Its log/receipt records recovery, reuse, exact values and no source-pickle mutation, with zero model calls, GPU dispatches or real video decoding. I read those artifacts and source; I did not rerun the fixture or inherited checks. Earlier CPU receipts retain their original identities, while this targeted receipt binds the new freeze.

The prior F1-related CPU release hold is lifted. Root still owns release and launch; this review did not start production and does not certify complete1244-video output.

GPU/formal work remains pending: real JPEG/native TSN feature completion, exclusive GPU/RAM gates, actual model/kernel execution, batch64 FP32 and extractor memory/pace/budget fit, CUDA recovery, full80 training/selection and RC131/full357 prediction/scoring. Clearing F1 is not a GPU-runtime, scientific-performance or SOTA result.

This followup was limited to original F1 and its bindings. Actions were static artifact reads, existing source/contract digest reads and raw contract comparison. Only this followup JSON/MD was written; original review/CPU artifacts were not overwritten.
