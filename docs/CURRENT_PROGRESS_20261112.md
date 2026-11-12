# TempoRAC live continuation — 2026-11-12

This is a running experiment snapshot, not a completed scientific result. Updated from the08:12UTC RepNet completion and08:17UTC full_s1 durable checkpoint receipts. The latest explicit user resumption supersedes the06:40UTC pause. Use `aris-work/current_status.json`, actual server process/log receipts and immutable source snapshots for later updates.

| Item | Actual state | Evidence / remaining work |
|---|---|---|
| Inherited research |15 framework/version entries;29 original complete runs and all cached features preserved | Literature ledger and original predictions are inherited; known source-overlap and selection qualifications remain |
| Durable checkpoints | Server engineering and bounded fresh Sol review complete; actual first epoch checkpoints saved for s0 and s1 | `evidence/resume_20261112/checkpoint/REVIEW.md`, `ENGINEERING_REPORT.json`, `first_checkpoint_full_s0.json`, `first_checkpoint_full_s1.json` |
| Corrected main model | GPU0 full_s0 and GPU1 full_s1 running,50 epochs each; full_s2 finite GPU0 successor | s0 worker399036/trainer399051; s1 worker416529/trainer416540; source489dc77. Old12/13epoch attempts never saved weights and are not falsely called resumed |
| Corrected population | RC756/UCF331/CX4222 unique training videos;8570 balanced entries/epoch | Manifest96c068ee..., fixed validation131/90/1374; evaluation unchanged RC152/UCF105/CX2403/QUVA100 |
| Official external comparison | ESCounts released-weight results inherited; RepNet updated2025 ckpt-70 completed357/357 | `evidence/resume_20261112/repnet-full357/`: exact226 weights, full identities/output hashes/metrics verified;196 predictions reused unchanged,161 newly inferred. Other literature rows are not our reproductions |
| Resource order | GPU0 s0 then s2; GPU1 RepNet completed, then s1 launched08:15UTC after actual success and free-resource verification | `checkpoint/GPU1_LAUNCH_STATUS.json`; no remaining24 release, no scheduler, no other-project process changes |
| Method and metrics | Existing frozen framework, single density MSE; primary positive rounded NMAE with denominator g+0.1, OBO/OBZ/RMSE and /g/all/zero diagnostics | No architecture, split, metric, seed or hyperparameter change during the group |
| SOTA / component contributions / final paper | Not yet accepted | Require complete matched seeds, official comparisons and paired ablation evidence; no incomplete aggregate or hidden failed domain |

Training source is immutable `489dc77b91a21d21215784ff49961edb20dab4fb`. Completed RepNet v4 uses `8facb655a897e01b421bb68b4214f65178f7b009`, with all seven runtime files verified against its clean snapshot. The two exact UCF decoder-tail exceptions preserve bitwise-identical annotated-ROI inputs; the189- and196-record failed attempts and adoption chain remain preserved. Each run keeps its exact source/config and result identity; publication metadata may advance without changing a running job.

## Completed official RepNet comparison

| Frozen evaluation population | Videos | Positive rounded NMAE (+0.1) | Positive OBO |
|---|---:|---:|---:|
| RepCountA test |152 (151 positive)|0.287127|0.410596|
| UCFRep test |105|0.311727|0.571429|
| QUVA test |100|0.240458|0.520000|

These are our full evaluations of the authors' released updated2025 ckpt-70, not our own RepNet training or a reproduction of the2020 original checkpoint. The primary score is `abs(round(pred)-gt)/(gt+0.1)` on positive-count videos, with NumPy ties-to-even rounding; OBO is absolute count error<=1. Full/all/zero-count, OBZ, RMSE, denominator-gt and continuous-output diagnostics are retained in `metrics.json`. RepCount's zero-count case and two ambiguous labelled inputs remain in the full population. Countix is outside this RepNet357 contract. Corrected TempoRAC3-seed results are still pending; historical results must not be pooled into that group.

The next concrete milestone is full_s0's50-epoch completion and unchanged four-population evaluation, followed by the existing full_s2 successor. Full_s1 runs concurrently. At complete three-seed group closure, interpret results once and decide the next frozen group; do not tune against intermediate test results.

New experiments run only on the server. E: holds editing/control/evidence. Existing caches/checkpoints/negative results are preserved. A component or research route can be retired without deleting the evidence that motivated that decision. Final manuscript claims wait for the complete group; existing historical draft build/layout repair is not final acceptance.
