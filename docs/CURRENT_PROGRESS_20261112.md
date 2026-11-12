# TempoRAC live continuation — 2026-11-12

This is a running experiment snapshot, not a completed scientific result. The latest explicit user resumption supersedes the06:40UTC pause. Use `aris-work/current_status.json`, actual server process/log receipts and immutable source snapshots for later updates.

| Item | Actual state | Evidence / remaining work |
|---|---|---|
| Inherited research |15 framework/version entries;29 original complete runs and all cached features preserved | Literature ledger and original predictions are inherited; known source-overlap and selection qualifications remain |
| Durable checkpoints | Server engineering and bounded fresh Sol review complete; actual formal first epoch checkpoint saved | `evidence/resume_20261112/checkpoint/REVIEW.md`, `ENGINEERING_REPORT.json`, `first_checkpoint_full_s0.json` |
| Corrected main model | GPU0 full_s0 running,50 epochs; full_s2 finite successor | worker399036/trainer399051, source489dc77; old12/13epoch attempts never saved weights and are not falsely called resumed |
| Corrected population | RC756/UCF331/CX4222 unique training videos;8570 balanced entries/epoch | Manifest96c068ee..., fixed validation131/90/1374; evaluation unchanged RC152/UCF105/CX2403/QUVA100 |
| Official external comparison | ESCounts released-weight results inherited; RepNet updated2025 ckpt-70 awaiting isolated environment installation and full357 inference | Exact official core/checkpoint plus harmonized raw-prediction scoring; other literature rows are not our reproductions |
| Resource order | GPU0 s0 then s2; GPU1 RepNet then s1 after complete success and free-resource verification | No remaining24 release, no scheduler, no other-project process changes |
| Method and metrics | Existing frozen framework, single density MSE; primary positive rounded NMAE with denominator g+0.1, OBO/OBZ/RMSE and /g/all/zero diagnostics | No architecture, split, metric, seed or hyperparameter change during the group |
| SOTA / component contributions / final paper | Not yet accepted | Require complete matched seeds, official comparisons and paired ablation evidence; no incomplete aggregate or hidden failed domain |

Training source is immutable `489dc77b91a21d21215784ff49961edb20dab4fb`. The later RepNet authorization-only update is `bb588614a589d8a1c45298875ec1379d0af977a1`; it does not alter the active training snapshot. Both have separately verified server snapshots. Each run keeps its exact source/config and result identity; publication metadata may advance without changing a running job.

New experiments run only on the server. E: holds editing/control/evidence. Existing caches/checkpoints/negative results are preserved. A component or research route can be retired without deleting the evidence that motivated that decision. Final manuscript claims wait for the complete group; existing historical draft build/layout repair is not final acceptance.
