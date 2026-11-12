# TempoRAC live continuation — 2026-11-12

This is a running experiment-group snapshot, not a completed scientific verdict. Updated from the08:48UTC full_s0 completion and08:51UTC live handoff receipts. The latest explicit user resumption supersedes the06:40UTC pause. Use `aris-work/current_status.json`, actual server process/log receipts and immutable source snapshots for later updates.

| Item | Actual state | Evidence / remaining work |
|---|---|---|
| Inherited research |15 framework/version entries;29 original complete runs and all cached features preserved | Literature ledger and original predictions are inherited; known source-overlap and selection qualifications remain |
| Durable checkpoints | Server engineering and bounded fresh Sol review complete; actual first epoch checkpoints saved for all three runs | `evidence/resume_20261112/checkpoint/REVIEW.md`, `ENGINEERING_REPORT.json`, `first_checkpoint_full_s0.json`, `first_checkpoint_full_s1.json`, `first_checkpoint_full_s2.json` |
| Corrected main model | full_s0 completed50 epochs and all four evaluations; full_s1 and full_s2 running | s0 complete08:48:26UTC; existing GPU0 worker399036 continued to s2 trainer419980; GPU1 s1 worker416529/trainer416540. At08:51:48UTC s1=17/50,s2=1/50. Source489dc77; old12/13epoch RAM-only attempts remain historical |
| Corrected population | RC756/UCF331/CX4222 unique training videos;8570 balanced entries/epoch | Manifest96c068ee..., fixed validation131/90/1374; evaluation unchanged RC152/UCF105/CX2403/QUVA100 |
| Official external comparison | ESCounts released-weight results inherited; RepNet updated2025 ckpt-70 completed357/357 | `evidence/resume_20261112/repnet-full357/`: exact226 weights, full identities/output hashes/metrics verified;196 predictions reused unchanged,161 newly inferred. Other literature rows are not our reproductions |
| Resource order | GPU0 completed s0 then entered s2; GPU1 completed RepNet then entered s1 | `checkpoint/GPU1_LAUNCH_STATUS.json`, `FULL_S0_COMPLETION_S2_HANDOFF.json`; no remaining24 release, no scheduler, no other-project process changes |
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

## Completed corrected full_s0 (single seed, group still running)

| Fixed evaluation population | Videos | Positive rounded NMAE (+0.1) | Positive OBO |
|---|---:|---:|---:|
| RepCountA test |152 (151 positive)|0.125608|0.728477|
| Countix test |2403|0.241028|0.692468|
| UCFRep test |105|0.201620|0.723810|
| QUVA test |100|0.195373|0.560000|

These are verbatim saved two-view scores from full_s0, selected by the frozen original-view validation rule. Original-only, shifted-only, all/zero-count and every secondary metric remain in [SAVED_METRICS.json](evidence/resume_20261112/full-s0-results/SAVED_METRICS.json), with exact predictions,50-epoch log, checkpoint hashes, source/data identities and a complete small-evidence archive. No cross-seed aggregation, significance, component-effectiveness, SOTA or group acceptance is inferred from this seed. No split, metric, model, selected epoch or subsequent seed changed after observing it.

The next concrete milestone is full_s1/full_s2 completing the same50-epoch training and four-population evaluation. Both continue in their existing finite workers. At complete three-seed group closure, interpret results once and decide the next frozen group; do not tune against intermediate test results.

The bounded [single-UCF inventory](evidence/resume_20261112/ucf-reuse/INVENTORY.md) found three completed80-epoch candidates with exact saved configuration, unchanged331/90/105 populations and inherited output hashes matching the already frozen single-UCF specifications. They remain potential reuse candidates with missing contemporaneous historical source/environment/cache bindings disclosed; no replacement training was ordered. This frozen arm also omits token/repetition, so it cannot be described as removing only the joint population while holding every full-model setting constant. Reuse and scientific interpretation await the same group-end decision; current jobs and the matrix are unchanged.

New experiments run only on the server. E: holds editing/control/evidence. Existing caches/checkpoints/negative results are preserved. A component or research route can be retired without deleting the evidence that motivated that decision. Final manuscript claims wait for the complete group; existing historical draft build/layout repair is not final acceptance.
