# TempoRAC live continuation after local storage maintenance

The user explicitly resumed server-only research after C-drive maintenance.
This continuation reuses the existing input caches, all completed runs and
the unchanged official TransRAC/ablation contracts. The current backend is
`/root/autodl-tmp/TempoRAC`; November4 stage/server fields are historical.
New local control/evidence stays on E. No application scheduler, new scientific
matrix, model change, GPU migration or other-project interruption was introduced.

## Actual observations

At **2026-11-12 11:55:41 America/Chicago (17:55:41 UTC)**, TransRAC PID490003
was alive on GPU0 and had committed **34/200 epochs**, **816 actual optimizer
updates**, and **zero overflow skips**. Each completed epoch covered all756
training rows. Its latest297,628,466-byte checkpoint exists; the progress receipt
binds its SHA. All five runtime source files still match the launch gate.
The exact immutable training publication remains39dcafb8d77de9e43f9b61ae0346d726eddd18b5.
The latest ten epochs took a median181.11seconds of training; the remaining
training-only estimate is8.35hours, excluding future scheduled validation and
the final357-video evaluation. Full performance results remain pending.

The new ablation `wo_shift_s1` finished all50epochs and fullRC152/Countix2403/UCF105
evaluation at17:49:31UTC. New-arm completion is now **4/21**. Only this newly
completed arm was collected:42 files matched their server byte counts and SHAs.
Saved predictions were rescored on the server without training, inference or
cache reconstruction. At17:55:33UTC, GPU1 worker435833 was running `wo_shift_s2`
(PID500464), with2/50epochs and a durable first checkpoint; `wo_token_s0` follows.
GPU0's exact ten remaining ablation arms remain saved while TransRAC runs.
Their [one-shot continuation entry](../scripts/resume_gpu0_after_transrac.py)
has a completed [fresh static engineering review](evidence/resume_20261112/live-continuation/ablation-successor/INDEPENDENT_REVIEW.md).
Its read-only check returned WAIT; execution is not started. It will reuse the
original worker after explicit root invocation and full baseline/free-GPU gates,
without a watcher, periodic job or failed-experiment retry.

At17:56:34UTC, the same RACnet acquisition worker496667 was healthy. It had
6,605,133,199 unique durable bytes including reusedG1 ranges out of128,629,792,634
(5.135%). Two of61 author objects were fully SHA-verified:RC.00 and the complete
UCF archive, totaling5,241,984,399bytes. The133 completed extracted NPZ members
(28RCtrain+105UCF) passed the worker's CRC/finite checks. This is partial input
readiness, not complete population/ROI parity or a training release.
Two of64 bounded retries had been used, consecutive failures were zero, and
the original deadline/resource limits were unchanged. Observed throughput
suggests about14–17hours of remaining transfer; this excludes later qualification
work and is not a completion guarantee. The existing worker continues missing
ranges rather than restarting downloads.

## Acceptance and concrete next work

| External family | Counting weights / inputs actually used | Current full-performance state |
|---|---|---|
| ESCounts | Acquired author RC task checkpoint | Inherited complete RC152 + UCF105 + QUVA100; no rerun |
| RepNet | Author updated2025 checkpoint70 | Inherited complete357; no rerun |
| TransRAC | Fresh official counting model, official K400 Swin-T backbone; own full200-epoch retraining | Same seed1 trajectory running; full357 follows validation selection |
| RACnet | Fixed author feature archive; no acquired counting task checkpoint | Full assets acquiring; native200-epoch retraining and full357 still pending |
| DeTRC | Official TSN/R50 backbone; no acquired counting task checkpoint | Released TSN/R50 route under preparation; neither the paper ViT result nor a completed reproduction |

These rows distinguish released-weight evaluation from own retraining. A prepared
route is not an evaluated baseline. Native population/input differences and their
previously frozen qualifications remain in each method's contract.

| Required deliverable | Current state | Next actual work |
|---|---|---|
| TransRAC official-code full performance | Running53/200 at18:52UTC; final357 pending | Continue the same durable trajectory, scheduled validation selection and complete final evaluation |
| Frozen21 new ablations +3 historical single-UCF reuse | 4/21new complete | GPU1 continues its frozen lane; restore the exact GPU0 residual after its current baseline is complete and the GPU is free |
| RACnet official baseline inputs | 7/61objects verified at18:47UTC; full acquisition running | Complete all frozen objects and QUVA production using the reviewed native bridge; training remains conditional |
| Further official baseline implementation | DeTRC native integration and focused repair review complete; full CPU JPEG production running | Finish all1244 JPEG inputs, then native TSN features/full80/full357 on a released GPU; current GPU jobs remain unchanged |
| Component contribution | Pending paired three-seed groups | Preserve primary/all/zero results and judge completed corresponding groups, without modifying current runs |
| Requested benchmark breadth/SOTA/final paper | Still unmet | Complete external performance, full ablations, qualified statistics/efficiency and evidence-bound writing |

The corrected full3 model and ESCounts/RepNet evaluations remain inherited and
are not rerun. Four existing evaluation datasets and two actually evaluated
official-release families do not satisfy the requested six-dataset/seven-framework
scope. Literature-only or merely prepared baselines are not counted as reproduced.
Negative results and historical selection/population qualifications remain visible.

DeTRC prerequisite work subsequently completed at18:14UTC:
the official97,579,098-byte TSN backbone has318/318 matched encoder states;
both native extensions compiled and imported in a private CUDA12.4 environment;
the compatibility candidate passed full-frame-index and checkpoint-state CPU
checks with zero forward calls or GPU dispatches. See its
[report](evidence/resume_20261112/live-continuation/detrc-prerequisites/REPORT.md).
Full JPEG/TSN production and full357 inference integration now continue from that
candidate. A prepared or CPU-checked baseline is not a completed performance row.

Routine execution/collection is assigned to Sol; the upcoming RACnet full-run
scientific interface was closed once by Astra before that new run starts.
The [frozen native contract](evidence/resume_20261112/live-continuation/racnet-contract-astra/RUN_CONTRACT.json)
binds RC755, seed2333, physical batch16, 200epochs, native H-norm, a single
RC-validation-native-MAE-selected checkpoint and full357 evaluation. Sol is
implementing the unchanged official model/decoder with the required input,
durability and evaluation plumbing. Full RC inputs and QUVA100 feature production
remain actual run prerequisites, not reasons to postpone source implementation.
All105 author UCF arrays have verified full-video lengths and matching embedded
counts; the fixed common ROI requires88 slices and leaves17 full intervals.
Requested identities are recorded, actual backend identities remain unattested,
and same-family scientific assurance remains provisional. Runtime observation
does not replace a scientific verdict.

New ablation evidence: [safe summary](evidence/resume_20261112/ablation-live-20261112T1755Z/SUMMARY.md).
Actual metadata and raw-evidence hash bindings:
[runtime snapshot](evidence/resume_20261112/live-continuation/RUNTIME_SNAPSHOT.json)
and [evidence bindings](evidence/resume_20261112/live-continuation/EVIDENCE_BINDINGS.json).
Publication commit and synchronized artifact manifests are recorded separately
from immutable training commits in the actual `source_sync.json` receipt.

## Native integration continuation

At18:47:14UTC, the same TransRAC process had committed51/200epochs and1224
optimizer updates; `wo_shift_s2` had25/50epochs. RACnet acquisition remained
on the same worker with7/61objects verified and17,069,921,679durable bytes
including reusedG1. No existing trajectory, cache or completed result was restarted.

The [RACnet native implementation](../scripts/external/racnet_native/README.md)
now implements the frozen200epoch/full357 route. One concentrated review found
an extraction-refresh identity issue and missing aggregate timing; both received
bounded metadata-only repairs and a same-session closure. Native model, losses,
decoder, validation selection and scientific contract remain unchanged.
The [review closure](evidence/resume_20261112/live-continuation/racnet-native-implementation/INDEPENDENT_REVIEW_FOLLOWUP.md)
is provisional engineering assurance; full author inputs and QUVA production
still precede actual training. Earlier input counts in that packet are preparation
snapshots; the live acquisition state has continued beyond them.

The [DeTRC production entry](evidence/resume_20261112/live-continuation/detrc-native-integration/PRELAUNCH.md)
implements full1244-video native JPEG preparation, resumable official TSN features,
full80training and one selected checkpoint's131diagnostic+357final predictions.
Its concentrated review found an H5-without-sidecar recovery defect. The bounded
repair reuses committed pickles, preserves original numerical assembly and uses
the same completion criterion for capacity estimates. The original source,
checks and first reviews remain archived; no passed full check was repeated.

CPU JPEG production is independent of occupied GPUs and is released separately
after this fixed engineering closure and source publication. It uses one worker,
two allowed CPU cores, nice10/ionice2-7, a64GiB derivative cap and128GiB free reserve
plus the remaining RACnet author bytes. Full GPU feature production, training,
evaluation and baseline performance are still pending actual execution. These
prepared routes do not increase the count of completed external performance rows.

At18:53:02UTC, after GitHub/server/E source publicationc7544c2, the full CPU
production was actually launched asPID510017. At18:53:56UTC it had durably
completed19/1244videos. The first complete300-frame video had matching source,
contract, producer, membership and file-size evidence; its already recorded
frame hashes were reused without a second decode or full frame rehash. Process
affinity was CPUs0/1, nice10, CUDA_VISIBLE_DEVICES empty. Both existing GPU
processes490003/500464 remained active. See the
[first durable receipt](evidence/resume_20261112/live-continuation/detrc-cpu-production20261112T1853Z/FIRST_DURABLE_SAFE.json).
The original video and feature caches, all completed experiments and all failure
records remain retained. Full JPEG production is in progress, not yet complete.
