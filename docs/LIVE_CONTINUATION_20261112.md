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
| TransRAC official-code full performance | Running121/200 at23:06UTC; final357 pending | Continue the same durable trajectory, scheduled validation selection and complete final evaluation |
| Frozen21 new ablations +3 historical single-UCF reuse | 6/21new complete; wo_shift paired3 closed | Preserve bound6arm receipt; GPU1 current arm completes before the requested whole-arm DeTRC handoff; restore each exact saved residual after its baseline closes |
| RACnet official baseline inputs | 57/61objects verified at23:06UTC;120719562127/128629792634 durable bytes | Finish the same acquisition, refresh only metadata using existing886 targets/152 native GT, then completeQUVA/full200/full357 on a released GPU |
| Further official baseline implementation | DeTRC full1244 CPU JPEG production and deterministic closure complete | Reuse all954550 native JPEG frames; TSN features/full80/full357 remain pending actual released GPU and launch/outcome receipts |
| Component contribution | wo_shift paired3 statistical closure; overall requirement still unmet | Preserve Countix seed1 negative and Countix/UCF Bonferroni18 intervals spanning zero; no independent scientific acceptance or inference from incomplete groups |
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

## Incremental evidence at22:48–22:50UTC

The newly collected `wo_shift_s2` and `wo_token_s0` completed50epochs and the
fixed RC152/Countix2403/UCF105 evaluations at19:42:05UTC and21:33:05UTC.
There are now **6/21 new completed ablations**. The [safe aggregate report](evidence/resume_20261112/continuation-2244/ablation/AGGREGATE_SAFE_REPORT.md)
and [completion summary](evidence/resume_20261112/continuation-2244/ablation/COMPLETION_PUBLIC_SUMMARY.json)
bind the immutable private collection. All71 transferred files matched bytes/SHA.
The four earlier collected arms and corrected full3 summaries were reused.
No training, inference, cache build or previous full-group statistics were repeated.

The closed wo_shift seeds0/1/2 have full-versus-removal primary NMAE means
0.141996913 versus0.169563887 on RepCount-A,0.243386496 versus0.252201503 on
Countix, and0.194626165 versus0.215744358 on UCFRep. The paired removal-minus-full
means are0.027566975,0.008815006 and0.021118193. Existing source-cluster bootstrap
units,10000draws/seed20261111 and the original18-comparison family were reused
only for this newly closed contrast. Bonferroni18 intervals are
[0.001033961,0.060436591],[-0.000692408,0.018213262] and
[-0.009656390,0.052671752], respectively, conditional on the fitted models.
Countix seed1 favors removal by0.002832460; its negative result is retained.
Countix/UCF adjusted intervals span zero. These intervals have finite Monte Carlo
tail precision and establish no equivalence or broad component-effectiveness verdict.
[All aggregate metrics, matched-seed deltas and saved-view factorial diagnostics](evidence/resume_20261112/continuation-2244/ablation/WO_SHIFT_PAIRED3_AGGREGATE_PUBLIC.json)
remain available; per-video predictions and labels remain private.

At22:48:09UTC, GPU1 was still the original worker435833/trainer531570,
`wo_token_s1` with36/50 committed epochs. The later [manual handoff request](evidence/resume_20261112/continuation-2244/detrc/GPU1_HANDOFF_REQUEST.json)
requires that entire current arm, all frozen evaluations and its durable completion
receipt finish before the original worker records the exact four-arm residual and
exits. That request is evidence of an intended handoff; it is not a DeTRC GPU launch.
The GPU0 ten-arm residual remains unchanged. The bound6arm receipt is retained;
this update does not collect or imply completion of a seventh arm.

At22:50:19UTC, [DeTRC CPU closure](evidence/resume_20261112/continuation-2244/detrc/CPU_COMPLETE_SAFE.json)
verified all1244 videos and954550 JPEG frames:756 RC train,131 RC validation,
152 RC test,105 UCF and100 QUVA. All1244 original identity receipts matched the
frozen source/producer/contract; complete membership, frame indices and recorded
byte sizes passed. The original frame SHA evidence was reused with zero decoding,
JPEG content rehash, model loads or GPU dispatches. The producer had exited;
its measured production wall time was3090.9897645s. Full TSN extraction,80epoch
training and357final+131diagnostic evaluation remain pending the actual root-owned
GPU release. [CPU closure and remaining stages](evidence/resume_20261112/continuation-2244/detrc/SAFE_SUMMARY.md)
do not constitute a measured baseline performance row.

The [incremental acceptance evidence](evidence/resume_20261112/continuation-2244/ablation/ACCEPTANCE.md)
records deterministic collection as passed and scientific effectiveness/breadth as
partial or pending. Requested executor Sol; actual backend identity remains
unattested; same-family assurance remains provisional. No current status,
source-sync, worker, training source or contract was modified by this collection.

## Current input reuse and runtime at23:06UTC

TransRAC remains the same GPU0 process,121/200 committed epochs and2904 actual
optimizer updates, with zero overflow skips. GPU1's current wo_token_s1 has44/50
epochs; its full evaluations and durable completion precede DeTRC release.
RACnet's same attempt3 worker has57/61 verified objects and120719562127 durable
bytes,10 retries and zero consecutive failures. Neither external baseline has a
new completed performance row at this observation.

The new [receipt-only input refresh](evidence/resume_20261112/continuation-2244/racnet/REFRESH_INPUTS_PUBLIC_SUMMARY.json)
avoids rerunning initial RACnet preparation. It inherits886 cached targets and152
RC-test native GT records, preserves targets/scoring/raw-QUVA manifest bytes,
and changes only author/QUVA file qualification metadata. One server metadata
fixture passed with zero science/GT/decoder/array/payload-hash calls; the
[fresh bounded source review](evidence/resume_20261112/continuation-2244/racnet/INDEPENDENT_REVIEW.md)
found no actionable blockers. Assurance is same-family provisional. The actual
refresh has not run: it requires all61 author objects formally complete. No
current training configuration, cache or trajectory changes with this repair.
