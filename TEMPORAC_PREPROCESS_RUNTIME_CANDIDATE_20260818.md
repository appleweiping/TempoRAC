# TempoRAC preprocessing/runtime candidate — 2026-08-18

## Implemented without changing frozen model files

- `src/pams/temporac/preprocess_resampled_v2.py`
  - physical-clock interpolation;
  - uniform 320-row resampling;
  - zero-geometry-edge-only deterministic tie break;
  - per-identity source/output hashes and repair counts;
  - typed `FeatureRecord` adapter.
- `src/pams/temporac/campaign_runtime_candidate_v1.py`
  - complete seven-block X0 teacher loss;
  - standard adapted-feature receipt binding;
  - natural teacher inference;
  - certificate and target-artifact construction;
  - differentiable natural and X0 response losses.
- `scripts/experiments/train_temporac_teacher_candidate_v1.py`
  - frozen seed/job initialization;
  - deterministic source cycles;
  - AdamW training, tune selection, no-overwrite checkpoint output.

The original `preprocess.py`, `teacher.py`, `certify.py`, `response.py`,
`objective.py`, `training.py`, and `runtime.py` were not modified.

## Validation

- Ruff on all new source/scripts/tests: PASS.
- New tests: 4 passed.
- v44 resampling entrance audit:
  - train 268/268 accepted; 1,049 missing rows interpolated; 552 degenerate
    edges repaired;
  - val 134/134 accepted; 330 missing rows interpolated; 50 degenerate edges
    repaired.
- GPU teacher runtime:
  - 1-step checkpoint SHA-256
    `e67e0254a10a2ffe5c1880c7ff8e1ee9f6efd92b7cb44bdf0ef75d9e265f68f6`;
  - 30-step checkpoint SHA-256
    `eca9e975b2ae4f8ff0e398f7cc1a17e534bfbe7d0e622b3f664d8d518aee6d4f`;
  - tune loss changed from about 0.320751 at step 1 to 0.318984 at step 30.
- Real teacher-to-target smoke on 20 train identities completed without a
  runtime exception, but produced zero certified targets.

## Newly proven hard blocker

The frozen natural certificate requires at least three geometry landmarks per
identity before teacher phase/reconstruction can affect the result. A complete
teacher-independent audit after resampling v2 found:

| split | identities | at least 3 landmarks | fraction | K1 requirement |
|---|---:|---:|---:|---:|
| train | 268 | 45 | 16.79% | 80% |
| val | 134 | 24 | 17.91% | 80% |

Therefore no teacher checkpoint, including a hypothetical fully trained one,
can make the current frozen certificate pass K1. Launching three 20,000-step
teacher jobs or the 24 response jobs would be scientifically ineligible and
waste GPU time. The next algorithmic decision must version and review the
natural landmark/certificate rule (or replace the input data with a population
that satisfies it) before campaign launch.

## Additive landmark/certificate candidate v2

The frozen-rule blocker was addressed in new files only:

- `src/pams/temporac/certify_adaptive_v2.py` preserves every frozen landmark
  result containing at least three points and otherwise applies a count-blind,
  pose-return fallback with a minimum excursion requirement;
- `src/pams/temporac/certify_target_adaptive_v2.py` copies the complete frozen
  certificate control flow and replaces only its landmark entry point;
- `src/pams/temporac/campaign_runtime_candidate_v2.py` routes natural teacher
  inference through the v2 certificate.

The original `certify.py` and candidate-v1 runtime remain unchanged. X0 ledger
equivalence passed. A complete v44 identity audit at excursion threshold 0.20
produced:

| split | identities | at least 3 landmarks | fraction | K1 requirement |
|---|---:|---:|---:|---:|
| train | 268 | 245 | 91.42% | 80% |
| val | 134 | 128 | 95.52% | 80% |

The 30-step teacher was then rerun on the same first 20 train identities using
the complete v2 certificate. Landmark abstentions fell from 16 to zero, while
the result remained 0/20 certified: 14 topology abstentions and 6 origin
abstentions. This isolates phase topology/origin consistency as the next
runtime bottleneck (the origin reason still needs subcondition diagnostics).
A separate deterministic 500-step candidate was launched
for a controlled learning-curve check; it does not overwrite prior checkpoints.

## Teacher identifiability diagnostics

The 500-step frozen-F11-style candidate reached X0 tune loss `0.0509076`, but
the same 20 natural identities remained `0/20` certified (17 topology, 3
origin). A newly added complete checkpoint evaluator then ran the model output,
not the analytic fixture, through all 56 X0 tune certificates:

- `0/56` certified;
- all 56 failed topology;
- negative observed phase-edge fraction `0.497229` versus the `<=0.01`
  checkpoint-eligibility threshold.

This demonstrates that the current F11 surrogate has a near-constant-phase
degenerate solution: `ReLU(-increment)` penalizes negative magnitude, but does
not prevent almost half of tiny increments from being negative.

An explicitly non-frozen diagnostic then added an X0-only analytic phase anchor
without changing the teacher architecture. Its artifacts are marked
`DIAGNOSTIC_NOT_FROZEN_F11` and are not paper-eligible. At 1,000 steps and
anchor weight 1, the negative-edge and alias-edge fractions both reached zero,
but X0 remained `0/56` certified: 34 topology, 21 reconstruction, 1 origin.
The mean/max traversal winding errors were `0.01747/0.07398` cycles. Increasing
the anchor weight to 10 and training 2,000 steps gave only
`0.01444/0.06149`, with the same abstention counts.

Therefore additional blind training is not justified. A reviewed protocol
revision must jointly address (1) an identifiable one-winding teacher
objective, (2) the interaction between direction-bearing memoryless inputs and
return-endpoint phase equality, and (3) whether the certificate's `1e-6`
winding tolerance is attainable for learned float32 outputs. The 27 formal
jobs remain blocked; no natural efficacy number is claimed.

## v5/v6 protocol-revision diagnostics

The next additive experiments isolated and fixed two more executable
interfaces while preserving the original model files:

1. `campaign_runtime_candidate_v5.py` masks the 66 direction values and their
   33 masks only at the phase-network input. This makes repeated-pose endpoint
   inputs invariant to adjacent speed blocks. With the unchanged teacher
   parameters, mean/max X0 traversal winding error fell to approximately
   `3.4e-8/1.2e-7`, below the `1e-6` certificate threshold.
2. Direct audit found a reconstruction target-domain mismatch. The old runtime
   reached raw masked Huber `0.01550`, while the actual certificate dense Huber
   was `0.06958` mean and `0.07112` maximum. Candidate v6 trains against the
   exact canonical `certificate_track_from_x0` continuous arrays and masks.

The selected v6 diagnostic checkpoint is:

- path:
  `/media/lenovo/data2/pams-rac/runs/temporac_candidate_v1/teacher_s20260815_v6_certtarget_w01_best_2000.pt`;
- SHA-256:
  `1382402451b176ff3a69e958daa42939c304789fd950cf7d0feec8d678cac2a6`;
- seed `20260815`, best/final step `2000`, anchor weight `0.1`;
- X0 tune: `56/56` complete certificates passed;
- negative/alias edge fractions: both zero;
- mean/max dense certificate reconstruction: `0.01121/0.01258`;
- max traversal winding error: `1.19e-7`.

This is the first candidate that executes the complete synthetic teacher gate.
It is still explicitly a protocol-revision diagnostic, not frozen F11.
Applying the matching direction-blind inference adapter and adaptive natural
certificate to the same first 20 real train identities produced `0/20`
certified (16 topology, 4 origin). Thus the remaining blocker is synthetic-to-
natural phase-domain generalization. Formal pseudo-target freezing and the 27
response/teacher campaign jobs remain blocked; no efficacy result is promoted.

## Runtime performance observation

The current candidate needs roughly 2–3 seconds per teacher step after
including complete seven-block computation and tune work. A 20,000-step job
would also exceed the frozen six-A6000-hour per-teacher ceiling. After the
certificate feasibility issue is resolved, the teacher graph still needs
profiling/optimization before formal launch.

## v7/v8 natural stable-pair diagnostics

All changes in this stage were additive candidate files. No tracked frozen
model source was modified.

Candidate v7 added count-blind positive phase closure on natural geometry
landmarks. Geometry-arc closure removed negative/alias winding failures, but
the first 20 real train identities still produced `0/20` certificates. A
label-free 1,000-step natural-decoder candidate was then trained from the v6
checkpoint while retaining an X0 loss. Its selected checkpoint is:

- path:
  `/media/lenovo/data2/pams-rac/runs/temporac_candidate_v1/teacher_s20260815_v8_naturaldecoder_1000.pt`;
- SHA-256:
  `5e5588cf2439154aff2ef40538e800e3035c6770c64318b623287c7c3b2bee50`;
- 258 natural train units were eligible for its training objective;
- final natural/X0 probe losses: `0.08537/0.01833`;
- the complete X0 retention audit remained `56/56` certified, with mean/max
  certificate reconstruction `0.01480/0.01623`, zero negative/alias edge
  fractions, and maximum winding error `1.19e-7`.

The natural certificate remained `0/20`. An independent periodic-template
oracle showed why: under the v2 hybrid landmarks, even the best shared
phase-conditioned reconstruction passed the `0.02` reconstruction threshold
for `0/20` identities (mean/worst maximum-traversal Huber
`0.06379/0.159995`). Some tracks had 30--39 landmarks, demonstrating that
local pose returns were being mistaken for complete cycles.

The additive v3/v4 landmark candidates therefore estimate a full period from
global 66-dimensional lag agreement, then select the most geometrically
consistent adjacent cycle pair. This selection is count-blind and does not
read count, period, or validation labels. With the v4 selection, the same
oracle passed `20/20` eligible train identities; mean/worst
maximum-traversal Huber was `0.00743/0.01823`.

Candidate runtime v8 and certificate v3 bind that selection to a label-free
periodic-template reconstruction. The executable certificate results were:

- first 20 real train identities: `15/20` certified (`75%`), four deliberate
  landmark abstentions and one topology/observed-phase abstention, with no
  reconstruction abstention;
- first 20 held-out val identities: `12/20` certified (`60%`), all eight
  failures deliberate landmark abstentions, with no topology or
  reconstruction abstention.

This is a substantial feasibility improvement from `0/20`, but it remains a
`DIAGNOSTIC_LABEL_FREE_PROTOCOL_REVISION`. Held-out coverage is below the 80%
preflight target, and every accepted identity currently contributes exactly
two stable events rather than the complete action sequence. It must not be
presented as a paper efficacy result or used to launch the 27 formal jobs.
The next gate is a train-only frozen threshold/selection audit that raises
held-out landmark recall while retaining zero held-out reconstruction
failures, followed by independent protocol review.

## v9/v10 coverage audit and structural limit

A complete train-only threshold sweep froze `0.025` as the v9 stable-pair
geometry threshold: it was the largest tested value with zero train oracle
reconstruction failures after structural phase prefiltering. Candidate v9 did
not modify v8 or any frozen source. Its complete executable audit was:

- train: `155/268` certified (`57.84%`), with 109 landmark abstentions and 4
  dense-phase abstentions;
- held-out val: `98/134` certified (`73.13%`), with all 36 failures being
  deliberate landmark abstentions and no reconstruction/topology failures;
- first-20 val improved from v8 `12/20` to v9 `13/20`.

The full train oracle then measured the limit of the current three-landmark,
two-traversal structure. Geometry-arc reconstruction could certify at most
`166/268` (`61.94%`) even when the best source-consistent pair was selected.
Monotone geometry DTW improved the bound to `188/268` (`70.15%`), but selecting
the best DTW pair among every structurally legal candidate did not increase
that `0.02`-threshold bound. The periodic-chain inventory itself covered
`224/268` (`83.58%`), so reconstruction/alignment, not merely landmark recall,
is the binding limit.

An additive v10 DTW runtime froze a conservative train-only row-Huber margin
of `0.018`. It reached `16/20` on the small val smoke test, but the complete
held-out audit was only `102/134` (`76.12%`): 26 landmark abstentions and 6
phase-class abstentions. Therefore the apparent 80% small-sample result did not
hold, and v10 is not promoted over v9.

The safe current diagnostic remains v9 because every held-out failure is a
conservative landmark abstention. Neither v9 nor v10 is paper-eligible: every
accepted identity still contributes exactly two local events, and the current
strict shared-template certificate cannot reach the 80% coverage target on
full data. The next protocol revision must address complete-sequence event
coverage and cycle-variation tolerance together; further threshold tuning on
the already-inspected val split is prohibited. The 27 formal jobs remain
blocked.

## Complete-sequence feasibility diagnostics

The next additive diagnostics removed the fixed two-event limitation. A v1
full-chain selector starts from the global-period landmark chain, trims origin
outliers, and uses dynamic programming to retain the maximum subsequence whose
every traversal satisfies the existing phase-sampling constraints. On all 268
train identities it produced:

- periodic-chain availability: `224/268` (`83.58%`);
- legal refined-chain availability: `212/268` (`79.10%`);
- mean retained-event fraction versus the original chain: `90.98%`.

Thus the selector retains most detected events instead of forcing exactly two.
With geometry-arc alignment and a shared mean template, the complete-chain
`q80 <= 0.04` diagnostic accepted `161/268` (`60.07%`) with 9.81 mean events.
A medoid-cycle multi-traversal DTW template improved this to `185/268`
(`69.03%`) with 9.64 mean events. The original strict maximum-Huber rule still
accepted only `49/268` (`18.28%`).

A coordinate-wise median DTW template improved the typical-cycle rule
(`median <= 0.02`) from `112/268` to `136/268`, but reduced the broader
`q80 <= 0.04` result from `185/268` to `177/268`. It therefore does not replace
the mean DTW template for coverage.

These train-only results establish a structural result: under the current
origin, topology, and reconstruction thresholds, complete-sequence coverage
cannot reach 80%. Even before reconstruction, the maximum legal refined chain
is `79.10%`; after the most effective tested multi-cycle alignment it is
`69.03%`. Relaxing thresholds or tuning on the already-inspected val split is
not authorized. A reviewed protocol revision must define a principled natural
cycle-variation model and complete-sequence certificate before any formal
pseudo-target campaign. v9 remains the conservative executable diagnostic;
the complete-chain and multi-cycle implementations are feasibility candidates
only.
