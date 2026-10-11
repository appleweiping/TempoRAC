# TempoRAC: Identity-Indexed Local Tempo Routing for Multi-Person Repetitive Action Counting

This repository was formerly named `pams-rac-reproduction`. The Python
distribution and import package keep their original names
(`pams-rac-reproduction`, `pams`), as does the console script `pams`.

> **Project status.** The newest dated records in this repository are
> ledger entries from 2026-08-19 and raw diagnostic traces from 2026-08-22.
>
> - **TempoRAC:** the Wave 0/1 contract modules are implemented, but the
>   native campaign launcher and pseudo-target runtime are not, and no
>   efficacy result exists. All 27 frozen training jobs are
>   `BLOCKED_PRECHECK`, every planned comparison and ablation is blocked,
>   and the recorded decision is `HOLD_INDEPENDENT_REVIEW`.
> - **PAMS reproduction:** the code base the project began from is a
>   `partial_reproduction` with negative, development-only results. The
>   sealed 105-video UCFRep test split is unscored.
> - **Side probes:** undated, count-supervised probes
>   (`results/autoresearch-v19` to `-v21`) were added after the v18 code and
>   may therefore be the most recent work in the tree. They are not TempoRAC
>   results and do not change the status above. (`density-seed20260925`
>   names a training seed, not a date.)

## Overview

This repository studies multi-person repetitive action counting (MRAC): one
repetition count per person, where people move asynchronously and each
person's tempo may change within a track. TempoRAC takes externally supplied,
identity-indexed pose tracks and processes each identity separately:

1. A shared causal encoder feeds three structurally identical slow, medium
   and fast response experts.
2. A detached, irregular-clock non-uniform discrete Fourier transform (NUDFT)
   cue measures relative window-local tempo and sets soft per-window gates
   over the experts' probabilities.
3. Overlapping windows are merged by exact, strictly positive normalized
   overlap-add (NOLA).
4. One fixed-threshold connected-component decode returns the count.

The frozen contract `temporac.execution.v4` fixes the method. Besides its
implementation, the repository holds a synthetic training population (X0), a
phase teacher with certificate-based pseudo-targets, additive candidate
runtimes, diagnostics and the ARIS evidence ledger. It also keeps the
independent, auditable partial reproduction of PAMS (single-person counting
on UCFRep) from which the project grew. The central TempoRAC hypothesis has
not been tested on any data. No formal response checkpoint exists, and both
the synthetic specialization test and the natural-data comparisons are
blocked.

![Conceptual task schematic: pooled single-stream counting versus identity-indexed, window-local responses with one count per identity](paper/figures/task_comparison.png)

*This is a schematic of the task setting, not a measurement or an
architecture diagram. Panel (a) pools all identities into one count. Panel
(b), the setting studied here, keeps one window-local response per identity
and outputs a person-wise count vector. The paper package lists the figure as
retired from its active draft
([`figure_manifest.json`](paper/figures/figure_manifest.json)).*

## Project status

| Line of work | Recorded status | Evidence |
|---|---|---|
| TempoRAC contract modules | Wave 0/1 contract work is implemented, and the Wave 1 post-fix review is recorded as passed. The effective contract in code is the 665-byte Wave 1 index (SHA-256 `c203f4aa…3d5c`). | [`TEMPORAC_HANDOFF_20260817.md`](TEMPORAC_HANDOFF_20260817.md); [`contract.py`](src/pams/temporac/contract.py) |
| Frozen 27-job campaign | All jobs are `BLOCKED_PRECHECK`. No formal training ran, and no MAE/OBO is assigned to any job. | [`ALL_EXPERIMENT_RESULTS_20260818.md`](results/temporac/ALL_EXPERIMENT_RESULTS_20260818.md) |
| Natural-data entrance | Fails. The frozen preprocessor accepts 172/268 train and 91/134 validation identities. Only 45/268 train and 24/134 validation identities have the at least three landmarks the frozen certificate needs, so it cannot reach the 80 % K1 coverage requirement. | [`TEMPORAC_EXPERIMENT_STATUS_20260818.md`](TEMPORAC_EXPERIMENT_STATUS_20260818.md); [`TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md`](TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md) |
| Additive candidates v1–v18 | Diagnostics only; none is accepted. The prose ledger ends at v15. v15 was formulated after inspecting the validation population and has only an author preflight (`AUTHOR_PREFLIGHT_PASS / INDEPENDENT_REVIEW_REQUIRED / NO_LAUNCH_AUTHORITY`). v16–v18 exist only as code, raw traces and diagnostic JSON (`results/temporac-v16/`, `results/temporac-v18-diagnostic/`), with no ledger entry or review. | [`TEMPORAC_V15_PROTOCOL_AUTHOR_PREFLIGHT_20260819.md`](refine-logs/temporac/TEMPORAC_V15_PROTOCOL_AUTHOR_PREFLIGHT_20260819.md) |
| Contract amendment A005 | Unresolved. The fixed aliases still hold Round 8, which was reviewed `REVISE`. The Round 9 pair `20260819_195424` has author prechecks only (`AWAITING_INDEPENDENT_REVIEW`). Round 10 preparation scripts exist, but no Round 10 artifact does. | [`TEMPORAC_HANDOFF_20260817.md`](TEMPORAC_HANDOFF_20260817.md); [review request](refine-logs/temporac/TEMPORAC_A005_ROUND9_INDEPENDENT_REVIEW_REQUEST_20260819_195424.json) |
| Authority | No P0–P3, S0, G or K gate has passed, and no server experiment, result promotion or launch is authorized. | [`TEMPORAC_HANDOFF_20260817.md`](TEMPORAC_HANDOFF_20260817.md) |
| PAMS reproduction | `partial_reproduction`, with no verified benchmark claim. Three-seed dev84 results are negative, and the test split has no predictions. | [`results/README.md`](results/README.md) |
| WARP-PHASE pilot | Terminal Gate 2 (K5) failure, after which the work pivoted. Recorded as an engineering failure, not an efficacy result. | [`CLAIMS_FROM_RESULTS_WARP_PHASE.md`](CLAIMS_FROM_RESULTS_WARP_PHASE.md) |

The absent native launcher and pseudo-target runtime are listed as blockers
in the 2026-08-18 audit
([`TEMPORAC_EXPERIMENT_STATUS_20260818.md`](TEMPORAC_EXPERIMENT_STATUS_20260818.md)).
All ARIS semantic reviews were run within one model family and are recorded
as same-family and provisional
([`TEMPORAC_HANDOFF_20260817.md`](TEMPORAC_HANDOFF_20260817.md);
[`ARIS_LOCK.md`](ARIS_LOCK.md)). The handoff calls this snapshot
fail-closed; it claims no completed gates, server runs or paper results.

Gate labels (P0–P3 and P00–P06, S0, G, K1–K7) and contract components
(F11, F14) are defined in
[`FINAL_PROPOSAL.md`](refine-logs/temporac/FINAL_PROPOSAL.md) and
[`EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md) (in Chinese). Wave 0/1
denotes the contract-implementation waves reviewed under
`refine-logs/temporac/`, and A001–A006 are the contract amendments stored
there.

## Method

### Claim under test and invariants

The input is a set of supplied, person-indexed COCO-17 pose tracks with
joint confidences. The output is one count per identity.

The only normative specification is
[`refine-logs/temporac/FINAL_PROPOSAL.md`](refine-logs/temporac/FINAL_PROPOSAL.md),
`temporac.execution.v4`; "TempoRAC" itself is a working name. That file is
byte-identical to [`refine-logs/FINAL_PROPOSAL.md`](refine-logs/FINAL_PROPOSAL.md),
and its SHA-256
`3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` is
pinned as `PROPOSAL_SHA256` in `contract.py`. Its single dominant claim is
an untested hypothesis:

> On frozen supplied identity tracks, identity-indexed window-local
> relative-tempo routing among shared slow/medium/fast response experts
> improves one-decode counting under deterministic within-track pace drift
> relative to the strongest frozen matched nonlocal comparator, while
> retaining performance on the paired unwarped/clean tracks.

The seven contract invariants are:

1. All state, windows, NOLA accumulators and normalization are private to one identity.
2. The branches share operations and parameter counts and differ only in weights and frozen dilation.
3. Routing is a detached, cue-only soft function of relative window-local tempo.
4. Sigmoid probabilities are fused, never logits or counts.
5. Overlap is reconstructed through the exact, strictly positive NOLA denominator.
6. One threshold-0.5 connected-component decode runs per identity.
7. No human count, period, density, boundary, box or evaluator field enters any objective, checkpoint or prediction.

The contract excludes `L_cont`, PAMS-TCC initialization, ACF fusion and a
learned router. Its non-goals disclaim introducing MRAC, person-wise output,
pose counting, tracking, temporal windows, phase learning, mixture-of-experts,
ACF/FFT, PAMS-TCC, NOLA or peak detection, and disclaim being first,
annotation-free, label-free, end-to-end, constant-time or state of the art.
The paper-package ledger records the claim that the intended counting
objective uses no person-wise count or period labels as
`BLOCKED (SYNC-REQUIRED)`: its supervision-firewall evidence is still
blocked, and any such wording must also disclose the supervision in the
pose/tracking front end
([`CLAIMS_EVIDENCE_MATRIX.md`](CLAIMS_EVIDENCE_MATRIX.md), row C10).

### Graph, training signals and campaign

| Element | Implementation |
|---|---|
| Preprocessing | COCO-17 `[T,17,3]`, joint-confidence threshold 0.20; duplicate clocks collapsed; at least 17 valid samples. Produces 66-D geometry (34 joint coordinates plus the 32 coordinates of 16 bone vectors), 215 teacher channels and 269 response channels. |
| Cue and router | 128-sample windows, stride 32. A detrended NUDFT over 48 log-spaced periods (5–128) gives `u_w` and a confidence `gamma_w`, which is set to 0 below 0.15. Gates: `a_{w,k} = gamma_w * softmax_k(-(u_w - u_ref - mu_k)^2 / (2 sigma^2)) + (1 - gamma_w)/3`, with `mu = (-log 1.5, 0, log 1.5)` and `sigma = log(1.5)/2`. All detached. |
| Experts | Encoder `Linear(269,64)`, then two causal residual depthwise-separable blocks (kernel 5, dilations 1, 2). Slow/medium/fast branches each apply two causal residual blocks at dilation 4/2/1, then `LayerNorm-Linear(64,1)-Sigmoid`. |
| Fusion, NOLA, decoding | `r = sum_k a_{w,k} * sigmoid(z_k)`. Taper `h_n = 1e-3 + (1 - 1e-3) sin^2(pi (n + 1/2) / 127)`. `R_e = N_e / D_e`, with compensated sums and `D_e >= 1e-3` and no clamp. One float32 quantization; each maximal run of edges with `R >= 0.5` is one event. Abstentions serialize as `count = -1`. |
| Phase teacher (F11) | Static `Linear(298,128)-GELU-Linear(128,32)`; phase `Linear(247,256)-GELU-Linear(256,128)-GELU-Linear(128,2)` with unit-L2 output; reconstruction decoder. |
| Certification | Integer-landmark, winding-checked certificates convert teacher phase into pulse pseudo-targets, or abstain with a reason code. |
| X0 population | 40 deterministic synthetic source orbits (24 train / 8 tune / 8 held-out). Each has one stationary block and six slow/medium/fast drift orders. |
| Response objective (F14) | Balanced BCE plus positive-margin (0.75) and negative-valley (0.25) terms, each weighted 0.5. Step loss `0.5 * L_X0 + 0.5 * L_natural`. |
| Campaign | Seeds 20260815, 20260816 and 20260817. 3 teacher jobs (20,000 steps, base learning rate 3e-4); 24 response jobs (10,000 steps, base learning rate 1e-3): 3 canonical, 3 capacity-control and 18 shortcut (six adapters x three seeds). |
| Optimizer and budget | AdamW with betas (0.9, 0.999), eps 1e-8 and weight decay 1e-4; 5 % linear warm-up, then cosine decay to 0.1x; checkpoint every 500 steps. 93 of 100 RTX A6000 hours allocated; at most 2 concurrent jobs. |
| Planned metrics | Per-identity relative error and OBO, averaged within each video, then over videos, then over seeds. Arms: `local`, `global`, `uniform`, `capacity-control`, each on natural-clean and natural-drift input. 10,000-draw paired video-cluster bootstrap; fixed K7 decision rule. |

### Component map

| Component | Source (`src/pams/temporac/`) |
|---|---|
| Contract constants, job inventory, seeds, budget | [`contract.py`](src/pams/temporac/contract.py) |
| Preprocessing and quadrature | [`preprocess.py`](src/pams/temporac/preprocess.py), [`quadrature.py`](src/pams/temporac/quadrature.py) |
| NUDFT cue; routing modes `local`, `global`, `uniform`, `blocked`, `shuffled` | [`cue.py`](src/pams/temporac/cue.py) |
| Encoder, experts, capacity control, six shortcut adapters | [`response.py`](src/pams/temporac/response.py) |
| Fusion and exact NOLA; decoder; per-identity predictions | [`nola.py`](src/pams/temporac/nola.py), [`decode.py`](src/pams/temporac/decode.py), [`prediction.py`](src/pams/temporac/prediction.py) |
| Teacher (F11), certification, X0, response objective (F14) | [`teacher.py`](src/pams/temporac/teacher.py), [`certify.py`](src/pams/temporac/certify.py), [`x0.py`](src/pams/temporac/x0.py), [`objective.py`](src/pams/temporac/objective.py) |
| Runtime record, schedule, fail-closed training audit | [`runtime.py`](src/pams/temporac/runtime.py), [`training.py`](src/pams/temporac/training.py) |
| Feature-only I/O, feature/vault packing, hashing, receipts | [`feature_io.py`](src/pams/temporac/feature_io.py), [`trusted_packer.py`](src/pams/temporac/trusted_packer.py), [`hashio.py`](src/pams/temporac/hashio.py), [`receipts.py`](src/pams/temporac/receipts.py) |
| Metrics, paired bootstrap, K7; evaluator skeleton; gate state | [`metrics.py`](src/pams/temporac/metrics.py), [`evaluator.py`](src/pams/temporac/evaluator.py), [`gates.py`](src/pams/temporac/gates.py) |
| Same-checkpoint inference ablations | [`ablation_v1.py`](src/pams/temporac/ablation_v1.py) |

### Additive candidates (diagnostic, not accepted)

The frozen modules `preprocess.py`, `teacher.py`, `certify.py`,
`response.py`, `objective.py`, `training.py` and `runtime.py` remain
normative. Later revisions were added beside them
([`TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md`](TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md)):

| Modules | Role |
|---|---|
| `preprocess_resampled_v2.py`, `preprocess_interpolated_v1.py` | Physical-clock interpolation with 320-row resampling; non-contract gap filling |
| `campaign_runtime_candidate_v1.py` … `v18.py` | End-to-end glue and successive teacher and pseudo-target revisions |
| `certify_adaptive_v2.py` … `v5.py`, `full_chain_landmarks_candidate_v1.py` … `v4.py` | Landmark selectors |
| `certify_target_adaptive_v2.py` … `v10.py` | Copies of the frozen certificate that replace only the landmark entry point |
| `dtw_*`, `multi_cycle_dtw_template_*`, `periodic_template_*`, `phase_closure_*` | Dynamic-time-warping (DTW) pair selection and phase, multi-cycle and periodic templates, count-blind phase closure |

The newest candidate, `campaign_runtime_candidate_v18.py`, keeps every v17
certification. Only after a v17 abstention does it fall back to
complete-chain landmarks with a return-RMS threshold of 0.085, which its code
labels train-selected. A validation sweep of 0.075/0.085 is also recorded
([`val-threshold-0075-0085.json`](results/temporac-v18-diagnostic/val-threshold-0075-0085.json)),
so v18's validation coverage is not an untouched estimate. None of these
candidates is part of the contract.

## Lineage: PAMS reproduction and WARP-PHASE

**PAMS reproduction.** The repository began as an independent, auditable
reproduction of *Count What Repeats: Period-Adaptive Multi-Scale Consistency
for Self-Supervised Repetitive Action Counting* (CVPR Findings 2026; link in
[`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md)).

The paper publishes no implementation and gives no training loss for its
Period Head. Its comparison table also mixes UCFRep (526 videos) and
UCFRep-pose (110 videos) results ([`docs/PAPER_AUDIT.md`](docs/PAPER_AUDIT.md)).
Each inferred closure is registered in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

The implementation, frozen in [`configs/pams.yaml`](configs/pams.yaml),
combines a four-layer Transformer encoder (width 512, 16 heads) over 33
MediaPipe joints with a stop-gradient FFT period estimate, PAMS-TCC
contrastive training at scales 0.5/1.0/1.5 and a Fast/Medium/Slow
multi-expert peak-counting consensus. **PAMS-Literal** keeps the seeded
random Period Head untrained, as a diagnostic. **PAMS-SSHead** freezes the
encoder and trains the head for 30 epochs with the inferred, collapse-guarded
loss `L_cycle + L_spectral + 0.1 L_variance + 0.01 L_smooth`; it is an
inferred repair, never the authors' method.

**WARP-PHASE.** This intermediate candidate used a signed local
warp-integral objective. It failed its preregistered Gate 2 (K5 selector
stability) terminally and was pivoted. The code remains in
`src/pams/warp_phase` and `pams warp-phase` as research history
([`CLAIMS_FROM_RESULTS_WARP_PHASE.md`](CLAIMS_FROM_RESULTS_WARP_PHASE.md)).

## Repository structure

```text
src/pams/                 installable package (console script `pams`)
  temporac/               TempoRAC contract modules and additive candidates
  warp_phase/             WARP-PHASE pilot (terminated; research history)
  baselines/              baseline registry, official-checkpoint runners, clean-room scaffolds
  *.py                    PAMS reproduction (model, losses, period, consensus, training, CLI, ...)
configs/                  PAMS frozen config, protocols, ablation ledger, experiments, smoke, baselines
scripts/experiments/      TempoRAC teacher/decoder candidates, manifest generators, side probes
scripts/diagnostics/      TempoRAC entrance, landmark, certificate, threshold and ablation diagnostics
scripts/server/           container build/launch, pose-cache audits, PAMS server gate runners
tests/                    unit, property and integration tests (tests/temporac for TempoRAC)
results/                  result records (PAMS, TempoRAC diagnostics, side probes); some, e.g. under
                          results/dev-negative/ and results/autoresearch-v21/, retain absolute server paths
refine-logs/              ARIS proposals, plans, trackers, reviews; refine-logs/temporac holds the
                          TempoRAC contract, amendments, reviews and raw traces
idea-stage/, .aris/       ARIS research contract, run records and review traces
paper/                    ARIS-generated pre-results draft package
docs/                     PAMS specification, paper audit, assumptions, baselines, server runbook
docker/                   PAMS server image and environment-only baseline sidecar images
data/                     UCFRep split lists and committed test fixtures (see Data)
TEMPORAC_*.md             top-level TempoRAC handoff and status documents (2026-08-17/18)
```

Many top-level ARIS documents also exist as timestamped snapshots (for
example `NARRATIVE_REPORT_20260815_013000.md`); the un-suffixed file is the
current one. [`.gitattributes`](.gitattributes) disables Git's
trailing-whitespace checks for these snapshots so that they stay
byte-identical to their recorded hashes.

[`paper/`](paper/README.md) is an ARIS-generated, evidence-gated draft
package; its README calls it `provisional/data-pending`, with every
controlled result rendered `PENDING` and no proposed-method number. Its
framework diagrams (`framework.png`, `icassp_framework.png`), which the
package also lists as retired from its active draft, are
synchronization-pending proposal drawings showing elements the v4 contract
excludes (PAMS-TCC, routing and track/continuity losses and, in
`icassp_framework.png`, an ACF + FFT cue).

## Requirements and installation

The package supports Python 3.10–3.12. It depends on NumPy, SciPy, PyTorch,
scikit-learn, Pydantic, PyYAML and Typer ([`pyproject.toml`](pyproject.toml));
a `uv.lock` is also tracked, and the 2026-08-15 baseline test receipt was
produced with `uv run --extra dev pytest -q`
([`BASELINE_TEST_RECEIPT.json`](refine-logs/temporac/BASELINE_TEST_RECEIPT.json)).
The `pose` extra (NumPy < 2), the WARP-PHASE training lock (NumPy 1.26.4) and
the formal TempoRAC runtime (NumPy 2.1.0) require separate environments.

```bash
git clone https://github.com/appleweiping/TempoRAC.git
cd TempoRAC
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"                   # extras: dev, pose, countllm
pams --help                                         # equivalently: python -m pams --help
```

| Extra or pinned environment | Specification |
|---|---|
| `pose` extra | `mediapipe==0.10.14`, `opencv-contrib-python==4.10.0.84`, `protobuf==4.25.3`, `numpy<2`; required for `pams pose extract` |
| `countllm` / `dev` extras | CountLLM-Lite baseline recipe / pytest, ruff, mypy |
| Formal TempoRAC campaign ([`runtime.py`](src/pams/temporac/runtime.py)) | Linux x86_64, CPython 3.12.4, NumPy 2.1.0, SciPy 1.14.1, PyTorch 2.4.1+cu124, CUDA 12.4, cuDNN 9.1.0.70, driver 550.54.14, RTX A6000; deterministic, no TF32 or autocast. No container definition in the repository matches it. |
| WARP-PHASE training lock ([`gates.py`](src/pams/warp_phase/gates.py)) | CPython 3.12.13, NumPy 1.26.4 and the exact SHA-256 of `pyproject.toml` and `uv.lock`. Editing either file, including for the repository rename, invalidates the frozen lock. |
| PAMS server image ([`Dockerfile`](docker/server/Dockerfile)) | PyTorch 2.5.1/CUDA 12.4, Python 3.11, MediaPipe 0.10.14; run on one GPU with [`run_gpu1.sh`](scripts/server/run_gpu1.sh) per [`docs/SERVER_RUNBOOK.md`](docs/SERVER_RUNBOOK.md) |

Credentials (passwords, private keys) are not stored in the repository; see
[`SECURITY.md`](SECURITY.md) for the reporting policy and
[`.env.example`](.env.example) for the path-only environment template.

## Data and external assets

| Tracked path (no videos, pose caches or weights) | Contents |
|---|---|
| [`data/splits/`](data/splits/README.md) | Canonical UCFRep ID lists (421/105 and 337/84) |
| `data/temporac_p00_v4/` (about 31 MB) | Synthetic X0 and operator-manifest candidate fixtures with receipts, environment locks and replay witnesses |
| `data/warp_phase_pilot_v1/audit/` | WARP-PHASE gate 0–3 receipts and selector unit fixtures (JSON, Markdown, small NumPy files) |

The TempoRAC fixtures are read by `tests/temporac/test_operator_manifest.py`,
`test_x0_manifest.py` and `test_gates_fixtures.py`. The WARP-PHASE gate
receipts and selector unit fixtures are the artefacts bound by
[`gates.py`](src/pams/warp_phase/gates.py) and cited in
[`CLAIMS_FROM_RESULTS_WARP_PHASE.md`](CLAIMS_FROM_RESULTS_WARP_PHASE.md).
`tests/test_warp_phase_selector.py::test_fixture_id10_failure_preserves_computed_diagnostics`
also needs the historical pack
`data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/pack/`, which is
not distributed. [`.gitignore`](.gitignore) ignores everything under `data/`
except `data/splits/`; the fixture directories above are tracked explicitly,
so new files there must be added with `git add -f`.

**MultiRep v44 pose cache: TempoRAC natural data, not distributed.** The
only permitted natural inputs are `train.pkl` (SHA-256
`c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb`) and
`val.pkl` (SHA-256 `4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251`),
both under `data/counting_multirep_skeleton_pose_expanded_v44_len320/`. The
trusted loader in [`packing.py`](src/pams/warp_phase/packing.py) checks these
digests before deserializing.

[`PILOT_DATA_SCHEMA_AUDIT.md`](PILOT_DATA_SCHEMA_AUDIT.md) records the cache
as follows:

| Property | Recorded value |
|---|---|
| Protocol name and labels | *GT-bbox-assisted AlphaPose supplied-track pilot*; `partial-cache` / `GT-bbox-assisted` / `supplied-track` |
| Coverage | 110 train / 51 validation videos (268 / 134 identities). Only 110 of 794 train and 51 of 183 validation videos have pose conversions, a plausible selection bias. |
| Tracks and labels | Person slots come from IoU association of ground-truth boxes with AlphaPose detections. The pickles store count and density targets next to the inputs. |
| Permitted use | A controlled pilot, not eligible for main results. It cannot support predicted-track, end-to-end, HOTA, IDF1 or IDSW evaluation, and test data are forbidden inputs. |

The repository does not record the licence terms of the underlying MultiRep
data; obtain the data under its original terms.

**UCF101 videos and UCFRep annotations: PAMS data, not distributed.** The
official annotation archive is accepted only with SHA-256
`08b2b8a88c2728e9aa6de6c62dc6b02f1941dfa2c3adb28dab5852c12695ae3c`;
`pams data prepare-ucfrep` downloads it to
`data/annotations/ucf526_annotations.zip` when absent
([`src/pams/cli.py`](src/pams/cli.py); see also [`DATA.md`](DATA.md)).
Preparation accepts `UCF-101/<Action>/<video>.avi`, `<Action>/<video>.avi`
or `flat/<video>.avi`, hashes all 526 videos in strict mode, rejects
cross-layout duplicates and maps UCFRep `HandStandPushups` to UCF101
`HandstandPushups` as its only alias; it performs no recursive or
case-insensitive fallback ([`src/pams/ucfrep.py`](src/pams/ucfrep.py)). Pose
caches are keyed by a pose fingerprint that excludes seeds and optimizer
settings, so seeds 42/2026/3407 share one cache ([`DATA.md`](DATA.md)).
UCFRep-pose (89/21) sealed scoring is disabled until its official ID/count
digest is frozen ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).

**Checkpoints and serialized objects.** No checkpoint is tracked in git
(`*.pt` is git-ignored), and no TempoRAC teacher or response checkpoint is
distributed. This includes the diagnostic teacher
`teacher_s20260815_v8_naturaldecoder_1000.pt` (SHA-256 `5e5588cf…2bee50`), so
checkpoint-based TempoRAC diagnostics cannot be replayed from a checkout. The
only recorded weight release is the development-only PAMS-SSHead v8 seed-2026
encoder and head, listed as published under the prerelease
`v0.4-v8-partial`, with hashes in
[`summary.json`](results/dev-negative/pams_sshead_longest_track_v8_seed2026_07192de/summary.json).
The file
`results/autoresearch-v21/seed20260725-unseen-person-count/prototype_calibrator.joblib`
is pickle-based; loading it can execute arbitrary code.

## Usage

### Lint and tests

[`ci.yml`](.github/workflows/ci.yml) runs the first two commands below on
Python 3.10–3.12 for every push and pull request:

```bash
ruff check src tests scripts
pytest --cov=pams --cov-report=term-missing
pytest tests/temporac            # TempoRAC subset
```

No passing full-suite run is recorded for the current tree, and the CI job
should be expected to fail on generic runners. The lint step alone reports
218 `ruff` findings on the current tree, most of them in the v19–v21 probe
scripts. Some WARP-PHASE assertions
replay frozen environment locks: the training lock requires CPython 3.12.13
and NumPy 1.26.4 ([`gates.py`](src/pams/warp_phase/gates.py)), which
`tests/test_warp_phase_gates.py::test_training_environment_lock_is_canonical_exclusive_and_live`
exercises against the running interpreter. In addition,
`tests/test_warp_phase_selector.py::test_fixture_id10_failure_preserves_computed_diagnostics`
reads the untracked pack
`data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/pack/` and fails
on a fresh clone.

| Date | Scope | Outcome | Evidence |
|---|---|---|---|
| 2026-08-15 | Full suite, before `pams.temporac` existed | 1022 passed, 6 skipped | [`BASELINE_TEST_RECEIPT.json`](refine-logs/temporac/BASELINE_TEST_RECEIPT.json) |
| Wave 1 freeze (2026-08-17 handoff) | TempoRAC tests | 134 passed, 1 skipped | [`TEMPORAC_HANDOFF_20260817.md`](TEMPORAC_HANDOFF_20260817.md) |
| 2026-08-18 | Container `pams-rac:07192de` (scope not recorded) | 133 passed, 2 failed | [`TEMPORAC_EXPERIMENT_STATUS_20260818.md`](TEMPORAC_EXPERIMENT_STATUS_20260818.md) |
| 2026-08-22 | Raw trace named `full-pytest.log`; 165 tests run (command not recorded) | 163 passed, 2 failed | [`full-pytest.log`](refine-logs/temporac/independent-traces-v18-20260822/full-pytest.log) |

Both 2026-08-22 failures were in `tests/temporac/test_operator_manifest.py`.
The 2026-08-18 and 2026-08-22 failures were both caused by the then-absent
`operator_candidate_20260816` fixture directory, which is now tracked.

### TempoRAC scripts

TempoRAC has no `pams` sub-command and no YAML configuration. None of these
scripts constitutes or authorizes a formal campaign job.

```bash
# Synthetic X0 only. The seed must be 20260815, 20260816 or 20260817. The script refuses to
# overwrite an existing file and labels its output CANDIDATE_PENDING_PROTOCOL_REVIEW.
python scripts/experiments/train_temporac_teacher_candidate_v2.py \
  --output runs/temporac/teacher_v2_smoke.pt --seed 20260815 --steps 30 --device cuda   # short smoke run
python scripts/diagnostics/audit_temporac_teacher_x0_checkpoint_v1.py \
  --checkpoint runs/temporac/teacher_v2_smoke.pt --device cuda   # all 56 X0 tune certificates
# Natural data: run from the repository root with the v44 pickles under
# data/counting_multirep_skeleton_pose_expanded_v44_len320/
python scripts/diagnostics/audit_temporac_v44_preprocess.py --repository .     # frozen entrance
python scripts/diagnostics/audit_temporac_v44_resampled_v2.py --repository .   # resampled-v2 entrance
python scripts/diagnostics/audit_temporac_landmark_coverage_v2.py --repository .
python scripts/diagnostics/smoke_temporac_pseudo_targets_v14.py --repository . \
  --checkpoint TEACHER.pt --split val --limit 134 --direction-blind --summary-only  # v18 cascade
```

The teacher script accepts up to 20,000 steps, but full-length runs are
neither budgeted nor authorized: the 2026-08-18 candidate runtime measured
about 2–3 s per teacher step, so a 20,000-step job would exceed the frozen
six-A6000-hour ceiling per teacher, and the candidate report states that
additional blind training is not justified
([`TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md`](TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md);
[`runtime.py`](src/pams/temporac/runtime.py)). No timing is recorded for the
v2 script.

The recorded v8–v18 rows did not use `train_temporac_teacher_candidate_v2.py`.
The candidate report describes their teacher `5e5588cf…` as a label-free
1,000-step natural-decoder run started from the v6 protocol-revision
checkpoint (seed 20260815, 2,000 steps, anchor weight 0.1, certificate
targets). The scripts whose options and runtime labels match these two runs
are `train_temporac_natural_decoder_candidate_v1.py` and
`train_temporac_teacher_phase_anchored_v1.py`; the exact command lines are not
recorded.

The same directories hold threshold and landmark sweeps, earlier
pseudo-target smokes, A005 prechecks, manifest generators and the ablation
runner and scorer (`run_temporac_ablation_suite_v1.py`,
`score_temporac_ablation_predictions_v1.py`). No script trains the response
model (`TempoRACResponse`): `run_temporac_ablation_campaign_v1.sh` calls a
trainer outside the repository, and its summary records `metrics: null`.

### PAMS reproduction

```bash
pams config validate configs/pams.yaml
pams synthetic evaluate --config configs/pams.yaml   # spectral proxy, counts 2-40; --stress adds stress cases
pams synthetic counter-gate                          # 576-case counter test with the exact period
pams synthetic period-counter-gate                   # same matrix with estimated periods
pams synthetic sshead-collapse                       # SSHead collapse diagnostic; currently fails and exits nonzero
pams data prepare-ucfrep /path/containing/UCF-101 --output data/manifests/ucfrep_526.json
pams data split data/manifests/ucfrep_526.json --output data/manifests/ucfrep_337_84_105.json
pams pose extract data/manifests/ucfrep_526.json data/pose-cache --config configs/pams.yaml --skip-existing
pams train encoder MANIFEST POSE_CACHE RUN_DIR --config configs/pams.yaml --device cuda --epochs 150 --microbatch-size 32
pams train sshead RUN_DIR/encoder.pt MANIFEST POSE_CACHE SSHEAD_DIR \
  --encoder-progress RUN_DIR/logs/encoder.jsonl --config configs/pams.yaml --device cuda --epochs 30 --microbatch-size 8
```

`--no-hash-videos` skips video hashing, and `--annotation-only` yields a
manifest that is not training-ready. `pams data split` accepts only seed
2026, and `--skip-existing` skips a cache entry only when its source hash and
provenance match. The formal procedure for seeds 42/2026/3407, including the
one-attempt sealed-test evaluation through `pams evaluate checkpoint`, is in
[`docs/SERVER_RUNBOOK.md`](docs/SERVER_RUNBOOK.md). `pams baseline predict`
runs only the diagnostic `spectral-proxy`. `pams warp-phase --help` lists the
retained WARP-PHASE tooling, whose `train` command returns a blocked receipt.

## Results

All numbers are copied from the cited files. **None of them is a TempoRAC
counting-accuracy result on natural data.**

### TempoRAC: entrance and certification diagnostics

Population: the v44 pilot, with 268 train and 134 validation identities.
The rows report entrance acceptance and label-free certification coverage
(identities with a certifiable pseudo-target), not counting accuracy. From v8
onward they use the diagnostic teacher `5e5588cf…2bee50`. v9 certifies only
two stable events per identity. The v13, v15 and v18 cascades fall back to v9
when the complete-chain certificate abstains, so their coverage mixes
complete-sequence and two-event pseudo-targets. The validation split has
already been inspected, and further threshold tuning on it is prohibited.

| Diagnostic | Train | Validation | Recorded status |
|---|---:|---:|---|
| Frozen preprocessor entrance | 172/268 (64.18 %) | 91/134 (67.91 %) | FAIL (hard pre-training failure) |
| Resampled-v2 preprocessor entrance | 268/268 | 134/134 | Diagnostic pass; not in the contract |
| Frozen certificate, ≥ 3 landmarks after resampling | 45/268 (16.79 %) | 24/134 (17.91 %) | Below the 80 % K1 requirement |
| v9 filtered stable pair | 155/268 (57.84 %) | 98/134 (73.13 %) | Conservative diagnostic |
| v13 fixed cascade | 202/268 (75.37 %) | 104/134 (77.61 %) | FAIL precheck (minimum 108/134) |
| v15 certificate-bound cascade | 213/268 (79.48 %) | 109/134 (81.34 %) | Validation clears the 108/134 numeric threshold (train is below 80 %). Diagnostic only: designed after inspecting validation; no independent review |
| v18 cascade | 217/268 | 109/134 | Raw trace and diagnostic JSON only; validation threshold sweep also recorded; no ledger entry or review |

Sources: [status audit](TEMPORAC_EXPERIMENT_STATUS_20260818.md),
[candidate report](TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md),
[ledger](results/temporac/ALL_EXPERIMENT_RESULTS_20260818.md); for v18,
[`train-full-v18.json`](results/temporac-v18-diagnostic/train-full-v18.json)
and [`val-full-v18.json`](results/temporac-v18-diagnostic/val-full-v18.json).

| Other TempoRAC record | Recorded outcome |
|---|---|
| Frozen-F11-style teacher, 500 steps | 0/56 X0 tune certificates. Negative phase-edge fraction 0.497229 against a limit of ≤ 0.01, a near-constant-phase degenerate solution. |
| v6 protocol-revision teacher, 2,000 steps (not frozen F11) | 56/56 X0 tune certificates; 0 of the first 20 natural train identities |
| Non-contract X0-only checkpoint (4,000 steps and seed 20260818 per its file name, outside the frozen seed set) on 134 v44 validation identities (X0 tune loss 0.004117) | Raw count MAE 21.604477611940297; relative NMAE 0.9915196743554954; OBO 0.0; predictions 0–4 against targets 4–92. Labelled "X0-only diagnostic; not TempoRAC result". |
| Planned evaluation (unmeasured) | Main comparisons, seven paired ablations, synthetic 3x3 specialization, nonstationary-tempo stress and bootstrap intervals: `BLOCKED_UPSTREAM`. Tracking diagnostics: `BLOCKED_SCOPE`. |

The implemented cue is the NUDFT ([`cue.py`](src/pams/temporac/cue.py)).
`src/pams/temporac` scores only per-identity relative error and OBO, with a
paired video-cluster bootstrap and the K7 rule
([`metrics.py`](src/pams/temporac/metrics.py)); it has no Period-AP or
tracking metric. A supplied-track Period-AP adapter exists only in the
WARP-PHASE [`evaluator.py`](src/pams/warp_phase/evaluator.py).

### PAMS reproduction (foundation)

The evaluation uses the UCFRep-526 fixed 84-video development split, a
stratified seed-2026 subset of the 421 training videos, at source revision
`f99f3fda91b8bf2c25fc75a42814d9c97df71ae0` with seeds 42, 2026 and 3407.
NMAE is mean(|round(pred) − gt| / gt), OBO is the fraction with
|round(pred) − gt| ≤ 1, and raw MAE and RMSE are in counts. Values are the
three-seed mean ± sample SD and are **development-only negative results**
([`results/README.md`](results/README.md)).

| Variant | NMAE | Raw MAE | RMSE | OBO | Seeds passing gate |
|---|---:|---:|---:|---:|---:|
| PAMS-Literal | 0.729925 ± 0.155861 | 4.309524 ± 1.000071 | 6.071914 ± 1.131865 | 0.250000 ± 0.083333 | 0/3 |
| PAMS-SSHead (inferred) | 4.941204 ± 0.323411 | 18.876984 ± 1.284630 | 21.509048 ± 1.463711 | 0.051587 ± 0.018185 | 0/3 |

| Seed-2026 dev84 diagnostic | NMAE / OBO | Note |
|---|---|---|
| PAMS-SSHead, v8 longest-contiguous-track preprocessing | 0.643123 / 0.297619 (matched v7: 0.735728 / 0.190476) | Missed the extension gate (NMAE ≤ 0.60, OBO ≥ 0.30), so seeds 42 and 3407 were not run ([record](results/dev-negative/pams_sshead_longest_track_v8_seed2026_07192de/README.md)). [`configs/pams.yaml`](configs/pams.yaml) keeps the v2 policy. |
| Frozen-v8 direct readout, PAMS-Literal | 0.664627 / 0.261905 | Development-only |
| Frozen-v8 pose-only local-frequency readout | 0.529058 / 0.333333 | Not a PAMS component; table-ineligible |
| `PAMS-TeacherPeriodDirect-inferred-v1` | 2.860120 / 0.166667 | Post hoc; permanently table-ineligible |

A later strict, target-isolated rerun of the paper-config full multi-scale
model
([`pams_full_multiscale_three_seed_strict.json`](results/dev-negative/pams_full_multiscale_three_seed_strict.json))
gives PAMS-Literal 1.489633 ± 1.439463 / 0.226190 ± 0.154762 and PAMS-SSHead
5.022471 ± 0.505117 / 0.051587 ± 0.013746 (NMAE / OBO, three-seed mean ±
sample SD). It also passes on 0/3 seeds and is also development-only
([`results/dev-negative/README.md`](results/dev-negative/README.md)).

Official-checkpoint runs of RepNet, TransRAC, ESCounts and IVAC-P2L are
development-only dev84 sanity checks under this repository's protocol, not
original-protocol parity; TransRAC and IVAC-P2L ran in labelled
modern-compatibility environments. The released PoseRAC-v1 checkpoint was
run only as an oracle-free dev84 diagnostic on the 256-frame PAMS cache,
which disables the released evaluator's ground-truth-count oracle and adds an
independently inferred channel selector
([`results/README.md`](results/README.md)). Official checkpoints are used only
in this separately labelled sanity track. Independent baseline
implementations must be clean-room and may not read any held-out count,
action identity or cycle boundary; blocked methods raise
`BaselineUnavailableError`, and the sealed-report allowlist is empty
([`docs/BASELINE_PROTOCOLS.md`](docs/BASELINE_PROTOCOLS.md)).

Synthetic acceptance passed both 576-case counter gates and failed the
55-case pose-level spectral-proxy gate
([`results/synthetic/README.md`](results/synthetic/README.md)). The
paper's UCFRep values (NMAE 0.208, OBO 0.686) are targets, not results of
this repository ([`docs/PAPER_AUDIT.md`](docs/PAPER_AUDIT.md)); verification
would require a three-seed mean NMAE ≤ 0.228 and OBO ≥ 0.666, at least two
passing seeds and the preregistered ablation directions
([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)). The sealed 105-video test split
has no predictions or metrics, although an older development CLI did
deserialize its manifest.

### WARP-PHASE and supervised side probes

| Record | Outcome |
|---|---|
| WARP-PHASE Gate 2 (K5), terminal failure | Agreement 246/1000 (needs ≥ 950). Overall half/double selections 55/144 (needs ≤ 20/≤ 20). Symmetric half/double selections 50/66 (needs ≤ 12/≤ 12). The integrity audit is `FAIL` ([claims](CLAIMS_FROM_RESULTS_WARP_PHASE.md); [audit](EXPERIMENT_AUDIT_WARP_PHASE.md)). |
| `results/autoresearch-v19/`, `-v20/` | Undated, count-supervised probes on v44: a validation-selected regressor (v19) and a density counter (v20); `20260925` in `density-seed20260925` is the v20 default seed, not a date |
| `results/autoresearch-v21/` | Supervised MotionBERT prototype calibration on the MPCounter `seed20260725/unseen_person_count` protocol (not v44). Its artefacts are byte-identical to the replay bound in [`MOTIONBERT_GENERALIZATION_BINDING_20260818.md`](MOTIONBERT_GENERALIZATION_BINDING_20260818.md), and it scores that protocol's test split once: AvgMAE 0.16792673992673993, AvgOBO 0.725. The `PASS` in its [README](results/autoresearch-v21/README.md) is against a target set at the historical values (AvgMAE ≤ 0.168, AvgOBO ≥ 0.725). Protocol seed 20260725 is the best of three (AvgOBO 0.06875 / 0.30625 / 0.725; mean about 0.36667). Not a TempoRAC, UCFRep or v44 result. |
| Server candidates v46, v62, v63 | Audited as ineligible ([`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md)) |

The side probes train on count labels. They must not be presented as
TempoRAC, unsupervised or three-seed results.

## Reproducibility and provenance

| Item | Record |
|---|---|
| Research workflow | ARIS (Auto-Research-In-Sleep), pinned at `e12e07c7b85ee1a4dc07e5463089aa16836af2bf` ([`ARIS_LOCK.md`](ARIS_LOCK.md)). Proposals, plans, trackers and reviews are in [`refine-logs/`](refine-logs/). Contract amendments A001–A006, implementation reviews and raw traces are in `refine-logs/temporac/`. Run records are in `.aris/`. Generated artifacts are logged in [`MANIFEST.md`](MANIFEST.md). |
| Checksum ledger | [`MANIFEST.sha256`](MANIFEST.sha256) covers selected ARIS artifacts only. In the current tree, 7 of its 411 entries do not match, including `MANIFEST.md` and both copies of the proposal; the current proposal digest is the one pinned in `contract.py`. |
| Contract binding | `PROPOSAL_SHA256` and `CONTRACT_SHA256` in `contract.py` bind the proposal and the effective contract index; the trusted loader binds the v44 data. The handoff orders the next steps: A005 Round 9 review, contract rebinding, native runtime, then the P00–P06 gates. |
| PAMS run receipts | Every formal command writes an immutable `*.started.json` and, on success, a hash-linked `*.completed.json`. They bind the Git commit, configuration and dataset hashes, seed, command, hardware, pose fingerprint and an ordered pose-cache digest. An atomic registry allows one sealed-test attempt per method and seed, and `pams data validate-run` re-validates receipts ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)). |
| Automation | [`ci.yml`](.github/workflows/ci.yml) runs lint and the full test suite. [`publish-server-image.yml`](.github/workflows/publish-server-image.yml) publishes the PAMS server image to GHCR on pushes to `main` or `agent/initial-reproduction` that touch `docker/server/**`, its two build scripts or the workflow itself, on any `v*` tag push (GitHub does not apply path filters to tag pushes), and on manual dispatch. |

## Documentation index

| Document | Content |
|---|---|
| [`refine-logs/temporac/FINAL_PROPOSAL.md`](refine-logs/temporac/FINAL_PROPOSAL.md) | Normative TempoRAC specification `temporac.execution.v4` |
| [`idea-stage/docs/research_contract.md`](idea-stage/docs/research_contract.md) | Supplied-track pilot research contract |
| [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md), [`refine-logs/EXPERIMENT_TRACKER.md`](refine-logs/EXPERIMENT_TRACKER.md) | TempoRAC experiment plan and tracker (in Chinese) |
| [`TEMPORAC_HANDOFF_20260817.md`](TEMPORAC_HANDOFF_20260817.md) | Handoff state, open A005 issues, continuation order |
| [`TEMPORAC_EXPERIMENT_STATUS_20260818.md`](TEMPORAC_EXPERIMENT_STATUS_20260818.md) | Status audit and blockers in execution order |
| [`TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md`](TEMPORAC_PREPROCESS_RUNTIME_CANDIDATE_20260818.md) | Preprocessing, teacher and certificate candidates v1–v10 |
| [`results/temporac/ALL_EXPERIMENT_RESULTS_20260818.md`](results/temporac/ALL_EXPERIMENT_RESULTS_20260818.md) | Append-only TempoRAC ledger and 27-job campaign ledger (entries through 2026-08-19) |
| [`PILOT_DATA_SCHEMA_AUDIT.md`](PILOT_DATA_SCHEMA_AUDIT.md) | MultiRep v44 pilot data audit |
| [`CLAIMS_EVIDENCE_MATRIX.md`](CLAIMS_EVIDENCE_MATRIX.md) | Paper-package claim-to-evidence ledger |
| [`CLAIMS_FROM_RESULTS_WARP_PHASE.md`](CLAIMS_FROM_RESULTS_WARP_PHASE.md), [`EXPERIMENT_AUDIT_WARP_PHASE.md`](EXPERIMENT_AUDIT_WARP_PHASE.md) | WARP-PHASE decision and integrity audit |
| [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md), [`docs/PAPER_AUDIT.md`](docs/PAPER_AUDIT.md), [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) | PAMS specification, paper audit, assumption register |
| [`docs/BASELINE_PROTOCOLS.md`](docs/BASELINE_PROTOCOLS.md), [`docs/baselines/`](docs/baselines/) | Baseline registry and official-checkpoint audits |
| [`DATA.md`](DATA.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), [`docs/SERVER_RUNBOOK.md`](docs/SERVER_RUNBOOK.md) | UCFRep preparation, PAMS reproducibility contract, GPU server procedure |
| [`results/README.md`](results/README.md) | PAMS result status and the intentionally empty fair-comparison tables |
| [`configs/experiments/README.md`](configs/experiments/README.md), [`configs/smoke/README.md`](configs/smoke/README.md) | Inferred experiment and smoke configurations (not table-eligible) |
| [`paper/README.md`](paper/README.md) | ARIS pre-results draft package |
| [`MOTIONBERT_GENERALIZATION_BINDING_20260818.md`](MOTIONBERT_GENERALIZATION_BINDING_20260818.md) | Replay binding and metric definitions for the supervised v21 side probe |
| [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md), [`PAPER_PLAN.md`](PAPER_PLAN.md), [`PAPER_ACCEPTANCE_CONTRACT.md`](PAPER_ACCEPTANCE_CONTRACT.md), [`NARRATIVE_REPORT.md`](NARRATIVE_REPORT.md), [`CLAIMS_FROM_RESULTS.md`](CLAIMS_FROM_RESULTS.md), [`findings.md`](findings.md), [`SERVER_METHOD_INVENTORY.md`](SERVER_METHOD_INVENTORY.md), [`PILOT_PERIOD_SCHEMA_ADDENDUM.md`](PILOT_PERIOD_SCHEMA_ADDENDUM.md), [`IMPLEMENTATION_SYNC_REPORT_ZH.md`](IMPLEMENTATION_SYNC_REPORT_ZH.md) | ARIS planning, audit and findings records for the draft package; their semantic reviews are same-family and provisional |
| [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md) | Contribution and security policy |

## License and third-party notices

The original code is licensed under the Apache License 2.0
([`LICENSE`](LICENSE)). Datasets, papers, third-party code and model weights
keep their own licences; this repository neither relicenses nor redistributes
them. The baseline sidecar images under `docker/*-official/` define runtime
environments only: per their `ENVIRONMENT.md` files, upstream sources,
checkpoints and videos are mounted at run time, and installed packages keep
their own licences. `paper/vendor/icassp2026/` is vendored from the official
ICASSP 2026 LaTeX kit (source and hashes in
[`paper/TEMPLATE_PROVENANCE.md`](paper/TEMPLATE_PROVENANCE.md)); it is not
original code of this repository and is not relicensed.

## Citation

```bibtex
@software{yan2026temporac,
  author  = {Yan, Weiping},
  title   = {{TempoRAC}: Identity-Indexed Local Tempo Routing for Multi-Person Repetitive Action Counting},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/appleweiping/TempoRAC}
}
```

If you use the PAMS reproduction, please also cite *Count What Repeats:
Period-Adaptive Multi-Scale Consistency for Self-Supervised Repetitive Action
Counting* (CVPR Findings 2026; link in [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md)).
The tracked [`CITATION.cff`](CITATION.cff) still carries the earlier title
(`PAMS RAC Reproduction`) and the former repository URL.

## Acknowledgements

| Upstream work | Reference |
|---|---|
| Original identity-indexed, window-local tempo-routing idea | Project collaborators ([`idea-stage/docs/research_contract.md`](idea-stage/docs/research_contract.md)) |
| PAMS (CVPR Findings 2026) | [`docs/METHOD_SPEC.md`](docs/METHOD_SPEC.md) |
| ARIS (Auto-Research-In-Sleep) workflow | Upstream repository and pinned commit in [`ARIS_LOCK.md`](ARIS_LOCK.md) |
| UCF101 and the UCFRep annotations | [`DATA.md`](DATA.md) |
| MultiRep and the official MultiCounter+ evaluator | Pinned commit in [`PILOT_DATA_SCHEMA_AUDIT.md`](PILOT_DATA_SCHEMA_AUDIT.md) |
| AlphaPose (v44 pose estimates) and MotionBERT (v21 side probe) | [`PILOT_DATA_SCHEMA_AUDIT.md`](PILOT_DATA_SCHEMA_AUDIT.md); pinned commit in [`MOTIONBERT_GENERALIZATION_BINDING_20260818.md`](MOTIONBERT_GENERALIZATION_BINDING_20260818.md) |
| MediaPipe Pose, PyTorch, NumPy | [`pyproject.toml`](pyproject.toml) |
| Official baseline releases (development-only sanity checks) | [RepNet](https://github.com/google-research/google-research/tree/ec7c3d346277b737bc2decffcd1b533d4b7ec105/repnet), [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC); ESCounts (EveryShotCounts), IVAC-P2L and PoseRAC in [`docs/BASELINE_PROTOCOLS.md`](docs/BASELINE_PROTOCOLS.md) |
