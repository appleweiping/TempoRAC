# RepNet official adapter independent review — 2026-11-11

Verdict: **MUST_FIX before environment witness or full evaluation**. This is an execution-readiness/static implementation review, not a performance, SOTA, or project-completion verdict. No code was changed and no experiment, dependency installation, model import, checkpoint restore, or GPU operation was performed by this reviewer.

Reviewer session: `/root/repnet_official_review`; requested model `gpt-6.1-sol`; project-required reasoning floor `xhigh`. Actual backend/model/effort are not independently attested. Native OpenAI same-family review; semantic assurance remains **provisional**. This fresh review received primary paths and read their contents directly.

## Acceptance checklist and evidence

| Requirement | Status | Direct evidence |
|---|---|---|
| Literal released model/count/preprocess implementation | Pass, static | All 13 active class/function definitions match the pinned notebook exactly after trimming only surrounding whitespace; full positional mapping matches exactly. Model, preprocess, get_score, get_counts, read_video and get_repnet_model included. |
| Correct released asset identity | Pass, static receipt only | Notebook local SHA is `886367d108ef270baa4bcd9ecef58eff6552f6b3eec8e675268e2ef9f1f4129c`; three adapter source SHA values and config SHA match acquisition receipt. Server asset bytes/restoration were not reverified here. |
| Runtime matches released source | **Not met** | TF2.15.1/Keras2.15 conflicts with literal `tf.keras.Variable` constructors at core lines107/113. |
| All 357 evaluation records, zero and ambiguity retained | Pass, static | Snapshot manifests contain rc_test152, ucf_test105, QUVA test100. RepCount `train143` count0, `stu4_5` count51, `stu4_27` count56 are retained. Runner has no count filter or skip-on-error branch. |
| Same evaluation temporal inputs | **Not met** | All105 UCF rows have `frame_range`;51 have a nonzero start. Runner passes the entire AVI to get_counts. |
| Frozen harmonized and native-derived metrics separated | Pass, static | frozen_metrics.py is byte-identical to snapshot `temporac_cvpr/metrics.py`; np.round, positive +0.1, positive /g, full population and zero rows retained. Continuous metrics are explicitly described as derived, not an official notebook scorer. |
| No ground-truth tuning or action-class oracle | Pass, static | Model receives frames plus fixed notebook inference settings. Count/action/bounds are not used for stride selection, threshold selection, or model/channel choice. UCF temporal segmentation is part of the fixed evaluation protocol and must be applied. |
| Full run failure/completion behavior | Partial | Missing/size/hash-mismatched assets abort; failures preserve completed JSONL/NPZ records; no success marker until loop completion. Claimed incomplete-decode detection is absent. No resume implementation is claimed/available: existing output directories are refused. |
| Authorized resource gate | **Not met** | Config selects GPU0 although authorization is GPU1 after full_s1 plus fixed evaluation. Runner checks only free memory, not predecessor completion or workers. |
| Reviewable runnable commands | **Not met** | SPEC and BUILD_NOTE commands omit required --mode/--output or use nonexistent --preflight/--witness flags. |
| Server model restore/kernel/full357 evidence | Pending | Correctly not run. Current draft cannot be labeled runnable or restored from static evidence. |

## Findings requiring correction

### R1 — P1: declared environment cannot construct the literal model

`repnet_official_core.py:107–117` calls `tf.keras.Variable(initializer, shape=..., trainable=True, ...)`. The selected `tensorflow==2.15.1` package requires `keras>=2.15,<2.16`; that is the legacy Keras2 API, not the Keras3 Variable API used by the 2025 notebook. The runner's comment that only `weight.value.shape` needs a Keras2 workaround does not address construction. A model witness will fail before restore.

Prefer preserving the literal author core and choosing one coherent, explicitly pinned Keras3/TensorFlow/Python stack. Update env-spec, assumptions and restoration commentary together, then perform complete restored-variable and finite-kernel checks on the server in the authorized window. Do not silently replace the positional variables or reinterpret an environment repair as a new model. NumPy1.26.4 satisfies TF2.15.1's declared `>=1.23.5,<2` requirement; active core uses `np.float32`, not removed `np.float`/`np.int` aliases.

Primary package evidence: [TF2.15.1 setup.py](https://github.com/tensorflow/tensorflow/blob/v2.15.1/tensorflow/tools/pip_package/setup.py#L81), [Keras compatibility](https://keras.io/getting_started/#compatibility-matrix). Current Keras documentation also requires Python3.11 for current Keras3, so copying its latest pins while retaining Python3.10 is not a complete repair.

### R2 — P1: GPU choice and predecessor gate do not implement authorization

`evaluation_config.json` sets `runtime.gpu=0`; `evaluate_repnet.py:111–112` obeys it. `gpu_check` at90–100 only reads free memory. It does not establish that GPU1's `full_s1` and its fixed full evaluation have completed, that the corresponding workers are gone, or that GPU1 is free for this job. Starting this draft can contend with locked full-method training or its evaluator.

Set the intended GPU to1. Before any witness or evaluation import, require a finite, explicit predecessor gate using the actual `full_s1/complete_ks.json`, expected worker/process state and fresh GPU1 ownership/free-memory evidence. Do not interrupt workers, run an unbounded poller, or add a scheduler. The configured8GiB cap and10GiB free check are proposed limits only; their sufficiency is not proved statically.

### R3 — P1: UCF score is for a different temporal input

`evaluate_repnet.py:163–170` decodes the full file and immediately calls get_counts. It ignores `frame_range`. Snapshot `prepare_ucfrep.py:3–7,72–75` specifies an annotated repetitive segment, with zero-based inclusive endpoints. Snapshot `extract_features.py:107–114` applies that crop and records its existing end clamp. All105 UCF evaluation rows carry this field. For example `v_JumpRope_g22_c04` has `[2,224]` and target11; the target refers to223 selected frames, not all frames of its source AVI.

Apply the frozen UCF crop outside the unchanged official core before inference, using the same inclusive/clamp convention as the existing evaluation inputs. Preserve requested and effective range, full decoded length and model input length in output evidence. Preserve all105 rows; do not change counts or choose boundaries from observed RepNet errors. No crop change is needed for the current rc_test or QUVA full-video rows.

### R4 — P2: incomplete decoding is silently accepted despite A006

`read_video` at446–461 treats any failed read as EOF. Adapter165 only rejects empty/nonfinite arrays; no metadata count or existing authoritative decode-length check exists. Thus a positive-length truncated decode is accepted and can reach completion.json, contrary to A006's explicit fail-on-truncated-metadata-disagreement promise. File SHA verification proves asset identity, not successful full decoding in this runtime.

Implement the declared decode-completeness check in the thin adapter and record the evidence, or explicitly narrow the claim while providing the already-frozen equivalent full-input check. Account for the frozen legitimate UCF endpoint clamp separately; do not introduce a new silent input convention. Retain the author's pixel transforms and sequential order.

### R5 — P2: published success/check commands do not parse

SPEC's success command has no required `--mode`. BUILD_NOTE lists nonexistent `--preflight` and `--witness`, and those commands also omit required `--output`. The actual argparse interface at218–223 is `--config PATH --output UNIQUE_DIR --mode preflight|witness|evaluate`. Correct every command to that interface and distinct output directories. The readiness document's full-evaluation command is already structurally correct.

## Correct and bounded aspects

The core is not an internal-style approximation. Exact matching includes ResNet50V2 through conv4_block3_out,64 frames,112 model resolution, the two positional variables, temporal convolution, similarity/transformer heads, normalization, official stride selection and threshold/median filter counting. Settings match the notebook: strides1/2/3/4, batch20, threshold0.2, within0.5, median filter enabled, constant-speed/fully-periodic disabled. The complete mapping literal also matches; the adapter adds count/shape/equality checks but their server outcome remains pending.

Snapshot metrics copy is exact. No train rows are passed to the model, no data-dependent parameter selection appears, no class oracle is present, no row is silently skipped, and the updated2025 checkpoint is accurately separated from original2020 paper and third-party retraining numbers. Per-row raw continuous counts and frame traces are retained. Refusal to reuse an existing output directory is consistent with the declared first-run no-auto-resume policy; a restart requires a new directory and keeps the old partial records.

## Scope/read record

Read complete E:/AGENTS.md, repository AGENTS.md, ARIS_LOCK.md, WORKFLOW.md, current progress/training contract, pinned research-implement-feature and research-review skills, and their applicable reviewer/assurance/scope/language/trace/effort/injection references. Pinned and installed research-implement-feature SKILL.md both hash to `33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8`.

Read every primary adapter file, the original notebook, acquisition_receipt and BASELINE_READINESS. Parent then authorized exact snapshot input-protocol sources; read `prepare_ucfrep.py`, relevant `data.py`/`extract_features.py` input sections and UNION/QUVA manifests, and compared frozen_metrics to snapshot metrics.py. Comparisons, JSON inspection and file hashes are static file inspection, not local experimental validation.

Next gate is a corrected-draft static recheck of R1–R5, followed by authorized server CPU/runtime evidence and full357 execution only after GPU1's predecessor condition. No SOTA, independence-clean dataset, accepted semantics, or goal-completion claim follows from this review.
