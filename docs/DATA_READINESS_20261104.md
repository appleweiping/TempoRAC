> Current full-production status: [FULL_NATIVE_CHECKPOINT_20261104.md](FULL_NATIVE_CHECKPOINT_20261104.md). Eight full label-free stages are complete; full isolated diagnosis is complete; once-only continuation provisionally permits the frozen full method test. Older entries below are historical snapshots.

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


Actual full generation now runs on GPU1 (wrapper1567147/child1567156); see `evidence/native_launch_20261104.json`. First native raw artifact has216physical presentation-frame rows,2identities,FP32COCO17,216rawPTS and430/432 valid bbox/pose person-frames; missing rows are zero+false masks. This is one necessary full-run interface receipt, not a separate GPU pilot or complete training-input acceptance. The loaded extractor file is frozen while this job runs.


## Native learner and clock qualification

NB1 engineering implementation/review is complete at same-family provisional assurance; full977/2364 build is not yet run. All original raw records remain, with no forced320 sampling or invented observations. Three reviewed target/ledger defects were corrected; see `NATIVE_LEARNER_BRIDGE_REVIEW_20261104.md`.

Before complete NB1 build, freeze the977 clock overlay only after both full raw and decode finish. Overlay binds original raw artifacts, exact decoded row bytes/CSV and source identity, and resolves only genuinePTS; it cannot count decoder best-effort or FPS as observed PTS. This adds no pose observations and requires no re-extraction. NT1/LF1 and full target/error/coverage diagnosis remain separate next gates.


NT1/LF1 interfaces now have independent same-family provisional engineering review (18CPU tests, including certified-success numerical parity and autograd), under explicitly unavailable production hybrid. It does not supply empirical natural target coverage or units. The isolated evaluator implementation is separate; actual fulltrain/dev diagnosis waits for complete qualified artifacts, while alltestlabels remain outside this preparation stage.


The immutable clock-overlay/qualified-builder engineering path is checked, including an actual isolated NT1 consumer fixture. Full physical-clock/data coverage remains pending until the complete977 freeze/build. It does not rerunpose, add observations, reorder timestamps or modify the existing raw data. The diagnostic evaluator has two concrete input-reader corrections in progress and remains unaccepted.


All preparation readers now have checked engineering paths, including the isolated diagnostic evaluator after two narrow fixes. It has not opened naturalGT. Full overlay/qualifiedNB1/NT1 production and complete train/dev diagnostic still await the full raw cache. NTR1 runtime code is being prepared separately; complete training and quality requirements remain unmet.


## Full decode closure and NTR1 R2a checkpoint

The original full CPU decode job finished all794 train/183 development videos:695625 frames, rc0 for977, zero row findings or closure discrepancies. All977 raw clock CSVs and1954 empty stderr files are archived; E restoration verified2942 artifacts/30800644 bytes with zero mismatches. PTS and best-effort clocks have no missing/duplicate/reversed ticks;1954 missing DTS remain an explicit diagnostic. Five time bases and observed rates are retained. This closes the existing job; do not decode it again. `FULL_DECODE_COMPLETION_20261104.md` and `evidence/full_decode_completion_20261104.json` contain exact definitions, hashes and limitations, including prior media SHA copied rather than recomputed during decode and original process exit code not separately captured.

The healthy full native pose job remains active (293/977 complete at12:18UTC); immutable raw artifacts, clock overlay, complete qualified NB1/NT1 and target-unit/error/coverage diagnosis still precede training. This data completion is not a scientific performance result.

Astra NTR1 R2a clarifies that another PID gaining1MiB while actual free memory remains10GiB does not stop this job; actual2/5/6GiB bounds and the same arm's cumulative24h budget govern resource exits. Full train/dev maximumN and guarded prediction share the budget. Original R2 stays in Git401d8a5. The original runtime review found four concrete blockers; the author is applying bounded fixes and the original reviewer is checking them. No GPU training or naturalGT evaluation has occurred. CPU MKLDNN validation produced nonfinite gradient failures; explicit CPU-native convolution validation passed with unchanged numerical tolerances. Both traces remain. A newly frozen two-process identical X0 bank has all40 old orbit-grid SHA differences; no unavailable old array comparison is inferred to prove roundoff equivalence.


## NTR1 checked runtime checkpoint

The new native training runtime and analytic X0 adapter have completed the original independent Sol/xhigh review and bounded correction checks. All four original findings and intermediate failing probes remain in `NATIVE_TRAINING_RUNTIME_REVIEW_20261104.md`: actual producer receipt/provenance interoperability, loss-side whole-traversal eligibility, harmless shared-GPU growth, and full train/predict setup/runtime failure accounting. The final CPU suite passed26 tests; original mathematics/sampling/selection and full-population mocked entry probes were separately checked. This is same-family provisional engineering assurance, not scientific acceptance or empirically measured GPU feasibility.

CPU convolution uses the explicitly disclosed native backend after observed MKLDNN nonfinite gradients, without relaxed tolerances. Two fresh X0 processes produced identical40-source/224-view arrays under the recorded new environment; all40 old orbit-grid SHA differences remain. The first16-pass/2-fail owned-suite full log path was reused: only its explicitly labeled excerpt survives, while separate raw gradient/backend failure traces remain. No original full-log preservation is claimed. The first full794/183 canonical/control jobs still require complete raw/qualified inputs, NT1 and isolated full target-unit/error/coverage diagnosis, the evidence-grounded continuation decision, and the exact frozen workload/live lease. No naturalGT evaluation or actual GPU training/prediction occurred in this checkpoint.


## DDF1 and fractional-support correction

Astra DDF1 freezes the first full target diagnostic's proposed [s,e), e-s, count/period and bootstrap0 policy before new natural-GT access/results. Official endpoints/count parity remain uncertified; target-quality tolerance, PeriodAP and statistical/scientific claims remain unmet. Count0/abstention and every original population/alias/conflict row remain visible. See `DIAGNOSTIC_ENDPOINT_FREEZE_20261104.md`.

The old diagnostic code was actually reproduced on synthetic inputs: a positive tail shorter than1e-12 could be classified both full/partial and give aggregate unsupported=-1; a fully unsupported short interval could also be labeled full. The minimal classification correction uses exact positive-overlap mask membership, preserves the raw fraction and makes the three categories disjoint. Original38 tests and new regressions pass53 owned CPU checks; a fresh narrow independent review is recorded in `NATIVE_DIAGNOSTIC_SUPPORT_REVIEW_20261104.md`. The old findings/negative probes remain. No naturalGT, production build, GPU training or model performance was run by this correction. Full raw extraction remains the live prerequisite for complete977/2364 production and isolated diagnosis; do not rerun the already completed decoder.
