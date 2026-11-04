# TempoRAC execution checkpoint — 2026-11-04

This is an engineering/data checkpoint. Formal model experiments, final scientific review and writing remain pending. Current private state and server run logs take precedence over this snapshot.

| Item | State | Evidence / next concrete work |
|---|---|---|
| Continuation / ARIS | Passed mechanically | Existing `3e9ae828` method retained; exact ARIS `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`; stage full-read receipts; no method restart |
| Literature / novelty | Provisional | Closest-work evidence and Astra caution preserved; MultiCounter+/TWCRAC full reading and original result reconciliation still incomplete |
| New server / separation | Passed infrastructure | Authenticated isolated TempoRAC root; all GPUs occupied at startup; STaR and others untouched |
| Complete candidate population | Preserved | 794 train / 183 development / 180 A candidate test; release/recipe identity not certified; no test labels read |
| Composite byte audit | Passed mechanically | 977/977, 4.62GB, zero failures; four train duplicate pairs, no train/val byte duplicate |
| Original-source byte audit | Passed mechanically, independence partial | 906/906, 8.41GB, 301 byte IDs; A180 retained, B78 candidate only, 102 overlaps; 9 B source components |
| Conflicting train labels | Observed; unresolved | Four exact-media pairs disagree 58/48 on aliased person count/period; bbox/periodicity agree; both labels retained |
| Full decoded frames / clocks | Complete mechanically | Original full977 job finished695625 frames with zero findings; archive/E restoration verified2942 artifacts. See FULL_DECODE_COMPLETION_20261104.md; do not rerun |
| Fresh input audit review | No remaining engineering blocker; same-family provisional | Fresh Sol/xhigh caught decoded-counter defect, fixed/rechecked; 16 CPU tests pass. Raw original finding retained in INPUT_AUDIT_REVIEW_20261104.md |
| Native pose frontend / contract | Checked implementation; full extraction running | Full977/2364 projection checked; GPU1 child1567156/wrapper1567147 healthy,300/977 complete at12:21UTC; immutable extractor source and raw originals retained |
| Learner / natural targets / scorer | Engineering paths checked; full scientific qualification pending | NB1, immutable clocks, NT1/LF1, isolated diagnostics and NTR1 runtime have same-family provisional reviews; full production/target diagnosis/training/official scorer parity still pending |
| Main results / 3seed / baselines / ablations | Not met | No natural full experiment result; six-dataset/seven-framework quality rule remains unmet |
| Observation / hyperparameters / efficiency / loss<3 | Partial | LF1 single class-balanced BCE criterion checked, replacing old margins; full component/mechanism/hyperparameter/efficiency results still unmet |
| CVPR final paper | Not met | Preserve prior drafts/numbers with reconciliation; experiments precede final writing, all-page render required for final PDF changes |
| Source synchronization | Previous common HEAD ae00d2f7dd7b7cbf347b29639757108408d4928e verified local/GitHub/server | Checked NTR1 source is checkpointed next; latest exact source/config/data receipt lives in server source_sync.json and E current_status.json; full scientific completion remains pending |

Storage startup: approximately81GiB free on project SSD and41GiB shared-data mount. Existing videos are reused read-only; no raw-frame duplication or cleanup. Detect-only ARIS watchdog uses E-disk state, stale threshold6h; it cannot create scientific verdicts or restart jobs.

Operational deviation: one author pytest invocation omitted TEMP/TMP/basetemp and created four fixture files (261 payload bytes) in a D-disk pytest temp directory. Subsequent checks explicitly use E for temp/cache/basetemp. Automatic policy rejected recursive cleanup and then a fixed-four-file cleanup before execution, giving only `blocked by policy`; files remain and no bypass was attempted. The all-E artifact requirement is therefore partial for that invocation. Research execution continues on E/server.

No Word/PDF was changed at this checkpoint. Private credentials, original inventories, restricted media/weights and transport logs remain outside Git.


Latest code checkpoint: full977/2364 bbox projection complete; fresh Sol/xhigh extractor review has no remaining concrete blocker after two resume corrections,12CPU tests pass. `NATIVE_POSE_CACHE_20261104.md` contains exact full-generation command; no cap. Raw generation is pending only the source-bound GPU allocation launch at this snapshot; no full CPU or GPU micro inference was used. Astra metric contract is documented, while official count parity and final primary freeze remain pending. Actual later source sync and job PID/log state are recorded in private current_status.json/server source_sync.json, rather than editing prior snapshots into completion claims.


Full977 native pose extraction actually launched10:16:56UTC on GPU1:wrapper1567147/child1567156, launch8326588, exact extractorSHA in `evidence/native_launch_20261104.json`. First216-frame2-person raw artifact passed local schema/hash validation;430/432 person-frames had valid supplied boxes, and the two unsupported rows remain. Native raw and full CPU decode continue; the latter was227/977 with0findings at10:17UTC. This is generation progress, not model performance. Lower-priority own FastSAM job is reversibly paused with matching-PID recovery; STaR/other-user jobs unchanged.

Named FULL-v1 CPU reference implementation passed27synthetic tests, original AP-kernel parity and independent hand calculations; freshSol review found no remaining concrete implementation blocker. Original documentation collection failure and correction retained. Oldmetrics unchanged, no real labels scored, no official count/main metric freeze claimed. Astra native learner bridge design is underway; deliberate sampling and observed support remain distinct.


## NB1 checked implementation checkpoint

NB1 now has a complete native-N implementation and fresh Sol/xhigh review with no remaining concrete engineering blocker. Initial scoped75 CPU tests passed; after three concrete corrections the independent impacted suite passed30 tests and Ruff. Raw original findings/reproductions and correction check remain in `NATIVE_LEARNER_BRIDGE_REVIEW_20261104.md`. Whole-traversal support loss now removes all its training weights; per-traversal/mixed provenance is retained; corrupt NPZ is hash-checked before loading and enters the full failure ledger. This is same-family provisional engineering assurance, not full-data/scientific acceptance.

At11:02:03UTC the full native job had127/977 complete, decoder556/977 with0 findings; actual source was common cb2fe0f and clean on server. Both full jobs continue. `evidence/runtime_snapshot_20261104_1102.json` is this older observed snapshot; private current state and server logs override it.

Astra fixed NT1/LF1 in `NATIVE_TARGET_LF1_CONTRACT_20261104.md`: exact existing per-run branch order, default unavailable hybrid, full-edge supervision ledger and one balanced-BCE criterion across branch/fused readouts. Implementation and independent review are proceeding; true target units/quality and complete production artifacts remain pending. Official release/scorer certification limits corresponding claims; it does not indefinitely block already authorized complete794/183 development experiments after actual data/target/diagnosis/budget freezes.

The native extractor reads decode receipts only once at launch. A separate immutable clock overlay is therefore required before the full NB1 build, to attach later genuinely observed PTS without changing raw poses or restarting jobs (`NATIVE_CLOCK_OVERLAY_DECISION_20261104.md`). Missing/duplicate/regressing PTS stay explicit support defects. Full physical-clock acceptance remains pending.

Bounded full-text retrieval plus connected-library metadata search found no usable primary PDF for MultiCounter+ or TWCRAC; coverage remains3/5full,2partial. Static official Plus interface discrepancies are recorded, without claiming a model-runtime or paper-experiment verdict. No Word/PDF modified.


NT1/LF1 checked implementation: the frozen hybrid-unavailable producer, G-NT1 counts and differentiable single-BCE criterion passed final18 CPU tests against NB1 SHA50aecf64/builder1822d120; a fresh independent Sol/xhigh reviewer separately passed18 tests and certified-success/offset/run/weighting probes, with no concrete remaining blocker. Raw executor evidence and review are retained. This validates the engineering path only. Complete target build, true-unit/count/coverage diagnosis, training sampler/budget/checkpoint freezes and formal experiments remain pending.

Clock overlay and the isolated target diagnostic evaluator are being implemented in separate new files, preserving all checked old and NB1/extractor APIs. No actual natural GT was read by these implementation tasks, no partial real-cache build was substituted for the complete population, and no training was launched.


Clock overlay and the separate qualified NB1 builder now have a completed freshSol/xhigh engineering review:61 impacted tests/Ruff passed, and actual NT1 diagnose-inputs consumed synthetic qualified2video/4identity artifacts in isolated server CPU scratch. No concrete blocker remains in that scope; semantic assurance is same-family provisional. OldNB1/extractor files and raw originals remain unchanged. Stream selection uses pinned OpenCV4.11 first-video code and ffprobe v:0, without claiming single-stream media or pixel-equivalence proof. Full977 clock freeze and2364 qualifiedbuild wait for actual fullraw/decode completion.

The isolated target evaluator review requested two bounded fixes:bind original projection-provenance SHA to the census, and validate actual complete frozenNB1 feature/derived schema. Writer is correcting them; original findings remain. It has not read naturalGT or run a full diagnostic.


The isolated target diagnostic evaluator has completed its original freshSol review and bounded correction check:38 independent CPU tests/Ruff passed; the two original provenance/schema defects are preserved and now rejected before output. No naturalGT or actual diagnostic was run. This same-family provisional engineering result prepares the required full population/error/unit/coverage evaluation; it does not supply those results.

Astra NTR1 R2 fixes the first complete paired training API/schedule, X0 analytic adapter, source sampler, checkpoint selection and resource bounds (`NATIVE_TRAINING_CONTRACT_20261104.md`). R1 is retained in docs/aris. R2 corrected an unnecessary22GiB-free/blanket-no-pause condition: planned process cap6GiB/free8GiB, actualmaxN/lease still pending, and only already authorized matching-user low-priority reversible scheduling. STaR/other users stay untouched. The 24h/arm value is an upper budget, not measuredETA. Runtime implementation is underway; no GPUtraining/pilot was launched.


## Full decode closure and NTR1 R2a checkpoint

The original full CPU decode job finished all794 train/183 development videos:695625 frames, rc0 for977, zero row findings or closure discrepancies. All977 raw clock CSVs and1954 empty stderr files are archived; E restoration verified2942 artifacts/30800644 bytes with zero mismatches. PTS and best-effort clocks have no missing/duplicate/reversed ticks;1954 missing DTS remain an explicit diagnostic. Five time bases and observed rates are retained. This closes the existing job; do not decode it again. `FULL_DECODE_COMPLETION_20261104.md` and `evidence/full_decode_completion_20261104.json` contain exact definitions, hashes and limitations, including prior media SHA copied rather than recomputed during decode and original process exit code not separately captured.

The healthy full native pose job remains active (293/977 complete at12:18UTC); immutable raw artifacts, clock overlay, complete qualified NB1/NT1 and target-unit/error/coverage diagnosis still precede training. This data completion is not a scientific performance result.

Astra NTR1 R2a clarifies that another PID gaining1MiB while actual free memory remains10GiB does not stop this job; actual2/5/6GiB bounds and the same arm's cumulative24h budget govern resource exits. Full train/dev maximumN and guarded prediction share the budget. Original R2 stays in Git401d8a5. The original runtime review found four concrete blockers; the author is applying bounded fixes and the original reviewer is checking them. No GPU training or naturalGT evaluation has occurred. CPU MKLDNN validation produced nonfinite gradient failures; explicit CPU-native convolution validation passed with unchanged numerical tolerances. Both traces remain. A newly frozen two-process identical X0 bank has all40 old orbit-grid SHA differences; no unavailable old array comparison is inferred to prove roundoff equivalence.


## NTR1 checked runtime checkpoint

The new native training runtime and analytic X0 adapter have completed the original independent Sol/xhigh review and bounded correction checks. All four original findings and intermediate failing probes remain in `NATIVE_TRAINING_RUNTIME_REVIEW_20261104.md`: actual producer receipt/provenance interoperability, loss-side whole-traversal eligibility, harmless shared-GPU growth, and full train/predict setup/runtime failure accounting. The final CPU suite passed26 tests; original mathematics/sampling/selection and full-population mocked entry probes were separately checked. This is same-family provisional engineering assurance, not scientific acceptance or empirically measured GPU feasibility.

CPU convolution uses the explicitly disclosed native backend after observed MKLDNN nonfinite gradients, without relaxed tolerances. Two fresh X0 processes produced identical40-source/224-view arrays under the recorded new environment; all40 old orbit-grid SHA differences remain. The first16-pass/2-fail owned-suite full log path was reused: only its explicitly labeled excerpt survives, while separate raw gradient/backend failure traces remain. No original full-log preservation is claimed. The first full794/183 canonical/control jobs still require complete raw/qualified inputs, NT1 and isolated full target-unit/error/coverage diagnosis, the evidence-grounded continuation decision, and the exact frozen workload/live lease. No naturalGT evaluation or actual GPU training/prediction occurred in this checkpoint.
