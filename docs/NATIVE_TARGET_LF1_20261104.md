# Native target NT1 / LF1 implementation — 2026-11-04

## SPEC (pre-code)
Implement contract c4e489c6bd1f26117efa2bce2f993b9bc9c8fbc5ead9abe71d00415d5ed8aae0 on base cb2fe0f55774e5d2d819ea8ef0070f0c383f813c. Inputs are NB1 full-N numeric identities with verified provenance, never labels or models. Outputs are temporac.native-target.nt1 NPZ/JSON, full-population ledger and G-NT1 label-free counts; differentiable torch LF1 is one balanced-BCE criterion applied to branch/fused readouts. CPU-only CLI plan/build/validate/diagnose-inputs. Success: scoped synthetic pytest, direct numerical parity and CPU autograd evidence, Ruff. Full-cache production build, natural GT evaluation, target-quality acceptance, training and deployment excluded. Only six named new implementation/test files and this document are owned. Parent owns fresh Sol review; semantic assurance remains provisional.

## ASSUMPTIONS (ASK=never; before dependent code)
| ID | Choice | Class | Source |
|---|---|---|---|
| A001 | Fixed per-run v12_geometry -> unavailable v9_hybrid -> v15_geometry; reuse original kernel/certifier, preserve complete-run rejection and strict positive masked arc. | semantic | NT1 contract |
| A002 | First build hybrid_mode=unavailable; future compatible artifact intake remains pending; current public producer rejects available hybrid, never loads a teacher or invents its SHA. | semantic | NT1 contract |
| A003 | CLI accepts bridge_root,census,output,hybrid_mode; production population frozen through NB1 census validator; missing NB1 artifacts blocks any target build. Fixture API may explicitly supply expected census. | interface | default |
| A004 | NT1 provenance input SHA is actual NB1 feature NPZ SHA, raw/bridge bindings come from verified NB1 sidecar; config/availability/producer hashes are actual bytes/canonical JSON. Adapter returns narrow numeric LF1 supervision dictionary; mixed sources stay in outer NT1. | interface | default |
| A005 | NT1 diagnostic is label-free only. GT count/error/unit/period evaluator is separate and deferred; no fake acceptance of true units. G-NT1 count sums certified geometry traversal starts and never scales partial coverage. | semantic | NT1 contract |
| A006 | LF1 targets, responsibility and weights detached; missing support rejects whole traversal, positive/negative weight underflow raises implementation error; exact traversal, identity/source/family mean interface. | semantic | NT1 contract |
| A007 | No manifest/state/commit modifications; this document composes SPEC/ASSUMPTIONS/BUILD_NOTE per parent directive. Tests are engineering evidence only. | interface | parent |

## BUILD_NOTE
Balanced / draft / five MUST rungs. Pinned research-implement-feature fully read (pin f76ec8d6b77792e2242cca049e08d0cbb7d54e19), mandatory output-versioning/output-manifest/output-language/output-composition and acceptance/reviewer/effort references read. Installed stage source is not edited.

| Rung | Acceptance command | Status |
|---|---|---|
| F0 entry to plan artifact | Python -B CLI plan fixture | passed; E-only plan artifact |
| F1 native numerical target and kernel parity | scoped test_native_target_nt1.py | passed synthetic CPU |
| F2 full payload/provenance/CLI resume | accumulated target tests | passed fixture only |
| F3 LF1 hand calculation/autograd/reduction | scoped test_objective_lf1.py on CPU torch | passed CPU torch |
| F4 accumulated tests + Ruff + parent fresh review | scoped suite; parent review | 18 tests + Ruff passed; fresh review pending |

## Run record
Pre-code spec and assumption ledger written before implementation. No full-cache build or natural labels accessed. Existing extractor and all old scientific modules untouched.

## Deferred / blockers
Complete977/2364 target production and label-free census require completed NB1 artifacts. Full target unit/quality diagnostics require isolated evaluator. Formal sampler manifest, full training budget, scientific contribution and semantic acceptance remain pending. No training launcher is added.

A008 (semantic; Source=parent/Astra contract): production hybrid intake is explicitly unavailable until actual existing phase/checkpoint artifact validation is frozen. Public produce_native_target rejects hybrid_mode=available even if caller supplies plausible SHA strings. Internal numerical_branch mock phase is only synthetic parity evidence; it does not certify production availability. This delivery fully implements the frozen unavailable production configuration; future compatible intake remains pending.


## Engineering receipt / acceptance (11:02 UTC)

Scoped round3: **18 passed in 20.70s**, Python 3.10.12, torch 2.0.1+cu118, SciPy 1.15.3; CUDA_VISIBLE_DEVICES empty and observed torch CUDA device_count=0. All thread counts fixed 1; TMP/HF/TORCH/XDG caches below `/mnt/4tssd/lilei/TempoRAC/work/native-target-implementation`. Local Ruff `check --no-cache` passed all five Python files. No install/shared environment change, model/checkpoint load, GPU task, natural GT/test label, raw-cache build, training, commit or push occurred.

Raw CPU receipt: `E:/ICASSP_TempoRAC/aris-work/tmp/native-target-lf1-cpu-receipt-round3.txt`; detailed SHA/package receipt same stem `.json`. Remote raw receipt: `work/native-target-implementation/cpu_receipt_3.txt`. Existing local audit Python lacked SciPy; existing server CPU torch environment lacked pytest. Existing pytest 8.4.2/pluggy/iniconfig/packaging/pygments were copied into isolated vendor; exceptiongroup 1.3.1 and tomli 2.3.0 wheels were downloaded to E-only scratch and extracted there, never installed. Package/file/archive SHA evidence is in the JSON. A first tomli Windows wheel carried an inert extension and pure fallback; a pure `py3-none-any` wheel and separate pure extra archive now exist for the final runner; no extension was loaded by Linux. Historical failed readiness attempts are preserved.

| Requirement | Evidence / status |
|---|---|
| Original v12/v15 numerical reuse and exact field parity, no teacher fiction | passed: direct original CertificateTrack/certifier array parity tests |
| N216/N900, multiple runs, unavailable hybrid, v15 fallback | passed synthetic tests; availability is logged separately from rejection |
| Strict stationary/zero arc, no traversal across response/PTS gaps | passed synthetic and real NB1 preprocessing fixture |
| Full E payload, exact traversal union, one start pulse, normalized chi, endpoint N-1 | passed validation/mutation/terminal tests |
| Public hybrid availability cannot be asserted by plausible SHA strings | passed explicit rejection; production compatible phase intake pending |
| Mixed provenance bookkeeping | passed fixture-only injected phase intake; no production hybrid artifact produced |
| Atomic targets, input hash/tamper rejection, CLI missing-cache gate/full fixture resume | passed synthetic CPU; full977/2364 population not built |
| LF1 one BCE criterion, exact hand value/gradient, head/fused gradient, no target/cue-weight gradient | passed CPU autograd and LF0-margin difference tests |
| Traversal/source/family weight interfaces | passed deterministic cycle, unit-mean tolerance1e-15 and 0.5/0.5 tests; actual training manifests pending |
| True target quality/coverage/count error/period unit legality | pending isolated full-population evaluator; no scientific claim inferred |
| Fresh selected Sol review and silent-assumption sweep | parent-owned pending; same-family positive result can only be provisional |

The first CPU round had 15 passed and one overly strict bit-equality assertion on different floating summation orders. The criterion did not change; a 1e-15 tolerance checks identical unit weights. Final tests additionally bind actual NB1 opaque identity to its census key and retain failed input/short-run reasons. The tested NB1 source copy SHA was f27d2335fc32c22771403eed5a06f8ebc874fa8e714e410d9e2f36548454a8e5; final NB1 review fixes must be matched before declaring current-endpoint integration complete. This is engineering evidence and a complete frozen hybrid-unavailable implementation, not target-quality or formal experiment acceptance. No live numerical stub remains. Full source-component joined distributions and all label-aware error/unit diagnostics belong to the later complete census/evaluator stage.


## Final NB1 dependency integration narrow rerun

After parent explicitly resumed the dependency gate, only the isolated server scratch NB1 module and builder were synchronized: `native_bridge_v1.py` SHA256 `50aecf64e85667570b656553789e255763b755d3f252d82b3c21f7585071c4f3`; `build_temporac_native_learner_bridge.py` SHA256 `1822d120a2d1f6e00fb031dae7e535b75cde28a9b99fd18fd0afea1b5460806a`; unchanged cue SHA256 `97d4c537813757ca03f83e3c695abf63418a9465fcba481ec1d21b159252a880`. Runner asserted actual dependency hashes before execution. Original 18 scoped tests passed again in **25.52s** using pure-Python `vendor_pure`, Python3.10.12/torch2.0.1+cu118/SciPy1.15.3, CUDA visible count0, torch/OMP/BLAS/MKL/NUMEXPR threads1, all caches/TMP explicitly below project scratch. NT1/LF1/CLI/test source hashes stayed unchanged.

Raw final receipt: `E:/ICASSP_TempoRAC/aris-work/tmp/native-target-lf1-cpu-receipt-final-nb1.txt`, structured binding/environment receipt same stem `.json`; remote `work/native-target-implementation/cpu_receipt_final_nb1.txt`. This supersedes the earlier f27 dependency-copy limitation for these 18 CPU engineering tests only. Fresh Sol code review remains parent-owned; full population production, natural target quality/unit diagnostics and formal training remain pending. No feature/source/test change, model loading, labels, GPU, shared environment mutation or production cache build occurred in this rerun.
