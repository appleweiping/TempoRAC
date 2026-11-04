FINAL: Review complete; no concrete remaining engineering blocker in final stable727becc2 six-file NTR1/R2a scope. Independent26CPU tests passed7.21s. Same-family provisional, actual backend unattested. Actual inputs/diagnostics/lease/GPU training/prediction/scientific quality remain separately pending. The original findings and every bounded recheck are preserved below.

# Independent NTR1 runtime technical review — 2026-11-04

Status: prerequisites and checklist prepared; waiting for the author's STABLE six-file dispatch. The six WIP implementation artifacts have not been opened. No substantive implementation verdict is assigned yet.

## Identity and boundaries

Fresh reviewer session `/root/native_training_runtime_review`; requested `codex / gpt-6.1-sol / xhigh`. Actual serving model/effort are not independently attested and remain null. OpenAI same-family semantic assurance is provisional. Review is engineering only: random CPU initialization, synthetic tensors and analytic X0, no trained weights, natural GT, GPU or resource-lease action, production build, code edit, commit or push. Parent owns shared manifests/traces/state.

The active NTR1 R2 contract was read in full and independently hashed to `65503a1ddb1d4cd83b75de4e3c2e47d8763dfdb4e722810f15eadd567bb969fd`. The original R1 is historical. Pinned research-review SKILL was read in full; installed and exact pin f76ec8d6b77792e2242cca049e08d0cbb7d54e19 source hashes both equal `17505d5982c82e821124de898b0beb7de4a98ecc4c1cd173b8687b74ace9d4c5`.

## Checklist prepared before implementation access

| Requirement | Required evidence | Current status |
|---|---|---|
| Complete inputs and population | Frozen794 train/183 dev,977 videos/2364 identities; aliases retained and same-source component vote; raw/decode/clock/NB1/NT1/diagnosis/plan bindings | Pending implementation review |
| X0 analytic adapter | 40 sources,24/8/8 partition,224 train/tune views; immutable bank provenance, two-process bytes; no heldout selection; ten pulse/terminal/chi/guard/trim/5-edge checks; no F11 | Pending implementation review |
| Paired sampling and initialization | CPU fan-in/LayerNorm initial bytes equal across arms; domain-separated PCG64 cycles and shared full draw ledger; complete per-identity exposure | Pending implementation review |
| Fixed schedule | T=500*ceil(max(10000,C*M)/500),C>0; warmup ceil(.05T),old cosine parity at10000; candidates every500, no early stop | Pending implementation review |
| LF1 and full cue support | Natural cue/NOLA use full NB1 response support independent of target mask; canonical vs capacity probability fusion; eight sequential graphs, exact0.5/7 and0.5 weights; one clip and AdamW | Pending implementation review |
| Vector fusion | Float64 D/pi precompute, positive denominator threshold and unsupported zeros; reference float64 forward <=1e-12,float32 <=2e-7,non-saturated gradient relative <=1e-5 | Pending implementation review |
| Selection and resume | Full8x7 X0 tune objective; finite lowest/tie earlier; completeT/candidates only; recovery separate; optimizer/RNG/sampler/schedule/input/code/config/init binding continuity | Pending implementation review |
| Full population prediction | Actual all-population outputs, no target restriction, NB1 float64 reference NOLA -> oncefloat32 -> exactly one identity decode; partial/abstain retained | Pending implementation review |
| Resource R2 and launch gates | Shared GPU permitted; processcap6GiB,allocator5GiB,startfree8GiB,runheadroom2GiB; max_N envelope and disk checks; missing/unknown lease rejects; only owned authorized lowpriority reversible scheduling, no STaR/otherusers | Pending source/CPU-only review |
| Scope and scientific claims | No auto-train after validation, no realGPU probes or fullbuild; required diagnostic continuation decision precedes training; no quality/SOTA/fullmatrix upgrade | Pending implementation review |

## Primary prerequisites read

E:/AGENTS.md; project AGENTS.md,ARIS_LOCK.md,docs/aris/WORKFLOW.md,docs/CVPR_RESUME_20261104.md,current .aris/runs JSON; full NB1,NT1/LF1 andclock overlay contracts; checkedNB1/NT1/clock reviews including final bounded rechecks. No nested AGENTS.md was found in src/scripts/tests/docs. Research-review references read in full: reviewer-backends,reviewer-routing,review-tracing,output-composition,integration-contract,reviewer-independence,review-scope-limits,output-versioning,output-manifest,output-language. User/project explicit Sol/xhigh assignment overrides the skill's default Astra/ultra.

## First stable review — original findings (preserved)

The first stable six-file SHA table was independently reproduced. The initial independent CPU suite passed18/18 in6.45s on existing Python3.10.12/Torch2.0.1, NumPy1.24.4/SciPy1.15.3, CUDA visibility empty and threads1, own native-training-review staging/cache only. No natural GT/model weights/GPU/lease/job operation occurred. A first source snapshot assertion stopped before tests because the author's remote source staging omitted the documentation; the document hash was independently checked on E, and the code-only source assertions then passed. This was a reviewer preparation issue, not an implementation finding.

### R1 — real producer receipts cannot qualify; independent hashes do not enforce existing provenance links

Blocking interface defect: `_artifact_receipts` recognizes only generic complete/complete_population/passed/frozen. Actual overlay emits frozen_complete_census; qualified NB1 emits validated_population; raw population/progress and isolated diagnostics have no generic top-level status. Independent synthetic actual-status probe returns clock_overlay_not_complete and nb1_not_complete despite success states. Read actual producer source confirms the others. Consume each existing producer schema/terminal-population fields and its existing bindings. In particular the diagnostic report already exposes receipt.inputs_sha256 for frozen census/source-components/NB1/NT1 and individual artifacts; NB1 binds its overlay/census. Supplied independent file hashes do not prove the diagnostics describe the currently loaded targets or component mapping. Reuse these existing links, including four same-source train alias groups; no new hashing framework is requested.

### R2 — one target traversal without raw class mass removes the whole identity

Blocking LF1 eligibility defect in `_load_inputs`: any traversal with W+ orW- zero sets eligible=False for the entire identity. Independent synthetic one-good/one-zero-positive-mass fixture yielded eligibleFalse,traversals2,reason no_complete_positive_and_negative_LF1_traversal. Contract/root clarification: retain the complete identity whenever at least one complete traversal has legal raw positive/negative mass. Only LF1 bounds/mask filter whole ineligible traversals; retain original target/pulse/G-NT1, full cue/NOLA/prediction and all population rows, and record reasons. Soft responsibility underflow remains an implementation error, never a filtering route.

### R3 — harmless shared allocation growth forces resource_incomplete

Blocking execution policy mismatch: `_resource_check` rejects any growth of another PID, even if actual free remains far above the2GiB running headroom. CPU fake telemetry with another PID growing1MiB while actual free=10GiB reproduced resource_incomplete. R2 permits sharing and requires exit for actual free<2GiB or own process>6GiB; it does not prohibit harmless changes in another allocation. Keep telemetry and the actual bounds; do not touch other processes.

### R4 — predict bypasses live R2 resource protections

Blocking resource path defect by direct source inspection: `_predict` allocates oncuda:1 after `_require_train` (frozen inventory only), without live resource checks, the5GiB allocator cap or running process/free/headroom/wall accounting. `build_training_plan` also uses only train max_N for the envelope; dev identities may be larger. Route prediction through the same scoped admission/monitoring/bounded accounting and include full dev_N for its memory shape. CPU mocked checks suffice; no GPU probe requested.

All findings were sent directly to author/root. Parent authorized immediate bounded fixes of R1/R2 while resource checks continued. Original stable code is retained in the reviewer's own server snapshot. No reviewer implementation edit was made.

## Assumption sweep — first stable snapshot

A001 new schema is declared, but actual producer-format interoperability in A002 is stale until R1 is fixed. A003 correctly declares vectorized floating operation order and probability fusion; the implemented tolerances passed. A004 is stale for prediction (R4) and the undeclared any-growth abort contradicts the stated R2 policy (R3). The observed CPU MKLDNN-disabled backend is transparently disclosed in BUILD_NOTE but is not yet an assumption-ledger row; record it as an interface/runtime choice with the observed failure evidence (no method redesign). A005 review separation and A006 independent X0 bank are accurately disclosed. Same-family sweep can flag these gaps but cannot independently acquit semantics.

Initial raw evidence: E:/ICASSP_TempoRAC/aris-work/tmp/ntr1-review-independent-pytest.txt; ntr1-review-independent-probes.json; reproducible probe ntr1-review-probe.py. Author's actual two-process X0 byte check and old-grid differences were read:40 sources/224 views raw digest c595a3b4e89fe795949e4df3738a8742b75fa313790037db35181eb183de86fd; accepted attempts/coefficients unchanged, all40 old orbit-grid hashes differ in the actual new environment. These are disclosed compatibility differences, not altered seeds or hidden regeneration.

Final narrow recheck is pending. Initial substantive verdict: changes required. No full training or target-quality acceptance is assigned.

## First bounded R2a fix recheck — b25780f3 snapshot

The six review-fixed hashes were independently reproduced. Primary contract was fully re-read and matched R2a `47d739d3e7d8922fdeb70169d9013e9d4d8f429d675900462d5fc3927d3e3078`. R1 producer-specific formats/existing diagnostics bindings, R2 whole-traversal LF1 eligibility and R3 harmless other-allocation growth have source-level corrections. Prediction now calls live admitted_device and monitoring. Before the final independent run, narrow R4/R3 residuals were returned to the author: R2a separately requires max_train_identity_N/max_dev_identity_N; GPU setup/checkpoint loading must follow cumulative-budget precheck and be accounted; produced prediction NPZs need a bound per-identity progress record; running free<2GiB must not produce inconsistent `failed` vs `resource_incomplete` budget status. Author acknowledged these bounded corrections. No new generic review or scientific change was proposed. The b257 snapshot is not called final-pass; final recheck awaits the next stable hashes.

## Second bounded R2a recheck — 6dc9c05e snapshot

Independent source assertions passed; all24 owned/impacted CPU tests passed in6.71s. Added independent CPU-only probes confirmed a synthetic full2364 population/person-frame plan uses max_train_identity_N744,max_dev_identity_N2000,max_N2000,T10000, and interrupted mocked prediction retains1 bound complete record with output SHA/count/status and cumulative budget12.001423s rather than resetting12 priorseconds. All GPU/model/checkpoint operations in the prediction entry probe were patched with explicit CPU fakes; no actual train/predict/GPU or trained weights were used. CUDA_VISIBLE_DEVICES empty, actual device_count0, threads1.

One bounded original-R4 residual remained: a resource failure during admitted_device/model/checkpoint setup still precedes the running budget marker and try/failure handler. Mocking admitted_device to raise resource_incomplete left the prior budget at statuscomplete/wall12, no prediction_failure.json. Train has the same source-level gap. Author was sent this concrete probe and asked to encompass setup in the existing budget/failure scope, write running beforeGPU admission, and preserve prior recovery/progress on failure. This is not a new scientific/runtime redesign. Final semantic verdict remains pending until this narrow correction is checked.

## Final bounded recheck and verdict — stable727becc2

**Final substantive engineering verdict: no concrete remaining blocker found within the six-file NTR1 implementation and R1–R4 bounded correction scope.** Review completed=true; review_scope_complete=true. Same-family semantic assurance remains provisional; requested Sol/xhigh, actual serving backend/model/effort not independently attested. This is not GPU validation, full input/diagnostic completion, target-quality approval, trained performance, official qualification, SOTA or submission acceptance.

The final outer accounted-stage scope checks the prior cumulative arm budget before any GPU admission, saves setup_running first, and encompasses admission/model/checkpoint setup as well as the existing training/prediction bodies. A setup exception advances the same arm budget, records resource_incomplete and retains recovery/prediction progress. Unaccounted prior setup/running states refuse automatic continuation. The reproduced setup-failure counterexample is resolved.

### Final independently verified artifact SHA256

| File | SHA256 |
|---|---|
| src/pams/temporac/native_training_runtime_v1.py | 727becc2f13de9346522215de36be3efbb55c9d7ef27e14ad156c856fb86aaaa |
| src/pams/temporac/native_x0_adapter_v1.py | b61aa7d86864b8de1718b2ed6f8a88f0f6b2047e927b69ff4341f8e70934d2db |
| scripts/train_temporac_native_response.py | b092eff0427b97ebb5f34a0072e3c1afd54a42f82c3ec7f297cf66d8ebee2edf |
| tests/temporac/test_native_training_runtime_v1.py | 430fe5e6e247c87e48c043ce87a93c893f1ff6901ce663e3a82a95ce720a8684 |
| tests/temporac/test_native_x0_adapter_v1.py | 413b073c2ca5241ca91a41ca5c79bfb197062262c4b261e61b79004096d37495 |
| docs/NATIVE_TRAINING_RUNTIME_20261104.md | ddb8d704bf31ec4da5408dcb1b3d6cdae0753feea70c77b3ba72f49cc03457ac |

Active primary contract is R2a SHA47d739d3e7d8922fdeb70169d9013e9d4d8f429d675900462d5fc3927d3e3078; initial65503a1d R2 and originalR1 remain historical. Initial stable, b257 and6dc review evidence above is preserved rather than recast as final verification.

### Final checklist disposition

| Requirement / original finding | Status | Final evidence and limits |
|---|---|---|
| Complete794/183 and2364 population, alias/source votes, input/diagnostic freezes (R1) | Passed interface/fixture check; actual full qualification pending | Exact production census is cap-free; actual producer-specific terminal schemas/populations; immutable loaded artifacts/census/source-component/NB1/NT1 diagnostic links and NB1 overlay linkage checked. Existing source_membership enforces four same-media train alias groups; continuation binds the complete diagnostic report. Stale component binding regression rejects. Actual full production artifacts were not loaded in this review. |
| X0 source/units/bank/guards | Passed stated analytic checks | Both actual isolated40-source/224-view banks independently validated and compared raw hashes;24/8/8 partition, ten pulses, chi normalization, terminal exclusion,5-edge synthetic cycle, trim/guards and synthetic_tick units tested. Heldout never enters training/selection. Old40 grid-byte differences remain disclosed, accepted attempts/coefficient bytes unchanged; no F11 or trained model. |
| Paired initialization/sampler/source exposure | Passed CPU checks | All three fixed seeds yield matched CPU initial state bytes across arms; PCG64 purpose/cycle separation has no variant; component and X0 complete cycles, per-component identity exposure and sampler resume checked. Complete plan ledger and source/identity eligibility records remain visible. |
| T/LR/checkpoint selection/exact resume | Passed CPU checks | T formula, C=0 refusal, all10000 old LR points, every500 candidate set, finite/tie/NaN/missing/hash refusal and optimizer/sampler continuation passed. Full8x7 X0 LF1 tune selection is source-inspected; no natural dev/GT selection. Actual full training/candidate production is pending. |
| Full cue/NOLA support and LF1 eligibility (R2) | Passed engineering checks | Cue/cache built from full NB1 response support before target use. One good/one raw-class-mass-invalid traversal retains legal complete LF1 bounds only, original target/mask unchanged; zero-good identity excluded with traversal reasons. Soft-pi/head underflow still faults. No inference restriction or G-NT1 mutation. |
| Vector NOLA /8 graphs /AdamW | Passed CPU parity checks | Float64 forward<=1e-12, finalfloat32<=2e-7 and nonsaturated gradient-relative<=1e-5 reference checks pass on33/216/900 lengths and gaps. Eight fresh weighted backward graphs match the same total loss, independent globalL2/manualclip and oneAdamW update. Capacity uses probability mean and omits responsibility weights. |
| Full population reference prediction | Passed interface/CPU fixture checks; actual full predict pending | Model sees full support; reference float64 NB1 NOLA then onefloat32/oneidentity decode, including abstention. Complete486/183 output enforced. No target argument on prediction function. Interrupted fake-device execution preserves bound completed records and NPZ hashes; partial is never declared complete. |
| Shared-resource R2a policy (R3) | Passed CPU-mocked policy checks | Other PID+1MiB/free10GiB continues and is logged. Actual free<2GiB, ownNVML>6GiB and allocatorreserved>5GiB reject; startup8GiB/capboundary/unknownlease checks pass. No actual nvidia-smi/GPU/resource/lease action was taken by reviewer. |
| Predict resource/budget/setup/recovery (R4) | Resolved in source and CPU-mocked regressions | Shared admitted_device before model placement sets GPU1/5GiB cap; complete train/dev max fields/envelope. Same-arm budget spans train/tune/predict, exhausted arm never reaches admission. Setup_running precedes admission, failure advances original budget and preserves progress. Source-inspected monitoring/partial records; empiricalGPU memory/time remain pending. |
| Assumption sweep A002/A004/A009 | Resolved within reviewed scope; same-family provisional | A002 now consumes actual producers and existing diagnostic links; A004 behavior/documentation includes live predict/shared limits and cumulative accounting; A009 explicitly discloses the post-failure CPU MKLDNN choice/negative evidence. A007/A008 name provenance and traversal interface clarification. No remaining undeclared consequential decision was identified in this bounded re-sweep; semantic_undeclared0. This does not independently acquit scientific meaning. |
| Scientific/full campaign acceptance | Not achieved / outside this review | Full977 raw/clock/NB1/NT1 and isolated complete diagnostics/continuation/plan/lease readiness, actual GPU training/prediction, empirical efficiency, true pseudo-unit quality, six datasets/seven frameworks, three-seed/component/mechanism/hyperparameter/SOTA results and final paper remain separate obligations. |

### Independent final execution

Existing server CPU interpreter only: /home/lilei/PoseRAC_Experimental_Code_20260828/.venv/bin/python -B, Python3.10.12,Torch2.0.1+cu118,NumPy1.24.4,SciPy1.15.3. CUDA_VISIBLE_DEVICES='', actual device_count0; OMP/BLAS/MKL/NUMEXPR threads1; no auto plugins/cacheprovider/bytecode; all source snapshots/temp/cache/fixture files under reviewer-owned /mnt/4tssd/lilei/TempoRAC/work/native-training-review. Existing vendor_pure reused; no installation/shared-environment change.

Final command: `python -B -m pytest -v -p no:cacheprovider --basetemp <own-root>/tmp/pytest_final_setup tests/temporac/test_native_training_runtime_v1.py tests/temporac/test_native_x0_adapter_v1.py` against asserted stable727 source snapshot. **26 passed in7.21s, exit0.** This includes both train/predict setup failure regressions; no actual training/prediction job or CUDA execution.

Independent final probe raw result: setup failure now writes resource_incomplete plus failure record and cumulative12.000401s; interrupted fake-device prediction retains1 complete bound prediction record/SHA and cumulative12.001674s; full-size synthetic plan retains2364 identities/person-frame sum1758034, T10000, max_train744/max_dev2000/max_N2000. Actual device_count remains0. The prior6dc probe showing unchanged12s/statuscomplete/no failure record is retained separately as negative evidence.

Raw E-side evidence:

- tmp/ntr1-review-final-26-tests.txt
- tmp/ntr1-review-final-probes.json; reproducible tmp/ntr1-review-final-setup-probes.py
- tmp/ntr1-review-independent-banks.json
- tmp/ntr1-review-independent-pytest.txt and ntr1-review-independent-probes.json (initial snapshot)
- tmp/ntr1-review-6dc-setup-failure.json (preserved counterexample)

Remote original/fixed source snapshots and logs are retained under own native-training-review root. Optional first-snapshot Ruff diagnostic reported import/semicolon/zip-strict style notices; no automatic edits or functional lint-pass claim was made, and style-only work was outside the bounded final recheck.

No reviewer implementation edit, natural GT/test-label read, trained checkpoint/model weight load, real GPU/train/predict job, production build, lease/process signal, shared environment change, commit/push or cleanup occurred. Parent owns shared manifest/tracing/state integration and the later full-readiness decision. Final raw report, metadata, completed receipt and final assumption-sweep JSON are saved for the parent heartbeat.

