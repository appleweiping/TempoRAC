# Independent NT1 / LF1 technical review — 2026-11-04

Verdict: **No concrete remaining engineering blocker found for the frozen hybrid-unavailable NT1/LF1 implementation. Same-family semantic assurance remains provisional.** The five Python artifacts were independently read and tested against final NB1 dependency SHA `50aecf64…` and NB1 builder `1822d120…`. This is not target-quality acceptance, full-population completion, training authorization, or an official/SOTA claim.

## Review identity, independence, and scope

- Fresh native reviewer session: `/root/native_targets_lf1_review`. Primary artifacts and contracts were supplied without an inherited executor verdict.
- Requested route/model/effort: `codex / gpt-6.1-sol / xhigh`, following the explicit task and project override. The actual serving model and actual reasoning effort are not independently attested; reported actual values remain null. No cross-family acceptance is claimed.
- `review_independence: same-family`; `acceptance_status: provisional`. Substantive outcome and identity assurance are separate.
- Read E:\AGENTS.md; project AGENTS.md, ARIS_LOCK.md, docs/aris/WORKFLOW.md, docs/CVPR_RESUME_20261104.md and current run JSON. No narrower AGENTS.md was found in src/scripts/tests/docs.
- Fully read the installed and pinned-original research-review SKILL at pin `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`, plus reviewer-backends, reviewer-routing, review-tracing, output-composition, integration-contract, output-versioning, output-manifest, and output-language references. Skill defaults were overridden only by the explicit routine Sol/xhigh assignment and project policy.
- Fully read the Astra NT1/LF1 contract SHA `c4e489c6bd1f26117efa2bce2f993b9bc9c8fbc5ead9abe71d00415d5ed8aae0` and NB1 design SHA `f0622e664ed6ed53f123cfc7f7151a71f04b2d15f347e84114e548ddbaa9487a`.
- Read all six new artifacts, original v12/v13/v15/v15_landmarks/v9 runtime paths, phase closure and template kernels, CertificateTrack and v6/v7 certificate implementation, and the relevant NB1 census/preprocess/support code. Existing certificate functions are reused directly, rather than replaced by a new approximation.

## Acceptance evidence

| Requirement | Status | Independent evidence and limits |
|---|---|---|
| Exact native-N kernel/certifier reuse; original branch order | Passed engineering check | Full source inspection; direct positive certified v12/v15 parity tests passed, comparing phase/reconstruction/pulse/chi/masks/bounds/seams/canonical phase/winding/reasons. Fallback tests preserve v12 -> unavailable v9 -> v15 order. |
| Nonvacuous certified-success mapping | Passed engineering check | Both direct parity tests require `direct.certified and result.certified`. An extra independent offset probe certified 5 traversals on run [102,215), exactly mapped to [102,122), [122,142), [142,162), [162,182), [182,202); pulse/chi exactly equal the original local certificate and zero outside. This is not parity by matching rejection. |
| N216/N900, final N-1 endpoint, complete full-E payload | Passed engineering check | Independent 18-test rerun; N201 terminal fixture accepts right endpoint 200 while retaining exactly 200 edges and no fictitious sample N. Validation enforces domain/dtype, ordered disjoint bounds, exact traversal-mask union, one start pulse, finite nonnegative normalized chi, and zero payload outside. |
| Per-run complete rejection; no zero-motion or unsupported traversal salvage | Passed engineering check | Additional probe: one response-unsupported edge in an otherwise executable fixture rejected both geometry branches for that complete run. Repeated exact-zero masked arcs rejected the complete run. A separate successful run retained K=4 when the second run was stationary/rejected. Attempt reasons and unavailable hybrid remain separate. |
| PTS gap, short runs, no across-gap cycle or fake negative | Passed engineering check | Real NB1 preprocessing fixture with a >=5-median-step clock gap splits into two runs; target does not cover gap edge99 or cross it. Short/static fixtures keep all full-E zeros and false target masks. Inspection confirms local sample slice [a:b+1] with edge slice [a:b] and offset a. |
| Honest geometry/hybrid/mixed provenance | Passed for frozen unavailable configuration | Geometry has null teacher SHA. Public production hybrid intake rejects even plausible caller SHA strings, and CLI requires hybrid_mode=unavailable. v9 numerical mock-phase checks and mixed-source bookkeeping use visibly fixture-only injected intake; they do not establish production hybrid availability. |
| Target artifacts, input binding, resume and population preservation | Passed interface/fixture check; full population pending | NPZ content digest, sidecar metadata, policy/input binding, census opaque-key/slot/N check, complete expected census gate, atomic writes, missing-input return2, changed raw-binding resume refusal and rejected identity retention verified in source/tests. No production target output was built. |
| G-NT1 direct baseline | Passed interface check; quality pending | With production hybrid unavailable, both geometry branches execute in original priority and count=sum(pulse)=successful traversal count; rejected output is count0+abstain, with no coverage scaling. geometry_count refuses a hybrid-containing target. True repetition unit is explicitly unvalidated. |
| LF1 is one BCE criterion applied to 3 branches + fused readout | Passed engineering check | Source matches class-balanced formula, detached target/chi/mask/pi, positive finite W+/W- with no epsilon fallback, 1/3 branch mean + fused, equal traversal mean. Old objective remains unchanged and LF0-LF1 equals exactly the two legacy margin contributions in the test. |
| Correct LF1 values, responsibility weights, gradients | Passed engineering check | Existing exact hand-BCE/gradient checks passed. Independent nonuniform edge-wise pi hand calculation matched 0.7578704159212301; capacity_control omitting pi matched 0.7479006604747293. Heads/fused have gradients, unsupported edge gradients are zero; targets/chi/cue/pi do not receive loss gradients. Missing class mass and responsibility zero/underflow fail explicitly. |
| Traversal/identity/source/family reductions | Passed interface check; formal manifests pending | Split-run traversal averaging is equal within 1e-15. Independent unequal-identity-count source example produced source mean 7, matching ((2+6)/2+10)/2 rather than global identity mean 6. SourceIdentityCycle retains empty components in the census while cycling eligible sources/identities deterministically; family mixture is exactly 0.5/0.5. Actual X0/native sampling manifests and training schedule remain pending. |
| Final dependency integration | Passed engineering check | Independent copy asserted the five Python SHAs plus NB1 50aecf64… and builder1822d120… before 18 tests. Executor final receipt separately records 18 passed/25.52s against the same dependency hashes; document now discloses that supersession of the earlier f27 copy. |
| Full977/2364 diagnostics, true target units/quality and formal training | Pending / outside this scoped implementation review | No raw production build or GT read occurred. Full source-component/branch/clock/failure distributions and isolated train/dev error/unit/event-coverage evaluation are still required by the contract. Successful synthetic certificates do not establish one true repetition or adequate coverage. |

## Independent execution

Existing environment only: `/home/lilei/PoseRAC_Experimental_Code_20260828/.venv/bin/python`, Python3.10.12, torch2.0.1+cu118, NumPy1.24.4, SciPy1.15.3. Reused the existing isolated `vendor_pure`; no install or shared environment mutation. CUDA_VISIBLE_DEVICES was empty and observed torch device_count was 0. OMP/BLAS/MKL/NUMEXPR thread counts were 1. All reviewer source copies, temporary files, cache variables and fixture writes stayed beneath `/mnt/4tssd/lilei/TempoRAC/work/native-target-review`; E-only task transport/evidence stayed under aris-work/tmp. No GPU/model/checkpoint/natural GT/test-label load, training launch, process signal, code edit, commit, push, or healthy-job change was performed.

Command: existing CPU interpreter `-B -m pytest -v -p no:cacheprovider --basetemp <review-scratch>/tmp/pytest_independent tests/temporac/test_native_target_nt1.py tests/temporac/test_objective_lf1.py`, with PYTHONDONTWRITEBYTECODE=1 and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.

Result: **18 passed in 24.73s**. Extra independent strict-run, offset-success, nonuniform-pi and source-reduction probes all passed.

Local evidence:

- `E:/ICASSP_TempoRAC/aris-work/tmp/nt1-review-independent-pytest.txt`
- `E:/ICASSP_TempoRAC/aris-work/tmp/nt1-review-independent-probes.json`
- Reproducible probe source: `E:/ICASSP_TempoRAC/aris-work/tmp/nt1-review-probe.py`
- Remote evidence: `work/native-target-review/independent_pytest.txt`, `independent_probes.json`, `independent_probes.py`; source snapshot is under that directory's `source/`.
- Executor dependency receipt checked separately: `tmp/native-target-lf1-cpu-receipt-final-nb1.{txt,json}`. It supports the named integration rerun only.

A first attempt to use the E audit Python for SSH transport failed because that isolated interpreter lacks paramiko; the existing system Python transport then worked. No package was installed to address this. The failure had no scientific execution side effect.

## Findings, action boundary, and allowed claims

No actionable code defect remains identified in this scope after direct inspection, the independent test execution, and the added adversarial probes. No source fix is requested. The current document accurately distinguishes the earlier f27 dependency copy from its final 50aecf64 rerun and keeps quality/training gates pending.

Allowed: the hybrid-unavailable producer and LF1 numeric/reduction interfaces have passed the stated CPU fixture checks, original certified-success parity, and fresh same-family provisional technical review. A completed required input census can now support the one full label-free target build and its later contract-mandated isolated evaluation.

Not established: full target population completeness, natural pseudo-event unit validity, natural count error, event/time/identity coverage adequacy, component efficacy, official period output legality, full training completion, SOTA, independent-test status, or cross-family acceptance. Production hybrid availability is intentionally not implemented/accepted in this delivery. These are disclosed outstanding contract stages, not reasons to repeat generic source review before the authorized next concrete stage.

## Artifact identities

The accompanying `.meta.json` records complete SHA256 values for the six reviewed artifacts, two contracts, final NB1 dependencies, raw reviewer test/probe evidence and fully read pinned review references. The fixed report path is owned by this reviewer; parent owns shared project state, traces integration, fixes, and commit.
