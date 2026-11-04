# NB1 independent native learner bridge review — 2026-11-04

Status: first review complete; three concrete findings sent to the parent; author fixes and bounded recheck pending. This report preserves the original findings. It is engineering review only, not research, model, target-quality, training, or publication acceptance.

## Review identity, independence, and source receipt

- Fresh native review session: `/root/native_bridge_review`; parent `/root`.
- Requested reviewer: `gpt-6.1-sol / xhigh`. Actual backend model/effort: not independently attested; requested values are not identity evidence.
- Executor/reviewer relation: OpenAI same-family; positive semantic outcome can be **provisional** only. Current substantive verdict: **changes required**.
- Primary design read in full: `E:/ICASSP_TempoRAC/aris-work/native_learner_bridge_design_20261104.md`, SHA256 `f0622e664ed6ed53f123cfc7f7151a71f04b2d15f347e84114e548ddbaa9487a`.
- Later target/loss clarification read in full: `E:/ICASSP_TempoRAC/aris-work/native_target_loss_contract_20261104.md`, SHA256 `c4e489c6bd1f26117efa2bce2f993b9bc9c8fbc5ead9abe71d00415d5ed8aae0`.
- Read `E:/AGENTS.md`, project AGENTS.md, ARIS_LOCK.md, docs/aris/WORKFLOW.md, docs/CVPR_RESUME_20261104.md, and current `.aris/runs/temporac-cvpr-20261104.json`.
- Full pinned `research-review/SKILL.md` read at commit `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`; source and installed-copy SHA256 both `17505d5982c82e821124de898b0beb7de4a98ecc4c1cd173b8687b74ace9d4c5`.
- Full-read shared references and source/copy hash equality: reviewer-backends `e0aea16315faaec6ea468272aed7bf9d4a1b1d5261302a8ba357a9dce7bd9c33`; reviewer-routing `b0ce923454caf49e22742ae4de0bd9847e6f0a4d3c9ca95a1b48288cc1ea6619`; review-tracing `fd1e524d1d78dd2dfb6cc51e404a303f98aabc40849bc16149676c8904814fa4`; integration-contract `0d6ec981e51b85da6d3c24d8fef4bbeb64c27fef87a57e1129413aff59f2ddd1`; output-composition `79a72c7ea02102c460ae3ad55853c21a4054b9df93f377c29dd34555e6d3c3a7`; output-versioning `de8e7ac23069de6c5d0c482d4cef117dd5a543e00b57e2dc6c62560a457290c5`; output-manifest `e767083e0ef97a7adf7a4e8a574090067a59c0655d5aca422a3747cd32b6afde`; output-language `0f1447579bd7a2195fd4a074be70ae6345b5c368f1a70d327e89748bf5a378f9`.

I did not open the six implementation files until the parent supplied STABLE hashes. Each hash below was independently reproduced before review. Full reads covered all six files plus directly used legacy preprocessing, timebase, cue, quadrature, decoder, response interface, and raw validation/load interfaces. No other new producer/loss implementation was reviewed.

| Stable reviewed file | SHA256 |
|---|---|
| src/pams/temporac/native_bridge_v1.py | f27d2335fc32c22771403eed5a06f8ebc874fa8e714e410d9e2f36548454a8e5 |
| src/pams/temporac/native_time_cue_v1.py | 97d4c537813757ca03f83e3c695abf63418a9465fcba481ec1d21b159252a880 |
| scripts/build_temporac_native_learner_bridge.py | a475efe632c4c25065dffcc45cd230308d41dc89b3425e5cc1628d51cdfbbb0a |
| tests/temporac/test_native_bridge_v1.py | 387c4182377c1a83fcf237605aeed225f1cddbc7b93694fc08ad4d698b1e0814 |
| tests/temporac/test_native_time_cue_v1.py | d4d9750fb1fb33f181dbb471f5bea9cc8df6cbab03e66dfe1c380b5d53bcaf16 |
| docs/NATIVE_LEARNER_BRIDGE_20261104.md | c3870b73b2a10dd3beac5405dee7391db0994169159edefa900d60e51e21ae05 |

## Acceptance checklist prepared before code review

| Requirement | First-review status and evidence |
|---|---|
| Arbitrary N, full source population, feature-only raw projection, no320 resampling | Pass engineering: N216/N900 fixtures and strict ten-field numeric loader; no annotation/count/period/density/boundary input; source i=0..N-1; production census is fixed and cap-free. Actual full977/2364 build remains pending. |
| Honest observation/derived/time masks and q/tau/PTS mapping | Pass engineering: raw observations distinct from frontend/source/derived masks; source-only fallback only when all PTS absent; Python integer subtraction, video-level tau, separate units/segments, candidate interpolation never fed to model. |
| >=8 joints/hip and >=32 common geometry coordinates; zero-motion exact extension | Pass engineering: legacy normalization reused; paired geometry makes31 unreachable, realizable30/32 tested; zero lengths retained, direction false and static cue uniform. |
| Runs, lag, cached response windows, NOLA do not cross gaps | Pass engineering for response path: native runs reset lag and window schedule; supported denominator >=1e-3; unsupported numerator/denominator exactly0. Target traversal support has finding NB1-R1. |
| Float-clock NUDFT formulas/configuration, detached routing, conservative grid fallback | Pass engineering: inherited48 periods5..128 and five interventions; native q=i exact quadrature/spectra/gates; real available-reference nonuniform-gate equivalence additionally checked; float Δq=.5 remains accepted; clock-grid guard explicit. All cue arithmetic is NumPy detached data. |
| Once-per-identity decode, retained partial/abstain, coordinates/endpoints | Pass engineering: one full component call, empty once when unsupported, full response/support retained, count=-1 abstention; source component ranges distinct from periods; boundaryN seconds unavailable without verified terminal. |
| Target mask domain/provenance and W± | Partial: W+=sum(m*chi*pulse), W-=sum(m*chi*(1-pulse)) are present. Whole traversal support and mixed per-traversal provenance require NB1-R1/R2 fixes. |
| Full census, failure retention, atomic writes and resumable validation | Partial: fixed full census, aliases and slots retained; two-file/sidecar atomic protocol and binding checks exist. Truncated feature ZIP bypasses failure ledger in NB1-R3. |
| Frozen old APIs/extractor unchanged; no training launcher | Pass within local evidence: git has no tracked modifications, only new untracked files; extractor SHA remains d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1. CLI exposes plan/build/validate only. No server/process controls were used. |

## Original findings (preserved)

### NB1-R1 — whole target traversal remains eligible after losing internal response support

Priority: blocking semantic interface defect under the NT1 clarification. Initial file `native_bridge_v1.py:852–865`, especially `ok = wp > 0 and wn > 0` at863.

`native_supervision` intersects response and target support, then checks only surviving positive and negative weight. It can therefore silently train on a truncated certified traversal. NT1 page2 requires any response support loss to make the entire traversal ineligible, with a reason; it cannot retain eligibility merely because both class weights survive.

Reproduction used only synthetic CPU data:

```python
r = preprocess_native(feature(80))
valid = np.zeros(79, bool); valid[10:60] = True
pulse = np.zeros(79); pulse[10] = 1
chi = np.zeros(79); chi[10:60] = 1/50
t = NativeTargetV1(valid, pulse, chi, np.array([[10,60]], np.int64),
                   'full-chain geometry', 'fixture', 'a'*64)
ok = r.feature.frame_source_ok.copy(); ok[40] = False
gap = preprocess_native(replace(r.feature, frame_source_ok=ok))
print(np.flatnonzero(valid & ~gap.response_edge_support).tolist())
print(native_supervision(gap,t)['traversal_eligible'])
```

Actual output:

```text
[39, 40]
(True,)
```

Required narrow fix: preserve `supervised` as the intersection but require complete response support over each declared traversal before declaring eligibility; retain the traversal and explicit reason. Keep the correct W± formulas.

### NB1-R2 — minimal target API cannot retain per-traversal mixed provenance

Priority: blocking scoped API requirement, with an explicit scope distinction. Initial file `native_bridge_v1.py:789–848`; `branch_provenance: str`, three-value scalar validation, scalar JSON loader.

The original NB1 minimum target placeholder had one scalar source, and the independent NT1 envelope was intended to retain mixed per-run/per-traversal provenance. That distinction is real; this finding does not claim NT1 was already integrated. However, the parent explicitly included provenance preservation by the target mask API in the review scope. At the reviewed stable source, a per-traversal tuple is rejected as `unknown branch provenance`, and the only accepted form loses mixed source identity.

Actual output from `replace(t, branch_provenance=('full-chain geometry',))`:

```text
ValueError: unknown branch provenance
```

Parent accepted the scoped narrow correction: retain and validate per-traversal provenance plus a truthful mixed summary; allow a single-source scalar to broadcast for existing callers. This must not change producer branch selection or claim NT1 integration. Author coordination with the separate NT1 writer remains outside this review's implementation ownership.

### NB1-R3 — truncated resumed feature NPZ escapes the full census failure ledger

Priority: blocking resume/failure handling defect. Initial builder `:220–231` calls `load_native_feature` at224 before checking its saved digest at226. Its catch at266 handles ValueError/OSError/KeyError, while a truncated ZIP raises `zipfile.BadZipFile`.

The reviewer mutated only the reviewer-owned synthetic fixture feature NPZ created by the baseline test, at `aris-work/tmp/nb1-review-baseline/test_cli_real_artifacts_resume0/output/identities/..._0.npz`, to bytes `b'PK\x03\x04broken'`. Its original sidecar digest was left untouched. Calling `execute(cfg,'validate',expected=fixture_census)` produced:

```text
corrupt NPZ escapes census ledger BadZipFile File is not a zip file
```

It raised before returning2 or refreshing the per-identity failure ledger. The production/raw files and source were not edited. Required narrow fix: check saved feature/derived digests before loading feature contents, so ordinary byte corruption becomes the existing handled hash mismatch and retains the complete ledger. No new hash scheme or generic security gate is proposed.

## Independently executed verification

All executions used `E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe -B`, explicit TEMP/TMP `E:/ICASSP_TempoRAC/aris-work/tmp`, PYTHONPATH=repo/src, PYTHONDONTWRITEBYTECODE=1. No install, model loading/forward, GPU, natural benchmark labels, server command, process signal, code edit, commit, or push was performed.

Baseline command:

```text
python -B -m pytest -p no:cacheprovider tests/temporac/test_native_bridge_v1.py tests/temporac/test_native_time_cue_v1.py tests/temporac/test_preprocess_geometry.py tests/temporac/test_quadrature_cue.py tests/temporac/test_timebase_audit.py tests/temporac/test_decode_audit.py tests/temporac/test_feature_io_firewall.py tests/temporac/test_native_pose_cache.py --basetemp E:/ICASSP_TempoRAC/aris-work/tmp/nb1-review-baseline -q
```

Raw output:

```text
........................................................................ [ 96%]
...                                                                      [100%]
75 passed in 31.49s
```

Independent Ruff command on the five scoped Python files used the isolated environment's `ruff.exe check --no-cache`:

```text
All checks passed!
```

Additional numerical checks, independent of authored tests:

```text
quadrature exact True
routing modes 7 [False]
subunit clock accepted 0.5 True 7
nonuniform routing reference True 16 16 gate deviation 0.2547141891820734
all interventions exact True
```

The first fixture's reference was unavailable, so an additional all66-coordinate sine fixture with600 samples was used; it had16 reliable windows, an available reference, and genuinely nonuniform gates. Native vs old local/global/uniform/blocked/shuffled gate arrays were all bit-exact. This avoids relying only on a vacuous uniform-gate equivalence.

## Boundaries and pending work

The actual full977/2364 bridge build/census has not run in this review and cannot be inferred from synthetic fixtures. Raw/decode completion, target production/quality, full development diagnosis, metrics/failure policy, losses/budget/selection, formal training, and scientific acceptance remain separate. PeriodAP remains unavailable. Healthy native extractor/decoder were not contacted or modified. The current review does not reintroduce obsolete A005 security gates or re-audit unrelated routes.

Author fixes and a bounded, same-session recheck of the three findings will be appended below. The original findings above will not be deleted.

## Final same-session narrow recheck — R1/R2/R3

This final section supersedes the initial pending status at the beginning of the report. The original findings and first-review evidence above remain preserved. The parent supplied a second STABLE snapshot and required a bounded recheck only of the three findings and impacted existing regressions. I followed that boundary; no new overlay/scientific redesign or unrelated audit was reviewed.

### Final independently verified SHA256 snapshot

| File | Final SHA256 |
|---|---|
| src/pams/temporac/native_bridge_v1.py | 50aecf64e85667570b656553789e255763b755d3f252d82b3c21f7585071c4f3 |
| src/pams/temporac/native_time_cue_v1.py | 97d4c537813757ca03f83e3c695abf63418a9465fcba481ec1d21b159252a880 |
| scripts/build_temporac_native_learner_bridge.py | 1822d120a2d1f6e00fb031dae7e535b75cde28a9b99fd18fd0afea1b5460806a |
| tests/temporac/test_native_bridge_v1.py | 1ed1647397279792b52eb3b24db301b38f20821e71e2bb980f5025f3f54a4e8d |
| tests/temporac/test_native_time_cue_v1.py | d4d9750fb1fb33f181dbb471f5bea9cc8df6cbab03e66dfe1c380b5d53bcaf16 |
| docs/NATIVE_LEARNER_BRIDGE_20261104.md | 25e998690a517d9b76aee97f56519650fd6062c5fe88b0927d0249a5e49eb1c2 |

The final changed-file hashes were checked again after execution and still matched. The extractor SHA remained `d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1`.

| Original finding | Recheck evidence | Final engineering status |
|---|---|---|
| R1 complete traversal support | At native_bridge_v1.py:883–892, any lost response edge yields response_support_gap and clears the complete traversal mask. Re-running the original sole-positive-pulse N80/frame40 fixture now yields eligibility(False,), reason(response_support_gap,), supervised0 and both weight sums0. Regression additionally preserves a separate healthy traversal and verifies exact raw chi*pulse / chi*(1-pulse) weights there. | Resolved for reviewed interface |
| R2 mixed per-traversal provenance | Optional traversal_provenance is validated one-to-one against bounds and frozen as a tuple. Legacy single-source scalar broadcasts. Mixed summary requires actual distinct certified sources; contradictory summaries/missing lengths/rejected sources reject. Actual mixed NPZ+JSON loader roundtrip and old scalar loader remain covered. Producer branch selection is unchanged and NT1 remains a separate envelope. | Resolved for scoped target mask API |
| R3 corrupt resumed NPZ | Builder now checks both saved learner hashes before either feature/derived archive is loaded. Four existing regression variants cover truncatedZIP/appended corruption × feature/derived. Both build/resume and validate return2 with complete two-identity ledger, first artifact_failure(hash mismatch), second complete. Reading the generated JSON confirmed this for all four variants. | Resolved for reproduced failure |

Independent narrow command:

```text
python -B -m pytest -p no:cacheprovider tests/temporac/test_native_bridge_v1.py tests/temporac/test_native_time_cue_v1.py --basetemp E:/ICASSP_TempoRAC/aris-work/tmp/nb1-review-narrow -q
ruff.exe check --no-cache src/pams/temporac/native_bridge_v1.py src/pams/temporac/native_time_cue_v1.py scripts/build_temporac_native_learner_bridge.py tests/temporac/test_native_bridge_v1.py tests/temporac/test_native_time_cue_v1.py
```

Actual output (both exit0):

```text
..............................                                           [100%]
30 passed in 23.88s
All checks passed!
```

Additional direct recheck output:

```text
R1 original reproduced fixture now (False,) ('response_support_gap',) supervised 0 weights 0.0 0.0
R2 scalar broadcast ('full-chain geometry',)
R2 mixed retained mixed ('full-chain geometry', 'stable-pair hybrid')
```

All four current R3 generated artifact pairs reported:

```text
validate=population_failures, identities=2, statuses=artifact_failure,complete,
resume=population_failures, error=ValueError: learner artifact hash mismatch
```

No unrelated old tests were re-run after these fixes. The initial75-test baseline is historical evidence on the first snapshot; the final30-test result is the impacted NB1 regression evidence on the final snapshot. The author's documented first narrow fixture failure (duplicate media_path, later corrected to a distinct path) was read and retained in its owned document; it was not treated as a production change or omitted from this review's interpretation.

### Final verdict and checklist disposition

**Substantive verdict: no remaining concrete engineering blocker in the reviewed six-file NB1 scope after the three bounded corrections.** The first-review checklist rows for target mask/provenance and atomic-resume failure handling are now engineering-pass for the reproduced contracts; the other engineering rows retain their first-review evidence. This is **same-family provisional** semantic review, with requested Sol/xhigh and actual backend identity/effort still unattested. It is not cross-family acceptance or scientific approval.

Review completed: true. Review scope complete: true for the six-file NB1 scope, the supplied NB1/NT1 interface clarification, and the three narrow corrections. The parent owns external trace/meta integration. Reviewer changed no implementation, legacy file, extractor, producer/loss source, runtime configuration, job, server, model, or environment; only this report and isolated synthetic test outputs were written.

Actual full977/2364 bridge build/validation, raw/decode completion, NT1 integration/target quality, full development diagnosis, metric/failure protocol, loss/budget/selection, formal training, and scientific acceptance remain pending or separately scoped. Later Astra overlays and the separately implemented NT1/LF1 files are outside this verdict. PeriodAP remains unavailable.
