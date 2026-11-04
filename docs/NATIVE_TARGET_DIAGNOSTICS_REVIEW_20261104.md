# Native target diagnostic evaluator — independent review, 2026-11-04

Verdict: **CHANGES_REQUESTED** for two concrete input-contract defects below. The supplied synthetic suite passes, but it does not establish the promised complete, consistent artifact intake. No natural-data diagnostic, scientific target-quality PASS, official metric freeze, PeriodAP qualification, or training acceptance is granted.

Requested reviewer: `gpt-6.1-sol / xhigh`; reviewer session: `/root/target_diagnostics_review`, fresh primary-artifact review. The actual backend model/effort is not independently attested (`reported_model=null`, `reported_effort=null`, `backend_attested=false`). `review_independence=same-family`, `acceptance_status=provisional`. The requested identity is not substituted for observed identity. Root owns the shared ARIS trace and any project-state update.

## Acceptance criteria and observed evidence

| Requirement | Status | Evidence / limit |
|---|---|---|
| Review exact stable four-file set and primary contracts | Pass | All four SHA256 values match the request; the complete primary NT1/LF1 four-page contract is SHA256 `c4e489c6bd1f26117efa2bce2f993b9bc9c8fbc5ead9abe71d00415d5ed8aae0`. Read the metric contract, NB1 design/schema, existing NB1 and qualified-NB1 builders, NT1 schema/validator and builder. |
| Full 977-video / 2364-identity population, no cap or silent exclusion | Partial | Production census hard-codes train794/1878, val183/486, frame totals695625/1758034, complete slots, unique rows and complete NB1/NT1 ledgers. Counts and population hashes are checked. F1 and F2 below leave original-provenance consistency and full NB1 schema intake incomplete. |
| Proposed video→person macro count diagnostics | Pass, synthetic only | `macro_metrics` computes people within each video, then videos for NMAE, raw MAE, standard OBO≤1, signed and relative signed error, count/GT ratio. Unequal-people fixture distinguishes this from person pooling. No rounding or coverage inflation. |
| Failure / partial / conditional behavior | Pass, synthetic only | Rejected target has count0+abstain and remains in full denominator; partial count is not rescaled. Certified-only subgroup has explicit smaller video/person denominators, and an empty conditional subgroup is undefined. |
| Isolated pure-geometry baseline | Pass for current availability freeze | Geometry-only traversals and `hybrid_mode=unavailable` are required. Existing NT1 branch order skips unavailable v9 and tries v12 then v15. Hybrid/mixed intake is rejected rather than relabeled pure geometry. A future hybrid run needs a separate frozen G artifact. |
| Support funnel, grouped distributions and unknown time | Pass for checked numerical logic; full intake incomplete | Observed→response→target→training masks, identity/edge/source-frame denominators, known seconds subtotals and unknown-time edge counts are retained. Groups include split/source/branch/clock/reason and per-split views. Source-frame-only time remains null. Full NB1 feature validation is missing (F2). |
| Period/pulse unit diagnostics and endpoints | Pass, synthetic only | Explicit `e_minus_s_v1` uses raw [s,e), no+1, source-edge seam/start pulse; 0/1/≥2 bins, full/partial/unsupported support and outside-union pulses are preserved. `[N−1,N)` remains unsupported even with all N−1 actual edges. `unresolved` returns unavailable and does not guess official endpoints or legal PeriodAP output. |
| Count/period inconsistency and exact-media aliases | Pass for preservation; provenance binding incomplete | Raw GT counts/periods and disagreement flags are retained. Four train alias groups are required in production; aliases must share source component, remain separate rows, and label conflicts are exposed. F1 can admit an incompatible provenance document. |
| Source-cluster bootstrap | Pass, synthetic only | Fixed seed20261104; sorted cluster list records the cluster population, selected indices and replicate metrics. Whole source groups are sampled with replacement, repeated video copies stay separate, and video/person macro is recomputed. Unequal cluster-size hand expectation is tested. No significance claim is emitted. |
| GT isolation and claims | Pass, static and synthetic | Only train/val paths from census are opened after artifact validation. No producer, model, SciPy, Torch or test-data imports/calls in evaluator. Labels do not flow back to thresholds. Outputs explicitly say official count not certified, PeriodAP unavailable and quality tolerance not frozen. |
| Authorized engineering checks | Pass | Independent E-only Python3.11 synthetic pytest24/24; Ruff clean; CLI help exit0. No natural labels, server/GPU, package installation, producer call, code mutation or commit. |
| Complete real target-quality diagnosis | Pending | Requires complete qualified NB1/NT1 population and frozen diagnostic config; this review did not read natural GT or run production scoring. |

## Concrete findings

### F1 — P1: verify the existing projector provenance→census binding before GT access

At `src/pams/temporac/native_target_diagnostics_v1.py:422–426`, the reader accepts the provenance schema, complete media keys and record count, but never compares its existing `projection_sha256` with the actual census digest. The trusted projector writes this binding in `scripts/build_temporac_native_pose_cache.py:863–868`. Checking independently frozen input bytes does not establish that those two inputs belong together.

Reproduction uses only the supplied synthetic fixture. Add `projection_sha256='0'*64` to its provenance, freeze that changed provenance's own digest in config, and call `run(config, _expected=fixture_expected)`. It emits both identity rows and metrics although the actual census SHA is `24b9b568f17c93c9a466fb6315e250111da080f4ab6ef4728f407e1055b79242`. A separately valid original mapping/annotation provenance from a different projection can therefore be paired with this feature census. This defeats the intended source binding without any runtime file mutation.

Required narrow fix: require the already-defined provenance projection digest to equal the selected census digest before opening annotation files. Make the fixture use the actual full provenance field, and reject a correctly self-hashed provenance that binds a different census. No new hashing scheme is requested.

### F2 — P2: validate the frozen NB1 numerical schema rather than four routing fields

At `native_target_diagnostics_v1.py:145–157`, feature intake validates only `source_length`, `slot`, `opaque_key` and `source_frame_index`. It neither checks the exact NB1 feature array set nor the frozen dtypes, shapes and observation/PTS invariants. The production NB1 schema in `native_bridge_v1.py:49–107` requires ten fields, including `xyconf_raw`, `joint_observed`, `frame_source_ok`, `pts`, `pts_present`, and `time_base`.

Reproduction: the supplied `fixture_bundle` itself writes only those four routing arrays. The same successful run in F1 accepts it, even though all six additional mandatory arrays are absent. Its derived NPZ is likewise only a selected subset of the actual `derived_arrays` schema. Thus the synthetic “full immutable reader” pipeline currently exercises an invalid NB1 artifact and can report a valid support/clock funnel without validating the corresponding feature schema. This is an explicit existing input invariant, not a hypothetical security threat.

Required narrow fix: validate the frozen feature schema and relevant complete derived schema/dtypes/domains locally without importing a producer/model. Bind clock/support interpretation to valid NB1 data. Update the synthetic bundle to be a valid NB1-shaped artifact and include a missing-member/schema mismatch rejection. The existing positive tests should then exercise the real input interface.

## Independent checks and reproducibility

Environment: `E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe`, CPython3.11.2, NumPy1.26.4, pytest8.4.2. `TEMP=TMP=E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu`, `PYTHONPATH=E:/ICASSP_TempoRAC/TempoRAC-CVPR/src`, `PYTHONDONTWRITEBYTECODE=1`. Python invoked with `-B`; pytest cache disabled, Ruff `--no-cache`. No background process was created.

```text
python.exe -B -m pytest tests/temporac/test_native_target_diagnostics_v1.py -p no:cacheprovider --basetemp E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/pytest_independent_review_20261104 -q
24 passed in 10.09s; exit0

ruff.exe check --no-cache src/pams/temporac/native_target_diagnostics_v1.py scripts/evaluate_temporac_native_targets.py tests/temporac/test_native_target_diagnostics_v1.py
All checks passed; exit0

python.exe -B scripts/evaluate_temporac_native_targets.py --help
exit0
```

Raw evidence:

- `E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/pytest_independent_review_20261104.txt`
- `E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/ruff_independent_review_20261104.txt`
- `E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/independent_handchecks_20261104.txt`
- Synthetic fixture/output directory: `E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/independent_handchecks_20261104/projection_binding/`

The handcheck constructed the fixture, changed only the synthetic provenance as described, updated that input's config SHA, and ran the evaluator's internal fixture entry. It did not weaken production expected counts or add a production CLI cap. All generated files remained on E:.

## Reviewed SHA256 and read receipt

| File | SHA256 |
|---|---|
| src/pams/temporac/native_target_diagnostics_v1.py | `85cf3392bee7bcf067e675f406f6341916c723180b3dee82201ebcc01102dba5` |
| scripts/evaluate_temporac_native_targets.py | `add1421442fcc5e9ca69cb289f061a8b2f3052f7760b39965d21eb6b0e51a913` |
| tests/temporac/test_native_target_diagnostics_v1.py | `18a11b0802b9fff6ad69b6053dc6c21b34161a57377c41e9963daef7e455d246` |
| docs/NATIVE_TARGET_DIAGNOSTICS_20261104.md | `916a685472d60b81578b1b2a2053eb77748cc8ceffaabfaa4008b370980e6a4d` |

Read E:/AGENTS.md and project AGENTS, ARIS_LOCK, WORKFLOW, CVPR_RESUME and current run state. Full-read pinned `research-review/SKILL.md` and mandatory routing/backends/tracing/composition references plus integration-contract, reviewer-independence, output-versioning/manifest/language. Pinned and installed skill SHA256 are equal: `17505d5982c82e821124de898b0beb7de4a98ecc4c1cd173b8687b74ace9d4c5`, ARIS pin `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`. This task's requested raw-report path and root-owned shared trace/state override generic standalone/shared-output instructions. No shared manifest or project state was edited.

After author revisions, continue this same reviewer session with a narrow recheck of F1/F2, their affected fixtures and the exact changed artifact hashes. Avoid another generic audit. Historical failures and all natural quality/coverage obligations remain unchanged and pending.

## Same-session narrow recheck — F1/F2 closure, 2026-11-04

Final scoped verdict: **F1 CLOSED; F2 CLOSED; no remaining concrete engineering blocker in the requested F1/F2 recheck and affected regressions.** The initial CHANGES_REQUESTED findings and their successful pre-fix reproductions above remain historical evidence. This positive code-review conclusion is **same-family provisional**, backend model/effort unattested, requested Sol/xhigh, in the same `/root/target_diagnostics_review` session. It is not a scientific quality PASS or official metric/main freeze. Full natural977/2364 diagnosis remains pending complete qualified inputs and protocol freeze.

The recheck covered only the two revised intake paths, their N17 synthetic positive fixture, negative cases and affected regression suite. It did not restart a general audit or inspect natural labels.

| Finding / criterion | Recheck status and concrete evidence |
|---|---|
| F1 existing projector→census binding | Closed. Evaluator line476 now requires `provenance.get('projection_sha256') == digest(census_path)` before identity iteration or the annotation path/read loop. Missing and correctly self-hashed mismatched projection bindings are tested with GT-read and metric-call guards. The original wrong-binding handcheck now raises `original provenance projection/census SHA mismatch`; no output directory is created. |
| F2 exact frozen NB1 feature schema | Closed. `load_identity` invokes existing `load_native_feature`, which enforces the exact10-array set and the frozen dtypes/shapes, original identity axis, observed finite/zero missing payload and valid PTS rational base. Replaying the original four-routing-array artifact, with its own correct new hash and target input binding, now raises `feature-only schema member mismatch`; no output directory is created. |
| F2 complete derived schema and domains | Closed for the requested invariant. `validate_derived` checks the existing full field set and dtypes/shapes; finite nonclock values; clock mode and every source mapping array against feature-derived native clock/mapping; usable/observed prerequisites; valid executable/short-run domains and exact mask unions; edge-length and direction payload domains. It calls only existing feature-only clock/mapping helpers, not native preprocessing or a target producer. |
| Valid positive interface fixture | Pass. Integration bundle now constructs a full10-field N17 `NativeFeatureV1` and actual complete derived arrays from synthetic NB1 CPU preprocessing. All16 response edges are supported; target's six supported edges remain distinct. The old incomplete positive fixture is explicitly retired. No natural pose/model or target generation is used. |
| Affected regression and isolation | Independent38/38 tests pass, including14 added binding/schema/domain rejection cases; rehashed malformed inputs exercise the validator rather than only stale digests. Ruff clean. Independent handcheck confirms no Torch, SciPy or NT1 target-producer module imported. |

Reviewed revised SHA256:

| Artifact | SHA256 |
|---|---|
| src/pams/temporac/native_target_diagnostics_v1.py | `95b8686caa725d3c220e68fb74e13dab7b822f41fe3d3dc1e12c10155ab3fee5` |
| scripts/evaluate_temporac_native_targets.py (unchanged) | `add1421442fcc5e9ca69cb289f061a8b2f3052f7760b39965d21eb6b0e51a913` |
| tests/temporac/test_native_target_diagnostics_v1.py | `a091c84419c78412488780785500ea0c087635d84bb2e29fcc44522471a3e111` |
| docs/NATIVE_TARGET_DIAGNOSTICS_20261104.md | `d310f770433d324d2d19623daf1fac1b29fb242d0cb47f63eb54382860bbbdfe` |

Independent rerun, same existing E-only CPython3.11.2 / NumPy1.26.4 / pytest8.4.2 environment, E-only TEMP/TMP/PYTHONPATH, `-B`, `PYTHONDONTWRITEBYTECODE=1`, disabled pytest and Ruff caches:

```text
python.exe -B -m pytest tests/temporac/test_native_target_diagnostics_v1.py -p no:cacheprovider --basetemp E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/pytest_independent_recheck_f1f2_20261104 -q
38 passed in 19.76s; exit0

ruff.exe check --no-cache src/pams/temporac/native_target_diagnostics_v1.py scripts/evaluate_temporac_native_targets.py tests/temporac/test_native_target_diagnostics_v1.py
All checks passed; exit0
```

Raw recheck receipts:

- `E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/pytest_independent_recheck_f1f2_20261104.txt`
- `E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/ruff_independent_recheck_f1f2_20261104.txt`
- `E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/independent_recheck_f1f2_20261104.txt`

The independent handcheck built only synthetic valid N17 bundles, replayed each original defect separately with self-consistent hashes, observed rejection before outputs, and retained the generated E-only scratch evidence. No natural GT/test access, production build, target producer, model/server/GPU operation, install, scientific code edit, commit, push or shared state/manifest update occurred in this recheck. Only this raw review report was appended by the reviewer.
