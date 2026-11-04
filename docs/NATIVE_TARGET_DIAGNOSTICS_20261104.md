# Native target diagnostics — 2026-11-04

## SPEC (declared before code)
Target: isolated CPU evaluator for immutable full-population NB1/NT1 artifacts and train/val evaluator-only labels. Inputs: full bbox-only census, projection provenance, source-components manifest, complete NT1/NB1 population manifests, original annotation root, frozen config. Outputs: all identity raw prediction/GT/support/event rows, full and conditional video→person macro proposed NMAE/raw MAE/OBO/bias/ratio, grouped support distributions, optional fixed-seed source-cluster bootstrap and receipt hashes. Base commit: 2458eb2906f39dde42f3fcd3119b2acefa11bcc3. Success: isolated synthetic pytest and Ruff under E-only TEMP/TMP. No natural GT access, production build, model, GPU, producer call, test labels, threshold selection, commit/push, shared manifest/state writes in this task. Four assigned new files only; root owns independent review.

## Acceptance table / ladder
| Rung | Requirement/check | Status |
|---|---|---|
| F0 | CPU CLI help exits0 before feature code | pending |
| F1 | Immutable complete census/readers; invalid/missing artifacts fail whole evaluation | pending |
| F2 | Proposed macro metrics, failure/partial rows and source bootstrap | pending |
| F3 | Explicit endpoint unit diagnostics/support/disagreement preservation | pending |
| F4 | Integrated synthetic CLI fixtures, receipts, Ruff; root fresh review | pending |

## Assumption ledger — ASK=never, EFFORT=balanced, assurance=draft
| ID | Chosen before dependent code | Class | Source |
|---|---|---|---|
| A01 | Production fixed train794/1878 + val183/486, 695625 native frames/1758034 person frames; no cap/fixture CLI override. Internal test entry may inject expected census. | interface | user |
| A02 | Count baseline is G-NT1 only if all traversals geometry and hybrid_mode unavailable. Existing NT1 hybrid intake is unavailable; evaluator rejects hybrid/mixed rather than filter pulses or misname. A separately frozen pure-geometry artifact is required if hybrid later becomes available. | semantic | user/NT1 |
| A03 | Mean over people within each original video, then videos; count>0 GT required or entire run fails. OBO abs error≤1, raw MAE, signed bias, relative signed error and count/GT ratio follow same macro. Named proposed FULL-v1 supplied-track diagnostics, no official/SOTA claim. | semantic | user/metric contract |
| A04 | Period endpoint default unresolved→unit diagnostics unavailable. Only explicit e_minus_s_v1 interprets raw numeric periods as [s,e), length e−s; no +1 and no PeriodAP. Domain 0≤s<e≤N. Pulse coordinate is its NT1 seam/start source edge, not midpoint. Support intervals [e,e+1) only for E=N−1 actual edges; final [N−1,N) unsupported without invented sample. | semantic | user/default |
| A05 | Seconds use NB1 stored map_seconds and positive finite deltas in pts mode only; unknown seconds remain null when none known, counted unknown edges, never zero. Stage duration excludes undefined intervals, reports known subtotal plus unknown edge count. | semantic | default/NB1 |
| A06 | Conditional metrics select certified identities but retain video→person macro on videos with ≥1 certified identity; subgroup denominators and excluded rows explicit. Full metrics retain failed count0+abstain; partial never scaled. | semantic | default/NT1 |
| A07 | Source cluster bootstrap samples archived source components with replacement within split; each selected component contributes all member videos, including repeated copies; recompute macro. Fixed seed20261104, optional replicates0(default), no automatic significance claim. | semantic | default/metric contract |
| A08 | Config explicitly names original annotation count_field and periods_field (no guessing), provenance binds exact annotation SHA and original object→slot mapping; all objects/slots must match. Same-media aliases remain independent original rows with raw label disagreements recorded, components must bind aliases together; production expects four train alias groups. | interface | user/default |
| A09 | Read-only artifact reader validates NPZ schema/invariants/hash, actual census binding, complete NT1/NB1 population ledgers and per-identity binding. Uses the checked NumPy-only NB1 feature loader and clock/mapping validation after A11; no target producer or heavy model imports. Integrity/NaN/missing mapping fails whole run; certificate rejection is a valid zero row. Outputs use a new directory and retain raw JSONL plus every input artifact digest. | interface | default/user |

## Build note
No natural results or target-quality acceptance. Historical X0 NMAE .9915197/OBO0, v15/old80% coverage failure and original negative evidence remain untouched. Full real diagnostic waits for complete artifacts and protocol freeze. Root owns fresh Sol review; same-family/backend unattested means provisional only.

## Implemented input/output interface

`python -B scripts/evaluate_temporac_native_targets.py --config E:/.../frozen_diagnostics.json` is the sole production entry. Exact config keys:

```json
{
  "protocol": "temporac.native-target-diagnostics.v1",
  "inputs": {
    "census": {"path": "E:/.../projection.jsonl", "sha256": "<actual SHA256>"},
    "provenance": {"path": "E:/.../provenance.json", "sha256": "<actual SHA256>"},
    "source_components": {"path": "E:/.../source_components.json", "sha256": "<actual SHA256>"},
    "nb1_manifest": {"path": "E:/.../NB1/validation.json", "sha256": "<actual SHA256>"},
    "nt1_manifest": {"path": "E:/.../NT1/population_ledger.json", "sha256": "<actual SHA256>"}
  },
  "annotation_root": "E:/.../MultiRep",
  "bridge_root": "E:/.../NB1",
  "target_root": "E:/.../NT1",
  "output": "E:/.../new-diagnostic-run",
  "count_field": "<frozen actual original count key>",
  "periods_field": "<frozen actual original periods key>",
  "endpoint_policy": "unresolved",
  "bootstrap_replicates": 0
}
```

There is no CLI census override or cap. Input annotation paths are obtained solely from validated train/val projection paths. Complete artifact/census validation occurs before GT opens. Only original count/period fields named in config are used; no action/category or test-label read. Immutable original JSON bytes must match trusted projector provenance. Mapping must be exact object-set and contiguous complete slots. Source components use the existing audit `[{id,members:[split/composite-name,...]}]`; test membership strings in an existing graph are ignored as nondevelopment metadata and never cause test file access. Same-media aliases must map to the same source cluster. Production requires the four frozen train alias groups; labels/conflicts remain as separate original rows.

New output directory contains `identity_rows.jsonl`, `diagnostics.json`, `receipt.json`. All raw identity rows retain original object/count/periods, count-period disagreement, count errors, pseudo-pulses/traversals and attempt/reason ledgers, support funnel and per-GT-period pulse/support records. Summary includes split full/conditional count metrics, groups for split/source/branch/clock/reason (with per-split views), frame/edge/known-duration distributions, explicit unknown-time edges, event denominator/full/partial/unsupported counts and pulse bins. Bootstrap outputs include cluster order, sampled indices and replicate metrics for exact reproducibility. Receipts bind input manifests, original annotations, every NB1/NT1 artifact, config, evaluator/CLI code and raw output rows; inputs are rehashed after computation. Output-directory overwrite is refused.

NMAE = mean_video mean_person abs(c−g)/g; raw MAE = corresponding abs(c−g); OBO = corresponding 1[abs(c−g)≤1]. Signed bias, relative signed error, and c/g use identical macro weights. Conditional population retains certified identities and reports its smaller video/person denominators. Count0 abstention remains in the full denominator; partial count is never divided by coverage. Source-cluster resampling keeps whole video/identity groups together, permits repeated copies, and recomputes these macros. It does not change the original population metric to source-equal macro.

Under unresolved endpoints, raw periods and mismatches remain available but event-unit diagnostics are unavailable, including outside-pulse rate. Explicit `e_minus_s_v1` alone enables half-open numeric [s,e), pulse at seam/start-edge coordinate, and support-overlap length e−s with no +1. A GT period ending at N can contain a final unsupported [N−1,N) portion because NB1 has N−1 real edges. Endpoint arithmetic is a named proposal, not a claim about official annotation semantics. Empty period lists have zero period denominator and undefined support fractions; nonpositive/nonfinite GT count fails the whole count evaluation. Pseudo-traversals are never supplied as legal PeriodAP output.

## Build record / acceptance evidence

F0: initial minimal CLI `--help` ran exit0 before feature implementation (tool receipt). The stub never emitted result numbers and has been retired. F1–F3 landed as a coupled implementation before the first accumulated check; this is a procedural departure from one green check per coded rung, not evidence those rungs were independently green earlier. No code preceded the SPEC/ledger.

Initial synthetic suite:22 passed in9.67s. Follow-up suite:24 passed in9.63s, exit0; Ruff final exit0/All checks passed. Final stable receipt will be appended below after the small summary-distribution completion. Tests cover unequal people/video weighting; all failed and partially supported counts; GT count-period mismatch; 0/1/≥2 pseudo-pulses per period; boundaries0/N and half-open end exclusion; full/partial/unsupported periods; outside-union pulses; empty period lists/invalid GT counts; deterministic unequal-size source resampling; full synthetic immutable artifact/mapping/receipt pipeline; alias conflicts; unknown versus known duration; missing/changed artifacts, NaN pulse/hybrid targets, mapping errors and incomplete ledger rejection. Production CLI denies fixture census/caps. No Torch/SciPy/producer imports were needed.

| Requirement | Status/evidence |
|---|---|
| Isolated CPU evaluator and full immutable census validation | Passed synthetic engineering; production census rejects fixture population, all artifact/mapping failures nonzero with no output |
| Count/support/event definitions and raw failure preservation | Passed synthetic engineering; unequal-video macro, abstention, partial support and unit counterexamples |
| Complete977/2364 natural diagnostic | Pending complete NB1/NT1/protocol freeze; no natural GT opened here |
| Official count parity, legal period output, target-quality tolerance | Pending; deliberately no official/SOTA/PeriodAP/quality-acceptance claim |
| Fresh silent-assumption review | Root-owned pending; any positive Sol review same-family/backend-unattested provisional |

Read receipt: pinned and installed `research-implement-feature/SKILL.md` both SHA256 `33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8`; pin f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Read full skill and mandatory acceptance-gate/reviewer-backends/reviewer-independence/effort-contract/capture-antipatterns/injection-hygiene plus output-language/versioning/manifest/composition; source/copy SHA equality verified. Requested four-file scope overrides standalone skill auxiliary outputs/manifest writes/commits. Only this owned report consolidates SPEC/ledger/build record; root handles review trace and shared outputs. Initial output-versioning file is new; this record is append-only. No new environments, installs, processes in background, server/GPU/model calls, natural GT/test reads, commits, pushes or shared state/manifest edits.

## Stable final execution receipt

24 passed in10.95s, pytest exit0; Ruff All checks passed, exit0. E-only TEMP/TMP/basetemp, -B/PYTHONDONTWRITEBYTECODE and disabled pytest/Ruff cache. F0–F3 mechanical checks complete; F4 deterministic checks complete, root review still pending. No live stubs; no natural diagnostic result. Full command: E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe -B -m pytest tests/temporac/test_native_target_diagnostics_v1.py -p no:cacheprovider --basetemp E:/ICASSP_TempoRAC/aris-work/native-target-diagnostics-cpu/pytest_stable -q (PYTHONPATH=project/src).

| Artifact | SHA256 |
|---|---|
| E:\ICASSP_TempoRAC\TempoRAC-CVPR\src\pams\temporac\native_target_diagnostics_v1.py | 85cf3392bee7bcf067e675f406f6341916c723180b3dee82201ebcc01102dba5 |
| E:\ICASSP_TempoRAC\TempoRAC-CVPR\scripts\evaluate_temporac_native_targets.py | add1421442fcc5e9ca69cb289f061a8b2f3052f7760b39965d21eb6b0e51a913 |
| E:\ICASSP_TempoRAC\TempoRAC-CVPR\tests\temporac\test_native_target_diagnostics_v1.py | 18a11b0802b9fff6ad69b6053dc6c21b34161a57377c41e9963daef7e455d246 |
| E:\ICASSP_TempoRAC\aris-work\native-target-diagnostics-cpu\pytest_stable.txt | a11995809cb2ba43048de2a4dffe4b69a3fa5ecdebb70c4561ee3047cfbd90f7 |
| E:\ICASSP_TempoRAC\aris-work\native-target-diagnostics-cpu\ruff_stable.txt | a4443afdcfb6d7363adb285762515ccf7cf50473b1a05c20c1a50f6bed4d26b0 |

## Narrow reviewer corrections — declared before dependent changes

Original fresh review `E:/ICASSP_TempoRAC/aris-work/native_target_diagnostics_review_20261104.md` remains unchanged: CHANGES_REQUESTED, F1/P1 and F2/P2. The original synthetic24-pass receipts do not establish complete NB1 intake. Root requested only these two narrow fixes and same-session recheck.

A10 (interface, Source=sweep): require the existing `provenance.projection_sha256` to equal the selected census file SHA before any annotation/GT/metric evaluation. Correctly self-hashed provenance with an incompatible projection digest fails whole evaluation.

A11 (interface, Source=sweep): reuse the checked NumPy-only NB1 `load_native_feature` validator for the exact10-field feature schema and existing dtype/shape/observation/PTS invariants. Validate the complete existing derived array schema, dtypes/shapes, finite numeric payload, run/support domains and clock/mapping consistency in the reader; no target producer invocation or model/heavy import. Synthetic positive fixtures must use a full valid NB1 feature and its actual synthetic CPU-derived arrays (N≥17), with regressions for missing arrays, wrong shape/dtype and observation/clock violations. This corrects intake without changing any statistics or endpoint definitions.


## Narrow correction completion receipt

R1/F1 implemented: `projection_sha256` is now required and compared with the selected census digest before artifact iteration/annotation opening/metric computation. Correctly self-hashed mismatched or missing projection bindings fail the whole evaluation. New regressions instrument label opening and metrics to ensure neither happens.

R2/F2 implemented: feature intake calls existing NumPy-only `load_native_feature` for the exact10-field schema and its original dtype/shape, full-axis identity, observed finite/zero missing payload and PTS time-base invariants. Derived intake checks the exact existing full field set, all dtype/shapes, finite nonclock numerical arrays, actual native clock/mapping equality, observed joint/frame prerequisites, complete executable/short-run domains, exact run/support unions, nonnegative supported edge-length domain and zero unsupported direction payload. This is read-only validation of existing NB1 artifacts, not target production or method/threshold selection. A09's old bridge-import statement has been corrected explicitly by A11. Neither native preprocessing nor target production is invoked by the evaluator.

Positive integration fixture is now N17 with all10 actual feature arrays and complete arrays from actual synthetic NB1 CPU preprocessing. The earlier four-routing-array/six-edge fixture was not a valid NB1 artifact and is no longer presented as complete intake evidence. Original F1/F2 findings and old receipts are preserved in the independent review and prior build record. New parameterized negative cases cover missing/extra feature and derived members, feature shape/dtype/observation/PTS defects, derived shape/dtype/clock/run-domain defects. Malformed artifacts are rehashed with updated metadata so rejection tests exercise schema/domain validation rather than only stale digests. Statistics, endpoint policies and natural-data gates are unchanged.

Necessary final CPU suite: **38 passed in28.30s, exit0**. Ruff: **All checks passed, exit0**. E-only TEMP/TMP, PYTHONPATH, -B/PYTHONDONTWRITEBYTECODE, `-p no:cacheprovider`, E-only `pytest_narrow_final` basetemp and `--no-cache` Ruff were used. Output receipts below replace no historical receipts. The prior24 tests first passed again on the complete N17 fixture in12.08s before adding the narrow14 regressions. No natural GT/test labels, model/GPU/server operations, installs, target producer calls, commits, pushes, shared manifest/state edits or files outside the assigned four were changed. Root owns same-session reviewer F1/F2 recheck, which remains pending and same-family/backend-unattested provisional.

| Narrow stable artifact | SHA256 |
|---|---|
| E:\ICASSP_TempoRAC\TempoRAC-CVPR\src\pams\temporac\native_target_diagnostics_v1.py | 95b8686caa725d3c220e68fb74e13dab7b822f41fe3d3dc1e12c10155ab3fee5 |
| E:\ICASSP_TempoRAC\TempoRAC-CVPR\scripts\evaluate_temporac_native_targets.py | add1421442fcc5e9ca69cb289f061a8b2f3052f7760b39965d21eb6b0e51a913 |
| E:\ICASSP_TempoRAC\TempoRAC-CVPR\tests\temporac\test_native_target_diagnostics_v1.py | a091c84419c78412488780785500ea0c087635d84bb2e29fcc44522471a3e111 |
| E:\ICASSP_TempoRAC\aris-work\native-target-diagnostics-cpu\pytest_narrow_final.txt | bab3106ec11914ef18f45c9710cfb858ced2ae618527aab876656afdeb7d3e49 |
| E:\ICASSP_TempoRAC\aris-work\native-target-diagnostics-cpu\ruff_narrow_final.txt | a4443afdcfb6d7363adb285762515ccf7cf50473b1a05c20c1a50f6bed4d26b0 |
