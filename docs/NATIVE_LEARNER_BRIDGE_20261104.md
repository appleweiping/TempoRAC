# Native learner bridge NB1 — 2026-11-04

## SPEC (before code)
Target: implement the provisional NB1 contract (design SHA256 f0622e664ed6ed53f123cfc7f7151a71f04b2d15f347e84114e548ddbaa9487a) in independent native modules. Inputs: validated native raw numeric arrays and complete census; outputs: feature-only `temporac.native-learner.v1` NPZ plus provenance JSON, full-length support/mapping/predictions, independent `temporac.native-target.v1` mask interface. Base commit: cb2fe0f55774e5d2d819ea8ef0070f0c383f813c. Success: E-only Python -B scoped pytest without cache; CLI plan/build/validate CPU fixtures and complete production census enforcement. Excludes actual partial-cache build, model forward, target producer, loss changes, training, GPU/server mutation, commit/push. Existing APIs and active extractor remain unchanged. Root owns fresh Sol review.

## ASSUMPTIONS (ASK=never; recorded before dependent code)

| ID | Gap / choice | Class | Source |
|---|---|---|---|
| A001 | NB1 full-N rows; zero-motion edges allowed; confidence >.20, >=8 joints plus hip, >=32 geometry coordinates; short runs <17 samples retained but not executable. | semantic | user / NB1 |
| A002 | Raw failures and L/D mismatch project all fixed slots to unusable full annotation-length features, with no guessed decode alignment. | semantic | user / NB1 |
| A003 | CLI config requires raw_root, census (trusted bbox-only projection JSONL), output. Production census fixed train794/1878 + val183/486. Test functions may pass explicit fixture census; CLI cannot override it. | interface | default |
| A004 | Numeric feature schema includes opaque key bytes, slot, N, i, raw xyconf, observed/source support, integer PTS/present and rational base. Status and hashes stay in JSON sidecar. Strict member whitelist excludes annotations/bbox/targets. | interface | user / NB1 |
| A005 | Clock anchor first present PTS; video tau uses all positive present-adjacent differences before identity masks. Missing/nonpositive clocks split segments; float q never enters old integer APIs. No PTS gives source_frame_only. | semantic | user / NB1 |
| A006 | Preserve old normalization and channel construction exactly, copying only geometry routine to allow precise zero length. Native NumPy NOLA mirrors existing NumPy algorithm because existing nola imports torch unconditionally; no torch needed here. | interface | default |
| A007 | Full unsupported prediction retains full response/support separately, calls existing component decoder on empty arrays exactly once, count=-1. Partial counts are observed response components only; PeriodAP unavailable. | semantic | user / NB1 |
| A008 | Target loader verifies numeric masks/values and producer/provenance/bounds; producer certification remains external. Traversal qualification requires supervised positive and negative weights; no new loss or target production. | semantic | user / NB1 |
| A009 | Atomic NPZ then JSON receipt; resume only exact config/raw/source binding and validated payload. Missing raw produces explicit retained failure rows and nonzero CLI completion status; never full-cache success. | interface | default |

## BUILD_NOTE (predeclared ladder)

| Rung | Feature / one check command | Tier | Status |
|---|---|---|---|
| F0 | plan entry -> plan.json: Python -B CLI plan --config E fixture | MUST | pending |
| F1 | strict full-N input, geometry, clocks, maps: scoped test_native_bridge_v1.py | MUST | pending |
| F2 | float cue, physical-unit equivalence: accumulated two new test files | MUST | pending |
| F3 | masked NOLA, one decode, target interface and CLI build/validate fixture | MUST | pending |
| F4 | legacy CPU regressions + Ruff; independent review by root | MUST | pending |

## Run record
Pre-code ledger composed into this owned document per root authorization. Pinned skill and mandatory references read; balanced five-rung budget, draft implementation assurance. No manifests/state writes or commits (task override). Full population build waits for completed cache, and is not part of fixture evidence.

## Deferred / blockers
Actual full977/2364 build and census, dynamicN v15 producer, target quality/coverage, metric freeze, fewer-than-three-term loss revision and full training budget remain separate gates. Root owns fresh Sol silent-assumption sweep; no semantic PASS claimed.

Pre-fix ledger A010 (semantic, Source=user/Astra clarification via root): W+=sum(supervised*chi*pulse), W-=sum(supervised*chi*(1-pulse)); both must be positive for traversal eligibility. Only defining mask/eligibility, not a new loss. A011 (interface, Source=NB1): one-row identities retain N=1 and zero edges as unusable/abstain; model and cue are not executed. A012 (semantic, Source=NB1): production census additionally checks frozen totals sumN=695625, sum(persons*N)=1758034; aliases are never deduplicated.

## Implementation and final engineering evidence

F0 ran before feature code: `python -B scripts/build_temporac_native_learner_bridge.py plan --config E:/ICASSP_TempoRAC/aris-work/native-bridge-cpu/plan_config.json`, exit0, actual `plan/plan.json`, status `prepared_not_built`. The original minimal plan spine has been replaced by the complete census-aware CLI. F1–F3 were implemented as a coupled change before their first accumulated check; this is a procedural deviation from checking each rung before coding the next, not evidence of separate earlier green rungs. The final accumulated suite verifies their stated mechanical obligations.

The learner NPZ has exactly the ten dataclass numeric fields; bbox, annotation paths and targets never enter that loader. Trusted raw projection preserves available raw observations/PTS on aligned failed videos while setting all source-response support false; L/D mismatch projects fixed annotation rows unsupported without inventing alignment. Preprocessing preserves every row and uses the unchanged old root/shoulder normalization. The native geometry routine is an explicit copy of the old F4 routine with finite length zero permitted; lag rows still reset at run boundaries. A one-row identity remains a zero-edge unusable record, without model execution. Timebase candidate interpolation/extrapolation weights are audit metadata only; no resample_values call exists.

The native clock/cue path uses video-level tau and subtraction of Python integers before binary64 conversion. It preserves duplicate/backward/missing PTS rows, segments support, accepts positive subunit q differences, and rejects delta q>=5. NUDFT formulas, constants, quadrature summation, reliability, reference and gates reuse the old implementation; only the float clock entry and NB1 clock-grid guard are added. Window/lag/convolution units remain native rows. No new antialiasing or continuous-time-network claim is made.

The NumPy NOLA path mirrors the existing NumPy implementation without its module's eager torch dependency. It fuses cached full-identity branch probabilities, applies the existing taper, verifies positive supported denominators and exact zero unsupported numerator/denominator, then decodes once per identity. Existing response.py already accepts `[N,269]` and explicit edge run bounds; callers pass `identity.response_input` and `identity.run_bounds` to its cached_responses once, then slice its cached probabilities for native routing/NOLA. No model forward or torch validation was run here. Partial counts are response components and source ranges, never repetition period labels; all-unsupported outputs retain N-1 arrays and serialize count=-1 after the required empty component call.

The target loader is a separate strict numeric/mask interface with explicit producer/hash and certified traversal bounds. W+/W− include both chi and pulse as A010 specifies. This is not a dynamicN target producer or a training loss. Single branch provenance is the NB1 narrow interface; the separately designed NT1 outer envelope must retain mixed per-run/per-traversal provenance, not relabel it as geometry here.

CLI config has exactly `raw_root`, `census`, `output`. `census` is the trusted complete bbox-only projection JSONL. All production plan/build/validate paths enforce train794/1878, val183/486, total695625 native video frames and1758034 person-frames, preserving aliases. Build refuses to write identity artifacts while any raw record is absent; it emits pending status and exit2. After raw completeness, each slot receives feature/derived artifacts or an explicit failure ledger row. Existing artifacts require the same config/census/raw/source binding, valid hashes and exact recomputed feature/derived arrays. Derived NPZ stores mappings, response channels, masks/runs and lengths; sidecars store binding/status. The complete report includes per-slot observed/derived/clock/response support and duration, plus distributions by clock mode. Target support remains explicitly unavailable.

### Actual check record

- First new-suite run: **1 failed,15 passed in17.79s**. Failure `test_confidence_observation_and_hip_thresholds`: the fixture assigned float32(.2), then wrongly expected it to fail the old binary64 `> .20` comparison. Actual float32(.2) converted to binary64 is above binary64(.20); actual mask was true. The low-score case was changed to .19. The production threshold and old preprocessing were not changed. A separate regression now checks predecessor float32(.2), float32(.2), and successor against both the old and new entry points, explicitly preserving this bit boundary.
- Subsequent accumulated check: **55 passed in27.54s** (new interfaces plus old geometry/cue/timebase/decoder/firewall).
- Expanded accumulated check: **73 passed in34.15s**, Ruff direct executable all checks passed.
- Final stable code: **75 passed in31.68s**, pytest exit0 and Ruff exit0. Exact stdout receipts: `E:/ICASSP_TempoRAC/aris-work/native-bridge-cpu/pytest_final.txt`, `ruff_final.txt`. Final basetemp `E:/ICASSP_TempoRAC/aris-work/native-bridge-cpu/pytest_final`.

Final command (Python `E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe`, NumPy1.26; TEMP/TMP and PYTHONPATH explicit on E; PYTHONDONTWRITEBYTECODE=1):

```text
python -B -m pytest -p no:cacheprovider tests/temporac/test_native_bridge_v1.py tests/temporac/test_native_time_cue_v1.py tests/temporac/test_preprocess_geometry.py tests/temporac/test_quadrature_cue.py tests/temporac/test_timebase_audit.py tests/temporac/test_decode_audit.py tests/temporac/test_feature_io_firewall.py tests/temporac/test_native_pose_cache.py --basetemp E:/ICASSP_TempoRAC/aris-work/native-bridge-cpu/pytest_final -q
ruff.exe check --no-cache src/pams/temporac/native_bridge_v1.py src/pams/temporac/native_time_cue_v1.py scripts/build_temporac_native_learner_bridge.py tests/temporac/test_native_bridge_v1.py tests/temporac/test_native_time_cue_v1.py
```

| Requirement / rung | Evidence and status |
|---|---|
| F0 real entry -> artifact | PASS engineering: exit0 before feature implementation, actual E plan.json |
| F1 N216/N900/no fabricated320, masks/normalization | PASS engineering: full lengths, missing two frames, score bits, hips/eight joints, exact old feature equivalence; old FeatureRecord rejects216/900 |
| F1 support/time/mapping | PASS engineering: 16/17 samples, one-row abstention, large integer ticks, rational base, missing/duplicate/backward/4.999/5 steps, observed-only mappings and unsupported all-false channel; source boundaries0/N-1/N |
| 31/32 geometry boundary | Actual geometry mask has paired x/y coordinates, so31 is unreachable. Test uses realizable30 and32 and confirms actual >=32 gate; this does not claim an artificial31-coordinate fixture was run |
| F2 float cue/gates | PASS engineering: full valid q=i spectrum/gates exactly equal old; physical rational-unit change preserves spectrum/gates; static uniform, grid fallback, per-identity isolation |
| F3 reconstruction/readout/target | PASS engineering: overlap/gap support, positive NOLA, exact zero unsupported, one decoder call partial/empty, target hash/members, unsupported-only-positive traversal rejection |
| F3 complete CLI behavior | PASS synthetic engineering: actual fixture NPZ/JSON artifacts, build/validate/resume, corruption rejection, census override inaccessible to CLI, pending raw prevents partial identity build |
| F4 regressions/lint | PASS engineering: final75 tests and Ruff exit0; active extractor SHA unchanged d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1 |
| Fresh Sol review / assumption sweep | PENDING, owned by root; executor has not supplied a Type-B PASS. Positive same-family review can only be provisional |
| Actual full977/2364 build/validation | NOT RUN: complete raw cache still generates. Fixture build is not full data evidence |
| Target producer/scientific quality/formal training | NOT COMPLETE; NB1 does not authorize training, fix the three-term old loss, establish SOTA or provide PeriodAP |

No live production stubs. No legacy source/API, active extractor, server, GPU/job, environment, manifest, workflow state or training configuration was mutated; no install, benchmark test-label read, commit or push. Full-data failures and original negative evidence remain untouched.

### Read/source receipt

Pinned implementation SKILL and installed copy SHA256 both `33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8`, upstream commit f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Read complete SKILL plus effort-contract, output-language, acceptance-gate, capture-antipatterns, injection-hygiene, reviewer-independence, reviewer-routing, review-scope-limits, reviewer-backends, output-versioning, output-manifest and output-composition. Parent explicitly authorized composing SPEC/ASSUMPTIONS/BUILD_NOTE here, and owns the fresh review. Project policy/current status, NB1 full contract, native raw contract/cache docs and relevant preprocess/cue/quadrature/NOLA/decode/timebase/response code were inspected. Requested execution model follows root's Sol routing; actual backend variant is not independently attested. Scientific NB1 design remains same-family provisional.

## Narrow review corrections — original findings retained (before dependent fixes)

Fresh root-owned reviewer reported three concrete blockers after the initial75-test receipt; that receipt remains historical, not final acceptance.

- R1: N80 target traversal[10,60), pulse=.10, chi=1/50, missing response frame40 leaves edges39/40 unsupported, yet the original native_supervision could return eligible=True. A013 (semantic, Source=sweep/NT1 clarification): any response support loss anywhere inside a certified traversal invalidates the whole traversal. Clear its complete supervised interval and positive/negative weights, preserve response_support_gap reason; leave successful other traversals intact. Missing positive/negative mass also invalidates and clears its traversal interval. Returned positive_weights/negative_weights are raw masked chi*pulse and chi*(1-pulse), not newly normalized losses.
- R2: the original NB1 target interface only stored identity-scalar branch provenance. A014 (interface, Source=sweep): add optional traversal_provenance tuple/list with exactly one full-chain geometry/stable-pair hybrid value per traversal; old single-source scalar broadcasts. Scalar mixed is valid only with actual distinct per-traversal sources; mismatched summaries reject. Sidecar accepts the additional traversal_provenance list while preserving old single-source files. NT1 writer confirmed its independent envelope and lf1_supervision adapter do not construct NativeTargetV1; no NT1 source or science changes.
- R3: resume/validate originally loaded feature NPZ before comparing its recorded digest. A015 (interface, Source=sweep): check both feature and derived recorded digests before any NPZ load, so a truncated ZIP produces existing ValueError failure ledger/nonzero result. Preserve complete census and continue subsequent video records using the existing loop. No new threat model or hash framework.

These original findings are not erased or reclassified as passed. Only the narrow corrections and their regression evidence will be appended below; root owns the same-session narrow recheck.

### Narrow correction receipt (R1/R2/R3; no expanded review)

Implemented exactly A013–A015. `native_supervision` now returns full-edge immutable `positive_weights` and `negative_weights` alongside its existing fields; these are raw masked masses, and every rejected traversal has its entire mask and both masses zero. Eligibility checks complete response support before accepting either mass. Regression uses N80/frame40 missing, traversal[10,60) with both a sole positive pulse at edge10 and a constant pulse=.10, chi=1/50; a distinct successful[60,79) traversal stays supervised with its exact original weights.

`NativeTargetV1` adds final optional field `traversal_provenance: tuple[str,...] | None`. Legacy scalar single-source callers/files broadcast to K traversal entries. New metadata stores `traversal_provenance` as a JSON list aligned exactly with bounds. Allowed certified sources remain full-chain geometry/stable-pair hybrid; actual distinct sources require scalar summary mixed. Missing/mismatched/malformed per-traversal metadata and false mixed/single-source summaries reject. Rejected targets still cannot supervise. Actual NPZ+JSON mixed roundtrip and backward scalar tests pass. NT1 writer explicitly confirmed its separate envelope/adapter has no dependency conflict; no NT1 file was changed.

Builder now verifies both recorded learner NPZ hashes before loading either archive. Regression parameterizes truncated ZIP bytes `b'PK\x03\x04broken'` and appended corruption against feature/derived artifacts. Both validate and resume/build return2, retain all fixture population rows, record hash mismatch for the failed video, and complete the subsequent healthy video. This uses the existing failure ledger and loop, with no new framework.

First narrow suite attempt:4 failed/26 passed in19.23s because the newly added second fixture video reused the first fixture media_path, correctly rejected by the existing full-census duplicate-path check before artifact validation. The fixture now uses a distinct path. This did not trigger a production threshold or policy change. Final impacted suite (the two owned NB1 test files): **30 passed in23.44s**, pytest exit0; all five owned Python files Ruff exit0. E-only TEMP/TMP, -B/PYTHONDONTWRITEBYTECODE and `-p no:cacheprovider` retained; basetemp `E:/ICASSP_TempoRAC/aris-work/native-bridge-cpu/pytest_narrow_final`. Exact stdout receipts `pytest_narrow_final.txt` and `ruff_narrow_final.txt` in that directory's parent. The earlier75-test record remains intact; no unrelated old-scope tests/reviews were repeated for these narrow fixes.

Same-session independent narrow recheck is pending with root's existing reviewer. No semantic acquittal is claimed by this correction receipt. All full-cache/target-quality/metric/training gates remain pending as previously recorded. Frozen active extractor and all other modules/jobs remain untouched.
