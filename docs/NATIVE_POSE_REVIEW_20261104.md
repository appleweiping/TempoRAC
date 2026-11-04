# TempoRAC native pose cache — independent technical review, 2026-11-04

## Review identity, authorization, and scope

Fresh native reviewer session: `/root/native_cache_review`. Requested route: codex/native, `gpt-6.1-sol`, `xhigh`; canonical pinned route resolver returned that pair. Actual host is an OpenAI Codex/GPT-family reviewer; the exact backend variant and effort are not independently attested by the host. Requested identity is not substituted for actual identity. `review_independence: same-family`; `acceptance_status: provisional` throughout. This is engineering review of raw-cache construction, not scientific/model-readiness approval.

Review was prepared from the contract before opening implementation. Root supplied STABLE hashes; source first read and independently hashed: `scripts/build_temporac_native_pose_cache.py` da0e6b8c8b5eef1a301aa2b2800bdaacd1a9baf6433b69cf8ae092da3cd9961e; original tests 8a2cee18693dc58ab2db06840194f7dc60cca70bb6fdd3f8a886cc6fee1f757c; initial implementation document 6035e467e98745ccdb2d3e62703212e3b7511da1e34fecf6397a860357a683bc. During reading the author added the disclosed resume regression; its next observed hash was 169dee7c65ccd237eecb9fdd4bff043de09b3e5edf19ebb99123511f27679f79. Root explicitly authorized reviewing current tests and documentation updates. No reviewer source/test changes, installs, shared-environment changes, GPU access, model construction, model forward, server model run, or scientific-method changes were performed.

Primary scope: the three implementation artifacts above, `docs/NATIVE_POSE_CONTRACT_20261104.md`, and supporting exact API/decode interfaces. Objective: find concrete blockers to full977 bbox-only projection and complete offline supplied-bbox raw extraction. Existing A005 security proof requirements were not reintroduced.

## Preparation and read receipt

Read in full: E:\AGENTS.md; project AGENTS.md, ARIS_LOCK.md, docs/aris/WORKFLOW.md, docs/CVPR_RESUME_20261104.md, current .aris/runs/temporac-cvpr-20261104.json; native pose contract; pinned research-implement-feature and research-review SKILL.md; mandatory shared references reviewer-backends, reviewer-routing, review-tracing, acceptance-gate, reviewer-independence, review-scope-limits, effort-contract, capture-antipatterns, injection-hygiene, output-language, output-composition, integration-contract. Pin independently verified: f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Applicable copied skills equal source hashes: research-implement-feature 33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8; research-review 17505d5982c82e821124de898b0beb7de4a98ecc4c1cd173b8687b74ace9d4c5.

Supporting reads: E:\ICASSP_TempoRAC\aris-work\pose_frontend_inventory_20261104.md and its raw `_4.txt` API receipt; relevant audit_temporac_decode.py clock/stream output; full projection and totals command/stdout receipts. No test annotation was read.

## Predeclared acceptance checklist and first-round outcome

| Requirement | Outcome | Artifact-grounded evidence and limit |
|---|---|---|
| Exact label-isolated projection | Pass at code/fixture level | Exact top-level/person white lists, recursive privileged-key rejection, explicit train/val paths; only trusted projector reads source JSON; frontend receives only RGB images and boxes |
| All977 videos/2364 supplied slots and aliases retained | Pass for projection receipts; code checked | main uses exact794/1878 and183/486 census, no cap; aliases are sidecar records without deduplication. Actual full project/validate receipts report977/2364; reviewer did not independently download source projections |
| Native frame/slot ownership, xyxy→xywh, COCO17, low score retention, FP32 | Pass at code/fixture/API level | `(frame,slot)` owners; same image one supplied crop per call row; no clipping, threshold=None, use_fast=False, no target_sizes, exact0..16 ordered labels; matching installed4.49 API receipt |
| Dynamic N and independent annotation L/decode D | Pass at fixture/code level | Raw L pose/bbox axis and D clock axis retained; mismatch invalidates pose support; no fake320/reindex/temporal fill |
| Honest clocks | Pass at fixture/code level | Integer pts and independent best-effort ticks/masks; rational time_base only when present; duplicate/missing PTS retained; source_frame_only fallback does not claim index/FPS is measured PTS |
| Failure support and zero payload, all slots retained | Pass at fixture/code level | Absent/invalid boxes, nonfinite joints, frontend errors, zero/partial/dimension mismatch fixtures; false masks and zero invalid payload; failure/attempt and population ledgers exist |
| Offline/lazy production dependency interface | Pass at source/API level | stdlib projection path; NumPy/OpenCV/torch/transformers lazy; exact four local assets verified; local_files_only and offline env; model eval/float32, no autocast; no actual model/device run |
| Resume matches current media | **Unmet, finding R1** | Completed-cache reuse occurs before existing media SHA check; changed media bytes are silently reused |
| Failed attempts can retry after bounded repair | **Unmet, finding R2** | Binding rejection runs for failed artifacts as well; batch repair cannot retry failed artifact with --resume |
| Atomic writes and no silent failed-video exclusion | Pass at reviewed code-path level, bounded | Each numeric/JSON/log file uses temp+replace; failed-video path preserves slots/artifacts/ledgers; no claim that multi-file writes are transactional or interruption recovery has been exercised |
| Full raw/model/device/learner qualification | Not complete, explicitly deferred | No raw977 inference/model forward; old fixed320 bridge not implemented; full decode/split/target/scorer scientific obligations remain separate |

## First-round concrete findings, reported immediately

### R1 — validate current media before completed-cache reuse

Source first-round lines742–757 load and reuse complete artifacts before lines761–766 resolve/hash the current MP4. The binding compares the projected SHA to the same projected SHA saved earlier, so it cannot discover changed/missing source bytes on this path. This contradicts the existing exact-media reuse invariant, rather than requiring a new security mechanism.

CPU reproduction creates a valid complete injected fixture cache, modifies media bytes without changing projection, then resumes. Actual output:

```json
{"case": "changed media resume", "projection_sha": "45a15d0c886daa87c755995a6352c77c2331eea7d7533e488d0a8473501c9be4", "actual_sha": "e6d36ed4f0a7d7a4e3b69813063f898529997f816312becaf9ea54c3c3e1234d", "result": [{"media_key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "status": "reused_complete", "persons": 2}]}
```

Bounded remedy: run the existing media-path/SHA verification before allowing completed reuse, preserving explicit failures and population accounting.

### R2 — allow failed-artifact retry under a repaired binding

First-round lines743–746 reject binding mismatch before testing whether status is complete. A video that failed because its batch was too large cannot retry with a smaller batch; changing script bytes for an actual repair has the same problem. The contract permits reuse only of matching complete outputs and requires retained failed attempts plus retry.

CPU reproduction first uses a fixture frontend that raises `fixture batch too large`, obtains status failed, replaces it with a succeeding frontend, changes batch2→1, and resumes. Actual output:

```json
{"case": "retry failed with repaired batch", "initial": [{"media_key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "status": "failed", "persons": 2}], "exception": "ValueError: resume artifact binding mismatch"}
```

Bounded remedy: apply reuse/binding rejection to complete artifacts; preserve failed attempt evidence and retry failed/unresolved artifacts with the current binding.

Reproduction script: E:\ICASSP_TempoRAC\aris-work\tmp\native_cache_review_repro_20261104.py. First-round fixture output directory: E:\ICASSP_TempoRAC\aris-work\tmp\native-review-repro-x5pbsvz4. These are synthetic engineering fixtures, not natural media or performance results.

## Commands and observed verification

All reviewer Python used E:\CodexRuntime\temporac-audit-py311\Scripts\python.exe -B. TEMP/TMP explicitly E:\ICASSP_TempoRAC\aris-work\tmp; pytest basetemp/cache were under E-only native-review-pytest/native-review-pytest-cache. NumPy/pytest/Ruff were existing dependencies; torch is absent from this local environment and projection/import/fixtures still succeeded.

- `python -B -m pytest tests/temporac/test_native_pose_cache.py --basetemp=E:\ICASSP_TempoRAC\aris-work\tmp\native-review-pytest -o cache_dir=E:\ICASSP_TempoRAC\aris-work\tmp\native-review-pytest-cache -q`: **11 passed in3.72s**.
- `python -B -m ruff check --no-cache scripts/build_temporac_native_pose_cache.py tests/temporac/test_native_pose_cache.py`: **exit0**.
- The E-only standalone reproduction above completed both cases and demonstrated R1/R2.
- Full projection execution receipts (author-run, reviewer-read) give train794/1878, val183/486, native annotation frames695625, person-frames1758034, projection bytes40091158 and SHA b53569b09c643ecdb041df8d4109fddf7d1bdeadfb2163a045f301b68b9761da. project and validate both REMOTE_EXIT0. Validation snapshot157 decoded audit records available,820 pending,0 mismatches among available; this is not full decode qualification.

## Current verdict

**Two concrete resume blockers require bounded fixes and same-session recheck.** No other concrete blocker was found in the reviewed original scope. The positive engineering observations are provisional and limited to source/fixtures/existing API receipts. No claim of complete raw cache, actual GPU/model integration, formal learner input, source-disjoint evaluation, method efficacy, or scientific acceptance follows.

Follow-up fixes and final hashes will be appended after root supplies STABLE revised artifacts.

## Same-session narrow recheck — final verdict supersedes first-round blocker status

Root supplied revised STABLE artifacts, independently rehashed before recheck:

| Reviewed artifact | Final reviewed SHA256 |
|---|---|
| scripts/build_temporac_native_pose_cache.py | d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1 |
| tests/temporac/test_native_pose_cache.py | 05f5a8628df3d42d53d02d4632a5ba3f9d33389782359bc3a19b9d3b11b523e8 |
| docs/NATIVE_POSE_CACHE_20261104.md | 7a82568cb219c309144945c237f8b9a396d289063695d4ed40e63275e71621dc |

R1 resolved: current source path/SHA verification now runs before completed reuse (lines742–747). Changed bytes produce explicit ValueError before reuse; the added regression verifies the previous completed NPZ remains byte-identical. Remaining ledger entries retain pending status when this deliberate integrity failure aborts; this is not represented as a complete extraction.

R2 resolved: strict binding comparison is now inside the completed-artifact branch (lines751–753). Failed artifacts may retry under the repaired batch/config, preserving historical attempt JSON. The added regression verifies old attempt file bytes unchanged and two attempt receipts after failed2→successful1 retry. Complete artifacts with changed effective configuration still cannot be reused.

Independent recheck commands (same E-only interpreter, -B/TEMP/TMP policy):

- Full scoped suite, separate recheck basetemp/cache: **12 passed in5.60s**, exit0.
- Ruff `check --no-cache` on both Python artifacts: **exit0**.
- Independent standalone recheck script E:\ICASSP_TempoRAC\aris-work\tmp\native_cache_review_recheck_20261104.py, retaining original repro script unchanged. New output root E:\ICASSP_TempoRAC\aris-work\tmp\native-review-repro-ebs7ujg5. Raw output:

```json
{"case": "changed media resume", "projection_sha": "45a15d0c886daa87c755995a6352c77c2331eea7d7533e488d0a8473501c9be4", "actual_sha": "e6d36ed4f0a7d7a4e3b69813063f898529997f816312becaf9ea54c3c3e1234d", "result": {"exception": "ValueError: media path/SHA changed since projection"}}
{"case": "retry failed with repaired batch", "initial": [{"media_key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "status": "failed", "persons": 2}], "result": [{"media_key": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "status": "complete", "persons": 2}]}
```

The revised document records semantic A011 (Source=sweep) and truthful outstanding real-model/full-raw/learner requirements. The projection generator predates extraction-only fixes; its stored generator SHA remains historical provenance rather than being relabelled with final script bytes.

Final acceptance update: both previously unmet resume rows are **Pass for reviewed source + CPU regressions**. All other checklist rows retain their bounded first-round evidence and limitations. **No remaining concrete engineering blocker found within the requested full977 projection/offline supplied-bbox raw-cache implementation scope.** `review_independence: same-family`; `acceptance_status: provisional`; exact backend/effort remain not independently attested. This positive scoped verdict does not attest actual model/device integration, completion of all977 raw artifacts, scientific validity, or formal learner-input qualification. No model forward or extraction job was started by this review.
