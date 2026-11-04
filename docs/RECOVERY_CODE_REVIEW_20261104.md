# TempoRAC CVPR recovery / timebase audit: independent raw review

Date: 2026-11-04. Reviewer session: /root/recovery_code_review.
Requested role: fresh independent routine code reviewer (Sol policy). Runtime tool metadata does not expose verifiable actual backend identity/effort to this session; parent launch receipt should be retained separately. No author implementation summary was used as a substitute for reading primary artifacts. This is an OpenAI same-family review: any positive semantic conclusion is PROVISIONAL, not cross-family acceptance or scientific validation.

## Verdict and findings

No blocking correctness defect found in the reviewed timebase module. Its interpolation, endpoint holding, original source-row indices, support kinds, explicit support threshold, immutable copied arrays, and sample-center coordinate mapping match the stated standalone audit contract. No training integration or modification of the frozen v2 implementation was found. No claim of scientific efficacy is supported by this review.

- **P3, non-blocking — stale manuscript status in the intake table.** `docs/CVPR_RESUME_20261104.md:32` still describes the manuscript source only as “旧 ICASSP pre-results draft”, while line 14 now records an existing CVPR pre-results worktree. Update that table cell to acknowledge the CVPR draft, retaining the explicit absence of new results. This is a local documentation consistency fix, not a reason to restart writing or experiments.

No other actionable finding. Do not reinterpret that statement as a complete project, data, model, scorer, or publication approval.

## Direct evidence and checks

1. Read `E:\AGENTS.md`, `docs/aris/WORKFLOW.md`, and pinned `skills/skills-codex/shared-references/reviewer-independence.md` plus `review-scope-limits.md` directly before review. Their independence and proposal limits governed this review. No new safety framework, hash mechanism, or million-byte certificate was proposed.
2. Read the complete current versions of `docs/CVPR_RESUME_20261104.md`, `docs/TEMPORAC_TIMEBASE_AUDIT.md`, `src/pams/temporac/timebase_audit.py`, `tests/temporac/test_timebase_audit.py`, `src/pams/temporac/preprocess_resampled_v2.py`, and `src/pams/temporac/types.py`. Untracked artifacts were read directly, not omitted from the diff review.
3. Inspected `git status`, `git diff 3e9ae828 --stat`, tracked workflow/document changes, and confirmed `git diff 3e9ae828 -- src/pams/temporac/preprocess_resampled_v2.py src/pams/temporac/types.py` is empty. HEAD remains `3e9ae828979bb7538da733835d286fcb395ccfbb`; new audit files are untracked at review time.
4. In the old v2, `output_time` uses physical time at lines 137–142, but emitted `resampled_clock` becomes `arange` at line 147; adapted `frame_mask` is all ones at line 167. The new docs accurately identify that retained legacy outputs do not preserve physical clock/original support. The old module also determines support per joint from frame validity, finiteness, and confidence at lines 79–84; audit assumption A-006 explicitly preserves this granularity distinction rather than treating a frame-level mask as per-joint observation.
5. New `build_timebase` lines 126–147 brackets only observed anchors while retaining original array indices, distinguishes exact equality from interpolation, holds the nearest observed endpoint outside support, and rejects an implicit gap threshold. The one-anchor path has matching endpoint indices and remains unsupported away from the anchor. Missing values at unobserved rows do not participate in interpolation.
6. Coordinate conversion operates over the output sample-center domain [0,N-1], with explicit rejection outside that domain. The existing target/prediction bounds in `types.py` are edge intervals whose upper limit can equal the edge count (N-1 for N samples); this does not establish a conflict with sample-center endpoint mapping. It also does not establish compatibility with an unreviewed official period scorer or intervals whose endpoints lie outside the sample-center domain.
7. Independent execution: `C:\Python314\python.exe -B -m pytest tests/temporac/test_timebase_audit.py -q -p no:cacheprovider --basetemp=E:/ICASSP_TempoRAC/aris-work/tmp/pytest-timebase-review` returned **13 passed in 2.44s**, exit 0. Set `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONPATH` to this checkout's src, and TEMP/TMP to the E: work directory. Python 3.14 is only an available engineering check, not certification of the project's supported runtime.
8. An independent in-memory assertion script on existing E: Python **3.12.13 / NumPy 2.4.6** passed fixed expected coordinates, weight sum, endpoints, and a linear signal. For output clock [5,7,12,14], rows [0,0.5,1,1.75,2,3] produced [5,6,7,10.75,12,14]; clocks [5,6,7,8,12,14] produced [0,0.5,1,1.2,2,3]. Source [5,6,14] with values [50,60,140] resampled to [50,70,120,140]. No test source was changed.

## Test and claim limitations

- The committed candidate test at `tests/temporac/test_timebase_audit.py:73–86` checks inverse consistency and domain rejection. By itself, roundtrip equality cannot exclude two mutually consistent but wrongly scaled conversions. The independent fixed-value check above addresses the current implementation; those assertions are not yet a regression test in the candidate source. This is a coverage limitation, not evidence of a present mapping bug.
- The support and index tests use explicit expected arrays, rather than only implementation-shaped self-consistency assertions. Stationary-pose testing checks exact preservation and source immutability. Legacy CLI testing checks actual NPZ reading, JSON output, return status, and unchanged input bytes. These are real CPU checks, not proof that all legacy caches are recovered.
- The half-open interval example only establishes endpoint mapping inside the declared sample-center domain. It does not validate event semantics, period AP, count conservation, arbitrary official annotations, or supervision support over complete natural videos. A gap threshold being explicit does not make that threshold scientifically justified.
- The module is intentionally a standalone audit. No adapter, model input, loss, training graph, pseudo-target branch, dataset manifest, full-population support distribution, or GPU execution was validated here. No scientific direction was substituted or rejected route reactivated by the reviewed code.
- R1 explicitly preserves negative results and marks full data, comparisons, components, loss-count compliance, and metrics freezing incomplete. It does not upgrade CPU checks to model efficacy. The new-server statement explicitly avoids inheriting old path/GPU/account facts; authentication and three-way deployment remain unverified. The heartbeat sentence does not claim experiment completion.
- The R1 additions for an existing CVPR draft and the heartbeat were read, but their worktree commit and scheduling service state were not independently inspected by this code review. Likewise, historical 110/51 video and 268/134 identity counts, v9/v12/v15 dependencies, literature novelty, and current official CVPR policy/date claims are outside this source review and were not independently re-established. No external contact, web request, SSH connection, or GPU job was made.
- Supported Python 3.11 project-wide verification remains outstanding; neither the 3.14 candidate test run nor the focused 3.12 assertions closes it.

## Reviewed source SHA-256

These hashes identify the already-reviewed files as requested by the review assignment; this review does not propose a new hash scheme.

| Primary artifact relative to E:\ICASSP_TempoRAC\TempoRAC-CVPR | SHA-256 |
|---|---|
| docs/CVPR_RESUME_20261104.md | d2e51e2234fdaec1b2edd098bd0106bd8b0e350734686461466bde38ed6e6cc3 |
| docs/TEMPORAC_TIMEBASE_AUDIT.md | c68872f13c0ccdcb778d614b3843be8d2902faaa3f9c955968bbdc2e594e3914 |
| src/pams/temporac/timebase_audit.py | 5f8d829c01a64e60c1e984bc9d872af1e5bcbaf4157aa1de2aeff360b8f5bd11 |
| tests/temporac/test_timebase_audit.py | 94e359643e58f891daefddcd3337415f5cb0d9a0735c2373ae6806ec4b8f82e2 |
| src/pams/temporac/preprocess_resampled_v2.py | da7481c1783ddf1e757d3b8194367c497fc87d044b87c55de0b93ca9415940ff |
| src/pams/temporac/types.py | 6d575ff34ccdcadaada925e5d3b97fb59c76580cbe48ce119cec3d5ed50cf84b |

The read receipt covers the exact primary versions above; later edits require a scoped follow-up. Report-only mutation: this file. Pytest temporary fixtures are confined to the named E: temporary path.

## Scoped follow-up: R1 factual updates and P3 resolution

Date: 2026-11-04. Re-read the complete current `docs/CVPR_RESUME_20261104.md` directly. No code or tests were re-run for this documentation-only follow-up.

- **P3 RESOLVED.** R1 line 32 now names both the old ICASSP and old CVPR pre-results drafts, consistently with line 14. Neither is promoted to completed experiments.
- R1 lines 13 and 21 now record successful SSH and four busy RTX 4090 GPUs; lines 33 and 72 consistently distinguish authentication from deployment, full data, and formal experiments. No stale “authentication unresolved” statement remains within this R1 version. This reviewer checked document consistency; the parent session supplied the new SSH status, and this follow-up did not independently connect to the server or inspect its raw remote receipt.
- R1 line 52 now explicitly gives TempoRAC and STaR-Body equal high priority, separates directories, processes and GPU quotas, limits this chat to TempoRAC, and states that STaR-Body must not be interrupted. It does not authorize killing unidentified jobs or deleting other projects. This matches the new resource-priority instruction supplied for this scoped review.
- No new actionable issue found in these R1 updates. Scientific scope and all other prior review limitations remain unchanged; the earlier report's “authentication unverified” describes its earlier version, and is superseded only as a parent-reported status by this addendum. This review still does not certify deployment, experiments, scientific validity, official CVPR rules, or external service facts.
- Identity clarification supplied by the parent from the original native spawn call: requested/native override `model=gpt-6-astra`, `reasoning_effort=xhigh`, `fork_turns=none`. This supersedes the initial role-only “Sol policy” label as the actual requested launch configuration. No separate backend identity attestation was available to this reviewer; do not claim one. Same-family positive semantic conclusions remain **PROVISIONAL**.

Final reviewed R1 SHA-256: `6e7c00b249c79e32770c2b1661b8ff060a5b79108308ddd7c42eb6d002e0f5e0`.

Unchanged source identities rechecked without test reruns:

- `src/pams/temporac/timebase_audit.py`: `5f8d829c01a64e60c1e984bc9d872af1e5bcbaf4157aa1de2aeff360b8f5bd11`.
- `tests/temporac/test_timebase_audit.py`: `94e359643e58f891daefddcd3337415f5cb0d9a0735c2373ae6806ec4b8f82e2`.
