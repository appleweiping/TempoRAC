# Independent review — RACnet receipt-only input refresh

**PROVISIONAL_PASS. No actionable blocking defect found.**

Requested: gpt-6.1-sol / xhigh. Exact actual model and effort: unattested. Fresh native reviewer session: /root/racnet_refresh_prelaunch_review_20261112_2259. OpenAI same-family; acceptance remains provisional.

PROVISIONAL_PASS. No actionable blocking defect found in the exact receipt-only RACnet input refresh delta.

The new entry inherits the existing checked supervision and manifest identities, qualifies author/QUVA inputs through existing committed metadata and byte sizes, and preserves targets.json, score_labels.json and quva_raw_inputs.json. It does not call preparation, GT/decoder, model code or array loading and does not rehash feature/target payloads. The original native input/science behavior is unchanged outside the addition. The new CLI branch is CPU-only and the existing cpu-refresh bridge remains unchanged.

The existing synthetic server fixture covers pending-to-qualified receipt transitions, byte stability, nonqualification fields, the cpu-refresh bridge, changed-supervision rejection and formal author completion. Its source identity agrees with the exact reviewed snapshots and E sources. I did not rerun it. No uncovered concrete defect required another server fixture.

The captured production author STATE is running with 53/61 verified, so actual refresh remains ineligible. This source review does not establish production input readiness, scientific correctness, performance, or GPU release. The existing root publication/review/ownership gates still apply to these exact sources.

ASSUMPTIONS I01-I07 match the implementation; undeclared=[], stale_rows=[], semantic_undeclared=0. Requested reviewer gpt-6.1-sol/xhigh; actual exact model/effort are unattested. This fresh native OpenAI review is same-family and provisional.

| Acceptance requirement | Result | Evidence |
|---|---|---|
| Only refresh function and CLI action change | PASS | Static before/after comparison restores both original files when only additions are removed. |
| No science, GT/decoder, array/payload open or payload rehash | PASS | New call graph uses JSON/JSONL receipts and size only; supplied server fixture reports zero forbidden calls. No rerun. |
| 886 targets / 152 RC-test native_gt and frozen manifest bytes inherited | PASS | Original CPU/readiness/manifest identities match EVIDENCE_BINDINGS; source does not write frozen manifests; fixture asserts exact bytes. |
| Population/count/order/ROI/GT/TSM semantics and initial prepare unchanged | PASS | Original native_inputs.py is text-identical outside addition; fixture checks nonqualification fields. |
| Formal61 completion fail-closed | PASS | native_inputs.py:220 requires61 verified plus complete/completed; supplied53/running snapshot is ineligible. |
| Dispatch, cpu-refresh bridge and exact candidate identity | PASS | New branch at run.py:198; unchanged cpu-refresh at145; supplied fixture and E source hashes agree for all5py and inherited science/env/runtime. |
| Ledger agreement | PASS | I01-I07 match; undeclared=[], stale_rows=[], semantic_undeclared=0. |
| Actual production input readiness / GPU release | NOT ASSESSED | Synthetic fixture only; no actual refresh or GPU dispatch; existing root publication/ownership gates remain. |

Read/scope receipt: E:/AGENTS.md, repository AGENTS.md, ARIS_LOCK.md and docs/aris/WORKFLOW.md read directly. Complete pinned research-implement-feature SKILL and12 total skill/reference files read; installed copies match their pinned originals. Exact paths and identities are preserved in INDEPENDENT_REVIEW.json. Primary before/after source, raw diff, assumption/build/spec documents, fixture source/JSON/log, inherited CPU/readiness, initial manifests, author STATE, EVIDENCE_BINDINGS and public summary were inspected directly. Original large pending lists were summarized as metadata; no feature/target payload was opened. The native_common identity/receipt/GPU gate was inspected only to trace changed-code integration.

No new fixture, local experiment, actual refresh, SSH probe, signal, launch, Git mutation, source_sync/current_status mutation or published-packet modification was performed. Only this review JSON/MD is written. No additional experiment or review matrix is proposed.

The following is the exact original review request, including its scope block:

# Affected-interface review request — root fresh Sol

Read primary files directly; executor summary is not a substitute.

- SPEC.md, ASSUMPTIONS.md, BUILD_NOTE.md
- INPUT_REFRESH.diff; before/native_inputs.py, before/run.py; after/native_inputs.py, after/run.py
- metadata_fixture.py, TARGETED_INPUT_REFRESH_FIXTURE.json/log
- before/CPU_CHECK.json, before/READINESS.json; ../raw-private/ frozen initial inputs/targets/scorer/rawQUVA and EVIDENCE_BINDINGS.json
- Current candidate W=/root/autodl-tmp/TempoRAC/baselines/racnet_native_v1 (5py identity in targetedfixture)
- ../REFRESH_INPUTS_PUBLIC_SUMMARY.json

Scope: only receipt-only input qualification, inherited supervision and manifest stability, fail-closed formal-author gate, run.py dispatch, and reuse through existingcpu-refresh. Verify unchangedprepare/native math/population/count/ROI/GT/TSM semantics and exact886 targets/152RC-test native_gt inheritance; code may not open oldtarget/feature payloads, call science/GT/decoder, regenerate caches or redoCRC/finite/SHA overpayloads. Changed code requires its newexactidentity and root publication/review receipts before GPU. Source/CPUmetadata fixture is synthetic and cannot be counted as actualauthor/QUVA production. No need to rerun oldmodel/GT/CRC checks or inspect unrelated currenttraining.

Please return findings with concrete existingtrigger, site/severity and focusedfix; audit undeclared/stale ledger assumptions. Say plainly if no issue. Same-family positive outcome staysprovisional; requested Sol/xhigh, actualidentityunattested unless independently provided. Root handles dispatch; no review spawned by executor.
=== SCOPE LIMITS (these bound what you PROPOSE, never what you look for) ===
Report anything that is actually wrong here — including a rare-looking case, if
this repo actually produces it. Then keep the fix in scope:
1. This is a RESEARCH-WORKFLOW tool, not a security paper. Verification is
   welcome; over-defense is not. Assume a cooperating operator on their own
   machine — a malicious local user is NOT in the threat model.
2. Do NOT propose SHA / hash / content-fingerprint / digest-binding schemes.
   Reporting a real defect in hashing code that already exists is fine.
3. NO speculative machinery: do not add feature flags, migration frameworks,
   compat layers, wrappers, pins, or similar mechanisms unless evidence shows
   a current repo defect they fix or an explicit existing invariant they must
   preserve. "Load-bearing", "compatibility", and "not scaffolding" are labels,
   not evidence. Point to the failing path/artifact or invariant, and check the
   proposal's factual premises, such as whether a named package version exists.
4. NO corner-case obsession: exotic encodings, symlink races, RTL text and
   millisecond races are out of scope unless you can show the case arises here.
5. Where a rubric or checklist is genuinely needed, do not over-mechanize
   judgement. A clear sentence a human reads beats a scored table nobody
   maintains.
Exception: code that runs remote commands, starts a network service, or installs
an MCP server runs on the user's machine with their credentials — trust-boundary
findings there are in scope and the default is strict.
Say plainly when something is correct. Do not manufacture findings.
