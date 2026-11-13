# Independent bounded review — GPU1 after RACnet

Verdict: **PROVISIONAL_PASS**. The complete affected implementation scope was reviewed. There are no actionable findings, no undeclared interface or semantic decisions, and no stale assumption rows in the reviewed candidate. This is a positive bounded engineering review; actual backend/model identity is unattested and acceptance remains provisional under the project's same-family policy.

Candidate read and verified: `9a4857a68f54ac526a95b008e4e222943e4983f6e17fcdb8b7bc2971d64abcad`. The E packet entry, repo entry and server entry have these same bytes. The 13 existing packet bindings match locally and remotely. The actual original worker is `/root/autodl-tmp/TempoRAC/resume20261112/ablation-continuation/ablation_worker.py`, discovered from the entry and existing launch; it is not a guessed continuation_worker.py.

## Identity, completion and limits

- Review session: `/root/racnet_ablation_successor_review_20261113`; fresh bounded context, no executor history.
- Parent-supplied resolved request: Codex native, `gpt-6.1-sol`, `xhigh`, regular tier. The route helper was already resolved by the parent and was not rerun here.
- Actual returned reviewer model and effort: unknown/null; no independent backend attestation is available. Requested executor is `gpt-6.1-sol`; actual executor is also unattested.
- Transport: native reviewer session completed this report. Review scope is complete; findings: 0; semantic undeclared: 0; stale rows: 0.
- Review independence/acceptance: same-family policy / provisional; this is not cross-family acceptance.
- The fixture's fabricated publication/review objects are test inputs. They do not constitute this review or a real production release.

I read the full pinned research-implement-feature skill, reviewer-backends and review-scope-limits, E and repo AGENTS, ARIS_LOCK and workflow. I read the primary packet files, both successor sources, the actual original worker and GPU1 controls, the native common/engine/recovery schemas, and the full finite RACnet runner. I inspected existing receipt metadata without opening model, checkpoint, feature, array, label or prediction payloads. I did not run the fixture, entry, --check, --execute, GPU query, model or job. No scientific protocol, result comparison or SOTA assessment was reopened.

## Findings

```json
{"undeclared":[],"stale_rows":[],"semantic_undeclared":0,"findings":[],"verdict":"clean"}
```

## Completed scope and evidence

1. **Declared decisions match code.** A1 matches canonical-release selection in assess: one PIPELINE_COMPLETE with the canonical launch release SHA, never timestamp-latest selection. A2 matches validate_release and main: explicit release, exact residual, entry identity, physical GPU identity, pipeline receipt and startup expiry. A3 matches the retained worker --resume path and metadata-only RAC checkpoint checks. A4 matches the existing 137438953472-byte shared reserve. There are no new model, metric, data, epoch, seed or selection choices in this successor.

2. **Exact queue and frozen provenance.** ablation_control (candidate lines105–129) checks worker, contract, hold, held lane and launch against their existing identities, frozen source HEAD/clean tree/train source, spec and manifest. The native command equals the original launch plus --resume. Remote metadata shows GPU1's five completed prefix arms wo_shift_s0/s1/s2 and wo_token_s0/s1, all50epochs with matching saved configs and worker/contract identities; the held suffix is exactly wo_token_s2 and wo_joint_cx_s0/s1/s2. All four residual output directories are absent. The inherited worker maps wo_token_s2 to50epochs and the three wo_joint_cx arms to80epochs through the unchanged contract/spec.

3. **Completed work is reused.** The new gate checks completed prefix provenance and epoch metadata without rehashing old experimental payloads. The unchanged original worker's complete/validate_residual path remains authoritative at real --resume and checks saved completion digests before skipping those arms. No feature/cache rebuild or rerun command was added. The original contract SHA is dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167; original worker SHA is d23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144.

4. **Canonical whole RACnet sequence.** racnet_complete (lines54–103) validates the canonical release/publication/review/runner/candidate chain and the ordered extract-quva, refresh-inputs, train, evaluate sequence. Each stage must be COMPLETE/exit0 or REUSED_COMPLETE/no exit code. The same finite helper functions qualify full105/51349-frame feature metadata, full1243 readiness, full200 training and full357 evaluation metadata. Incomplete, failed or nonunique canonical receipts block; the successor does not restart RACnet.

5. **Native completion accounting and selection.** Candidate checks align with the existing native_engine writer: 200 indexed epochs,755 training rows,48 attempts each,9600 total attempts, actual updates plus AMP skips,189 native validation candidates (epochs11–199), finite native MAE/OBO and earliest-epoch minimum native MAE. The selected training ref must equal COMMIT.best_mae, its next_epoch must agree, and COMMIT.latest must reach200. The fixture correctly accepts9599 actual updates plus1 overflow skip rather than requiring9600 successful updates. This review verifies gate/schema correspondence, not scientific effectiveness.

6. **Final identity and checkpoint preservation.** The final binding uses the same training identity, selected checkpoint SHA and prediction-input metadata identity as the original native writer. The gate checks357 unique exact keys and RC152/UCF105/QUVA100 membership, finite counts, per-row input identity and shared binding. Existing final receipt digests remain checked by finite.evaluation_complete. Checkpoint payloads are only required to exist, be nonempty and resolve under the existing checkpoints directory, as explicitly declared; this review did not load or rehash them. The successor never writes RACnet checkpoints, caches or completion artifacts.

7. **Ownership and live resources are separate gates.** assess (lines131–165) checks recorded root/child PID plus exact command, excludes zombies, and takes both existing RAC writer locks before metadata transfer. The resource query is restricted to physical GPU1, requires the recorded UUID, memory below500MiB and no listed processes, and never signals occupants. Canonical root alive, writer lock held or GPU occupied yields WAIT. A matching live child or invalid/partial completion blocks without dispatch. The live-child BLOCKED outcome is a stop with retained evidence, not a waiting/retry loop; the fixture specifically tests the root-alive WAIT path. The old worker PID435833 is currently absent.

8. **Once-only transfer and recovery.** main (lines175–207) holds the successor lock during assessment/transfer, creates the durable one_shot claim exclusively, then renames the original hold into the packet runtime archive. It commits no-repeat state before one original-worker Popen, preserves original lane/events/checkpoint/output paths, and waits only for that finite child. Existing claim, partial residual output, changed control or failure blocks. Any post-claim exception retains the claim/archive/logs for explicit manual recovery; there is no automatic second dispatch, GPU waiter, scheduler or retry.

9. **Fixture fidelity and evidence boundaries.** I read metadata_fixture.py and its existing PASS16 result, bound to the candidate SHA. Its positive RAC metadata follows the actual finite/native field shapes, including AMP skips, validation eligibility and final membership. Negative cases exercise failed/order-invalid stages, update inconsistency, wrong best-MAE, duplicate/missing final membership, worker-source tamper and invalid/expired root release. Later lifecycle cases intentionally replace ablation_control, racnet_complete, modules, process/resource interfaces and Popen; they demonstrate only fake resource/transfer/once-only behavior. This is accurately labelled metadata-only in the packet. It is not an end-to-end production run or evidence of an actual free GPU. Unexercised real-lock/process branches were inspected statically; no fabricated fixture review marker was accepted as actual review.

10. **Publication envelope and operation limits.** SOURCE_ALLOWLIST contains only the new public source entry. Packet/repo/server candidate bytes agree. The approved GPU0 reference is unchanged at6116d44c3bff0cdeaf22854eb123c537f06a9aaf5333edcd9504406d109aa0cb. Server and repo native_common/native_engine/native_recovery/full-runner bytes agree. This review wrote only its own review/acceptance artifacts within the E packet and mirror. It did not publish Git/source_sync, create a release, alter original controls, touch another project or start any background process.

## Current runtime boundary

The actual canonical launch path `/root/autodl-tmp/TempoRAC/baselines/racnet_native_v1/GPU_PRODUCTION_LAUNCH.json` was absent during this review. The successor correctly has an explicit WAIT path for that state. No canonical full105/full200/full357 completion is claimed here, and no real readiness check was executed. A real root invocation and release remain deferred until the actual RACnet whole-arm completion and fresh resource gate. Existing complete experiments remain outside this review.

## Source identities

| Source | Verified SHA256 |
|---|---|
| E packet, repo and server resume_gpu1_after_racnet.py | 9a4857a68f54ac526a95b008e4e222943e4983f6e17fcdb8b7bc2971d64abcad |
| Repo resume_gpu0_after_transrac.py | 6116d44c3bff0cdeaf22854eb123c537f06a9aaf5333edcd9504406d109aa0cb |
| Actual server ablation_worker.py | d23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144 |
| Actual server RUN_CONTRACT.json | dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167 |
| Actual GPU1 hold / lane / launch | 8017bcc146e537b408af49f2baa62567c66af6575c5eb905a0678ebbb4a00a40 / 71cf338ea8ad0fd28c0976bf7832753d883cd842ea85531c11bdcd5ba0cb00ff / 858fa67c3efd06c5e1c689393564e38d874b7c76756c0db542e0ac712c4c745e |
| Repo and server native_common.py | 8fa18e68260c5376463816ae1a36c369071cc58f417d67bb1198fa64adc669a3 |
| Repo and server native_engine.py | e74f300f6eaac84a3eed54e154c9e705913d7b892f4d8f2a1f4125743e437739 |
| Repo and server native_recovery.py | 9abc552789ba8dd326b59b9277e501766f46e46de6125ccb072d47a605ecdc15 |
| Repo and actual server run_racnet_missing5_full.py | 00106817d7ffd2a9921a86bfb583ad32766641dd093dbf9c887fabdf4dc3763e |

The JSON companion preserves the complete review request verbatim, this complete response verbatim, exact current paths, packet identities, scope completion and the read-only server metadata observations. Acceptance evidence is in ACCEPTANCE.md. No code fix is requested.

