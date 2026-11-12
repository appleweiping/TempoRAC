# TempoRAC continuation — 2026-11-12

The user explicitly resumed this chat after the November11 pause. Resume from the verified25b8c47 checkpoint, the actual AutoDL project `/root/autodl-tmp/TempoRAC`, and `aris-work/current_status.json`'s `continuation_20261112`. The old November4 scheduler/server/stage fields are historical and must not restart work.

## Inherited work and execution boundary

- Keep the29 completed original-protocol runs, their checkpoints and raw results. The source-overlap and test-informed-selection qualifications still apply; do not relabel those runs as independent corrected-protocol results.
- Keep the15 framework/version literature entries (14 families if MultiCounter+ is collapsed),13 verified comparison-table counts, prior full-read receipts and explicit unavailable-source gaps. No repeat literature/novelty pass is requested.
- ARIS remains pinned to `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`. README and installation hashes match the inherited full-read receipt. Applicable stage skills and mandatory references are read before use.
- The paused corrected runs retain12/13 epochs of logs but no saved model checkpoint. Those weights cannot be reconstructed. Fix durable epoch-boundary checkpoint/resume before relaunch; never present a fresh decoder as resuming lost weights.
- Continue only the already released three corrected full seeds, using the existing50-epoch configuration, frozen manifest and feature caches. The remaining24 specifications are not released automatically. No new matrix, model, metric or split changes during the group.
- GPU0 executes full_s0 then full_s2. GPU1 executes the official RepNet357-video comparison, then full_s1 only after successful baseline completion and a fresh free-resource check. No healthy job is migrated. The prior `full_s1 completed` scheduling dependency may be replaced by explicit user-resume/stop receipts plus actual absent predecessors and an unowned GPU; paused work must not be falsely marked completed.
- Reuse the RepNet adapter's completed bounded review. Review only the new resource gate. Keep official weights/core, all357 inputs, inference settings and scoring unchanged. Installation and runtime failures are repaired in isolation.
- All experimental computation stays on the server; local editing/control/evidence stays on E:. No new scheduled tasks, no other-project interruption, no submission or automatic main-branch merge.

## Acceptance for this continuation

| Requirement | Current state | Completion evidence |
|---|---|---|
| Inherit completed work without rerunning it | In force | Existing29 results/cache manifests and15-entry ledger retained |
| Durable interrupted training | Engineering and bounded review complete; first real epoch persisted | Server fixture and final review in docs/evidence/resume_20261112/checkpoint; actual full_s0 latest/best checkpoint receipt |
| Corrected full model, matched3 seeds | 1/3 complete: full_s0 finished50 epochs and four evaluations; full_s1/full_s2 running | Launch source489dc77; all first durable checkpoints verified. Require all three completions before group aggregation |
| Official external performance comparison | Partial overall; RepNet357 complete | ESCounts released-weight evidence inherited; RepNet official updated checkpoint357-video execution/integrity complete. Broader official comparison coverage remains outstanding; other literature rows remain correctly labelled |
| Every component contributes | Not yet accepted | Use paired multi-seed removals after the current group closes; retain counterexamples and uncertainty |
| SOTA / requested benchmark breadth | Not yet accepted | Same-scope measured external comparisons; no hidden exclusions or changed primary metrics |
| Code/config/results/manuscript correspondence | Run-specific sources verified; result publication in progress | Training489dc77 and completed RepNet8facb65 are separate immutable snapshots. Latest publication and result-manifest sync must be checked against live source_sync.json; the historical rendered paper predates these results |
| Final paper | Pending | Complete evidence, claim/citation checks and all-page main/supplement rendering |

Scientific continuation is assigned to Astra; routine implementation/execution and fresh bounded code review to Sol. Requested model identities are recorded, actual backend identities remain unattested, and same-family semantic assurance remains provisional. Review ends at the group's pre-run gate; runtime monitoring does not reopen method design.

## Second explicit resumption and checkpoint closure

The user resumed again after the brief November12 pause; `docs/evidence/resume_20261112/resume_authorization_2.json` supersedes that pause for the same frozen scope. No new formal training/evaluation had begun before that pause. Partly downloaded RepNet environment wheels remain reusable.

The durable checkpoint repair is complete: atomic latest/best files, optimizer/scheduler/EMA/RNG state, explicit identity-checked resume and historical-output refusal. The existing server-only CPU fixture establishes exact interrupted/continuous equality at three committed boundaries; it does not establish scientific performance or CUDA bitwise equality. Fresh Sol review is closed for the final source hashes, with same-family provisional assurance. See `docs/evidence/resume_20261112/checkpoint/REVIEW.md` and `ENGINEERING_REPORT.json`; no repeat test or scientific review is required at dispatch.

The finite worker is committed as `scripts/continue_known_source_v1.py`. The production source commit, resource ownership, environment, manifest and actual first durable checkpoint are recorded separately at launch. At08:51UTC, s0 has completed50 epochs and all four fixed evaluations; the existing worker has continued to s2, with s1 on GPU1. All use489dc77 and have actual durable checkpoints. RepNet v4 completed357/357 from8facb65, preserving196 earlier predictions and computing only161 remaining rows. See `docs/CURRENT_PROGRESS_20261112.md` and the linked raw receipts. Old29 results and the paused12/13epoch logs are preserved. Complete three-seed aggregation and scientific acceptance remain pending.
