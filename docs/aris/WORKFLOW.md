# TempoRAC CVPR continuation: ARIS project workflow

## Actual continuation — 2026-11-11

Read `docs/CURRENT_PROGRESS_20261111.md` and `docs/KNOWN_SOURCE_TRAINING_CONTRACT_20261111.md` before interpreting the older snapshots below. The active code is `temporac_cvpr/`, inherited from local commit `489d3ab`; the active backend is the isolated AutoDL TempoRAC project. Twenty-nine frozen-framework runs have completed and their saved predictions were rechecked on the server. The old MultiRep control state is not the active-stage gate.

Two RepCount train/test pairs have identical media and feature inputs with conflicting labels; two Countix train/validation source segments overlap. Preserve all old results. A versioned training-only correction excludes four optimization rows while retaining all evaluation rows and all frozen feature caches. Three complete decoder runs are authorized under the written contract; source/data manifests and live receipts determine their actual state. Do not pool old and corrected protocols or claim the known-source correction resolves inherited evaluation-label ambiguity. External literature numbers, official released-weight evaluations and internal custom heads must remain distinctly labelled. No new timer, no local experimental compute, no C:/D: project writes, no submission or other-project interruption.

## Current execution overlay — 2026-11-05

Latest user execution constraint: all experiments and computational validation run on the server only. Local training, model forward, evaluation, preprocessing/cache construction and experiment environment installation are disallowed. Local editing, static reading, synchronization and necessary connection/process control remain permitted. The local CPU validation started earlier in this session was stopped; retain its existing evidence without completing or repeating it locally. Server unavailability does not authorize local compute fallback.

The dated installation text below preserves the initial setup snapshot. Current facts come from `E:\ICASSP_TempoRAC\aris-work\current_status.json`, actual run logs and the active server source-sync receipt. Authentication previously succeeded as `lilei`; subsequent transport failures are not an unresolved username or evidence of job failure.

The user's latest instructions cancelled timed work and clarified the research framing. Automation `temporac-cvpr` was deleted; do not create a replacement scheduler. Continue through active user-directed sessions. TempoRAC remains the research mainline; use related conference papers as the reference for benchmark datasets and scale, the number and coverage of external baselines, metrics, ablations and experimental organization, as recorded in `docs/EXTERNAL_BENCHMARK_PROTOCOL_20261105.md`. External baselines serve whole-method performance comparisons; component effects are tested through ablations. Preserve the original canonical/control pair, caches, checkpoints and negative results; do not automatically release remaining seeds. Current model/data source751 remains frozen while that pair closes. Newly prepared documentation does not establish a new server deployment.

This is a workflow integration, not a new method or a restart of research. The active project is `E:\ICASSP_TempoRAC\TempoRAC-CVPR`; historical TempoRAC directories, results, designs, and rejection evidence remain intact. Read the current project handoff and user policy before making any scientific change.

## Source and installation

- Repository: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git
- Exact source commit: `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`.
- Detached immutable source checkout: `E:\ICASSP_TempoRAC\aris-work\aris-pinned`.
- Project installation: `.agents/skills/`, copied from `skills/skills-codex/`; manifest `.aris/installed-skills-codex.txt` records mode `copy`.
- E: is exFAT. A junction creation returned `Incorrect function`; real copies are used. `INSTALL_VERIFICATION.json` records source/destination hashes. Do not claim junction installation.
- Canonical helper source resolves through manifest `repo_root` to the pinned checkout's `tools/`. Do not fall back to C:/D: or a different project's helpers.
- The official Windows installer was inspected but not applied: it calls `Ensure-GlobalPointer` and writes the home `.aris/repo`. This installation does not modify global Codex/Claude/MCP configuration.
- Local project reviewer configuration is `.aris/reviewer.json`. Its portable equivalent is `reviewer-policy.json` in this directory. No external reviewer has been invoked by this installation.
- Do not edit copied skills as though they were upstream. Re-read pinned originals and compare hashes before each stage. Updates require a deliberately selected new commit, renewed read receipt, and controlled replacement; never `git pull` the active pin automatically.

## User policy overlay (takes precedence over upstream defaults)

1. Routine execution and routine independent review use `gpt-6.1-sol`. Direction, design, major scientific-interface review, and unresolved conceptual disputes use `gpt-6-astra`. Preserve required reasoning tiers: ordinary verdict review at least `xhigh`; deep audit `ultra` when available, otherwise explicit capability fallback to `xhigh`. Do not increase model tier just because an upstream example says Astra.
2. Executor and reviewer are different fresh sessions. Native review starts with `fork_turns: none` and primary artifact paths, no inherited executor narrative. Sol/Sol, Sol/Astra, and Astra/Sol are all OpenAI same-family: positive semantic verdicts are **provisional**, never cross-family accepted. Log requested and actual identity, session, complete scope, raw response, and verdict separately. Unknown identity or incomplete review never becomes a PASS.
3. Continue the existing TempoRAC method and repository. Preserve accepted freezes and old failed routes. Do not reactivate rejected approaches, redesign the model, regenerate all evidence, or retarget the paper merely because an upstream default suggests doing so.
4. User full-data requirements override toy/pilot/debug defaults. Before formal jobs, use only necessary CPU checks for schema, labels, split disjointness, metric interfaces, configuration, and launcher correctness. Do not launch small GPU training as a substitute for complete experiments or call engineering success a scientific result. GPU environment readiness remains unverified until the real required validation is performed in the authorized scope.
5. Stage contracts explicitly name primary metric and direction, exact datasets/splits, comparison population, seeds, validation-only selection, frozen method/config, resource budget, success/failure criteria, and evidence paths. Keep all raw negative results and selection history. Never drop a stronger baseline or a non-significant result to manufacture success. Generic upstream three-baseline or page-count defaults do not override the user/project contract.
6. CVPR is explicit. Use `venue: CVPR`, `anonymous: true`, and `assurance: submission` when writing or auditing the submission. The actual target-year official page limit, template, supplementary rules, anonymity, disclosure, and citation requirements must be checked and frozen before writing; do not import generic IEEE or ICLR assumptions. Final Word/PDF changes require all-page rendering, including appendices; maintain white/black three-line tables and dark red `#B00000` relative improvements.
7. Workflow and research deliverables stay on E:. No new C:/D: project artifacts, no global configuration modifications. Windows background roots use hidden/no-console creation; Python subprocess flags preserve existing flags and include CREATE_NO_WINDOW for noninteractive jobs; Node uses windowsHide. Do not interrupt healthy jobs to adjust display behavior.
8. No scheduler, automation, Goal Pulse, external notification, publication, submission, or push is started by installing this workflow. These remain governed by the current user authorization. An explicit ARIS research action authorizes its necessary child skills/reviewers, not arbitrary external communication.
9. Reuse synchronized artifacts by a common source SHA and manifest. The user identifies the endpoint as a **new server**: its username, environment, GPU, data, and project path are unverified. `/media/lenovo/data2` and any old host/GPU/account setting are historical evidence only, never deployment defaults. Before server deployment, verify actual remote path, access, clean/dirty state, current job ownership, commit, and data/config provenance. Local, GitHub, and server synchronization are three separate receipts; never mark all three complete when only one or two are verified. Server authentication is unresolved; full deployment and data migration are not complete.

## Stage handoffs and gates

Reading a skill is preparation only. The statuses below do not upgrade old experiments or reviews.

| Stage | Executor artifact and evidence | Gate / reviewer | Current integration status |
|---|---|---|---|
| Existing-project intake | Current branch/SHA, handoff, method freeze, raw evidence inventory, requirement acceptance table | Reuse/retire decisions grounded in original files; design questions to Astra | Workflow prepared; scientific intake owned by main task |
| Literature + novelty refresh | Source table, canonical paper IDs, exact search/date coverage, closest works and testable delta | Fresh reviewer; actual sources required; no inference from paper title | Skills read/installed; refresh not run here |
| Experiment contract | Claim → evidence → run matrix; primary/secondary metrics, complete splits, baselines, seed/budget plan | Astra for major design; no silent scope cut or changed method | Skill read/installed; contract not replaced here |
| Implementation | SPEC/ASSUMPTIONS/BUILD_NOTE; changed-file diff and CPU checks | Fresh Sol code reviewer, Astra for material scientific interface; blocking issues fixed | Skills read/installed; no scientific source changed |
| Formal execution | Immutable command/config/source/data, launcher PID/job identity, logs, raw JSON/CSV, completion status | Deterministic job completion only; no semantic performance PASS | Skills read/installed; no experiment launched |
| Training monitoring | Actual run/log path, metric/NaN/divergence evidence, decision | Preserve ownership and distinguish unreachable from failed | Skill read/installed; no monitor started |
| Integrity + analysis | Raw tables, mean/std/uncertainty, exact deltas, integrity report | Fresh artifact-grounded integrity reviewer; result-to-claim verdict yes/partial/no | Skills read/installed; no old result reclassified |
| Ablation and revision | Removal vs replacement separated, contribution/variance evidence, frozen budget, full negative history | Independent reviewer; no significance/efficacy inference from absence of significance | Skills read/installed; not executed |
| Auto review | AUTO_REVIEW, persistent reviewer memory, raw traces, append-only receipt | score >=6 AND ready/almost only ends drive loop; same-family remains provisional | Skill read/installed; not started |
| Paper plan/write | Evidence-complete NARRATIVE_REPORT, PAPER_PLAN, frozen acceptance contract, current official CVPR rules | Claims-evidence mapping; no missing evidence disguised as prose | Core skills read/installed; paper not rewritten |
| Final assurance | Fresh paper-claim and citation audits; proof audit or explicit NOT_APPLICABLE; actual rendered pages | External `verify_paper_audits.sh`; failed/missing/stale audit blocks; same-family output at most provisional | Core audit skills read/installed; no submission readiness claim |

The pipeline orchestrator is a routing reference. This continuation does not invoke Stage 1 idea discovery again by default. `AUTO_PROCEED=true` removes redundant selection questions within the authorized scope; it does not authorize method changes or fabricate unmet acceptance criteria.

## Conditional child skills: read in full before activating

The 20 installed stage skills are recorded as fully read in READ_RECEIPT. Additional child skills are deliberately **not marked read or installed**: `idea-discovery`, `idea-creator`, `research-refine`, `experiment-queue`, `paper-figure`, `figure-spec`, `paper-compile`, `auto-paper-improvement-loop`, `proof-checker`, `kill-argument`, `render-html`, `research-wiki`, and `integrity-forensics`. Activate only if the current authorized stage needs them, after full SKILL.md plus required support files from the same pinned commit are read and added to the receipt. This is not permission to silently skip a mandatory writing audit. `paper-writing` may not be declared completed until all applicable child stages are actually performed.

Shared `external-cadence.md`, `fan-out-pattern.md`, and `skill-governance.md` remain conditional before those features are used. No cadence/fan-out worker or skill rewriting was run during setup. Existing-project synchronous work can proceed without them.

## Recovery and synchronization

Every future stage records `pending/running/done/provisional/accepted/failed/skipped` accurately. `done` is output existence, not scientific acceptance. Save a new run id when starting a real workflow; do not fabricate one for installation or overwrite historical `.aris/runs` records. Resume from the first outstanding gate and reuse completed evidence.

Before sync, inspect status on each endpoint. Transfer only intended source/config/docs and evidence manifests; never sweep private data, credentials, caches, or another project's files. Do not launch a remote job until source/config/data identities and the authorized budget match. Archival/cleanup is a separate end-of-project action with verified backup and recovery; not an installation step.
