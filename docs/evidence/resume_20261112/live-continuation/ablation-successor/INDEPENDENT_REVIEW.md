# Independent engineering review — one-shot GPU0 successor

Completed 2026-11-12 18:20:23 UTC. Verdict: **clean static review; same-family provisional**. No concrete in-scope engineering finding, producer/consumer mismatch, undeclared consequential assumption or stale assumption row was identified. This is not a runtime PASS or execution authorization.

Requested reviewer: native `gpt-6.1-sol`, `xhigh`, fresh session `/root/ablation_successor_review_20261112`. Actual model and effort were not authenticated or returned and remain unknown. The project/caller declares an OpenAI same-family relationship; positive semantic assurance remains provisional. Review completed and the requested pre-execution static scope was covered.

## Scope and source

Reviewed only the new unexecuted `scripts/resume_gpu0_after_transrac.py`, its packet and the original producers it consumes. Scientific merit, frozen settings, active jobs, hardware assignments and completed input/model checks were not reopened.

The repository HEAD observed was `59bec947fbe7a6afc63efe847b868c6b355df0c6`; the candidate was untracked. The actual pinned ARIS checkout was `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`. The pinned `research-implement-feature/SKILL.md` and reviewer independence, scope, backend/routing and tracing references were read in full. All requested packet artifacts and project policy files were read directly. The candidate source, evidence copy and packet digest agree.

| Reviewed source | SHA256 |
|---|---|
| `scripts/resume_gpu0_after_transrac.py` | `6116d44c3bff0cdeaf22854eb123c537f06a9aaf5333edcd9504406d109aa0cb` |
| `scripts/continue_known_source_ablations.py` | `d23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144` |
| `scripts/external/transrac_native/native_run.py` | `514cad57fa517ef718ba258e4d477e55dcbb48c589463d893732901b7b71a964` |
| `scripts/external/transrac_native/native_inputs.py` | `79fb206356e29183464cedb5e1197f51dcd5517991dcd6d9fbe67c4153d9c0b9` |
| `scripts/external/transrac_native/run.py` | `d491b40a0ef7d6379b299d1b3720fbb5a11c233d3cab395a9aad0c5a062b8ae8` |

The reviewed TransRAC source identities match the actual saved preflight and launch code map. No new identity mechanism is proposed.

## Findings and acceptance evidence

Findings: **none identified**. Undeclared rows: **none identified**. Stale rows: **none identified**. Semantic undeclared count: **0**.

| Requirement | Assessment | Evidence |
|---|---|---|
| Resume exactly the held ten-arm suffix and reuse completed prefix | Pass, static | Candidate lines 68–110 bind original controls, call unchanged `checked_plans`/`validate_residual`, verify complete-prefix provenance and output file sets, require exact `RESIDUAL`, and reject any existing pending directory. Original worker lines 35–98 and 153–175 agree; saved held lane has the same suffix. |
| Wait for full200/full357, all frozen evaluations and exit | Pass, static; runtime unexecuted | Candidate lines 129–168 wait on the original PID, acquire the existing run lock, check full completion/accounting, reject a failed completion attempt, require writer absence, 50 exact validation candidates and every final record. Producer lines 295–327 and 373–490 supply those records and hold the lock around the complete run. |
| Match the actual producer schema and permit normal completion | Pass, static; runtime unexecuted | Producer lines 484–489 emit exactly the completion fields consumed at candidate lines 137–159. `native_inputs.py` lines 18–26 and 28–58 use the same manifest, population sizes and `population/video_id` key. Producer lines 251–260 and 468–481 supply the consumed scalar/density/selection/checkpoint/member fields. No mismatch was found. |
| Release only genuinely idle physical GPU0 | Pass, static; runtime unexecuted | Candidate lines 169–178 compare GPU index/UUID to START, require memory below 500 MiB and an empty XML process list. Query/parse failures block; no signal command exists. The unchanged native worker also rechecks GPU memory/processes at lines 222–224. |
| Prevent duplicate workers and automatic failed-run restart | Pass, static; runtime unexecuted | Candidate lines 67, 108–112 and 187–208 combine frozen held history, absent pending directories, exclusive execute lock and persistent exclusive one-shot claim. Every extant claim blocks future calls. Original worker lines 71–88 and 177–208 stop at child failure; no retry loop is introduced. |
| Preserve completed outputs, checkpoints, HOLD bytes and historical events | Pass, static; runtime unexecuted | Candidate lines 198–204 rename and verify the original HOLD bytes and create a separate handoff event. Original events remain intact and the unchanged worker later appends them. The candidate does not write training outputs/checkpoints. |
| Current read-only WAIT with no control mutation | Pass by inherited receipt; not rerun here | `READ_ONLY_CHECK_RECEIPT.json` binds this exact candidate and records 18:14:05 UTC WAIT on PID 490003 plus identical seven before/after control digests. It explicitly leaves completion/READY/execute branches unexercised. |

The operator exclusivity assumption between GPU checking and child start is explicitly declared in `EXECUTION_BOUNDARY.md`, together with the native start-time recheck. The persistent claim intentionally makes every interrupted consumed attempt manual recovery, even when a crash may precede spawn. `WORKER_EXITED_OK` explicitly means child return code; a later manual HOLD can still stop a lane early, and native lane/events remain authoritative. These are declared operational choices consistent with the code.

## Execution limits and remaining branches

This reviewer executed only local read-only text/JSON/hash/Git inspections. No SSH command, candidate `--check`/`--execute`, training, inference, model import, checkpoint load, environment installation, test suite, process signal, input scan or cache rebuild was run. No source, control, Git, `current_status` or `source_sync` edit was made. Only these two requested review output files were written.

Commands used were PowerShell `Get-Content` (including numbered source reads), `Get-ChildItem`, `Get-FileHash -Algorithm SHA256`, JSON field reads with `ConvertFrom-Json`/`ConvertTo-Json`, `rg --files`, a targeted `rg -n` for producer key/entry functions, and the following exact Git reads:

```text
git -C E:\ICASSP_TempoRAC\TempoRAC-CVPR rev-parse HEAD
git -C E:\ICASSP_TempoRAC\TempoRAC-CVPR status --short
git -C E:\ICASSP_TempoRAC\aris-work\aris-pinned rev-parse HEAD
```

An initially guessed `verdict-trace.md` filename was absent; the actual `review-tracing.md` was then read in full. A PowerShell JSON read of historical differently cased status keys failed and was retried with `-AsHashtable`; it changed no artifact. These were inspection errors, not candidate failures.

Unexecuted: PID-exited/run-lock-busy handling; missing/inconsistent completion/failure receipts; full normal completion/member/checkpoint validation; GPU busy/error/identity/idle READY branches; exclusive claim, archive, spawn and child wait; duplicate execution; filesystem/process interruption at each persistent state boundary; native child failure or a later operator HOLD. The source review found no concrete blocker in these paths, but their behavior has not been dynamically demonstrated. The earlier WAIT receipt must not be described as having exercised them.

The machine-readable companion contains the same verdict, exact reviewed identities, scope, acceptance evidence, assumptions and limitations. No fix is requested from this review.
