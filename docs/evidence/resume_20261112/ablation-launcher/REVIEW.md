# TempoRAC frozen21 launcher: bounded independent review

Final substantive outcome: **no remaining concrete release-blocking launcher defect found after one targeted repair and recheck**. This is same-family **provisional** assurance, not an accepted scientific or submission verdict. Requested reviewer: gpt-6.1-sol / xhigh minimum; actual backend model and reasoning effort are unattested and recorded as null. No semantic PASS is inferred from the requested identity.

Session: /root/ablation_launcher_review_20261112. Review workspace is outside the shared repository. Root retains Git, source synchronization, resource checks and production dispatch ownership. This reviewer started no production job, model execution, local experiment, GPU toy, scheduler, or repeated checkpoint-durability fixture.

## Final identities

| Artifact | SHA256 |
|---|---|
| ablation_worker.py | d23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144 |
| check_launcher_fixture.py | 7c82577bfad9c89e1bd4365b9afe4e72f43efe9aa7295e008267eb8d48c5d522 |
| Astra RUN_CONTRACT.json, unchanged | dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167 |
| frozen_27_spec.json, unchanged | 5e6750b0ff6a894be85c171e5ae31c150e9cb5288a3cb9ab4df8f0f5984babe4 |
| ENGINEERING_REPORT_v2.json | f065799b0fabd2f2e7f60ed427db37fd253600644ae28d3e0a7fadb7641cbbd1 |
| raw old-worker-to-final worker.diff | ff2b9d9cd9034e13e9ed8d06bf4fc946b1523ec772a2f2c87432a21602444dfe |

The final server receipt SERVER_SHA256_v2.txt matches the locally read worker, fixture, unchanged contract and raw fixture_v2 report. The executor records fixture_v2 actual exit0, session48538 completion964511, under /root/autodl-tmp/TempoRAC/resume20261112/ablation-launcher-work/fixture_v2. This reviewer inspected those transferred artifacts and the production functions they exercise; it did not rerun the fixture.

## Acceptance table and evidence

| Obligation | Status in this bounded review | Evidence |
|---|---|---|
| Fixed21 new arms/1320epochs; seeds0/1/2;50/80epochs;12/9lanes | Pass, engineering scope | checked_plans lines35-69 binds every approved cfg digest, seed, features and splits, exact unique lane membership and epoch sum. Raw fixture has all21 mappings. The3 old singleUCF and3 corrected full runs are reuse entries and are not launched by this worker. |
| Training/evaluation core and source489 unchanged | Pass, static scope | Worker is launcher-only. main requires clean immutable Git HEAD489dc77b91a21d21215784ff49961edb20dab4fb and train.py SHA33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4. Read source489 train/eval_saved/metrics interfaces directly; no core edit was proposed or made. |
| Config, population, feature and evaluation mapping | Pass, engineering/static scope |15 union arms evaluate RC152/151positive, CX2403/2403 and UCF105/105.6 singleRC/CX arms evaluate corresponding test only using best.pt. Union jobs use best_rc_valid.pt/best_cx_val.pt/best_ucf_dev.pt. wo_depth uses UNION_stmae_p4 with no aux; others preserve b8 main plus cfg aux. eval_saved obtains aux from checkpoint cfg, so omission of a CLI --aux override is correct. Runtime checks config, full prediction-key sets, finite both-view counts, n/n_positive and checkpoint name. |
| Validation-only selection and fixed two-view inference; no new QUVA | Pass, static/engineering scope | Frozen train selects original-view validation by (mae_p,-obo_p); launcher omits --with-test and calls unchanged eval_saved only after training. eval_saved averages original/shift counts before unchanged NumPy rounding in score. All new jobs exclude QUVA. Existing confounds/disclosures remain frozen. |
| Explicit identity-checked partial and completed-prefix resume | Pass after targeted repair, provisional correctness judgment | Source/contract/spec/manifest/cfg provenance and completed-file hashes remain bound. --resume is explicit, partial latest.pt uses existing trainer identity check, completed outputs form a prefix. New validate_residual lines90-98 permits only verified forward advancement within saved canonical suffix. Fixture_v2 reproduces the specific stale-state interruption and rejects missing/out-of-order completion. |
| Failure stop, hold-after-current boundary and lane isolation | Pass, engineering/static scope | Production run_lane validates completion before successor/hold, propagates run_one exceptions and records current+unstarted suffix. Hold is read only after training, all fixed evals and durable complete receipt. GPU-specific control/PID/state/hold paths and one sequential child per process keep lanes independent; no process kill/migration or cross-lane action is present. |
| Scope and execution restrictions | Pass for this review | Local actions were static reading/hash/diff inspection and writing this E: review folder. No experimental computation, production launch, training-core change, matrix expansion or scientific redesign occurred. |
| Independently accepted semantic assurance | Not established | OpenAI same-family route; actual backend identity/effort unattested. The result remains provisional and cannot certify science, claims, SOTA, manuscript readiness or actual future dispatch success. |

## Finding and its closure

Initial candidate77572af3 had one blocking recovery defect at lines162-166: saved state remaining had to equal current pending, while terminal state was only updated on held/failed/queue_complete. A held lane could resume, complete another full arm, then be hard-interrupted before a new terminal state. Its valid progress would cause the next explicit --resume to reject the stale state. ROUND1.md preserves the finding and original candidate/fixture identities.

The narrow final repair introduces validate_residual and invokes it in the existing explicit-resume path. It requires saved remainder to be a canonical lane suffix, calls the existing complete function for all names (thereby checking provenance and file hashes), rejects holes, and allows pending only as a suffix of saved. All removed saved entries must therefore be verified complete; no seed/arm can be omitted by moving the remainder forward. It handles empty completion/residual tails correctly. LedgerA8 records the interface choice.

The single targeted recheck read the new helper/call site, revised fixture and ledger, unchanged training/contract identities, raw report and final server SHA receipt. The shared-function fixture covers held resume -> another durable arm -> abrupt BaseException before state advancement -> explicit recovery without repeating that arm; missing/holed completions reject. All original mapping/hold/failure checks remain present and the server report records them true. **Finding closed; no further repair or review loop requested.**

## Workflow receipt and limits

Read E:/AGENTS.md, project AGENTS.md, ARIS_LOCK.md, docs/aris/WORKFLOW.md, dated resume protocols and current status context. The applicable installed implementation-review stage is research-implement-feature Phase4; there is no separately named installed code-review skill. Read its complete SKILL.md and relevant reviewer/acceptance/scope/tracing references. Installed/source skill SHA33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8 matches pinned f76ec8d6b77792e2242cca049e08d0cbb7d54e19; the ten inspected skill/reference copies match pinned originals.

This review closes the requested pre-run launcher gate only provisionally. It does not repeat the closed training-checkpoint durability review, reinterpret scientific confounds, authorize extra QUVA inference or add arms. Root still verifies final publication/source/contract receipt and current free GPU ownership before its explicit finite dispatch. No new approval or test campaign is requested by this report.
