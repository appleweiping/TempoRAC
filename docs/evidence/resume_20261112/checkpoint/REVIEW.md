# Bounded checkpoint review — 2026-11-12

Verdict: **no remaining concrete blocking finding in the bounded checkpoint/resume patch or the single finite-launcher static pass**. Scope closed. Same-family semantic assurance is **provisional**; actual backend/effort is unattested. This is implementation assurance only.

## Acceptance and evidence

| Requirement | Status | Evidence |
|---|---|---|
| Preserve frozen training, loss, data, sampling, selection and evaluation behavior | Pass (static) | Diff from `25b8c479d59e7d56476ea69fe4c505b33747d415` limits trainer changes to persistence/resume and overwrite guards. Training loop computation is unchanged. |
| Persist model/optimizer/scheduler/EMA and relevant randomness at a completed epoch | Pass within recorded CPU runtime; CUDA state statically covered | `train.py:58`, `train.py:450`, `train.py:551`; final fixture compares model, optimizer, scheduler, EMA, best states, RNG and outputs exactly. BF16 uses no scaler, saved as `None`. DataLoader has no private generator; default CPU RNG is captured. Deterministic collate needs no worker RNG checkpoint. |
| Atomic latest/best writes retain the prior checkpoint on failed serialization | Pass (static + focused fixture) | Same-directory temp file, file flush/fsync, atomic replacement and directory fsync at `train.py:27`; best selection materialized before latest commit. Injected partial serialization preserves the prior loadable checkpoint and removes the temp file. |
| Recover log/best state at the committed boundary | Pass (static + focused fixture) | `train.py:470` rejects a log shorter than the checkpoint, truncates uncommitted tail, reconstructs or deletes best files from committed state. Fixture appends a fake epoch999 and inserts a stale best before resume; final logs/bests match continuous training. Short-log refusal is statically inspected. |
| Reject config/manifest/source drift, completed output and uncheckpointed historical output | Pass (static + focused fixture) | Nine drift refusals, three completed-run refusals, and two historical-partial refusals have expected nonzero exits. Protocol includes resolved cfg, seed, split/test policy, feature-root path, ordered manifest byte identities, package Python source identities, Torch/CUDA versions. |
| Separate uninterrupted versus genuinely terminated/resumed CPU training agrees | Pass within recorded CPU runtime | Final server fixture exits0; children exit73 after committing epochs0/1/3, then resume at1/2/4. Tensor/optimizer/RNG comparisons use exact equality; losses, validation selection and predictions agree. All21 raw child logs and the fixture command/completion receipt were read. No reviewer local runtime or experiment executed. |
| Finite launcher releases only corrected seeds 0/1/2 under specified resource ownership | Pass (one static pass plus bounded ownership-guard follow-up); production launch not performed by reviewer | Final worker restricts GPU0 to full_s0→full_s2 and GPU1 to full_s1; clean source/commit, frozen spec/manifest, fresh group, explicit provenance-matched resume, live-trainer refusal, failure stop, first checkpoint observation,50 epochs, and fixed evaluation counts152/2403/105/100. Before each run, the selected GPU must have memory below500MiB and an empty compute-PID query. Root must explicitly release GPU1 after full RepNet success and verify actual ownership/environment at dispatch. |
| Fresh executor/reviewer separation and honest assurance | Pass, provisional | Reviewer `/root/checkpoint_review_20261112` differs from executor `/root/durable_checkpoint_20261112`; actual backend identity unattested. Same-family positive review remains provisional. |

## Scope and limits

Inputs are the pinned ARIS stage instructions, project policy, final implementation SPEC/ASSUMPTIONS/BUILD_NOTE, the trainer diff, the server-only fixture plus raw logs/receipts, the final finite launcher, its prepared launch/environment records and the inherited frozen full-seed specs/preflight. No architecture, literature, method, metric, benchmark or scientific-claim re-review. No new experiment matrix; no local tests/forward/evaluation/dependency installation; no GPU experiment; no active/historical source mutation. Lost old RAM-only epochs remain unrecoverable.

The deterministic fixture can establish equality only in its recorded CPU runtime. CUDA RNG restoration is statically covered; GPU kernel bitwise determinism and scientific performance are not established. Identity is intentionally an input/source/runtime compatibility check, not a rehash of frozen feature caches. Runtime identity currently records Torch/CUDA, not every installed dependency. Mid-epoch recovery is outside the declared interface.

## Findings and closure

No concrete blocking defect found. The finite worker was added after the original trainer ledger; the author recorded its scope and interface in A007–A009 and updated F0–F2 from pending to actual server evidence. No semantic assumption gap remains in this scope, and no additional runtime check was needed. The synthetic fixture's tiny CPU head is explicitly labelled and never used as scientific evidence.

This closes the pre-run implementation review for the hashes below. Formal runs remain the root task's execution responsibility; source freezing, actual GPU ownership, project-contained cache environment and the post-RepNet GPU1 release are dispatch conditions, not another scientific review campaign. Do not reopen this review during full runs absent a concrete runtime fault or changed reviewed source.

### Final closure delta after interrupted bookkeeping

The root identified that a memory-only resource guard could permit a small existing CUDA process. The bounded worker-only follow-up at lines122–124 was checked before the pause: `nvidia-smi -i <requested GPU> --query-compute-apps=pid` must return no PIDs, alongside the existing memory guard. This refuses a conflicting GPU; it has no process-kill path. The reviewed final worker is `e8844834850bfdb749d6240d82c2541e9c82d796891fe84dfebba6d98fd009d4`, matching the already-read local hash and raw server receipt. It supersedes worker `488714e4...` in the original response. Trainer, test and fixture report hashes are unchanged.

User resumption is recorded in `E:/ICASSP_TempoRAC/aris-work/resume-20261112/resume_authorization_2.json` (observed2026-11-12T06:58:41.2933466Z). This update only completes the interrupted review records. The original `001-bounded-review.response.md` remains intact; `002-final-closure-delta.response.md` records this delta. No test, scientific review, source edit, commit, training or launch was repeated. Verdict remains no remaining concrete blocker, scope closed, same-family provisional with actual backend/effort unattested.

## Server evidence

- Exact command, isolated server cwd, observed tool session74247 completion exit0/chunk148283: `../FIXTURE_COMMAND_RECEIPT.json`.
- Final parent report/stdout: `../ENGINEERING_REPORT.json`, `../fixture_v2.stdout.log`; all21 child logs: `../raw-logs/`.
- Raw SSH hash receipt: `../SERVER_SOURCE_HASHES.txt`; local static hashes independently agree with the server receipt.
- Existing server runtime recorded by author: Python3.10, Torch2.5.1+cu124, NumPy1.26.4. No installation or rebuild.

## Artifacts reviewed

- Base: `25b8c479d59e7d56476ea69fe4c505b33747d415`.
- `temporac_cvpr/train.py`: SHA256 `33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4`.
- `tests/check_checkpoint_resume.py`: SHA256 `2deb43c05acaacefac76a9117ec1388a11f6f7cf0fbdbaaf20f1fc169bcffd34`.
- `continuation_worker.py` (final ownership guard included): SHA256 `e8844834850bfdb749d6240d82c2541e9c82d796891fe84dfebba6d98fd009d4`.
- `ENGINEERING_REPORT.json`: SHA256 `d8d06b61f5299b5fb503006a2cd085d2aae940e632d8c1be03bafc6bdd5d819b`.
- Inherited corrected manifest: SHA256 `96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2`; frozen27 spec: SHA256 `5e6750b0ff6a894be85c171e5ae31c150e9cb5288a3cb9ab4df8f0f5984babe4`. Only the initial three are released.
- ARIS source pin verified: `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`.
