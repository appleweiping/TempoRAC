# Build note

ASK=never; effort=balanced; explicit scope overrides generic pilot/repeated-review defaults. Frozen source/model semantics come from Astra's contract. No local experimental computation.

| Rung | Feature | Server acceptance command | Tier | Status |
|---|---|---|---|---|
| F0 | CLI to immutable spine receipt | `python run.py --phase spine` | MUST | PASS, server exit0 |
| F1 | Native virtual-replica/scheduler/epoch-resume CPU primitives | `python run.py --phase primitives` | MUST | PASS server arithmetic/RNG/scheduler and serialized epoch resume; no GPU |
| F2 | Full input binding and raw sampler/target checks | `python run.py --contract RUN_CONTRACT.json --phase prepare --build-rgb-cache` | MUST | PASS full1244 input receipt committed2026-11-12 14:57:30.346UTC |
| F3 | Exact backbone graph/coverage and environment | `bash cpu_preflight.sh` | MUST | PASS exit0, CPU_PREFLIGHT committed14:58:39.557UTC |
| F4 | Full finite trainer/evaluator and source review | root's bounded fresh Sol review plus static/server checks | MUST | PASS engineering; review/readiness clean15:03:19.001UTC,0blockers,same-family provisional |

## Run record

Implementation isolated under E:/ICASSP_TempoRAC/aris-work/resume-20261112/transrac-native-implementation and server baselines/transrac_native_v1. Shared repo/running jobs remain untouched.

Base publication4c5cc763ce9c33148a118ee574cca3baacc5db05. Pinned research-implement-feature SHA33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8 and run-experiment SHA8325128eab14c7f82213d44c8e78e98f44ca680baa9c78f3bee8766f8e9f464a match pinned originals, fully read; required compute-env, acceptance, independence, routing/backend, effort/output, capture/injection/scope references read.

F0 server exit0 and checks/spine.json exists. F1 server existing Torch2.5.1+cu124 CPU fixture: B32 four8/B20 four5 loss/gradient/Adam/primaryBN maximum absolute errors all0; shared BN invoked3 times; independent RNG restore passes; full200 scheduler sequence passes. Production atomic_torch/capture_rng/restore_rng epoch fixture passed exact model, Adam, scheduler, Python/NumPy/Torch/data generators. This is engineering evidence only.

Python311 declared attempt log preserved; only own pip438023 terminated after measured125KB/s exact Torch wheel transport. Root authorized native310 read-only base reuse; Astra contract1b986ab freezes amendment. Own overlay setup exit0, original official models.TransRAC import exit0 with no patches or fake modules. Main base env never upgraded/uninstalled. Exact imported module versions are2.5.1+cu124/.20.1+cu124 although installed dist metadata omit local +cu124 suffix; both required by final checks.

Code frozen for root's one bounded fresh Sol review; review gets actual files/contract, no semantic verdict supplied. Code allowlist run.py/native_engine.py/native_inputs.py/native_run.py/env-spec.json. No official TransferModel forward executed, GPU allocation or formal launch.

## Root review receipt interface

`ENGINEERING_REVIEW.json` is root-written after completed bounded review: `{status:"clean" or "PASS", complete:true, contract_sha256:<exact>, code:{<the five filename:SHA entries>}, requested_model, actual_model, actual_effort, session, review_independence:"same-family", acceptance_status:"provisional", raw_review_path}`. Launch checks completion plus exact code/contract binding; CPU_PREFLIGHT must also match all actual source/input/environment identities.

## Direct formal launch (root only)

`nohup env TMPDIR=/root/autodl-tmp/TempoRAC/tmp/transrac-native XDG_CACHE_HOME=/root/autodl-tmp/TempoRAC/cache/transrac-xdg XDG_CONFIG_HOME=/root/autodl-tmp/TempoRAC/cache/transrac-config TORCH_HOME=/root/autodl-tmp/TempoRAC/cache/transrac-torch CUDA_CACHE_PATH=/root/autodl-tmp/TempoRAC/cache/transrac-cuda HF_HOME=/root/autodl-tmp/TempoRAC/cache/transrac-hf MPLCONFIGDIR=/root/autodl-tmp/TempoRAC/cache/transrac-mpl PIP_CACHE_DIR=/root/autodl-tmp/TempoRAC/cache/transrac-pip PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_REQUIRE_VIRTUALENV=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /root/autodl-tmp/TempoRAC/envs/transrac-native310/bin/python -u /root/autodl-tmp/TempoRAC/baselines/transrac_native_v1/run.py --contract /root/autodl-tmp/TempoRAC/baselines/transrac_native_v1/RUN_CONTRACT.json --phase train-evaluate --gpu N > /root/autodl-tmp/TempoRAC/baselines/transrac_native_v1/formal.log 2>&1 < /dev/null &`

Only after root's manual full-arm/eval/receipt handoff. Exact explicit `--resume` resumes committed epoch and reuses matching per-video final records, preserving all failed attempt logs. Failure exits without retry or alternate configuration. First actual formal complete epoch emits cost receipt; validation cost remains unknown until first scheduled full validation.

## Deferred

GPU witness and complete formal200/full357 are root-launched only after full current-arm boundary. They are not CPU-preflight claims.

## Initial blockers (historical; resolved by final CPU receipts)

At start backbone exact bytes, full RGB preparation and final CPU preflight were pending; no mismatching file was loaded. They are now complete as documented below. Same-session normal receipt closeout finished15:03:19.001UTC. Root still owns publication, review gate file and manual GPU handoff.

R1 bounded review finding and repair: worker default_collate's typed shared out fails real int64 empty-bound-first mixed target batches; exact seed1 first train trigger epoch68/batch10, val epoch68/batch3. Raw checks/target_collation.json preserved. Reviewer requested a single repair in collate_native and both DataLoaders; applied explicit torch.stack targets while keeping default RGB/indices collation. Source sampler functions, RGB bytes, target values and native integer-only dtype unchanged. E08 declared before this repair. Targeted actual-cache worker closure pending required sidecars; then same selected reviewer closes R1, no second generic review.

R1 closure actual cached-target worker fixture passed all5 cases, including actual epoch68/batch10, without RGB/model/GPU. Raw repair receipt SHA30cd6f37083d24d69163d77beb18c13e7f74e85d48b9a9fe3a8af78e8e5d99a7. Runtime temp/cache variables above keep new artifacts underR without overriding HOME. The completed install process had a temporary child-process HOME redirect; that instruction mismatch is retained here and removed from the saved installer for future use. No shared/global HOME configuration was edited.

## Final CPU acceptance — actual artifact commit times

Full inputs were atomically committed2026-11-12 14:57:30.346067UTC; normal CPU_PREFLIGHT at14:58:39.557472UTC (later read/collection timestamps are not completion time). The once-run foreground shell exited0. No official model forward, GPU kernel/training pilot or formal training was run.

| Requirement | Outcome | Actual evidence |
|---|---|---|
| Exact full corrected input population | PASS |1244: RC756/131/152, UCF105, QUVA100; inputs/IDENTITIES.json SHA9341a1b74120c5fd12172a1c8af1ecdba32403db32a6392c86a6343ba3e6882d |
| Original raw RGB/target interfaces | PASS | checks/sampler.json SHAf0af24f82585aaddad9364d4a95e9aa6616715130c03dc718ba157178ffc5f6a; longN1441/zeroN275 exactly match official tensors and targets, short11 and inclusiveROI2..8 exact |
| Original backbone coverage | PASS |183 state entries/29,694,438 parameter+buffer elements exactly equal verified9950…76acc weight; checks/backbone.json SHA006a9b6b0bc002e9439693f3eb79b4cf7749fbf32bab6193b6f0521bced9064f |
| Virtual replicas, scheduler, exact epoch recovery | PASS engineering scope | Actual raw primitive/resume receipts and normal CPU_PREFLIGHT; B32/B20 arithmetic errors0, all200 rates and state restore verified |
| Native zero-target worker compatibility | PASS | Same selected review R1 closure and actual-cache5-case receipt; no cache/sampler/source-model changes |
| Exact complete CPU binding | PASS | checks/CPU_PREFLIGHT.json SHAdd6b2810f5913997fca7396a94e5b112a092eca28c2042ba1dd737809fb0d898 binds fivecode files, contract, official479tracked-file hashes, input and environment |
| Server-only scoped temp/cache and actual origins | PASS | checks/RUNTIME_SCOPED_ENV.json SHA60740d1aefb3a9afb496f4dc466949ea43cefad6d1f1b435b2dc64b80e275457; separate module-origin receipt; HOME unchanged for preflight |
| Full200/full357 and baseline performance | NOT RUN | Root-owned formal launch after same-session receipt closure, publication and manual full-arm boundary; no scientific result claimed |

Native target count-difference ledger has339 rows including the preserved null scalar case; this is the unchanged original target behavior, not a corrected target or synthetic annotation. Raw label/manifests/CSV/media remain only in the authorized local/server evidence. Publish compact inputs/FULL_INPUT_SUMMARY.json (counts/hashes only), code and nonrestricted normal CPU receipts, without raw per-video label manifests.

The original14:34 candidate BUILD_RECEIPT is retained as BUILD_RECEIPT_20261112_1434_candidate.json. Final BUILD_RECEIPT.json records completed CPU checks and remaining root actions. No further tests/audits were started after normal preflight.

Final same-session review/readiness receipt: ../transrac-native-review/REVIEW.json SHA92687026778bb7ee87cbcc7667d3616cf5ae28e7d178b3d83b282821c5181f03; REVIEW.md SHA24839b18f0f872bdbd5040c3d446bcc356f9279c7458e71b169838f4d79e5eec. Complete=true,0engineering blockers, semantic_undeclared=0. Requested Sol/xhigh; actual backend identity/effort unattested. Same-family positive verdict remains provisional, not scientific acceptance. All implementation MUST rungs' engineering checks have passed; formal full200/full357 remains unrun.

## R2 actual startup fault and serialization-only supersession —2026-11-12

First formal attempt20261112T155146Z-c79b9363 failed15:51:48UTC at identity equality before START/resource/model/step. Original formal.log, LAUNCH_RECEIPT if present, FAILURE and prior preflight/review/source are preserved under repairs/identity-serialization-20261112/prior. No checkpoint exists and no GPU was initialized by this repair.

Typed metadata diagnosis found126 unequal live tuple versus JSON list distribution pairs. Complete live/stored JSON canonical identity bytes were identical SHA98acdabc70a8ec7007c8d57389d9beae5e9312ca588fb147a11f271c2e130ce9. One additional TorchVersion/str subclass difference was already Python-equal. E09 declared before repair. Exactly one native_run.py line now constructs sorted list pairs; all package facts and every scientific/source/data/model/metric/contract setting remain unchanged. This also keeps checkpoint/final-record JSON identities compatible at their unchanged equality gates.

Targeted actual binding/JSON-roundtrip comparator passed15:57:50.187135UTC, without full1244/sampler/backbone/primitive/resume/model checks or GPU. New native_run SHA514cad57fa517ef718ba258e4d477e55dcbb48c589463d893732901b7b71a964; other4code files unchanged. Rebound CPU_PREFLIGHT SHAca79905b34591587ec9899660140550dd6b18e815942e8f1c87587875777955d explicitly supersedes olddd6b2810f5913997fca7396a94e5b112a092eca28c2042ba1dd737809fb0d898 only for changed code identity, retaining the original completed test evidence and completion time in provenance. Rebinding is not a fresh full CPU test pass. Raw diagnosis/delta/REBIND_PROOF are in the repair packet; independent bounded R2 closure and root publication/gate/fresh launch remain required. No launch by executor.
