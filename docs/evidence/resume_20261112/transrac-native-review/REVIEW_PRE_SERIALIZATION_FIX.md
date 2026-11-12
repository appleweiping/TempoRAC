# TransRAC native full-training engineering review

Final bounded engineering/readiness verdict: **CLEAN, COMPLETE — 0 remaining engineering blockers; same-family provisional.** Full normal CPU/input receipts are now read and bound to the repaired final source. Root publication/synchronization, final gate and manual complete-arm GPU handoff remain dependencies. Formal GPU forward/training is untested; no scientific-performance acceptance is implied. Initial and code-closure snapshots are preserved separately below and in REVIEW_INITIAL / REVIEW_CODE_CLOSURE files.

Initial verdict: **BLOCKED — one concrete code defect (R1)**. This is one fresh bounded Phase 4 review, followed by one targeted closure after repair. Requested reviewer `gpt-6.1-sol`, reasoning `xhigh`; actual backend model/effort are not attested. Native reviewer session `/root/transrac_native_review_20261112`; same-family, provisional. This is not scientific acceptance or GPU readiness.

Review target: isolated candidate `E:/ICASSP_TempoRAC/aris-work/resume-20261112/transrac-native-implementation`, frozen 2026-11-12 14:34:25 UTC; contract SHA `1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a`. No shared repository edits, local experiment, GPU allocation, training probe, process interruption or launch was performed by this reviewer.

## Acceptance checklist

| Requirement | Initial status | Direct evidence / limit |
|---|---|---|
| Exact frozen source and scientific contract | PASS | All five supplied hashes match; candidate RUN_CONTRACT exact SHA; SPEC/ASSUMPTIONS/BUILD_NOTE and definitive designer contract read directly. |
| Official graph/raw sampler/target/internal autocast retained | PASS static | Original 68bdd4d loader, target, model, training and test paths read directly; native adapter imports unchanged TransferModel and original preprocess. Actual full loader comparisons pending. |
| Four native shard BN/weighted gradients/RNG and fixed budget | PASS bounded | native_engine.py read directly; existing server primitives.json: B32/B20 loss, gradient, Adam and primary BN max errors all 0, scheduler 200 rates and separate RNG restore PASS. No GPU parity claimed. |
| Real input population and batch interface | BLOCKED | R1: int64 empty-bound target at first position of mixed worker batch triggers typed-output collation failure; actual seed1 train epoch68/batch10. Full1244 receipt also pending. |
| Validation-only selection and full357 native/common evaluation | PASS static | Every epoch51..199 validation; candidate51,54,...198; canonical mae_p/obo_p/epoch key; exact full role sets and single selected checkpoint. Native and common arithmetic separated. |
| Atomic epoch/best/final-record recovery and preserved attempts | PASS bounded | native_run.py transaction/resume read directly; existing server epoch_resume.json proves CPU primitive exact model/Adam/scheduler/RNG/generator recovery. Final GPU crash witness not claimed. |
| Read-only runtime isolation and exact backbone | PARTIAL | Static own-prefix overlay, runtime prefix/package gates and exact asset READY receipt 9950e3be...76acc read; imported backbone state coverage/CPU_PREFLIGHT not yet complete. |
| Release and actual runtime readiness | PENDING | One targeted closure, normal full input/sampler/backbone CPU receipts and root-only complete-arm lane handoff remain required. No production launch authorized by this review file. |

## One consolidated repair list

**R1 [P1, interface; release blocking] — dtype-dependent worker collation aborts the frozen full run.** At native_inputs.py:178 an empty-bound source target remains int64 while Gaussian targets are float32, exactly as the native constructor infers. Both production DataLoaders use the default worker collator (native_run.py:223–226 and 379–380). In a worker, its shared stack output is allocated from the first target dtype; when that target is integer and the same batch contains float targets, PyTorch rejects the stack with `torch.cat(): input types can't be cast to the desired output type Long`. The actual retained train row `stu6_44` is index583, and the frozen seed1 sequence first puts it at the head of a mixed batch at zero-based epoch68/batch10. This is a deterministic full-run failure, not a hypothetical malformed input.

Minimal repair: use the existing collate_native interface to default-collate inputs and indices, stack targets explicitly without a first-item typed `out`, and pass this function to both production DataLoaders. Preserve single-item/all-integer target int64 and normal mixed-batch float32 promotion. Record this interface choice as `Source: sweep`. Do not alter original model, target values, sampler, input cache or scientific contract.

Targeted closure: server CPU worker integer-first, float-first, all-integer and actual trigger-batch target collation; changed lines and final hashes; then read the normal pending full1244/sampler/backbone CPU receipts. No repeated primitive suite/model import/GPU probe is requested solely for this repair. Raw witness `W/checks/target_collation.json` and helper `W/checks/check_target_collation.py` are preserved; local copy pending when this initial report was written.

No second concrete release-blocking static defect was found in the bounded graph, optimizer, selection, durability or membership review. Positive static judgment remains same-family provisional.

## Workflow read receipt

Read E:/AGENTS.md, project AGENTS.md, ARIS_LOCK.md, docs/aris/WORKFLOW.md, current/resume state and complete pinned research-implement-feature SKILL (Phase 4 applied; SHA33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8 matches installed copy), plus reviewer-backends, reviewer-independence, reviewer-routing, review-scope-limits, acceptance-gate, effort-contract, capture-antipatterns, injection-hygiene, output-language, integration-contract and review-tracing. Pin remains f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Project reviewer config resolves to codex/gpt-6.1-sol; no provider switch or secondary reviewer was invoked.

## One targeted repair closure — 2026-11-12 14:49 UTC

**R1 resolved. No unresolved code blockers found in this bounded review; normal full input/backbone CPU readiness gates remain pending.** Read the complete raw `checks/target_collation_repair.json` directly and verified SHA `30cd6f37083d24d69163d77beb18c13e7f74e85d48b9a9fe3a8af78e8e5d99a7`. Five actual cached-target worker cases PASS: integer-first and float-first mixed batches float32; all-integer and single-integer batches int64; the actual seed1 epoch68/batch10 targets float32. Every case is bitwise equal to ordinary torch.stack. Raw initial failure SHA `c0a029f9575fb1914ea45d17b1e63bd289baddd392d162afc87b407e61e71523` remains preserved. This closure ran on server CPU only without RGB decode/model/GPU work.

Directly inspected the exact small repair: explicit target stack in collate_native, import and use in train/evaluate DataLoaders, and E08 Source:sweep ledger row. Updated native_inputs.py SHA `79fb206356e29183464cedb5e1197f51dcd5517991dcd6d9fbe67c4153d9c0b9`; native_run.py SHA `f4a630ab16a56b1147ee225c858bb08c6640c317407597f41dd1a36a66e937b7`; other three frozen hashes unchanged. Original sampler functions, cache interface and native target values/arithmetic remain unchanged. No new sweep or scientific review is requested.

Read full ENVIRONMENT_ORIGINS.json: actual imported Torch/torchvision/NumPy and CUDA distribution origins are the read-only project base; SciPy/OpenCV/MMCV/timm/einops/kornia and installed overlays resolve to the new native310 prefix. Both exact source Git HEADs are verified: pinned ARIS f76ec8d and original TransRAC 68bdd4d; original TransRAC tracked tree is clean. Full1244 input, sampler and CPU backbone coverage receipts are still pending, so this is **not an all-ready declaration**. Root retains source synchronization, review receipt and complete-arm manual GPU handoff.


## Targeted readiness closeout — final normal receipts

This closeout reads the normal server CPU preflight and input artifacts only. It does not reopen source review, add tests or change scientific scope. Every required copied receipt is closed, directly read and SHA-verified. CPU_PREFLIGHT is PASS and its final five-file code map, contract, input receipt, official source and environment match the final reviewed artifacts. Its historical `fresh_review: pending` field is fulfilled by this review; root still owns the final ENGINEERING_REVIEW gate.

| Required evidence | Final status | Concrete result |
|---|---|---|
| Full prepared inputs | PASS | 1244 entries and 1244 unique keys: RC train756/valid131/test152, UCF105, QUVA100. Positive validation130 and test151 retained; stu6_44 scalar remains null with the native empty-bound target. Input sampler-source SHA matches final native_inputs.py. |
| Raw sampler and target | PASS | Original long rc_valid/stu3_3 N1441 and zero rc_valid/stu9_67 N275 RGB tensors/targets exact; short11 fixture equals original loader; inclusive ROI2..8 exact. |
| Official backbone state map | PASS | Exact 9950e3be...76acc asset; all183 parameters/persistent buffers match values and shapes, 29,694,438 loaded elements. No counting checkpoint and no network forward. |
| Final CPU preflight/source binding | PASS | Exact contract1b986ab and repaired final five-file map; official68bdd4d with479 tracked files and103 imported official modules under the original source tree; source and environment bindings identical. Existing B32/B20 primitives have all four errors0,200 scheduler rates, RNG/epoch recovery PASS. |
| Scoped runtime and read-only origins | PASS | Native Python3.10.22 overlay; exact inherited Torch2.5.1+cu124/torchvision0.20.1+cu124/NumPy1.26.4 and new-prefix overlays. HOME unchanged; actual tempfile root and every scoped cache/config path under R; GPU hidden during CPU preflight. |
| Review scope and remaining release work | COMPLETE / root dependencies | One initial review + one R1 closure + this normal-receipt closeout. No remaining code/readiness engineering blocker. Root must finish publication/synchronization, write exact final ENGINEERING_REVIEW and hand off only at a complete current-arm boundary. No reviewer GPU/process action occurred. |

Exact receipt SHA256 values:

| Artifact | SHA256 |
|---|---|
| inputs/IDENTITIES.json | `9341a1b74120c5fd12172a1c8af1ecdba32403db32a6392c86a6343ba3e6882d` |
| checks/sampler.json | `f0af24f82585aaddad9364d4a95e9aa6616715130c03dc718ba157178ffc5f6a` |
| checks/backbone.json | `006a9b6b0bc002e9439693f3eb79b4cf7749fbf32bab6193b6f0521bced9064f` |
| checks/CPU_PREFLIGHT.json | `dd6b2810f5913997fca7396a94e5b112a092eca28c2042ba1dd737809fb0d898` |
| checks/RUNTIME_SCOPED_ENV.json | `60740d1aefb3a9afb496f4dc466949ea43cefad6d1f1b435b2dc64b80e275457` |
| checks/ENVIRONMENT_ORIGINS.json | `588552816f93207dfde0ff42b443e9a41baf564e153b9693f0857059962075ac` |
| CODE_IDENTITIES.json | `5ed78c1a935bf5a39e3dd7ef8c58b8d564b5a9d5aacfd612d5a48bff68d37a46` |

Final code and unchanged contract identities are in REVIEW.json. Requested reviewer gpt-6.1-sol/xhigh; actual backend model/effort remain unknown, so this positive engineering judgment remains same-family provisional. Actual4090 full training, its AMP behavior, full357 predictions and scientific performance remain unmeasured.
