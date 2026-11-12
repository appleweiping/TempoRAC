# TransRAC native full-training engineering review

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
