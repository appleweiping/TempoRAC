# RepNet 2026-11-12 scheduling amendment — bounded closure

**Verdict: CLEARED_FOR_EXACT_JOB_RUNTIME_GATE; same-family provisional.** No concrete launch, ownership or source-identity blocker was found in the reviewed amendment. This closes only the new execution gate/launcher. The 2026-11-11 R1–R5 adapter closure is inherited; no full scientific review was repeated. After the existing environment/startup checks, the exact launcher may perform one restored-weight/kernel witness then the fixed full357 evaluation when its live gate passes. No further review is required by this closure during that run.

Requested reviewer: gpt-6.1-sol; session /root/repnet_official_review; actual model/backend/effort independently unattested. Native OpenAI same-family semantic assurance is provisional. No experiments, local computational validation, installation, model import, server query/mutation, source/run change or GPU operation was performed by this reviewer. Only this review file was written.

| Amendment requirement | Status | Artifact-grounded evidence |
|---|---|---|
| Explicit resume replaces paused predecessor completion | Closed, static | Actual resume_authorization.json matches configured SHA and explicitly authorizes the fixed official external comparison. gpu_check109–117 requires its exact identity/schema/excerpt/scope. The two actual historical stop receipts match configured SHAs and both retain full_s0=12/full_s1=13 completed epochs and complete=false. gpu_check118–124 verifies those immutable records and does not invent completion. |
| Old workers gone; GPU0 reserved; GPU1 live availability | Closed, static | Config fixes GPU1,8GiB process cap and10GiB minimum free. gpu_check105–147 rejects other GPUs, checks six stopped root/setup PIDs absent, scans for the old full_s1/GPU1 queue workers, requires fresh GPU1 free memory and an empty GPU1 compute-app owner list. It checks current conditions once and raises on failure; no worker is terminated. |
| Gate precedes TensorFlow/core import | Closed | Evaluation207–208 and witness331–332 call gpu_check before load_model; TensorFlow import is165, core import occurs inside load_model afterward. Authorization and ownership are rechecked separately for witness and evaluation. |
| Immediate bounded launcher and correct affinity | Closed | run_official_repnet.py uses the new20261112 adapter/output root, isolated repnet Python, spec environment with CUDA_VISIBLE_DEVICES=1 and isolated CUDA-library paths. Lines37–48 launch witness then evaluate once, wait for each, and stop on nonzero exit. No previous full_s1 wait, polling, automatic retry, ours training or GPU migration is present. |
| Existing scientific protocol unchanged | Inherited and unchanged | Whole-file diff of old/new evaluate_repnet.py contains only gpu_check replacement. Config diff changes only the execution authorization and source/launcher identities. Core and metrics SHA remain the previously reviewed values; all population/label/crop/inference/checkpoint settings remain unchanged. |
| Exact candidate/source consistency | Pass, static | Current config SHA and all six source_sha256 entries match actual files. Resume receipt and both historical stop SHA values match actual local primary files. Identities are listed below. |

Declared semantic amendment: the old full_s1 successful-completion/queue_complete condition is replaced by explicit latest user resume, immutable stop evidence that honestly records incomplete training, absent old workers, and live free GPU1 ownership. This is explicitly authorized by GATE_AMENDMENT.md and the bounded Astra continuation decision. The new GPU0 three-seed serial training has no scientific dependency on RepNet; this launcher never starts or modifies that training.

## Exact reviewed identities

| Artifact | SHA-256 |
|---|---|
| evaluation_config.json | 4b5f1814b2a6c7bc8d6ac818c4f553e0fff851d4824b3c5b76f5319e6a348e54 |
| evaluate_repnet.py | 757820028a63c5d37a911b8caedad576a3967131690725539db8ea0fbe3d8e90 |
| run_official_repnet.py | cb1bdc77b99451579aa4801cd42df55fd4ea43025f42abf9e8c6daf221e4578e |
| repnet_official_core.py | af46180e5f7671efbd2e95b73630592802ec75752382800f4ba8ad903abc6475 |
| frozen_metrics.py | fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0 |
| env-spec.json | 3a85c459413a5bd0f282a99fc3acd458096c8175c5198f334cd4404fd8746b39 |
| setup_repnet_env.py | 95f6be9a024e79b05294925b2c5a752bb5708eb11f2c2978a6e117e70e45b49b |
| resume_authorization.json | 8c564c92701879d8edf9ad6487c080176a74609f3199c7855d032d2bc59c67db |
| user_stop_receipt.json | 05cf0fa3fbc62cbb548866ba9aaae60c47c755bb4357e82bdcf290887718cd3d |
| user_stop_receipt_additional.json | bbb236ee3af047a0b61620393ba1d6731e419886ffa468bab19f52e55f4599de |

Exact dispatch interface from the verified server project root, after the already-planned environment/startup checks:

```sh
env/bin/python -u -B external-baselines-20261112/repnet-runner/run_official_repnet.py --review PATH_TO_THIS_COMPLETED_GATE_REVIEW
```

The configured runtime paths are /root/autodl-tmp/TempoRAC/external-baselines-20261112/repnet-runner and its new runs directory; assets and isolated environment are intentionally reused from existing verified locations. Old output/stop evidence is preserved.

Read GATE_AMENDMENT.md, actual resume and historical stop primary files, complete new launcher/config, old/new runner and config diff, relevant new gate/import/call sites, and bounded Astra decision. Re-read the full pinned research-implement-feature SKILL; installed/pinned copies still match33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8. Applicable reviewer/independence/acceptance/scope/tracing/effort/language/injection references retain the complete read record of the earlier review in this session; no new scientific or reviewer stage was opened.

Executor-reported server py_compile/PREFLIGHT_OK357 is not presented as this reviewer's independent execution. Environment repair, real GPU1 ownership,226-variable restore, finite kernel outputs and full357 performance remain actual runtime evidence to collect. This closure does not assert installation success, current GPU idleness, full evaluation completion, SOTA or project completion. Review slot released to the authorized exact run.
