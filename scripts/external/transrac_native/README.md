# Official TransRAC full-run adapter

This is the frozen execution adapter for official TransRAC source
`68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979`, not a replacement counting model.
The exact contract is `RUN_CONTRACT.json`, SHA256
`1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a`.
It specifies seed 1, 200 complete epochs on 756 training videos, 131 validation
videos, and 357 final evaluation videos. Only the official K400 backbone weights
are loaded; task weights are trained by this run. Modern runtime, virtual replicas
and validation selection adaptations are disclosed in the contract.

The execution directory is
`/root/autodl-tmp/TempoRAC/baselines/transrac_native_v1`.
The five executable/spec files there must match the published `CODE_IDENTITIES.json`
and the CPU preflight and bounded engineering review receipts byte for byte.
This repository copy is the immutable source archive. Copying it to a fresh
directory alone does not supply the verified inputs, environment or launch gate.

Full RGB inputs are prepared once and reused. Production uses `run.py --phase
train-evaluate --gpu N` only after a completed ablation arm, all its evaluations,
its completion receipt, and a free GPU check. The other GPU lane continues.
Set the project-scoped cache/temp variables shown in `cpu_preflight.sh`; preserve
the user's HOME. Execution and all new experiment artifacts stay on the server.

After an interruption, the root explicitly uses `--resume` with the same identity.
It restores the last committed complete epoch and reuses verified final per-video
predictions. Failures exit without automatic retries or configuration changes.
No completion-tracking automation, score-based pruning or new experiment matrix
is implemented here. The root resumes the exact remaining ablation lane after
this finite baseline run completes.
