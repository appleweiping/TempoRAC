# Invocation and constraints

Preparation only. Root must publish/review the exact entry before real use. Check once at the actual whole-arm boundary:

    /root/autodl-tmp/TempoRAC/env/bin/python -B /root/autodl-tmp/TempoRAC/resume20261113/ablation-after-racnet/resume_gpu1_after_racnet.py --check

Root separately supplies a real release JSON under the project root, with released_by=root; authorized_residual exactly [wo_token_s2,wo_joint_cx_s0,wo_joint_cx_s1,wo_joint_cx_s2]; gpu_index=1; actual gpu_uuid; entry_sha256; start_before_epoch; racnet_pipeline_complete={path,sha256} exactly as READY output. This document is not a release receipt. Then explicit root command only:

    /root/autodl-tmp/TempoRAC/env/bin/python -B /root/autodl-tmp/TempoRAC/resume20261113/ablation-after-racnet/resume_gpu1_after_racnet.py --execute --release <actual-root-release-path>

No waiting/polling/automatic retry. --execute repeats assessment, holds one-shot and RAC writer locks during transfer, archives the original GPU1 hold by rename only after READY+release, records one_shot.json, and invokes exactly originalworker --gpu1 --resume. It waits only for its ordinary finite child. Any failure retains claim/archive/logs for root, with no further scheduling. No scientific/model/data/metric/seed/epoch changes; wo_token_s2 keeps original50epochs and wo_joint_cx_s0/s1/s2 keep original80epochs.

Runtime durable outputs: resume20261113/ablation-after-racnet/runtime/{one_shot.json,successor.lock,HOLD_ORIGINAL_GPU1.archived.json,worker_gpu1.successor.stdout.log}; original worker continues existing authoritative lane/events/output schema. Metadata fixture is never the actual release or scientific result.
