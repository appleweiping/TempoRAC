# Prepared finite production launch — not launched

Root owns final Git/source freeze and process dispatch. Source must be a clean Git snapshot under `/root/autodl-tmp/TempoRAC` containing the final reviewed train.py SHA33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4. Worker SHAe8844834850bfdb749d6240d82c2541e9c82d796891fe84dfebba6d98fd009d4. Stage worker under `resume20261112/continuation/continuation_worker.py` after review; current isolated validation copy is under checkpoint-work.

Before starting the worker, create and export these project-owned locations in its parent environment; the worker inherits them:

```bash
export CUDA_CACHE_PATH=/root/autodl-tmp/TempoRAC/.cache/cuda
export XDG_CACHE_HOME=/root/autodl-tmp/TempoRAC/.cache
export TORCH_HOME=/root/autodl-tmp/TempoRAC/.cache/torch
export HF_HOME=/root/autodl-tmp/TempoRAC/.cache/huggingface
export TMPDIR=/root/autodl-tmp/TempoRAC/tmp
export PYTHONDONTWRITEBYTECODE=1
```

Record the actual selected environment in the production launch receipt before dispatch. This does not modify the frozen scientific configuration. Root creates the directories under the verified project root and starts the detached worker with these variables. No local training, environment installation or model execution.

GPU0 arguments: `--gpu 0 --runs full_s0 full_s2 --source <clean-source-snapshot> --commit <final-40-character-SHA>`.

GPU1 arguments: `--gpu 1 --runs full_s1 --source <same-clean-source-snapshot> --commit <same-final-SHA>`. Root starts this worker only after the full RepNet process succeeds with all expected outputs and releases GPU1; worker additionally requires free GPU memory below500MiB. No baseline readiness shortcut is encoded.

Fresh group is `runs/frozen_known_source_v1_resume20261112/`; original partial/full historical directories are preserved. A necessary restart is a manual invocation with the same source/SHA and `--resume`; checkpoint protocol/provenance must match. No automatic retry. A failed child stops its queue. Worker writes event logs, observed first durable checkpoint receipt,50-epoch completion checks, best checkpoints and complete evaluations for RC152/CX2403/UCF105/QUVA100. Only the three initial full seeds are allowed; the other24 remain unreleased.
