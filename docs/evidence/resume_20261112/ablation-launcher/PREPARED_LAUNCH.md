# Prepared21 finite lanes — no production launch yet

Root first freezes the reviewed launcher and contract in its authorized publication, then stages exact files under `/root/autodl-tmp/TempoRAC/resume20261112/ablation-continuation/`. The launcher publication can advance independently; training remains the clean immutable489dc77 snapshot. Candidate final worker SHAd23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144, contract SHAdc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167.

Both lane invocations use:

```text
/root/autodl-tmp/TempoRAC/env/bin/python -u -B <control>/ablation_worker.py
  --gpu <0 or 1>
  --source /root/autodl-tmp/TempoRAC/source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb
  --commit 489dc77b91a21d21215784ff49961edb20dab4fb
  --contract <control>/RUN_CONTRACT.json
```

There is no arbitrary run-list override: the exact12/9 ordered contract lanes determine all21 jobs. Outputs use the fresh `runs/frozen_known_source_v1_ablation20261112/` group. Verify the reviewed hashes, current authorization/final Astra JSON, exact frozen manifest/spec, existing cache reuse and both free GPUs/no compute PIDs immediately before root dispatch. Use detached hidden/noninteractive process roots and preserve launch command, PID, source/worker/contract identities and selected environment.

Before starting each worker, export CUDA_CACHE_PATH=`R/.cache/cuda`, XDG_CACHE_HOME=`R/.cache`, TORCH_HOME=`R/.cache/torch`, HF_HOME=`R/.cache/huggingface`, TMPDIR=`R/tmp`, PYTHONDONTWRITEBYTECODE=1 and OMP/MKL/OPENBLAS_NUM_THREADS=2, where R is the verified project root. Child environments retain these project locations; no cache rebuild or environment installation.

To release one lane for an inherited official comparator under its separately frozen runtime-ready contract, root may create `<control>/HOLD_AFTER_CURRENT_GPU0` or1. This is an explicit resource action, never a response to scores. The worker reads it only after the current arm's full training, all authorized evaluations and durable completion receipt, writes `lane_gpuN.json` containing the exact unstarted remainder, then exits. The other lane continues. No kill, migration, timer or retry.

Root clears the hold file and explicitly invokes the identical worker/source/contract with `--resume` to continue. Completed prefix arms are verified and skipped. A stale saved residual after a hard stop may advance only past currently valid complete-provenance/hash receipts; missing or out-of-order completions reject. Partial training requires its existing latest.pt and the trainer's exact protocol identity. Failed evaluations are not silently overwritten or automatically retried. Failed lanes stop before a successor.

The21 jobs contain1320 frozen epochs, singleRC/CX evaluate only their corresponding test population, union arms only RC/CX/UCF. There is no QUVA ablation inference; completed full-model QUVA is reused. Three historical singleUCF arms are reused under their disclosed provenance qualification. No new scientific matrix or training-core change is authorized here.
