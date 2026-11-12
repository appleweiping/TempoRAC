# Exact next commands (server)

CPU-only refresh after genuinely new completed assets; passing primitives are reused:

```bash
R=/root/autodl-tmp/TempoRAC
W=$R/baselines/racnet_native_v1
cd "$W"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=$W/tmp XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/matplotlib
nice -n 15 "$R/envs/racnet-native310/bin/python" -B run.py prepare --work "$W"
nice -n 15 "$R/envs/racnet-native310/bin/python" -B run.py cpu-refresh --work "$W"
```

No formalGPU command is currently authorized. After root has an actual completed-arm/lane receipt and all mode-specific gates, the full workload command (not a pilot) is:

```bash
CUDA_VISIBLE_DEVICES=<released_index> "$R/envs/racnet-native310/bin/python" -B run.py extract-quva --work "$W" --ownership-receipt "$W/ROOT_EXTRACT_OWNERSHIP.json"
# After all1243inputs are ready, the samecode/source/rootreview are bound and a new actual ownershipreceipt exists:
CUDA_VISIBLE_DEVICES=<released_index> "$R/envs/racnet-native310/bin/python" -B run.py train --work "$W" --ownership-receipt "$W/ROOT_TRAIN_OWNERSHIP.json"
# Explicit continuation only: add --resume. Never change batch16/T/native math after realOOM.
CUDA_VISIBLE_DEVICES=<released_index> "$R/envs/racnet-native310/bin/python" -B run.py evaluate --work "$W" --ownership-receipt "$W/ROOT_EVAL_OWNERSHIP.json"
```

Ownership schema is implemented by native_common.gpu_gate: released_by=root; mode; contract_sha256; exact candidate_identity; authorized_gpu_count=1; current_arm_complete_receipt{path,sha256} whose status is complete; gates{source_runtime,CPU_interfaces,lane,resources,all1243_inputs} (last excludes extraction); immutable_publication_receipt{path,sha256} bindingcandidate_identity; fresh_review_status + fresh_review_receipt{path,sha256} bindingcandidate_identity/verdict; gpu_index/gpu_uuid; expires_epoch; pending_author_bytes; additional_bytes_reserved. Actualnvidia-smi checks the chosenlane is empty; sharedreserve and candidate32GiB cap are checked. The receipt must be produced from actual rootobservations, not copied placeholders.
