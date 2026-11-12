# Official RACnet native bridge

Frozen contract SHA: 77c3d84443fb57415db3b2ed7416b3b16af5f8e8af614c41a05812802e7f8de8. Original source51b8e3f. Native model, GenerateGT and cal_count are copied byte-for-byte from the fixed server export by `plan`. The wrapper implements the published loss and fixes validation unpack/start handling and same-device masks. No neural forward is allowed in CPU preparation.

All commands run on the verified server, from `/root/autodl-tmp/TempoRAC/baselines/racnet_native_v1`, using `/root/autodl-tmp/TempoRAC/envs/racnet-native310/bin/python -B run.py`. Set `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; set `TMPDIR` and `XDG_CACHE_HOME` to this candidate's own `tmp` and `cache` directories. CPU commands require `CUDA_VISIBLE_DEVICES=''`.

Preparation order: `plan`, `runtime`, `prepare`, `cpu-check`. `prepare` writes all1243 ordered rows, all886 original targets,357 label-free inference records and a separate scorer manifest. It reuses completed author CRC/finite receipts and media identities. PENDING records preserve exact missing inputs. CPU_CHECK remains PENDING until all full inputs are present. Do not rerun passing CPU primitives; concrete fixes receive targeted closure.

Future formal entries are `extract-quva --ownership-receipt <root receipt>`, `train --ownership-receipt <root receipt>` (add `--resume` only for the same committed trajectory), and `evaluate --ownership-receipt <root receipt>`. They require exact candidate/environment/input bindings, fresh focused root review and an unoccupied released GPU receipt; no GPU is currently released. QUVA extraction must precede complete input qualification and full200 training.

Epoch checkpoints use fsynced generation files and one atomic COMMIT pointer. Only completed epoch boundaries resume. Partial attempts remain as durable per-step journals with replay accounting. Checkpoint storage retains initial/first/latest/bestMAE/bestOBO references and reclaims only this candidate's unreferenced generations. Final outputs are individually atomic and identity-bound. Scoring opens labels only after all357 predictions persist. Native H-norm p=0, batch16 and full temporal length stay unchanged; actual OOM/nonfinite stops the route.

This is source preparation and CPU engineering evidence, not full-performance completion. It does not edit active baselines, acquisition workers, queues, source_sync, or current_status.
