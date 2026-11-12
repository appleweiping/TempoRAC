# TransRAC native training assumptions — fixed before implementation

ASK=never. These decisions implement the user's authorized official-method reproduction. No implementation or experiment is claimed here.

| ID | Under-determined choice | Frozen decision | Class | Source |
|---|---|---|---|---|
| A01 | Baseline identity | Official 68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979; TransferModel, 64 frames, scales [1,4,8], OPEN=False; fresh counting weights and official Swin-T K400 backbone. | semantic | user/default |
| A02 | Raw and NPZ normalization disagree | Use the documented raw branch actually imported by train.py: RGB float32 / 255, floor(i*N/64), 224x224. Use this branch consistently for train, validation and evaluation. | semantic | default |
| A03 | Fine-cycle annotation source | Existing EveryShotCounts repcount/*_with_fps.csv, bound by SHA; inherited full count/bound agreement with Lance. Serialize L fields as integer strings only after exact integral-value checks. Do not use open_set splits. | semantic | default |
| A04 | Qualified training population | Filter the unchanged corrected union manifest to rc_train=756; retain rc_valid=131. Only stu4_50/stu3_27 are excluded from optimization. Every evaluation row and original label remains. | semantic | user |
| A05 | Training budget | Native seed 1; 200 epochs indexed 0..199; logical batch 32; Adam defaults, base lr=8e-6; MultiStepLR milestones [0,40,80,120,160], gamma=.8; source FP16 AMP. | semantic | default |
| A06 | Four native replicas on one GPU | Four virtual shards formed with torch.chunk(batch,4): 8 each for B=32 and 5 each for final B=20. Weight shard losses by b/B; step once. Restore all BN buffers from the pre-step snapshot before each shard; retain only shard 0's resulting buffers. Preserve separate CUDA RNG streams. | semantic | default |
| A07 | Frozen backbone mode | Preserve model.train(), including Swin drop_path_rate=.1 within no_grad. Do not cache learned backbone features or force the backbone to eval during training. | semantic | default |
| A08 | Native graph and targets | Preserve cumulative scale-4 to scale-8 zero-padding, shared BN1 across scales, signed final density and native overlapping-target assignment. mae_error=False means only density MSE contributes gradients. | semantic | default |
| A09 | Validation selection | Among native saved-checkpoint candidates 51,54,...198, minimize harmonized positive RC validation NMAE; break ties by higher OBO then earlier epoch. Preserve native continuous/density-GT and batch-mean diagnostics separately. | semantic | default |
| A10 | Complete final population | RC152, UCF105 inclusive annotated ROI, QUVA100; same raw start-aligned sampler for all. One RC-trained checkpoint transfers to UCF/QUVA; no Countix addition. | semantic | user/default |
| A11 | Native versus common scoring | Keep native density-derived RC count separately; harmonized metrics use dataset CSV counts and canonical NumPy rounding/(g+.1), plus all/zero/OBZ/RMSE,/g/continuous outputs. No clamp or TTA. | semantic | default |
| A12 | Compatibility environment | Root-authorized Python3.10.22 venv at R/envs/transrac-native310, inheriting R/env Torch2.5.1+cu124/CUDA/torchvision/NumPy read-only. Pinned MMCV1.4.0, timm0.4.12 and remaining TransRAC overlays install only in the new prefix. Record import paths, versions and origins; no original Torch1.7 or Python3.11 bitwise-parity claim. | interface | root explicit resource amendment |
| A13 | RGB input cache | Optional complete sampled uint8 RGB cache only; preserve exact frame indices, resizing, channel order and float conversion, with CPU comparisons to the official raw loader. | interface | default |
| A14 | Durable execution | Atomic epoch checkpoints with model, optimizer, scaler, scheduler, RNG, virtual streams, data order, selected best and log offsets; explicit exact resume, preserving attempts. | interface | user/default |
| A15 | Resources | At most one 4090 after an ablation arm's full training/evaluation/receipt and root's manual hold; the other lane continues. Existing 21-arm contract remains unchanged. | interface | user |
| A16 | Claim label | Official-code full training under raw-loader/modern compatibility and corrected RC756, one seed 1, disclosed single-GPU virtual replicas and common validation selection. Not an authors' checkpoint or exact paper-number reproduction. | semantic | default |

No choice above remains a user permission question. Engineering failures are repaired within these choices; changes to scientific meaning require a new version before execution. Native AMP skips are explicitly logged and counted, not hidden or silently compensated by extra steps.
A03 implementation clarification from actual preparation: stu6_44 has empty CSV count / manifest null and empty bounds. Retain it in RC756 with the native empty-bound target; do not synthesize a zero scalar label. Validation runs every epoch51..199; only51,54,...198 are selectable. Canonical scorer is temporac_cvpr/metrics.py SHA fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0, not older pams metrics.



## Executor interface decisions declared before final source freeze

| ID | Interface gap | Decision | Class | Source |
|---|---|---|---|---|
| E01 | Cache preparation trigger/reuse | Explicit --build-rgb-cache only; two CPU threads, per-row atomic uint8 NPY+sidecar; verify exact source bytes/ROI/actualN/indices/samplerSHA; existing committed entries verified and reused. No learned feature cache. | interface | user/default |
| E02 | Actual QUVA role in asset manifest | Preserve population=test/dataset=QUVA for all100. No manifest relabeling. | interface | default |
| E03 | Native final batch size | Source test.py default batch1; validation logical32 split up to4; train only B32/B20 with nativefour shards. | interface | default |
| E04 | Final-eval durability | Atomic individual prediction records bound to exact best checkpoint/input/source/env; explicit resume reuses matching records and evaluates missing rows only. | interface | default |
| E05 | Epoch/best transaction | Atomic latest contains current and best model state; independently materialize best.pt after commit. On resume recover committed best only, preserve any orphan. | interface | default |
| E06 | Root review and launch gate | Require ENGINEERING_REVIEW.json complete=true,status=clean or PASS,exact code map/contractSHA; same-family provisional. Root writes only after bounded review. | interface | user/default |
| E07 | Package metadata versus imported binary identity | Existing torch dist METADATA says2.5.1, actual torch.__version__2.5.1+cu124; require both, same for torchvision. Record origins and Torch extension SHA. | interface | default |
| E08 | Native int64 empty-bound target first in mixed worker batch | Source:sweep R1; use collate_native with default_collate for RGB/indices and torch.stack(targets) without first-item-typed out; preserve int64 allinteger/testsingle and normal mixed float32 promotion. Worker target-only witness proved default failure at seed1 train epoch68/batch10; no labels, native arithmetic or RGB/cache sampler change. | interface | sweep |

These interfaces implement the unchanged semantic rows above. CPU primitive tests are engineering evidence only; no model dataset performance or GPU parity is claimed.


## R2 pre-GPU startup serialization repair —2026-11-12

| ID | Interface assumption exposed by actual runtime | Chosen | Class | Source |
|---|---|---|---|---|
| E09 | Live environment distribution tuple pairs were assumed equal to their JSON-loaded list pairs | Return native list pairs from environment_identity; preserve all package facts/order/versions.126unequal tuple/list items are JSON-canonically identical; TorchVersion/str difference is already Python-equal. Archive prior pass/failure, rebind only changed native_run codeSHA with explicit prior CPU receipt supersession; no fixture, dataset, model or GPU rerun. | interface | user |
