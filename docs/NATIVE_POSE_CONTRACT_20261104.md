# TempoRAC native pose 输入合同（2026-11-04；Astra；两页）

**决定：先完整生成动态 raw cache，另行桥接旧 320 learner。** 继续 TempoRAC，不改模型/loss、不重启 A005 门禁。本次只读设计，未推理/用 GPU/读 test 标签；同族设计意见 provisional。

| 验收项 | 结论/通过证据 |
|---|---|
| 完整人口与标签边界 | 设计通过：794 train+183 development videos；1878+486 supplied identities；失败保留 |
| 原时钟/观测/身份可追溯 | 设计通过：以下 raw schema；禁止 v2 的重编号+全 1 mask |
| 一次性全量缓存可实施 | 设计通过，生成未完成：已有离线 COCO17 资产/API；待专用入口、CPU 检查及资源 |
| 正式训练输入 qualified | **未达标**：SHA 不能代替 decode 对齐、pose/桥接、split/目标有效性 |

## 第 1 页：独立 raw 层与 bbox-only 投影

`temporac.native-pose.v1` 每视频保存 JSON 元数据、数值 NPZ（禁 pickle）、失败账本，**不由现有 feature_io 加载**。帧轴共享；每视频固定 local slots，source object→slot 对应仅在投影审计中。slot 不跨视频表示身份、不作为输入通道，也不随失败重编号。

| 必存字段 | 语义 |
|---|---|
| 媒体/帧 | `media_key,split,media_sha256,width,height,annotation_length=L,decoded_length=D`；正常 `N=L=D`，`source_frame_index:int64[N]=0..N-1` 是原呈现帧序号。原视频、原尺寸 RGB，不能用224派生帧 |
| 时间 | 整数 `pts[N]`+`pts_present[N]`、有理数 `time_base_num/den`、原起点/可得 duration、decoder/stream/version；`best_effort_timestamp` 单列为重建值。不可得则 `clock_status=source_frame_only`；FPS 仅元数据，index/FPS 不称实测 PTS |
| 每人 | `slot[P]`；原 `bbox_xyxy[P,N,4]`；`xyconf_raw:float32[P,N,17,3]` 为 native pixel xy+原 score、COCO0…16，第三维不是深度。不 clip score、不阈值删点、不归一化/时间补点 |
| 支持 | `decoded_ok[N],bbox_present[P,N],bbox_valid[P,N],pose_returned[P,N],joint_observed[P,N,17]`。observed=当前帧 frontend 返回有限 xy/score，**不证明关节可见**；低分保留。缺失/失败零 payload、false mask+reason；非有限输出另记 invalid mask。旧>0.20 是下游派生有效性 |
| 关联 | manifest 关联媒体/投影、精确模型/config/processor、源码 commit+实际脚本 SHA、环境/命令/设备/耗时；每 video/slot 必有 pending/complete/failed 状态。只用普通 SHA/日志，不加安全证明体系 |

**L≠D 不暗补。** 保留 L 个预期 annotation rows 和 D 行 decode/clock ledger；不能确认映射的 pose mask=false，视频记 `alignment_unresolved`，不将 extra decoded rows 塞进最后的 bbox。整视频失败也保留全部 slots。完整 raw=全人口有产物/失败交代，不等于全部可训练。

仅 trusted projector 读显式 train/val 源 JSON，输出严格白名单 `{schema,split,media_key,media_path,media_sha256,width,height,length,persons:[{slot,bbox_xyxy,bbox_present}]}`；其他 provenance 另存 sidecar。按数值 objectN 次序映射，frontend 只读投影/MP4，不接收源 annotation dictionary/path，不导入 counter/repair/evaluator。输出递归拒绝未知字段及 count/period/periodicity/density/boundary；不 glob test。

每原帧按 slot 处理全部 bbox，固定 `(frame,slot)` batch 归属；有限、正面积且与图像有交集才生成 crop，部分越界保留原框、交固定 processor 处理。xyxy→xywh；offline processor `use_fast=False`；postprocess `threshold=None`、不传 `target_sizes`，核对 labels=0…16。无补人/身份重关联/插值。披露为 **externally supervised ViTPose + supplied GT bbox/identity assisted, count-label-free learning**；checkpoint 精确训练集仍待核实。

4组相同媒体的 train 别名有 count/period 冲突（58/48），bbox/periodicity 相同：保留全部记录及别名关系，**不改标签、不静默去重、不阻塞 bbox-only 提取**；评估/监督基线另行处理。frontend 不读取这些计数或周期值。

## 第 2 页：实现与启动依赖；尚欠的 learner 桥接

新增独立入口 `scripts/build_temporac_native_pose_cache.py`，API：

```python
project_bbox_only(media_manifest, annotation_root, splits=("train","val"))
validate_projection(projection, decode_audit)
extract_native(projection, decode_audit, snapshot, output, device, batch_size)
validate_raw(cache)  # 全人口、数组、归属、时钟和失败；不读计数标签
```

一次 CPU fixtures 覆盖多人/多帧归属、bbox 转换、缺框/全失败/低分/非有限值、COCO顺序、标签拒绝、N≠320、L≠D、缺失/重复 PTS；随后验证完整投影 census。复用977 composite/906 original SHA及当前全帧 decode 审计。新的 ffprobe 已定位 `/home/lilei/miniconda3/pkgs/ffmpeg-6.1.1-hecf7045_5/bin/ffprobe`，复用已验证库路径 launcher；旧 PATH 缺失不再当工具不存在。若 clocks 只有浮点秒，应补整数/有理数或明确 unavailable，不能逆推伪 PTS。异常 PTS 保留，不排序 pose 来修复。

以下是**待实现后的接口，不是现成命令/启动回执**。在已核实 server checkout 执行：`R=/mnt/4tssd/lilei/TempoRAC`；`PY=/home/lilei/PoseRAC_Experimental_Code_20260828/.venv/bin/python`。Sol先写 `$R/config/native_pose_v1.json`，明确 source root、`$R/runs/media-integrity-20261104/media.jsonl`、`$R/runs/media-decode-20261104`、全部 train/val、输出 `$R/native_pose_v1`、固定 float32/batch、上述策略。模型使用 inventory 完整 snapshot `a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45` 及四份已核实资产 SHA（weights `853753…e4fdf`），local/offline，不能只按模型名字加载。

```sh
CUDA_VISIBLE_DEVICES='' "$PY" scripts/build_temporac_native_pose_cache.py project --config "$R/config/native_pose_v1.json"
CUDA_VISIBLE_DEVICES='' "$PY" scripts/build_temporac_native_pose_cache.py validate --config "$R/config/native_pose_v1.json"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$ALLOCATED_GPU" \
  "$PY" scripts/build_temporac_native_pose_cache.py extract \
  --config "$R/config/native_pose_v1.json" --device cuda:0 --resume
```

无 limit/debug cap；逐视频 atomic 写；resume只复用相同媒体/投影/模型/有效配置的完成产物，保留失败尝试并可重试。全量循环首个必要 forward 同时验证设备/API，不另开 GPU 微实验。当前 GPU 全忙：先实现/投影/CPU检查及并行其他科研准备，取得已分配资源再全量推理，**不抢 STaR-Body**。全量 CPU 亦需显式线程配额/完整 ETA；不能暗开不限线程任务。Windows后台根进程隐藏，服务器保存PID/日志。

**桥接仍未完成：** FeatureRecord 恰七字段、motion=(320,17,3)、clock=int64；preprocess 拒绝内部无效帧、少于17点及原时钟间隔≥5。不可为通过而改成0…319、全mask、插值当观测。可立即加纯 `audit_bridge(raw)->{admissible,reasons}`；其可接收子集不能冒充完整训练集。

下一项只解决动态N/支持runs如何进入现有每人推断；若保留320，必须版本化冻结采样、原时钟单位/频率限制、缺失支持及原时域输出映射。外置 mapping 保存 `output_row,output_physical_clock,left/right_source_index,weights,observed/interpolated/extrapolated,supported,gap_span`，逐关节支持不能被帧mask吞掉。已有 build_timebase 可复用，但要求严格递增时钟和至少一个观测；全失败另记状态，其 sample-center 映射不自动定义半开 period 边界，端点须单测。**可逆默认：raw不插值/丢帧，不定新gap阈值、不改旧训练契约/loss。完整raw不等桥接；正式训练仍等桥接、split/伪目标质量分别成立。**

审读依据：E:\AGENTS.md、活动AGENTS/ARIS_LOCK/WORKFLOW/run state、CVPR_RESUME、SCIENTIFIC_RECOVERY_REVIEW、frontend inventory；定向types.FeatureRecord、feature_io、preprocess_identity/_raw_joint_state、timebase_audit。设计交付通过；实现/生成/科学有效性未冒称通过。
