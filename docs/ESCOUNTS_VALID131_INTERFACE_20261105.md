# ESCounts VALID131 固定入口契约

版本：2026-11-05；状态：科学接口决定，尚未实现、未推断。请求的 reviewer 为 Astra/ultra，实际后端未 attested；本决定按 same-family provisional 披露。沿用已读取的 ARIS f76ec8d6b77792e2242cca049e08d0cbb7d54e19 规则。

**决定：首个完整开发评测使用 ESCounts 官方 `--only_test` 的双 offset 推断配方，数据始终是 RepCount-A VALID131。** 不采用训练过程的单窗口 validation 配方，不运行两套后择优。这是在查看本次预测前固定的推断接口选择，原因是它对应官方 README 的部署评测配方；没有读取最终 test 来选择模式。结果命名为 `ESCounts / official checkpoint / zero-shot / two-offset / VALID131 / adapted entry`，不称未修改官方 CLI 或论文 test 复现。

## 先验收的接口目标

| 要求 | 当前证据 / 验收条件 | 状态 |
|---|---|---|
| 真正外部模型 | 官方代码、官方 decoder 和 encoder；无 TempoRAC 模块、无新训练 | 资产与源码已核；实际执行待验 |
| 完整 131 样本 | valid CSV 131 个唯一 name，保留零次数视频，输出同序同集合且无跳过 | 清单已核；视频/特征待验 |
| 仅 valid 数据访问 | 不构造 train/test dataset、不读其 CSV/视频/token；模型权重作为外部固定资产可读 | 实现待验 |
| 模型、窗口与 native scorer 保真 | 以下固定契约及逐例预测、形状、GT、hash 证据 | 源码已核；一致性待验 |

## 不可变参数与数据

```yaml
contract_id: escounts-valid131-two-offset-v1
upstream_commit: f18fcf1933abb3d4f199fd12ee3cafc53581576a
dataset: RepCount-A
dataset_split: valid
expected_unique_videos: 131
valid_csv_sha256: 49695468686d894c06beb27fc4182675c762960175126a6eb8372b6148f2d148
evaluation_window_mode: two_offset
model: video_mae_cross_full_attention.SupervisedMAE
config: configs/pretrain_config.yaml
encodings: mae
use_precomputed: true
token_pool_ratio: 0.4
window_size: [4, 7, 7]
multishot: true
iterative_shots: true
shot_num: 0
select_rand_segment: false
compact: true
get_overlapping_segments: true
density_peak_width: 0.5
threshold: 0.0
scale_counts: 100
seed: 0
batch_size: 1
num_gpus: 1
shuffle: false
drop_last: false
training_or_finetuning: forbidden
selection_from_test: forbidden
```

decoder SHA256 `297fc53000417ac6ffa4e08d6876e033df89800c97632fa09d2f3945ba225559`；VideoMAE encoder SHA256 `b6d1d0b539dbdc992c3a3544a9bc5bbb0591179b615dc728f951285875d824e8`。decoder 按原 `model_state_dict` 严格加载，`model.eval()`、`torch.no_grad()`；decoder 不新增 autocast。保留官方 encoder 构造与权重映射，不把兼容改动伪称 bitwise 复现；依赖版本和任何未映射权重必须记录、解释后才执行。

## 三处允许的必要入口适配

1. **选择 valid，同时明确窗口角色。** 独立入口只构造 valid loader；把原 `Repcount_multishot_loader.py:112–137` 中控制窗口的两个 `self.split != 'test'` 判断换成显式 `evaluation_window_mode`。`self.split` 仍为 valid，其他 split 控制不能伪装为 test。特征入口只遍历 valid，不运行原 main 的三 split 构造。
2. **保留 CSV 的零次视频。** 此次新增源码/metadata 核对发现 `stu9_67.mp4` 为 count=0、275 帧、无 L 周期，segment 为 [0,275]。`Rep_count.py:78–83` 的 `count > 0` 会把特征集合减为 130；仅为 valid 的视频特征提取解除这个筛选，保留既有解码和变换。清单 131 行 count 非空、帧数均大于 64、均非 stu1_10，故不需要额外人群筛选。禁止悄悄报 VALID130 或丢掉零次数高误差。
3. **zero-shot 下消除无用 exemplar I/O。** 原 loader 在空周期上先 `durations.argmin()` 报错，随后还会尝试打开不存在的 exemplar；而 `SupervisedMAE.forward:509–520` 在 `shot_num=0` 时完全不读取 `yi`，使用已训练 `shot_token`。因此本专用入口对全部 131 行统一跳过未使用的 `select_exemplar` 计算与 exemplar 加载，提供固定零占位 tensor（单样本 C×T×H×W = 768×1×1×1，collate 后 B×1×768）并显式传 `shot_num=0`。不生成、读取、挑选任何 exemplar token；不从其他视频借示例；模型 forward 不改。这是删除未执行模型分支的输入依赖，不是新 zero-shot 算法。实现必须断言 shot_num 恒为 0；非零直接拒绝。

第 3 项修正 readiness 早先“必需 131 份 exemplar NPZ”的原生入口依赖判断：**原生 loader 确实有该依赖，但本获准的 zero-shot 专用适配可删除它。** 标签仍用于 segment 边界与 density target；不能把整个评测称作标签盲流程。以上适配在隔离运行副本/独立 wrapper 完成，保留 upstream 与补丁 diff；不改 active TempoRAC、原 CSV、原官方 checkout 或其模型实现。

## 必须逐项保持的数值路径

- **视频特征：** `Rep_count` 原 RGB 解码、`sampling_interval=1, encode_only=True`；原 test transform（min/crop 224、原 mean/std）。`save_swim_features.save_tokens`：视频尾部追加 64 个零帧；窗口起点每 16 帧；每个 64 帧窗口按原 `np.linspace(j,j+64,17)[:16].astype(int)` 取 16 帧；保留原 VideoMAE、video-feature autocast 和 token 排列。只导出 valid 视频 token；不运行 exemplar 提取。无新裁剪、重采样或分辨率调整。
- **双 offset：** 原 token 数组分别取 `[0::4]`、`[2::4]`，按原排列展开；low=`segment_start//8`，up=`ceil(segment_end/8)`。第一份截 low:up，第二份截 `max(low-4,0):max(up-4,0)`；各自保持原空间池化 `ceil(spatial_size*0.4)`。两次 forward 使用原 `thw` 传递方式，不自造位移校正。
- **预测合并：** 原 only_test 路径 FP32 输入，分别得 p1、p2；形状不同时给 p2 前置 **4** 个零，然后 `(p1+p2)/2`。仅新增断言，若该规则后形状仍不一致则报该视频与形状并停止；不能补更多零、截断、插值或切回单窗口。相同形状则直接平均。
- **target：** 完整复用原 valid density 构造（周期端点到 8 帧网格、同位置赋 1、sigma=0.5 高斯平滑），包括转 half 后求和的顺序；`g=item[3]` 是该 density sum，不能替换为 CSV count。零周期样本的 target 是同长度全零 density，g=0，不能访问空数组 argmin。
- **计数/round：** `c=sum(pred)/100`，按原 torch 路径转 FP32 后除法，再 `q=torch.round(c)`；后续保留原 NumPy round（ties-to-even），不 clamp、不 round-half-up、不拟合比例、不调阈值。保存未 round 的 c 和 q。
- **原生评分：** `MAE_native=mean(abs(q-g)/(g+0.1))`；`OBO_native=mean(abs(q-g)<=1)`；`OBZ_native=mean(abs(round(q)-round(g))==0)`；`RMSE_count_native=sqrt(mean((q-g)^2))`。OBO 不先 round g。另可保留原 density MSE/RMSE，但不能混作 count RMSE。零 GT 的 0.1 分母是官方规则，保留并单独披露该样本。该 native MAE 不能直接冒充 CountLLM 的无 epsilon 社区公式；跨论文校准另立任务。

**零 GT 与论文式指标边界：** 必须产出全 131 个预测，主开发记录为上述 `native_VALID131`。论文式除以 GT 的 MAE 在零次视频上无定义，不能加 epsilon 后仍称相同指标，也不能默删该行。若同步补充社区公式，使用原 CSV count 记为 N，在预先确定的 130 个 N>0 视频上单列 `MAE_paperform_CSV_positive130 = mean(abs(q-N)/N)`，明确这是条件子集且 GT 与 native density sum 可能不同；不得冒充全 VALID131 或论文 test 主表。零次视频另列 name、raw c、q、绝对误差和 `false_positive = (q>0)`，即使其表现很差也完整报告。社区 OBO 如需补充可按 CSV N 在全 131 行计算，但必须独立命名；不覆盖 native OBO。六域/SOTA 的资格不能由这个子集指标自行获得。

## 实现顺序与一次性验收

1. 在独立运行目录写两个窄入口：valid-only 视频特征准备；valid-only 双 offset zero-shot 评测。运行前输出固定参数、权重/源码/CSV SHA、131-ID 清单；只设置 valid 数据路径。不得先 import 并调用原三 split main。路径访问护栏允许代码/配置/固定权重，拒绝 train/test 数据路径。
2. 先做 CPU 静态/数组一致性验收：valid 清单正好 131（含 stu9_67），上述两个窗口切片和原 test 分支在相同 token 数组上完全一致；非空周期的 density/native scorer 与逐行复用原函数一致，空周期输出零 target 且可组 batch。检查模型 AST/源码确认 shot_num=0 不读取 yi，placeholder 不进入运算；这只证明代码路径，尚非推断实测。
3. 盘点真实 131 视频，按固定视频编码流程导出并 hash 131 token 文件。缺文件/解码失败不跳过；只修对应 I/O 或环境，保留失败记录。显存不足先调整资源与串行执行，不能静默降 pool ratio 或换窗口配方。任意进一步分块/数值兼容修改需单独记录并做同输入一致性，不能趁入口适配改算法。
4. 一次完整 VALID131 推断，输出逐例 name、两窗口长度、合并长度、raw c、q、native g、独立的 CSV count、native 各误差，以及参数与 token manifest。以 131 同序同 ID、有限数值、无缺失为完整性验收。对非零周期的既有真实 exemplar 如已可用，可在同一固定权重上一次核对 placeholder 与真实 exemplar 在 shot_num=0 的输出相同；不要为该核对额外生成 exemplar 或借用 train/test。
5. 独立重算 native 聚合分数并与入口分数核对；记录容差和框架版本。获得完整结果后立即进入外部比较分析，不为该入口反复 toy-test、不同时试单窗口择优、不触动健康 control。开发结果不等于论文 test SOTA。

本契约仅授权上述入口边界；执行实现端可立即据此写代码并做 CPU 验收。此设计审计没有启动 GPU、服务器、定时器、训练或下载。尚无 VALID131 性能结果。

## 核验来源

以下文件均为固定官方 checkout 的本地源码；此次直接阅读了对应路径与 relevant 全函数。只打开 valid CSV，未打开 train/test CSV 内容。

| 文件 | SHA256 |
|---|---|
| exemplar_counting_train.py | 0a87420259a023abbd4012a213337a62913c9bfd5216c3eeec2212b9a74848fc |
| Repcount_multishot_loader.py | 2cdb7c7369d3cc206fcb552b0a01c9f1b835694091d9169821aaa56a5175d80f |
| save_swim_features.py | ccc7b94c10a8619176e7d71b17c579e8348220ed719a3bae46febb4b916fb1f1 |
| Rep_count.py | efe61403e61d52f7e3861c51fd6bb245178068be1633b252cb4843d628450724 |
| video_mae_cross_full_attention.py | e1597a0ec2fcf5f76deb7b1396a9059de8b0edcca7ff8a64ba25762b5f498ace |
| configs/pretrain_config.yaml | 38960518be9cd640573c58a5aaf3e850f721dc92d445d5b8e787ddbbfed0bf67 |

## ADDENDUM：已触发的 encoder 权重映射缺陷，2026-11-05

**裁决：批准在隔离入口采用同一官方 `demo.py:119–136` 已有的完整 encoder 映射；不能照搬 `save_swim_features.py:205` 的外层提前退出。** 这是恢复既定官方 checkpoint 的必要加载修正，不改 encoder 架构、预处理、窗口或精度路径。此前接口版本已另存 `ESCOUNTS_VALID131_INTERFACE_pre_mapping_addendum_e57e87c6.md`，SHA256 为 `e57e87c6e0b3160c0868f46b3921de3bb50c306cb522e5c159500ccf0f0ba845`。

已核实而非假设的触发证据：

- 对既定 SHA 的 `VIT_B_16x4_MAE_PT.pyth` 先静态查看 ZIP/pickle 键，再用 **PyTorch 2.12.0+cpu、`torch.load(map_location='cpu', weights_only=True, mmap=True)`** 读取结构。`model_state` 有 272 项；encoder blocks 的分离 q/k/v weight+bias 共 **72** 项、fused qkv **0** 项。第 0 层 q/k/v weight 各为 768×768，bias 各为 768；第 11 层与末尾 norm 权重均存在。没有启动 GPU 或模型 forward。
- 当前 `pretrain_config.yaml` 未覆盖的官方 defaults 是 `POOL_FIRST=False`（451 行）、`SEPARATE_QKV=False`（547 行）；`attention.py:200–205` 因而构造 fused `qkv`。第一个目标 `blocks.0.attn.qkv.weight` 无直接匹配，实际必经三矩阵拼接分支。
- `save_swim_features.py:197–205` 位于内层 checkpoint 遍历结束之后；205 的 `break` 退出 177 行的**外层目标参数循环**。因此第 0 层 qkv.weight 复制后，后续参数未从 checkpoint 恢复（仍为初始化状态，不能一概称全部随机）。`demo.py:129–136` 的同类拼接及 break 则在内层循环，外层继续；该官方代码给出了既有修正依据。213 行的 v1 分支有相同缩进问题，但本契约 `use_v1=False`，不扩展修复范围。

**允许的最小实现：** 保持完整外层遍历；直接匹配照原复制，只有 `blocks.*.attn.qkv.{weight,bias}` 按官方 demo 的 **q、k、v 顺序沿 dim=0 拼接**（权重 2304×768、bias 2304）。可去掉导致外层退出的 break，或抽取与 demo 等价的映射函数。`module.` 仅作为 DataParallel 名称前缀规范化；不能通过把模型改成 separate-QKV、改 config、允许随机缺项或重新训练绕过此问题。

**执行前的强制加载验收：** 按 `_mae_forward_encoder:433–476` 的实际可达参数集合，逐项列出目标名、shape/dtype、来源键、direct/concat、复制后数值相等检查。必须覆盖 `patch_embed.*`、`cls_token`、全部启用的 `pos_embed_*`、12 个 `blocks.*`、`norm.*`；缺失、形状冲突、重复或任何可达初始化残留均停止提取。checkpoint 中预训练 decoder 等未使用来源与 encoder 对象中仅计数分支可达的参数单独列明，不能用宽泛的 `strict=False` 吞掉。将适配结果与官方 demo 映射在同一 CPU state 上核对，要求所有 encoder 可达参数逐元素相等；参数覆盖核对通过后直接继续既定 VALID131 流程，不新增方法调参实验。

附加源码 SHA256：`demo.py`=`bce255b1274d6f3feedd42cf671125e844354f158c6865590e302fd704216f38`；`slowfast/models/attention.py`=`8d166eb33cc6a6dedc2042cde225e6c95d4e072f9e518fb0b4172a003c8dcfc3`；`slowfast/config/defaults.py`=`23675eecf2492df01c75e7c63d033116ae236d4796935b26b392c876ad41253b`。本补充只核实触发条件与允许边界；完整参数覆盖验收待实现端执行，不冒充已通过。

