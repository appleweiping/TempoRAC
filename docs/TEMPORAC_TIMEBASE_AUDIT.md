# TempoRAC 时间坐标审计 v1

## 范围与验收（实施前记录）

基线：`3e9ae828979bb7538da733835d286fcb395ccfbb`。本变更只新增 CPU 时间映射、
支持信息和旧缓存诊断，不修改 `preprocess_resampled_v2`、训练图、loss 或冻结协议。
采用 pinned ARIS `research-implement-feature`，`ASK=never`、balanced/draft。
依本次仅三个文件的范围约束，SPEC、假设和 build note 合并在本文；不自动提交。

| Rung | MUST 验收 | 状态/证据 |
|---|---|---|
| F0 | CLI 能读取旧 NPZ，输出 JSON 诊断且不伪造坐标 | E 盘现有 Python 3.12 执行 `assert main([fixture]) == 2`，退出 0；JSON 报不可恢复 |
| F1 | 保留源/目标物理时钟、左右原始源索引和权重，拒绝重复/逆序 | `test_irregular_clock_retains_original_indices_weights_and_support` 与 6 个非法 clocks 用例通过 |
| F2 | 区分 observed/interpolated/extrapolated；记录 gap、阈值和支持 | `test_support_threshold_is_explicit_and_interpolation_is_never_observation` 等通过 |
| F3 | 不规则时钟往返、端点、静止轨迹、缺失数据 CPU 测试 | 边界/静止/single-anchor/JSON/只读 CLI 用例通过；见执行记录 |

## 假设账本（实施前确定）

| ID | 未确定项 | 采用决策 | Class | Source |
|---|---|---|---|---|
| A-001 | 时钟单位 | 调用者显式提供非空 `clock_unit`，所有 gap 使用相同单位；不推断 FPS | interface | default |
| A-002 | 重复时间的多值含义 | 源/输出时钟必须有限且严格递增；重复不静默平均 | semantic | default |
| A-003 | 插值何时可用 | `max_interpolation_gap` 必填、非负且有限；仅两已观察锚点间跨度不大于它才支持；没有科学默认阈值 | semantic | default |
| A-004 | 外推数值 | 端点保持，左右索引相同，权重 1/0；标记 extrapolated 且始终 unsupported | semantic | default |
| A-005 | 坐标端点 | 行坐标为 sample center，可为小数；闭区间 [0,N-1]，禁止越界或隐式端点夹紧；半开事件区间分别映射两个端点，不增减一 | semantic | default |
| A-006 | 观察支持粒度 | 必须显式传一维 source_observed；不从有限数值猜观察。逐关节/通道缺失须各自建 map，不能用帧级 map 冒充逐关节支持 | interface | default |
| A-007 | 旧 v2 缺失信息 | CLI 只审计现有 NPZ；缺少源时钟/原观察 mask 不可恢复，source_length 和哈希不能逆推 | semantic | user |
| A-008 | 输出/调用接口 | schema `temporac.timebase-audit.v1`；独立模块 API/CLI，不接入旧流水线 | interface | user |
| A-009 | 浮点时钟匹配 | 仅精确相等的 source/output time 标为 observed，不以容差吸附；近邻标插值 | semantic | default |
| A-010 | gap 计数 | `gap_span` 为完整两锚点间物理跨度（外推为离端点距离）；`missing_source_rows` 仅两锚点之间原数组缺失行数，不猜丢失视频帧数量，外推为0 | semantic | default |

## API（预定）

`build_timebase(source_clock, output_physical_clock, source_observed, *, clock_unit,
max_interpolation_gap)` 返回 `TimebaseMap`。`resample_values(values)` 只按 map 权重
生成数值，不把插值数值当 observed；调用者必须保留 `supported`。`row_to_clock` /
`clock_to_row` 做 sample-center 连续坐标往返。`as_dict()` 提供完整可保存 JSON。

没有输出真值计数、没有 scorer 或训练入口。CPU 测试通过仅是机械证据；完整自然数据
支持分布、后续新 adapter 接入和独立语义 review 仍须单列完成状态。

## 使用

```python
from pams.temporac.timebase_audit import build_timebase

mapping = build_timebase(
    [10., 11., 12., 20.], [10., 11., 15., 20.], [1, 1, 0, 1],
    clock_unit="seconds", max_interpolation_gap=2.,
)
filled = mapping.resample_values([2., 3., float("nan"), 7.])
receipt = mapping.as_dict()  # 保存此完整映射，不只保存填补数值。
```

输出同源时钟单位；插值阈值由后续协议明确设定。源/目标都至少两行，支持一个观察
锚点（其余输出全部外推且不可用），全无观察则拒绝。缺失行可含 NaN，观察值必须有限。
`TimebaseMap` 数组为不可写副本，原始时钟和 mask 后续修改不会改变 receipt。

旧缓存命令（仓库已安装或 `PYTHONPATH=src`）：

```text
python -B -m pams.temporac.timebase_audit path/to/legacy-v2.npz
```

stdout 为 JSON；退出码 2 表示缺少可验证 provenance，1 为读文件/格式错误。
`index_shaped_clock=true` 仅提示存在行索引式时钟，不能单凭它证明由 v2 生成。
`source_length`、旧哈希和全1 mask 无法恢复原时钟及逐关节缺失；即使 NPZ 带同名
扩展字段，此 legacy 诊断入口也不验证新 receipt、不宣称可恢复。

## 执行记录与限制

- F0：Python 3.12.13，E 盘已有环境，CPU JSON fixture 实际读取完成；无 stub。
- F1–F3：现成 Python 3.14.5 / NumPy 2.4.6，初测 `13 passed in 9.26s`，最终
  `13 passed in 7.67s`，退出0；两份 Python 文件 `ruff check --no-cache` 退出0。
  这只是临时工程验证，不表示整个项目支持3.14。Python 3.11补验由主任务环境完成后记录。
- pytest 命令：`python -B -m pytest tests/temporac/test_timebase_audit.py -q -p no:cacheprovider --basetemp=E:/ICASSP_TempoRAC/aris-work/tmp/pytest-timebase-314`。
  设置 `PYTHONDONTWRITEBYTECODE=1`、`PYTHONPATH=<repo>/src`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，TEMP/TMP 指向 E 盘。
- 仅新模块单测，无自然数据/GPU/网络下载/冻结 v2 语义修改；没有证据说明旧缓存已修好。
- Type-B 假设审查由主任务统一进行；本文不把自测通过称为独立正确性鉴定。

## Supported-version verification — 2026-11-04

Root repeated the scoped test under CPython 3.11.2 / NumPy 1.26.4 in the E-only `temporac-audit-py311` environment: 13 passed in 2.35 seconds. Ruff 0.14.4 exited 0. `PYTHONDONTWRITEBYTECODE=1`, TEMP/TMP and pytest basetemp were on E:. No model/GPU or natural-data test was run.
