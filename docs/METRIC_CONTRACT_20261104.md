# TempoRAC 评测契约建议 R1 — 2026-11-04

**结论：可立即实现 CPU 参考评分器与反例，但“官方 AvgMAE/AvgOBO 完全复现”仍 pending，不能启用独立测试或替换现有主指标冻结。** 本文只规定评测接口，不改模型/解码器/训练目标。请求角色：`gpt-6-astra`，任务 `/root/metric_contract`；无 backend attestation，实际后端身份未独立证实；OpenAI 同族科学判断至多 provisional。

| 验收要求 | 本次状态与证据 |
|---|---|
| 官方实际源码、固定 SHA、精确行号 | 通过；下列 S1–S5 及本地 `metric_sources/plus/` |
| 明确匹配、分母、区间、失败策略 | 通过，限原样行为识别及**拟议**规范；§2–3 |
| 官方 count 指标与完整可执行转换链 | 未达标；论文公式歧义、count scorer 未找到、预测字段/列数/时钟未闭合 |
| A180/B78、历史负结果、test 隔离 | 保持；B 仅候选，9 个相关 source components；本次未打开任何数据标签、未跑模型 |
| 下一项具体实现 | §4 的一个 CPU scorer/adapter 交付；不先启动正式评分 |

## 1. 可复核的一手证据

官方仓库 `yinntag/MultiCounterPlus` 固定提交 **`d15c767d82dac2f7a5df9fde1d7f6b5a4f491062`**。下列链接均锁该提交；已通过 GitHub connector 读取实际文件，先读 main 的五文件又与固定提交内容逐一相等核验。

- **S1** [multirep_eval_api.py](https://github.com/yinntag/MultiCounterPlus/blob/d15c767d82dac2f7a5df9fde1d7f6b5a4f491062/mmdet/datasets/multirep_eval_api.py)：blob SHA `e5c188aa032957fa628062df645a7e1fe8d59b6d`；181–260 空间 tube-IoU，311–394 贪心匹配，438–463 仅匹配实例，534–564 首配置，589–686 period AP，775–784 默认参数。
- **S2** [eval_multirep.py](https://github.com/yinntag/MultiCounterPlus/blob/d15c767d82dac2f7a5df9fde1d7f6b5a4f491062/tools/eval_multirep.py#L32)：blob `ab680b06d5c4d1c62b54b3562f843c4eba5a959a`；32–45 固定 stride=3、bbox、evaluate→accumulate→action_ap。
- **S3** [data_pre/video2frames.py](https://github.com/yinntag/MultiCounterPlus/blob/d15c767d82dac2f7a5df9fde1d7f6b5a4f491062/data_pre/video2frames.py#L21)：blob `9bfdb73cb979cd156cb81bb8df244da1e9115a97`；21–30 period 长度=e−s、range(s,e)；106–114 periodicity binary 却含 end，且**确实输出 GT periodicity=原 period 区间**。
- **S4** [result_convertor.py](https://github.com/yinntag/MultiCounterPlus/blob/d15c767d82dac2f7a5df9fde1d7f6b5a4f491062/tools/result_convertor.py)：blob `4a8e17610d0980c50a2010465d60eef82a0f97ff`；5–30 以0.3阈值输出连续活动段 `periodicity_converted=[start,last_active,mean_score]`，不是逐周期边界。另一 [demo/period_convertor.py](https://github.com/yinntag/MultiCounterPlus/blob/d15c767d82dac2f7a5df9fde1d7f6b5a4f491062/tools/code_for_demo/period_convertor.py#L26) blob `234d24099b236358eed24c181715df50df1be8c9`，26–69 从累计周期质量输出 `periods_converted`，但每项只有两个端点，S1:547 要求第三列 score；其阈值变量未参与周期解码。
- **S5** [MultiCounter 论文](https://arxiv.org/html/2409.04035v1#S5.SS2)，§5.2 Eq9–10（[PDF p5](https://arxiv.org/pdf/2409.04035) 同样可见）：N 是视频数、M 是该视频重复动作人物数；Eq9 写成有符号相对差，缺绝对值；Eq10 的 ≤1 位于人物平均后。不能自行把排版意图当可执行规范。论文描述 Period-AP 对已匹配 TP 对求值，没有公开证明“必须平均所有实例 IoU”；因此 first-config 是**可验证的限制**，并非已证明违反作者意图。

原 [MultiCounter 仓库](https://github.com/yinntag/MultiCounter/tree/8fdaa7c9fa02cec8ed3b22091aabc380f60c3bf3) 该提交根只有 README 与两个预处理脚本，无 scorer。Plus 仓库限定关键词 AvgMAE/MAE/OBO 搜索只定位 README/无关字符串；不声称穷尽所有历史分支。**候选树差异**：旧审计的 `tools/prepare_multirep_data.py:116–142` 使用 e−s+1 且遗漏 GT periodicity，不能归罪于 S3 官方版本。matched-only/first-config/预测转换不兼容则在官方版本独立复核成立。候选树尚无 exact upstream parity，保留其原报告。

## 2. 原样复现通道 OFFICIAL-CODE-CONDITIONAL（诊断，尚未跑通）

**输入与时钟。** GT 是 COCO-like `videos/categories/annotations`；每人 annotation 需唯一 `id,video_id,category_id,bboxes,periodicity`。bbox 为逐帧 xywh 或 null；periodicity 为**逐周期** [s,e]。DT 是对象列表，字段 `video_id,category_id,score,bboxes,periods_converted`；每个 period 为 [s,e,event_score]。loadRes 给预测顺序编号（`multirep_api.py:208–256`）。视频顺序/GT ID 必须满足 S1:456 的 `real_gt_id=gt_index+1` 假设；若不满足，原样通道报告不适用，不静默修复。

**实例匹配。** 每视频/类别按 track score 降序、稳定 mergesort；最多100预测。tube-IoU = 全帧 bbox intersection 面积之和 / union 面积之和；一侧 null 时其另一侧面积进 union。逐预测选择当前未匹配且 IoU≥门槛的最大 IoU GT，一对一；不是 Hungarian，也不是周期内时空乘积。默认类别 person=1、area=all [0,10¹⁰]、IoU=.50:.05:.95、maxDets=[1,10,100]、useCats=1。S1 的 period 分支实际只用首个 config：默认 **实例 IoU=.50、首类别、all**；不可声称平均了十个实例阈值。

**Period 分母。** 对该 config，只有匹配 GT 身份的所有周期进入 N_GT；其对应 DT 的所有周期进入预测集合；未匹配 GT/DT 整个人都被排除。再把 GT 人物ID作为 temporal video-id，在同一身份内匹配周期，跨所有身份/视频按 event_score 排序。对 τ=.50:.05:.95 分别贪心锁定最高 tIoU 尚未占用 GT；重复/无匹配是 FP。tIoU=交长/并长，长度严格 e−s、无+1。R=累计TP/N_GT，P=累计TP/(TP+FP)；AP 是 precision envelope 对 recall 增量积分，**不是101点平均**；mAP=十个 temporal AP 均值、AP50=第0项、AP75=第5项。非视频/人物宏平均。内部 [0,1]，论文显示百分数时×100。event score 的 ties 原实现 argsort 不保证稳定，不能承诺重排不变。

**不完整项必须直报。** S2 要 `test_stride3.json` 和 `result/results_periods_converted_stride3.json`；S3/S4 与 test.sh 生成的名称、字段及采样坐标均不闭合。S1 默认不校验轨迹长度，zip 可截短；不应在拟议评分器继承。无 GT period/空集合的数值行为也未认证。只允许为兼容运行记录独立 patch（例如 np.float），不得称修改后的结果为未改官方原码。AvgMAE/OBO 的身份漏检、误检、rounding、零 GT 处理**全部仍为官方未知**。

## 3. 拟议完整人口通道 FULL-v1（待冻结；明确非官方定义）

**共同输入。** 对完整冻结视频列表每视频必须有成功或失败记录；每条预测有 `prediction_id,track_score,count,periods`、完整源时钟 bbox/support 映射、decoder/config/checkpoint SHA。count 为非负有限数，使用各方法冻结的原生读出；不在 scorer 统一四舍五入，也不从 count 造区间。GT count/period 仅由隔离 evaluator 读入。缺失整个预测视频视作所有人漏检、零输出，并保留失败原因；格式错误/NaN 先使整次评测失败，不能删行或悄悄记零。GT count≤0 或无重复人物视频须先在 dev 形成明示处理补丁；未冻结前停止正式 count 评分，禁止 ε 掩盖分母问题。

**匹配与坐标。** supplied/oracle-track 设定按已冻结 identity 映射；common predicted/native 设定按上述全视频 tube-IoU≥.50、score 贪心一对一，使用真实显式ID，**所有输出 track** 均保留，无100条截断。固定 tie key=(video_id,prediction_id)，GT IoU并列按GT ID；不以 count/period 选择身份。所有轨迹长度须等于完整源帧数。区间统一为零基源帧边界坐标的实数 [s,e]，0≤s<e≤T，tIoU仍用 e−s；官方GT数值不擅加1。明确标为 inclusive-frame span 的外部输出才经具名适配器 [a,b]→[a,b+1]。输出行号/stride须经可验证映射回源时钟，不能猜乘3。插值/外推/缺口不删除对应 GT 周期。TempoRAC 事件峰尚非完整周期：period decoder 未核定时 PeriodAP 标记 unavailable，不能把相邻峰或活动段充作合法周期。

**计数：分清人物宏平均与完整端到端。** 对视频 i，全部GT人物集合 G_i、人数 M_i、GT计数 g_ij>0；匹配预测计数 c_ij，漏人置0；U_i 是未匹配预测集合。

- 提议 supplied-track 主指标 `AvgNAE_GT↓ = mean_i [sum_j |c_ij−g_ij|/g_ij / M_i]`；`AvgOBO_GT↑ = mean_i [sum_j 1(|c_ij−g_ij|≤1)/M_i]`。这与本地 metrics.py:89–95、215–266 的人物→视频层级一致，但**不是已认证官方 AvgMAE/OBO**。未匹配预测对这两个 GT 指标不加分也不计罚，必须同时报下一行。漏人 g=1 在标准 OBO 仍为成功，不能把 OBO 解释为身份检出率。
- 提议完整端到端主指标 `FullCountNE↓ = mean_i [(sum_j |c_ij−g_ij| + sum_{u∈U_i} c_iu) / sum_j g_ij]`。每个多余预测的重复次数按相同计数单位加罚；分母仅GT总计数，不能通过多报身份稀释错误。它是按GT计数加权的误差，**不得改名 AvgMAE 或与其数值混排**。
- 配套 `FullIdentityOBO↑ = mean_i [sum_{j matched}1(|c_ij−g_ij|≤1)/(M_i+|U_i|)]`：漏人永不成功、多余人扩大分母，即使其count=0也计罚。该名字明确区分标准 OBO。同报 matched/M_i、未匹配预测数、原始 count；不能仅选择有利指标。

**Period：** `FullPeriodAP@τ` 沿用 temporal匹配/AP积分，但 N_GT=**该轨道全部GT人物的全部周期数**；未匹配GT周期永久为FN，未匹配预测人物的所有周期为FP；匹配人物错误/重复周期也为FP。按全体event score全局稳定排序，ties=(video_id,prediction_id,event_id)。所有输出预测进入集合，不根据质量/支持率删区间。零预测且N_GT>0时AP=0；N_GT=0的轨道记 undefined，不能用0与有GT轨道混平均。mAP/AP50/AP75仍是 temporal阈值均值/对应项，不平均实例阈值。并排保留 OFFICIAL-CODE-CONDITIONAL；两者差值是分母变化，不是模型提升。

## 4. 数据、统计、下一项交付

A 保留**180**视频；B 保留**78候选**、overlap **102**、当前byte/path unknown **0**。历史暴露和合成recipe未完结，B不能称已认证独立测试集。使用已归档同源连接分量：A相关23、B相关**9**；同一checkpoint/预测评分A/B，不为B调阈值。三训练seed分别完整评分，报告mean和sample SD（ddof=1）；AP必须每seed独立汇总再平均，禁止把3seed预测合并一份AP。成对cluster bootstrap按完整来源图分量有放回抽样、成员视频成组加权、重算整套指标；B九簇区间与删一簇敏感性均显示，不能当78独立样本。不得用显著性或仅报B掩盖A失败。

**一个具体下一动作：** Sol 在 TempoRAC 项目新增独立 `multirep_metric_contract_v1` CPU adapter/scorer及fixture，不改冻结旧metric。一次交付覆盖：不等人数视频区分macro/pool；漏人g=1和g=10；多余/重复身份；高置信FP在TP前；未匹配GT周期；乱序/非连续ID；边界0/T、inclusive span/连续端点、stride往返；空预测、NaN、长度不一致。用人工区间两GT一TP证明 conditional AP=1 而 full AP=.5；将一次高置信错误预测放在该TP前应得 full AP=.25。前者须**原样官方S1函数**与新参考函数双算，仅使用人工fixture，不访问test。完整dev对照只在适配器及数据时钟契约接受后运行。

正式冻结仍需：官方count实现/作者可核实说明（若不可得就明确采用并命名FULL-v1）、release/split身份、GT/DT边界及采样映射、TempoRAC合法period读出、共同track与阈值/checkpoint选择规则；均 pending。保留 X0失败、v15覆盖失败和不利诊断。CPU契约通过不等于性能/数据/主结论通过。

读回执：ARIS pin `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`；完整读 experiment-plan 与 output-versioning/output-manifest/output-language/output-composition；安装/源skill SHA256 均 `61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a`。本次无需独立全实验计划/追踪表，任务指定交付覆盖其评测子契约。源码及取证索引在 `E:\ICASSP_TempoRAC\aris-work\metric_sources\`；未启动GPU、安装、SSH、外发、commit/push。

