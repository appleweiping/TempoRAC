# TempoRAC：面向 CVPR 的科学恢复审查

审查日期：2026-11-04。范围：只读审计与恢复设计；未修改模型、未读取自然数据或封存测试集、未启动 GPU、未推送。本文是供主任务采用的具体设计建议，不把任何历史未通过协议改写为已经通过。

## 1. 结论与可核对验收表

保留 TempoRAC 的核心问题和推断图：每人独立状态、共享编码器、局部相对节奏、slow/medium/fast 响应专家、detached NUDFT 软路由、正分母 NOLA、每人只解码一次。现在最该交付的是完整自然数据上的可信计数证据，而不是 A005 下一轮证明文本。旧 27-job campaign 既不是完整 CVPR 实验，也不应按原依赖直接恢复。

当前存在两个新的代码级优先问题：重采样适配器把物理时钟改为 0…319；高覆盖自然候选的主要分支并未使用学习到的 teacher phase/reconstruction。解决这两个接口及全序列伪目标来源问题，比追加训练或修改 coverage 阈值重要。

| 验收项 | 当前状态 | 现有证据 | 真正的通过条件 |
|---|---|---|---|
| 明确主要科学主张 | 部分 | v4 主张是局部软路由在节奏漂移下优于 matched nonlocal control，并保留 clean 性能 | 新协议同时写清自然主指标、drift 次级机制检验、比较集和适用轨迹设定 |
| 完整、无泄漏数据 | 未达标 | 仅 110/51 videos、268/134 identities 的 partial GT-bbox-assisted supplied-track cache | 完整合法来源、源视频 connected-component 去重、split manifest、所有合格样本不按结果筛选；独立评估集未参与候选选择 |
| 输入时钟和评测坐标可信 | 未达标 | resampled_v2 插值后 clocks=arange，mask=ones，source_length 保留原值 | 物理时间与重采样坐标显式双向映射；观测与插值 mask 分离；边界往返和 irregular-clock 测试通过 |
| 自然伪目标可训练且与真实计数单位一致 | 未达标 | v15 213/268、109/134 为探索性 coverage；v12/v15 几何分支绕过学习输出；v9 可只保留两个事件 | 全序列 target provenance、事件/时长 coverage、几何计数器 baseline、独立 unit-error 统计；不能只报 identity acceptance |
| 原始自然有效性 | 未达标 | X0-only NMAE 0.9915197、OBO 0、预测 0–4 对真值 4–92 | 完整自然评测有效预测与误差分布，优于冻结比较集，有三 seed 和成对不确定性 |
| 至少三个耦合组件及消融 | 部分 | 图中已有 local cue、tempo experts、NOLA/one-decode；没有实测消融 | 分别移除/替换后完整训练和评测；不把一次 decoder 调用等工程不变量单独宣称新颖性 |
| Observation 动机证据 | 未达标 | 仅问题假设和候选故障分析 | 完整 development 分布下 tempo 非平稳性、全局 cue 偏差、global counter 错误随漂移强度增加的图/表 |
| Loss 少于三项 | 未达标/有冲突 | F11 有 reconstruction、correspondence、两种 phase penalties；F14 有 BCE、正负 margin | 明确当前项目旧冻结目标优先，或版本化简化；不能把多项重新命名成一个 family 假装满足 |
| 强基线与公平协议 | 未达标 | 旧 27 jobs 多为 shortcuts，不包含完整竞争方法复现 | native MRAC 与 common-track 结果分表，输入、监督、数据、预算披露；实际复现清单见第 6 节 |
| 三 seed、统计、超参和效率 | 未达标 | 正式 jobs 未跑，主表空 | 三 seed mean/sample SD、video/source-component cluster CI、预先固定比较；超参曲线、吞吐/显存/人数量曲线 |
| CVPR 格式与匿名合规 | 未达标 | 旧稿是 ICASSP 五页 pre-results draft | 官方 CVPR 模板、八页正文、匿名、LLM Usage Statement、参考文献核查、逐页渲染 |
| 6 数据集×7 框架、6 域 SOTA 偏好 | 未达标，范围须明示 | 当前 RAC 项目原协议不是六域推荐项目，已有数据不足 | 不把六种扰动叫六个数据集；若该通用偏好被指定为本次硬要求，需另列真实数据/资源且实际满足，不静默打勾 |

## 2. 本次实际审读与证据边界

完整读取：`E:\AGENTS.md`；活动候选 checkout 的 `AGENTS.md`、`RESEARCH_BRIEF.md`、`TEMPORAC_HANDOFF_20260817.md`、`docs/TEMPORAC_SYNC_STATUS_20260930.md`；`E:\ICASSP_TempoRAC\TempoRAC-main` 的 20260818 experiment status 和 preprocessing/runtime candidate 两份长报告。

定向读取：活动候选 checkout 的 `refine-logs/FINAL_PROPOSAL.md`（问题锚点、七不变量、数据、teacher、F11、F14、比较和评测约束）及 `EXPERIMENT_PLAN.md`（claim map、baseline、job inventory、预算、部分数据晋级限制）；代码 `preprocess_resampled_v2.py`、`campaign_runtime_candidate_v9.py`、`v12.py`、`v13.py`、`v15.py`、`v15_landmarks.py`。没有逐字重新审计百万字节级 A005，也没有声称验证 server checkpoint 原始字节。

本次已查官方 CVPR 2027 dates/author guidelines；论文一手范围仅 MultiCounter arXiv 摘要和 PAMS 的 CVF 搜索可见摘要（PDF 获取返回 403），未完整精读两篇；没有完成 MultiCounter+、所有 2026 新作的穷尽查新。以下 novelty 是待验证判断，不是通过 novelty gate。

20260930 状态优先于 20260818 旧状态：当时本地 TempoRAC 测试 157 passed/1 skipped，全套 1179 passed/7 skipped，原 27 jobs 仍未运行；这说明旧“缺失 operator_v1”不应在未检查当前文件前继续当今日阻塞。这里没有重跑测试，其结果只作为历史记录。

## 3. 保留的科学约束与可以一次性撤销的工程门禁

必须保留：

1. 人级输出和状态隔离；不回到场景总数、不恢复 WARP-PHASE 或其他撤回路线。
2. count/period/density/boundary 标签不进入 learner、伪目标、路由或 checkpoint selection。外部 detector/pose/tracker 的监督与 GT bbox 帮助必须披露。
3. 完整合格人口、缺失/拒绝样本和负结果保留；v46/v62/v63 和监督 MotionBERT 数字不能移入 TempoRAC 行。
4. 源视频级 split 防泄漏；当前反复检查的 val 属于 development，不能再叫独立未见证据。新 independent test 的访问范围仍要按本次授权落实。
5. 每个结果绑定代码/config/seed/environment/数据 manifest/checkpoint/raw predictions；三 seed 和成对统计；固定预算、健康检查、后台无窗口。
6. 当前 partial cache 只用来开发和附录定位；主表必须完成正式数据协议，不能把旧 K7 PASS 当晋级许可。

可以在一份简短、明确的恢复协议中废止并保留旧档案：

- 不再把 FD、ptrace、token origin、57-owner 顺序等 A005 安全状态机证明当科研训练依赖；用实际进程隔离、feature-only manifest、只读 evaluator labels 和运行日志实现需要的边界。
- 把全量重复预审改成一次代码/数据/指标 preflight；变更后只重验受影响链路。事实性哈希不等于必须维护百万字节双份规范。
- evaluator 因基础设施故障可在相同冻结 predictions 上幂等恢复，保留尝试记录；“进程重启一次即永久不能评测”不是统计独立性的必要条件。不得借恢复改变模型、阈值或预测。
- 固定 40/20 个 checkpoint 和失败不能恢复可以改成可验证 resume、预定 schedule、固定选择规则。不能因此减少正式完整训练成 debug runs。
- 80% certificate coverage 是旧项目选择的有效性要求，不是 CVPR 规则；应如实记录未满足，不能调低后叫旧门禁通过。新协议改为统计完整 identity/time/event coverage 和伪目标误差，其科学理由是旧证书不对应真实任务损失。
- 自然数据“每个 interval 恰一 pulse、union 外零 pulse，任一错即杀死”要求零错误，不是实际计数算法的合理总准入条件。新协议应报告漏计/多计、period metrics 和完整计数误差；synthetic fixture 的精确不变量保留。

这些变更由主任务依据用户本轮授权一次性写入新版本及 superseded 条目即可；本文不要求先补完旧 A005 Round 9 再做科研。

## 4. 两个代码事实改变了恢复依赖

### 4.1 入口全通过仍可能破坏物理时间和支持信息

`preprocess_resampled_v2.py` 约 139–173 行先在 `[time[0], time[-1]]` 上均匀插值，随后返回 `np.arange(pose.shape[0])` 并把所有 frame masks 设为 1。原 `source_length` 却保留。于是模型实际看到的是归一化采样序号，不能据此声称测试过真实 irregular clocks；原始 period interval 到响应的转换也必须额外恢复，否则 localization 评分不可靠。对零几何边累积小扰动也可能制造非零运动，应保留计数并测敏感性。

具体恢复项：保存 source clocks、output physical clocks、row-to-source map、original observation mask、interpolation/support mask 和 gap length；禁止把长 gap 当观测到的完整周期。先用 CPU 人工已知 period 边界验证正反映射、端点/半开区间、重复 clocks、长短 gaps、静止段；再对全 train/development 做支持分布审计。完整原视频重建 pose 时优先保留有效物理时间，避免用适配器永久掩盖旧 320-row 缓存局限。

### 4.2 v15 高覆盖不证明 learned teacher 已成功

`campaign_runtime_candidate_v15.py` 顺序是 v13 成功则返回，否则 v15 landmark fallback；v13 又是 v12 完整链或 v9 stable pair。`v12.py` 和 `v15_landmarks.py` 都调用 model，但其 `_code/_phase/_raw_reconstruction` 没有进入 target。后续 phase 来自常量输入加 geometry landmark closure，reconstruction 来自多周期 DTW。v9 才把 model phase 送入 closure。这是混合来源系统。

因此：

- 不应为了 v12/v15 target 再训练三个 synthetic teachers；它们对这些分支的有效输出没有因果作用。
- 不能全局直接删 teacher，因为 v9 分支仍读取 phase，且 X0 路径未在此证明可替换。
- 必须给每个 target 标记 `full-chain geometry / stable-pair hybrid / rejected`，统计每条路径的 accepted identity、事件数、覆盖时长、缺失段与真实评估误差。
- stable-pair 只产两个局部事件时，未标注的其余时间不能当负例。要么从完整序列目标剔除该路径并记 coverage 下降，要么目标显式提供 validity mask，仅在支持区监督。不能用“每条 track 已有 target”掩盖漏掉多数周期。
- 必须有独立的几何伪计数器 baseline：直接对 frozen target chain 计数，不训练 response。若 response 不能超过这个来源，论文贡献不能归因于 learned counter。

## 5. 推荐下一段：一次性、最小的科学修订

推荐恢复包 R1 只修输入/目标语义和执行依赖，不改 TempoRAC 推断骨架、不增加模块：

1. 将现有完整链几何生成器明确命名为“训练用、无计数标签的几何伪目标生成器”；保留来源记录。把 teacher 成功泛化的措辞撤下。v9 两周期 fallback 默认不晋级为完整序列 target；若保留，必须带监督有效区。
2. 先冻结一个候选（优先审计现有 v15 路径，不再并行发明 v16/v17 阈值），在完整训练集做 count-blind 时间变换、周期保留、静止/非周期负例、gap 及 full-sequence coverage 检查。门禁允许揭示失败，不允许只为 acceptance 更高选参数。
3. 在已作为 development 的自然集合，对冻结几何伪计数器和现有 X0-only checkpoint做一次完整 evaluator-only 误差报告。该报告是路线质量判定，不反向选择同一报告中的若干阈值；所有观察纳入开发历史。若连简单全局几何计数器都显著更差或大量整段漏失，应先停止响应训练并修目标语义，不继续昂贵 teacher pretraining。
4. 对来源明确且可训练的目标，沿用现有 response 图完成第一个全数据、全 schedule canonical run 和一个 capacity-matched global/uniform control；这两个就是正式矩阵首 seed，可复用，不额外发明 smoke GPU。代码级检查只在 CPU 完成。初始完整 run 健康后再并行其余 seed/基线。
5. Loss 简化若本次必须满足“少于三项”：推荐使用现有 class-balanced pulse BCE 作为唯一响应优化项，X0 与 natural 只是数据混合；正负 margin 两项同时删除并留下原 F14 对照。不要新加 consistency/regularizer 把恢复变成新方法工程。X0 analytic targets 已存在，几何 target 分支也无需 F11；任何仍依赖 learned teacher 的路径必须单独审查并列明其真实目标项。若主任务保留旧 F14 以便最快复现，则该偏好记未达标，不能改名糊弄。
6. 仅在确定数据完整性和 train/development 边界后，用一次 evaluator 冻结确认协议接入独立评估；未授权 test 不碰，已有 val 不重新命名 test。

这里第 1/5 项是明确的协议/训练设计修订建议，不是“代码原本如此”。它们需要在主任务的恢复协议中列差异和理由，不能静默实施。若当前授权只允许无语义恢复，先完成 CPU 审计、完整数据构建和现有 counter evaluation；不启动语义不明的 27-job campaign。

## 6. CVPR 完整实验的最小矩阵

主评测先固定为完整 MultiRep、按 supplied/oracle 与 common-predicted tracks 分设定。自然 clean 的官方 AvgMAE 为主、AvgOBO 为共同报告项，Period-mAP/AP50/AP75 按能输出合法 period intervals 的方法报告；路由 drift 检验是预先固定的机制证据，不能替代自然主表。计数峰不是完整周期 interval，不允许未经验证直接以相邻峰拼接 Period AP。

| 实验块 | 最小内容 | 是否需新增训练 |
|---|---|---|
| 主表 A：native MRAC | MultiCounter、MultiCounter+ 官方完整协议；明确各自数据和监督 | 能用合法官方权重则复评；否则完整训练 |
| 主表 B：common tracks | Track-PAMS、适配后的 PoseRAC；按输入可比性接入 RepNet、TransRAC、DeTRC、D2-STX 等强相关方法 | 逐个标明权重/监督/训练数据；不把不可比 published number 混为 matched result |
| 内部强控制 | Geometry pseudo-counter；global NUDFT；uniform；capacity-matched single/expert control；TempoRAC | geometry 零训练；其余正式三 seed，若后融合干预可缓存则单列 post-hoc 行 |
| 组件消融 | local→global cue；三专家→参数匹配单专家；soft→hard/uniform；NOLA→固定 boxcar merge；decode once→window decode+merge | 训练图改变的重新完整训练；纯解码改动可复用 frozen response，并明确区别 |
| 机制与 Observation | tempo variation 分布；3×3 expert 专长；速度轨迹/路由/响应对齐；clean→piecewise drift 误差与CI | 复用正式模型；预固定 drift，不按效果挑曲线 |
| 支持/鲁棒性 | irregular clocks、periodic/nonperiodic missingness、occlusion、identity switch、人员数变化 | 复用正式模型，固定 corruption seeds；与有效输入支持范围对应 |
| 超参与效率 | window/hop、路由温度或可靠度阈值的完整 validation 曲线；显存/吞吐/人数量；包括 pose/tracker 和排除 pose/tracker 两种耗时 | 缓存可支持的推断超参不重训；需训练超参预算单列 |

主表 B 的候选不是本次已经核实实现的完成清单。先做一页 baseline manifest：官方来源、license、checkpoint、输入、监督、能否 common-track、完整运行命令和预计资源。PAMS 已覆盖 pose+self-supervised+multi-scale+adaptive counting；需要实验证明 TempoRAC 的局部路由与人级连续重建增量，而不是泛泛“多尺度”新颖性。MultiCounter 已定义 MRAC，不声称首次多人计数。[MultiCounter 一手摘要](https://arxiv.org/abs/2409.04035)；[PAMS 的 CVF 论文入口](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)。

最少自然 clean 三 seed；point estimates 和 paired video/source-component cluster bootstrap CI 都报告。若视频共享源片段，按源 connected component 聚类。比较 multiple arms 应预定 primary contrast，并对其余主张控制多重比较或明确探索性。stationary clean 保留不能用“p>0.05”证明等效；需预先指定实际容忍界及其 CI。原 v4 的 5% drift gain、0.01/0.02 clean tolerance 可作为历史目标保留，但必须写清指标单位与分母，不机械套在不同 AvgMAE 定义上。

## 7. 最小依赖图与下一交付

```text
现有代码/负结果/数据位置清点 ──> 一份恢复协议 R1 + 预算 + split/metric manifest
                                      │
       ┌──────────────────────────────┼─────────────────────────┐
       v                              v                         v
完整合法 pose/track cache       CPU 时钟/target/scorer检查     强基线 manifest/环境准备
源视频去重+清晰train/dev         target分支与支持区审计          官方native/common-track适配
       └──────────────┬───────────────┘                         │
                      v                                         │
             冻结几何伪计数器全dev误差与覆盖报告                  │
                      │                                         │
               目标语义可信？──否──>一次有界修订/记录退出原因     │
                      │是                                       │
                      v                                         │
            首seed全数据canonical + matched control              │
                      │运行健康                                 │
                      ├──────────>余seed/完整消融<───────────────┤
                      │                                         │
                      └──────────────┬──────────────────────────┘
                                     v
                    frozen predictions + 全人口指标/CI/效率/图
                                     v
                独立确认评估（在实际授权边界内）+ claim audit
                                     v
                 CVPR定稿/全页渲染/匿名与引用检查/复现包
```

最先可交付的不是论文或“SOTA”：应是 `R1 protocol + full cache/source manifest + target-provenance/full-sequence coverage report + scorer/time-map CPU checks + baseline runnable manifest`。这组交付允许后面的正式训练直接积累论文证据。完整数据构建、baseline 安装/适配可以与 CPU target 审计并行；除非依赖共享资源，无须等 A005。

预算沿用历史事实仅作上限参考：旧 93 A6000h/100h ceiling 只覆盖 pilot 的 3 teacher+24 response，不能承诺也覆盖完整数据、七个外部框架、消融。实际可用 GPU、余量、授权预算与完整数据规模由主任务清点后填写，不擅自扩额也不缩成小样本。

## 8. CVPR 时间与稿件范围

官方当前列出 CVPR 2027 注册 2026-11-10、论文 11-16、补充 11-23，均 AoE。[官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)。以今日 11-04 计，论文还有约十二日，不能把未经测量的训练耗时假设成能赶上；最迟在数据/基线 manifest 有实际 ETA 后作一次交付可行性判断，不每天重复 gate。

正文八页，参考文献和 LLM Usage Statement 可另页；稿件与补充必须匿名，LLM 声明必需。旧 ICASSP 的五页和完整作者元数据不适用于当前匿名稿。[官方作者指南](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)。用户要求的三线表、红色 improvement、逐页渲染和 Method 约两页继续保留；它们是用户质量规则，不冒称 CVPR 硬规则。

建议正文叙事只承诺证据支持的一条主线：全局节奏在非平稳人级轨迹上出现系统偏差，局部相对节奏路由+共享专家+连续重建能在自然计数和预定漂移条件下改善。三个组件是耦合机制，不保证分别新颖。若自然数据、主表、强控制或独立确认仍缺失，稿件仍为 pre-results/未完成；不通过拿掉强方法、隐藏负域或编造有效性赶 deadline。也不自动投稿、公开或清理服务器。

## 9. 收尾验收

本次审查交付通过：明确保留与废止项、找到两个代码因果问题、更新到 9 月证据、提供一次性 R1 建议、主表/消融/统计依赖图和官方 CVPR 约束。科研有效性仍未通过，详见首表；没有将单测、coverage 或 X0 loss 改称自然有效性。后续按最小依赖推进，遇到真实证据再修订科学设计，不恢复无期限门禁循环。
