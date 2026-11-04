# TempoRAC 独立创新性审查 — 2026-11-04

**判定：PROCEED WITH CAUTION。现有 TempoRAC 主线明确可以继续科学验证；没有发现足以作 ABANDON 判定的具名已发表论文。这个正向判定是 same-family positive provisional，不是最终 novelty PASS、效果通过或 CVPR 接收判断。**

CAUTION 有两个具体根据：PAMS 已同时具备三种速度专家、局部双窗动态阈值和局部变速实验，TempoRAC 必须证明“可训练响应分支的局部概率路由”带来的增益；而当前 NOLA 在数学上等价于对局部门权重的归一化平滑，并非一个新的独立响应重建算子。MultiCounter+、TWCRAC 全文尚未取得，属于明确的查新缺口，不能据摘要宣布精确机制已排除碰撞。上述事实不要求换路线或停止已有科学准备。

审查方式：独立子代理在同一 Codex/GPT 模型家族内重新阅读方法证据并作判断；没有调用跨家族 reviewer、没有联系作者。具体模型后缀未由本子任务独立 API 回执核定，不沿用旧文档的 GPT-5.6-Sol 字样。正式角色/型号以父任务运行回执为准。未读取自然 test、未执行模型、未启动 GPU、未修改方法、未恢复 WARP-PHASE。

## 1. 可核对验收表

| 本次要求 | 判据 | 收尾状态与证据 |
|---|---|---|
| 忠实审查现有三主张 | 以 FINAL_PROPOSAL 的核心图及 CVPR_RESUME R1 的新授权边界为准 | 通过；§2–4，不把旧安全流程当新前置条件 |
| 独立核读近邻而非照抄 verdict | 从原文检查方法和关键实验 | 通过；MultiCounter、PAMS、TReViS 的指定正文段落见 §3；不等于逐页全篇复审 |
| 明确全文/摘要覆盖范围 | 所有结论绑定阅读层级 | 通过；核心库可取得全文 3/5，另外 2/5 仍为部分覆盖；见 §3 |
| 逐主张 closest work / delta / 禁止表述 | 能写成审稿人可验证的差异 | 通过；§4–6 |
| 决定性实验可证伪 | 区分训练、推断干预、容量和有效替代 | 通过（实验方案层面）；§7；没有任何新效果结果 |
| 穷尽检索/全部精确碰撞排除 | 全部高相关正文以及别名/引用链均得到核查 | 部分；复用原 9 个查询并补 23 条查询，仍有全文和索引缺口；不宣称领域穷尽 |
| 校准 verdict | 无具名精确先例不得 ABANDON；具体薄点才用 CAUTION | 通过；上述正向 provisional 判定，限制原文保留于附录 |
| 六数据集×七顶会框架、六域 SOTA、三组件真实贡献 | 必须由同口径实验支持 | 未达标；当前材料没有此证据，本审查不把文献表当实验表 |
| 保存完整审查并保留来源绑定 | 本文可读、输入 SHA 可核对 | 通过；§9；本次只交付本审查 Markdown |

## 2. 审查对象与一句话 delta

审查的是共享 encoder、共享 slow/medium/fast 响应分支、每身份独立状态、detached 局部 NUDFT 相对节奏 cue、概率融合、严格正分母 NOLA、每身份一次固定 connected-component decode。不是 identity embedding，不是动态学习 router，不是检测跟踪端到端 MRAC。FINAL_PROPOSAL §2 明确没有 identity token；§5 和 §8.1 明确专家在整条 run 上执行一次，窗口只切片缓存 logits。

当前可以诚实检验的一句话是：

> 在同一外部身份 pose 轨迹和相同计数监督预算下，以脱离网络梯度、按原始时钟计算的局部相对节奏 cue，对共享可训练响应分支作逐窗概率融合，经归一化重叠平均后每人仅作一次固定事件解码；检验这一组合是否比最强匹配非局部对照更好地保留非平稳计数，并保持 clean 表现。

这是一项具体可证伪的方法假设，不是“从未有人想过”的宣称。可以作为核心科学贡献候选；C1 的状态隔离和 C3 的正确重建首先是其有效性条件。若最终只能证明代码隔离正确、NOLA 分母非零，却没有自然数据上的路由效益，现有论文核心贡献尚未建立。

接续协议 R1 已将目标切到现有 TempoRAC 的 CVPR 接续，并允许科学诊断；旧 proposal 中 ICASSP 页限、旧服务器、旧权限凭证和一次性 kill 流程不能重新成为本轮创新性否决条件。这里不重开设计、不自行改 loss、不扩展训练预算。

## 3. 阅读覆盖与 closest work

这里的“全文证据”指直接从完整正文中核读所列段落，不表示本子任务逐行读完参考文献、所有附录或渲染了每一页。输入文献报告的 3/5 全文覆盖被保留；本 reviewer 在其中重新核读以下关键部分。

| 工作 | 本 reviewer 实际覆盖 | 可以确认的重叠与距离 |
|---|---|---|
| MultiCounter，ECAI 2024 | 本地全文文本 §§3、4.1–4.3、5.1–5.3、5.4 开头、5.5–5.7；公式及 Table1/5 | 已有身份级多人输出、异步/变速处理、64 帧与 32 帧重叠窗口、跨 clip IoU 链接；同时已有 ByteTrack+单人 RAC 的逐轨计数对照。C1 不是新任务 |
| PAMS，CVPR Findings 2026 | 本地完整正文中 §§3.1–3.3、Algorithm1、§4.3–4.4；补充 §7.2.1、Tables4–5 | 最接近 C2。训练为 period-adaptive multi-scale representation learning；推断为同一 period stream 的 Fast/Medium/Slow 参数化峰检测，再 count vote。每个专家自身有逐时刻长短双窗阈值，因此不能称其推断仅有全局节奏适应 |
| TReViS，arXiv:2609.24367v1；输入记录说明 Image and Vision Computing 接受稿 | 本地全文 §3.1.1–3.1.3、Algorithm1、§4.7–4.9，以及相关工作与参考文献45 | TSSM 粗周期/计数统计引导合成训练数据；不是已证明的每原始物理周期精确目标。相关工作还指向更早 adaptive gate fusion；不能用“自监督挖掘周期”作宽泛首次主张 |
| MultiCounter+，TCSVT 2026 | IEEE 摘要/引言可见片段、官方 README、官方仓库根目录核查 | 时间一致实例头、anchor-guided 长短周期模块与合成增强已确认；完整最终方法、融合和解码仍未读到。README 的 `output (2).pdf` 链接所指文件当前未出现在仓库根目录清单，不能把 badge 当全文 |
| TWCRAC，Pattern Recognition 115199 | 出版社摘要、highlights、引言、section snippets；SSRN 搜索可见元数据 | 局部统计动态阈值、adaptive dual-window、tempo drift、video-level supervision 已明确。不能推定没有软融合/NOLA，也不能推定它已有 TempoRAC 完整方法。SSRN PDF 本轮打开仍 403 |

一手来源：[MultiCounter 全文](https://arxiv.org/html/2409.04035v1)、[PAMS 正文](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)、[PAMS 补充](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf)、[TReViS 全文](https://arxiv.org/html/2609.24367v1)、[MultiCounter+ IEEE](https://ieeexplore.ieee.org/abstract/document/11421441/)、[MultiCounter+ 作者仓库](https://github.com/yinntag/MultiCounterPlus)、[TWCRAC 出版社](https://doi.org/10.1016/j.patcog.2026.115199)。

新增检索发现而未升级为全文审查的邻居：

- [D²-STX 官方 CVF 页](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.html) 已核定 **CVPR Findings 2026，8205–8214**。官方摘要是 pose/video 双分支时空跨注意力；只核摘要和 venue，不据第三方长摘要认定其内部 gating 细节。
- [RepProbe，arXiv:2610.18844](https://arxiv.org/abs/2610.18844)，2026-10-13，及[作者项目页](https://arenaa.github.io/RepProbe-webpage/)：近六个月新发现，VLM 重复计数诊断 benchmark；只读摘要/项目页，不属于已确认的 TempoRAC 方法碰撞。其 RepCount 衍生样本不能直接算新增独立源域。
- TReViS 参考文献45：Zhang 等，*Repetitive Action Counting with Feature Interaction Enhancement and Adaptive Gate Fusion*，MMAsia 2024，[DOI](https://doi.org/10.1145/3696409.3700271)。本轮 ACM 页面 403，只有一手论文中的引用及检索元数据；**未读其摘要或全文**。精确 gate 对象、是否局部 tempo 应列为待补，不能因标题相似就判撞车，也不能称首次 adaptive gate fusion。
- [CEPMS-Net 出版社记录](https://www.sciencedirect.com/science/article/abs/pii/S095219762601691X)，EAAI 181(2)，2026-10，文章115407：本轮只读摘要/可见引言及方法概述。它覆盖 pose、多尺度卷积和局部/全局融合的邻域；不是已证实的 identity-local tempo MoE。应纳入强 pose 对照候选可用性核查，不能凭作者摘要结果给 TempoRAC 排名。

自源污染：搜索仍命中用户作者主页的 TempoRAC 简介。该页面及镜像不作独立先例、性能或发表事实的证明。其“identity-conditioned encoder”“extensive experiments”等与当前可审计状态有差异；本子任务不修改网站、不核定是否已实际投稿。

## 4. 逐主张判定

| 主张 | 最近邻 | 剩余可检验 delta | 不能声称 |
|---|---|---|---|
| C1：多人 identity-local 计数 | MultiCounter / MultiCounter+；MultiCounter 的 tracker+单人 counter 对照 | 在固定轨迹输入下局部路由收益；干预 A 的全部数据时 B 的响应和计数不变；不同人数、排序和异步组合下结果等变 | 首次 MRAC、首次 person-wise、仅靠 private state 成为新的多人建模方法；GT bbox supplied-track 结果代表端到端检测/跟踪性能 |
| C2：局部非平稳节奏软专家 | PAMS；TWCRAC 精确机制待查；更早 gate fusion 待查 | 训练得到的不同感受野响应分支，用身份内相对局部 cue 在概率层混合；与实际局部节奏匹配而非均匀容量/正则化效应；跨自然局部变速验证 | 首次三速度专家、首次局部窗口/局部阈值、首次非平稳计数、首次局部变速评估；“PAMS 只适应全局周期” |
| C3：NOLA 后一次 decode | MultiCounter 已重叠 clip；PAMS 同一连续 stream 的专家 count vote；一般归一化重叠平均 | 概率融合发生在离散事件决策之前，能否实测降低窗口 seam 的 split/merge；具体 taper 与 one-decode 对因果收益的作用 | 发明 NOLA；分母正就保证无重复计数；调用一次 decoder 就保证每个物理周期只计一次；归一化自动保证任意 offset 不变 |

PAMS 和 TempoRAC 的可核实差别是“多个参数化 detector 产生 counts 后投票”与“多个可训练 response branches 在事件决策前融合”的不同计算图。这个差别有研究价值；若实验显示它只相当于一个更大模型/更平滑门权重，而无局部选择的因果贡献，则 delta 会薄。不能单靠更换聚合位置和三条分支名称证明创新强度。

## 5. NOLA 的精确代数边界（独立推导）

FINAL_PROPOSAL §5/§8.1 缓存 run 级 `z[e,k]`，不依赖 window `w`；§8.2 定义 `r[w,e]=sum_k a[w,k]*sigmoid(z[e,k])`。对有效边 `e`：

```text
D[e] = sum_w h[w,e] v[e] > 0
a_bar[e,k] = sum_w h[w,e] v[e] a[w,k] / D[e]
R[e] = sum_k a_bar[e,k] sigmoid(z[e,k])
sum_k a_bar[e,k] = 1,   a_bar[e,k] >= 0
```

因此当前 NOLA **等价于对 window-local gates 作带 taper 的归一化插值/平滑，再融合已经连续的 run 级专家响应**。这是实数算术的代数等价；浮点运算次序可能导致小差别，应结合阈值 margin 做数值核验，不宣称 bit-identical。

这不是否决理由，也不是恢复旧路线的理由。它让机制描述更准确，并给出了很便宜的强诊断：

1. global/uniform 的 gate 不随窗变化时，重建在实数算术中退化为相同混合响应；不能期待这个 arm 用“去 NOLA”创造独立增益。
2. 分母正仅保证可定义且保持 convex range，不保证激活集合有正确拓扑。单个真实动作的响应若有两个超过0.5的峰团，one-decode 仍计2；两个动作中间谷值超过0.5则可能合为1。
3. 改变窗口起点会同时改变 cue、gate 和 taper 权重，因此没有无条件 offset 不变性。有限人工 operator fixtures 的零错不能替代真实模型重算 gates 后的 offset 测试。
4. 将全部相同贡献等比例复制会被归一化抵消；只重复一个不同 gate 的窗口通常会改变 `a_bar`，不能声称任意窗口重复都不影响结果。
5. 一个可证明的有限稳定性条件是：若同一有效支持与 adjacency 下，两种重建的逐边差异严格小于参考响应到0.5的最小 margin，则 threshold bits 和 component count 不变。该条件与“是否一物理周期一事件”仍是两件事。

合理比较应包括 normalized rectangular overlap-average、合法的直接逐边 gate 插值/平滑、以及保持事件归属正确的窗口解码基线；去掉分母让幅值随重叠数增加，只证明归一化必要，不足以证明科研组件贡献。

## 6. 可以写、不能写与暂缓写

现在可以写方法的真实结构和研究问题；不能写效果结论。可采用“针对 supplied identity tracks 的局部节奏响应融合”，而不要“解决端到端多人跟踪计数”。

以下边界需要在稿件与实验表中保持：

- 共享参数与 private identity state 不等于 identity-conditioned embedding；因果 encoder 不等于在线系统。当前 reference weighted median 依赖整条 run，窗口也包含局部前后文，不能因此声称 streaming/zero-lookahead。
- “计数目标不用人工 count/period labels”不等于全系统无监督。外部 pose/tracker、GT bbox、模型预训练、目标生成过程必须分别披露。
- PAMS 的本地重实现不是 author-native；其训练描述歧义也不是删除它的理由。TWCRAC 文本写 video-level labels 时，不能未经核查并列称完全同监督预算。
- one-decode 的事件位置/激活片段不是天然合法的 full-cycle period intervals；Period-mAP 必须使用经核定的坐标与官方 scorer，不能用峰间距自动造合法区间。
- 原始负结果、unsupported identities、伪目标失败及已观察 development 需要保留；不能仅报可认证部分后推广至全部人/全部周期。
- 六数据集、七个有说服力顶会框架、六域 SOTA 目前均没有实验证据。Findings/期刊身份按实记载，不为凑框架数改叫主会；pose 版本、变速版本、RepCount 衍生组合不是自然新增独立域。
- 用户“loss 少于3”尚未满足：旧 F11 明列 reconstruction、correspondence、两个 phase penalties，F14 也含 BCE 与正/负 margin。应在实际采用版本逐项数清，不能靠一个 `L_teacher` 或 `P` 名称把加项折叠成一项；也不能无证据认定现在采用的 v15 必然走旧 teacher 分支。本 reviewer 不修改它。

## 7. 哪些实验能决定这个 delta 是否站得住

下面是对现有路线的证据要求，不是本子任务启动新增训练/打开 test 的指令。需要新的比较训练时，由主流程在已授权预算和当前设计审查中固定；不能悄悄扩展旧冻结 job 表。

| 决定性问题 | 最小有说服力证据 | 什么结果削弱主张 |
|---|---|---|
| D0：自然输入是否支持物理计数？ | 显式 source/output 时钟和支持掩码；完整人群的观测/插值/外推时长及事件覆盖；几何伪计数器逐事件 split/merge/miss/extra 与 teacher 依赖来源；标签仅在诊断/评估读，不进入优化 | 仅在极少数高置信或几何易例成立；总 count 抵消掩盖事件错位；行号冒充物理时钟。此时先诊断输入，不把失败误归于 novelty |
| D1：局部 cue 是否有因果价值？ | 同权重 local/global/uniform/blocked/shuffled 推断干预；相同参数、数据、steps、tuning 预算的独立训练 global/uniform/capacity 对照；自然 paired clean/drift 至少三 seeds | 只赢把 local-trained 网络推断时关路由的受损对照，却不赢训练好的 matched baseline；shuffle 不破坏收益；clean 损失超过预设容忍 |
| D2：三分支是否真的专门化？ | heldout X0 3×3 matrix 作为必要诊断；自然数据按独立真实局部周期分层的分支误差/责任覆盖，标签不作训练或 router；报告所有 strata 与 CI；single-branch/参数匹配替代 | 人工 responsibilities 下必然出现的对角优势在自然轨迹消失；一个分支普遍最佳；少数 gate 状态占尽时长。非显著差异不能证明三分支等效有效 |
| D3：C2 是否优于现有局部自适应？ | 同 supplied tracks 的 PAMS 原生可用实现或明确标注的忠实重实现；保留其局部双窗参数与正常验证选参；同时给 native MRAC 的 MultiCounter+ 单独表；TWCRAC 完整机制取到后补最相关对照 | 仅将三专家 PAMS 简化成全局 FFT/固定 threshold，或靠不同输入、注释/预训练预算制造优势；不能把 prior abstract 数值充作复现 |
| D4：概率先融合与 NOLA 到底贡献什么？ | 冻结响应的代数等价 check；taper 对 normalized rectangular；连续融合对合法 count-level consensus；模型实际重算各 offset/长度/overlap；逐事件 boundary split/merge/miss/extra 与正负 margins | 只有漏除分母的错误实现显著变差；仅 operator bank 零错，真实模型 seam 仍变；gain 全由阈值随幅值变化产生 |
| D5：identity-local 是否正确且作用边界清楚？ | 单轨运行与同轨放入多人 batch 一致；A→A′ 不改变 B；人数/顺序/异步、长短轨及 gap reset；真实 detector/tracker 轨迹另测并披露 ID switches | isolation 单测成功就声称解决遮挡/跨人身份；仅 GT-box 人工组合上有效而推广真实场景 |
| D6：完整证据是否可推广？ | source-video/component-disjoint 划分，已看 validation 标为 development；三 seeds、sample SD、按源依赖聚类的 paired bootstrap CI，强对照选择与多比较进入统计；全 dev 的 observation 图；window/stride/router bandwidth/reliability/threshold 性能曲线和含前处理效率 | 以扰动次数当独立样本，挑最强 seed/子集，或用未显著当 clean 等效；只报速度扰动不看自然非平稳；删最强 comparator 才达 SOTA |

D1 与 D2 是主线成败最直接的证据；D0 是解释它们的前提。D4/D5 是完整方案的必要有效性与贡献证据。这里没有声称所有实验都是会议硬性门槛；三组件、六域和 loss 项数属于用户质量要求，必须另行逐项验收。

旧 §10.1 的 global/uniform/blocked/shuffle 共用 canonical logits 是很好的推断因果干预，**不等价于分别训练到同等预算的竞争方法**。旧独立 capacity control有价值，但若想声称胜过“最强非局部”，至少应明确训练/选参口径以及真正优化过的非局部 comparator。不可在论文表里把 inference ablation 写成完整 retrained baseline。

当前 proposal 主要把 identity isolation 和 NOLA 当 validity prerequisites。用户又要求三组件都真实贡献。两者可同时诚实满足：证明 local cue、trainable experts、连续融合/一次解码的实际因果作用，但不把三项都冒称各自首次发明。如果第三项只提供正确性而没有性能/稳定性贡献，最终必须标为部分满足，不能改名后自动算通过。

## 8. 检索回执与剩余范围

复用输入 evidence JSON 的 Q01–Q09 与近六个月 arXiv 查询，并在本子任务 2026-11-04 实际补充以下23条 web 查询。六批返回包括大量无关命中、官方原文链接、上述四个新增邻居及失败记录。第三方仅用于发现，机制结论只绑定本地一手全文或官方可见内容。

```text
01 "repetitive action counting" "soft" "experts"
02 "repetition counting" "overlap-add"
03 "MultiCounter+" "pdf"
04 "TWCRAC" full text pdf
05 "Toward Efficient and Robust Multi-Instance Repetitive Action Counting"
06 "repetitive action counting" "mixture of experts" -site:appleweiping.github.io
07 "repetition counting" "normalized" "overlap" -site:appleweiping.github.io
08 "repetitive action counting" "local tempo" -site:appleweiping.github.io
09 "D^2-STX" site:openaccess.thecvf.com
10 "repetitive action counting" "routing" -site:appleweiping.github.io -TempoRAC
11 "action counting" "frequency" "experts" -site:appleweiping.github.io -TempoRAC
12 "repetition counting" "overlap" "fusion" -site:appleweiping.github.io -TempoRAC
13 "Decoupling Spatial-Temporal Cross-Attention for Dual-Branch Repetitive Action Counting"
14 "TWCRAC" "adaptive dual-window"
15 "MultiCounter+" "output (2).pdf"
16 "Repetitive action counting with feature interaction enhancement and adaptive gate fusion"
17 "RepProbe" "Counting"
18 "repetitive action counting" "nonstationary"
19 "repetition counting" "non-uniform" "window"
20 "repetitive action counting" "local" "gating"
21 "person-wise" "repetition" 2026 -TempoRAC
22 "repetition counting" "expert" after:2026-05-04 before:2026-11-05 -TempoRAC
23 "temporal repetition counting" 2026 multi-scale fusion experts
```

还直接打开了 RepProbe arXiv、TWCRAC SSRN PDF、D²-STX CVF 页面、MMAsia ACM DOI，并用 GitHub connector 查询官方 MultiCounterPlus 根目录。SSRN/ACM/D²-STX 的 web 直接打开有403；CVF检索摘要仍返回官方 venue 内容。不把可见摘要升级为成功全文下载。

优先未决：MultiCounter+ 最终正文；TWCRAC 完整方法/补充；MMAsia adaptive gate fusion 正文；各 native scorer 和实现；其它频率/多尺度/边界建模邻居的引用链。RepProbe/CEPMS-Net 的发现说明“精确短语 arXiv 近六个月只返回一篇”不等于整个新文献域仅一篇。本轮没有遍历全部会议录、专利或完整学术索引，不给“穷尽通过”标签。全文补齐应继续，但不是停止现有 CPU 诊断/数据准备的借口。

**重新判定规则：**若新取得的一篇已发表论文具体包含相同目标、同一核心机制以及本文所拟检验的结果，必须具名定位并重新判断；仅共享词汇、三分支、局部窗口或相似实验，不足以 ABANDON。若 D1/D2 建立自然收益并且 D4 解释融合价值，当前 crowded-but-deltaed 可升级为更强 PROCEED；若效果不成立，应如实报告实证失败，不倒推伪造“早就没有新颖性”。并发/竞争工作报告给用户决定竞速，不自动 veto。

## 9. 输入绑定与交付

读取对象均来自当前 E 盘接续目录；本文写入 `E:\ICASSP_TempoRAC\aris-work\novelty_review.md`。本次未修改输入。

| 输入（相对 `E:\ICASSP_TempoRAC\TempoRAC-CVPR`） | SHA-256 |
|---|---|
| docs/LITERATURE_REVIEW_20261104.md | b6e325edef52f9c767ac3f21f4b255bc43f0c70b2bff90f75fa0933aa83cee4d |
| docs/literature/evidence_20261104.json | f1e21074d1e2bdb6e75933b6e7f9f89aae8a926228c01d17d2d8e2602f4bb4ff |
| refine-logs/FINAL_PROPOSAL.md | 3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391 |
| docs/CVPR_RESUME_20261104.md | 6e7c00b249c79e32770c2b1661b8ff060a5b79108308ddd7c42eb6d002e0f5e0 |

原始文献文本：`E:\ICASSP_TempoRAC\aris-work\literature\multicounter.txt`、`pams.txt`、`pams_supplement.txt`、`trevis.txt`、`multicounter_plus_readme.md`。本轮没有输出论文 Word/PDF，因此没有假称做了其逐页排版验收。

收尾时父任务提供的状态更新：已找到完整 MultiRep 候选目录，train/val/test 数量为794/183/180，full train/val 媒体 hash 审计开始。本 reviewer 未独立验收该目录，不据此宣布数据完整或模型结果达标；它不改变本审查 verdict。

## 附录：本次原样适用的 ARIS 判定边界

```text
=== NOVELTY VERDICT LIMITS (these bound how you judge, never how widely you search) ===
Search exhaustively; judge calibrated. Two failures waste months equally:
passing an idea a published paper already contains, and killing a viable idea
because the territory has neighbors.
1. Proximity is information, not a verdict. Someone working nearby goes in the
   report; it is not by itself a reason to reject.
2. ABANDON has exactly one qualification: a specific published paper already
   contains this result — name that paper. No named paper, no ABANDON.
3. Crowded-but-deltaed is PROCEED: state the delta in one sentence a reviewer
   could verify. Thin or contested delta is PROCEED WITH CAUTION — say what
   would make it carry, not why it should die. CAUTION is not a safe middle:
   if you cannot name the specific thing that makes the delta thin, the
   verdict is PROCEED.
4. Concurrent or competing work is not a veto. That is a race — report it and
   let the user decide whether to run it.
5. A direct attack on a central problem is legitimate novelty when nobody has
   executed it well. "This area is hot" does not mean "this area is taken."
6. This check is an early gate, never the last one — more triage, pilots, or
   external review still stand between any idea and a paper, whatever order
   this run uses. A wrongly passed idea dies cheaply at one of them; a wrongly
   killed idea is never seen again. When torn between two verdicts, choose the
   more permissive one.
Say plainly when an idea clears the check. Do not manufacture overlap.
```
