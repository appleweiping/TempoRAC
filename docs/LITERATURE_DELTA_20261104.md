# TempoRAC 新增文献证据与对照清单（进行中）

本轮查阅时间：2026-11-04。以下是官方摘要/官方仓库的初步证据，不是正文精读完成或创新性最终通过。历史 `idea-stage` 与 `refine-logs` 保留。主张应聚焦局部相对节奏路由与身份内连续重建的可验证增量，而非首次多人计数、首次自监督或首次多尺度。

| 工作 | 已读一手证据 | 与本方法的关系/下一核查 |
|---|---|---|
| MultiCounter | https://arxiv.org/abs/2409.04035 | 同时检测、跟踪、多人计数和 MultiRep 已存在；需核对正式 split/scorer、源视频去重与 native checkpoint |
| MultiCounter+ | https://github.com/yinntag/MultiCounterPlus ; https://doi.org/10.1109/TCSVT.2026.3670243 | 官方仓库提供多人任务、时间一致实例头和长短周期感知头；必须作为强近邻与 native 对照，不能宣称首次处理多人异步节奏 |
| PAMS | https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html | 自监督骨架、多尺度周期一致性与推断专家投票均已存在；核对正文与本地复现歧义，不能直接将报告数值认定同口径实测 |
| TWCRAC | https://www.sciencedirect.com/science/article/pii/S003132032602162X | 出版社记录涉及 TCC 和 adaptive dual-window，应精读区分 tempo drift 的增量；当前在线日期/期刊卷期不同，查新按首次可公开版本核对 |
| TReViS | https://arxiv.org/abs/2609.24367 | 2026-09-21 首次提交；利用无标签视频的重复结构合成序列和伪标签，直接关联训练目标恢复；需全文对比伪监督及时间变化机制 |
| PoseRAC | https://github.com/MiracleDance/PoseRAC ; https://arxiv.org/abs/2303.08450 | pose saliency 监督和自监督设置不同，需明确输入/监督；未核实顶会发表身份，不以年份或名字推断 venue |
| RepNet | https://research.google/pubs/counting-out-time-class-agnostic-video-periodicity-in-the-wild/ | 自监督合成训练与 self-similarity bottleneck；需核对官方评测及完整序列采样，采用原作者关于评测差异的说明 |
| TransRAC | https://github.com/SvipRepetitionCounting/TransRAC ; https://arxiv.org/abs/2204.01018 | 多尺度 temporal correlation、density 和长视频对照；官方仓库已有 RepNet 实验口径澄清，复现时必须读取 |
| DeTRC | https://arxiv.org/abs/2403.01543 ; https://github.com/lizishi/DeTRC | dynamic queries 的高效计数对照；需正文、checkpoint、依赖与合法数据接入核对 |
| D2-STX | https://github.com/wwwwwxa/D2-STX | 视频/pose/融合与 exemplar 设置不同；仓库 pose 特征提取仍注明待发布，不能把未跑通流程算完成框架 |

上述官方论文报告的数字未抄入 TempoRAC 结果表。当前不声称这些框架均跑通、不声称满足七个顶会框架，也不声称六数据集 SOTA。实际完整实验矩阵需绑定合法权重、输入模态、监督、训练数据、完整 split、seed、scorer 与资源。

## 待完成的查新动作

- 对主张分别检索至少三种表述：多人身份内计数；局部非平稳节奏与软专家路由；重叠窗连续重建与一次解码。
- 完整阅读主要近邻正文/补充，记录各主张的支持段落与公式；下载正文仅放 E 盘研究目录。
- 区分 pipeline 共享组件与真正方法增量；Astra 独立判断最接近工作和可检验差异。不得把当前元数据/摘要级清单标成 full novelty PASS。
- 确认六真实数据集的任务适配、来源、完整性和独立评估边界。当前历史项目协议为 MultiRep 主任务，不能用数据名列表替代已完成实验。
