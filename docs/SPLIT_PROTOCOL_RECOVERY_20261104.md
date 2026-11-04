# TempoRAC 双轨数据恢复说明（2026-11-04）

**决定：保留完整 official benchmark，另设预先冻结的 source-disjoint 确认子集；复用同一组模型与预测，不改 TempoRAC 方法，不以删测试样本提高 official 分数。** 本页是可实施协议建议，不访问测试标签、不启动 GPU。已读服务器 inventory 和本地 JSON；`protocol_audit.md` 尚未出现。794/183/180 是候选目录数量，正式释放身份及 canonical source 仍待核验。

**已知证据。** 父任务报告按目录名 `+` 分解的 source-name proxy：train/val/test 为174/85/121，train∩val=0、train∩test=36、val∩test=42。本次独立枚举同一 inventory 全部180个测试目录：有78个目录的所有 constituent proxy 均不在 train∪val，102个至少含一个重叠 proxy。这只是无标签的结构统计，不是 canonical source 证明，更不是性能筛选。旧110/51视频 pilot 的 val 已用于设计，永久属于 development。

## 两条轨道

| 轨道 | 数据与使用方式 | 可支持的结论 |
|---|---|---|
| A：Official 全量 | 确认官方释放后，用完整794 train训练、183 val开发；测试全部180视频，禁止漏掉重叠/困难/拒绝样本。保留标准指标和原官方划分。 | 标准 benchmark 性能；明确存在源片段复用，不能直接称对未见源视频的泛化。 |
| B：Source-disjoint 确认 | 用**同一冻结 checkpoint**，在测试集中所有来源均未进入训练或开发暴露集合的整视频上评分；先无标签冻结 membership。78仅是当前代理候选数。对整段混合视频有任一已见来源即不属于B，但仍完整进入A。 | 对此次确认集合的未见来源泛化；不是另一个数据集，不保证代表全部自然场景。 |

## 最小执行顺序

1. **完成来源图。** 977个 train/val 媒体SHA继续跑；核验官方manifest和合成配方。用 canonical原视频ID、裁剪区间、生成元数据核对 `+` 名称；字节SHA只能证实完全重复，不能排除转码/裁剪同源。无需打开test标签：来源表仅含目录/媒体与合成元数据。canonical无法证明者标记unknown；A保留、B不当作已证实独立，并单列数量。
2. **冻结暴露集合 E。** 包含此次完整train/val，以及此前pilot、已知v4x审计/开发、任何用于本项目参数选择或设计的源视频；按已存在provenance核对，不去读取旧test标签补历史。训练权重/外部预训练来源另行披露；无法排除的预训练重复限定泛化措辞。B定义为 `test视频全部canonical sources与E不相交`。不能把“未看test标签”误称“从未看过源视频”。
3. **一次写定 split manifest。** 保存A全部ID、B全部ID、overlap/unknown分层、每项来源理由、生成代码/输入hash、预定指标与cluster bootstrap规则。确定身份/视频聚合和失败样本计分规则；冻结时不读取任何test count、period或指标。若canonical核验改变78/102，保留前后清单及原因；这是事实修复，不按结果选样。
4. **所有方法共用协议。** TempoRAC、matched controls、common-track baselines 使用相同train/val、轨迹、A/B列表及评测器；checkpoint和阈值仅按事先约定的开发规则选定。native官方方法单列监督/输入差异。无法证明来源安全的官方权重可以进入A上下文行，不冒充B同口径训练结果；B需来源可核验的复现权重。A/B复用冻结逐视频预测，不分别为B调模型。
5. **指标逐行核对，结果同时发布。** 父任务新增scorer疑点包括只执行首config、未匹配检测未进入PeriodAP、`periods/periodicity_binary`与`periodicity`字段不一致、inclusive与continuous端点差异，待Sol提供代码证据；不可直接采用本方video-first AvgMAE冒充官方。后续固定“原实现复现”和“按公开定义校正的规范scorer”两个版本，各方法统一评分、并排保留差异，本次不重写整套评测。报A、B及overlap/unknown各层指标、样本/人/来源数与成对CI；bootstrap按共享canonical source的connected component聚类。反例、拒绝预测、raw artifacts和不利结果全部保留，不只报B或优势层。B若小/类别窄/独立clusters少，如实报告宽CI，不放宽独立定义补人数。

**有界后备。** 若canonical与历史暴露核对后B为空或无法支持有意义的区间估计，保留A并标记“独立泛化尚未建立”；下一确认集从真正未暴露、合法来源建立，所有方法同一冻结协议评测。不能把已见源视频重新分组、删掉难例或重新命名旧val获得“独立测试”。本轮无需等待旧证书续审；CPU数据完整性、完整pose构建及baseline适配立即按现有授权继续，测试标签仍到预测与协议冻结后才由隔离评测器访问。

**验收：** 双轨不丢A样本、B membership完全无标签、历史暴露/unknown可追踪、方法协议一致、同源聚类、全量负结果保留，均须有manifest/原始预测证据；目前仅代理目录审计通过，其余待实现核验。
