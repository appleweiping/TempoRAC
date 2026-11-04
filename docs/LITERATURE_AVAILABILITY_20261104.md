# MultiCounter+ / TWCRAC 全文补齐回执（2026-11-04）

**结论：两篇仍不能标成全文精读完成。** 本轮取得了 MultiCounter+ 固定提交的官方实现关键文件、完整树清单和两篇 Crossref/出版社元数据；所有可解析的正式正文入口仍被拒绝或只返回元数据。没有取得任何一页两篇目标 PDF，不能提供虚构的论文页码、公式号、完整 loss 或 appendix 结论。原核心库 3/5 全文、2/5 部分覆盖的状态保持。本交付补强可复核的可用性与实现证据，不赋予创新性 PASS，不修改 TempoRAC 主线。

请求角色为 routine Sol executor；本会话实际 backend 身份没有独立 attestation，OpenAI same-family 科学解读至多 provisional。未另作独立 reviewer 裁决。

## 验收表

| 要求 | 状态 | 证据 |
|---|---|---|
| 只补 MultiCounter+、TWCRAC，复用原身份/URL | 通过 | 两篇 DOI 与输入 evidence 相同；没有重读 PAMS/MultiCounter/TReViS 原文 |
| exact pinned research-lit 与必读支持文件 | 通过 | ARIS f76ec8d6b77792e2242cca049e08d0cbb7d54e19；源/安装 skill SHA 相同；READ_RECEIPT.json |
| 优先 E 盘已有 PDF | 部分 | 完整扫描 E:/ICASSP_TempoRAC 未找到目标 PDF；全 E:/ 扩展扫描耗时而中止，不声称整个 E 盘穷尽 |
| MultiCounter+ 全部正文、实验及相关附录 | 未达标 | 0 页；IEEE 418、空202响应或验证页；官方 pinned 树无 PDF |
| TWCRAC 全部正文、实验及相关附录 | 未达标 | 0 页；SSRN/ScienceDirect 403；Elsevier TDM XML只有 coredata |
| 官方代码及版本、机制和适配影响 | 部分 | Plus commit d15c767d82dac2f7a5df9fde1d7f6b5a4f491062 的12个文件；TWCRAC未定位可用作者代码 |
| 数据集/监督/权重与协议边界 | 部分 | Plus README与实际配置可核；TWCRAC保留摘要级信息与未知项 |
| 不改仓库/状态、不访问服务器/GPU/test标签 | 通过（本会话动作记录） | 所有新文件只在本目录；未运行模型、安装、Git修改、提交或推送 |

## 来源与版本

1. Tang, Yin; Zhang, Deyu; Luo, Wei; Zhang, Jinrui; Huang, Wei; Jing, Ruihai; Zhang, Yaoxue. **MultiCounter+: Toward Efficient and Robust Multi-Instance Repetitive Action Counting**. IEEE TCSVT 36(7), 9462–9476, July 2026, DOI [10.1109/TCSVT.2026.3670243](https://doi.org/10.1109/TCSVT.2026.3670243). 本轮 Crossref `created` 是2026-03-04，不能把此字段自行解释为已确认首次 online 日期。出版社题名是 Toward，官方 README 使用 Towards，按 DOI 去重，同一工作。
2. Li, Jun; Gao, Shizhao; Hu, Tianwen; Xie, Yinhui; Liu, Zhonghua; Li, Qiming. **TWCRAC: Leveraging temporal cycle consistency and dynamic thresholding for robust repetitive action counting**. Pattern Recognition 183, article 115199，显示 March 2027 issue，DOI [10.1016/j.patcog.2026.115199](https://doi.org/10.1016/j.patcog.2026.115199)。Crossref `created=2026-10-23`，本轮能解析正式记录；不能因issue年份2027就排除其截至2026-11-04已公开的邻近工作。此字段也不能单独证明首次公开日。原 SSRN 版本 [10.2139/ssrn.6630216](https://doi.org/10.2139/ssrn.6630216) 与正式版不混作逐字一致；未取得两者正文，无法比较修订。

机器可核的原始响应：`plus_crossref.json`、`twcrac_crossref.json`、`twcrac_tdm_xml.html`（实际为XML，扩展名来自通用抓取器）、`plus_tree.json`。Crossref链接标明 Plus 的 version-of-record PDF与TWCRAC的出版社TDM路径，均已尝试。Elsevier XML HTTP200 **不等于全文成功**：仅1,853 bytes，只有 coredata，明确 `openaccess=false`，没有正文、公式或参考文献内容。

## MultiCounter+：可确认的机制及限制

**论文级贡献描述仍来自官方 README/先前可见摘要，不能冒充最终正文。** 作者将方案描述为 RGB端到端检测、跟踪、人物内重复计数；混合时空交互；带 Siamese query动态更新的时间一致实例头；anchor-guided similarity reconstruction 的长短周期感知头；合成预训练及运动速度等增强。[固定README](https://github.com/yinntag/MultiCounterPlus/blob/d15c767d82dac2f7a5df9fde1d7f6b5a4f491062/README.md)

本轮独立核对实际代码，得到以下更窄、可复现定位的事实。它们是**release代码的行为，不是已核准的论文公式**：

| 机制 | 精确代码证据 | 对解释的约束 |
|---|---|---|
| 共享RGB特征和query检测 | `configs/instcount/instcount_r50.py:5–76`：R50-FPN、20 proposals、4个stage、STQIHead、PeriodHead | native RGB任务不同于给定外部pose轨迹；不是可直接喂入TempoRAC特征的原生counter |
| 空间后时间query attention | `mmdet/models/roi_heads/bbox_heads/stqi_head.py:39–66`，随后query作dynamic-conv filter与RoI交互 | 代码确有时空query交互；不能据此自动核准最终文中的Siamese更新细节 |
| 单个64帧内自相似表示 | `mask_heads/period_head.py:29–63,103–128`；`mask_heads/tsm_transformer.py:43–62,112–127,145–171` | ATSM计算帧间负平方L2距离、除sqrt(256)、逐行softmax，2D卷积后Transformer；未证明这就是最终anchor-guided模块 |
| 读出与监督候选 | `period_head.py:62–63,130–177`：64类period、1维periodicity；period CE、periodicity Focal、TSM MSE | 配置的MSE权重0.1，类默认5；最终真实使用取决于调用链，不可把defaults冒充实验设置 |
| 演示period解码 | `tools/code_for_demo/period_convertor.py:26–54`：argmax period类别+1，长度>1取倒数，逐帧累计质量，再按整数累计质量找起止 | 输出两端点、无event score；文件级阈值0.3未参与这个函数；不能自动当官方完整PeriodAP/AvgMAE转换链 |

数学复述（本轮由代码推导，**无论文公式号**）：在ATSM内 `S[i,j]=softmax_j(-||x_i-x_j||²/sqrt(256))`；演示器每帧 `p_t=argmax softmax(logits_t)+1`，`r_t=1/p_t` 当p_t>1，否则0，然后累计 `C_t=sum_{s<=t}r_s`。这不是局部NUDFT节奏cue、三个可训练tempo响应分支或NOLA的证据，也**不是它们在最终论文中不存在的证明**。

额外文件 `memory_query.py:6–43` 展示点积相似度Hungarian匹配和指数式记忆更新；`matrix_res.py:5–40`展示去对角、自相似行归一化和掩码重建。两者包含独立示例代码；本轮所读的活动调用链未确认接入。不能将文件名/孤立函数拼成作者最终方案。尤其不能从这些文件重新构造一个自称“论文忠实”的MultiCounter+。

**release适配阻塞是实质性的。** `instcount_roi_head.py:185–190`只传 `attn_feats`给period head，但所选 `PeriodHead.forward` (`period_head.py:103`) 需要额外 `num_object_batches`；ROI训练调用的get_targets/loss仍按单一period输出接口 (`instcount_roi_head.py:212–227`)，而PeriodHead定义需要periodicity/TSM输入并返回tuple。配置没有另一个periodicity_head，但ROI `simple_test` 的period-only分支仍把输出当tensor（`instcount_roi_head.py:487`起），与新PeriodHead的tuple接口不一致。以上是静态读码定位，未运行依赖或模型；足以阻止把当前仓库存在称为native baseline已能运行，不足以断言作者论文实验无效。

### 数据、训练、指标和可用性

- 作者README列MultiRep 1,157 videos、52,590 periodic events；模型预训练用RepCount-A生成的合成数据，再MultiRep fine-tune。配置 `configs/_base_/datasets/multirep.py:2–63` 指向train/val/test JSON、64帧clip和训练 `gt_bboxes,gt_labels,gt_periodicity,gt_periods,gt_ids`。本轮未读任何标签或复核真实样本数；不能从文件名推定split来源独立。
- 配置有bbox L1、GIoU、classification Focal，另period CE、periodicity Focal、TSM MSE；这证明公开实现候选使用密集监督字段，不能与TempoRAC计数目标无人工count/period监督混称同预算。最终论文loss组合/TSM target来源仍未知。
- 数据公开入口：[MultiRep作者Drive文件夹](https://drive.google.com/drive/folders/1kz9pk2WJlneFai7cjxvGx9gpdWk6jLqa?usp=drive_link)；权重入口：[作者R50-FPN checkpoint](https://drive.google.com/file/d/17eHYy4JRxF2bAMjMnLqyXNLmN7zrtw4H/view?usp=drive_link)。仅复用README链接；没有下载模型/数据，没有checkpoint SHA、加载或论文结果复现证据。
- README报告Period-mAP14.56、AvgMAE0.223、AvgOBO0.571是作者结果，未移入TempoRAC实验表。论文的完整其他RAC数据集协议、训练seeds、预训练预算、选参规则和所有表格仍未核。
- 评测沿用输入 `METRIC_CONTRACT_20261104.md` 的官方scorer静态结论；本轮没有重审已读MultiCounter论文或重跑scorer。AvgMAE/OBO公式及全预测转换链仍pending；匹配条件/完整人口两通道不能混名。演示器二维区间不能直接满足scorer要求的三列event输出，source-clock/stride也必须明确映射。

## TWCRAC：仍为部分覆盖

可核的出版社摘要/Highlights信息限定为：单视频内TCC用于周期表示学习；以video-level supervision替代昂贵dense frame labels；周期分数的局部统计动态阈值；adaptive dual-window面向tempo drift与signal decay；作者列RepCount、Countix、UCFRep和43+FPS。它们不能确定最终输入pose提取器、损失、窗口公式、计数操作、监督标签究竟是类别还是计数、数据split及预训练泄漏控制。[出版社可索引摘要](https://www.sciencedirect.com/science/article/pii/S003132032602162X)

本轮搜索未定位可实际读到的作者官方代码、权重或supplement。**未定位不等于不存在。** 无法确认独立方法页码、公式编号、Algorithm、所有实验章节、附录、超参、训练seed、模型选择或checkpoint身份。SSRN原记录所称31页仅是前轮记录的元数据，本轮没有PDF核实；正式版页数也未知。

可作的对比推断：其局部阈值和双窗已经覆盖“局部处理非平稳节奏”的广泛动机；不能写首次local adaptive counting/首次tempo drift。视频级监督的具体预算必须对齐后才能比较。将其强行改造成NUDFT/三专家/NOLA对照没有全文依据；目前只允许保留为closest comparator pending，不能以不可得为由从文献相关性中删掉，也不能声称已排除精确机制碰撞。

## 对现有TempoRAC的具体验收影响

| 现有主张 | 本轮可确认的邻近性 | 剩余比较工作（不启动、不改设计） |
|---|---|---|
| C1 identity-local多人计数 | MultiCounter+明确native MRAC、query与时间关联 | native RGB端到端表与supplied-track表分开；不能把轨迹裁剪适配称未经修改原生基线 |
| C2 detached局部NUDFT + slow/medium/fast软响应融合 | Plus长短周期、自相似；TWCRAC局部统计与dual windows | 最终正文机制仍未取得，不能给“无相同机制”的负断言；保留对现有方法的因果检验要求 |
| C3 正分母NOLA后每人一次decode | 已读Plus代码中可见累计period质量解码，与当前概率融合不同 | 这个局部代码差别不能提升为最终论文排除碰撞；输出period坐标/scorer仍待闭合 |

没有本轮证据支持SOTA、三个组件均有效、六域/七框架矩阵、完整baseline可运行或新颖性accepted。主线保持共享encoder、既有三tempo experts、detached NUDFT soft routing、NOLA；不恢复作废路线。两篇全文缺口保持具名，后续只需补真实primary PDF/作者完整稿及supplement，然后按全章节与页码登记，而不是再次重复完成的三篇。

## 检索、读取和失败记录

- 本地先扫描E:/ICASSP_TempoRAC目标文件名及PDF，发现的只有既有PAMS/项目稿等，目标未命中；未重读这些PDF。全E扩展搜索中止，记录为不完整。
- IEEE document/abstract、stamp和Crossref PDF；SSRN记录/官方Delivery、ScienceDirect正式页/abs/pdfft、DOI/linkinghub、Crossref所列Elsevier TDM XML/plain均尝试。具体响应见 `retrieval_receipt.json`、`followup_receipt.json`、`READ_RECEIPT.json`。未进行登录、付费、反机器人绕过或作者联系。
- 官方GitHub connector取完整recursive tree，`truncated=false`且PDF paths为空；README的 `output (2).pdf`链接404。12个关键源码保存于 `code/`，按固定SHA链接，不取移动main。
- exact-title/Siamese/supplement/GitHub/作者主页补检没有找到合规全文；第三方结果只作发现，不作最终机制依据。没有自源TempoRAC网站证据。
- pinned arxiv helper第一次宽泛query输出遇GBK编码失败；显式UTF-8后精确title query成功返回 `[]`，保存在 `arxiv_identity_search.json`。空查询结果不是领域无先例。
- PDF skill已读，但目标无PDF获得，无渲染/逐页阅读，不伪造render成功。未创作Word/PDF，论文排版验收不适用。

哈希、输入绑定、各文件读取范围与未读正文项见 `READ_RECEIPT.json`；完整产物SHA见 `SHA256SUMS.json`。报告已完成并可供父流程审查；科学覆盖仍部分，不应自动更新主库成5/5。
