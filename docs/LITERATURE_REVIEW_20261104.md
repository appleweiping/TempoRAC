# TempoRAC 有边界文献刷新（2026-11-04）

本轮结论：现有主线可以继续做有边界的科学验证，不能据此写“首次多人计数”“首次三节奏专家”“首次局部变速评估”或新颖性已通过。已经取得并读过 MultiCounter、PAMS、TReViS 的完整方法与实验正文；PAMS 的局部变速补充也重新核验。MultiCounter+ 和 TWCRAC 的完整最终方法仍未取得，不能用摘要代替排除精确机制碰撞。

本报告是检索/阅读交付，不是独立 reviewer 判定，不改冻结方法、不启 GPU、不重跑历史、不接触原始自然 test、不联系作者、不推送。参考 docs/CVPR_RESUME_20261104.md 的当前授权；旧 proposal 中已被接续协议修订的流程约束不重新引入。旧 WARP-PHASE / pivot notes 只复用文献事实，不恢复路线。

## 1. 验收表与覆盖回执

| 检查项 | 结果 | 可核对证据 |
|---|---|---|
| 三主张各至少三种查询措辞 | 通过 | §2 的 Q01–Q09；JSON searches；E盘 web_retrieval_raw_20261104.json |
| 近六个月优先 | 通过，但不是穷尽 | 2026-05-04 至 2026-11-04 的日期查询；ARIS arXiv exact-phrase 日期过滤返回 TReViS |
| 至少三篇可获取近邻全文方法/实验精读 | 通过 | MultiCounter HTML §§4–5；PAMS pp8145–8150+补充§7.2.1；TReViS HTML §§3–4、§6.1 |
| 逐主张近邻、差异与未排除项 | 通过 | §4；JSON claims/core_papers |
| MultiCounter+ / TWCRAC 完整方法排除碰撞 | 未达标 | 两篇仍为摘要/官网说明/片段级，分别列出未知项 |
| 原生基线已能运行 | 未达标 | 只核对公开代码/权重入口，不把链接当成功下载或实测 |
| 独立创新性审查 | 未在此子任务执行 | 由主执行流程接续；本报告不赋予最终 PASS |
| 六数据集×七顶会框架与 SOTA | 未达标且非本轮实验任务 | 不能把下列论文/数据名或作者报告值当已完成实验矩阵 |

冻结核心单元：MultiCounter、MultiCounter+、PAMS、TReViS、TWCRAC。全文覆盖 3/5；部分覆盖 2/5；未覆盖的完整正文为 MultiCounter+、TWCRAC。其余 RepNet/TransRAC 等是历史 notes 复用或仓库级可用性核查，没有冒充本轮全文精读。

起始 checkout 的 papers/、literature/ 未发现旧 PDF 库；已复用 idea-stage/PIVOT_LITERATURE_LANDSCAPE.md、idea-stage/NOVELTY_CHECK.json、idea-stage/TWCRAC_ADJUDICATION.json。旧 novelty 中的科学候选和 verdict 不沿用到 TempoRAC。Zotero/Obsidian 未配置且未被指定；S2/DeepXiv/Exa/Gemini/OpenAlex API 未被明确要求，未调用。搜索结果中的第三方聚合只用于发现，一切机制判断绑定下列一手来源。

## 2. 实际检索记录

全部实际执行于 2026-11-04。保留原始返回（含无关命中和失败），没有把查询结果少解释为领域空白。

| ID | 主张 | 实际查询 |
|---|---|---|
| Q01 | C1 多人身份内计数 | `"multi-instance" "repetitive action counting" after:2026-05-04 before:2026-11-05` |
| Q02 | C1 | `"person-wise" "repetition counting" 2025 2026` |
| Q03 | C1 | `"MultiCounter+" "temporal" "period"` |
| Q04 | C2 非平稳局部节奏/专家 | `"repetitive action counting" "non-stationary" 2025 2026` |
| Q05 | C2 | `"repetition counting" "mixture" "experts" 2025 2026` |
| Q06 | C2 | `"period-adaptive" "multi-scale" consistency counting PAMS` |
| Q07 | C3 连续重建/一次解码 | `"repetition counting" "overlapping windows" reconstruction` |
| Q08 | C3 | `"repetitive action counting" "sliding window" "density" 2026` |
| Q09 | C3 | `"action counting" "overlap-add" decode` |

另实际检索了 PAMS 精确标题/代码、TReViS 代码、TWCRAC dual-window、2026-05-04 后 tempo、arXiv 2026 mixture/overlap，以及 OpenReview、NeurIPS proceedings、PMLR 的 repetition counting 2025/2026 组合。后面三项只是限定站点检索，不是逐篇查完 ICLR/NeurIPS/ICML 全部论文；Google Scholar/Semantic Scholar 的完整索引没有系统导出，未宣称覆盖。

ARIS arxiv_fetch.py 的普通主题检索成功；随后错误地尝试了不受支持的 `--sort submittedDate`，CLI 拒绝，未产生证据。读取 help 后使用原生 query `all:"repetitive action counting" AND submittedDate:[202605040000 TO 202611042359]`、max20 成功，仅返回 arXiv:2609.24367。这是精确短语结果，不覆盖所有别名和旧论文的新修订。结果在 E:/ICASSP_TempoRAC/aris-work/literature/arxiv_recent_window.json。

## 3. 核心论文：机制与实验阅读

| 工作/身份 | 本轮证据级别 | 已核实的关键内容 | 与 TempoRAC 的关系 |
|---|---|---|---|
| Tang 等，MultiCounter，ECAI 2024 | 全文 HTML：§§3–5，Eq1–10，Tables1–5 | instance-query RGB 联合检测/跟踪/计数；64帧窗、32帧重叠、相邻clip用bbox IoU链接；周期速度/周期性监督 | 多人异步、身份输出、变周期处理、重叠clip均已有 |
| Tang 等，MultiCounter+，TCSVT 2026 | IEEE 摘要/引言可见内容+作者 README；完整正文未获 | Siamese-query时间一致实例头；anchor-guided similarity reconstruction period head；长短周期感知、合成预训练 | 必须保留的更强 MRAC 近邻；不能据摘要断言其没有局部路由 |
| Gao、Li、Li，PAMS，CVPR Findings 2026 | 官方全文 PDF+相关补充正文 | period-adaptive多尺度自监督、Fast/Medium/Slow推断峰检测专家、计数投票；补充含局部变速 | 最直接的专家/tempo邻居；可训练响应专家与概率级重建才是需验证差异 |
| Yu 等，TReViS，2026作者接受稿 | arXiv全文方法/实验+§6.1；官方comments/README | TSSM估计粗周期/计数统计，合成带伪标签训练数据；模型无关、训练期数据生成 | 占据“从无标签重复结构恢复监督”的邻域；粗统计不等于每原始物理周期都准确 |
| Li 等，TWCRAC，Pattern Recognition，article115199 | 出版社abstract/introduction/highlights/section snippets；完整正文未获 | intra-video TCC、video-level labels、local-statistic动态阈值、adaptive dual-window处理tempo drift | C2最高未决项之一，精确机制不能从片段判同/判异 |

### 3.1 MultiCounter

一手：[arXiv全文](https://arxiv.org/html/2409.04035v1)、[ECAI DOI](https://doi.org/10.3233/FAIA240494)。作者：Yin Tang、Wei Luo、Jinrui Zhang、Wei Huang、Ruihai Jing、Deyu Zhang。§4.1 MSTI 使用实例queries与RoI特征，先做实例内时间通信，再跨实例空间通信；这里的 TCC 是 Temporal Conditional Convolution，不能与 PAMS 的 Temporal Cycle Consistency 混为同一模块。§4.2 period head输出逐帧周期长度分类与周期性；§4.3 对存在/box/周期/周期性做匹配监督。

§5.1 MultiRep 是从 RepCount 按FPS分组后横向拼接的1157条合成视频，含52590个periodic events，论文按7:2:1分割。它去除了 A+B+C 与 B+C+A 这样的组合重复，不能据此认为不同split的源视频也独立；TempoRAC仍要审 source-video provenance。§5.3 已使用64帧、重叠32帧的推断窗口并跨clip关联。因此“重叠窗避免重复计数”的论文定位必须比简单使用滑窗更具体。

作者 Table1 报 Period-mAP 10.60、AvgMAE 0.239、AvgOBO 0.554，仅作原文定位，不导入 TempoRAC 表。§5.2 Eq9 的HTML展示似缺少绝对值，Eq10的括号层级也需官方scorer判定；不能凭名字把 per-identity pooled MAE 当官方 video-averaged AvgMAE。§5.5 有period-varying actions对照，§5.7有逐实例变速读出；异步/变速问题本身不是空白。

### 3.2 MultiCounter+

一手：[IEEE记录](https://ieeexplore.ieee.org/abstract/document/11421441/)、[作者仓库](https://github.com/yinntag/MultiCounterPlus)。IEEE显示 Date of Publication 2026-03-04、TCSVT36(7) pp9462–9476；不能把July期刊卷期当首次公开日期。官方README明确主张实例时间一致性、anchor-guided长短周期建模，以及speed augmentation。其 Table 报14.56/0.223/0.571，与MultiCounter作者Table1基线同类但不是TempoRAC实测。

仓库有 configs、mmdet、train/test脚本及原生checkpoint链接；本轮没有下载权重、安装依赖或执行。完整方法/损失/最终实验未读到，所以本报告对其局部时间窗口、路由、碎片处理和重建机制均保留未决，不制造“未包含”的断言。

### 3.3 PAMS

一手：[官方正文](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)、[官方补充](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf)。出版身份明确是 CVPR Findings 2026，不能记成main-track。

pp8145–8147 §§3.1–3.3：Transformer产生embedding；对stop-gradient embedding/pose-energy估计周期，使用0.5/1/1.5倍周期窗口构建多正样本InfoNCE，并加跨视频/cluster负样本。Algorithm1的Fast/Medium/Slow是同一period stream上的不同平滑和峰检测参数组合；每个专家先形成count，再多数投票，平票靠FFT参考count。TempoRAC若成立，差异是共享的可训练响应分支在每个身份局部窗按cue融合概率，重建后只调用一次decoder；不能把专家名称或三尺度作为区别。

原文Table1报告RepCount0.129/0.842、Countix0.098/0.879、UCFRep0.208/0.686；同表CountLLM在UCFRep0.116/0.839更强。不能照搬摘要“普遍优于所有监督方法”的表述。正文§4.3有训练/推断消融，§4.4含0.1x–2x全局速度；补充§7.2.1更已在每个152视频随机选5–7秒、0.5x–1.5x局部重采样、重复10次（1520 trials），报0.1336 MAE/0.7447 OBO，Tables4–5给count分层。局部tempo drift评估不是TempoRAC首次。

可复现未决：§3.1.1称PeriodHead只用于推断、loss直接作用embedding，其head准确训练路径仍需author-native实现或补充说明；§3.1.2存在sum/平均的表述差异；§4.1 MAE仅文字描述，归一化公式未给。它们是复现问题，不是证明原论文无效，也不允许用本地历史复现冒充原生。此轮未定位可验证的官方可运行代码/native权重。

### 3.4 TReViS

一手：[官方元数据](https://arxiv.org/abs/2609.24367)、[全文HTML](https://arxiv.org/html/2609.24367v1)、[作者仓库](https://github.com/yfqi/TReViS)。首次arXiv2026-09-21，在近六个月窗口内。官方comments说明已被 Image and Vision Computing 接受，是作者接受稿、非最终版本，DOI待补；不能简单称未审稿预印本，也不虚构最终卷期。

§3 Eq3–5、Algorithm1：VideoSwin把非重叠chunk编码成TSSM；随机anchor及数据相关相似度阈值提供粗cycle/count统计；高斯采样周期长度与重复数、重复原始片段并以q插入无关段，形成伪标签训练现有RAC网络。§4.1明确c8、K20、q0.1、tau0.2及采样边界/fallback。它是训练数据生成，不是推断时identity-local专家，也不直接保证原始序列每个物理周期恰对应一个伪事件。

Table1的作者结果TReViS+TransRAC为0.4955/0.3046、+SimPer为0.4671/0.3320（normalized MAE/OBO）；§4.5的UCFRep/Countix为从RepCount-A迁移，不能算三个独立in-domain训练。§4.7 Table12直接比对粗统计误差：absolute count MAE2.1385、cycle-length MAE0.8065，已说明估计并非准确标签。§4.9 Table14用148个至少两周期视频的cycle-duration CV三分位做自然变速分层；MAE近似稳定但OBO随高变异下降，非显著相关不证明等效。TempoRAC应直接报告目标/事件误差与支持覆盖，不能仅凭最终count接近就宣称伪监督有效。

仓库README实际只有论文说明和“Coming Soon”；没有可运行发布或checkpoint。摘要“code available”不能覆盖这个实际状态。正文还有758训练视频与§4.10计时757条的不一致，后续复现需核对，不自行替作者补数。

### 3.5 TWCRAC

一手：[出版社DOI页面](https://doi.org/10.1016/j.patcog.2026.115199)、[ScienceDirect记录](https://www.sciencedirect.com/science/article/pii/S003132032602162X)，旧笔记另有[SSRN记录](https://doi.org/10.2139/ssrn.6630216)。页面显示March2027 issue，但2026-11-04已可检索到在线记录，不得因卷期年份将其排除；首次公开日期仍需确认。

本轮能读的是摘要、引言、highlights和方法/实验section snippets，不能称全文。可确认其同时说video-level labels、intra-video TCC、local-statistics动态阈值与adaptive dual-window、tempo drift；这已经要求TempoRAC缩窄“自监督/局部时间窗口/动态节奏”主张。精确loss、局部周期估计、专家/重建/decoder、训练和评估全流程均未排除碰撞；43+FPS和三benchmark优越性仅是作者摘要主张，本报告不据此建立同口径性能排序。旧SSRN403/超时不是本轮重新尝试成功。

## 4. 逐主张边界

| 主张 | 现有最近证据 | 可以检验的残余差异 | 尚欠证据 |
|---|---|---|---|
| C1 身份内多人计数 | MultiCounter/Plus已定义MRAC、person-wise输出、异步变速 | 相同外部轨迹输入下，private state是否保证一人tempo干预不影响另一人；是有效性条件，不单独称首次贡献 | 真实provenance、完整轨迹、干预与native端到端对照；不能用GT bbox assisted pilot替代 |
| C2 非平稳局部节奏与软专家路由 | PAMS训练period adaptation+三推断专家+内部局部变速；TWCRAC双窗动态阈值；TReViS粗周期自监督 | 三共享可训练响应分支在每身份window-local相对节奏下做detached软概率融合；与global/uniform/shuffled及capacity control相比的可重复增益 | 完整训练、专家专门化、cue置乱/取消、clean保持、三seed不确定性；TWCRAC全文 |
| C3 连续重叠重建后一次解码 | MultiCounter64/32重叠clip关联；PAMS连续stream和多专家count vote | 概率先融合→严格正分母NOLA→每身份一次固定事件解码，能否消除offset/边界重复事件 | 不同offset/长度/缺口/重叠布局的事件级边界证据，重建移除/替换；完整近邻decoder查阅 |

建议的可验证定位句（是研究问题，不是已取得结果）：

> 在外部给定的身份pose轨迹上，检验window-local相对节奏cue驱动的共享响应专家软融合，是否在连续重建及每身份一次解码下，比最强匹配非局部对照更好地保留变速计数。

PAMS已经把同一曲线的多个decoder-count投票做好了；要证明TempoRAC不是仅换聚合位置，必须观察局部cue与分支响应的对应、破坏对应关系后的效应，以及NOLA边界不变性。近邻密集并非否决理由，但未找到完全相同组合也不是创新通过证据。暂列 PROCEED WITH CAUTION，仅指继续现有研究；不赋数字新颖性得分，不宣布最终novelty PASS/ABANDON。

## 5. 基线可用性与公平比较

| 基线 | 本轮官网核查 | 当前可用性/口径限制 |
|---|---|---|
| MultiCounter+ | [官方仓库](https://github.com/yinntag/MultiCounterPlus) | 有训练/测试/权重链接；原生RGB端到端，含人框与period监督。不能直接与GT-box pose计数阶段称同预算同输入 |
| PAMS | 上述官方正文/补充；精确标题+GitHub检索 | 未定位可审计原生代码/权重；历史本地版本须标复现及歧义 |
| TReViS | [官方仓库](https://github.com/yfqi/TReViS) | Coming Soon；不能计作已可运行框架 |
| TWCRAC | 上述出版社记录 | 完整方法、官方代码与权重均未核实 |
| TransRAC（Hu等，CVPR2022） | [官方仓库](https://github.com/SvipRepetitionCounting/TransRAC) | 有代码与checkpoint链接；README2025-02-06给RepNet评测差异澄清，2024-11-14说明比较框架在RepCount-A重训；RepCount-B原视频不公开 |
| PoseRAC（Yao等，arXiv2023；本轮未核定最终venue） | [官方仓库](https://github.com/MiracleDance/PoseRAC) | model/eval/原生权重文件可见；使用pose-saliency额外人工监督，不能称同自监督预算 |
| DeTRC（Ma等，IJCV2026，README口径） | [官方仓库](https://github.com/lizishi/DeTRC) | TSN特征、MMAction/MMCV和1D自定义算子；有Baidu权重链接；非仅凭名字算顶会 |
| D²-STX（Wang等，本轮venue未核定） | [官方仓库](https://github.com/wwwwwxa/D2-STX) | video/pose/fusion与exemplar流程；pose特征提取仍待发布、预提特征要联系作者，本轮未联系 |

上述均未运行，链接不等于下载成功。比较表应分native端到端与相同supplied-track计数模块，并保留最强对照；不能静默删去输入更强但已比较的方法来保证SOTA。没有帧级/周期定位输出时，不能凭整数count生成伪合法period intervals填Period-mAP。

本轮未冻结六真实数据集的合法完整来源与任务适配：MultiRep、RepCount-A、Countix、UCFRep只是核心文献中的四个名字，MultiRep又由RepCount组合而来，不自动算四个独立源域。姿态版本、speed扰动、子集不能加成六数据集。当前也没有七个被核定为顶会且跑通的框架。

## 6. 未覆盖范围、污染与证据文件

最重要缺口是MultiCounter+及TWCRAC全文；其次是各原生scorer、权重实际可访问、完整source-disjoint split、最新别名文献、PAMS官方实现和TReViS正式发布。其他从旧notes继承的邻居（SimPer/CycleCL/DeepPhase、Rethinking TSSM、HTRM等）本轮未全部重读，不能声称最新精确碰撞已全排除；未为扩大表格重新跑旧路线。

检索中命中了[作者本人公开主页](https://appleweiping.github.io/)，其中TempoRAC已有Sep2026 submitted ICASSP2027、identity-conditioned encoder、NOLA及“extensive experiments”摘要。该页面是自源，严格排除独立新颖性和性能证据；当前接续协议仍说主要实验未完成，冻结方法也未用identity token。因此公开表述与当前可审计状态有差异，主执行流程需据实标明；本轮未修改网站或联系任何人。

机器证据：docs/literature/evidence_20261104.json，包含canonical IDs、作者/venue、阅读层级、段落定位、URLs、基线可用性、9查询和coverage receipt。

原始缓存统一在 E:/ICASSP_TempoRAC/aris-work/literature/：multicounter.html/.txt、trevis.html/.txt、pams.pdf/.txt、pams_supplement.pdf/.txt、两个官方README、download_receipt.json、web_retrieval_raw_20261104.json、web_retrieval_extra_20261104.json、arxiv_recent_window.json。官方PDF通过本地HTTPS成功下载，而web工具对部分CVF/ScienceDirect URL返回403；下载成功与web失败均保留，不把403解释成无论文。此轮只做文本阅读，没有输出Word/PDF稿件或宣称完成其逐页排版验收。

