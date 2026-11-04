# TempoRAC 首次完整目标诊断冻结 DDF1（2026-11-04；两页子契约）

**决定：在本任务未访问自然 GT、未见完整自然诊断结果前，一次冻结 endpoint_policy=e_minus_s_v1、count_field=count、periods_field=period、bootstrap_replicates=0。** 仅用于首次完整 train/dev 的 G-NT1 目标单位/事件/支持描述；不再等待官方端点认证，不运行多个约定择优。官方端点与 AvgMAE/AvgOBO 仍未认证，无 target-quality PASS、PeriodAP、CI 或显著性结论。

请求角色按科学设计策略为 gpt-6-astra；实际 backend/model/effort 未独立 attested，同族判断 **provisional**。本任务仅写本文及同名 receipt，不改源码/测试/仓库/共享状态，不运行 GT、测试、训练、GPU、服务器作业或提交。

| 可核对验收 | 状态 / 证据 |
|---|---|
| GT/结果前固定口径，保留既有方法 | 通过（规则）；以下两页及 receipt，不代表实测质量 |
| 完整分母、失败、端点及源点/边区别 | 通过（规则）；第1页 |
| 支持分类与合法非整数端点一致 | 部分；第2页有一个静态反例，待窄修订 |
| 完整977/2364诊断、官方/科学质量 | 未达标；完整raw/NB1/NT1及后续证据仍待交付 |

## 第1页：唯一描述口径与完整分母

**依据与限度。** 已保存的官方 MultiCounterPlus 固定提交 d15c767d82dac2f7a5df9fde1d7f6b5a4f491062 中，video2frames.py:21–30 用 e−s/range(s,e)，multirep_eval_api.py:650–677 的 tIoU 也无+1；但前者106–114的 periodicity binary 含 end。故原值差是有来源的描述性选择，不能据此认证官方标注语义。字段名 count/period 继承 runbook 的已存 parser/aggregate 证据，本任务未重开自然标签。将来权威证据导致变更须另立版本、保留本次结果，不覆盖或择优。

**端点与支持。** NB1有N个源点0…N−1、E=N−1条相邻源边i=0…N−2；边i对应描述区间[i,i+1)。原GT period解释为[s,e)，要求0≤s<e≤N，分母L=e−s。保留合法有限非整数端点，不+1、round、裁尾或猜stride。对每种 observed/response/target/training mask m：

- o_i=max(0,min(i+1,e)−max(i,s))；coverage_fraction=Σm_i·o_i/L。
- H={i:o_i>0}；has_support=any(m_i for i in H)。
- fully_supported=(e≤N−1) AND all(m_i for i in H)。
- unsupported=NOT has_support。
- partially_supported=has_support AND NOT fully_supported。

分类互斥且穷尽，**不采用任何端点/支持容差**；真实正长度缺口即使<1e−12也保留为partial。fraction保留实算值，分类不靠isclose或clamp掩盖缺口。三类计数之和须等于全部合法period条数，负数/重叠分类是工程失败。

[N−1,N)必为unsupported；[a,N)前段有支持则partial；[a,N−1)可为full。N=1无边。源边界N不得裁成sample N−1，也不创造最后一条边或末帧秒数。此结构性尾部无支持须披露，不能自动解释为模型/动作错误。

**伪单位。** NT1 traversal [l,r)含源samples l…r，唯一pulse位置p=l，计入period当且仅当s≤p<e。不改成NB1响应组件的midpoint p+0.5。Σpulse=K只证明certified traversal数，不证明真实repetition。逐原period报0/1/≥2 pulse及full/partial/unsupported；重叠/重复period保留原顺序和全部分母，同一pulse可进入多个period。outside-pulse按原period union算一次。空period list分母0、覆盖比例null；有pulse则全在空union外，无pulse的outside比例null。0/≥2只能提示单位/漏检/对齐问题；one-pulse+full也不是单位有效证明。没有合法period decoder，PeriodAP仍unavailable。

**计数。** 仅纯几何G-NT1，hybrid_mode=unavailable，沿用v12→跳过unavailable v9→v15顺序。c=K=Σpulse；certificate rejected为count0+abstain且进入完整分母，partial不按覆盖率放大。缺artifact/错hash/NaN/schema或mapping错误使整次评估失败，不伪装certificate rejection或删行。

机器reference沿用 FULL-v1-supplied-track-proposed-video-person-macro。首要描述量nmae↓（展示名Proposed AvgNAE_GT/NMAE）=mean_video mean_person |c−g|/g，g>0。同权重报raw MAE、标准OBO=1[|c−g|≤1]、signed error、relative signed error、c/g；不round，不pool全部人物或改source等权。count0且g=1可满足标准OBO，不能叫检出成功。不得裸称官方AvgMAE或与官方数值混排；这是supplied-track/bbox-assisted目标诊断，不是端到端性能。

train794/1878与dev183/486分别报告，共977/2364；四组train同媒体alias、58/48冲突原行保留。另报certified-only的较小视频/身份分母，空条件人口undefined，不替代full。支持漏斗保留身份/源帧/源边/事件、known seconds及unknown-time edges；无已知秒为null，不以源帧比例冒充秒覆盖。首报告bootstrap0仅描述，后续三seed/source-cluster统计另冻。

## 第2页：原始值、一个必要修订与衔接

**原值与失败。** 原count、period、条数不一致、原annotation bytes/SHA与负结果均保留；不由period条数重建count。合法非整数端点直接计算；s=0/e=N合法；空list合法零period。[s,s]、逆序、负/超N、NaN/Inf、bool/string、非二端点/非list或GT count≤0/非法类型：保留原文件与错误/退出证据，整次诊断failed，不补ε、不修GT、不drop、不换unresolved。当前evaluator在evaluate_rows后才写identity_rows；非法值会在输出目录创建前失败，**不可称已输出完整诊断行**。必须保留实际命令/config/input provenance/异常并明示未完成；必要数据契约修订另记，不以GT调producer/阈值/分支/sampler/checkpoint。

**唯一必要实现修订。** 所读native_target_diagnostics_v1.py SHA：
95b8686caa725d3c220e68fb74e13dab7b822f41fe3d3dc1e12c10155ab3fee5。
301–302行用abs_tol=1e−12的isclose判full，却严格0<length<L判partial；369–386行独立计数后以total−full−partial得unsupported，可产生负数。

未执行的静态人工反例（不是测试PASS）：N=17、16条边均true，period[0,16+5e−13]、pulse[0]。旧规则full=true且partial=true，单period汇总unsupported=−1；冻结结果必须仅partial。period[16,16+5e−13]必须仅unsupported；[0,16]必须仅full。5e−13仅是可表示反例，不是容忍阈值。对四级mask分别验证，保留fraction原算法和pulse/count规则。

主任务仅需按第1页正重叠集合规则修分类/分区恒等式，加入上述人工回归并跑原38项isolated诊断CPU suite，独立窄复核后冻结新源码SHA。此前F1/F2及38项工程证据保留，不重开泛审；本任务未改源码/测试或运行反例。

**衔接。** Runbook既有exact config显式填 count_field=count、periods_field=period、endpoint_policy=e_minus_s_v1、bootstrap_replicates=0；使用已有新输出路径后缀e_minus_s_v1。本文/receipt SHA记录在外部运行证据，不擅自在exact config加键。完整产物生成后才填实际input SHA；修复源码另记SHA，不冒用旧95b8686c。

完整raw终态/原恢复责任→复用已完成977 decode→全overlay/NB1/NT1的977/2364验证→上述窄修订通过→独立CPU evaluator一次完整train/dev诊断→一次证据化“继续完整方法检验/语义失败先修复”判定→原NTR1 R2a完整plan/实际lease门槛。可并行准备，不能用部分生产诊断替代完整人口；不新增GPU/训练预算或访问test。

输出仍写 official_metric_status=not_certified、period_ap=unavailable_no_legal_period_decoder、target_quality_acceptance=not_evaluated_no_tolerance_frozen。共享encoder/三expert/detached NUDFT/正分母NOLA、NT1/LF1、NTR1 R2a、健康extractor及历史失败均不变。端点未决不再无限阻挡描述报告；完整数据、真实语义、质量容忍界、官方资格、三seed/强基线/三组件/六数据集七框架/SOTA/最终稿要求仍各自未达。

**读回执。** 已完整读取E:\AGENTS.md、项目AGENTS/ARIS_LOCK/WORKFLOW/CVPR_RESUME/当前run snapshot；pin f76ec8d6b77792e2242cca049e08d0cbb7d54e19的experiment-plan与literal ../../shared-references的output-versioning/manifest/language/composition。源/安装技能SHA同为61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a。相关NB1/NT1/LF1/诊断/审查/NTR1/runbook/metric文档及诊断源码全文、官方证据范围和所有SHA详见receipt；未读六个训练runtime源文件。指定两文件所有权覆盖默认计划/追踪表输出，无共享state/manifest写入。

