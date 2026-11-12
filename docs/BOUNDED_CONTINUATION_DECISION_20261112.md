# 2026-11-12 有界恢复决策与启动验收

**决策：继续既有 no-LLM 冻结框架；本组只恢复已释放的 full_s0/full_s1/full_s2 三个完整修正训练，并并行完成既定 RepNet 357 条官方发布权重评测。其余 24 个冻结规格保持未释放，不自动重跑 27 项，更不重跑原有 29 项。**

这是暂停后的一次性调度/恢复决定，不是新设计或再次查新。使用父任务提供的最新用户恢复授权。requested_model=gpt-6-astra；actual_model=null（未独立证明）；actual_effort=null；session=/root/resume_direction_20261112；review_independence=same-family；acceptance_status=provisional。本决定允许证据收集，不表示 SOTA、创新性或投稿质量已通过。

## 本组验收表（启动前冻结）

| 要求 | 必须留下的证据 | 当前判定 |
|---|---|---|
| 沿用现有有效资产 | 原 29 项完整模型/预测及缓存保留；暂停日志保留；新 attempt 与旧 attempt 不混淆 | 通过（既有状态证明保留；执行器仍须保存新 attempt 回执） |
| 修正训练协议不漂移 | manifest SHA `96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2`；仅四训练行排除；RC756/UCF331/CX4222，共5309，平衡采样8570/epoch；全部旧评估记录不变 | 已冻结；启动现场核验待办 |
| 三次完整正式训练 | seeds0,1,2，每次50 epochs；原配置、训练目标、验证选择、两视图推断不变；完整checkpoint与逐视频预测 | 未达标；旧12/13轮无权重，不能作为完成或恢复点 |
| 可恢复运行 | 原子保存每epoch latest训练状态及持久 best-by-dataset checkpoint；有效恢复和防重复日志/评估校验 | 启动前必须补齐；目前未验收 |
| 官方外部基线 | 原 RepNet 官方core/权重/预处理/参数、357输入清单与冻结评分；保留连续原输出及harmonized指标 | 静态闭环已通过；新调度条件、运行见证和完整评测待办 |
| 不扩大矩阵、不重复研究 | 继承15-framework ledger及缺口；本组仅3训练+既定357评测；无新数据/超参/架构搜索 | 本决策通过；按执行回执验收 |
| 真实科学质量 | 训练修正、历史test-informed选择、评价标签冲突、外部训练资源差异明确区分 | 仅部分；SOTA、六域×七框架、所有组件贡献、最终论文均未达标 |

## 顺序与释放边界

1. 先完成一次必要的**服务器端**启动核验及训练持久化修复；只复核受影响的恢复逻辑和冻结配置差异，不重做原有科学审查。不运行小样本GPU实验，不重建特征。将暂停的 attempt 日志与旧路径完整保留，在新 attempt 路径从随机初始化启动 s0/s1；s2同样从随机初始化开始。原12/13轮模型只存在于已终止进程内存，不能恢复，严禁报告“从12/13轮续训”。
2. GPU0先跑 full_s0。GPU1通过下述新调度核验后立即完成 RepNet witness 和 full357；这与 full_s0 没有科学依赖。RepNet不必等任何修正训练的测试分数。
3. 正式任务启动前一次性修订资源顺序：GPU0执行 full_s0 → full_s2（各自含既定评估）；GPU1执行 RepNet witness → full357，只有RepNet完整成功且确认其进程退出、GPU1空闲后，才启动尚未运行的 full_s1及既定评估。RepNet失败时 full_s1保持未启动，先解决具体故障，不跳过成功门槛。该安排仅将同组三seed中尚未启动的s1排到基线之后，不迁移健康任务、不改变科学配置、不释放其他24项。最多两项GPU任务同时运行、每卡一个。有限队列执行完退出，无定时器、自动重试或常驻服务。
4. 每个训练种子依原验证集选各数据集checkpoint，再按冻结规则对完整原评价人口评估一次。失败后的显式恢复只补未提交工作，保留失败证据。三seed与RepNet完整结果形成**一次组末汇总**；组中只监控/修故障，不据中途test结果改方法、选seed或放大矩阵。
5. 其余24项不是技术上必须等全部3seed才可启动：原合同仅需要首个完整full证明执行链正确，匹配消融的科学分析则需要对应full种子。但目前仅3项已释放，且先获得修正主结果与真实外部基线的信息价值最高，因此本次明确不释放24项。完成本组后可复用原冻结规格开展下一组，无需再做概念审查；不能把本决定当作整27项自动队列授权。先核验旧 single-UCF 三seed的完整有效配置/训练人群/源版本等价性，能证明等价就复用；其余受修正训练影响的旧消融只保留历史用途，不冒充修正协议对照。

三次修正full仍是本组必要交付：它们回答去除已知训练曝光后的主模型表现，并为后续配对消融提供三seed基准。它们不恢复数据集级独立性，也不能用新seed消除历史test-informed选择。名义训练预算沿用5–8 GPU-hours（三次总计150epoch）；RepNet沿用一次357条既定预算，不把未知运行耗时编造成精确数字。不得因超出估时缩短完整训练。

## 仅限运行恢复的工程要求

每epoch边界持久保存 model、optimizer、scheduler/其等价步数、AMP scaler（若使用）、已完成epoch/global step、Python/NumPy/Torch CPU及CUDA RNG、显式DataLoader/sampler generator状态（若存在）、各dataset最佳验证值及其checkpoint身份，以及配置/数据/源码身份。保留已有seed/初始化/采样顺序，checkpoint写入本身不消费训练随机流。先写临时文件再原子替换；恢复检查身份不匹配则停止。best权重必须独立持久化或包含于可恢复状态，不能再只放内存到训练末尾。中途结束的epoch可从最近完整epoch重新执行，日志不得把未完成epoch算入完成数。必要恢复测试在服务器CPU用可控状态/小张量完成，不运行本地实验或GPU缩短训练；真实正式首epoch检查持久状态实际存在。实现者和独立Sol复核一次具体变更即可，已闭环事项不重复审。

## RepNet 调度依赖的明确修订

既有 `repnet_official_closure_20261111.md` 已关闭五项静态阻塞并给出 CLEARED_FOR_SERVER_RUNTIME_GATE。复用该闭环，不再次审查整适配器。原合同要求旧 full_s1/worker387997 完成是为了避免占用GPU1；现在用户暂停使旧队列终止，该完成条件永远不应伪造。

本次只将**等待旧full_s1成功/旧queue_complete**改为：保存最新暂停/进程现场回执，证明旧TempoRAC训练与有限队列已退出，无待唤醒的旧 worker；核查GPU1无其他计算进程、至少10GiB空闲，并确认GPU0训练与GPU1评测各自绑定正确。保留原8GiB进程内存上限、全部source/config/input/metadata/checkpoint身份检查、逐变量恢复见证、一次64-frame有限kernel见证及357条正式运行。新launcher配置与源hash必须更新并保存；旧回执不可冒充新调度版回执。此改动是资源前驱替换，不改变任何科学输入、预测或评分。

full357=RepCount152+UCFRep105+QUVA100，**不包含Countix**；保留零计数及争议视频，UCF按既定inclusive segment处理，完整解码长度核验，357条原子记录及trace齐全才算完成。使用官方更新的ckpt-70，标注其2025发布身份，不能说复现了2020论文原checkpoint或同预算清洁训练。已完成ESCounts结果复用；既有可恢复RepNet记录若身份完全一致则复用，不为目录名变化重算。

## 本组后如何判读，什么工作退出

Primary仍是 positive-count `abs(round(pred)-g)/(g+0.1)` MAE（越低越好），保持NumPy舍入；报告OBO/OBZ/RMSE、/g次指标及零计数/全人口结果。历史29项与修正协议绝不混池。修正组完整后报告三个seed的均值/SD，既有source配对分析口径及多比较家族保留；不以不显著证明等效。完整结果与已读15-framework ledger作有标签比较，官方发布权重、文献值、内部head分开。

继续保留当前框架的实证检验价值；没有依据仅凭领域拥挤宣布无创新，也没有本轮完整查新通过。退休MultiRep/geometry/tempogram、失败LLM/encoder-adaptation路线及不适配的internal-style基线停止新增投入、不得恢复主张或自动重启；原代码、失败结果、配置和选择记录保留，不做清理。UCF/Countix历史不占优不能被改写成SOTA；用户“不在主比较报非SOTA域”的偏好与完整范围披露并存，应在最终稿阶段明确主表范围并保留可找到的全部结果，不能暗示六域均胜。

本组冻结后，结果差也不得中途改数据/指标/结构来追胜。若修正结果不支持优势，记录差距并停止把该优势当主结论；是否继续既有未释放消融由组末证据决定，不自动花满27项预算。六数据集和额外官方可运行框架的缺口仍是未完成目标，本组期间不新增数据集、不重做15篇阅读。既有官方代码/权重缺失不是改写内部head为官方复现的理由。

## 证据及读取回执

直接读取：E:\AGENTS.md；项目AGENTS.md、ARIS_LOCK.md、docs/aris/WORKFLOW.md、docs/CVPR_RESUME_20261104.md、.aris/runs/temporac-cvpr-20261104.json；current_status.json的continuation_20261111；resume-20261111/astra/{SCIENTIFIC_AUDIT,KNOWN_SOURCE_TRAINING_CONTRACT,KNOWN_SOURCE_EXPERIMENT_TRACKER}.md；literature/COVERAGE_15_20261111.json；repnet官方closure及RUN_CONTRACT。旧Nov4状态只作历史，当前事实以Nov11 continuation/暂停回执及新的现场回执为准。

固定ARIS源码HEAD实际核对为 f76ec8d6b77792e2242cca049e08d0cbb7d54e19。三项完整SKILL及安装副本SHA均匹配：experiment-plan=61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a；novelty-check=65e1684942c2de5f9791f6e2b1025b320a8a4d7b41032a1991b73bb91719082f；result-to-claim=2e67d8bc7eb178aba18f3a3b646b59d3ca25836a42154acabe35dfa705f9f0ec。完整读取适用支持文件：output-versioning、output-manifest、output-language、reviewer-backends、review-tracing、evidence-precheck、experiment-integrity、integration-contract、reviewer-routing。继承已存在README/install全文回执，不再安装/重读全库。

本任务是继承已有结论的有界设计决策；没有重开novelty搜索或宣称完成新的result-to-claim流水线；没有调用新reviewer、没有运行evidence_check/实验计算，没有服务器变更、共享status/论文/Git变更。本文件为完整决策响应，可供父任务保存原始委派trace。执行启动验收须由执行器的真实回执补齐，不能用本文件自证完成。

用户恢复授权的父任务回执：E:\ICASSP_TempoRAC\aris-work\resume-20261112\resume_authorization.json；父任务给出的SHA-256为8c564c92701879d8edf9ad6487c080176a74609f3199c7855d032d2bc59c67db。最终调度修订为GPU0 s0→s2、GPU1 RepNet成功→s1；训练总GPU-hours预算不变，墙钟时间取决于实际RepNet耗时，不据原估时缩短任何任务。


最终启动前调度修订：父任务确认前版SHA caa7ee099ea10e3081723c29591c4dc30e83332c52973259e1dd331dcf92f2c8 已归档。本次仅修改资源先后条件，无科学/配置/矩阵变化；未发现与冻结三seed范围的冲突。

