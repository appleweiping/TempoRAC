# TempoRAC 完整三种子组末判断与下一冻结组决策

**决定：继续当前冻结 no-LLM 方法，释放剩余24个规格，具体为21次新的完整训练和3次既有 single-UCF 结果复用。两张空闲GPU可立即进入已冻结队列；官方基线准备并行，满足预定条件后在完整实验边界交接一张卡。** 不新增组件、超参、数据或种子，不因某域或某种子得分改变本组范围。当前结果支持有界继续补齐证据，**不支持六域SOTA或“每个组件都有效”已经达成**。

本决定依据最新用户继续并完成目标的授权，在原三种子组结束后作出。旧“其余24项不自动释放”是上一组的调度边界，本文件作出明确的组末释放，不把旧文档当成永远需要用户再次批准的限制。没有重新研究选题、重新安装ARIS、重复查新或启动额外语义审查。

机器契约为 `RUN_CONTRACT_20261112_134530.json`，固定名 `RUN_CONTRACT.json`；SHA256均为 `dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167`。该契约在最终定量判断前已锁定，以下结论不改变它。

## 验收表

| 用户/项目要求 | 本次判定 | 实际证据与边界 |
|---|---|---|
| 复用原研究、有效缓存、29项旧实验和15条文献记录 | 通过（本次范围） | 原资产保留；只读取继承分析；24个规格原已冻结；single-UCF3不为目录或缺失旧回执而重训。15条是框架/版本记录，折叠MultiCounter+为14家族，部分仅局部/摘要可得，不改称15篇完整顶会正文。 |
| 修正full3正式完成 | 通过（执行与数值存在） | 每seed50连续epoch、有限loss、身份与validation选择校验、12份全人口评测、8280条seed×video记录；`group-completion/group_summary.json`、`COMPLETION_RECEIPT.json`及`evidence/`。 |
| 仅排除已知训练曝光、保留评价人口 | 通过已声明的已知来源修正；整体独立性仅部分 | manifest `96c068ee...21fb2`，RC756/CX4222/UCF331；评价RC152/CX2403/UCF105/QUVA100完整保留；旧test-informed方法选择及RC标签冲突仍在。 |
| 组件真实贡献，配对三seed判断 | 未达标，已明确释放补齐 | 新full3作为唯一修正协议基准；五移除×三seed和三个单域×三seed。旧受曝光影响的消融不得与新full混配。 |
| 外部整方法性能比较 | 部分 | 两个已运行官方权重框架：ESCounts、RepNet；三个共同评价人口。ESCounts的UCF/QUVA是RC权重迁移；Countix本轮没有这两个完整同人口结果。 |
| 六数据集×七有说服力框架，至少六域主性能SOTA | 未达标 | 现在四个评价数据集，非六个；当前最强文献对照也存在明确未胜项。文献行不等于我们的复现，不同backbone或重复权重行不等于新增框架。 |
| 至少三个紧密耦合组件；少于三项loss | 组件贡献待实测；loss通过 | 多率/双深度表示、shift训练与两视图推断、带token的平衡联合训练构成候选耦合机制；只有一项实际生效的density MSE。inactive配置字段不算实际分支或loss。 |
| observation、超参曲线/稳定性、完整指标与效率 | 部分/未达标 | 三seed和完整计数指标现有；机制观察、最终框架曲线及含特征提取/两次decoder的效率仍欠缺。不能用训练完成代替这些要求。 |
| 最终稿与质量保证 | 未达标，本次不写最终稿 | 旧稿为明确标注的历史结果；本次没有新论文/引用/全页渲染验收。当前会议与已注册论文不重开争论，不自动投稿。 |

## 三种子组实际支持什么

固定主指标为正计数视频上的 `mean(abs(np.round(pred)-g)/(g+0.1))`，越低越好；每seed先将原视图和shift视图的连续计数作算术平均，再用NumPy舍入。表中是三个独立拟合seed的均值±sample SD，不是seed ensemble，也不是置信区间。

| 评价人口 | 全部/正计数N | 主NMAE，mean±SD | OBO，mean±SD | 同预测/g NMAE，mean±SD |
|---|---:|---:|---:|---:|
| RepCount-A | 152/151 | 0.141996913±0.016629308 | 0.693156733±0.061176187 | 0.145644084±0.016839214 |
| Countix | 2403/2403 | 0.243386496±0.007086311 | 0.695242059±0.004105604 | 0.250028290±0.007553787 |
| UCFRep | 105/105 | 0.194626165±0.012612571 | 0.704761905±0.025197632 | 0.199745938±0.012959866 |
| QUVA迁移 | 100/100 | 0.190626881±0.008050887 | 0.530000000±0.060827625 | 0.192830209±0.008127374 |

RC三seed为0.125608148、0.158856754、0.141525836。必须报告整组，不能以s0的0.1256代表最终模型或选择最好的seed。每个目标的checkpoint仍由自己的original-view validation选择；RC的best epoch为34/13/47，CX为22/15/22，UCF为45/40/31；QUVA沿用对应RC checkpoint/token。因此是每seed一条共享训练轨迹、多个目标checkpoint，不是所有域共用一个最终权重。

原始完成回执中s0/s1/s2分别于08:48:26/09:58:58/10:31:44 UTC完成；13:38 UTC是父任务较晚的现场观察，不是s2结束时间。13:43:36 UTC组末分析确认各50epoch、无存活trainer/worker、GPU compute-apps为空。发布源码49b30a3和训练源码489dc77的职责分开；`source_sync.json`旧的completed=[0]不是新组完成的依据，父任务需以真实回执更新共享状态。

以下反例不能省略：

- RC唯一零计数视频`train143`的两视图舍入输出为5/4/4，三个seed均是假阳性。新模型全152人口NMAE为0.426150442±0.025387874，**高于**ESCounts官方权重的0.279472368。正计数主指标优势不等于对零事件视频同样有效；保留全人口与零计数列，不以主指标冻结为由隐去它。
- 现有full3的original-only→固定two-view主NMAE为RC 0.161811696→0.141996913、CX 0.264342636→0.243386496、UCF 0.190705453→0.194626165、QUVA 0.207705789→0.190626881。UCF的OBO也由0.752380952变为0.704761905。它们只是相同拟合模型的已保存视图诊断，已经反驳“所有域都受益”的强表述；不能据此在UCF测试集上改成original-only。训练shift的贡献必须等对应wo_shift三seed。
- 新训练修正不消除早期test-informed方法选择、不裁定RC冲突标签、不保证所有未知近重复或预训练曝光已清除。原29项与新协议不混池，不用旧五seed均值充当新消融full基准。

## 官方发布权重与文献的分层判断

下表每格为同一冻结计数口径的NMAE/OBO。ESCounts和RepNet均为**作者发布权重由我们评测**；没有重新训练这些基线，也没有相同训练预算保证。

| 方法/发布身份 | RC152（主指标151正计数） | UCF105 | QUVA100 | 训练/设置说明 |
|---|---:|---:|---:|---|
| TempoRAC修正full3均值 | 0.141997/0.693157 | 0.194626/0.704762 | 0.190627/0.530000 | RC/CX/UCF联合训练；目标各自validation checkpoint |
| ESCounts已获RC checkpoint | 0.215098/0.556291 | 0.341636/0.466667 | 0.299129/0.300000 | UCF和QUVA都是RC权重迁移，不是该方法各目标最强训练设置 |
| RepNet updated2025 ckpt-70 | 0.287127/0.410596 | 0.311727/0.571429 | 0.240458/0.520000 | 357完整输入；不冒称2020原论文checkpoint |

精确证据：`resume-20261111/server/baseline_source_audit.json`及其绑定原预测；`resume-20261112/baselines/runtime-v4/metrics.json`、完整357回执和trace。当前full每个seed在这三个人口的正计数NMAE都低于这两个确切发布权重，但**未做该比较的配对不确定性分析**；不称统计显著、不称同资源优越、不称整方法SOTA。

保留且正视更强的文献行：RC的PAMS报告0.129，比本组0.141997更低，虽然它是pose、自监督及不同公开协议，不能删掉它制造全模态最好。RC的CountLLM报告0.162；本方法同预测/g为0.145644，仍数值较低，但同分母不能解决人口、额外训练、checkpoint与选择差异。UCF的CountLLM0.116、Sight&Sound sight-only0.143、Context0.147均低于本方法对应0.194626（/g为0.199746）。Countix的CountLLM0.215也低于本方法0.243386（/g为0.250028）；PAMS还报告0.098，但其Countix子集/口径缺口必须逐行标注，不能一边缺口未知一边排除强项。QUVA只击败目前两个确切运行权重，不能推出击败全部已发表方法。

这些是继承15条ledger及已读primary表格的有标签比较，不是本次重新查新的全球最优排名。最稳妥的当前措辞是：**“在保留原评价人口、已知来源修正训练及固定三seed协议下，本方法在三个共同人口上取得低于两个确切官方发布权重的正计数NMAE；与更强的文献方法仍有未胜项和协议差异。”**

用户“主比较不报非SOTA域”的偏好仍未被满足或取消。当前不能把其解释成删掉UCF/Countix、隐藏PAMS或宣布六域均胜。最终主表范围需要真实证据支撑，并明确可找到的全域结果和后选择历史；本次不把该排版/范围冲突伪装成已经解决。

## 下一组一次性冻结

原27规格SHA `5e6750b0ff6a894be85c171e5ae31c150e9cb5288a3cb9ab4df8f0f5984babe4`保持不变。新训练只使用manifest `96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2`、原缓存、原源语义、seed0/1/2及原每臂完整cfg。新launcher可以适配多臂/50或80epoch、单目标checkpoint和边界hold；只做一次必要Sol工程核对，不重开泛审或新科学设计。

| 冻结臂 | seed与epoch | 新训练数 | 完整评价与解释 |
|---|---|---:|---|
| wo_pyramid | 0/1/2，各50 | 3 | 联合训练；RC152/CX2403/UCF105；仅保留s=4，是移除多率表示，参数/计算也变化，不是等参数“唯一原因”证明 |
| wo_shift | 0/1/2，各50 | 3 | 相同完整三域；训练shift_prob=0，shift_eval仍true；原视图及两视图构成预定2×2诊断 |
| wo_depth | 0/1/2，各50 | 3 | 相同三域；保留block12、去掉block8；不误写成删除深层block12 |
| wo_token | 0/1/2，各50 | 3 | 相同三域；不含dataset token，其他冻结项保留 |
| wo_balance | 0/1/2，各80 | 3 | 相同三域；5309单次条目/epoch，完整原80epoch |
| wo_joint_rc | 0/1/2，各80 | 3 | RC756/131/152；validation rc_valid选择best.pt，只评完整rc_test |
| wo_joint_cx | 0/1/2，各80 | 3 | CX4222/1374/2403；validation cx_val选择best.pt，只评完整cx_test |
| wo_joint_ucf | 既有0/1/2，各80 | 0，复用3 | UCF331/90/105；best.pt和原完整预测；不重新运行以消除旧paperwork缺口 |

联合臂目标映射为rc_valid→best_rc_valid.pt→rc_test、cx_val→best_cx_val.pt→cx_test、ucf_dev→best_ucf_dev.pt→ucf_test。主推断均为冻结两视图均值，保留所有原/shift输出以及OBZ、RMSE、/g、全人口、零计数和连续输出诊断。这个组**不新增QUVA消融评测**；完整full3的QUVA100已有且继续保留。

single-UCF复用依据：旧/新manifest的331/90/105有序UCF记录完全一致；三个cfg、seed、feature-root、输出hash均与已冻结规格及Nov11 provenance一致；80连续epoch完整；旧/新data/models/metrics/eval_saved/launch字节一致，训练helper AST一致，主函数改动为持久化/恢复/日志。没有观察到配置、人群或计算语义不匹配。缺失的是历史run-start完整source/environment/cache绑定，不能事后虚构；这限制可复现溯源强度，不是无意义重训的充分理由。保留原目录和资格说明，与新full的配对解释明确标注历史复用。

**三个single-dataset臂均同时取消联合人口、token及balanced repeat，并使用原80epoch预算。** 因而它们检验“单域训练方案 vs 完整联合方案”的组合差异，不隔离联合人口本身。wo_token/wo_balance提供各自更局部的对照，但现有矩阵不能完整识别高阶交互或证明synergy。wo_balance名义optimizer更新为53,120，full为53,600（接近而不完全相同）；epoch-based warmup/decay也不同。照实称采样/训练日程方案对照，不能称其他训练因素严格恒定。保持这些已冻结设计，避免为补一个因果措辞临时扩矩阵。

有限队列为：GPU0依次wo_pyramid s0/s1/s2→wo_depth s0/s1/s2→wo_balance s0/s1/s2→wo_joint_rc s0/s1/s2；GPU1依次wo_shift s0/s1/s2→wo_token s0/s1/s2→wo_joint_cx s0/s1/s2。最多两项GPU任务、每卡一项，启动前确认资源归属，不打断STaR-Body或其他任务。首批是pyramid_s0和shift_s0，随后按既定顺序推进，不根据测试分数决定后继。

新训练共21次、1320epoch；按冻结batch与完整人口共有952,080个名义optimizer更新，无nonfinite跳步时成立。已完成full的日志训练耗时分别5715.4/5656.5/5645.5秒，约1.57–1.59小时/seed。按更新数粗略折算约28 GPU-hours训练，具体多率/视频长度/validation评测导致偏差；建议保留约28–36 GPU-hours的工作预测，不把它当缩短完整epoch的硬限。原“全27约46 GPU-hours”也只是估算。两卡约14–20小时加基线交接和实际IO开销，不保证精确完工时刻；执行器用完成实验更新预测，不新增实验来测时。

父任务可以显式设置hold-after-current：**当前臂完整训练、全部冻结评测和完成回执写完后**worker退出，记录未启动余项。只有继承的官方baseline在独立固定的源码/权重或完整训练、完整输入、native与harmonized评分契约准备就绪时，手动把一条lane交给该baseline；另一lane继续。随后显式恢复相同余项。不得杀/迁移健康作业，不在epoch中途为腾卡打断，不用自动定时器、不根据分数hold，不因这种资源交接扩大21臂。

## 如何判断组件，以及随后才可做什么

全组关闭后只对seed0/1/2匹配比较。先平均每视频的三seed误差，再按既定source/video cluster配对重采样；另报每seed delta及sample SD。五移除×三数据集加三个单域对照构成原18项主NMAE家族，沿用nominal及Bonferroni18口径。现有10,000次bootstrap的调整尾部只有约14个抽样，须保留Monte Carlo精度限制；这些区间条件于已拟合模型，并非训练随机性总体置信界。没有显著差异不能证明等效、有效或无效。

当前只接受“组件贡献待证”，不凭旧消融或新full一个seed撤组件。下一次组末将每组件标为：配对证据支持、效应不稳定/不确定、或完整指定比较中不支持；同时展示OBO等相反方向。弱depth也保留这次完整三seed移除，不能提前取消以保住三组件叙事。原视图/两视图的2×2来自现有与新wo_shift保存输出，不再训练第五种设定。

若配对证据显示某组件无稳定贡献，随后先用**已冻结validation人口**诊断误差：各域/原有tempo与长度分层、零计数假阳性、crop/边界敏感性、原/shift误差相关性及validation选择稳定性。检验疑点来自val，所有test仍保留为已观察数据；不在test上选新的视图、域、loss权重或阈值。当前组中不启动改进模型。

最终框架超参曲线仍必要：继承曾提出的p_shift=1三seed及sigma=0.125/0.5六seed最多9项方案可在本组后再判定是否必要，中心p_shift=.5/sigma=.25与p_shift=0复用；**本文件不自动释放这9项**。只有明确validation诊断和既有配对组件证据支持时，才将一项具体改进或必要稳定性研究一次冻结；不得复活MultiRep/geometry/tempogram/失败LLM路线，不能用一串新搜索拖延现有21项和外部比较。新方法如果被采用，必须与当前版本分开记录，并承担新的完整配对验证，不悄悄替换现有证据。

## 外部比较与六域缺口的实际下一步

固定外部池继续是CountLLM、ESCounts、TransRAC、RepNet、Context、Sight&Sound、MFL，并保留强相关PAMS等。MFL是WACV、ESCounts是ACCV、PAMS是CVPR Findings；不能把它们改写为七篇CVPR/ICCV/ECCV main方法。文献论文研究规模供组织参考，不代替用户的更高目标。退休internal-style head不计官方框架。

读取本次并行的`next-baselines/AVAILABILITY.md`及`summary.json`后，当前**没有第三个已就绪发布权重**。优先推进TransRAC的确切官方task权重/原生输入环境闭环；若权重仍不可得，采用官方完整200epoch训练的独立契约，而不是把内部head当复现。源码200epoch/global-batch32/lr8e-6及原raw/NPZ输入归一化差异必须落实，原4GPU绑定只能作资源适配。完整fine-cycle输入和legacy环境未就绪，故这份组末决定不假装它已能运行。

RACnet是备选原生整方法路线，ICIP2024不是新的顶会资格替身。作者链接的feature repo已固定至`2196a15bb50de973f245cff318acd1cbe997bdc8`；RepCount60分片125,484,960,235字节、UCF3,144,832,399字节，说明有具体可获取输入，不是任务权重已到位。实际split/ROI身份、fine-cycle CSV、环境和完整200epoch训练契约尚未闭环；不能用ST-MAE缓存代替官方Swin特征。无需等这些准备完成再训练已有21臂，也不无条件下载125.5GB来展示“进度”。CountLLM无验证官方release、MFL/PAMS等资产缺口继续显式保留；可行训练/资产准备由routine执行器推进，不再重复全库文献检索或泛审。

六数据集缺口仍沿原合同的YTSegments/ActCount官方完整合法源可得性路径推进。先证明全人口、任务相关性、来源关系并冻结评价，不按预期能否赢选择数据。不能将MultiRep衍生集、速度扰动、子集或backbone重复行凑成六域。第三方发布权重的历史训练暴露仍需区分；从零训练的clean-protocol官方基线与released-checkpoint评测应分栏。

## 结果到主张字段、身份与证据回执

- `claim_supported: partial`。完整方法当前在三个共同人口上取得低于两个确切官方发布权重的正计数NMAE；所声称的机制贡献和六域SOTA尚不成立。
- `what_results_support`：已知来源修正后的完整full3可重复计分结果；四个固定完整评价人口；固定推断下的确切权重比较；一项实际训练loss；接续配对消融具有明确科学价值。
- `what_results_dont_support`：独立选定的test确认、所有来源泄漏已解决、全人口/零事件普遍优越、所有组件有效、全域view收益、统计显著或等效、同预算全方法优势、全球SOTA、六域×七框架完成、最终论文已验收。
- `missing_evidence`：剩余配对组件结果、更多真正官方整方法比较、至少两个合格独立数据集、observation/最终框架曲线/完整效率、最终科学与稿件审查。
- `suggested_claim_revision`：使用上文精确权重/人口/指标限定的整方法比较措辞，保留post-selection与训练资源差异；组件主张暂挂，不靠理论修辞补证据。
- `next_experiments_needed`：本文件唯一释放的21次完整训练+3次single-UCF复用；其余基线/数据/曲线分别在输入齐全后固定独立执行契约，不借此增加本组训练矩阵。
- `confidence`：对已核对执行状态和保存数值高；对因果组件归因及广泛泛化低/尚未建立；组末继续决策为medium confidence、same-family provisional。

本次是父任务明确请求的**一个新鲜native Astra/xhigh科学组末判断**。requested_model=gpt-6-astra，requested_effort=xhigh，actual_model=null，actual_effort=null，session=/root/group_end_astra_20261112，review_independence=same-family，acceptance_status=provisional。显式调用覆盖项目默认codex/Sol；没有声称发生ultra能力fallback，也没有把它冒充固定result-to-claim默认ultra深审或提交级完整保证。没有嵌套第二reviewer。

本地`EXPERIMENT_AUDIT.json`为2026-08-15旧v46/v63的FAIL，来源和范围与当前489dc77不同。原FAIL保留；`current_group_integrity_status=unavailable_for_fresh_full_integrity_audit`，不得复制旧FAIL来证明本组失败，也不得以本次执行存在性校验将其改成PASS。已知标签冲突和历史test选择作为当前真实限制继续生效。

固定ARIS HEAD实际核对为f76ec8d6b77792e2242cca049e08d0cbb7d54e19。完整读取experiment-plan/result-to-claim两SKILL及mandatory output-versioning、output-manifest、output-language、reviewer-backends、evidence-precheck、review-tracing、experiment-integrity、integration-contract、reviewer-routing；源/安装副本SHA分别为61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a及2e67d8bc7eb178aba18f3a3b646b59d3ca25836a42154acabe35dfa705f9f0ec。README/install全文回执继承；本次未重装、未重读整库。

组末`evidence_precheck.json`实际31/31 verified，只证明引用数值存在。文件hash已静态核对：group_summary=`a4f5349fa187d9511e381176dafc3ac63b7cd867802e1a49a9d33e7ed6c9e0ee`；claims=`d242fda65c1f3443d485935de33da789d660230c33e88088ae9dfe79461070a9`；precheck=`8e1fca860643a0f6636b6e4a59f7905f59929b07d99014f340f9bce0b6bfa7cb`。helper由安装manifest解析、固定源码SHA2edbd901d760d22e7e43a14e8e88ab0e089303f489055cd9b5e9d1f4dce50382在服务器执行。元回执的prepared文字早于实际结果；本判断依据真实precheck输出，不依据prepared状态宣称通过。

直接阅读了新组summary/完成回执、抽取raw source metrics、逐seed/view数值、旧baseline原result与harmonized source audit、RepNet metrics、single-UCF inventory/raw assessment、冻结27规格和当前科学/文献/训练契约。全部预测及77项小证据在父任务group-completion/evidence目录；archive SHA `c52b972142977e385439c394bba1c99e5659ec6988aff11303d751e6d72647f8`。没有在本地进行训练、推断、实验统计或计算验证；没有服务器写入、启动/停止job、改共享status、改论文或仓库。本文与冻结契约供父任务保留完整原始响应和实现回执；发布/运行事实不能由本文件自证。