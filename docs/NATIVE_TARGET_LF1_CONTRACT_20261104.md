# TempoRAC 原生目标 NT1 与响应损失 LF1（2026-11-04；四页子契约）

**决定：沿用 NB1 和现有 v15 的分支次序，移植为原生全长目标；响应候选 LF1 仅优化 class-balanced pulse BCE。** 不发明 v16、不新增 teacher training、不重启 TempoRAC。请求角色 `gpt-6-astra / xhigh`；实际 backend 未独立 attested，OpenAI 同族判断为 **provisional**。冻结的是待实现接口，不是 target quality/coverage 已通过。本文覆盖 NB1 的目标占位接口与旧 F14 响应损失候选；不改旧文件或撤销历史失败。

| 可核对验收项 | 本次状态 / 证据 |
|---|---|
| 保留方法、准确分支与依赖 | 设计通过；第1页及源码哈希回执；无模型/GPU/服务器操作 |
| 原生 N、有效监督区、零运动与完整人口 | 设计通过；第2页；全977/2364实测仍 pending |
| <3项优化目标、明确对照与归约 | LF1设计通过；第3页；实现/训练/贡献未验证 |
| target单位、真实覆盖、几何baseline | 未达标；第4页给出完整诊断及开跑门槛，不把证书等同真实周期 |

## 第1页 / 4：来源、冻结分支与可用性

依据 `aris-work/current_status.json`（10:25:35 UTC快照，10:31接续）而非旧pilot：HEAD `cb2fe0f55774e5d2d819ea8ef0070f0c383f813c`；977视频、2364身份、1758034人帧。原生提取wrapper1567147/child1567156和decode1543699仍running；不据旧快照断言当前完成量。extractor SHA `d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1` 不动。NB1设计 SHA `f0622e664ed6ed53f123cfc7f7151a71f04b2d15f347e84114e548ddbaa9487a` 原样继承。

**NT1每个NB1可执行run按以下固定顺序尝试，第一份完整分支证书成功即返回；各run独立，失败run保留：**

| 次序 / branch_id | 沿用实际代码 | target因果来源及认证 |
|---|---|---|
| 1 `v12_geometry` | `full_chain_landmarks_candidate_v1` → `multi_cycle_dtw_template_candidate_v1` → geometry phase closure | v6证书；natural reconstruction Huber的0.8分位≤0.04；纯几何 |
| 2 `v9_hybrid` | `stable_pair_integer_landmarks_v5` → `periodic_template_candidate_v1` → closure | v4证书，每traversal Huber≤0.02；最多3 landmarks/2 traversals；需已有兼容phase证据 |
| 3 `v15_geometry` | `full_chain_landmarks_candidate_v2` →同一multi-cycle DTW→geometry closure | v7证书，同v6 reconstruction规则；纯几何 |

这就是 `v15→v13→v12/失败后v9→失败后v15_landmarks`。v12/v15_landmarks虽调用teacher，三个输出全被丢弃：NT1删除这两个无因果作用的调用，`teacher_sha256=null`。v9把learned phase送入closure；但传入geometry后，选中区间内相位由几何弧长覆盖，learned phase仅留在区间外且仍受全数组数值/归一化检查。保留其真实hybrid来源，不宣称已证明learned phase改善单位；本轮不把v9悄悄改造成新的纯几何stable-pair。

**可用性冻结。** 首次全量build配置显式 `hybrid_mode=unavailable`，除非开build前已有真实冻结的兼容phase artifact：绑定同一NB1 identity/input/run/215-channel与static定义、完整N坐标、有限单位phase、实际checkpoint字节SHA、生成代码/config SHA及既有选择来源。旧320 phase不得猜映射或补齐。本轮不读取/加载模型、不新训teacher。unavailable按第2顺位写日志后进入第3，不记为certificate failure；坏hash/模式不符是完整性错误，停build，不能按结果静默跳过。若以后接入已有phase，另冻availability manifest并重做完整诊断；不能对dev表现挑identity或启停hybrid。

所有数值、tie-breaking、lag候选、origin/拓扑/collision/三tau一致性沿用对应源码：lag 6…160，return RMS .075，origin RMS .05，stable-pair Huber .025，至少6 edges/5正increments/max弧长份额<.25；tau=1e−8,1e−7,1e−6。**lag单位明确为native rows**，随N取上限min(160,(run_samples−1)//2)，不按时长调参；这不是旧320采样的物理范围等价，也不承诺PTS重参数化不变。参数是否覆盖自然动作是诊断问题，不用dev真值寻找新阈值。最大合法子序列可跳过中间return，把多个真实重复合成一次闭合；one winding因此绝非one true repetition的证明。

## 第2页 / 4：全长目标、数学定义与失败语义

API `produce_native_target(identity: NativeIdentityV1, policy, optional_phase_artifact=None) -> NativeTargetNT1`，不接收annotation文件、count/period/density/boundary或评估结果。NT1外层为 `temporac.native-target.nt1`，独立于NB1尚在实现的最小 `temporac.native-target.v1` loader；learner通过显式adapter读取数值，不能借旧FeatureRecord/teacher receipt填假SHA。NT1 JSON绑定input/raw/bridge/producer/config/availability SHA、opaque identity与slot、N、状态、所有run attempt/reason、各traversal branch_id及teacher_sha（几何为null）；NPZ为 `target_valid[E] bool,pulse[E] uint8,chi[E] float64,traversal_bounds[K,2] int64,seams[K] float64,run_id[K],branch_id[K]`，E=N−1。完整人口manifest恰2364行，失败也有全E零payload和false mask、K=0；零占位不是负监督。identity来源可为geometry/hybrid/mixed/rejected，绝不把mixed整条标纯几何。

run边界NB1为边域[a,b)，对应samples[a,b]。算法在局部samples `[a:b+1]`、单run `[0,b−a+1)` 上工作，再加a回填；不重新归一化身份尺度、不合并run、不插值PTS或缺帧。215-channel首149列及mask沿用geometry66+direction66+confidence17；mask为coordinate66+repeat(direction_element,2)66+joint17。算法的129/513 grid、证书128中点只是几何内部数值计算，不能登记为真实新增观测。

对成功分支的有序landmarks，traversal为 `[l,r)` 源边，完整samples l…r；K≥2/run，禁止截断周期补齐。沿用 `close_phase_on_landmarks` 的geometry弧长闭合（含既有1e−12数值项），但先要求每个候选traversal的每条**masked F4弧长严格>0**、≥32共同坐标、全NB1 response support，且所有实际插值分母严格正。NB1可观察/响应的精确静止边仍保留；它不能被扰动为运动，在该branch严格证书中导致拒绝，不额外把它变为negative或把两侧拼接。保留原分支整run证书成败，不从失败证书中捞出几条cycle过关。若失败也不要将其余有效run清空；这是NT1明确的逐run扩展。

令归一化canonical phase为u，`d_e=atan2(cross(u_e,u_e+1),dot(u_e,u_e+1))/(2π)`。每traversal沿用证书要求：d≥−1e−12、d<.25、dense winding=1±1e−6、≥5正increments及phase classes、observed gap<.25、dense occupancy/collision/重建与三tau结果一致。定义 `chi_e=max(d_e,0)/sum_[l,r)max(d,0)`；有限非负且abs(sum chi−1)≤1e−12（float64数值校验，不是数据选参）。现 `_seam_ledger` 唯一seam就是l，故 `pulse_l=1`、该traversal其他边为0，终端r不重复记pulse。`target_valid` **恰为**成功traversal并集；prefix/suffix、拒绝run、缺口、partial cycles、其他未认证时间均false且pulse=chi=0。

`m=response_edge_support & target_valid`；合法完整traversal不能靠交集剪掉缺失正边后继续训练：任一response支持损失先使整个traversal无资格并记原因。推断只用NB1 response support，**不能用target_valid限制模型预测区域**。pulse/landmark/traversal是伪事件单位；响应component、伪traversal均未获PeriodAP合法周期资格，不能据此偷偷输出官方periods。

## 第3页 / 4：LF1 单一BCE、匹配对照、单位平衡

**显式删改：** 新 `objective_lf1.py` 沿用旧BCE的clamp `[1e−7,1−1e−7]`，同时删除positive margin和negative valley；不加consistency、重建、phase或regularizer优化项。DTW/证书Huber只是离线资格规则，无梯度，不是训练loss。F11旧teacher有4个优化项，本候选不启动F11；若将来训练teacher须重新说明，不能计作LF1只有一项。

对traversal与一个读出x，p为binary pulse，`w=m*chi`，分支k再乘detached NOLA平均responsibility π_k（matched capacity-control按旧定义省略π）：

`W+=Σwp >0, W−=Σw(1−p)>0; B(x,p,w)=−0.5[Σwp log(clamp x)/W+ + Σw(1−p) log(1−clamp x)/W−]`。

`L_identity=mean_traversal[(1/3)Σ_k B(r_k,p,m*chi*π_k) + B(R,p,m*chi)]`。

这是**一种BCE优化准则，显式施加于三branch与一个fused读出**；两个class归一化、head/traversal/source归约不是新增margin项，四次BCE计算不隐瞒。shared encoder/专家/概率融合/NOLA保持可导；cue、π、chi、pulse、mask和分母D detached。所有head的W+/W−均须有限正；不补ε、不换分母、不暗中放宽router。目标本身缺正负质量则标training-ineligible；固定soft责任的数值下溢导致W=0是实现故障，不能把该identity悄悄删除。

**具体已发现的WIP问题：** 当前NB1 `native_supervision` 写成 `sum(chi[m])` 与 `sum(1−chi[m])`，没有pulse，须改为上述W+/W−；否则唯一正pulse无支持仍可误通过。同时NT1验证binary p、完整域/有序不重叠bounds、mask恰为bounds并集、每traversal恰一start pulse、chi归一及input SHA，不只检查[0,1]。

沿用旧source平衡语义：一个自然source component等权进入确定性cycle，内部eligible identities等权确定性轮转；每身份所有成功run的traversals合并后等权平均，不能按run/窗口数多投票。全部无资格component仍留人口/coverage分母；四组train别名保留并绑定同源component，不能制造额外source票。X0使用既有analytic pulse/chi/责任及七block/source平均，不依赖新增teacher。若使用X0+natural，step仍0.5/0.5，一族为空则不能静默变X0-only。仅X0 tune的同LF1目标选checkpoint、tie较早step；自然GT不进选参。完整X0与native sampler/schedule须在启动前给出实际manifest。

**对照冻结：** 首seed主比较为NT1+LF1 canonical与同target/初始化seed/预算/selection的capacity-matched control，避免loss变化混入路由差异。额外 `LF0_legacy3` 是同NT1、同NB1、同canonical图下 `B+0.5M+ +0.5M−`，诚实标3项；用于检验简化代价，预算若未覆盖则pending，不能默认新开job。旧 `objective.py` 原样保存；没有恢复旧失败路线或把旧pilot数值当matched结果。继承三seed候选 `20260815/20260816/20260817`，完整步数/epoch、GPU时限/内存、checkpoint schedule及LF0是否纳入预算仍须单独冻结，旧27-job/93 A6000h不能冒充4090完整数据预算。

## 第4页 / 4：一次实现、完整诊断、训练与声明分门槛

**交Sol一个交付：** 新 `native_target_nt1.py` + `objective_lf1.py` + `scripts/build_temporac_native_targets.py`（plan/build/validate/diagnose-inputs，仅CPU，不启动训练）。复用动态N数值内核与严格证书；`CertificateTrack`自身已支持动态T，几何分支可用provenance=None执行数值证书，再由NT1外层绑定真实producer provenance；不调用旧natural FeatureRecord/伪teacher receipt。写全run attempt与失败账本、原子NPZ+JSON、内容hash resume；配置/input/code变化拒绝复用。独立eval诊断程序才允许加载完整train/dev GT；不能与producer同入口或把其产物回传调阈值。

**必要CPU测试（人工fixture，无自然标签/模型）：** ① N216/N900、多run及首末边映射，候选branch顺序/缺hybrid/混合来源，第三branch仅前两不成功时启用；② exact zero/零分母、缺PTS/≥5 gap、短run、裁掉首/尾cycle均不能产生跨gap traversal或fake negative；③ 单run同数值输入对原v12/v15数值内核逐字段parity，v9用确定的mock phase验证区间内geometry覆盖与区间外数值检查；④ 每traversal一pulse、sum chi=1、mask并集、整数端点N−1合法但不伪造N帧sample；⑤ W+缺失但chi sum>0、W−缺失、非binary pulse、SHA错、bounds重叠拒绝；⑥ 手算BCE及梯度、unsupported边零梯度、head/fused均有梯度而cue无梯度、拆同一身份为多窗口/run不改变单位权重；LF0−LF1恰为两margin贡献。测试通过仅为工程证据。

**完成cache后一次完整977/2364 label-free build/census；随后隔离evaluator在train794/dev183完整人口各报告：**

- 每身份原始attempt/选中branch、available/rejected/numeric/input失败、N、K、观测/response/target/training边数；有效PTS时用ΣΔt报target时长/可观测时长，另报未知秒数及source-frame比例，不能把unknown当0。按split、source component、branch、clock mode和失败原因汇总分布；既报conditional也报全人口分母。
- 冻结纯几何baseline `G-NT1` 独立执行第1→第3branch、跳过第2，`count=Σ成功traversals=Σpulse`；全失败输出count0并abstain标志，partial不按覆盖率放大。若启用hybrid，再单列 `T-NT1 direct` 直接目标计数，不能叫纯geometry。所有身份保留；count error/有符号bias、绝对/相对误差、OBO、count ratio及source-cluster区间按明确拟议指标报告，未认证official不冒名。
- 单位诊断：每个GT period内伪pulse数0/1/≥2、period fully-supported比例、部分重叠支持比例、落GT周期union外pulse比例；完整周期与partial cycle分别列，保留GT count与period条数不一致。明确period边界口径后再计算；endpoint未知则该诊断unavailable，不猜边界。用这些显示half/double/missed事件，不把每target一pulse的构造事实当真实unit通过。
- 同报纯几何全人口误差和仅certified子群误差；observed→response→target→training支持漏斗、identity/time/event覆盖联合表，不能以identity接受率替代event覆盖。保留旧80%未过、X0-only NMAE .9915197/OBO0与所有不利counter结果；本轮未实际运行这些诊断。

**训练门槛与结论门槛分开。** 完整固定train794/dev183的已授权预注册训练，在全raw/decode/NB1与NT1哈希/人口/接口有效、上述完整诊断已交付、source component/sampler、LF1/三seed/完整预算/selection与checkpoint策略冻结后可以推进；不得换split、截小样本或把unresolved unit failure掩作通过。目标质量数值容忍界目前未给定：这是“质量验收通过”真实未决项，不由本设计拍数值。完整诊断后允许作一次有理由、带噪声/覆盖披露的“继续完整方法检验”决定；若存在跨gap、虚构负例、错误单位定义等语义错误，先修复再训练。仅误差不够好不触发无限泛审，也不授权用dev GT改上述阈值。

release正式checksum、历史暴露与official scorer parity仍candidate/pending，应限制official/SOTA/独立测试声明，**不是永久阻止上述完整开发实验的理由**。自然标签仍仅evaluator-only；A180完整保留，B78仅候选且9 source簇，不能改称独立test或抛掉A。当前无test标签访问授权；正式独立评估另需协议/指标/合法输出资格。PeriodAP未定只阻其period结论，不阻已命名且冻结的count开发实验。通过训练、低loss、接口证书都不等于真实target质量或论文贡献通过。

读回执另见同目录 `native_target_loss_contract_20261104.read_receipt.json`。完整读pin `f76ec8d6b77792e2242cca049e08d0cbb7d54e19` 的experiment-plan及mandatory output-versioning/manifest/language/composition；安装与源SKILL SHA均 `61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a`。本次按指定四页子契约交付，不覆盖原proposal/experiment-plan；无自然GT读取、模型运行、安装、服务器信号、commit/push或extractor修改。

