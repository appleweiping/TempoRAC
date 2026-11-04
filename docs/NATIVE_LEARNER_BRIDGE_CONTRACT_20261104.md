# TempoRAC 原生 learner bridge 冻结 NB1（2026-11-04；三页）

**决定：新增 `temporac.native-learner.v1`，完整原生 N 行、分支持段执行；不转换成旧 320 格式。** 继续共享编码器、slow/medium/fast 三专家、detached 局部 NUDFT 软路由、正分母 NOLA、每身份一次解码。请求模型/推理档为 `gpt-6-astra / xhigh`；本会话后端身份未独立 attested，OpenAI 同族科学设计 **provisional**。这是可实施的接口冻结，不是完成缓存、训练或效果验收。

| 验收要求 | 本次结论与证据 |
|---|---|
| 不伪造观测、不挤压时间、不改变旧入口 | 设计通过：NB1 独立 schema；下述恒等帧映射与掩码；旧代码不改 |
| 全人口、真实缺失、PTS、输出位置可追溯 | 设计通过，运行验证待做：977 videos / 2364 identities / 695625 native frames，失败留账 |
| 可实施的推断/监督接口和 CPU 检查 | 设计通过：第2–3页；实现、全缓存覆盖仍 pending |
| 正式训练、少于3项 loss、主指标、PeriodAP | **未达标**：第3页列具体阻塞；不由接口通过代替科学通过 |

## 第1页：输入、支持与两个坐标

**证据优先级。** `aris-work/current_status.json` 优先于公开旧快照：最新三方源码 `cb2fe0f55774e5d2d819ea8ef0070f0c383f813c`；提取启动源码仍为 `832658815c7baed2ddf2f71d751ad1ab7f607f63`。10:25:35UTC时raw完成19/977、decode277/977且0 findings，均仍running。GPU1 wrapper1567147 / child1567156；extractor SHA256 `d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1`，健康作业及该文件不动。首件 `(2,216,17,3)`、L=D216、PTS216、time_base1/12288、430/432人帧返回、7310 observed joints；2人帧无支持，不能补造。本次仅看已有验证回执。

**勘误。** `src/pams/temporac/preprocess.py:26–30,182,192,355–357,463,471,573–577` 实际要求：score clip 后 **>0.20**，每帧 **≥8 joints 且至少1 hip**，边两端共同 geometry coordinates **≥32**，旧边长严格>0，内部无效帧拒绝，至少 **17 samples**，旧整数时钟差 **1…4**。旧摘要“17关节都有效”不准确；17是最少样本数。`types.py:150–178` 的320仅是旧七字段归档限制；`response.py:155–199` 的编码器已经接受任意T及多个run。

**独立入口。** trusted bridge 只从经验证 raw 投影出每身份 feature-only 数值记录；model loader 不接收 bbox、源annotation路径或count/period/density/boundary。旧 `FeatureRecord`、`feature_io`、`preprocess_identity`、v2 和其测试完全保留，禁止造一个假的320 `FeatureRecord`。NB1 原始字段固定为：opaque key/slot、`source_length=N`、`source_frame_index=i=0…N−1`、`xyconf_raw[N,17,3]`、`joint_observed[N,17]`、`frame_source_ok[N]`、整数PTS及present、rational time_base；状态/哈希/映射在sidecar，不进269通道。L≠D或raw失败保留记录并整身份 `unusable`；不猜帧对齐。

**支持规则。** 所有原生行保留；不插值、不外推、不重编号、不抽帧。`frame_source_ok=decoded_ok & bbox_valid & pose_returned`；`observed`是有限frontend返回，非可见性；`usable_joint = observed & frame_source_ok & (clip(score,0,1)>.20)`。沿用上述≥8/hip、全身份 root–shoulder median scale（下限1e−3）、≥32共同坐标。缺关节零数值只是带false mask的占位。有效边必须连接原始相邻帧 i,i+1；失效处断run，绝不跨过缺帧把两端接成邻居。每个run少于17 samples时保留但不作为可执行run，记 `short_run`；未支持帧/边不删除。

**唯一几何扩展明确冻结：** NB1允许有限、精确零长度的静止边作为观测/响应支持，长度保存0；方向零范数时direction mask=false。旧 `preprocess.py:471` 的 `length<=0` 拒绝不改；新入口区别于它。理由是静止确为观测，不能加微扰制造运动，也不能因静止抹掉真实输入。要求正弧长的旧target算法仍须拒绝不满足其分母的区间；NB1不替它制造正弧长。

**坐标。** 永远保留离散源帧 i；另设 clock mode：

- 有真实PTS：先对整数差运算，再转 `t=(PTS−anchor)*num/den`；`tau` 固定为该视频所有present、相邻源帧、正PTS差的中位数所对应秒数，`q=t/tau`。不逐run重缩放，不用FPS或best_effort代替PTS。缺PTS节点和非正差处断时间支持，重复/倒退PTS仍逐行留存；没有正差则 `clock_invalid`。
- 全视频没有真实PTS：显式 `source_frame_only`，q=i；不报告秒、Hz或真实不规则时钟结论。部分缺失不能降级整视频来掩盖缺口。
- 原生相邻边还须 `0<Δq<5`；这是旧1…4整数源帧规则的**有单位扩展**，不是把PTS ticks直接送旧gate。旧入口仍严格1…4；native小于1的正间隔允许，≥5断run，原因记明。

## 第2页：全长响应、NUDFT、输出与监督

**映射复用。** 在每个严格递增clock segment、每个joint上调用现有 `timebase_audit.build_timebase(source=q,output=q,source_observed=raw_observed,max_interpolation_gap=0)`；记录output_row→原始i、t/q、左右anchor/weights、support_kind、supported、gap_span。全无观察channel与不足2行segment显式记unsupported，不能强行调用后伪造anchor。模块可能给缺失行“interpolated/extrapolated”的候选权重，但NB1 **不调用其resample_values、不把候选值送模型**；只有exact observed能有观测支持，derived有效性另列。此模块只做sample-position映射，不能凭它定义最终视频边界。

**模型及窗口。** 输出仍为N行269通道、N−1边；沿用geometry/confidence/mask与lag4/2/1，lag不能穿越run。ragged逐身份处理，不为批处理补到320；模型的因果左零padding仅是原有run边界条件，不登记为观测。每run一次共享编码器/三个专家，缓存完整响应；局部窗口128 samples、stride32、末端右对齐，短run按现有窗口规则。窗口只切缓存，不截断训练序列或各自解码；继承相同参数数目与dilation4/2/1。专家卷积/lag仍以native rows为单位，不声称已变成连续时间网络。

**时间cue。** 新native clock wrapper接受float64 q，在各run内执行旧NUDFT/共享quadrature公式；不round q进整数API。固定48个period=5…128（单位tau或source frames）、f=1/period；可用PTS时Hz=f/tau。保留旧可靠度、三可靠窗口reference、soft gates和fallback规则，cue/router无梯度。额外固定clock-grid检查：窗口 `2*f_max*max(Δq)>=1` 时gamma=0、uniform gate并记理由，不更换频率网格来挑峰。它是保守频率支持检查，不是非均匀采样的无混叠证明。**NB1完全不降采样，所以无需新增降采样滤波器；原生采样本身仍可能混叠。** 旧320的主动均匀抽样不是“缺帧”，不能搬进NB1或复用其5…128的物理范围；以后任何降采样须新版本、先抗混叠再抽样并保存支持核，本次不实现。

**一次读出。** 所有run/windows回填同一身份N−1边数组，masked NOLA在支持边要求D≥1e−3，不支持边N=D=0。解码wrapper每身份只调用一次现有 `connected_components`（含无支持时的empty调用），阈值0.5不变；输出全长support、run bounds、组件与count。无支持记abstain；部分支持记partial，不把组件数宣称完整观测真值、不按覆盖率放大计数；保留全部2364结果行，正式evaluator按其冻结failure政策处理，不删难例。每边i对应原始相邻帧(i,i+1)，事件位置映为i+0.5及可得的PTS中点；组件[s,e)边范围映为源位置[s,e]。**这是响应组件范围，不是重复周期。** PeriodAP保持unavailable，禁止相邻峰配对猜周期。

源帧边界[0,N]单独命名；j<N可映到frame j实际PTS，j=N仅能用经验证的末帧duration/terminal timestamp，否则seconds endpoint=unavailable。不得把source-frame边界N裁成sample位置N−1，或用FPS补末帧秒数。往返必须保存坐标类型、unit及segment；跨断点不进行物理时间插值。

**监督接口 `temporac.native-target.v1`。** 独立target loader保存 `branch_provenance`（full-chain geometry / stable-pair hybrid / rejected）、完整源边域 `target_valid[E]`、pulse、chi、明确traversal bounds、producer/hash。`supervised = response_edge_support & target_valid`；区间内只有producer已经验证的正/负定义可监督。stable-pair只支持其实际认证区间，不能把其余时间置负；缺口和截断cycle不能补成完整traversal。v15现有算法必须另作动态N接口移植和资格诊断，不能接旧v2再把target坐标猜回来。训练按完整身份/源单位归约，不按窗口或run数增加权重；loss权重仅在supervised边非零，W+/W−缺一则该traversal无训练资格并记原因。此处仅定义掩码和坐标，不改loss或推断组件。

## 第3页：交给Sol的具体实现与未完成门槛

**下一交付按顺序执行（CPU、E盘、无新模型权重/标签/GPU）：**

1. 新增 `src/pams/temporac/native_bridge_v1.py`：严格schema/feature-only投影、全长支持/归一化/269通道、run/映射/prediction封装；API固定 `load_native_feature(path)->NativeFeatureV1`、`preprocess_native(feature,clock_meta)->NativeIdentityV1`、`decode_native_identity(response,identity)->NativePredictionV1`。新增 `native_time_cue_v1.py:compute_native_routing(identity)->RoutingResult`，窄幅float-clock入口，保留旧整数API及常量。新增 `scripts/build_temporac_native_learner_bridge.py`，只做 `plan/build/validate --config <json>`；配置显式raw root/完整census/output，逐身份结果与失败账本、原子写、配置+raw+source哈希绑定resume。不得挂上训练启动。
2. 一次有意义的CPU fixture组：N216真实形状及2帧缺失、N900无抽帧；缺失/低分/非有限区别；≥8/hip与31/32共同坐标边界；16/17 samples；Δq=0/倒退/缺失/4.999/5、全无PTS；大整数PTS先相减、rational单位转换；全失败不丢slot；静止保持精确零及uniform cue；任一缺口不串lag/run/窗口/NOLA；known sine在物理单位转换下spectrum/gates不变；不同身份互不影响。数值fixture和预置branch probabilities即可，不读取已训练模型。
3. 验证映射及读出：0、N−1、边界N；不可得末端拒绝秒转换；重复PTS不合并；all-false joint显式unsupported；缺失候选插值永不进模型；NOLA支持边正分母、无支持边零、重叠窗口仍一次解码、partial/all-failed人口保留。对完整有效q=i、正边长fixture验证native cue/通道与旧实现等价（零边长与新增clock-grid规则单列）；旧FeatureRecord仍拒绝216/900，旧scope测试通过。随后fresh Sol差异审查，记录backend身份及same-family provisional，不重做旧泛审。
4. 缓存完成后只运行一次**完整977/2364 CPU build+validate+census**（不抽子集替代），汇总observed/derived/时间/response/target支持的identity、frame、duration分布，分clock mode及失败原因，保留4组train别名和全部失败。首件fixture通过只证明接口；target event coverage需后续真实target产出，不能由帧覆盖推断。

**正式训练仍blocked，且阻塞不阻止上面的实现：** 全raw/全decode最终验证与完整bridge census；独立审查；动态N target producer及完整dev伪计数器质量/覆盖；release/source/exposure及固定split；主count指标/失败处理的最终冻结；完整训练预算和selection/seed配置。旧27-job/100h pilot不能直接替换完整数据预算。`objective.py:45–74` 当前是BCE+positive margin+negative valley **3项**，不满足<3；本桥接不改F11/F14，也不以“一个functional”重命名遮掩。训练前须单独冻结满足要求的loss修订。合法period decoder仍缺失，仅阻塞PeriodAP，不得靠猜区间使它通过。以上没有一项被写成已完成效果证据。

**审读回执与范围。** 完整读E:\AGENTS.md、项目AGENTS/ARIS_LOCK/WORKFLOW、CVPR_RESUME、SCIENTIFIC_RECOVERY_REVIEW、NATIVE_POSE_CONTRACT/CACHE及运行状态；定向核对上述源码、feature_io、preprocess、timebase、response、cue、quadrature、NOLA、objective、decode、旧proposal/plan与runtime schedule。REVIEW_SUMMARY/REFINEMENT_REPORT开头属于已作废WARP-PHASE历史，未沿用。完整读pin `f76ec8d6b77792e2242cca049e08d0cbb7d54e19` 的experiment-plan及output-versioning/output-manifest/output-language/output-composition；安装与源skill SHA同为 `61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a`。任务指定本三页子契约，未覆盖原实验计划；无安装、服务器写入、模型运行、评分、标签读取、commit或push。
