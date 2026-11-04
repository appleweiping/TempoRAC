# TempoRAC 首组完整开发训练合同 NTR1（2026-11-04；三页）

**决定：先实现原生完整训练runtime；数据与target诊断齐备后，顺序执行seed20260815的canonical和capacity-control。** 后续相同合同扩展20260816/20260817，不以首组代表完整论文矩阵。沿用NB1、clock-overlay 955b、NT1/LF1 c4e489；不重训F11、不改健康extractor或原始缓存。请求Astra/xhigh，backend未attested，同族设计 **provisional**。

| 验收 | 本次状态 |
|---|---|
| X0来源、单位、源平衡与matched执行 | 接口设计冻结，下面给出代码依据；runtime未实现 |
| 完整population与无目标泄漏推断 | 794 train/183 dev、977/2364全人口保留；完整缓存与诊断pending |
| schedule/选择/资源可检查 | 规则固定；T/人口工作量及实际环境hash待完整census填入，未经实测不保证ETA |
| 质量/SOTA/三组件/三seed/最终矩阵 | 未达标；首组仅是全量开发实验，不升级旧失败或官方资格 |

## 第1页 / 3：X0 analytic adapter与完整输入冻结

本地HEAD实际为 `46b00ea936795849ecc076e8c8fbbe77aaafb2b2`；current_status中的cb2旧快照不覆盖主任务新回执。按新回执raw/decode健康运行；不据快照宣布结束。开跑输入manifest须绑定：全977 raw/decode最终回执、955b全视频clock选择、NB1派生输入、NT1完整target/census、完整train/dev opaque身份表与train source-component映射、X0 bank、代码/环境/配置SHA。四组train别名不删除，共源只占一个component票；GT不进入任何loader。

**现成X0来源。** `x0.py:generate_orbit/generate_view/x0_manifest`：40个固定source；U0000…23训练、24…31仅tune、32…39只heldout；每source七block、每block十traversals。训练/tune使用linear、offset0；stationary durations=(5,8,16,32,64,127,128)，另六块SMF排列以64/32/16实现。边界B_j由duration累计，`p[B_0…B_9]=1`、terminal B_10不计，`chi[B_j:B_{j+1}]=1/d_j`，target mask恰[B_0,B_10)，故每traversal sum chi=1。这些由解析生成器产生，不需natural证书、不需teacher SHA；5-edge synthetic traversal亦不套NT1 natural的≥6门槛。

新增 `X0AnalyticAdapterNTR1` 直接取 `generate_view` 与既有 `preprocess_identity(view.motion,ones,view.clock)`；记录trim_bounds，按同一first偏移切p/chi/target/responsibility和traversal，不伪造320 FeatureRecord、不把X0冒作natural NPZ。269通道与分run响应图沿用既有定义；guard提供输入上下文，但不监督、不用于X0事件计数。`clean_coordinate,phase,p_star,chi,durations`仅target/责任侧，绝不拼入encoder通道。

**时间与roles。** X0实际clock是每个整数q=0…Q；adapter标 `synthetic_tick`、q=float64(clock)、Δq=1，无真实秒/Hz。自然q=t/tau或source_frame_only，单位是每视频nominal sampling interval；不把q误称秒，不给X0编FPS。X0的log(Δu)与log(Δu/32)只差常数，减reference后相同；slow/medium/fast是相对速度、dilation4/2/1仍按rows。X0责任由解析窗口速度给出，自然由detached NUDFT；不是跨FPS绝对Hz专家，也不保证不规则q下row卷积物理等价。既有5…128 cue grid、128/32窗口和fallback不变，无需重采样X0或改变角色。

**可复用bank而非重造数据。** 优先找到已有40-row manifest与对应冻结数组，校验source key、accepted attempt、coefficient/grid SHA及生成contract SHA；复用相同字节。若只有源码，CPU一次构建40源manifest及32×7=224个train/tune view，保存每数组SHA、原v4规范SHA、x0/preprocess/adapter SHA与实际Python/NumPy/SciPy版本；不得伪称已在旧3.12.4/2.1.0/1.14.1环境重现。新环境产生的bank以NTR1独立manifest冻结，所有来源按原64-attempt算法，失败不补seed/换source。在两个CPU进程逐字节复核；任何已有参考字节差异先记入必要兼容说明，不静默覆盖旧bank。40源分割完整保留，heldout view不参与训练/选择。

## 第2页 / 3：配对API、采样、完整schedule与selection

新增 `native_training_runtime_v1.py` 与 `train_temporac_native_response.py`，独立schema `temporac.native-training.ntr1`；不要强塞旧 `RuntimeConfig/JobSpec`（写死A6000/27jobs/10k）或旧CPU-only four-fresh-graph入口。最少API：

`build_training_plan(population,targets,x0_manifest,config)->FrozenPlan`；`prepare_natural_unit(identity,target)->PreparedUnit`；`prepare_x0_unit(source)->7 PreparedUnit`；`paired_draw(plan,seed,step)->(x0_source,component,identity)`；`forward_unit(model,unit,variant)->LF1 loss`；`optimizer_step(...)->StepReceipt`；`score_x0_tune/commit_checkpoint/resume_exact/predict_population`。plan/CPU validate与train/predict分命令，绝无“validate后自动训练”。

**cue/NOLA支持必须正确。** natural `compute_native_routing(identity)`和NOLA始终使用NB1完整 `response_edge_support`，目标不改变窗口、reference或推断区域；仅 `lf1_supervision`给loss提供target交集。旧 `natural_response_loss`把target.edge_mask送cue/NOLA，不能照搬。canonical使用local gates和LF1责任；capacity同样模型/参数形状，所有window gate=1/3、fused为三概率均值、LF1 `capacity_control=True`省略责任权重；不能平均logits。X0 canonical沿用analytic edge责任融合，capacity改uniform并省略责任权重，p/chi/bounds完全相同。

旧torch NOLA逐边逐window建scalar autograd graph，不作为完整N训练热路径。CPU预计算完整支持上的float64 `D_e=Σ_w h_we`、`π_ek=Σ_w h_we a_wk/D_e`及mask；可导融合 `R_e=Σ_k π_ek sigmoid(z_ek)`。这是线性NOLA的代数等价，浮点运算次序改变须具名记录：与现有Neumaier参考做forward/gradient误差检验，不声称bit-identical；D≥1e−3及无支持零规则保持。LF1继承旧训练的fused float32入口、target/责任detach；预测仍以NB1参考float64 NOLA→一次float32→一次decode形成实际可评结果，禁止用target mask预测。

**配对随机性。** 每seed生成一份CPU初始state，canonical/control加载相同参数字节SHA；沿用旧按ASCII参数名的fan-in uniform与LayerNorm初始化，不分别用含variant的旧job_name hash。新RNG域 `temporac.ntr1.paired`、seed和purpose区分init/X0/component/identity；PCG64每cycle独立permutation，variant不进seed。每步一X0 source（24源循环）+一eligible自然component（C个等权循环）；component内n_c个eligible身份按自身访问次数循环。两arm共用完整draw ledger。所有无资格身份/component留在全population，仅无合法监督者不入conditional训练cycle；不得结果导向排除。

**完整固定horizon。** 冻结census后计算 `M=max_c n_c`、`T=500*ceil(max(10000,C*M)/500)`，C=0或X0为空拒绝。这是旧10,000-step基线加上覆盖完整eligible人口所需的最低周期长度，绝非debug cap；启动前将C、所有n_c、T、逐身份曝光次数/总frames写死。每C步是一次source cycle，不能称所有身份一个epoch；前C*M步保证每eligible身份至少出现一次，报告每身份完整访问次数。绝不把candidate资格等于原train人口已全部获得监督。

每step整身份（所有run/traversals）与X0七block：逐block新graph反传 `0.5*L_block/7`，自然新graph反传 `0.5*L_identity`，共8次backward、一次clip(global norm1)和AdamW step；不truncate/run切片造单位，不同时留七张graph。AdamW lr1e−3、betas(.9,.999)、eps1e−8、decay1e−4、foreach/fused=false。warmup=ceil(.05T)，其后旧cosine形状降至0.1×lr；float32 model/optimizer，float64 NOLA，AMP/TF32关、deterministic开。每500步保存全部候选，T为500倍数，无early-stop；原T=10000时须与旧LR逐点相同。

每候选仅用完整8 tune sources×7 linear/offset0 blocks、同LF1/各自variant定义，按block→source均值选择最低finite objective，tie较早step；不得自然dev/GT选checkpoint。完成T且候选齐备后才宣称完整训练。resume绑定所有input/code/config/initial-state SHA、optimizer、step、RNG及sampler计数；保存独立recovery checkpoint不增加selection候选，禁止更换target或重置schedule。三seed固定20260815/16/17，首组之后的四job沿用同plan；LF0三项对照单列pending预算，不自动增训。

## 第3页 / 3：可执行资源门槛、诊断与下一交付

**首组资源合同：** 仅现有服务器GPU1，单进程/单job、两arm顺序；待raw提取及其wrapper正常结束并核对既有作业恢复和GPU1新lease，不能假定提取结束就可抢恢复的FastSAM，不发SIGSTOP/杀进程、不占STaR/其他用户卡。每arm累计GPU占用及运行wall上限各24h，首seed合计≤48 GPUh；该值是本次保守**执行上限，不是实测ETA或A6000换算**。同合同六job计划上限144 GPUh，余四job仅在首组诊断与资源安排成立后顺序推进；不新购云资源。

VRAM上限20GiB（torch peak allocated/reserved与NVML进程占用均记录），启动时GPU1实际可用≥22GiB；不继承旧“24GiB可用”假设。loader_workers=0，CPU线程上限4，输入缓存CPU只存无梯度数组；单identity/单X0 block顺序上卡。超过内存或壁时保存恢复点并标 `resource_incomplete`，不能截短N/T、跳身份或降完整数据量冒充成功。只保留明确可恢复中断；累计消耗不清零，扩预算需另冻字段，不静默延时。

**为何能计划、为何不能保证24h够用。** 现图269→64、8个64-wide depthwise/pointwise residual blocks、3头，主线约52736 MAC/sample-forward，激活随当前最长run线性增长；NT1/DTW和cue均预计算不在训练步重做。plan可在CPU精确统计整个draw ledger的forward rows、最大N/run、X0七块rows、window数/π字节、tune和dev预测rows；训练backward工作粗略约数倍forward，但Python/kernel/IO开销不能从FLOPs精确推时。必须落字段 `C,n_c,T,max_N,max_run,scheduled_rows,cache_bytes,environment_sha,device_uuid,free_vram,free_disk,lease_id`，现在真实值pending。模型/optimizer/checkpoint大小按参数数与dtype计算，所有候选+recovery+预测+cache写入峰值预留两倍并加10GiB安全余量；超过可用磁盘不得删原证据腾出“通过”。首个正式job自身日志统计step/tune耗时与peak memory，无额外GPU微试；初始ETA是unknown，完成前不承诺deadline。

**开跑前一次收口。** 完整raw/decode/955b/NB1/NT1与794/183人口校验、隔离evaluator的完整train/dev target-unit/identity-time-event覆盖和G-NT1全人口误差已交付；不存在跨gap/fake-negative/坐标/数值错误；以上plan字段与训练代码CPU验证齐备。质量容忍界尚未给定，不能写target quality PASS；需在看到完整冻结诊断后记录一次明确“继续完整方法检验/语义失败先修复”的证据判定，不能用dev GT搜索阈值。natural dev仅在checkpoint固定后做全183预测与隔离评分。count开发指标/失败策略必须在评分前具名冻结；官方实现不明就用明确非官方名称，不拖住已授权完整训练。release身份、历史暴露及独立test资格保持pending；A180不删、B78仍候选/9源簇，无test标签访问。

**一个下一Sol任务：** 实现上述两个runtime/CLI文件、X0必要adapter与CPU fixture；先交 `plan/validate` 可审产物，不启动GPU。测试覆盖：40/24/8/8来源与224view哈希；X0十pulse/chi/terminal/trim及5-edge周期；q平移/单位说明与analytic慢快顺序；canonical/control初始化及draw逐项相同、最大component身份完整曝光、resume连续性；目标改变不得改变cue/NOLA；vector NOLA对参考的float64 forward误差≤1e−12、最终float32误差≤2e−7、非饱和梯度相对误差≤1e−5；8次micrograph梯度对同一总loss与clip/step等价；T/LR/500-step保存与tie选择、NaN/缺候选/错hash拒绝。使用人工CPU tensors及解析X0，不读自然GT/不加载已训练模型；独立Sol只复核这次差异。

最终六数据集×七框架、至少六域主性能、三耦合组件及去除消融、三seed统计、observation/超参/效率与复现仍分别未达；不以这两个job替代完整矩阵。记录前处理与纯模型耗时、三seed原始预测与source-cluster统计；损失对照/外部baseline/独立测试按后续明确预算完成，所有负结果保留。

审读：阶段前完整读取pin f76ec8d6b77792e2242cca049e08d0cbb7d54e19的experiment-plan及其literal ../../shared-references四份全文；继承已读AGENTS/ARIS/NB1/c4，c4与955b原件hash重新相符；现LF1/NT1/NB1 core hash分别89d8d4ac/a1c84ef7/50aecf64，新读955b；定向核对x0、campaign_runtime_v1、runtime/training/response/NOLA、现NB1/NT1/LF1/cue与冻结proposal/schedule。此任务仅写本三页子契约，无安装、自然GT读取、模型执行、GPU/服务器操作、commit/push；未改任何活跃文件。

