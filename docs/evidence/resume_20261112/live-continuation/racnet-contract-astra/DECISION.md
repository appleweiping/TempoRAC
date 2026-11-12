# RACnet：一次闭合的完整原生训练与 357 评价合同

2026-11-12。**决定：继续既有作者资产路线，科学接口冻结为 RC755 × seed2333 × 200 epochs，单一 RC validation-native-MAE 权重，最终 RC152 + UCF105 + QUVA100。当前状态是 `CONDITIONAL_NOT_LAUNCH`。** RC 资产尚未完成、QUVA 特征尚未生成、隔离桥接和运行环境尚未验收。本文件允许据此准备实现和完整输入；不冒称训练已释放，不改变下载进程、GPU0 TransRAC、GPU1 消融或既有剩余队列。

本次只检验 C1：同一完整评价人口与共同评分下，TempoRAC 相对真实官方整方法的表现。必须排除的替代解释是自制仿冒头、删行或不同评分造成优势。此路线不增加 RACnet 种子/消融/调参矩阵，不恢复废弃主线。输出存在不等于 C1 获支持。

## 验收表

| 验收要求 | 本次状态 | 证据与后续客观门槛 |
|---|---|---|
| 固定 ARIS、适用技能与原生科学源码全读 | 通过设计范围 | experiment-plan 与三项 mandatory refs 全读；README/SETUP 三项继承 SHA 重核一致；RACnet 训练、验证、测试、模型、loader、计数、GT/TSM、RGB extractor 与配置全读；实际源码 SHA 见 SOURCE_SCHEMA_EVIDENCE.json |
| 全量 UCF 时间轴及标签闭合 | 通过元数据范围 | 105/105 作者 T=full_N、embedded count=冻结标注；88 需要 ROI 切片，17 无需缩短；逐项范围、源 SHA、NPZ SHA 已保存 |
| RC 完整原生监督输入 | 部分，未就绪 | 17:56:34 UTC 仅28 train NPZ 通过既有 CRC/finite；其余及七个异常端点的实际 TSM 尚待固定获取完成 |
| QUVA 完整可执行输入路线 | 设计通过，执行未完成 | 固定100原视频、48030帧、官方逐单帧Swin；复用已有原视频身份/官方权重/原始骨干源码；无再选归一化或特征替身 |
| 训练/选模/评分/官方权重界限一次冻结 | 通过设计范围 | RUN_CONTRACT.json；native MAE 选择、OBO 只保存次要记录、全部357评价与原生/共同GT分开 |
| 必要机械修复与耐久恢复有确定边界 | 通过规格范围 | 本文及 one-shot-closure-checklist.md；尚未实现或验收 |
| 完整200训练和357正式结果 | 未达标 | 尚未启动；不把取得资产、CPU通过或第一epoch当作完成 |
| 六数据集×七框架、六域SOTA、组件贡献、最终论文 | 未达标/不属本次闭合 | 原目标保留；RACnet 是 ICIP2024 Oral，不伪称 CVPR 顶会方法；无 SOTA、显著性或投稿就绪判定 |

## 1. 结果身份与范围

官方源固定 `51b8e3f2decdeb10dbc7fd506a93a9f7d2d4d4fb`，服务器导出 `/root/autodl-tmp/TempoRAC/external-baselines-20261111/code/RACnet-51b8e3f` 无 .git，按原来源回执和逐文件 SHA 绑定。作者 task checkpoint 仍未取得，SHA=null。现有 Swin K400 权重 SHA `9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc` 不是 RACnet 任务权重。

唯一结果标签：**official-code native-default retraining; fresh RACnet task weights; corrected full native-labeled RC755; seed2333; 200 epochs; RC-validation-selected transfer; author RC/UCF features + official-video-path QUVA features; qualified modern runtime**。作者发布特征不等于作者发布计数权重。若作者任务权重今后可得，保留为独立来源，不能替换本次已经冻结的 fresh trajectory 或混入其结果。

原论文相关实现段的既有阅读证据说100epochs、batch16、A6000 48GB、少于9小时；当前 YAML 明确200epochs。本次遵循代码默认，不声称论文数值或预算精确复现，也不把不同人口/预训练/训练预算称为完全公平匹配。

## 2. 完整输入与监督边界

复用现存 manifest SHA `61cbbd8c98c7a9764284dd2f032d85d2cfb9403fedabf695fe90271f31fa2333` 与已完成解码身份 SHA `9341a1b74120c5fd12172a1c8af1ecdba32403db32a6392c86a6343ba3e6882d`。它们只提供原媒体身份、标注、full_N/ROI_N；**不读取或复用 TransRAC 的64帧 RGB 缓存、目标或特征**。

训练755：原758去掉已知 train/test 同媒体 `stu4_50`、`stu3_27`，再排除原始 scalar/cycles 均缺失的 `stu6_44`。这是既定原生监督资格，不是选择性删难例；三个原始记录/媒体/作者文件仍归档。真实 scalar=0 保留。RC validation131、test152 原样保留；UCF105、QUVA100 不作训练/选参。总必要身份1243，最终357；RC test151正计数+1零计数，全部都预测并保存。

每个 population 内以 qualified key 的 Unicode/ASCII 字典序固定列表，记录完整列表 SHA；训练用原生 shuffle=True，validation/test 不 shuffle。目录枚举不决定资格。固定列表排序是可复现性适配，不能声称复现作者未知 os.listdir 顺序。

**RC**：直接读取作者 `img_feature[T,37632]` 作为原 loader 的 `frames` 别名，不改原 NPZ、不另存双份特征；若未来同时有两个键，要求 dtype/shape/payload 完全一致后才视为别名，否则失败。`length` 为整数 scalar，等于 feature T、已完成 full_N；train/validation 的作者 `gt_tsm` 必须有限、shape[T,T]，原生 FloatTensor 转 FP32。保留作者原始 float64 TSM，不重新生成、不裁端点。

`GenerateGT` 由原 CSV、原列顺序、`int(float(...))` 转换生成 start/end，保留原边界行为。七个已知越界位置全部是 end：`stu1_10` 的1840对T1681，其余 `stu11_11/8, stu12_0/1/2/27` 的末端等于T；全部 start 在范围内。原 `local2globel` 对这类 end 在写入前 break，start 监督不需裁剪。CPU全量检查须验证此原行为和所有886个train/val目标；作者七个实际TSM仍必须存在且兼容。若缺失或shape不符，路线输入失败并归档，不能丢七行、换TSM、额外循环探测或静默改训练人口。

**gt_tsm 是监督张量，不是网络输入。** 优化仅使用train755的target_start与TSM。validation可读取其标签计算原生诊断/选择，绝不反传。模型 forward 仅接收 feature、从有效长度产生的 masks、固定 sim_mode/feat_norm。final evaluator 的feature loader不接收count/bounds/TSM；RC native GT 由隔离 scorer 生成。test TSM 不属于预测前提，不能因没有 test TSM 删评价行。所有掩码长度来自 feature/已冻结ROI，不能从计数或目标非零位置生成。

**UCF**：已完成105个作者文件全部是 full_N，并非105个都已裁ROI。冻结 `SOURCE_SCHEMA_EVIDENCE.json` 的105个一一映射与五个已定 HandstandPushups/HandStandPushups 大小写别名；仅这些显式别名允许。共同输入为 `img_feature[start:end+1]`，start/end来自原 manifest 的含尾端ROI；88条缩短，17条为全区间；所有输入仍105。UCF native TestData 本来从feature.shape[0]取长度，实际缺 `length` 键不是阻塞。嵌入 count 与冻结标注105/105一致。原归档目录名 `valid` 不将它变成本项目validation，仍是外域test，绝不参与选模。

本次将“全人口共同评价”定义为同一视频及标注ROI的时间行范围；作者逐帧特征是否使用同一媒体字节、色彩/归一化、ROI外上下文均无法完全证明。采用可验证的作者namespace/revision/LFS SHA、成员SHA、唯一ID、全帧长度、scalar/ROI匹配作为有限准入标准，并明确记录上述未知。**不再将像素同源或训练/QUVA特征数值一致证明设为无限期门槛**；亦不得据此宣称像素等价、公平性无保留或精确论文复现。不额外做full-video UCF推断矩阵；此105 ROI结果是官方模型在共同ROI的适配评价。

## 3. QUVA：固定完整官方逐帧提取

仅100个现存QUVA原视频，全长48030帧、最长1820，预计 FP32 payload 7,229,859,840 bytes。一次正式完整提取，逐视频原子保存与SHA，断点只复用已完成且身份匹配的文件。它是完整357的必要输入准备，不是额外性能实验或GPU pilot。不得用首几个视频改参数后重来。

严格遵循 `test4video.py.get_item` 的实际行为：OpenCV完整解码，BGR→RGB，`cv2.resize((224,224), INTER_LINEAR)`，保留 `transpose(0,3,2,1)` 的 H/W 交换，FP32 `(x-127.5)/127.5`；每次一帧 `[1,3,1,224,224]` 输入 backbone，eval/no_grad、无外部autocast。输出 `[1,768,1,7,7]` 按channel、height、width顺序展平为 `[37632]`，逐帧顺序得到 `[T,37632]`。可以流式解码以节约RAM；不得改成64/32帧clip、时间池化、ImageNet mean/std、[0,1]、批内多帧或空间转置“修正”。QUVA count/bounds 完全不用于提取。

已有 `third_party/TransRAC/mmaction/models/backbones/swin_transformer.py` 是可复用的完整 Video-Swin backbone 源码依赖（SHA `775982f84fedbeb364f62daafb63e79d5d5ae11208b9eab6ec484b4f9b1f2734`），不是采用TransRAC模型/特征。复用其完整原始计算图，必要时仅隔离注册/导入；不走 TransRAC counting/model/cache 入口。缺失 `_base_` 可由已有同名Swin base config作显式展开：patch(2,4,4)，embed96，depths[2,2,6,2]，heads[3,6,12,24]，window(8,7,7)，mlp4，qkv_biasTrue，qk_scaleNone，drop/attn_drop0，drop_path0.1，patch_normTrue，LayerNorm，use_checkpointFalse。完整加载现有9950权重的 backbone 状态，严格验证所有骨干参数和buffer，不静默遗漏；分类head可明确不实例化。每个单帧的第二个时间patch位置由官方PatchEmbed3D原始零padding产生，不能复制帧替代。

RACnet配置中的K400 classifier test_pipeline不执行。`test4video.py` 中 E-norm只影响返回TSM，计数头不读取TSM；我们统一按已冻结H-norm final forward，decoder使用 `test.py` 的一次 `count_peaks_scaling`。不再重复video demo的外层min-max。作者数组可能由不同预处理产生的问题作为 transfer资格披露，不做test驱动的归一化选择。

## 4. 原生训练、选模与评分

200个epoch索引0..199，seed2333，单GPU逻辑且物理batch16，drop_last=False：每epoch47个16样本batch+末尾3样本，48次尝试更新，合计9600。全部755每epoch一次，不长度分桶、不裁T、不microbatch/accumulation替代BatchNorm。原生模型4stages/10layers/64maps，FinetuneF Conv3d/BatchNorm3d/两次dropout0.2，残差层dropout默认0.5；计数任务权重fresh，Swin仅作冻结离线提取。

Adam lr6.4e-5、betas(0.9,0.999)、eps1e-8、weight_decay0，MultiStepLR milestones40/80/120/160、gamma0.9，每个epoch后step。训练原生torch.cuda.amp.autocast FP16 + GradScaler；validation/final和QUVA extractor FP32，无外部autocast。现代PyTorch的TF32/cuDNN设置显式记录并保持启动冻结；不为内存换精度。两项loss：`sum((p-target_s)^2) + 1e-5 * sum((TSM_out * (TSM_gt != 0) - TSM_gt)^2)`；mae_error=False，不加SmoothL1或其他第三项，不改sum归一化。原生padding及其常数概率/损失贡献保留。

H-norm真实为 `cdist(emb,emb,p=0)**2`、负号、mask、row min-max+1e-8。既有PyTorch2.5.1官方CUDA源码证据显示p=0 backward为0，因此此TSM项对embedding的梯度为0；它仍是实际原生loss的记录项，不能宣称训练中实现了有效TSM约束或修成E/COS。计数分支仍由start SSE学习。此资格不阻止忠实复现已发布实现，必须跟结果一起披露。

**唯一主选模：RC validation131，epoch索引11..199共189次，最小native MAE，精确值相同取最早epoch。OBO不作tie-break。** native MAE保留验证batch16/不shuffle、target与预测的 `count_peaks(pro=0.3)`、denominator(count+0.1)、九个batch mean再等权平均的源码语义（末batch3样本）。在日志同时保存每视频值和pooled mean，后者不选模；不按共同metric或测试域重选。原生初始best_mae=1会漏掉所有MAE>=1的有效权重，外围归档改用+inf，保存所有候选标量，最终严格argmin；这是保存/选择定义，不改变优化。另存native bestOBO权重及其validation记录，**不用于final357、不额外按它跑一个test矩阵**。无有限合格validation则判失败，不回退到test最佳或last。

一次选定bestMAE后同一SHA用于RC/UCF/QUVA，不目标域选模、无ensemble/shift。final batch1、eval/no_grad、FP32，原生 `count_peaks_scaling(pro=0.2)`：每序列min-max，再SciPy find_peaks及两端规则。保留无epsilon源码；有限常数序列导致0/0缩放，但原峰规则返回count0，作为明确native常数序列诊断记录，不改decoder。输入概率非有限则为健康故障，不能被count0掩盖。

主表评分固定 source489dc77 `temporac_cvpr/metrics.py` SHA `fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0`，原始scalar：positive NMAE=mean(abs(np.round(pred)-gt)/(gt+0.1))↓，OBO=|rounded error|<=1↑；RC151、UCF105、QUVA100。所有357保存预测及全部/零行NMAE/OBO/OBZ/RMSE、positive /gt、raw-continuous。RACnet预测count是整数，保留原输出而非捏造连续估计。

另报native RC GT=`count_peaks_scaling(GenerateGT.start,pro=0.2)`与native全152 MAE/OBO，逐条保留native GT和scalar差异；UCF native scalar即作者count，QUVA无作者协议，写“official decoder transfer”。不能把native RC GT替换主共同scalar，不将ROI适配评价冒充作者full-video数字。全部概率序列、原count、对应checkpoint/input/config SHA都持久化。一个baseline seed不支持训练方差显著性或普适SOTA主张。

## 5. 一次实现与资源边界

独立候选目录 `R/baselines/racnet_native_v1`。原始作者树、已下载数组、原媒体和所有活跃环境只读。允许的机械修复集中为：tools缺Optional/torch/np/random imports；phases的tools.misc指向真正tools.AverageMeter，移除未使用且不存在的obo_thres导入；validation五值解包错误改为四个实际使用张量，start_end判断改为已冻结start；manifest-backed key/ROI/length adapter；mask显式送同device；日志/atomic checkpoint/selection/export。不改model.py计算图、targets、loss或decoder。最终test loader不要求无用test TSM。

优先新隔离 `R/envs/racnet-native310`，只读继承已验证project Python3.10.22 / Torch2.5.1+cu124 / torchvision0.20.1 / NumPy1.26.4，独立overlay限定已用兼容版本SciPy1.14.1、OpenCV4.10.0.84、MMCV1.4.0、timm0.4.12、einops0.3.2及PyYAML实际绑定。作者推荐Python3.11.5、Torch2.3.1、SciPy1.14.0的差异必须披露，不能叫原环境逐位复现。现有TransRAC环境/依赖可作只读来源，不能安装、升级、卸载或写其site-packages。只导入实际需要的包，不安装未用pysdtw/wandb等。当前此环境未创建，本会话没有安装或model import/forward。

先完成一次全量CPU closure：复用worker已经完成的SHA/CRC/finite，不为审查重复解包/解码；完整1243名单和105ROI映射；886原生监督目标；精确源/环境/权重键；必要decoder/mask/恢复原语。仅由Sol做一次集中实现检查，修明确缺陷后定点闭合，不重开科学接口。随后正式完整QUVA提取（可先于200训练），输入完整后在root实际资源回执下启动唯一200训练。不得先拿一小组GPU做试训；第一正式epoch必须纳入完整轨迹并保存。

耐久保存从epoch0开始：initial seed/model identity、latest、bestMAE、bestOBO、first-full-epoch；完整model/optimizer/scheduler/GradScaler、Python/NumPy/TorchCPU/CUDA RNG、next_epoch、训练顺序/loader RNG、最佳选择状态、epoch日志和源/输入/合同SHA。每epoch训练+验证+scheduler完成后临时文件fsync、原子rename、目录持久化，再提交完成标记。若中断在epoch内，保留失败日志，从上一个完成边界恢复同一epoch RNG与顺序；不得宣称失败尝试的工作未发生，记录replayed_steps，不重复计入正式完成。checkpoint保存与额外诊断不消耗训练RNG。

资源预算是**一条200epoch/9600 attempted-step轨迹 +189全validation +一次357final +一次100/48030帧QUVA提取**，最多一个实际释放GPU，不新增seed/重跑/调参。GPU墙钟尚未实测；仅从第一正式完整epoch和正式提取进度计算ETA，不引用TransRAC吞吐或把论文A6000耗时当4090承诺。作者batch16在24GiB fit未验证：最长训练T2550的单份FP32输入约6.14GB，另有dropout副本/卷积/反传/TSM；不能声称“肯定装下”。首次真实OOM/不可恢复非有限故障保留现状，停止本路线的当前运行并给出资源原因，不自动改batch、T、BN、loss或丢样本；这就是有限退出标准，无小试循环。

现有作者获取48h原deadline、384GiB cap、128GiB free reserve及原61对象不变，当前worker仍按其已接受的操作恢复合同工作。另给本run/QUVA/隔离overlay最多32GiB新增空间，始终满足同卷128GiB reserve，启动按实际存储与尚未落地的既定资产余量核算；不得从作者384GiB中偷占或删除其他项目。任务本身有限200epochs，不将资产48h误作训练早停时限。只在已授权lane完成当前完整训练+评价后由root手动交接；不主动停止健康TransRAC/消融，不为显示方式或本设计重启它们。

## 6. 关闭条件，不再循环讨论

one-shot-closure-checklist.md 是唯一执行清单。所有科学选择已冻结；后续只能填写真实资产/实现/runtime/SHA证据。未知作者像素/预处理来源以已列资格准入，不触发额外科学审查。缺失原生train/val TSM、固定身份不符、decoder输入非有限、原生资源不可运行才是具体停止理由。任何原生科学语义改变均是另一个协议，不能塞回本次run。

请求身份 gpt-6-astra/xhigh；actual backend/effort 未认证；OpenAI同族 **provisional**。本次不是独立跨族接收、ultra深审或论文提交审计。未改Git/current_status/source_sync/运行中源码，未读private signed URLs，未下载、安装、GPU计算、信号或启动任何任务。
