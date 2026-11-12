# TransRAC 官方完整训练执行契约 v1

**决定：推进 TransRAC 官方源码的完整原生训练，seed 1、200 epochs、RC756 训练/131 validation、357 条完整最终评价。** Task checkpoint 不再是此路线的前置依赖：计数网络随机初始化，只有官方 Swin-T K400 backbone 需预训练权重。这是独立基线任务，不改变正在执行的21项消融。使用一张4090时保留原四卡分片的计算语义；所有安装、输入准备、检查、训练和评分只在服务器进行。

本契约是实施与执行决定，不是假称环境或训练已经可运行。Root 在下列必要工程验收完成后，手动接收一条已经完成当前消融臂全部训练、评估和回执的lane，随即启动正式200轮任务；不再需要用户重复选择常规配置。另一条lane继续原队列。作者源码固定于 `68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979`，原树不修改，独立driver导入其模型/target函数；只记录兼容性必要改动。

## 验收表

| 条目 | 冻结要求 | 当前证据/状态 |
|---|---|---|
| 确切官方模型 | TransferModel，64帧，scales=[1,4,8]，OPEN=False | 已完整静态读取原train.py、model、train_looping、label_norm、两loader和Swin配置/相关代码；未运行模型 |
| 输入和标注 | raw RGB[0,1]；训练756、val131、test152；UCF105/QUVA100 | 原CSV已实际SSH读到，SHA与本地源一致；完整prepared/cache核验待执行 |
| 预训练权重 | 官方127,414,861字节Swin，SHA9950e3be...76acc；无计数task权重 | 确切URL/预期SHA已存在旧审计；并行asset执行器取得真实bytes后验收 |
| 一卡等效逻辑batch | 四个native shard、正确BN缓冲/梯度/RNG语义 | 静态机制已明确；服务器CPU最小原语对照待实现 |
| 完整训练 | seed1，200轮，4800个逻辑优化机会；每次AMP skip可见 | 未启动；不以小样本、短epoch或GPU探针替代 |
| 完整比较 | best由预定val规则选；357逐视频原始预测、native/common双计分 | 待训练完成；不会从测试表现选择输入分支或epoch |
| 实际性能/复现强度 | 标签为官方代码重训、现代兼容、修正训练人口和声明资源适配 | 待测；不是2022原论文数值精确复现，也不是作者发布task checkpoint |

## 源码已经消除的歧义

`train.py`实际导入`dataset.RepCountA_raw_Loader.MyData`，而非NPZ loader。原始视频完整顺序解码；N>=64时取索引`floor(i*N/64), i=0..63`；N<64时取全部N帧并重复最后帧到64；BGR→RGB、OpenCV默认线性resize到224×224、ToTensor转float32/255。没有ImageNet mean/std、随机crop或flip。Swin配置文件里写的MMAction训练/测试归一化pipeline没有被此手工loader调用。

NPZ loader使用[-1,1]，`test.py`默认导入它，但也明确提供raw替代。**本次raw训练必须使用相同raw分支评价**，不能把两个入口默认值机械拼成归一化错配。所有人口都在整视频或已冻结UCF inclusive ROI内应用相同start-aligned采样。官方UCF helper另有end-aligned索引，本次不混用。此选择在训练和测试前冻结，写入结果标签。

`OPEN=False`只让backbone forward处于`no_grad`，并没有使backbone永久eval。训练时`model.train()`会启用Swin的stochastic depth（有效drop_path_rate=.1）和计数网络dropout。不能把此模型改成eval-backbone、离线固定learned-feature输入或我们的ST-MAE tokens。输入RGB解码缓存可以复用，学习到的backbone特征不能用静态缓存替代。

原`TransferModel.forward`在scale4时对x补零，scale8继续使用已经补过零的x；shared BN1在三个scale各调用一次。尽管可以提出其他写法，本次保留这些实际官方代码行为。完整网络、attention、时序Transformer、所有dropout、BN参数、卷积与density读出不改。

## 输入文件和准备步骤

根目录`R=/root/autodl-tmp/TempoRAC`；独立执行目录建议`R/baselines/transrac_native_v1/`，环境`R/envs/transrac-native311/`，原始官方树`R/third_party/TransRAC/`。新文件均留在R内，E盘只收控制文件和小证据。

必须绑定的既有输入：

| 文件 | 内容与身份 |
|---|---|
| `R/data/UNION_known_source_v1/manifest_dev.json` | SHA256 `96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2`；过滤rc_train/rc_valid/rc_test得到756/131/152 |
| `R/third_party/EveryShotCounts/datasets/repcount/train_with_fps.csv` | 575440字节；SHA `4cfee3cfd904da8f701cf8a812ae809cfc2c94e56d55dfec7f2c5253afd7f857` |
| `.../valid_with_fps.csv` | 70154字节；SHA `49695468686d894c06beb27fc4182675c762960175126a6eb8372b6148f2d148` |
| `.../test_with_fps.csv` | 82683字节；SHA `9779090e122aaa0c19d16d167d9b7bb1258f5d3e0d7a1739cda92b99c43a12e2` |
| `R/data/UCFRep/manifest_full.json`及冻结union UCF test记录 | 完整105条、相同媒体和inclusive frame_range；以既有冻结评价身份核对，不重新拆分 |
| `R/data/QUVA/manifest.json` | 完整100条test；沿用full/RepNet已有输入SHA和计数 |
| 原media文件 | 保留字节、标签、帧边界及角色；不重下覆盖已有文件 |

本次实际SSH确认原CSV存在；它们不是“仍需寻找的作者标注”。继承`prepare_report_train_valid.json`和`prepare_report_test.json`显示原758/131/152行与Lance的count和完整bounds全部匹配。准备器再将这三个CSV与冻结manifest逐行绑定，排除训练`stu4_50`、`stu3_27`，不排除两条争议test或零计数记录。CSV与manifest不一致时报告具体键，禁止静默选择一种标签。

生成`inputs/train.csv,valid.csv,test.csv`和有序`inputs/manifest.json`。L1..Lk保持原始列顺序和全部值；原raw loader用`int(string)`，故对形如`72.0`的**精确整数值**输出`72`，先检查finite和value.is_integer；不是四舍五入、重新标注或改变边界。count=0的两条已核对记录在valid/test中bounds=[]，原target函数可输出全零density。不要使用TransRAC仓库的`open_set/new_*.csv`，其重新分组不是当前标准split。

可在服务器建立**完整uint8 RGB输入缓存**来消除200轮重复解码。每条保存64×3×224×224字节数组、source SHA、ROI、完整实际解码长度N、64个原帧索引及sampler版本。先ROI再采样；resize的实际OpenCV结果、RGB通道及float转换必须与官方raw loader一致。N来自实际完整解码，不拿CSV num_frames字段冒充。可使用顺序流式解码以节省RAM，必须验证最终完整帧数且保留原索引。CPU检查覆盖长/短/ROI/零计数代表输入，以及完整人口的身份/shape/finite检查；不使用小样本性能测试。若缓存接口对不上，就直接用已固定raw loader，不改归一化。1244条训练/验证/三个评价人口的未压缩uint8上界约12GB；887条train+val约8.55GB。这是RGB输入缓存，不是新特征研究。

## Backbone与现代环境

官方Swin预训练URL：
`https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth`

预期127,414,861字节，SHA256 `9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc`。来自项目既有官方资产审计和runner BACKBONE_SPEC。复用已验证bytes；没有则由asset执行器在R下获取并检查。该文件是K400识别backbone，不是TransRAC计数器task权重。新decoder不能加载任何旧TempoRAC/ESCounts/TransRAC counting state。

直接沿用项目`docker/transrac-official/ENVIRONMENT.md`已实际成功的modern兼容路线：Python3.11、Torch2.5.1+cu124、torchvision0.20.1+cu124、MMCV1.4.0、timm0.4.12、einops0.3.2、kornia0.5.11、NumPy1.26.4、OpenCV4.10.0.84、SciPy1.14.1。该历史receipt在A6000验证的是推断，不冒称当前4090训练ready。新建隔离环境，不修改现有TempoRAC环境；服务器已有Python3.10实验env和basePython3.12，按现有conda创建R内Python3.11环境即可，不强行套老Torch1.7/CUDA11.0。

继承已审过的直接包清单；tensorboardX和plot依赖可以用逐行JSON日志替代官方writer/未用绘图导入，这些是报告I/O，不改变训练运算。必要新依赖只为实际import错误补齐并记录。避免安装互相冲突的mmcv+mmcv-full两个版本、apex、TensorRT、mmdet等未被TransRAC实际forward路径使用的重依赖；已成功的MMCV1.4.0普通包优先。若当前import确实触发无关检测分支，限定改动到导入/注册或可证明不在本模型执行图的依赖；不得用假Swin层、替换网络、静默丢checkpoint keys来“跑通”。

CPU载入官方checkpoint并核对每个backbone参数/缓冲的映射、shape和loaded coverage；官方recognizer构造后取model.backbone的路径保持不变，不用另建近似Swin。所有计数层仍按seed1随机初始化。保留官方源码树和SHA；兼容性diff单独记录。现代环境的数值/随机实现与2022原环境不承诺bitwise相同，结果标签明确。

## 一张卡上的四分片语义

原train_looping使用DataParallel四张卡，global batch32，每副本8。模型含BatchNorm3d和BatchNorm2d，且默认DataParallel只将主副本的buffer更新留到下步。因此普通“batch8累积4次”会让BN running stats更新四遍，并不等价；“单卡batch32”会改变训练BN的归一化人口。

固定如下实现，物理GPU仍只有一张：

1. DataLoader产生完整逻辑batch32，最后batch20；drop_last=False。`torch.chunk(x,4,dim=0)`得到4×8或4×5，target使用相同分片。数据顺序有专用seed1 generator并保存在checkpoint；所有756训练行每epoch出现一次。
2. 每个逻辑batch开始zero_grad一次，克隆所有BN的running_mean/running_var/num_batches_tracked。每个virtual shard forward前恢复该共同起始buffer。不要恢复/清空模型梯度。
3. 恢复该virtual shard的独立CUDA RNG状态，调用**原TransferModel**训练forward和AMP loss。保存其更新后的RNG状态；仅保存shard0结束时的BN buffers为本逻辑step最终状态。共享BN1在一次forward的三个scale内照常连续更新。
4. 对b个样本的mean density MSE乘`b/B`后backward。所有shard使用同一GradScaler scale。加权和恰为完整DataParallel gather后的全B×64 cell mean；不额外除以4两次。
5. 全部shard backward之后恢复shard0的BN buffers；GradScaler.step(Adam)与update各一次。最后batch20的四分片均为5。保存该逻辑step的loss、当前scale、梯度是否finite、optimizer是否真正更新及overflow skip。

形式上，原损失 `L=(1/(64B))*sum_j sum_{i in shard_j,t}(d_it-y_it)^2 = sum_j (b_j/B)*L_j`；不同视频之间没有cross-batch attention，唯一batch耦合BN保留在每个原生副本分片内，buffer主副本规则也单独保留。因此梯度累积是原native局部batch语义的资源实现，不是新优化法。随机数由四个virtual流支持与四副本相同的stochastic分布；不声称与历史不同CUDA/设备调度逐位相等。

CPU必需工程证据：小型原语网络包含共享BN重复三次、第二个BN和线性MSE；比较四个从相同参数/buffer复制的reference replicas与上述串行累积。分别覆盖B32→4×8和B20→4×5，验证加权loss/参数gradient/一步Adam/主副本BN buffers；随机dropout关闭以检查算术，RNG分流另做状态连续/恢复fixture。FP64原语以紧容差比较，记录实际误差。它不模拟数据集性能或四GPU完整CUDA parity。另测scheduler步数和epoch checkpoint恢复。全部在服务器CPU，不能把这些fixture作为scientific run。

若B8的真实首个正式batch出现OOM，保留失败attempt和状态，报告准确memory/代码点；不能自动改microbatch4、eval-backbone、尺度或模型来凑过。可作保持上述原生BN人口的具体资源修复后显式恢复，但不得用一串GPU短跑选配置。

## 原生训练与选择

一个seed1；`torch.manual_seed(1)`在官方模型构建前执行，CUDA四virtual流也由seed1初始化；Python/NumPy种子和DataLoader generator固定并记录。训练200完整epochs索引0..199，global batch32、Adam(lr=8e-6, betas=.9/.999, eps=1e-8, weight_decay=0)，不加gradient clipping、AdamW、warmup或新augmentation。

源scheduler为MultiStepLR，milestones=[0,40,80,120,160]，gamma=.8，每epoch末step一次。Torch2.5的初始scheduler step会在milestone0把首次实际lr变成6.4e-6；之后epoch40/80/120/160起为5.12e-6/4.096e-6/3.2768e-6/2.62144e-6。CPU实例核对并落盘完整200轮lr序列；**不因为论文写8e-6就“修掉”milestone0**。

target直接调用原`preprocess/normalize_label`：按实际N将全部边界ceil映射到64网格、上界clip63、排序；每对边界积分高斯，重合格子的原赋值而非相加、塌缩周期设1，全部保留。原code计数为density sum再round，未必等于CSV scalar count；准备阶段按人口保存差异账本，不重新缩放target强制一致。

唯一训练目标是mean density MSE，未乘TempoRAC的100系数；mae_error=False。SmoothL1/count-MAE在原loop可计算为diagnostic但不参与梯度。输出末端是线性层，density及sum-count允许为负；不得加Softplus/ReLU或把负数clamp到0。FP16 autocast/GradScaler保留；native GradScaler overflow skip可以发生，全部记录实际successful/skipped更新，不静默补跑新的step或改lr。

每epoch写atomic latest；包括完整model/BN、optimizer、scheduler、scaler、CPU/Python/NumPy/CUDA与virtual RNG、generator、已完成epoch/global logical step/actual optimizer step、配置/输入/source/env、日志偏移和best身份。保存best独立文件。显式恢复从最近完成epoch开始，保留未提交attempt日志，不把重算部分计成额外完成epoch。

源代码epoch>50开始validation，epoch>50且epoch%3==0保存checkpoint。因此**主模型候选固定为51,54,...198，共50个**；训练仍继续至199。验证完整131条，每个候选均保存逐视频连续count。按130个正计数CSV GT的canonical rounded/(g+.1) MAE最小，OBO最大，earliest epoch确定best。原程序没有明确“best”函数，只按含native batchmean MAE的文件名保存；该统一指标选择是本次预先声明的选择接口适配，不称作者选择结果。保存native continuous-prediction/density-GT metrics以及原批均值diagnostic，但不据test或事后某个指标切换checkpoint。validation固定沿源在51..199每epoch记录，候选范围不变。validation按逻辑batch32分成最多四个原生shard，在eval模式顺序计算；保存固定seed1 validation generator及每轮顺序，以重算native未加权batchmean诊断。

## 一次完整最终评估

加载选中的唯一RC checkpoint，eval模式，全部冻结RC152、UCF105 inclusive segment、QUVA100 full video。每条一次原raw视图，无shift/TTA/exemplar/阈值调参；三个域共用相同RC权重，UCF/QUVA明确为transfer。CPU缓存只是已冻结RGB取样值。保存原64维density、原torch sum得到的连续count及sample/source/ROI/checkpoint身份；不要先把density升精度再求和改变原始计数运算。

native计分与harmonized计分分开：RC native target count采用原native density-derived count及其dtype/rounding；UCF/QUVA scalar GT单列。保存native的rounded误差/(g+.1)和OBO以及训练/val原continuous diagnostics。公共主指标只用原CSV dataset count，NumPy ties-to-even round，positive-count/(g+.1)，并保存/g、全部/zero、OBO/OBZ/RMSE及continuous诊断。完整152/105/100 key集合必须一致，争议/zero样本不能删除。公共scorer绑定`R/source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr/metrics.py`，SHA `fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0`；不是旧`src/pams/metrics.py`的half-up实现。继承现有冻结版本，不改变21项或已完成RepNet/ESCounts评分。

结果行应写“TransRAC official-code retraining, corrected RC756, raw loader, modern compatibility, seed1; common validation selection”。完整native训练不自动等于作者paper-number reproduction；一个baseline seed不证明其训练方差，不能用本方法三seed与它一个seed制造训练随机性显著性。旧84-dev、旧权重和作者文献值分别留存，不混入本run。

## 最小实施清单与命令

只需一个独立`run.py`入口和少量可测试函数，复用原模型/target/现有canonical scorer与已关闭的durability模式；不构建通用框架。建议固定CLI三个phase：

- `--phase prepare`：绑定全部source/CSV/media/population；输出整数CSV、有序input manifest、可选完整RGB缓存、target差异账本和hash；不加载GPU模型。
- `--phase cpu-preflight`：验证身份/shape/全部人口、backbone映射、官方loader对照、虚拟四副本梯度/BN/RNG原语、scheduler以及恢复fixture；输出明确PASS/失败点。不得运行本地计算或GPU训练probe。
- `--phase train-evaluate --gpu N`：资源归属检查，完整正式200epoch，然后唯一best完整357评价与双指标、结束退出。失败保留状态，不自动新seed/新设置重试。`--resume`只在身份一致且root显式指定时生效。

准备/安装/必要CPU检查可以与21项并行；CPU worker建议2，数据cache完整且只在R内。正式GPU命令在既定manual hold完成后运行，一张物理卡，绝不在脚本中重设CUDA_VISIBLE_DEVICES为0,1,2,3；逻辑cuda:0只对应root给出的那一张卡。所有Windows控制根进程隐藏/无控制台；Linux worker stdin断开、日志重定向。原队列残余清单保留，baseline完成后由root显式恢复，不设置timer或自动监控任务。

新的driver/兼容diff和CPU结果由一位fresh Sol做一次有界工程核对，重点是实际source graph、raw sampler、target不变、virtual BN/gradient/RNG、selection、durability和full357 membership。不要重做本次科学设计、已完成文献阅读或旧资产审计。缺陷修具体点，不展开大审查循环。

## 工作量、执行范围与当前状态

严格范围是**1个完整seed×200epochs**，75600×2=151200次训练视频呈现、4800个逻辑优化机会、每full batch四个native virtual shards。每个原TransferModel forward有64+32+16=112次backbone crop调用；200轮共19200个virtual shard forward，即2,150,400次批量Swin crop调用，未含完整validation和最终357。validation若按源51..199每epoch执行为149×131=19519次视频呈现。这个明确工作量比声称“轻量decoder重训”更准确。

当前没有此full trainer在4090上的实测吞吐，不能将TempoRAC缓存decoder的1.6小时搬过来。初始资源预留按单卡24–48 GPU-hours规划，**是调度预算预测而非实测上/下界或截断200轮的许可**；在正式第一完整epoch完成后用实际训练/validation耗时更新剩余预测。如果预测显著超过预留，root可在持久epoch边界报告预算影响并重新安排同一完整任务，不能把短跑当正式完成。禁止为了估时另跑小GPU训练。RGB输入约12GB未压缩，backbone127MB，计数模型/optimizer/checkpoint若逐epoch全存会增长，保留atomic latest+best+必要candidate identities，源候选权重可按冻结best选择持久化策略节约空间，原始数值/选择轨迹不删。

目前**设计已冻结，资产/环境/CPU实现/实际训练尚未完成**。不存在需要重开availability campaign的科学阻塞；CSV位置和raw分支已通过实际代码/SSH证据解决，modern compatibility有既有成功先例。真正当前依赖是已知backbone bytes、独立环境、driver和最小CPU验收。若出现特定兼容故障，先按原图机制修复；只有实际失败证明该路线在限定修复内不可运行时才考虑已列RACnet，不能仅因“legacy”一词跳过TransRAC。

## 读取和审查身份

完整experiment-plan及其必需references沿用本session同pin读取。新完整读取research-implement-feature及effort-contract、acceptance-gate、reviewer-independence、capture-antipatterns、injection-hygiene、review-scope-limits，其他routing/trace/integration references已读；只借用其声明假设与实施验收模式，本任务没有实施代码或宣称完成完整build/sweep流程。该skill源/安装副本SHA均为33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8；ARIS pin f76ec8d6b77792e2242cca049e08d0cbb7d54e19不变。

requested Astra/xhigh；actual model/effort未独立证明；same-family provisional。本文不启动、停止或修改任何job，不安装、不在本地运行实验、不改共享repo/status/paper。Exact upstream reads和CSV SHA的SSH读取属于静态证据；新的必要计算留给服务器实施器。主要假设在ASSUMPTIONS.md先登记，之前可能选择只在DECISIONS_HISTORY.md归档一次。
## 冻结时的实际输入补充

14:13:22 UTC的asset准备回执已成功生成756/131/152整数L字段CSV和1244人口manifest；原1041条RC记录的count/bounds完全对应，媒体存在和字节大小已核对。该回执尚不等于全媒体SHA、RGB缓存/loader等价或环境验收通过。训练`stu6_44`的源CSV count为空，冻结manifest count=null，bounds=[]；保留在756内。原empty-bound函数的零density是官方目标行为，而不是把缺失scalar count改标为0；原始null必须出现在provenance和target差异账本。此情况不影响131条validation或152条test的数值GT。准备器首次strict Decimal guard停止的attempt保留。

机器可读`RUN_CONTRACT.json`固定所有选择，SHA `892109aeff36dfe0d96f3203e8f8b08936e19073111ed58ef09a36399a8a6856`。完整预算和必要CPU证据已交接独立native执行器；本设计者没有安装、训练或修改正在运行的队列。

最终源码算术明确项：只有训练forward使用FP16 autocast；原training/train_looping.py的validation和testing/test_looping.py没有autocast，所以固定FP32 eval/no_grad。Native train/val GT为sum(target)再转FloatTensor再round；native final test GT为原sum(target).round()，不插入FloatTensor转换。连续prediction在原torch sum后、round前保存。公共指标仍只取原CSV scalar GT。这是原始运算接口明确化，不修改实验设计。

