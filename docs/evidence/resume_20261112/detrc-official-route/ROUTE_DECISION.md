# DeTRC 官方复现路线裁定

2026-11-12；固定作者源码 `538abfcd24e0a81b383ca81862ff4360a2389c97`，论文 `2403.01543v4`。本次只处理下一官方整方法对照的既有接口缺口，继承已完成文献工作；不改变 full3、21 项消融、TransRAC full200 或 RACnet 资产任务。

**决定：路线身份已收敛，执行尚未就绪。唯一有源码依据的下一候选是作者发布的 TSN/ResNet-50、2048 维逐帧特征、80 epoch 计数器重训。不能将其叫作论文最强 ViT 结果复现，也不能把“源码可以修”写成当前可运行。** Baidu 任务权重仍未持有；ViT 特征器、对应配置和任务权重的完整绑定未建立。未来若做此候选，应标为“DeTRC released TSN implementation, official-code retraining with disclosed compatibility and population adaptations”。本文件不是训练释放合同。

## 验收与实际证据

| 要求 | 判定 | 证据 |
|---|---|---|
| 精确作者版本、变体及输入可辨 | 通过静态范围 | 实际服务器路径及 HEAD；67 个文件内容、逐文件 SHA 见 `SOURCE_SNAPSHOT.json`；67/67 与固定官方 Git blob 完全一致，缺失0，见 `OFFICIAL_TREE.json` / `ROUTE_RECEIPT.json` |
| 区分机械修复和科学替换 | 通过设计范围 | 下文修复边界；模型、loss、Soft-NMS、训练窗口均不得顺手改好 |
| 数据、目标、指标、选择口径明确 | 通过静态范围 | `INPUT_METADATA.json`：已有 CSV 与解码身份回执核对；原生/共同口径分列 |
| 有当前可执行的完整作者模型 | 未达标 | 任务权重为空，TSN 特征未具备，运行时及默认验证报错尚未实施闭合 |
| 下一完整运行有具体范围与预算 | 部分 | 唯一条件路线及工作量上界；吞吐、编译与显存未测，不能许诺能在预算内完成 |
| 六数据集×七外部框架、六域 SOTA | 未达标 | 本次新增完成框架数 0、性能数 0、数据集数 0；IJCV 为期刊，不能自行充当“顶会框架”资格已满足 |
| 不影响当前工作 | 通过本次范围 | 服务器只读源码/现有 JSON/CSV；没有新数据/权重下载、网络资产探测、安装、解码、forward、训练、评测、Git 命令或共享状态写入 |

## 作者发布的究竟是什么

[论文 v4](https://arxiv.org/html/2403.01543v4#S4) 分别报告 TSN、I3D、ViT：RepCountA 的 TSN 为 MAE/OBO 0.2809/0.4570，ViT 为 0.2622/0.5430；UCF 跨域分别为 0.6016/0.2959、0.5435/0.4184。论文写 80 epochs、batch64、lr0.002、gIoU 系数0.4、validation 选模型。这里仅作身份/差异定位，不与本项目数值直接排位。

[作者 README](https://github.com/lizishi/DeTRC/blob/538abfcd24e0a81b383ca81862ff4360a2389c97/README.md) 明说 TSN；唯一 [配置](https://github.com/lizishi/DeTRC/blob/538abfcd24e0a81b383ca81862ff4360a2389c97/DeTRC/config/repcount_tsn_feature_enc_contrastive.py) 设 `feature_type=TSN,input_feat_dim=2048`。路径名 `feature-frame-mae` 不是 VideoMAE 身份证据。[提取程序](https://github.com/lizishi/DeTRC/blob/538abfcd24e0a81b383ca81862ff4360a2389c97/tools/data/activitynet/tsn_feature_extraction.py) 明确创建 Recognizer2D+ResNet50，而非 ViT。

[准备指南](https://github.com/lizishi/DeTRC/blob/538abfcd24e0a81b383ca81862ff4360a2389c97/docs/prepare_dataset_DetTRC.md) 的 wget 下载 VideoMAE，但实际 `--ckpt` 指定 `tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth`。这个精确文件有 [OpenMMLab 官方模型索引](https://github.com/open-mmlab/mmaction2/blob/v0.24.1/configs/recognition/tsn/README.md) 和 [原始权重 URL](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth)。因此可消除“应该下载哪个骨干”的语义歧义：按 TSN 执行行使用该文件；其 320p 是预训练身份，不能据此把本地提取 crop256 改成320。**本次只验证了官方索引，没有请求权重字节或验证传输，完整 SHA=null；短文件名后缀不是完整 SHA 验证。**

作者任务权重只有已知 Baidu 入口，继承此前不可用记录，本次不重试。固定仓库未给出可直接使用的特征 H5 或已验证任务权重镜像。[Issue 3](https://github.com/lizishi/DeTRC/issues/3) 是外部用户建议迁移 Hugging Face，当前无回复，不能当成已有镜像。[Issue 1](https://github.com/lizishi/DeTRC/issues/1) 的版本建议同样不是作者确认。

## 输入及目标的精确边界

1. 复用原 RGB 视频和既有媒体 SHA、完整帧数、UCF inclusive ROI 身份；不能复用 TempoRAC STMAE、RACnet Swin 37632维特征或 TransRAC 64帧224 RGB缓存。维度相似、同为视频特征都不构成数值等价。
2. 忠实 TSN 输入是官方 OpenCV raw-frame 路径：全部帧短边256、保存 JPEG，再 RawFrameDecode、短边256 Resize、CenterCrop256、RGB mean `[123.675,116.28,103.53]` / std `[58.395,57.12,57.375]`，逐帧 `frame_interval=1`；保持 UntrimmedSampleFrames 与 start_index 的原实现，未来以完整帧编号对照收据闭合其0/1基偏移，不能手写新采样器。直接从原视频 tensor 省去 JPEG 是不同预处理，不能默认为等价。
3. 官方 ResNet50 eval/no_grad，每帧一个 segment，空间平均池化到2048，FP32；提取内部批量200。原代码先把整个视频送入 CUDA，再分200帧 forward。H5 key 为视频名，值 `T×2048`，不额外 L2 normalize。拼接脚本硬编码 UCF/train，需只做路径/分片参数化，保留值和顺序。UCF 已冻结 ROI、QUVA 全视频是本项目跨域输入适配，需单独注明；作者未发布 QUVA 原生路线。
4. 训练沿用作者循环起止监督，归一化 center/length，clip512、stride128、interval1；窗口保留循环覆盖率≥0.75，空监督窗口由作者代码跳过，不引入新 sampler。测试 clip512、stride384，覆盖全序列并保留末窗，作者 padding/masks 不变。

服务器只读对照已有 CSV 和完整解码身份得到：

| 人口 | 作者 CSV / 本项目源名单 | 作者加载器实际行为 | 未来共同评价约束 |
|---|---:|---|---|
| RC train |758 /756| 本项目已冻结排除 `stu3_27,stu4_50`；作者 loader 再跳过无 count 的 `stu6_44`，因此可监督755 | 写明755，不填零标签或冒称756全部优化 |
| RC validation |131 /131| 默认 `exclude_empty=True` 去掉零计数 `stu9_67`，原生选择人口130 | 保留131共同诊断；若沿原生 OBO选重，明示130，不暗换选择指标 |
| RC test |152 /152| 默认去掉零计数 `train143`，原生人口151 | 共同评价必须完整152；对空 GT 只做空数组 shape 修复，不制造循环标签 |
| UCF/QUVA |105 /100| 源库没有这两套完整同口径适配 | 使用冻结ROI/全视频，输入只含特征/长度；真实scalar只在外部计分 |

有可解析count的共享行，其scalar、L端点、帧长在此次核对中均与现有manifest一致；无count行单列，未补标签。作者 loader 会忽略非正长/越界起点并裁剪终点：训练 `stu6_5` 为12 scalar/11有效循环、`stu4_3` 为4/3、`test118` 为6/5；测试 `stu4_5` 为51/41。这些是原生目标规则造成的既有差异，不能改标签使数字一致。共同计分仍用冻结scalar，原生循环数仅作另列诊断；已知媒体冲突和test-informed历史资格继续保留。

## 必须保留的运行行为

配置默认80epochs、单GPU batch64、AdamW lr0.0002、weight_decay0，step15/60、gamma0.1；FP16配置被注释。未给默认 seed，未来单次完整候选可固定 seed0，须标为本项目冻结值。特征器冻结离线；训练模型为投影2048→512、encoder2/decoder4、8heads、40queries，作者 deformable attention 和 relation attention、内容查询、全部层对比项保持原样。此配置和论文 lr/gIoU 不一致，选择发布代码默认值，并明确不声称论文数值精确复现。

实际 [模型代码](https://github.com/lizishi/DeTRC/blob/538abfcd24e0a81b383ca81862ff4360a2389c97/DeTRC/model/DetTRC.py) 还包含质量预测、GT查询辅助、各层辅助loss；`_parse_losses` 只加名称含 `loss` 的项，`iou_decay` 被记录但不优化。分类输出经过softmax后又进入cross_entropy/测试sigmoid；不擅自把它“修回logits”。分类主/辅助系数前25个已完成forward处于warmup，后续才生效；这些Python状态不在普通state_dict中，恢复必须另存 `cls_warmup_step/coef_ce_now/coef_count_now`、RNG/optimizer/scheduler，不能恢复后再次warmup。不能因为本项目方法要求loss少于3，就删改外部作者模型的原生优化项。

测试最终分数是 `sigmoid(softmax(class)) × predicted_IoU × predicted_centerness`，保留作者跨窗坐标合并。原生 evaluator 先计算 `ori_*`，再执行 Soft-NMS `sigma=.1,top_k=30,score_threshold=1e-4`，最终数正类 score>0.2 的查询。**30是整视频后处理总候选上限（包括类别竞争），不得提高上限、去NMS或test选阈值来改善比较。** 原生GT为有效循环个数，MAE分母 `gt+1e-6`；作者配置按每epoch的**后NMS validation OBO最大**选重，严格改善保存、平分保留较早权重。native/ori都存，最终357共同计数也从同一固定后处理得到。

共同主指标仍为 frozen `temporac_cvpr/metrics.py`：positive-count `mean(abs(np.round(pred)-gt)/(gt+.1))`，OBO≤1；完整保留OBZ、RMSE、/gt、all/zero与连续诊断。一个RC-validation选出的权重迁移全部357，不按UCF/QUVA测试选权重。不将作者论文表、原生GT口径或一个seed变成同口径显著性/SOTA证据。

## 修复与替换的分界

| 项 | 可接受的后续工程闭合 | 不能静默做的替换 |
|---|---|---|
| 文件/注册 | `configs→config`、真实文档名、`DetRC→DeTRC`的单一namespace适配；训练/测试入口显式导入注册模型与数据集；路径参数化 | 用本项目计数头替代官方模型 |
| 依赖 | 作者 `setup.py` 依赖缺失的 `mmaction/version.py`；应在隔离目录使用明确固定的上游旧API包，再把作者包加入路径。记录实际来源及差异 | 造一个version.py宣称完整安装；使用当前健康任务环境就地改依赖 |
| CUDA算子 | 编译作者 GridSample1d/ROIAlign；对新Torch仅做API拼写适配，保留kernel和浮点精度 | 换成相似grid_sample/ROIAlign实现并声称数值相同；这些kernel只dispatch float/double，不能开启AMP省显存 |
| 默认位置分支 | `nn.Embedding.unsqueeze`、`repeat(B,-1,-1)`确会失败。对**本次从零重训**，权重初始化全零且训练路径始终传None，因此可只修测试张量取值/合法repeat，保持严格零值，并在保存/恢复时断言零。该静态不变量须在实现中验证 | 给训练加可学习位置编码、切fixed-sine；对未知作者ckpt的非零位置权重套用同一修复。transformer的原permute顺序也须在张量契约中说明，不能用512==512掩盖泛化问题 |
| NumPy/CSV | `np.bool→bool`；固定pandas旧整数位置语义或显式iloc；恢复保存；空GT转(0,3)及标签隔离 | 改gt clipping、填补缺失count、换窗口/数据增强、恢复两条已排除污染行 |
| 完整评价人口 | label-free inference包装器让全部152+105+100输入通过，真实GT只供外部scorer；保留原生151诊断 | 过滤零计数或困难视频，然后称完整357；用人工cycle终点来适配UCF |

MMAction版本仍没有作者可验证的精确锁。README称1.3.10；官方该tag查询404，而源码使用旧API。`v0.24.1` 可提供这些API及该TSN特征提取逻辑，官方声明MMCV1.3.6–1.7.0，是**可准备的兼容性候选，不是已确定作者环境**；未检查的Torch/CUDA现代适配不能写成已通过。作者 [注册报错 Issue 2](https://github.com/lizishi/DeTRC/issues/2) 尚无答复。未来最小环境应先固定 MMAction/MMCV/Torch/CUDA/compiler 完整组合及作者kernel编译可用性；本次不安装、不做GPU小试。

## 唯一条件完整路线与预算

前提清单全部满足后，下一次单独冻结：**seed0 ×80epochs，RC755监督/原生RC130 validation-OBO选重，同时保留RC131共同诊断，最终完整RC152+UCF105+QUVA100**。这是未来候选；不扩展当前矩阵、不立即抢占GPU。先完成精确TSN权重与全部输入，完成必要服务器接口/CPU检查和一次实现审查，再在现有任务完整边界分配一张卡。任务权重不可得不妨碍从零完整重训计数器，但不能因此省略骨干/输入/内核就绪证据。

继承1244条输入身份（保守地包含那个不参加训练的缺标签视频）共954550帧；FP32 T×2048特征 **7,819,673,600 bytes（7.28 GiB）**，H5元数据另计。原生RGB→JPEG需新构建，已有64帧缓存不够；按视频流式提取并归档精确JPEG/特征身份，可不把所有JPEG同时常驻。保留原帧 JPEG 编码路径是数值合同，流式批处理只是I/O组织。原始视频不重复获取。

| 范围 | 帧数 | 不删空监督窗口前的窗口上界 |
|---|---:|---:|
| RC train756保守上界 |645860|3463 train windows/epoch；batch64最多55 updates/epoch、80epochs最多4400 |
| RC validation131保守上界 |112704|323 test windows/epoch，80次最多25840 |
| 最终357 |195986|627 test windows |

没有实测吞吐，不能借用TransRAC或论文全模型计时。规划敏感性：特征20/100fps约13.26/2.65 GPU小时；训练每update1–5秒约1.22–6.11小时；validation/test按窗口批处理的实际耗时另计。**建议未来候选的最大预算48 GPU小时：提取≤16、训练+全部validation≤30、最终评价≤2；48是硬上限建议，不是完成承诺。** 达上限保存可恢复状态并报告完整任务未完成，不改成短跑、降采样或小样本。当前没有授权该预算的正式 RUN_CONTRACT。

资源前提：一张独占/已获配24GiB卡是否能承载batch64尚未证明；单份训练输入为256MiB，测试batch64可能有7个padding窗口、仅输入就1.75GiB，均非峰值保证。保守派生物配额64GiB、保留服务器空闲128GiB；若需要常驻全部JPEG须重新用实际尺寸核账。环境/权重传输、CPU JPEG编码、编译时间单列，不能藏进GPU计时。OOM或kernel不兼容保留实际失败，不自动microbatch、混合精度或改算子。

## 尚缺什么，以及对总目标的贡献

`PREREQUISITES.json` 列明：精确TSN权重字节/SHA与完整特征；作者运行时未锁定及兼容环境/kernel就绪；上述机械修复、完整空标签推断与durable恢复；未来全量合同/资源。ViT最强论文版本与Baidu任务权重是独立未解缺口。获取作者完整可执行环境/任务权重能增加作者复现确定性，但本次没有联系作者或安排探测循环。

本次贡献是把模糊的“DeTRC有代码但不知怎么跑”收敛为一个明确、较小特征成本的TSN官方代码候选，并提前暴露其30候选上限、循环GT/人口、默认测试分支与论文版本差异。它将来至多新增一个实际外部家族和三个已有域的可比较结果，不能填满六数据集×七框架；不能用一个带发布实现限制的弱结果排除更强ViT文献来制造SOTA。当前官方方法完成数及SOTA判断不变。

身份：requested_model=`gpt-6-astra`，requested_effort=`xhigh`；actual_model/effort未独立证明，backend_attested=false，same-family provisional。此为一次静态科学接口设计，不是交叉家族验收、实现通过或submission audit。固定ARIS的experiment-plan/research-lit及适用输出协议已读取，安装副本与源SHA一致；用户指定单一报告、只读和范围限制优先于技能通用模板/广泛检索/toy默认。机器回执保留细节。
