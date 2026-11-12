# RACnet 合同假设、证据与有限退出

此表不是待用户反复批准的问卷。科学选择已在 DECISION.md / RUN_CONTRACT.json 冻结；后续工程证据按一张清单收齐。当前 `CONDITIONAL_NOT_LAUNCH`。

| ID | 事实或明确采用的假设 | 证据/资格 | 收敛标准 |
|---|---|---|---|
| A01 | 源码为作者51b8e3f导出，任务权重未取得 | 当前逐文件SHA与既有官方来源相同；task SHA=null | 完整fresh training，不能冒称released task weight |
| A02 | 作者数组按命名ID/长度/标签代表对应视频 | 61对象revision/LFS与完整成员CRC/SHA；像素/预处理无作者可证链 | 满足明确schema/ID/T/count/ROI准入后继续，保留未知资格，不无限追索 |
| A03 | 755是完整原生有监督训练资格 | 已知重叠2行排除+stu6_44 scalar/cycles缺失；原get_labels_dict跳过空scalar | 不造零标签、不删有标签行，评价全保留 |
| A04 | 七个异常端点可以由作者完整TSM加原生start目标使用 | 七处均end越界，所有start有效；原end函数break；实际TSM尚未取得 | 完整755+131的作者TSM/length/finite兼容即闭合；失败则停止路线，不裁/重造 |
| A05 | UCF105作者行按时间对应全帧 | 已有105 T=full_N；88需要ROI、17both；count全一致 | 固定SOURCE_SCHEMA_EVIDENCE中105切片，保存原文件；不重新提取UCF、不添加fullvideo性能矩阵 |
| A06 | 官方RGB示例是QUVA可执行的唯一固定提取路线 | 逐单帧、[-1,1]、H/W交换、原Swin+9950权重 | 完整100/48030帧各自SHA/shape/finite后闭合；不要求与作者RC特征数值相同 |
| A07 | 既有Video-Swin依赖可以只读复用 | 已完整读mmaction Swin代码及basecfg，hash在合同 | 骨干参数/buffer严格对应，计算图未替换；未认证checkpoint key不静默忽略 |
| A08 | H-norm的TSM梯度为0是已发布实现的资格 | 原p=0代码+继承官方Torch2.5.1 CUDA backward源码 | 原样保留，不能改善后仍标native；不能用此结果证明TSM训练组件有效 |
| A09 | 原生validation MAE是此次唯一选模准则 | parent明确确认；batch均值的均值，0.3，无test | epoch11..199 argmin，ties最早；OBO不tie-break、不跑额外test |
| A10 | 作者和共同标注/评分可能不同 | RC target峰计数与CSV scalar区分；已知冲突保留 | 同一357 raw输出支持两套明确标签，无隐藏负数值/零行 |
| A11 | 新现代隔离runtime足以执行源码 | 既有Py3.10/Torch2.5依赖可读，尚无RACnet runtime回执 | 一次CPU导入/源/权重/原语检查后真实完整运行；不声称原2.3环境逐位等价 |
| A12 | batch16全长可在实际释放GPU尝试完整正式任务 | 原模型BN使microbatch非等价；没有fit测量 | 仅第一正式epoch真实观察；OOM则保留失败停止，不小试/降batch/剪T |
| A13 | 预算以固定完整workload封闭 | 200×48 attempts、189×131 val、357test、100QUVA | 完成全部才done；实际step/skips/时间/存储分别记录，不静默短训 |
| A14 | 对比有资格限制但仍可提供真实证据 | 单baseline seed、不同预训练/人口/预算、历史test曝光、作者像素未知 | 报实测差异，无训练方差显著性、普适SOTA或精确paper reproduction主张 |
| A15 | 资产获取与正式训练的授权/状态分离 | 下载当前2/61完整对象；正式训练未发车 | 无资产/CPU/env/lane回执不能launch；设计文件存在不解除条件 |

只保留三个尚待实际证据的执行门槛：完整RC监督资产、完整QUVA100特征、隔离桥接/runtime及空闲lane。它们都有确定的成功或停止标准，不需要重新选择baseline、阈值、metric、种子或模型。

本次请求Astra/xhigh，actual身份未认证，同族provisional；无新实验结果或科学acceptance。
