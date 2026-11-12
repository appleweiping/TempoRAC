# RACnet 一次集中闭合清单

当前：**CONDITIONAL_NOT_LAUNCH**。执行者填写 `PASS / FAIL / PENDING` 和实际证据SHA；不以文字承诺代替结果，不对已绑定的通过项重复全量检查。科学参数只读 RUN_CONTRACT.json。

| Gate | 必需证据 | 当前 | 不满足时 |
|---|---|---|---|
| G0 合同/原源绑定 | 合同SHA、RACnet51b8e3f全部科学文件SHA、允许diff清单、README/技能读取receipt | 设计完成；实现待做 | 不部署未绑定代码 |
| G1 固定作者获取完成 | 同一61对象128629792634bytes LFS SHA，安全解包完成，保留原deadline/cap/reserve及历史失败恢复 | PENDING，17:56UTC为2/61 | 保留字节/失败原因，不换revision/删证据/扩Countix |
| G2 完整原生资格/监督 | RC755train/131val/152test一一映射；train/val原TSM[T,T]/length/T/finite；七个特定end异常实际数组；原始3排除记录；886原生target长度/finite | PENDING，RC已见28train | 完整监督资产不可用则停止本路线，不删7行或造TSM |
| G3 UCF ROI | SOURCE_SCHEMA_EVIDENCE105映射，105嵌入scalar一致，fullT105，88slice/17full；loader按既定切片，原NPZ不改 | 元数据PASS；实现待做 | 不改成任意模糊匹配/时间插值，不drop评价行 |
| G4 QUVA入口与完整生产 | 100原身份、48030帧；9950权重；完整Swin/basecfg/flatten/[-1,1]/转置源绑定；逐视频输出SHA/shape/finite/timing | 固定路线；未提取 | 停在具体媒体/模型错误，保留完成文件；不归一化试选 |
| G5 一次CPU实现闭合 | 复用G1 CRC/finite；1243名单及标签隔离；targets/decoder常数与border/native-vs-scalar/masks；严格bone keys；无模型forward的环境导入；checkpoint全状态/epoch边界恢复原语；single fresh Sol集中检查 | PENDING | 一批修明确工程错误，定点回查；不重开方法/metric |
| G6 正式资源/启动身份 | exact source/config/input/env/contract SHA；run独有目录；最多1释放GPU及owner/lane；32GiB新增cap、128GiBfree及未完成资产余量；健康任务完整arm后root交接 | PENDING | 等资源，不能打断TransRAC/消融/他项目，不占另一lane |
| G7 首正式epoch耐久 | seed2333/fresh权重；epoch0全部755一次、48attempts、actualupdates/AMPskips、有限loss/grad、完整latest状态SHA、全batch统计/峰值显存/时间 | 未运行 | 真实OOM/不可恢复非有限：记录失败停止；无小GPU重试或改batch |
| G8 全训练与选模锁定 | 200 contiguous正式epoch；每epoch755/48attempts；189validation131；min exact nativeMAE/earliest，单bestMAE SHA；bestOBO仅归档 | 未运行 | 不称完整、不用test/last填缺口 |
| G9 一次全357评价 | 同一bestMAE SHA；RC152/UCF105/QUVA100全ID；概率/原count/constantflags；native GT+metrics与共同scalar/metrics分开；零行输出 | 未运行 | 缺行则失败，不能以positive子集掩盖人口不完整 |
| G10 完整交付与比较资格 | run/输入/权重/差异/成本/失败史/完整raw预测SHA清单；主positive及all/zero表；未知预处理/单seed等限制 | 未运行 | 不宣称SOTA、组件有效、七顶会或投稿ready |

G0..G3/G5可在当前资产工作与GPU训练不受影响时准备；G4的完整正式提取需要实际GPU lane回执，但不是试验小样本。全部训练必需输入G1..G5闭合后才允许G7。G6也适用于G4的正式GPU占用；QUVA特征生成完先保存，再进入200训练。中途只做健康监控、耐久性和冻结成员完成检查，不按分数调参或重开设计。

一次fullCPU check使用已有元数据和首次必需的标签/桥接检查；不重复读取全部原视频、构建任何64帧cache、跑模型GPU smoke或重验已绑定Swin原文件下载。第一正式epoch和第一正式QUVA视频都属于持久化全量任务，不能用过后丢弃改参再叫“首跑”。
