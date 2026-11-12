# G1后唯一决策修订：完整获取官方作者资产

日期：2026-11-12。本文件是既定G1结束后的单次解释，依据真实新资产证据及父任务最新要求；不重开方法/文献/当前训练矩阵。此前设计与G1限制保留为历史。**决定：选择一次有限、可恢复的完整作者RC/UCF资产获取；不再做递增小探测。正式训练仍未释放。**

## 验收与本次变化

| 要求 | 判定 | 证据/执行边界 |
|---|---|---|
| G1按固定范围真实完成 | 通过 | G1_RECEIPT SHA19d3124926942077394ea07396c5086f09f19167a79f06c996b0b12fa772a0e4；8个Range、16HTTP hop、77,936,474字节、53.7224秒，零重试 |
| 获得足够依据决定是否继续作者路线 | 通过决策范围 | RC实见img_feature[930,37632]float32；UCF完整105个NPZ目录及完整样本；原生特征资产真实可达且与模型维度吻合 |
| 作者完整RC目标/755训练资产已到位 | 尚未 | 首RC NPZ仍不完整，gt_tsm/length不可臆测；完整获取将验证原始目标字节，不自行修标签 |
| native755有标签训练资格 | 可明确冻结为后续训练人口原则 | 现有756记录仅stu6_44的count为空，bounds为空；原get_labels_dict明确不纳入空count行。755是完整合格原生有标签训练集，不是抽样或分数筛选 |
| 同人口完整评价 | 继续要求357，未缩减 | RC152、UCF105、QUVA100；零计数与争议标签均保留 |
| 磁盘和传输代价实际核对 | 通过本次资产规划 | 15:28:55UTC空闲685,453,217,792字节；真实速度和存储预算见下 |
| 当前21组/TransRAC独立合同不变 | 通过 | 本会话未写实验源码/共享状态、未操作进程、未启动下载/安装/GPU；root决定实际部署该有限获取 |

## 755是原生资格，不是把缺失标签改成零

现有已知来源修正RC训练清单：原758，排除已有两条明确train/test同源记录stu4_50、stu3_27，保留756。其中stu6_44没有scalar也没有任何cycle boundary。RACnet的原始get_labels_dict在 `if not row['count']` 分支不把该行放入标签字典；字符串`'0'`不为空，真正零计数行仍被保留。现有服务器元数据静态核实**其余755行均有scalar**。因此，将来正式RACnet采用：

**RC755 labeled / validation131 / test152，且UCF105+QUVA100迁移评价。**

这是官方监督输入的完整合格人口。将目录枚举与非空原始标签字典作固定键交集属于显式eligibility适配，避免原loader遇到空标签返回None，并不发明新标签。stu6_44仍保存在原manifest、原CSV、原媒体和作者资产（若存在）中，单列`excluded_from_optimization: missing_native_scalar_and_cycles`。不能把其空bounds当作已知零动作、不能声称与TempoRAC/TransRAC756优化人口相同。

此决定据父任务最新指令，**只修订此前将“必须756”当作所有官方基线永久条件的过严理解**。不改变TempoRAC或TransRAC的756合同，不排除任何评价行，也不恢复两条已明确排除的曝光训练样本。未来结果标签必须写“official-code retraining; full native labeledRC755; source-overlap corrected; seed2333; native200epoch default”，而不是作者checkpoint、原论文精确复现或同训练预算比较。

其余七个越界端点仍不裁剪、不删视频。它们有合法监督记录，应使用**作者实际发布的完整gt_tsm数组**；先取得并保留其SHA/shape/内容。若缺失/不兼容，则完整资产获取的输入验收失败，不自行修成另一种TSM。原start-target路径与end-target原行为保留；不得以TransRAC的clip方式替代。

## G1五个UCF名字差异已具体定位

完整目录有105个regular NPZ。此前100个匹配及5个“missing”是严格大小写比较结果。实际存在五个相同g/c编号的`HandstandPushups`条目，对应冻结ID中的`HandStandPushups`：

| 作者stem | 冻结stem |
|---|---|
| v_HandstandPushups_g21_c04 | v_HandStandPushups_g21_c04 |
| v_HandstandPushups_g22_c02 | v_HandStandPushups_g22_c02 |
| v_HandstandPushups_g23_c04 | v_HandStandPushups_g23_c04 |
| v_HandstandPushups_g24_c02 | v_HandStandPushups_g24_c02 |
| v_HandstandPushups_g25_c06 | v_HandStandPushups_g25_c06 |

冻结这五条**文件名映射**，不运行任意模糊匹配，不把case-insensitive字符串相等当作媒体内容已验证。全量解包后要求105一一对应、无重复、embedded count与冻结标签一致、逐视频T与冻结full_N/ROI_N关系明确。原文件名和数组字节完整保留，映射仅在外围manifest，不能覆盖既有媒体。

已取得样本 `v_BenchPress_g24_c01` NPZ SHA1339221c652334894550f1cf868d42c2733494976edaebb4415a6c6ebd2513bc，只有img_feature和count两个键；它的T=69恰为full_N=ROI_N，因此不能证明其他视频被正确裁剪。未取得RC完整NPZ，仍不能声称gt_tsm/length实际存在。所有这些限制原样进入完整资产验收。

## 唯一下一动作：61对象完整有限获取

执行身份与完整列表继续使用[ACQUISITION_MANIFEST.json](ACQUISITION_MANIFEST.json)中的`fixed_full_object_inventory_reference_only`：

- 官方来源 `luoadore/RACnet_feature_npy`，唯一revision `2196a15bb50de973f245cff318acd1cbe997bdc8`。
- 60个RC分片合计125,484,960,235字节 + UCF ZIP3,144,832,399字节 = **128,629,792,634字节**。61个明确LFS SHA不可改变；Countix不获取。
- 原manifest SHA `7d315797bcfad9b3437b1d435c9032a024822231749c045837ca32109f605ae8` 不变。本修订将61对象从“仅参考”提升为**父任务可具体部署的有限资产范围**；G1原来的96MiB/12分钟只约束已经结束的G1，不能混写为新获取已经通过。
- 固定顺序建议：RC .00 → UCF完整ZIP → RC .01至.59。原始G1范围文件先核其记录SHA，再复用到确切偏移；不能重下已验证77,936,474字节。**待获取唯一新字节128,551,856,160**。
- 新目录 `/root/autodl-tmp/TempoRAC/baselines/racnet-author-assets20261112`；G1原目录保持不动。新文件、partial、索引与receipt均在该项目根下，不改已有官方导出/环境/原数据。不要回收其他项目或未经归档文件。
- 一个有限获取进程，**并发1、最多48小时墙钟、固定61对象**。允许保存每段实际新字节，显式恢复时仅请求缺口。每个完整对象校验已知LFS SHA；传输中记录Range/Content-Range/ETag/偏移/长度/segment SHA与逐对象完成状态。重定向签名URL可随请求刷新，canonical revision不得变。不得把过期签名URL当作新来源。
- 网络失败保留已写字节、偏移和日志并停在明确失败；不从0重跑，不自动换revision/下载其他替身，不用定时任务。一次进程内按既定连续缺口完成数据传输不是新的科研probe。
- 不复制合并成第二个125GB tar.gz：按固定顺序只读串流60分片。可在已有连续前缀上生成可恢复的归档索引；早期完整条目的schema回执是同一次全量资产任务的进度，不触发新的小探测或科学review。发现明确损坏/错误归档/必要训练键实际缺失即保留证据并结束，不能继续消耗预算以掩盖失败。
- 完整解包只做数据资产验证：安全路径、member size/CRC、所有NPZ的非object numeric键/shape/dtype、实际population与gt_tsm来源；无网络训练代码执行、无特征/目标生成、无框架依赖安装、无模型forward/GPU。未被选为训练的原作者条目仍保存。

**此设计会话没有启动全量获取。root可据本文件作一次实际部署；没有新用户审批要求，也没有附带GPU训练许可。**

## 磁盘与实际传输预测

[G1_TRANSPORT_AND_DISK.json](G1_TRANSPORT_AND_DISK.json)绑定原G1回执与15:28:55UTC服务器只读磁盘快照。当前free为**685.45GB（638.38GiB左右）**。本次原作者资产范围使用：

- 归档128.63GB；
- RC解包硬上限220GiB，UCF目录已知解包3,341,174,298字节；
- 归档、解包、partial、索引/日志与G1已保留范围的**合计占用硬上限384GiB =412,316,860,416字节**；
- 始终保留项目所在卷至少**128GiB =137,438,953,472字节free**，不占用/删除其他项目资产。启动必须重新读真实free；当前free高于二者之和549,755,813,888字节，剩余余量约135.70GB。每个分片/解包批次更新盘额；到限保留进度并停止。
- 此预算不包含后续QUVA特征或RACnet训练输出，它们不在本次动作范围内。实际解包尺寸若越过上限，不能以压缩档已下完为由越额写盘。

| G1已测速度口径 | bytes/s | 推算完整61对象 |
|---|---:|---:|
| 全部G1时间含小请求/CPU解析 |1450725|24.63小时|
| 4次RC数据请求合计 |1863675|19.17小时|
| 最后一次32MiB请求 |2380829|15.01小时|

这些只来自约78MB、53.72秒测量，不代表24小时稳定吞吐。**按20–30小时规划，48小时有限运行上限；不承诺GPU0约15:51交接前完成。** 该交接继续给已就绪TransRAC；RACnet资产没有GPU需求，不抢占当前21组或TransRAC。每完成一个完整对象以实际已传字节/时间更新预计余时，不额外跑测速样本。

## 全量资产之后的客观结论

完整获取完成不等于完整训练就绪。唯一资产完成报告须明确：

1. 61对象LFS SHA与全部安全解包通过。
2. 原始训练/val/test成员清单；native755均具有实际feature/gt_tsm及对应注释，尤其七个端点异常视频；stu6_44是否存在及其原始状态，始终不造标签。
3. 五条明确大小写映射之后UCF105一一对应、原始count/T/ROI关系。任何内容来源/预处理无法证明之处如实列出，不用匹配长度假装像素/归一化等价。
4. QUVA100作者数组仍未发布；保留固定官方RGB路线及既有原权重为后续完整特征输入准备。全量资产不得自行生成QUVA、不自动在test上选归一化或阈值。
5. 原代码导入/验证等机械缺陷、H-norm的native零梯度资格、batch16/BatchNorm资源、耐久恢复与validation selection，继续按先前设计记录。不能借资产取得改原生loss/decoder或恢复内部mock head。

输入闭合后才由root落实一个明确的**755×200epoch native训练合同 +357共同评价**；这不是再次选方法/重搜文献或增加矩阵。若原始作者资产不能提供完整原生监督，届时结论是此路线资产不足并保留已获原始证据，而不是开始一连串伪兼容试验。

本次父任务请求Astra/xhigh；actual backend仍未认证，同家族provisional。本修订没有实验结果、SOTA或七顶会框架完成主张。

