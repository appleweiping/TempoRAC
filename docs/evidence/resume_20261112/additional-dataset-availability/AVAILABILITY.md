# YTSegments / ActCount：一次有界官方来源可得性核查

核查日期：2026-11-12；服务器请求时间见 official_metadata.json（14:22UTC）。这是未来数据协议的准备材料。当前21项训练和TransRAC357不变；没有下载媒体/数据压缩包、建缓存、跑模型、选数据集或联系作者。

先继承15-framework ledger及LITERATURE_DELTA，而非重做文献综述。既有Nov8执行记录已说明YTSegments的TAU链接不可达。限定盘点未在当前R/data、R/features、R/resume20261111analysis三层目录找到YTSegments/ActCount命名的数据副本；E侧aris-work文件名盘点也未找到完整候选发布包。这不是全磁盘不存在证明。

| 验收项 | 结论 | 可核对证据 |
|---|---|---|
| 两个已发表任务与官方来源身份 | 通过 | 下列作者仓库、CVF记录和Springer DOI |
| 明确完整目标人口 | 通过文献级核实 | YTSegments100；ActCount172；不等于逐文件完整性通过 |
| 当前可取得全部媒体和标注 | 未达标 | YT官方云盘连接失败；ActCount未找到经作者/出版方确认的公开数据入口 |
| 明确数据许可和发布版本 | 部分 | YT代码仓库可固定commit；实际数据包版本/许可不明；ActCount仅论文版本明确 |
| 对既有四域证明来源独立 | 未达标 | 两者均缺可核对的完整source-ID/时间段/媒体SHA清单 |
| 不影响现有实验、保留负面/未知项 | 通过本任务边界 | 仅静态元数据，机器回执列出失败与未知；未改共享状态/Git |

## YTSegments100

原任务为Ofir Levy、Lior Wolf的ICCV2015 *Live Repetition Counting*（pp3020–3028），身份可由[CVF会议记录](https://openaccess.thecvf.com/ICCV2015.py)核对。作者[官方仓库固定版本](https://github.com/ofirlevy/repcount/tree/19b1d80a1827988c5253dfa9601414f5d3faa151)的README将100个YT_seg分段视频作为一个benchmark，另列25个YTIO视频；两者不能混成125条的同一原始人口。

README明确停用仓库内旧代码，改指向[TAU更新代码/数据分享](https://owncloud.cs.tau.ac.il/index.php/s/q5R5M9oHuoQeK9a)。当前GitHub树只有README、live_count源码及权重，没有data/YT_seg。master为19b1d80a1827988c5253dfa9601414f5d3faa151，2018-04-02；这是代码指针版本，不是已确认的数据包版本。GitHub许可证字段为null，树中未见LICENSE/COPYING；云盘数据许可未取得。

本次浏览器访问云盘为502；服务器单次HEAD在14:22:21UTC被远端断开，没有HTTP响应。因此没有取得可枚举的100条媒体/标注清单、下载字节或校验和，不能称数据已就绪，也不能从网络失败推断永久下线。后续[Context官方README](https://github.com/Xiaodomgdomg/Deep-Temporal-Repetition-Counting/tree/f7011d3ef29061b9c63cb7c12d5feeb2a93f7d35)仍转引该作者仓库；它不是媒体完整性的替代证明。

原始计数标签文件格式、是否含逐周期边界、后续补标版本与原版的对应关系仍未验证。1×/2×/4×等速度变换只能是同源诊断，不能增加独立数据集数。YTIO也不能未经源级核对就当作另一个独立域。

## ActCount172

[官方Springer章节](https://link.springer.com/chapter/10.1007/978-3-031-40289-0_24)为Gan、Su、Wen、Zhang，KSEM2023/LNCS14119，pp295–306，2023-08-09发布，DOI10.1007/978-3-031-40289-0_24。公开摘要确认172个视频、约1,870次重复动作标注及非人类中心内容；未公开可验证的文件清单、划分、标注格式或媒体下载入口。

本次出版方页面仍是订阅预览，未列数据/补充发布链接；有界作者/题名/ActCount搜索未找到可归属于作者或机构的公开数据仓库。结论仅为“本次未定位到权威公开入口”，不是“数据不存在”。论文的Springer版权条款不是数据集许可；数据使用/重分发条款、数据版本、完整172条媒体及计数/边界文件均未知。官方页面给出通讯作者Jia Su及机构联系信息，可供以后由用户明确安排的数据请求；本轮没有发信或购买章节。

## 来源独立性与具体后续取得步骤

两者都是任务相关的**待资格确认候选**，不能现在把“四域+两个名字”报告成六个完整独立域。YTSegments来自网络视频，ActCount的来源/裁剪ID清单未取得；与RepCount、Countix/Kinetics、UCFRep及QUVA的源级重合尚不能排除。仅论文名称、年份、分辨率或速度不同都不是独立性证明。

1. **YTSegments**：保留固定作者仓库及云盘URL；在将来的独立取得阶段，先取得作者确认的可读目录/替代公开地址、明确数据条款和原始100条清单，再下载完整媒体及其对应标注。不要拿废弃源码树或第三方不明镜像当完整原版。网络失败不触发本轮重复下载/替代集合。
2. **ActCount**：先通过合法机构访问或作者公开材料确认章节中的数据发布说明，取得作者确认的172条清单、媒体入口、划分和数据许可；若需要联系作者，由用户另行明确安排。付费论文可读不等于数据已发布或已获许可。
3. **取得后、任何评分前**：冻结每条video/source-ID、原始URL、裁剪时段/帧坐标、媒体SHA、计数及标注版本；核验100/172全人口和缺失账本，与既有四域全部训练/验证/测试清单做源级对照。保留非人类、长视频、零计数或冲突条目，不能按模型分数删选。
4. 只有资格确认后才另行冻结完整评价协议、比较方法、指标/舍入、来源暴露说明及预算。当前21项和357条baseline评价不增加任何新输入。

机器证据：AVAILABILITY_PROVENANCE.json及official_metadata.json。尚无可报告的新增数据集完整下载数、实际有效视频数或性能结果。
