# TempoRAC CVPR 接续协议 R1

日期：2026-11-04。任务授权：在现有 TempoRAC 主线上自主推进，采用固定版本 ARIS，日常 executor/reviewer 使用 Sol，重大科学方向和设计使用 Astra；新增实验与产物仅放 E 盘或指定服务器，定期同步 GitHub。

## 1. 接续来源与边界

- Canonical repository: https://github.com/appleweiping/TempoRAC
- 继承分支：`codex/aris-fullrun-icassp27`。
- 基准提交：`3e9ae828979bb7538da733835d286fcb395ccfbb`。
- 本轮分支：`codex/temporac-cvpr-aris-20261104`。
- 活动目录：`E:\ICASSP_TempoRAC\TempoRAC-CVPR`；旧 E 盘导出及旧工作树完整保留。新工作副本继承相同提交，不是另起研究路线。
- 最近可信的本地进度为 `docs/TEMPORAC_SYNC_STATUS_20260930.md`。157 项 TempoRAC 单测和 1179 项全套单测是历史工程记录，本轮需另记实际验证，不能冒充新结果。
- 用户本轮补充：目标是**新服务器**。已成功 SSH 登录并只读确认 4×RTX 4090；旧主机路径和环境不套用。四张卡当前都有作业，先建立 TempoRAC 独立部署目录并核查可复用数据，不把登录成功说成实验已运行。
- 另发现旧 CVPR pre-results 稿件工作树 `E:\Project\rac-paper-writing`，提交 `26e5b7176f1f7c678391163c3278b5c390d54115`。它可供后续复用叙事和匿名排版，仍无新方法结果；不必从 ICASSP 五页模板重新起稿，也不能将其视作已完成实验。
- 当前自然数据仅为 110/51 videos、268/134 identities 的 partial-cache / GT-bbox-assisted / supplied-track pilot。27 个正式训练尚未运行，主表与消融为空。X0-only 的失败及 v15 的探索性覆盖结果继续保留。

## 2. 本轮验收表

| 项目 | 本轮起始状态 | 可核对通过证据 |
|---|---|---|
| SSH 与资源 | SSH 已通过；4×4090 当前繁忙 | 已取得 GPU/磁盘/进程清单；TempoRAC 实际运行资源仍待调度 |
| 版本与工作流 | 已继承 3e9ae828；ARIS 安装中 | ARIS exact SHA、完整读回执、项目级安装与角色配置 |
| 文献与创新性 | 旧工作可复用，新增文献检索进行中 | 论文正文/官方代码证据、逐主张近邻表、独立审查；摘要级阅读单列 |
| 数据完整与独立性 | 未达标 | 全量合法源、source-video 去重、train/dev/独立评估 manifest、缺失账本 |
| 时间坐标 | 旧 v2 将物理时钟替换为行号 | 显式 source/output 时钟映射、观测/插值/外推支持记录、边界往返测试 |
| 自然伪目标 | 未达标 | 每序列分支来源、有效监督区、全序列事件/时长覆盖、几何伪计数器误差 |
| 主性能与强基线 | 未达标 | 完整实验、三 seed、原始预测、同口径统计；不得用不同任务旧结果填格 |
| 三组件贡献 | 未达标 | local cue、专家、连续重建对应移除/替换；训练与推断消融分别注明 |
| Observation/超参/效率 | 未达标 | 完整 dev 分布动机、超参曲线、稳定性、含前处理与不含前处理的资源统计 |
| 六数据集×七框架偏好 | 未达标，尚无合格完整矩阵 | 真实数据集与比较范围冻结；不把六扰动或六子集当六数据集，不静默缩减为已通过 |
| Loss 少于三项 | 旧候选有冲突，未达标 | 明确优化项及版本化删改；不通过重新命名隐藏实际项数 |
| CVPR 稿件 | 旧 ICASSP 与旧 CVPR pre-results drafts 均可复用 | 实验完成后形成 CVPR 匿名八页正文、LLM 声明、引用审计、逐页渲染含补充 |
| 三方同步 | 本地/GitHub 可推进，服务器已认证待部署 | 同一代码提交 + 配置/数据/结果清单与 SHA；服务器未达不得称三方完成 |

## 3. 保留的科学主线

继续研究 identity-local 计数：共享编码器、slow/medium/fast 响应专家、detached 局部 NUDFT cue 与软路由、正分母 NOLA、每身份仅解码一次。模型输入不得含 count/period/density/boundary 标签；外部 pose/detector/tracker 监督与 GT bbox 帮助据实披露。

WARP-PHASE、v46/v62/v63 和单人 PAMS/MotionBERT 数值属于历史或其他路线，不能恢复为主线或移入 TempoRAC 表格。已被开发过程观察的 validation 只作 development，不能改名独立 test。原始负结果和所有样本保留。

## 4. 一次性流程修订

用户本轮授权以实际科研继续推进。旧 A005 的 FD/ptrace/token 等安全证明、百万字节规范续审不再作为本轮科学诊断或训练的先决条件；旧文档保留，未被追认通过。以 feature-only 输入、清晰 split、只读评估标签、实际过程隔离和可恢复运行日志保证必要边界。

本修订不降低数据、统计或效果要求。旧 coverage 阈值失败仍是失败；新诊断应报告完整人口、时长和事件覆盖，而非调低阈值制造通过。恢复作业只能复用冻结预测/配置，不能借重启重新选参。基础设施等待期间推进 CPU 诊断、文献、基线准备；不得不断重问同一审批或无意义重跑已通过检查。

## 5. 下一执行顺序

1. 完成 ARIS 版本锁定和全文读回执，保留本次科学设计审查的真实模型身份及同族 provisional 状态。
2. 补文献正文阅读与逐主张查新；重点核对 MultiCounter/MultiCounter+、PAMS、TWCRAC、TReViS，以及具备合法可运行代码的对照。不因有近邻就否决现有路线，也不在证据不全时宣称新颖性通过。
3. 增加独立的时间映射/支持审计模块，不改写冻结 v2 输出，不把新审计宣称成已修复模型。
4. 已登录服务器，继续核对正在运行的健康作业、数据路径、剩余资源。TempoRAC 与 STaR-Body 均为高优先级，分开目录、进程和 GPU 配额，本聊天仅推进 TempoRAC，不中断 STaR-Body。对其他已识别、归用户控制的冲突作业，仅在 TempoRAC 完整输入就绪且保存恢复信息后安全调整；不杀不明进程、不删除其他项目。
5. 建完整数据/源视频清单并审计 v15 各分支。v12/v15 的 teacher 输出未用于主要几何目标分支，v9 则有依赖；先做几何伪计数器完整 dev 误差与监督支持诊断，避免为无效依赖重新训练 teacher。
6. 根据该诊断由 Astra 固定必要目标/输入修订，Sol 实现并接受独立代码审查。完整输入就绪后直接运行首 seed 的全数据 canonical 与 matched control，再并行其余 seeds、强基线与消融。CPU 接口检查不替代完整实验。
7. 统一收集预测、样本量、sample SD、video/source-cluster bootstrap CI、多比较处理、机制图、超参和效率。最终稿仅在实验与 claim audit 完成后形成。

自然主指标候选为官方 MultiRep AvgMAE，完整报告 AvgOBO、Period-mAP/AP50/AP75；需读官方 scorer 并冻结确切公式及 interval 坐标后才能启用。候选指标现在不是已完成冻结。不得将伪峰间距未经验证转换成合法 period intervals。

## 6. CVPR 与同步规则

官方当前给出 CVPR 2027 enrollment 2026-11-10、paper 2026-11-16、supplement 2026-11-23，均 AoE。八页正文，引用和 LLM Usage Statement 按官方范围另计；匿名、内容自包含，作者承担全部科学责任。来源：

- https://cvpr.thecvf.com/Conferences/2027/Dates
- https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines

本轮不自动投稿。用户的三线表、真实最优值加粗、深红 improvement、Method 约两页和所有 Word/PDF 逐页渲染偏好继续保留。服务器仅在项目结束并完成备份校验后清理，当前没有清理授权。

GitHub 仅提交代码、规范、非敏感读回执和可发布证据；不提交密码/私钥、原始视频、受限权重。同步以本轮分支为目标，不强推、不自动合并旧 main。服务器同步前检查 dirty tree，使用独立工作目录和 exact commit，保护已有结果。

周期接续在每次唤醒先读本协议与最新 run state：完成一项可验证交付再记录进度；不要复述无变化状态，不替换 reviewer 原始判定，不将“调度已创建”等同实验完成。

当前已创建本聊天每三十分钟一次的接续任务 `temporac-cvpr`；只有客户端实际调度且所需执行条件可用时才会继续。服务器认证已完成，部署、完整数据与正式实验仍需各自完成回执。
