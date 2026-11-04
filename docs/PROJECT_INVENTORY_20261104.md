# TempoRAC 接续盘点

审计日期：2026-11-04。角色：只读盘点；本报告写入得到主代理授权。未改科研源码、未启动实验、未推送。服务器实时检查由主代理负责；本报告将历史记录与当前本机证据分开。

## 仓库与接续位置

| 位置 | 状态/用途 | Git 证据 |
|---|---|---|
| `E:\ICASSP_TempoRAC\TempoRAC-CVPR` | 主代理从 canonical 仓库原样 clone 的新接续工作区，并非科研重来 | `origin=https://github.com/appleweiping/TempoRAC.git`; 当前 `codex/temporac-cvpr-aris-20261104`; HEAD `3e9ae828979bb7538da733835d286fcb395ccfbb`; 本次 tracked status clean |
| `E:\Project\rac-aris-fullrun-icassp27` | 已存在最新 TempoRAC 本地工作树 | `codex/aris-fullrun-icassp27`; HEAD 及本地 origin 对应 ref 均 `3e9ae828979bb7538da733835d286fcb395ccfbb`; tracked status clean |
| `E:\Project\rac` | PAMS/RAC 历史主 checkout，不能混成 TempoRAC 当前路线 | `agent/initial-reproduction`; HEAD `94fe7b2`; origin 仍使用旧名 `appleweiping/pams-rac-reproduction`; 两个 `recommend paper/*.pptx` tracked deletion |
| `E:\Project\rac-paper-writing` | 既有CVPR pre-results论文工作树 | HEAD `26e5b7176f1f7c678391163c3278b5c390d54115`，commit subject `docs: freeze CVPR pre-results paper package`；`.git`指向旧D盘；dirty未完成核验 |
| `E:\Project\rac-paper-writing-icassp27` | 既有ICASSP论文工作树 | HEAD `c33d1c5f604f094e000e1e92561a735d8610e6ef`，commit subject `paper: add delivery freeze manifest`；`.git`指向旧D盘；dirty未完成核验 |
| `E:\ICASSP_TempoRAC\TempoRAC-main` | 2026-09-25 导出，非 Git checkout | 无 `.git`; 含8月旧交接，不应覆盖9/30最新同步 |
| `E:\ICASSP_TempoRAC\TempoRAC` | 旧稿件与图资产 | 非当前源代码主线 |

迁移问题：`E:\Project\rac-aris-fullrun-icassp27\.git` 仍指向 D 盘。只读审计采用 `git --git-dir=E:\Project\rac\.git\worktrees\rac-aris-fullrun-icassp27 --work-tree=E:\Project\rac-aris-fullrun-icassp27 ...` 验证，没有修改旧指针。新 canonical clone 消除了对此失效指针的依赖。仓库重命名 `pams-rac-reproduction -> TempoRAC` 由主代理网络查验确认；两处当前 TempoRAC HEAD 一致。`rac`完整HEAD为`94fe7b2c624511436b09ea2abcf3e771928ba584`。旧paper-writing status耗时，由主代理指示中止本代理启动的只读进程；未把dirty状态误写为clean。

## 当前科研路线与不能混入的历史

权威起点：`E:\Project\rac-aris-fullrun-icassp27\refine-logs\FINAL_PROPOSAL.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`；近期状态以 `docs\TEMPORAC_SYNC_STATUS_20260930.md` 为先。

保留的主线是 externally supplied identity pose tracks 上的局部变速重复动作计数：shared causal encoder；三个共享 slow/medium/fast sigmoid response experts；detached irregular-clock cue 驱动 identity/window-local soft routing；概率融合；严格正 NOLA 分母重建；每个 identity 只做一次固定0.5阈值 connected-component decode。状态、窗口、邻接和归一化不能跨人混合。count/period/density/boundary evaluator labels 不进入学习。

旧 WARP-PHASE、PAMS-Literal/SSHead、MotionBERT/prototype calibration、v46 ExtraTrees 与 v62/v63 都是历史证据，不是本方法性能。尤其旧 `mae=0.087504, avg_obo=0.725` 不能填 TempoRAC 表。

## 数据与结果证据

最新公开同步状态文件说明：自然缓存仅 **110 train videos / 51 val videos，268 / 134 identities**；是 **partial-cache / GT-bbox-assisted / supplied-track pilot**。完整合格主表数据尚未建立。

- v44 train cache SHA256：`c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb`。
- v44 val cache SHA256：`4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251`。
- 8/18 frozen preprocessing 仅接受172/268 train、91/134 val；插值候选可覆盖268/134但当时未成为冻结协议。
- 9/30 v15 certificate-bound diagnostic 接受213/268 train、109/134 val；已依据val失败开发，只能探索性，不是独立门槛通过。
- 8/18 X0-only checkpoint 对134 val的诊断：raw MAE 21.6044776、NMAE 0.9915197、OBO 0、pred 0–4而target4–92。低X0训练损失没有转化为真实计数性能，不能进入主表。
- 9/30记录 `ruff check src tests scripts` 通过、`pytest tests/temporac -q` 157 passed/1 skipped、全套1179 passed/7 skipped。为历史工程证据，本轮未重跑；不能算科研达标。
- 27个正式training jobs仍未运行；旧计划3 seeds（20260815/16/17），3 teacher+24 response，93 A6000h，64GiB输出cap。当前正式performance、paired bootstrap CI与机制消融都缺。

## 稿件状态

`paper\README.md`（9/30）称未投稿 pre-results draft；既定ICASSP window已过，不能沿用旧会议模板作为CVPR合规证明。旧5页稿件是4技术页+参考文献页，含 `DRAFT -- RESULTS PENDING`。

`paper\evidence\method_manifest.yaml` 仍 `SYNC-REQUIRED`，绑定旧 `73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454` 单人 dominant-pose计数接口，proposed implementation_commit 为 null。`paper\sections\4_tables.tex` 主比较和消融均pending；reference delta0/0是定义，不是实测。需要实验完成后同步方法证据和稿件，而非先产最终稿。

## 服务器与后台（历史/本机）

旧记录项目根 `/media/lenovo/data2/pams-rac`；conda环境 `cja_py310`。公开 `.aris/compute/{ssh.md,provider-env.json}` 故意无用户名、密码、地址及key。本盘点未找到可公开报告的用户名，也未读取/输出秘密。

历史8/15资源为Ubuntu20.04.6、40逻辑核、125.4GiB RAM、2×RTX A6000 49140MiB、driver550.54.14、Python3.10.15、NumPy1.26.4、Torch2.5.0+cu118；当时输出盘96%已用、swap接近耗尽。9/30说明两GPU有其他项目占用。以上均非当前实时健康结果。当前TCP连接与远端目录、任务、资源由主代理独立核验。

本机CIM筛选Python/SSH/Node命令含`temporac`、`pams-rac`、`rac-aris`，未见相关后台进程；不能推断远端无任务。

## 接续优先项

1. 保留已完成代码/历史反例，从canonical同SHA接续；不要重新设计或恢复作废路线。
2. 对照服务器v15实际候选、cache完整性、失败identity和现有任务，形成可执行的自然数据与训练链差距；只做必需CPU工程检查。
3. 将CVPR目标、完整数据/主指标/公平基线与资源约束写成新验收表；用户完整实验和质量要求优先于旧短时pilot默认，但不得删除负结果、偷换数据划分或把val开发当独立test。
4. 独立审查方法/实现/评估后，满足完整有效输入即推进正式任务；不以重复小GPU smoke代替完整训练。
5. 实验后补齐主结果、组件去除、稳定性曲线、效率/复现、statistical证据，再同步和逐页验收最终稿。

尚未证明：完整主表数据就绪、正式三种子计数效力、6域SOTA及公平范围、所有组件贡献、CVPR当届官方格式、当前服务器容量。以上不能因为代码测试通过而标为完成。
