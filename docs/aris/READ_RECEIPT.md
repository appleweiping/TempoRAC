# ARIS full-read and integration receipt

- Upstream: https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git
- Exact commit: f76ec8d6b77792e2242cca049e08d0cbb7d54e19 (detached)
- Actual source: E:\ICASSP_TempoRAC\aris-work\aris-pinned
- Active project: E:\ICASSP_TempoRAC\TempoRAC-CVPR
- Full-read source files: 46; stage skills: 20.
- Preparation only: no experiment, review verdict, new research claim, remote deployment, automation, or push is represented by this receipt.

## Acceptance evidence

| Requirement | Result | Evidence |
|---|---|---|
| Read E:\AGENTS.md and user policy | PASS | Full E:\AGENTS.md read at start; precedence captured in WORKFLOW.md |
| Read actual README/install guide before use | PASS | README/AGENT_GUIDE full ranges confirmed by project_inventory; SETUP_GUIDE and Codex README read in full |
| Full requested stage skill reads | PASS | All 15 explicitly named skills (including both paper-write and paper-writing) plus 5 execution/evidence dependencies, table below |
| Fixed source revision | PASS | Full clone URL, exact detached SHA, clean upstream status |
| E-only project installation | PASS | exFAT requires copies; INSTALL_VERIFICATION.json hashes every installed file against pin; no upstream global installer applied |
| Reviewer/executor separation | PASS (configured, not executed) | Routine Sol; design/major-interface Astra; fresh isolated review; same-family provisional |
| Preserve scientific method and history | PASS for this integration | Only workflow/config/docs copied or written; historic root lock archived, existing evidence not replaced |
| Full CVPR research/quality gates completed | NOT EXECUTED | This is workflow setup; existing/new scientific results require separate evidence |
| New-server deployment and data migration | NOT COMPLETE | New host; authentication, account, GPU, environment, paths and data unverified |

## Full nontruncated reads

SHA256 covers local on-disk bytes at the pinned checkout. README and AGENT_GUIDE were read by the project_inventory agent; all other rows by aris_workflow. An early oversized README preview and installer preview were truncated and were not counted as a full read; README was subsequently read across complete non-overlapping ranges. The installer code was inspected for path/scope behavior, not claimed fully read. The installation uses a separate E-only copy script instead.

| File relative to pin | Lines | SHA256 | Reader |
|---|---:|---|---|
| `README.md` | 1819 | 04a20c46220d11b9f66b727be336108db23fd7f45645c8f75ef42d19c5c928e8 | project_inventory |
| `AGENT_GUIDE.md` | 275 | bd69ae118166370f261e5619e3f19b9d355f70a7d3647c97ff75f4871cff2e17 | project_inventory |
| `SETUP_GUIDE.md` | 279 | 50a9d67d000462806d20b3192c2f89416039b1968696270c3dc69554ef947ae6 | aris_workflow |
| `skills/skills-codex/README.md` | 182 | ffb30a7eace7b0e855f9d47e8a88b6a46dc9b8363a649f5ba881400b49c7113c | aris_workflow |
| `skills/skills-codex/research-pipeline/SKILL.md` | 370 | 8caec65ac3988cd619dcdada3f0e72069536e01c94bf91fe853bf799567eca9d | aris_workflow |
| `skills/skills-codex/research-lit/SKILL.md` | 501 | 87844f4afb7b71a4cdd71b4073b11c0aee3d22e5cfa4f00db1b84c2469059fd4 | aris_workflow |
| `skills/skills-codex/novelty-check/SKILL.md` | 141 | 65e1684942c2de5f9791f6e2b1025b320a8a4d7b41032a1991b73bb91719082f | aris_workflow |
| `skills/skills-codex/experiment-plan/SKILL.md` | 249 | 61812df053fd8717cccb248fcabec3863f930eaedba03463bc059294dcc3863a | aris_workflow |
| `skills/skills-codex/research-implement-feature/SKILL.md` | 419 | 33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8 | aris_workflow |
| `skills/skills-codex/run-experiment/SKILL.md` | 236 | 8325128eab14c7f82213d44c8e78e98f44ca680baa9c78f3bee8766f8e9f464a | aris_workflow |
| `skills/skills-codex/monitor-experiment/SKILL.md` | 98 | 66a42b21a7ec5a9c4ee2d4da6babfd533764d6c4188dcc178d5bcbd1d8a31072 | aris_workflow |
| `skills/skills-codex/analyze-results/SKILL.md` | 45 | 1a5f798558018fe6c8faa8b9997ac4cd1076aceb73cfb5bdc261e5de5727f7e2 | aris_workflow |
| `skills/skills-codex/research-review/SKILL.md` | 165 | 17505d5982c82e821124de898b0beb7de4a98ecc4c1cd173b8687b74ace9d4c5 | aris_workflow |
| `skills/skills-codex/auto-review-loop/SKILL.md` | 548 | 1ddf0c256958151ed330852044c0cf2d4595fa8842ef79ff71dfc7ac590402ec | aris_workflow |
| `skills/skills-codex/paper-plan/SKILL.md` | 292 | 715f69f3e6541256144fe29552c9cbab219d79629114f3fc49ab0d09be265078 | aris_workflow |
| `skills/skills-codex/paper-write/SKILL.md` | 503 | 31e08c10b6044776bace94c6e09f58ffd3fc9bbab04bfa888bf42784d6ede98c | aris_workflow |
| `skills/skills-codex/paper-writing/SKILL.md` | 759 | 34adf47ad317849ed8ba3d7610d1dfc79b0a8ee930594f415c15857f762f3374 | aris_workflow |
| `skills/skills-codex/paper-claim-audit/SKILL.md` | 363 | 583f275e2ed1d4c2f1d6fe4cb7e9733322f2b2c11c8948aa8d2fa66cfe910af2 | aris_workflow |
| `skills/skills-codex/citation-audit/SKILL.md` | 517 | 3f140b0fd75152b239c4fc2ec88c12828efab5da0fc8cbcd1710bafafbe36385 | aris_workflow |
| `skills/skills-codex/experiment-bridge/SKILL.md` | 369 | 2b7aaab38753f3aeb8a85d2e6bd77d8eeeb29a8850dc0637102567fd0ed0c1e5 | aris_workflow |
| `skills/skills-codex/result-to-claim/SKILL.md` | 261 | 2e67d8bc7eb178aba18f3a3b646b59d3ca25836a42154acabe35dfa705f9f0ec | aris_workflow |
| `skills/skills-codex/training-check/SKILL.md` | 94 | 94bcb09c9b2eae90f2bcfeb98d652dd72dbe00d42be47486650397ad5be69c44 | aris_workflow |
| `skills/skills-codex/ablation-planner/SKILL.md` | 126 | 8383bc55d53adb29a36e07cc7da560337ddb32140e268fe737d4c36f94110e7b | aris_workflow |
| `skills/skills-codex/experiment-audit/SKILL.md` | 300 | 355a22aa16bbd08ca78cd3e83784d77606fe4405790969273426f5cfdfec0da3 | aris_workflow |
| `skills/skills-codex/shared-references/reviewer-backends.md` | 156 | e0aea16315faaec6ea468272aed7bf9d4a1b1d5261302a8ba357a9dce7bd9c33 | aris_workflow |
| `skills/skills-codex/shared-references/reviewer-independence.md` | 66 | 326a2ed68aa25dd5ff7fbb74b34fcce7a595eb27cd0daf565890c208076206db | aris_workflow |
| `skills/skills-codex/shared-references/reviewer-routing.md` | 166 | b0ce923454caf49e22742ae4de0bd9847e6f0a4d3c9ca95a1b48288cc1ea6619 | aris_workflow |
| `skills/skills-codex/shared-references/assurance-contract.md` | 167 | 05dfc6d1d68b6c79c6aaaebc910b27efa5d15a540ea3b8806f3c993176419f93 | aris_workflow |
| `skills/skills-codex/shared-references/evidence-precheck.md` | 75 | 5114b5a6c5e4efe733c572fab8aaee90d60f95d69b2926e2a297fa6eecac3993 | aris_workflow |
| `skills/skills-codex/shared-references/experiment-integrity.md` | 49 | 22150fbdc5867e7790c88f968bccb367079c2db4c190ddce980709aea7e76f60 | aris_workflow |
| `skills/skills-codex/shared-references/review-tracing.md` | 239 | fd1e524d1d78dd2dfb6cc51e404a303f98aabc40849bc16149676c8904814fa4 | aris_workflow |
| `skills/skills-codex/shared-references/integration-contract.md` | 373 | 0d6ec981e51b85da6d3c24d8fef4bbeb64c27fef87a57e1129413aff59f2ddd1 | aris_workflow |
| `skills/skills-codex/shared-references/acceptance-gate.md` | 346 | 0d1ef8b2d60932b9c9739bdaf955ae4c614b62d659d357440e2606433affdb0a | aris_workflow |
| `skills/skills-codex/shared-references/effort-contract.md` | 143 | d4a53faf13d19d5d7f844d279b46ad5528de6572b07531cee4d49f4f405873e5 | aris_workflow |
| `skills/skills-codex/shared-references/compute-env-contract.md` | 163 | 27aee2eee75a705a7d3ed72086b05ae2aa499104489376596ecb5b683807eeb5 | aris_workflow |
| `skills/skills-codex/shared-references/resumable-runs.md` | 125 | 60cb895918c0ff394a46a27f69140b84578b1bc7457d8e95c2f5c20c06ca42d2 | aris_workflow |
| `skills/skills-codex/shared-references/output-language.md` | 45 | 0f1447579bd7a2195fd4a074be70ae6345b5c368f1a70d327e89748bf5a378f9 | aris_workflow |
| `skills/skills-codex/shared-references/output-versioning.md` | 111 | de8e7ac23069de6c5d0c482d4cef117dd5a543e00b57e2dc6c62560a457290c5 | aris_workflow |
| `skills/skills-codex/shared-references/output-manifest.md` | 40 | e767083e0ef97a7adf7a4e8a574090067a59c0655d5aca422a3747cd32b6afde | aris_workflow |
| `skills/skills-codex/shared-references/venue-checklists.md` | 73 | aef7a0c396078747a5723b93b289c379aa1dda3f08540eca6d2d321264ed0ec8 | aris_workflow |
| `skills/skills-codex/shared-references/review-scope-limits.md` | 81 | e383dbaa57aed6f8fd2c2bb6bcaefc6806648126d11b21cb6c39eeeec76100f6 | aris_workflow |
| `skills/skills-codex/shared-references/writing-principles.md` | 539 | 23142c901d034ab0b640a7f2c1ee3899f273020d9bf911134ef06ebfad61e5d8 | aris_workflow |
| `skills/skills-codex/shared-references/citation-discipline.md` | 452 | 1c51dc936bbc3dbb17580148767734e0c642ad59516a861d9503b9767ca7043c | aris_workflow |
| `skills/skills-codex/shared-references/capture-antipatterns.md` | 89 | 0339dc801cd1cf93d7d1e284c3b7034f49af83c7f06473cf1e922a9614de5e5e | aris_workflow |
| `skills/skills-codex/shared-references/injection-hygiene.md` | 136 | c5a96d7b004bd1c1ae2ffb874bec8027b0f849cc1b8581c69fa6e2a9c078b7d8 | aris_workflow |
| `skills/skills-codex/shared-references/output-composition.md` | 98 | 79a72c7ea02102c460ae3ad55853c21a4054b9df93f377c29dd34555e6d3c3a7 | aris_workflow |

## Stage status and pending dependencies

WORKFLOW.md is the stage delivery/gate map. Every stage skill above is read and installed, but scientific stages remain not executed by this setup. Existing historical artifacts retain their historical status and are not revalidated by being present.

Conditional child skills are intentionally not installed/read until an authorized stage invokes them: idea-discovery, idea-creator, research-refine, experiment-queue, paper-figure, figure-spec, paper-compile, auto-paper-improvement-loop, proof-checker, kill-argument, render-html, research-wiki, integrity-forensics. Conditional references external-cadence.md, fan-out-pattern.md, skill-governance.md must be read before using those mechanisms. A later phase cannot silently skip required child audits: read and install its dependencies first from this same pin and append the receipt.

## Reviewer/source obligations

Read installed skill plus applicable pinned references in full at stage activation and verify its SHA against this receipt. Preserve raw reviewer prompt/response, requested versus actual identity, successful completion, exact input scope and session. Separate deterministic completion from scientific/semantic acceptance; neither score threshold nor same-family agreement proves SOTA, correctness, or publication readiness. User/project dataset/model/budget freezes, full-data evaluation, honest negative evidence, CVPR rules and all-page render checks override upstream generic defaults.

## README and agent guide full-read evidence
# ARIS README / Agent Guide 全文阅读回执

日期：2026-11-04。reader：project_inventory 子代理。固定源：`E:\ICASSP_TempoRAC\aris-work\aris-pinned`，主代理固定commit `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`。

| 文件 | 物理行数（PowerShell Get-Content数组） | SHA256 | 结论 |
|---|---:|---|---|
| `README.md` | 1819 | `04a20c46220d11b9f66b727be336108db23fd7f45645c8f75ef42d19c5c928e8` | 全文已读 |
| `AGENT_GUIDE.md`（仓库根，非docs目录） | 275 | `bd69ae118166370f261e5619e3f19b9d355f70a7d3647c97ff75f4871cff2e17` | 全文已读 |

README 240533 bytes。最初1–100及236–335尝试因超长release-note行产生工具截断，未将那些截断输出算完成；最终以下完整非截断读取覆盖每一行：1–40，41–65，66–135，136–235，236–280，281–320，321–385，386–535，536–685，686–835，836–985，986–1135，1136–1285，1286–1435，1436–1585，1586–1735，1736–1819。AGENT_GUIDE以1–125、126–275完整输出读取。无以hash/目录枚举替代阅读。已向aris_workflow发送本回执事实供其合并。

## 关键要求与接续选择

- README/AGENT_GUIDE是入口与routing，具体SKILL及shared-reference是行为规范；发生冲突以具体skill为准，仍服从用户当前明确要求。
- 完整Codex pack路径为`skills/skills-codex/`；旧claude/gemini partial overlays冻结，不应新装它们作为完整工作流。Windows支持官方PowerShell flat per-skill junction安装；manifest管理已安装项；不编辑junction目标。
- 主链W1 idea discovery/refine/plan → W1.5 experiment bridge（复用代码、独立code review、部署、收集）→ W2 review/fix → W3 narrative/plan/figure/write/compile/improve。已有明确方法和历史证据可从相应阶段接续，不要求推倒重跑W1。
- 审计五层：experiment-audit → result-to-claim → paper-claim-audit → citation-audit → kill-argument；proof-checker按理论需求与submission规则运行。声明必须来自真实数据，作者不自判实现/实验诚信。
- audit-class reviewer应fresh，传原始文件路径，不用fix摘要引导。真实模型family和已完成scope决定独立性；same-family永远只provisional。正常auto-review-loop自身可按skill保留会话，其独立nightmare验证仍fresh。
- effort控制工作深度；assurance控制最终审计门槛，两者独立。submission须外部`verify_paper_audits.sh`通过，编译成功和高分不等于可投稿。
- AUTO_PROCEED=true默认继续已授权步骤；不可把每一步变成用户确认。GoalPulse仅主代理在显式Goal/长自治授权场景按相应规则处理，普通安装不启定时器。
- 默认跨模型regular xhigh、deep audit ultra；用户显式模型偏好覆盖默认，能力回退不可因单纯timeout触发。writer可与reviewer独立选择；未知身份不可冒充跨family。
- plain-text artifacts与Research Wiki保留失败结果和路线历史；rendered HTML只是派生视图。helper经skill/scripts→.aris/tools→tools→ARIS_REPO解析。

## 与本次用户要求的差异及处理

1. README某旧表仍列venue ICLR、illustration Gemini，后续guide明确无静默venue默认且新skill可选backend。此次用户明确CVPR、现有TempoRAC，不能套旧ICASSP页限或默认ICLR。
2. 通用W1/W1.5含30min–2h小pilot、smallest sanity，W2默认跳过>4GPUh实验/最多4轮/更愿意reframe。用户E盘科研偏好明确完整合格划分、最少必要CPU检查、完整训练与基线；已授权完整工作不因通用小pilot默认自动缩成debug成果。正式预算变更仍需如实记录。
3. 通用W5是已成熟稿件的text-only venue迁移，禁止新实验。本项目旧稿无正式结果且用户要继续高优先级CVPR科研，不应选W5来锁死必要实验；接续W1.5/W2及最终W3更符合现状。
4. 通用README强调挑赢的contest并保留不利数值。用户要求主表不报非SOTA域又要求保留反例/完整比较，需显式说明范围与结果冲突，不能用隐瞒或删对照制造普适优势。
5. Windows后台必须hidden/CREATE_NO_WINDOW/windowsHide；任何上游示例若会弹窗或写C/D，按当前用户限制改为E盘工作路径与无窗口启动。
6. 旧TempoRAC项目锁是gpt-5.6-sol same-family provisional；当前主代理指定普通工作sol、深审astra时应显式记录新选择，不冒称旧审查等于新CVPR接纳。外部消息/发表/投稿不在本次盘点授权内。

本回执没有另行执行安装、skills、外部review或实验；指定skills全文阅读与安装由aris_workflow负责，其独立回执合并本条。
