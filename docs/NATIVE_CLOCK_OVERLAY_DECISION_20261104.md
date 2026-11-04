# NB1 clock overlay 窄幅决定（2026-11-04；一页）

**采用独立不可变 `temporac.native-clock-overlay.v1`。** 全raw与原decode完成后，一次性冻结全977视频的clock选择清单，再做完整NB1 build；不重跑pose/decode，不改raw原件、当前6个桥接文件或健康extractor。请求Astra/xhigh，backend未attested，OpenAI同族设计 **provisional**。

| 验收项 | 当前状态/依据 |
|---|---|
| 说明缺PTS机制 | 通过：extractor `extract_native:734`只读一次audit，`:739`从快照取item；`attach_clock:358–367`默认source_frame_only |
| 真实时钟与原观测分离 | 设计通过：overlay只更换有效时钟视图，pose/masks/身份/源帧轴不变 |
| 全人口物理时间可用 | pending：须实际overlay+census；未补足时仍partial，不能把frame fallback称物理时钟通过 |

**接口。** 后续另增 `native_clock_overlay_v1.py`：`build_clock_overlay(raw_json,raw_npz,decode_row_bytes,clock_csv)->OverlayV1`；`resolve_clock(feature,overlay)->(NativeFeatureV1,provenance)`。JSON绑定schema、media_key/split、raw JSON/NPZ SHA、原始decode JSONL行字节SHA、CSV SHA、media SHA、N、stream_index、正有理数time_base、原clock状态、generator SHA；NPZ仅 `source_frame_index:int64[N], pts:int64[N], pts_present:bool[N]`，禁pickle。缺PTS为零/false；不放best_effort/FPS推算值。CSV/row SHA由此次冻结计算，不能冒称旧audit已有。

**绑定与选择。** raw须complete且L=D=N；CSV恰N行、frame_index逐项等于raw的0…N−1；decode成功无parse错误，decoded_frames/annotation length、逐帧尺寸、选定video stream一致。raw媒体SHA=decode prior_sha=冻结media清单SHA，decode前后stat稳定、无已知媒体变化。`audit_temporac_decode.py:175–176`的prior SHA是复制值，`:237–244`检查stat；须保留这个限制及raw逐视频SHA来源，**不能升级成decode时重哈希或逐帧像素等价证明**。沿用native连续呈现顺序契约；有流/顺序歧义须拒绝，不能只凭同帧数推断。`:223`的`physical_clock`来自best_effort，不得充当PTS。

overlay须与raw全部原present PTS及rational单位一致，不覆盖冲突。raw已完整且一致则no-op；否则使用整个有效CSV clock视图。`resolve_clock`仅替换`pts/pts_present/time_base`，永久保留原clock状态/哈希，在preprocess/cue/target之前解析。missing、重复、倒退PTS **保留并按NB1断段**，不排序/合并/插值，也不为绕过缺陷退回frame模式。完全无真实PTS且无冲突才保留source_frame_only。

**必须拒绝overlay并留账：** 未完成/failed/alignment_unresolved raw或decode；SHA/人口/长度/索引/尺寸/流绑定不符；CSV缺列、坏整数/越int64、非法time_base、parse错误；原真实PTS冲突；呈现顺序未能确认；读取期间变化或多份clock冲突。重复/缺失PTS是支持缺陷，非绑定错误。绑定错误不得静默fallback或发qualified回执；该视频和全部slots仍留失败行。

**后续CPU交付。** 当前6-file review继续；新文件独立测试：补PTS时非clock数组字节不变；绑定/原PTS冲突拒绝；缺失/重复/倒退原样且NB1断段；仅best_effort不升级；幂等及overlay变更使派生产物失效。新build ID绑定overlay清单SHA，不复用旧fallback的cue/target/feature。全977清单记raw-retained/overlay-attached/unavailable/rejected，汇总PTS帧、相邻时间边、可执行run/秒覆盖，保留2364身份。**不增加pose观测，不解除target/loss/metric/训练门槛。**

审读：extractor SHA `d972057c45f6e21f4d36638fabb21daa3bed66884ae9c4ba9b9c936700232da1`、decode源码、NB1/current_status；阶段前全文重读固定pin的experiment-plan及四份output支持。仅写此决定，无标签/模型/评分/GPU/服务器操作或源码修改。
