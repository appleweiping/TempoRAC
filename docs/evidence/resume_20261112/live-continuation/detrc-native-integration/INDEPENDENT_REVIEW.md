# DeTRC native prelaunch review — initial verdict

Reviewed 2026-11-12 18:40:58 UTC in fresh session `/root/detrc_native_prelaunch_review_20261112`. Requested native `gpt-6.1-sol` / `xhigh`; actual backend model/effort are unattested. This is a same-family, provisional engineering review. Overall verdict: **gaps**, with one concrete recovery defect. No scientific outcome is accepted.

The reviewed contract is `43b5fb151a7b3dc4471f14cf2fe3f229e9efb8c02b87e61fbdc5614fe1094570`, independently read by its existing digest. All eleven driver files matched the final PUBLIC_ALLOWLIST identities. Official source is `538abfcd24e0a81b383ca81862ff4360a2389c97`; prior candidate manifest is `3df586d987ba140f3705501cdd18ed0d5e6198d8208b67baa0024cc1b45eb34c`. The pinned ARIS stage is `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`.

## F1 — H5 interrupted-commit recovery fails (P2, interface)

Site: `scripts/external/detrc_native/produce_features.py`, lines 21–23, 34–35; related capacity estimate at line 13.

Assembly fsyncs `.h5.pending`, renames it to final H5, then computes the final H5 hash while constructing its identity sidecar. If the existing budget timeout or a process interruption occurs between the rename and sidecar commit, the final H5 remains without its sidecar. The next attempt sees `dest.exists()` and unconditionally reads the missing sidecar at line 22. It raises `FileNotFoundError` and can never reach `FEATURES_COMPLETE` through the documented resume path.

This is substantial work in an explicitly supported interruption path: RC-train H5 payload is `645860 × 2048 × 4 = 5,290,885,120` bytes. Hashing it after the rename is not a millisecond filesystem race. Completed per-video pickle evidence survives, but the population assembly cannot automatically finish from it.

Smallest repair: reuse final H5 only when its completion sidecar exists and validates. With no sidecar, consider the assembly uncommitted and replay it through `.h5.pending` from committed, validated pickles, then replace/commit normally. Use the same committed criterion in the missing-H5 space estimate. Preserve JPEGs/pickles and the frozen numerical recipe. One narrow server-only fixture of this interrupted state is sufficient; do not repeat whole-data or earlier prerequisite checks.

No other actual defect was identified. There are no additional undeclared assumptions or stale ledger rows, and no undeclared semantic choices. F1 violates the already declared recovery intent.

## CPU-production gate

**The CPU JPEG implementation has no identified code blocker. Hold this particular freeze for the bounded F1 repair and refreeze before its first release.**

The original OpenCV/VideoReader BGR, short256/int resize and default JPEG path are retained exactly, including the author's dimension-variable convention. Inclusive UCF ROI becomes JPEG1..T. Conservative1244, required1243, supervised755 and final357 populations remain distinct. The launcher hides GPUs, uses at most first2 allowed CPUs, one producer, nice10, ionice2/7 and one numerical-library thread. CPU JPEG production does not probe, allocate or wait for a GPU, and does not mutate other jobs or old caches.

The per-video pending-directory/IDENTITY commit supports reuse of completed videos and replay of incomplete videos. Capacity stops preserve64GiB derivative and128GiB free-reserve requirements without thinning/deleting frames. The RACnet reservation code matches the actual acquisition producer: CONTROL `objects[id,bytes]`, STATE `objects[id].coverage`, durable half-open byte intervals. It unions overlaps, reserves `128629792634 − durable covered bytes` in addition to128GiB and refreshes within60seconds. Sparse logical size and network traffic are not used as durable coverage.

The reason to fix F1 before CPU release is the current common binding: JPEG receipts include the entire RUN_CONTRACT, and release validation includes every driver file. Changing the downstream driver/contract after producing JPEGs would stale those completed JPEG bindings. This is not a CPU numerical defect or a requirement to wait for idle GPUs. After the one repair/refreeze, root release and actual current capacity are sufficient to start full CPU production; no repeat full-metadata check is requested.

## GPU/formal-run gate

**Blocked by F1 and by actual inputs/resources/runtime that remain pending.**

The added TSN extractor hooks change persistence only. Original transforms, interval1, center crop256, FP32 T×2048 and internal batch200 remain. The training gate requires all native feature populations and their shapes/dtypes/source/backbone identities. The fixed80epochs/batch64/seed0/FP32/AdamW/LR15/60 recipe is preserved. Native130 post-SoftNMS validation OBO with strict improvement/earliest tie selects one weight; RC131 common diagnostic does not select.

The label-free adapter calls the original testing-window construction and padding/masks, retains fixed64-row native-length batches across resume, and supplies empty(0,3) structural GT. Inference opens feature/length identities and checkpoint, not scalar labels. The separate CPU scorer opens scalar labels only after the GPU child exits. Original SoftNMS sigma.1/top30/floor1e-4/count>.2 is retained. Native RC130/151 cycle diagnostics remain separate from complete scalar common populations, including the asserted51scalar/41cycle counterexample.

Exclusive24GiB-class GPU,32GiB available RAM,16/30/2h phase and48h total allocation limits, persistent budget accounting and own-process-group termination are explicit. No AMP, microbatch, operator, sampler or epoch substitution was introduced.

Still unexecuted: real1244-video JPEG production; TSN extraction/native H5; actual first model/kernel dispatch; batch64 FP32 memory and the original extractor's whole-video device-input memory; throughput and cap fit; CUDA recovery; complete80epochs/selection; RC131/full357 predictions and all numerical/scientific outcomes. No performance, significance or SOTA conclusion follows from this review.

## Acceptance evidence

| Requirement | Classification | Evidence and remaining boundary |
|---|---|---|
| Official preprocessing/ROI/populations | Partial | Original builder/extractor/dataset and new source agree; raw CPU receipt covers full metadata and JPEG fixture equality. Real production pending. |
| Durable reusable production | Partial | JPEG/video and pickle commits present; F1 breaks interrupted H5 assembly. |
| Label-free prediction/scorer separation | Pass, static only | Distinct input files/process path; original model target handling read. No inference executed. |
| Native selection/postprocessing/metrics | Pass, static only | Original config/dataset/SoftNMS, candidate durability and external scorer read; existing primitive receipt. |
| Independent CPU/resource interface | Pass, static only | CPU launcher, common/rawframes source, actual RACnet writer schema and resource-delta receipt. |
| Full80/final357 completion | Partial | Frozen plumbing present; F1, actual inputs/resources and all real outcomes pending. |

CPU_INTEGRATION_CHECK is explicitly older than the resource increment. CPU_RESOURCE_DELTA_CHECK records the final contract and exact changed common/prepare identities; it inherits earlier primitives rather than pretending they were rerun. This is correctly declared, not a stale PASS for the final whole program.

## Read scope and actions

I read the project/pinned policies and complete implementation stage with reviewer independence/scope/tracing references; all eleven driver files; final contract/ledger/manifest/diff and both raw CPU receipts; relevant integration candidate dataset/model/window/padding/postprocess/config/durability source; prior raw diff, original JPEG/extractor/config/guide, prerequisite CPU/build/config receipts and frozen route; actual RACnet acquisition coverage writer and count metric source. Executor REPORT/BUILD_NOTE characterizations were not used as a basis for conclusions.

Commands were local static Get-Content/Select-Object, rg inventories/searches and Get-FileHash for existing identities. No SSH/interface checks, repeated tests, real video decoding, model loading/forward, GPU probes/dispatches, production, source/control edits, process signals, downloads/installations, Git or shared status/source_sync changes occurred. Only this verdict JSON/MD was written to the authorized E integration packet.

Available in this same session for one bounded F1 fix and affected-interface followup. Preserve this initial report and record the followup and new frozen identity separately.
