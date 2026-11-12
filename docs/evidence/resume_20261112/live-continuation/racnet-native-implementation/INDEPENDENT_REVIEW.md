# RACnet native implementation: independent engineering review

2026-11-12. **Verdict: GAPS; same-family provisional.** One P2 resume-interface repair is needed before formal extraction. One P3 timing-output omission should be closed before completion acceptance. No additional scientific-semantic mismatch was found in the reviewed paths. This is a static engineering review, not a model-performance or GPU-readiness verdict.

Requested reviewer `gpt-6.1-sol/xhigh`, fresh session `/root/racnet_native_engineering_review_20261112`. Actual model/effort are unattested and remain null. The pinned research-implement-feature skill and reviewer independence, scope, routing, backend and tracing references were read. The frozen prompt and its scope restrictions govern this report.

## Findings, reported together

**R1 [P2] Raw QUVA manifest changes when extraction outputs appear.** `scripts/external/racnet_native/native_inputs.py:159` serializes QUVA rows from `prediction_rows`. Initially those rows have no `sha256`. When `prepare` sees a completed QUVA marker, line115 adds the generated feature SHA; line159 then includes it in `quva_raw_inputs.json`. `native_extract.py:68-84` binds the entire raw-input file and compares it with the original `QUVA_BINDING.json` before reaching per-video reuse.

The supported sequence `prepare -> extract some complete videos -> prepare after new assets -> continue extraction` therefore rejects unchanged raw inputs at `native_extract.py:84`. The required final `prepare` after complete extraction likewise makes a subsequent idempotent extractor entry fail. This is a static reachable-path defect, not an observed server/GPU failure. Independent author acquisition and the documented metadata refresh make this a relevant boundary.

Minimal repair: construct the extraction manifest from raw input fields and destination paths, excluding the generated feature SHA. Keep the feature SHA in prediction inputs. Reuse the existing binding implementation; add no binding machinery and change no science. One CPU-only metadata fixture can show that a completion receipt updates prediction qualification while leaving extraction inputs identical. No full-data/model rerun is needed.

**R2 [P3] Completion evidence omits total timing.** `native_engine.py:270-278` stores per-video inference/total seconds, but `METRICS_357.json` and `COMPLETE.json` at lines307-311 contain no aggregate timing or observed invocation duration. `native_extract.py:141` similarly embeds per-video seconds without a total. The frozen `evaluation.outputs` explicitly requires per-video and total timing. This affects cost evidence, not count/metric correctness.

Minimal repair: add clearly labelled sums of retained per-video times and elapsed time for the current invocation, distinguishing reuse count and retained-output sums from invocation wall time. Do not rerun completed videos. A metadata-only aggregation fixture is sufficient.

## Correct within the reviewed scope

The actual source preserves the native model, GenerateGT, peak decoder and start-SSE plus masked tReCo loss. Feature-length masks, zero padding and same-device movement match the intended bridge. Training remains full755, seed2333, batch16/fullT, native global-RNG RandomSampler, Adam/MultiStepLR and native AMP; validation/final are FP32.

Selection is correctly restricted to epochs11..199, native prominence0.3 and nine equally weighted batch means. Strict-less MAE updates retain the earliest tie; OBO is archival only. Inference metadata excludes labels and score reads the separate scorer file after all357 outputs persist. RC152 includes the zero-count row; UCF105 uses the frozen inclusive ROI adapter. Common scalar and native RC GT remain distinct. Final inference uses the original scaling decoder once at0.2; finite constants retain count0, and nonfinite model probabilities fail.

Checkpoint payload/commit ordering, initial/first/latest/best retention, full state and RNG capture, explicit epoch-boundary restoration and replay journaling are coherent statically. Existing CPU state tests retain their own limited evidence. No actual GPU restoration equivalence or resource fit is claimed. Formal ownership, source/review/publication, lane and storage gates precede CUDA work; real OOM/nonfinite failures do not trigger a changed batch/T/method retry.

## Evidence and limits

Read the frozen contract/decision/assumptions, all candidate files, SPEC/ASSUMPTIONS/BUILD_NOTE/manifest/diff, original native training/model/test/loader/GT/decoder/extractor/config files, all105 frozen UCF ROI mappings, and server CPU evidence. Existing CPU receipts cover imports,886 targets, decoder/padding,183 strict backbone keys and state/RNG primitives without CUDA initialization or neural forward.

The supplied readiness snapshot is **PENDING**:4/61 author objects,170 usable author members,973 required author members missing and0/100 QUVA outputs. These are expected pending inputs, not failed CPU primitives. Do not repeat completed CRC/finite, decode, whole-data or model checks to satisfy this review.

Formal QUVA extraction,200 training epochs,189 validation passes, batch16/fullT fit, CUDA/AMP behavior, GPU continuation and full357 output/scoring remain unexecuted. Actual lane/resource release is also pending. No scientific redesign, extra trajectories, matrix expansion or current-job review was performed.

No stale assumption-ledger row was found. Two interface assumptions are undeclared: extraction continuation assumes `prepare` has not refreshed generated-output fields, and per-video timing is treated as sufficient total-cost output. No additional semantic assumption was found (`semantic_undeclared=0`), with provisional assurance.

## Source identity and commands

Contract SHA `77c3d84443fb57415db3b2ed7416b3b16af5f8e8af614c41a05812802e7f8de8`; official RACnet commit `51b8e3f2decdeb10dbc7fd506a93a9f7d2d4d4fb`; implementation base `59bec947fbe7a6afc63efe847b868c6b355df0c6`. The full candidate/environment/vendor SHA map is in the accompanying raw JSON, copied from the frozen source manifest and matching server CPU identities. No independent local hash/computational validation was run.

Commands were static `Get-Content`, `Get-ChildItem`, `rg`, and JSON field display for all105 ROI rows. No server command/check was necessary. Only this Markdown and `INDEPENDENT_REVIEW.json` were written, exclusively in this E: review packet. Candidate source, Git, controls, global status, processes and jobs were untouched; no downloads, installations, model forwards or GPU actions occurred.

| Acceptance requirement | Outcome | Evidence |
|---|---|---|
| Focused primary-artifact engineering review | Passed within static scope | Actual code, original source, contract and existing receipts |
| Additional undeclared scientific semantics | None found; provisional | Native interfaces and selection checked directly |
| Extraction refresh/resume | Not met | R1 |
| Total timing outputs | Partial | R2; per-video fields exist |
| Complete input and formal workload readiness | Pending | READINESS and unexecuted GPU branches |
| Cross-family accepted assurance | Not established | Requested-only identity, same-family provisional |

Retain this session for one targeted followup after concrete repairs. The existing deterministic checks need only affected, bounded closure.
