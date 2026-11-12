# RACnet targeted engineering followup

2026-11-12. **PROVISIONAL_PASS for the current candidate identity. R1 and R2 are resolved; no remaining or fix-introduced defect was found within this targeted scope.** The original review remains unchanged. This closes its focused engineering findings, without upgrading input readiness, GPU execution or scientific claims.

Same reviewer session: `/root/racnet_native_engineering_review_20261112`. Requested `gpt-6.1-sol/xhigh`; actual model/effort remain unattested. Assurance is **same-family provisional**, not accepted cross-family review.

| Original finding | Closure | Evidence |
|---|---|---|
| R1 [P2]: output SHA changed the raw extraction manifest | Resolved | `native_inputs.py:42-58,170` constructs raw extraction rows from fixed raw identity/length/path fields; generated feature SHA remains in prediction qualification. The existing binding now survives output qualification. |
| R2 [P3]: missing total timing | Resolved | `native_common.py:181-202` separates retained/reused/generated duration sums and current invocation elapsed. `native_engine.py:309-315` and `native_extract.py:143-145` persist these in completion artifacts. |

The one-shot server metadata fixture was inspected directly, including its source, raw JSON receipt and log. R1's before/after raw manifests have identical receipt SHA `cb1b06578c087349607bf2eb2d28caf853637b538dc6991e8fe432b92c9f34ac`, while prediction qualification receives the feature SHA. R2 checks mixed reused/new evaluation and extraction plus the all-reused case. The helper and output call sites label timing scopes accurately: retained completed-output durations exclude abandoned partial work; invocation elapsed includes current setup/reuse/scoring through summary creation and excludes final summary writes. Existing per-video definitions remain unchanged.

Ledger additions I08/I09 now declare these two interfaces and match the source. The qualifier refactor preserves the prior QUVA length/shape/binding/file-size checks. No scientific input, model, loss, normalization, decoder or selection change is introduced by these repairs.

The current complete `candidate_identity` is in the accompanying JSON, matching the inspected source manifest, targeted receipt and inherited CPU receipt. Changed source SHAs:

- `native_common.py`: `f2df43370dcd876a7e8929ac8cc9ddd672b9216ce508a6f485183f48249ac6d1`
- `native_engine.py`: `560ae967b2df291a1ccd1d62fbe12e485ff3eb6a7058533a8bbbdaa716e06d31`
- `native_extract.py`: `431603d6995bce4f5423d94acf6a6e4157666f104700da340e0c2c2c3773f661`
- `native_inputs.py`: `341512d3a0d4e2c5817ac8cddf93032907a281d9420ca16973c23421d1405f59`

The contract, runtime, ENV_SPEC, vendor scientific files and `run.py` identities are unchanged. Source identities are from the supplied raw receipts; this reviewer did not run local computational validation or recompute hashes.

Read `REVIEW_FIX.diff`, `TARGETED_REVIEW_FIX.json`, updated manifest/public summary/assumptions, fixture source/log, relevant inherited CPU fields and the actual affected source. Commands were static E: reads and line-number searches. No new server check was needed; the original CPU primitives were inherited and not repeated. Only the two followup review files were written. No source, Git, control, job, model or GPU action occurred.

**Remaining limits:** `CPU_CHECK` is still PENDING for full inputs. FullQUVA production, CUDA/AMP/resource fit, GPU continuation,200 training epochs and full357 scoring remain unexecuted. `PROVISIONAL_PASS` may satisfy the existing focused engineering-review gate for this exact identity; every input/publication/ownership/lane/resource gate remains independently required. It is not scientific, GPU or submission approval.
