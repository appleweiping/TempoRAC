# Native diagnostic support partition repair — 2026-11-04

This bounded engineering change implements the exact DDF1 support partition for legal raw fractional periods. It changes only `unit_diagnostics` classification; numeric overlap fractions, raw endpoints and e-s denominator, pulse membership, schemas, full-population validation and failure labels retain their existing implementation.

For overlap o_i=max(0,min(i+1,e)-max(i,s)), let H contain the strictly positive overlaps. `has_support=any(mask[H])`; full requires e<=N-1 and all mask[H]; unsupported is not has_support; partial is has_support and not full. No endpoint or support tolerance is used. Summary arithmetic remains total-full-partial, now fed disjoint exhaustive flags.

The finalized private DDF1 pair was fully read and hash-checked before editing: contract SHA256 `9cd0a7add1c82ce134184d80b4ade3094974a1f7197f908f645f4ad896137ed8`; receipt SHA256 `c8e776bf5883fd316fe7161d2718f5db69e25cd3d42d0c54e8f1c1809fe22daf`.

| Acceptance requirement | Status and evidence |
|---|---|
| Actual original defect reproduced and preserved | Mechanical pass: old95b8686c, N17/all16 supported edges/[0,16+5e-13] gave full+partial and unsupported summary -1; [16,16+5e-13] gave full+unsupported. Exact raw stdout and launch receipts retained privately. |
| Exact support partition; raw fraction retained | Mechanical pass: postpatch probe gives partial-only and unsupported-only respectively, with unchanged fractions 0.9999999999999687 and 0. |
| Integer/fractional boundaries, genuine small internal gaps, N1, all small-domain masks | Mechanical pass: new tests check all four stages and nonnegative summary counts, four internal-boundary cases, all 31 masks for N1..5, fraction preservation and illegal endpoint failures. |
| Existing isolated diagnostics behavior | Mechanical pass: 53 CPU tests passed in 16.85s, comprising existing 38 plus 15 new cases; no test failure/retry. `git diff --check` exit 0. |
| Fresh independent narrow semantic review | Pending at this executor handoff; root owns separate reviewer. Executor assigns no semantic PASS. Any positive same-family review remains provisional. |
| Complete natural-population target diagnosis / scientific quality | Not performed by this repair; no natural GT, GPU, production or training run. |

CPU execution used existing E audit CPython3.11.2/NumPy1.26.4, explicit E TEMP/TMP/basetemp/cache, no bytecode, one CPU thread, no auto-loaded pytest plugins, hidden/no-console process creation. Private scripts, raw before/after/suite stdout/stderr, receipts, read receipts and baseline snapshots are under `E:/ICASSP_TempoRAC/aris-work/diagnostic-fractional-support-fix/`. No commits, shared-state/manifests, producer/runtime/extractor, raw media or credentials were modified.

Official metrics, legal period decoding, target-quality tolerance and full natural diagnosis remain separate unfulfilled gates. Historical diagnostic review and its 38-test evidence are preserved; they do not cover this newly reproduced fractional defect.
