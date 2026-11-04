# Bounded diagnostics fractional-support review — 2026-11-04

Verdict: **NO_REMAINING_CONCRETE_ENGINEERING_BLOCKER** for the three frozen files below against b6b0efd6d2e5a99a34add2ef967e6233840a44c4. Review completed; scope complete; positive bounded semantic assessment remains **same-family / provisional**. Requested reviewer gpt-6.1-sol / xhigh; backend-reported model and actual effort are unavailable, not independently attested.

No new defect was found in the stable patch. The independently reproduced old defect is repaired: valid periods now have exactly one support category, and the raw fractional overlap value is preserved. This verdict grants no scientific target-quality, official metric, actual full-target diagnosis, GPU feasibility, training, or paper-quality PASS.

## Scope and sequence

The checklist was written before author patch access and is preserved verbatim at `E:/ICASSP_TempoRAC/aris-work/diagnostic-fractional-support-review-cpu/checklist_before_author_diff.md` (SHA256 950e41b1654307686c8a546c2669cc186436eb304de51366beff20306c2e1e76). Project policies/current status and the complete pinned Codex research-review SKILL plus mandatory references were read before review; the read receipt records exact hashes and pin f76ec8d6b77792e2242cca049e08d0cbb7d54e19. The explicit Sol/xhigh instruction overrides upstream Astra/ultra. No fallback occurred.

The finalized DDF1 contract and receipt were read directly in full and hash-verified before author diff inspection: contract 9cd0a7add1c82ce134184d80b4ade3094974a1f7197f908f645f4ad896137ed8; receipt c8e776bf5883fd316fe7161d2718f5db69e25cd3d42d0c54e8f1c1809fe22daf. Author STABLE was received before any patch inspection or test execution. The reviewed source/test/document hashes matched STABLE before and after the independent CPU run and on final recheck.

| Reviewed file, relative to E:/ICASSP_TempoRAC/TempoRAC-CVPR | Final SHA256 |
|---|---|
| src/pams/temporac/native_target_diagnostics_v1.py | b15d323111d3f65c589364884e63760f802031b1a1dd57a9ddbca3fc65023045 |
| tests/temporac/test_native_target_diagnostics_v1.py | 3840d53f222eabb80f8732bc7ec5be2346c672f4dab1a5157bb8b608953863e9 |
| docs/NATIVE_DIAGNOSTIC_SUPPORT_FIX_20261104.md | da26b6ed6fba288e455fdb614d93485cb3ebd9e747c1a4c8d47b39684ddc89a2 |

The actual diff has one source classification hunk and 15 added expanded test cases. The new document accurately describes the bounded repair and pending independent review at the executor handoff. Existing 38-test full-evaluator review, provenance/schema/runtime/full-science judgments were not reopened. The impacted original suite was rerun as requested, without extending its review scope.

## Original negative evidence and final positive evidence

The reviewer independently extracted the module with `git show b6b0efd6d2e5a99a34add2ef967e6233840a44c4:src/pams/temporac/native_target_diagnostics_v1.py`. Its exact raw bytes SHA256 is 95b8686caa725d3c220e68fb74e13dab7b822f41fe3d3dc1e12c10155ab3fee5, matching the defect-bearing source. No author repro script or fixture was used for the independent repro.

N=17, all16 source edges supported, pulse[0]. Each listed result was checked for observed/response/target/training:

| Raw period | Old flags (full,partial,unsupported) | Old summary (full,partial,unsupported) | Final flags and summary | Raw fraction in both |
|---|---|---|---|---|
| [0,16] | (1,0,0) | (1,0,0) | (1,0,0) | 1.0 |
| [0,16+5e-13] | (1,1,0) | (1,1,-1) | (0,1,0) | 0.9999999999999687 |
| [16,16+5e-13] | (1,0,1) | (1,0,0) | (0,0,1) | 0.0 |

The three-period final aggregate is exactly (1,1,1), denominator3, for every stage. Negative and corrected raw outputs remain separate, without overwriting.

The implementation directly follows the frozen exact rule: for H containing strictly positive source-edge overlaps, has_support=any(mask[H]); full=(e<=N-1 and all(mask[H])); partial=has_support and not full; unsupported=not has_support. For a valid positive period ending at or before N-1, H is nonempty and source edges cover its domain. Thus full implies has_support, making these flags disjoint and exhaustive. For N=1 there are no source edges and every valid period is unsupported. A tail-only interval cannot become full. No isclose, tolerance, endpoint snapping or new support edge is used. Summary remains total-full-partial; with this exact partition it equals the number of unsupported rows and is nonnegative.

An independent scalar exact-rational union-length oracle checked 5,852 period/mask combinations over N=1,2,3,4,5,17. It computed the exact measure of the union of supported edge intersections using Fraction.from_float, then classified by covered measure=duration, 0<covered measure<duration, or zero. This is distinct from the patch's positive-overlap-mask implementation. Included cases cover nextafter boundaries, 5e-13 gaps, internal unsupported edges, terminal tails, no edges, no support and full support. All cases matched; all 5,852 raw fractions remained exactly equal to the Git baseline. All341 integer cases retained their baseline category. Thirteen invalid-input cases retained identical DiagnosticError types/messages; three unresolved-policy cases retained exact results. Empty-period behavior remains covered by the unchanged suite and independent empty-pulse check.

## Frozen acceptance checklist and evidence

| ID | Requirement | Status and concrete evidence |
|---|---|---|
| P1 | Read project policies, current status, complete pinned research-review skill and mandatory references before review | Pass: own readreceipt JSON; exact Codex references fully read, pinned HEAD clean; installed SKILL SHA matched pinned original. |
| P2 | Wait for author STABLE, bind exact file SHA and read exact primary endpoint-freeze design/receipt | Pass: STABLE handoff preceded all patch/test access; DDF1 pair above; stable hashes verified before/after/final. |
| A1 | Independently reproduce original negative evidence from Git baseline | Pass: raw baseline_repro.json preserves full+partial and aggregate unsupported=-1, plus full+unsupported tail-only case. |
| A2 | Each valid raw period has exactly one frozen exact category | Pass: direct formula/proof inspection and 5,852 exact-rational oracle cases. |
| A3 | No tolerance-based full support; genuine gaps partial and last-frame-only tail unsupported | Pass: raw boundary repro, nextafter and internal-gap cases, all15 new expanded test cases. |
| A4 | Per-unit and aggregate partition exhaustive/nonnegative | Pass: all oracle flags sum1; allfour-stage three-period aggregate (1,1,1), denominator3. |
| A5 | Preserve descriptive raw fractions/endpoints/e-s and integer behavior | Pass: 5,852 identical baseline fractions,341 integer-equivalent categories; no fraction/pulse/validation/schema hunk. |
| A6 | Preserve invalid-period behavior | Pass:13 identical error-type/message cases,3 unresolved-policy equivalents, existing empty-period test. |
| A7 | Stable bounded regressions plus impacted38 CPU suite once; explicit E scratch/no-console children | Pass: single independent pytest run53/53 (38 existing+15 new),15.12s pytest/16.265s process wall; exit0, stderr empty; all owned temp/cache/basetemp E, -B/no bytecode, one thread. |
| A8 | Preserve nonofficial/descriptive scope in documentation and verdict | Pass: new document and unchanged output labels; official_metric_status=not_certified, period_ap unavailable, target_quality_acceptance not_evaluated. |

Command (working directory E:/ICASSP_TempoRAC/TempoRAC-CVPR):

```text
E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe -B -m pytest E:/ICASSP_TempoRAC/TempoRAC-CVPR/tests/temporac/test_native_target_diagnostics_v1.py -p no:cacheprovider --basetemp E:/ICASSP_TempoRAC/aris-work/diagnostic-fractional-support-review-cpu/pytest-basetemp -o cache_dir=E:/ICASSP_TempoRAC/aris-work/diagnostic-fractional-support-review-cpu/pytest-cache -q
53 passed in 15.12s; exit0
```

CPython3.11.2, NumPy1.26.4; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, PYTHONUTF8=1, PYTHONDONTWRITEBYTECODE=1, explicit E TEMP/TMP/TMPDIR/PYTHONPATH/cache, OMP/OPENBLAS/MKL/NUMEXPR/VECLIB thread limits1. The .NET root used UseShellExecute=false/CreateNoWindow=true; Python child launches explicitly used CREATE_NO_WINDOW. Launcher PID33796 exited0. No healthy job was inspected, signaled, interrupted or restarted. No C:/D: review-owned artifact, real annotation, server, GPU, training, source/doc/state/manifest write, commit or push was performed. A preparatory read found tests/conftest.py absent; this caused no test run or retry. Scoped git diff --check passed; no new generic lint/audit was launched.

## Raw evidence and integrity

All following files are in `E:/ICASSP_TempoRAC/aris-work/diagnostic-fractional-support-review-cpu/`:

| Artifact | SHA256 |
|---|---|
| baseline_repro.json | 7644e837d3d33b463dd5b51e11cd406e6e3fe00475272784ad9be06a611cbb09 |
| final_repro.json | 11226ddcda7861d49767a3d28681ed1ffa87d0bccc474ef615bdbfbb3256bfdd |
| independent_probe.json | 53c7f2775c10a3597153bcdf4ea3e728c173d1f20ea4e1e51e223ac17796279c |
| impacted_suite.stdout.txt | 464e7c34f9c8436751b5a578141a4fe90ea70d79a83ddfe0a5266f9a03934af2 |
| impacted_suite.stderr.txt | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| impacted_suite.receipt.json | 4f6f1856071033f527b3bad6a3dc23abc050eae231e1f3978b76194623e78ebd |
| completed_run.json | d8b08a0b01ada8a7a508bb1ba9eeae8faae6b47eef4511b6eb948e82ba21fe7e |
| review_cpu.py | 5faab35671c6995dafa7a5c24e2957b3477d88729c98c9cbfebceb5008de5fda |
| launcher.receipt.json | 92ed3ef09d35a0f735639b803486d9b6b009da2300e7b974ec291880da6cc742 |

Remaining scoped blockers:0. Parent owns shared trace/manifests/state, checkpointing, push and server synchronization. This review records a completed bounded arithmetic repair only; the complete natural-population diagnostic and all unfulfilled scientific requirements remain pending.
