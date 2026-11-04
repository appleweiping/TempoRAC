# FULL-v1 CPU metric reference — 2026-11-04

Status: proposed named contract; NOT official metrics, NOT an accepted main-protocol freeze. CPU artificial-fixture evidence only. Existing method, metrics.py, frozen main protocol, labels and models are outside scope.

## SPEC (recorded before code)

Base commit: `8fe7e76576b86d23adfc450fd690378711cadf9e`.
Target: standalone deterministic pure-CPU supplied-track and predicted-track scorer for the complete declared video population, implementing §3 of `aris-work/metric_contract_20261104.md`. Inputs: canonical JSON-like videos with source frame count and GT tracks; prediction video success/failure records with tracks, provenance, full-frame bboxes/support, count and scored periods. Output: four named count metrics, full and conditional temporal AP, identity matches, missing/extra counts, raw counts and failure reasons. Success: the self-contained PowerShell command under Reproduction command (including explicit PYTHONPATH, TEMP/TMP and E-only basetemp). No dataset scoring, inference, server action, install, commit or push.

## Assumptions ledger (ASK=never; declared before implementation)

| ID | Gap | Chosen | Class | Source |
|---|---|---|---|---|
| A1 | Canonical serialization | Lists of dictionaries; IDs nonempty strings or integers (no bool); typed ordering puts integers before strings; GT IDs globally unique, prediction IDs unique within video; category IDs explicit, default person=1 | interface | default |
| A2 | Supplied identity mapping | Each prediction explicitly has supplied_gt_id (null allowed); unknown, cross-video, cross-category and duplicate assignment fail; never infer identity from counts/periods | semantic | default |
| A3 | Absent videos versus malformed record | Omitted prediction videos become explicit missing-video failures and zero output; explicit failure requires reason and empty tracks; unknown/duplicate videos fail; no GT videos or no GT people fail until contract patch | semantic | contract + default |
| A4 | Period decoder not accepted | periods=null marks unavailable for the whole period channel; empty list is a valid zero-event output; period-ready metadata is not inferred from counts | semantic | contract + default |
| A5 | Canonical source clock | All tracks have T xywh/null bboxes and T support labels from observed/interpolated/extrapolated/missing; null iff missing; no masking GT periods by support; period intervals source-boundary real [s,e], no +1 | interface | contract + default |
| A6 | Conversion adapter scope | Named inclusive integer span converter and explicit piecewise-linear boundary map with inverse; map must cover 0..T, strictly increasing and verified externally; no implicit stride multiplication or bbox resampling | interface | default |
| A7 | Provenance fields | Every prediction track requires nonempty decoder_sha/config_sha/checkpoint_sha strings of 64 hexadecimal digits; scorer validates syntax, not file existence/authenticity | interface | default |
| A8 | Conditional output | Label CONDITIONAL-v1-reference-diagnostic; uses same declared identity matching then restricts GT and predictions to matched tracks; not exact official complete pipeline; official function parity only synthetic AP kernel, without tied scores | semantic | contract + default |
| A9 | Temporal matching tie | Global event order score descending then video/prediction/event IDs; equal tIoU GT periods use GT event ID; all ten exact thresholds .50..95; no 100-track truncation | semantic | contract + default |

## Build ladder / acceptance table

| Rung | MUST artifact/check | State |
|---|---|---|
| F0 | Importable public entry and explicit contract status | passed |
| F1 | Validation, source boundary adapters, count formulas and full identity ledger | passed |
| F2 | Deterministic tube matching; full/conditional period AP | passed |
| F3 | Manual counterexamples, strict failures and exact upstream AP kernel parity | passed |

Consolidation into this one file overrides the skill's default four implement-stage files to respect delegated file ownership. No commit per task instruction. Independent fresh review requested through parent at delivery; same-family positive opinion remains provisional.

## Source / skill read receipt

ARIS pin `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`; installed and pinned research-implement-feature SKILL SHA256 `33605606e3b04940b6b262e88a1a729a6a0c81f6a831c3a734c57ab75c57acf8`. Full skill and mandatory output-language, effort-contract, acceptance-gate, capture-antipatterns, injection-hygiene, reviewer-backends, reviewer-independence, reviewer-routing and review-scope-limits read before mutation. No review invoked by this executor; parent owns routing.

Official AP source: MultiCounterPlus commit `d15c767d82dac2f7a5df9fde1d7f6b5a4f491062`, blob `e5c188aa032957fa628062df645a7e1fe8d59b6d`, local `aris-work/metric_sources/plus/mmdet/datasets/multirep_eval_api.py`, SHA256 `50e9ede7792763d4f574b56528f728c9383a9b010456d51b720fdc5b33b455ab`. Test reads only original methods compute_average_precision_detection, segment_iou, interpolated_prec_rec at their original AST line identities, executes them unchanged with test-only np.float compatibility proxy and a minimal DataFrame-like fixture adapter because pandas is absent. No source copied to runtime module. These kernel receipts cannot establish official AvgMAE/OBO or end-to-end parity.

## API use and schema

Import `score`, `inclusive_frame_span`, and `map_boundaries` from `pams.temporac.multirep_metric_contract_v1`. `score(videos, prediction_videos, setting="predicted-track")` returns a plain dictionary; `setting="supplied-track"` uses explicit per-prediction `supplied_gt_id`. No file or data-loader entrypoint is provided, so isolated evaluators control GT access.

```python
from pams.temporac.multirep_metric_contract_v1 import score
T = 10
boxes = [[0, 0, 2, 2]] * T
support = ["observed"] * T
videos = [{"video_id": "manual", "num_frames": T, "tracks": [{
    "gt_id": 91, "count": 1.0, "bboxes": boxes, "support": support,
    "periods": [{"event_id": 0, "interval": [0, T]}]}]}]
predictions = [{"video_id": "manual", "status": "success", "tracks": [{
    "prediction_id": "p", "track_score": .9, "count": .8,
    "bboxes": boxes, "support": support,
    "periods": [{"event_id": 0, "interval": [0, T], "score": .9}],
    "decoder_sha": "a" * 64, "config_sha": "b" * 64,
    "checkpoint_sha": "c" * 64}]}]
result = score(videos, predictions)
assert result["count_metrics"]["FullCountNE"] > 0
assert result["FullPeriodAP"]["AP50"] == 1
```

Provenance in this artificial example is placeholder syntax only, not authenticated model evidence. Production callers must supply actual frozen identities. GT count and number of annotated periods are separate observations; this scorer does not infer or reconcile them. Missing whole videos receive `failure_reason="missing-prediction-video"`; explicit failures use `{video_id,status:"failure",failure_reason,tracks:[]}`. Malformed records throw `ContractError`; none are discarded. Count output retains every GT raw count and each extra prediction raw count, matched/missing/extra track totals and per-video results. All counts are native finite nonnegative reals, never rounded. FullCountNE uses total GT count denominator; all aggregate count outputs average videos equally.

`periods=None` on any predicted track disables both period channels for the complete call. It cannot be used to silently remove just that identity from AP; count scoring remains available. When GT has zero total periods, AP is undefined with null numeric fields. Explicit `periods=[]` means a qualified decoder produced zero events. Confidence ties in this proposed reference are deterministic; this differs from upstream's unstable tied-score argsort. Conditional reference uses all matched tracks without upstream's maxDets=100 or gt_index+1 assumptions, so its label does not claim exact official pipeline equivalence.

`map_boundaries` requires paired strictly increasing output/source control points covering source [0,T]; `inverse=True` reverses the same piecewise-linear map. It requires externally verified controls and cannot certify them, infer stride, resample boxes, discard gaps or turn event peaks into periods. `inclusive_frame_span(0,T-1,T)` is the only named +1 adapter; continuous intervals receive no +1.

## Build note / deterministic evidence

F0 ran before feature implementation: audit Python imported CONTRACT_STATUS and asserted `proposed-reference-not-official`, exit 0. F1/F2 complete real score path smoke printed `{AvgNAE_GT:1,AvgOBO_GT:1,FullCountNE:1,FullIdentityOBO:0}` for missing g=1, preserving the distinction between ordinary OBO and identity success. No runtime stubs remain.

F3 accumulated suite: **27 passed in 0.50 s**, exit 0; Ruff `check --no-cache` passed. Commands use `E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe -B`, E-only TEMP/TMP and pytest `-p no:cacheprovider --basetemp E:/ICASSP_TempoRAC/aris-work/tmp/metric-reference-pytest`. Formatting and final rerun evidence below supersede timings if changed. Initial fixture run had two test-authoring mistakes (wrong manual assertion and incorrect expected AST end lines); corrected by checking exact source and formulas, without changing contract behavior.

| Acceptance requirement | Status and evidence |
|---|---|
| Full/supplied population metrics, no rounding, missing/extra retention | passed fixtures: video-macro versus person-pool, missing g=1/g=10, 102 predictions, duplicate/zero-count extras |
| Whole-video failures and malformed records fail atomically | passed: explicit and missing-video records, NaN, malformed lengths, IDs, provenance, intervals |
| Full source coordinate and tube matching | passed: frame 0/T, inclusive [0,9]→[0,10], continuous adjacent intervals, measured nonuniform stride map roundtrip, missing-frame union |
| Full/conditional AP and stable ties | passed: unmatched GT, leading FP, repeated periods, reordered tracks/GT, equal-score ID ordering, decoder-unavailable and undefined GT-period channels |
| Exact original official AP kernel on artificial examples | passed: AST source at 566–648 / 650–677 / 679–686, SHA receipt; unchanged official functions return conditional 1/full .5/leadingFP .25 at all ten thresholds |
| Existing main protocol/method unchanged | scope-limited new independent module/test/doc only; parent handles repository-wide diff verification |
| Official count scorer reproduction / dataset or independent-test scores | not performed; pending and outside authorized scope |
| Fresh independent silent-assumption/code review | requested from parent; pending; no PASS invented |

No data labels, dataset files, models, network, installs, GPU or server were used. No experimental outcome, SOTA claim, freeze approval or three-way sync is implied. Loss/method design unchanged. The skill's Type-A ladder is green; semantic assurance awaits independent review, which remains provisional if same-family.

Final formatted rerun: 27 passed in 0.57 s; Ruff check passed. Source SHA256 `7cff8cbfb8b66ed6f2a4a1811ac2247e2878993be021bf678b8e2cfa2d2d0a9e`; test SHA256 `173c82e1a8186fea26114335e569fb09cf5a53bc8ace27d08d787b7d06280047`. Doc hash is reported externally to avoid self-reference.


## Reproduction command and retained review failure

Run this complete PowerShell block without installing the project. The original abbreviated command in SPEC omitted PYTHONPATH; the fresh reviewer observed collection failure `ModuleNotFoundError: pams`. That failed attempt is retained here as a documentation defect, not relabeled a test pass. The implementation runs above already set PYTHONPATH explicitly. The command below corrects the reproducibility instructions; the reviewer's corrected rerun/verdict is recorded separately by the parent.

```powershell
Set-Location -LiteralPath 'E:\ICASSP_TempoRAC\TempoRAC-CVPR'
$env:PYTHONPATH = 'E:\ICASSP_TempoRAC\TempoRAC-CVPR\src'
$env:TEMP = 'E:\ICASSP_TempoRAC\aris-work\tmp'
$env:TMP = 'E:\ICASSP_TempoRAC\aris-work\tmp'
& 'E:\CodexRuntime\temporac-audit-py311\Scripts\python.exe' -B -m pytest -p no:cacheprovider --basetemp 'E:\ICASSP_TempoRAC\aris-work\tmp\metric-reference-documentation-pytest' 'E:\ICASSP_TempoRAC\TempoRAC-CVPR\tests\temporac\test_multirep_metric_contract_v1.py' -q
& 'E:\CodexRuntime\temporac-audit-py311\Scripts\ruff.exe' check --no-cache 'E:\ICASSP_TempoRAC\TempoRAC-CVPR\src\pams\temporac\multirep_metric_contract_v1.py' 'E:\ICASSP_TempoRAC\TempoRAC-CVPR\tests\temporac\test_multirep_metric_contract_v1.py'
```

The documentation basetemp has a separate explicit E-only directory to avoid concurrent review/implementation fixture cleanup collisions. No new implementation/test mutation accompanied this correction.
