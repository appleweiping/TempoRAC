# FULL-v1 CPU metric reference: independent technical review

Review completed: 2026-11-04, final artifact check at 10:18 UTC. Review session: `/root/metric_reference_review`. Project: `E:/ICASSP_TempoRAC/TempoRAC-CVPR`.

**Substantive verdict: no concrete implementation blocker found in the final three-file CPU reference.** The four named count metrics, complete-population accounting, declared matching/tie rules, and full/conditional temporal AP match the proposed contract within its explicitly stated assumptions. One reproducibility-documentation defect was reproduced and corrected. Deterministic checks passed for the properties below; this semantic opinion remains **same-family / provisional**, with no official-metric acceptance, main-protocol freeze, scientific-performance claim, test-set qualification, or deployment approval.

Requested reviewer: `gpt-6.1-sol`, requested reasoning: `xhigh`; backend/model/effort attestation is unavailable. Those requested values are not presented as observed identities. Actual reported variant and effort: null/unknown. This is a fresh native review using the primary artifacts, without an inherited executor verdict. No additional reviewer was spawned. Parent owns the native-call trace and final workflow state.

## Fixed review scope and acceptance checklist

The requested checklist was fixed before executing repros. The proposed Astra document is a reference proposal, not an accepted official/main freeze. This review does not add a new method or external-acceptance gate.

| Requirement | Result | Concrete evidence |
|---|---|---|
| Exact proposed count formulas, video macro aggregation, native finite nonnegative counts without rounding | Passed deterministic checks; semantic provisional | Source 335–350, 387–406; existing artificial suite plus independent unequal-count probe gives AvgNAE=.5 and FullCountNE=1/11; video-macro fixture gives .8 instead of person-pool 2.6/3 |
| Complete declared population, missing/extra tracks and videos | Passed | Source 259–307, 335–391; missing whole videos retain a failure reason and all GT count/period contributions; extras retained even with zero count; artificial fixture keeps 102 predictions |
| Explicit IDs, greedy tube matching, tie rules, no count/period-based identity selection | Passed | Source 22–29, 105–167, 310–332; GT/prediction ID validation; independent greedy-vs-Hungarian counterexample matches one track; tied GT selection uses smallest typed GT ID |
| Source coordinates, full T boxes/support, missing-frame union, no support deletion | Passed within canonical adapter scope | Source 59–102, 112–132, 170–190; boundary 0/T, inclusive converter, nonuniform measured-map inverse, one-sided missing union; adapter does not claim to verify a measured map or decode periods |
| Malformed input/NaN/GT count<=0 fails instead of dropping people | Passed | Source validation and suite; independent NaN probe on otherwise unmatched extra rejects count, track score, event score, bbox and interval; GT 0/-1 and empty GT populations reject |
| Full AP denominator, global score order, duplicate FP, temporal thresholds, integral AP | Passed | Source 193–242, 352–372, 417–420; artificial 1/.5/.25 evidence plus independent threshold-spectrum, GT temporal tie and global pooling probes below |
| Conditional diagnostic accurately distinguished from official pipeline | Passed labeling/scope | Final doc A8/A9 and API limitations; diagnostic deliberately differs from upstream maxDets/ID/tie assumptions; no official count equivalence asserted |
| Actual original official AP functions used for artificial counterexample | Passed at kernel/function scope | Local SHA receipt and source/blob reconciliation below; original AST nodes at 566–648, 650–677, 679–686; test-only np.float compatibility proxy; all ten AP thresholds return 1/.5/.25 |
| Frozen old metric and existing runtime untouched | Passed scoped repository checks | `metrics.py` working and HEAD Git blob both `1645bec31901a486c09fa92c8e4a4d8cae47cde7`; tracked diff empty; search of src/scripts finds new scorer only within its new module |
| No test labels, datasets, models, GPU, network, installation, SSH, commit/push | Satisfied in this review | Read/CPU commands only against code/docs/source snapshots; no dataset paths opened, no model execution; no broad historical 1179-test run |
| Official AvgMAE/AvgOBO parity, accepted legal period decoder, real-data/scientific qualification | Not established; deliberately outside this reference review | Contract itself keeps these pending; artificial checks are not dataset results |

## Primary files and artifact identities

Initial reads captured these SHA256 values before the author's final formatting/doc updates:

| Artifact | Initial SHA256 | Final reviewed SHA256 |
|---|---|---|
| `src/pams/temporac/multirep_metric_contract_v1.py` | `9fc37ae03e728bd88b8be66352881595aef423a6510d945432769e503c8553ba` | `7cff8cbfb8b66ed6f2a4a1811ac2247e2878993be021bf678b8e2cfa2d2d0a9e` |
| `tests/temporac/test_multirep_metric_contract_v1.py` | `248246ab1c97a77f159d98a547ab961cecb040f2dd9af3bf23886b5e40ba4d1a` | `173c82e1a8186fea26114335e569fb09cf5a53bc8ace27d08d787b7d06280047` |
| `docs/MULTIREP_METRIC_REFERENCE_20261104.md` | `2695395fc747af805c7633ee8f80d6bc77719d9d3869699be402703b73fdef08` | `115535f4d6a2d78c7a02f685577df3dcbe62dab107f37f7919548f3cf5c4ba0b` |

All three final files were read. Source was reread in full after its final hash changed. Final suite and independent probes used final source/test hashes above. An intermediate documentation hash `6391866cb5787202edeb677101c87b84dfd056a7bb4671a104899869d63e291d` was superseded by the documentation-only fix below. No code/test mutation accompanied that fix.

Proposed contract: `docs/METRIC_CONTRACT_20261104.md`, SHA256 `43639bc8cb4a466e38c85f1efea493bbe53e5c9ffdb04b808fe2c39e05ab015c` (unchanged across this review).

At scoped final Git inspection, HEAD was `832658815c7baed2ddf2f71d751ad1ab7f607f63`; tracked `git diff HEAD --name-only` was empty and only the three reviewed new files were untracked. HEAD changed during parent work outside this review; this review did not commit, push or stage anything. `metrics.py` SHA256 was `1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd`; working/HEAD blob equality is an independent check that the old metric was not changed.

## Read receipt and governing instructions

Read in full: `E:/AGENTS.md`; project `AGENTS.md`, `ARIS_LOCK.md`, `docs/aris/WORKFLOW.md`, `docs/CVPR_RESUME_20261104.md`, `.aris/runs/temporac-cvpr-20261104.json`, `.aris/reviewer.json`; proposed metric contract; three primary artifacts; source receipt JSONs and relevant actual official scorer source.

Pinned ARIS checkout: `E:/ICASSP_TempoRAC/aris-work/aris-pinned`, commit `f76ec8d6b77792e2242cca049e08d0cbb7d54e19`. Full pinned `skills/skills-codex/research-review/SKILL.md` read, with full mandatory referenced reviewer-backends, reviewer-routing, review-tracing, output-composition, and integration-contract. Installed/pinned research-review SKILL hashes both `17505d5982c82e821124de898b0beb7de4a98ecc4c1cd173b8687b74ace9d4c5`. User/project scoped review and Sol/xhigh request control the route; no unavailable identity was upgraded into accepted assurance.

This is a technical scorer review. It does not invoke literature refresh, full experiment qualification, model redesign, new data access, external communication, or manuscript rendering.

## Mathematical review

For each declared video, all GT tracks have positive counts. Matched predictions provide their native count, unmatched GT uses zero. The implementation computes per-video mean of per-person absolute relative error and per-person ordinary OBO, then averages videos equally. FullCountNE sums absolute matched/missed errors plus all unmatched predicted counts, divides by total GT count inside each video, then averages videos. FullIdentityOBO counts only matched people within absolute count error <=1, divided by GT people plus extra predicted people. Consequently a missed GT with count=1 passes ordinary OBO but never identity OBO; a zero-count extra still penalizes identity OBO. This exactly preserves the proposed distinction.

Predicted-track matching uses descending track score with typed prediction-ID ties, greedy maximum available full-video tube IoU >=.50 within category, GT-ID tie resolution, all tracks retained. Supplied-track mapping is explicit and rejects unknown/cross-video/cross-category/duplicate assignments; null means an extra. Neither count nor periods chooses the identity. IDs 0 and nonconsecutive IDs do not collide with the missing-match sentinel, which is None.

Every canonical track must contain T boxes/support values; numeric validation precedes matching even for an extra track. Null boxes contribute the other side's area to tube union; support states do not mask GT events. Continuous source-boundary intervals use end-start, and only the named inclusive-frame adapter adds one. The boundary helper requires explicit monotone controls covering 0..T and reverses the same interpolation for inverse mapping. It correctly makes no claim to authenticate map controls, synthesize support, or turn event peaks into legal periods.

FullPeriodAP includes all GT periods in its denominator. Every extra-track event is an FP, every missed-track GT event remains a FN, and matched-track duplicate/wrong intervals remain FP. All events are ordered globally by score then typed video/prediction/event IDs. Temporal matching chooses the highest available IoU within the matched identity; equal IoU uses GT event ID. Integral precision-envelope AP has no 101-point discretization. Ten temporal thresholds are .50:.05:.95; AP75 is index 5. No-GT-period is undefined; zero predictions with positive GT is zero AP. Any predicted track whose periods are None makes both AP channels unavailable, including an unmatched extra. These agree with the documented canonical reference semantics.

The conditional diagnostic uses the same reference identity mapping and then retains matched GT/DT tracks. Its name and documentation correctly avoid equating it with the complete official pipeline. In particular official maxDets=100, first config, implicit gt_index+1 behavior, and unstable event ties are not silently claimed as reproduced here.

## Official source/function receipt

Official repository/commit declared in primary receipt: `yinntag/MultiCounterPlus`, `d15c767d82dac2f7a5df9fde1d7f6b5a4f491062`; declared upstream blob `e5c188aa032957fa628062df645a7e1fe8d59b6d`.

Actual file read and executed by test:
`E:/ICASSP_TempoRAC/aris-work/metric_sources/plus/mmdet/datasets/multirep_eval_api.py`.

Raw local bytes: 38653; SHA256 `50e9ede7792763d4f574b56528f728c9383a9b010456d51b720fdc5b33b455ab`, matching both test guard and `docs/evidence/metric_sources_20261104.json`.

An independent Git-blob calculation found an important exact distinction: the local file ends with **one additional trailing LF** relative to the declared upstream blob. Local raw Git blob = `71a37cb7c3c53ab992bba6ffd498c1f244c66eeb`; removing exactly the final LF gives upstream blob = `e5c188aa032957fa628062df645a7e1fe8d59b6d`. This establishes content provenance without mislabeling the whole local file byte-identical. The extra EOF newline cannot change any extracted method body/line identity.

Actual original methods used unchanged through their original AST nodes:

- `compute_average_precision_detection`, lines 566–648;
- `segment_iou`, lines 650–677;
- `interpolated_prec_rec`, lines 679–686.

The test compiles those actual AST nodes with the original filename/line positions. It supplies a test-only `np.float` compatibility proxy and a minimal DataFrame-like adapter because pandas is not present. I inspected its reached operations: descending score reindex, global original GT indexes through groupby/reset_index, numeric segment arrays, per-row lookups, and missing-group failure are faithful for the three unique-score fixtures. This is a valid original-AP-kernel counterexample, not an unmodified complete official runner or proof of pandas/end-to-end parity. No source in the runtime scorer is copied from the official file.

The original official source independently confirms tube IoU 181–260, greedy identity matching 311–394, matched-only denominator collection 438–463, first-config action_ap 534–564, and integral temporal AP 589–686. The artificial cases are:

| Function input | Original AP at all ten thresholds | Reference output |
|---|---|---|
| Matched GT identity 91 with one [0,10] event; same prediction at .8 | 1 | Conditional diagnostic 1 |
| Add unmatched GT identity 7 with one [0,10] event; same prediction | .5 | FullPeriodAP .5; conditional remains 1 |
| Also add unmatched identity's FP [0,10] at .99 before TP .8 | .25 | FullPeriodAP .25; conditional remains 1 |

These cases establish the denominator/leading-FP effect. They do not establish official count-scoring parity or accept any real decoder, data split, or independent-test protocol.

## Reproduction attempts and raw results

All actual Python invocations used `E:/CodexRuntime/temporac-audit-py311/Scripts/python.exe -B`; TEMP/TMP pointed to `E:/ICASSP_TempoRAC/aris-work/tmp`. No installs or GPU calls occurred.

### Retained initial failure and bounded author correction

Initial documentation's abbreviated pytest command omitted PYTHONPATH. Running it in the audit venv, with the project as cwd, produced this genuine collection failure:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.2, pytest-8.4.2, pluggy-1.7.0
rootdir: E:\ICASSP_TempoRAC\TempoRAC-CVPR
configfile: pyproject.toml
collected 0 items / 1 error

ERROR collecting tests/temporac/test_multirep_metric_contract_v1.py
tests\temporac\test_multirep_metric_contract_v1.py:14: in <module>
    from pams.temporac.multirep_metric_contract_v1 import (
E   ModuleNotFoundError: No module named 'pams'
1 error in 1.11s
```

This was sent to the parent immediately as documentation/launcher precision, not a metric-correctness defect. The author changed only the doc to provide a complete PowerShell block setting PYTHONPATH, TEMP/TMP and a separate E-only basetemp, and retained this failure in the final document. Final doc hash is recorded above. No source/test change followed the correction.

### Corrected focused suite

```powershell
$env:TEMP='E:\ICASSP_TempoRAC\aris-work\tmp'
$env:TMP=$env:TEMP
$env:PYTHONPATH='E:\ICASSP_TempoRAC\TempoRAC-CVPR\src'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:\CodexRuntime\temporac-audit-py311\Scripts\python.exe' -B -m pytest -p no:cacheprovider --basetemp 'E:\ICASSP_TempoRAC\aris-work\tmp\metric-reference-review-pytest' 'tests\temporac\test_multirep_metric_contract_v1.py'
```

Actual output, exit 0:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.2, pytest-8.4.2, pluggy-1.7.0
rootdir: E:\ICASSP_TempoRAC\TempoRAC-CVPR
configfile: pyproject.toml
collected 27 items
tests\temporac\test_multirep_metric_contract_v1.py ..................... [ 77%]
......                                                                   [100%]
============================= 27 passed in 0.55s ==============================
```

Direct Ruff command:

```powershell
& 'E:\CodexRuntime\temporac-audit-py311\Scripts\ruff.exe' check --no-cache 'E:\ICASSP_TempoRAC\TempoRAC-CVPR\src\pams\temporac\multirep_metric_contract_v1.py' 'E:\ICASSP_TempoRAC\TempoRAC-CVPR\tests\temporac\test_multirep_metric_contract_v1.py'
```

Actual output, exit 0: `All checks passed!`. A preceding `python -B -m ruff check --no-cache ...` also returned exit 0 but no output; the direct executable check supplies the explicit receipt.

### Additional independent hand-calculated probes

Only necessary checks not directly pinned down by the shipped suite were added in-memory; no source/test edits. The following Python was passed on stdin to the same Python `-B -`, with the same E-only environment. It defines independent artificial fixtures rather than loading any labels.

```python
import copy, json, math
from pams.temporac.multirep_metric_contract_v1 import score, ContractError

def g(i, count=1, x=0, w=2, periods=((0,10),)):
    return dict(gt_id=i,count=count,bboxes=[[x,0,w,2] for _ in range(10)],support=['observed']*10,periods=[dict(event_id=j,interval=list(p)) for j,p in enumerate(periods)])
def p(i, count=1, x=0, w=2, periods=((0,10,.8),), **kw):
    r=g(i,count,x,w,())
    r.pop('gt_id')
    r.update(prediction_id=i,track_score=.9,periods=None if periods is None else [dict(event_id=j,interval=list(e[:2]),score=e[2]) for j,e in enumerate(periods)],decoder_sha='a'*64,config_sha='b'*64,checkpoint_sha='c'*64)
    r.update(kw)
    return r
def v(i, tracks): return dict(video_id=i,num_frames=10,tracks=tracks)
def d(i, tracks): return dict(video_id=i,status='success',tracks=tracks)
def close(a,b): assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)

out={}
r=score([v('v',[g(0,1),g(10,10)])],[d('v',[p(0,10,supplied_gt_id=10)])],setting='supplied-track')
close(r['count_metrics']['AvgNAE_GT'],.5)
close(r['count_metrics']['FullCountNE'],1/11)
close(r['count_metrics']['AvgOBO_GT'],1)
close(r['count_metrics']['FullIdentityOBO'],.5)
out['unequal-count-denominators']=r['count_metrics']

a=p('high',x=1,w=3); b=p('low',x=0,w=3,track_score=.8)
r=score([v('v',[g(0,x=0,w=3),g(20,x=2,w=3)])],[d('v',[b,a])])
assert r['population']['matched']==1
assert r['videos'][0]['raw_counts'][0]['prediction_id']=='high'
out['greedy-not-hungarian']=r['population']

r=score([v('v',[g(0)])],[d('v',[p(0,periods=((0,6,.8),))])])
assert r['FullPeriodAP']['AP']==[1]*3+[0]*7
close(r['FullPeriodAP']['mAP'],.3)
assert r['FullPeriodAP']['AP50']==1 and r['FullPeriodAP']['AP75']==0
out['threshold-spectrum']=r['FullPeriodAP']

gt=g(0,periods=((0,6),(4,10)))
gt['periods'][0]['event_id']='b'; gt['periods'][1]['event_id']='a'
ps=[d('v',[p('p',periods=((0,10,.9),(4,10,.8)))])]
r=score([v('v',[gt])],ps)
assert r['FullPeriodAP']['AP']==[.5]*3+[.25]*7
reverse=copy.deepcopy(gt); reverse['periods'].reverse()
assert score([v('v',[reverse])],ps)==r
out['gt-event-id-tie']=r['FullPeriodAP']['AP']

r=score([v('a',[g(0)]),v('b',[g(20)])],[d('a',[p('tp',periods=((0,10,.5),))]),d('b',[p('fp',x=99,periods=((0,10,.9),))])])
assert r['FullPeriodAP']['AP']==[.25]*10
out['global-not-video-macro-ap']=r['FullPeriodAP']['AP50']

r=score([v('v',[g(0)])],[d('v',[p('matched'),p('extra',x=99,periods=None)])])
assert r['FullPeriodAP']['status']=='unavailable-period-decoder'
assert r['CONDITIONAL-v1-reference-diagnostic']['status']=='unavailable-period-decoder'
out['extra-unavailable-periods']=r['FullPeriodAP']['status']

rejected=[]
for field in ('count','track_score','event_score','bbox','interval'):
    bad=p('extra',x=99)
    if field in ('count','track_score'): bad[field]=float('nan')
    elif field=='event_score': bad['periods'][0]['score']=float('nan')
    elif field=='bbox': bad['bboxes'][0][0]=float('nan')
    else: bad['periods'][0]['interval'][0]=float('nan')
    try: score([v('v',[g(0)])],[d('v',[p('matched'),bad])])
    except ContractError: rejected.append(field)
    else: raise AssertionError(field)
for count in (0,-1):
    try: score([v('v',[g(0,count)])],[])
    except ContractError: rejected.append('gt-count:'+str(count))
    else: raise AssertionError(count)
for gs in ([],[v('v',[])]):
    try: score(gs,[])
    except ContractError: rejected.append('empty-videos' if not gs else 'empty-gt-people')
    else: raise AssertionError(gs)
out['rejected']=rejected

try: score([v('v',[g(0)])],[d('v',[p('p',supplied_gt_id=0,category_id=2)])],setting='supplied-track')
except ContractError: out['cross-category-supplied']='ContractError'
else: raise AssertionError('cross-category')
print(json.dumps(out,indent=2))
```

Actual output, exit 0:

```json
{
  "unequal-count-denominators": {"AvgNAE_GT": 0.5, "AvgOBO_GT": 1.0, "FullCountNE": 0.09090909090909091, "FullIdentityOBO": 0.5},
  "greedy-not-hungarian": {"videos": 1, "gt_tracks": 2, "predicted_tracks": 2, "matched": 1, "missing_gt": 1, "extra_predictions": 1, "failed_or_missing_videos": 0},
  "threshold-spectrum": {"status": "available", "n_gt": 1, "n_predictions": 1, "thresholds": [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95], "AP": [1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], "mAP": 0.3, "AP50": 1.0, "AP75": 0.0},
  "gt-event-id-tie": [0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
  "global-not-video-macro-ap": 0.25,
  "extra-unavailable-periods": "unavailable-period-decoder",
  "rejected": ["count","track_score","event_score","bbox","interval","gt-count:0","gt-count:-1","empty-videos","empty-gt-people"],
  "cross-category-supplied": "ContractError"
}
```

Numeric results are rendered compactly here without changing values; full key order preserved. These are engineering evidence on artificial inputs only.

## Final disposition

Resolved finding: incomplete reproduction command, corrected in final documentation without source/test changes. Remaining actionable implementation blockers: none found in this bounded review. No requirement to change the inherited method, run the historical whole suite, install packages, inspect test labels, run a GPU/model, or seek a new external gate follows from this result.

Reference implementation can be retained as the explicitly named proposed CPU reference and its deterministic evidence. Official count equivalence, a valid frozen decoder and source-clock adapter, release/split/exposure qualification and any real-data/full experiment claims remain pending exactly as the governing contract says. Same-family positive semantic review is provisional.
