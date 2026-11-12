"""Existing-artifact analysis only. Run on the research server, never locally."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
root, out = args.root, args.out
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((root/'data/UNION/manifest_dev.json').read_text())
bykey = {f"{x['split']}__{x['video_id']}": x for x in manifest}
assert len(bykey) == len(manifest), 'duplicate protocol keys'
main = [f'final_v2/wo_llm_s{i}' for i in range(3)] + [f'frozen_a/full_s{i}' for i in (3, 4)]
ds = {'rc': ('rc_valid', 'rc_test'), 'ucf': ('ucf_dev', 'ucf_test'), 'cx': ('cx_val', 'cx_test')}
provenance = {}
seen_runs = set()

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(1024*1024): h.update(b)
    return h.hexdigest()

def read(p):
    provenance[str(p.relative_to(root))] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    return json.loads(p.read_text())

def group(x):
    if x['dataset'] == 'Countix': return 'youtube:'+x['video_id'].rsplit('_', 2)[0]
    if x['dataset'] == 'UCFRep': return 'ucf:'+x['action']+':'+x['group']
    return 'sha:'+x['sha256'] if x.get('sha256') else x['dataset']+':'+x['video_id']

def predictions(run, short, single=False):
    seen_runs.add(run)
    val, test = ds[short]
    d = root/'runs'/run
    assert (d/'result.json').exists(), (run, 'missing completion result.json')
    suffix = '' if 'wo_joint_' in run else '_'+val
    p, r = read(d/f'predictions{suffix}.json'), read(d/f'result{suffix}.json')
    cfg = read(d/'config.json')
    expected = sorted(k for k, x in bykey.items() if x['split'] == test)
    assert sorted(p['test']) == expected, (run, short, 'population mismatch')
    pred = np.array([p['test'][k] for k in expected], dtype=float)
    metric = 'test'
    if not single:
        assert sorted(p['test_shift']) == expected
        pred = (pred + np.array([p['test_shift'][k] for k in expected]))*.5
        metric = 'test_tta'
    gt = np.array([bykey[k]['count'] for k in expected], dtype=float)
    good = np.isfinite(gt) & (gt > 0)
    assert np.isfinite(pred).all()
    error = abs(np.round(pred[good])-gt[good])
    a, b = error/(gt[good]+.1), (error<=1).astype(float)
    assert r[metric]['n'] == len(expected) and r[metric]['n_pos'] == len(error), (run, short, 'stored population mismatch')
    assert abs(a.mean()-r[metric]['mae_p']) < 1e-10, (run, short, 'MAE mismatch')
    assert abs(b.mean()-r[metric]['obo_p']) < 1e-10, (run, short, 'OBO mismatch')
    keys = [k for k, g in zip(expected, good) if g]
    full_error = abs(np.round(pred)-gt)
    return a, b, keys, {'run':run, 'seed':cfg['args']['seed'], 'n_all':len(expected), 'n_positive':len(keys), 'mae_p':float(a.mean()), 'obo_p':float(b.mean()), 'mae_positive_div_gt':float(np.mean(error/gt[good])), 'rmse_p':float(np.sqrt(np.mean(error**2))), 'obz_p':float(np.mean(error==0)), 'mae_all':float(np.mean(full_error/(gt+.1))), 'obo_all':float(np.mean(full_error<=1)), 'mae_raw_cont_all':float(np.mean(abs(pred-gt)/(gt+.1))), 'zero_count_rows':[{'uid':k,'prediction':float(p),'gt':float(g)} for k,p,g in zip(expected,pred,gt) if g==0], 'best_epoch':r['best_epoch']}

def bootstrap(delta, labels):
    groups = sorted(set(labels)); ix = {g:i for i,g in enumerate(groups)}
    sums = np.zeros(len(groups)); counts = np.zeros(len(groups))
    for d,g in zip(delta,labels): sums[ix[g]] += d; counts[ix[g]] += 1
    rng = np.random.default_rng(20261111)
    values=[]
    for start in range(0,10000,100):
        j=rng.integers(0,len(groups),size=(100,len(groups)))
        values.extend((sums[j].sum(1)/counts[j].sum(1)).tolist())
    return {'n_clusters':len(groups), 'ci95':np.quantile(values,[.025,.975]).tolist(), 'ci95_bonferroni18':np.quantile(values,[.05/(2*18),1-.05/(2*18)]).tolist()}

results={'method':'seed-averaged per-video errors; matched seeds0,1,2; no model execution', 'inference_scope':'Conditional on fixed fitted models; resampling videos/source clusters only. Seed SD separately describes training dispersion.', 'bootstrap_mc_note':'10000 draws give about14 samples in each Bonferroni18 extreme tail; adjusted percentile bounds have finite Monte Carlo uncertainty; no p-values or confirmatory significance claims.', 'bootstrap_draws':10000, 'bootstrap_seed':20261111, 'primary_family_size':18, 'benchmarks':{}}
for short in ds:
    full=[predictions(r,short) for r in main]
    assert [x[3]['seed'] for x in full] == [0,1,2,3,4]
    keys=full[0][2]
    assert all(x[2]==keys for x in full)
    base_a=np.stack([x[0] for x in full[:3]])
    base_b=np.stack([x[1] for x in full[:3]])
    report={'full_5seed': [x[3] for x in full], 'full_5seed_mean':{'mae_p':float(np.mean([x[0].mean() for x in full])), 'obo_p':float(np.mean([x[1].mean() for x in full]))}, 'full_5seed_sd':{'mae_p':float(np.std([x[0].mean() for x in full],ddof=1)), 'obo_p':float(np.std([x[1].mean() for x in full],ddof=1))}, 'matched_3seed_mean':{'mae_p':float(base_a.mean()),'obo_p':float(base_b.mean())}, 'ablations':{}}
    for arm in ['pyramid','shift','depth','token','balance','joint']:
        runs=[f'frozen_a/wo_{arm}_s{i}' if arm!='joint' else f'frozen_a/wo_joint_{short}_s{i}' for i in range(3)]
        ab=[predictions(r,short,single=arm=='shift') for r in runs]
        assert all(x[2]==keys for x in ab)
        assert [x[3]['seed'] for x in full[:3]] == [0,1,2] == [x[3]['seed'] for x in ab]
        a=np.stack([x[0] for x in ab]); b=np.stack([x[1] for x in ab])
        da=(a-base_a).mean(0); db=(base_b-b).mean(0)
        labels=[group(bykey[k]) for k in keys]
        obov, oboc = bootstrap(db,keys), bootstrap(db,labels)
        obov.pop('ci95_bonferroni18'); oboc.pop('ci95_bonferroni18')
        report['ablations'][arm]={'raw': [x[3] for x in ab], 'mae_removal_minus_full':float(da.mean()), 'relative_mae_removal_pct':float(100*da.mean()/base_a.mean()), 'obo_full_minus_removal':float(db.mean()), 'per_video_mae':bootstrap(da,keys), 'source_cluster_mae':bootstrap(da,labels), 'per_video_obo':obov, 'source_cluster_obo':oboc}
    results['benchmarks'][short]=report

# Do not remove discovered overlaps from the evaluation population.
allrows=manifest+json.loads((root/'data/QUVA/manifest.json').read_text())
audit={'population':dict(Counter(x['split'] for x in allrows)), 'manifest_sha256':sha(root/'data/UNION/manifest_dev.json'), 'limitations':['Exact bytes and recorded source IDs only; no perceptual near-duplicate scan.','RepCount source recordings beyond recorded IDs/SHA unavailable.','Pretraining exposure not audited.'], 'overlaps':{}}
for kind,fn in [('exact_sha256',lambda x:x.get('sha256')),('known_source',group)]:
    buckets=defaultdict(list)
    for x in allrows:
        identity=fn(x)
        if identity: buckets[identity].append(x)
    collisions=[]
    for identity, xs in buckets.items():
        if len(set(x['split'] for x in xs))>1:
            collisions.append({'identity':identity, 'rows':[{'dataset':x['dataset'],'split':x['split'],'video_id':x['video_id'],'count':x['count']} for x in xs]})
    audit['overlaps'][kind]={'cross_split_groups':len(collisions),'groups':collisions}

configs={}
for run in sorted(seen_runs):
    configs[run]=read(root/'runs'/run/'config.json')
    for p in (root/'runs'/run).glob('*.pt'):
        provenance[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
for p in sorted((root/'code/temporac_cvpr').rglob('*')):
    if p.is_file() and '__pycache__' not in str(p): provenance[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
center=configs[main[0]]
config_diffs={}
for run,c in configs.items():
    config_diffs[run]={}
    for section in ['cfg','args']:
        config_diffs[run][section]={k:{'center':center[section].get(k),'run':c[section].get(k)} for k in sorted(set(center[section])|set(c[section])) if center[section].get(k)!=c[section].get(k) and not(section=='args' and k in ['seed','out'])}
for run in main:
    assert not config_diffs[run]['cfg'] and not config_diffs[run]['args'], (run,'main configuration mismatch')

baseline={'escounts_evaluations':{},'internal_style_controls':[], 'other_official_weight_assets':[], 'historical_weight_binding':'Current asset hashes verified; historical per-evaluation hash receipt not located. ESCounts identity inferred from code/launch paths and existing records, not proven by directory name alone.', 'scope':'Explicit ESCounts allowlist, released decoder+encoder asset hashes and code/launch evidence; no download, model load or retraining; table literature numbers remain separately labelled. UCFRep is RepCount-weight transfer, not in-domain training.'}
for p in sorted((root/'runs/baselines').glob('*/records.jsonl')):
    if p.parent.name not in ['escounts_official','escounts_official_testwarp','escounts_official_quva','escounts_official_ucfrep']: continue
    lines=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    uniq={}
    for x in lines:
        assert x['key'] not in uniq or uniq[x['key']]==x, (p, 'conflicting duplicate records', x['key'])
        uniq[x['key']]=x
    b={'path':str(p.relative_to(root)), 'result_exists':p.with_name('result.json').exists(), 'records':len(lines), 'unique_keys':len(uniq),'splits':{},'decoder':'pretrained/escounts_repcount.pyth','encoder':'pretrained/VIT_B_16x4_MAE_PT.pyth', 'mode':'RepCount checkpoint transfer' if p.parent.name!='escounts_official' else 'RepCount official pretrained checkpoint'}
    bm = {'escounts_official':'data/RepCountA/manifest_all.json','escounts_official_quva':'data/QUVA/manifest.json','escounts_official_ucfrep':'data/UCFRep/manifest_full.json','escounts_official_testwarp':'data/RepCountA_warp/warp_pw3.json'}[p.parent.name]
    expected_manifest=read(root/bm)
    for split in sorted({x['split'] for x in uniq.values()}):
        rows=[x for x in uniq.values() if x['split']==split]; g=np.array([x['count'] for x in rows]); pred=np.array([x['raw_count'] for x in rows]); pos=g>0; e=abs(np.round(pred)-g)
        expected={f"{x['split']}__{x['video_id']}":x for x in expected_manifest if x['split']==split}
        assert all(x['key'] in expected and x['count']==expected[x['key']]['count'] for x in rows), (p,'baseline ID/GT mismatch')
        if b['result_exists']: assert len(rows)==len(expected), (p,'incomplete baseline population')
        b['splits'][split]={'n':len(rows),'n_positive':int(pos.sum()),'mae_p':float(np.mean(e[pos]/(g[pos]+.1))),'obo_p':float(np.mean(e[pos]<=1)),'mae_all':float(np.mean(e/(g+.1))),'mae_positive_div_gt':float(np.mean(e[pos]/g[pos]))}
        if b['result_exists']:
            stored=read(p.with_name('result.json'))[split]
            assert stored['n']==len(rows) and stored['n_pos']==int(pos.sum())
            for name in ['mae_p','obo_p']:
                if name in stored: assert abs(stored[name]-b['splits'][split][name])<1e-10
    b['completion']='complete' if b['result_exists'] else 'partial; no result.json (testwarp OOM in stdout)'
    baseline['escounts_evaluations'][p.parent.name]=b
    provenance[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
for p in (root/'runs').glob('*/*/config.json'):
    if any(s in p.parent.name.lower() for s in ['style','tsm','repnet']):
        c=json.loads(p.read_text()); baseline['internal_style_controls'].append({'run':str(p.parent.relative_to(root)), 'head':c['cfg'].get('head'), 'seed':c['args']['seed'], 'complete':p.with_name('result.json').exists(), 'not_official_reproduction':True})
for rel in ['pretrained/escounts_repcount.pyth','pretrained/VIT_B_16x4_MAE_PT.pyth','third_party/PoseRAC/best_weights_PoseRAC.pth']:
    p=root/rel
    if p.exists():
        item={'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)}
        baseline['other_official_weight_assets'].append(item); provenance[rel]=item
for rel in ['data/UNION/manifest_dev.json','data/QUVA/manifest.json','logs/escounts_extra.sh']:
    p=root/rel
    provenance[rel]={'bytes':p.stat().st_size,'sha256':sha(p)}
for name,obj in [('analysis_frozen.json',results),('overlap_audit.json',audit),('provenance.json',provenance),('configs.json',configs),('config_diffs.json',config_diffs),('baseline_source_audit.json',baseline)]:
    (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
print(json.dumps({'done':True,'out':str(out),'population':audit['population'],'overlap_counts':{k:v['cross_split_groups'] for k,v in audit['overlaps'].items()},'main':{k:v['full_5seed_mean'] for k,v in results['benchmarks'].items()}}))
