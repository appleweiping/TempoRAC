"""Artificial complete populations only: no natural labels or model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('score_candidate', ROOT/'src/pams/temporac/native_prediction_score_v1.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2))
    return {'path': str(path), 'sha256': s.digest(path)}


@pytest.fixture(scope='session')
def template(tmp_path_factory):
    base = tmp_path_factory.mktemp('all_artificial')
    arm = base/'runs/20260815/canonical'
    arrays = arm/'predictions'
    arrays.mkdir(parents=True)
    population, rows, records = [], [], []
    for split, nv, triple in [('train',794,290), ('val',183,120)]:
        for vid in range(nv):
            media = f'{split}{vid:059d}'
            for slot in range(3 if vid<triple else 2):
                key = media+'_'+str(slot)
                row = dict(split=split,media_key=media,slot=slot,source_length=4,
                           gt_count=1+((vid+slot)%2),source_component=f'c{vid//2}')
                rows.append(row)
                population.append(dict(key=key,media_key=media,slot=slot,split=split,N=4))
                if split!='val': continue
                abstain=vid==0 and slot==0
                count=-1 if abstain else (vid+slot)%3
                k=max(count,0)
                bounds=np.asarray([[2*i,2*i+1] for i in range(k)],dtype=np.int32).reshape(-1,2)
                response=np.zeros(3,dtype=np.float32);response[0:2*k:2]=.75
                np.savez_compressed(arrays/(key+'.npz'),response=response,
                    support=np.zeros(3,dtype=bool) if abstain else np.ones(3,dtype=bool),
                    run_bounds=np.empty((0,2),np.int32) if abstain else np.asarray([[0,3]],np.int32),
                    component_bounds=bounds,event_source_positions=2*np.arange(k,dtype=np.float64)+.5,
                    event_seconds=np.full(k,np.nan,dtype=np.float64))
                records.append(dict(key=key,count=count,status='abstain' if abstain else 'complete',
                    decoder_invocations=1,sha256=s.digest(arrays/(key+'.npz')),
                    period_ap_status='unavailable_no_period_decoder'))
    rows_path=base/'diagnosis/identity_rows.jsonl'
    rows_path.parent.mkdir()
    rows_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    rows_ref={'path':str(rows_path),'sha256':s.digest(rows_path)}
    prior={str(base/'projection.jsonl'):'1'*64,str(base/'source_components.json'):'2'*64,
           str(base/'nb1.json'):'3'*64,str(base/'nt1.json'):'4'*64}
    diag=dict(schema='temporac.native-target-diagnostics.v1',official_metric_status='not_certified',
        receipt={'identity_rows_sha256':rows_ref['sha256'],'inputs_sha256':prior},
        same_media_alias_groups=[{'media_keys':[f'train{i}',f'train{i+1}'],'label_conflict':True} for i in range(4)],
        splits={sp:{'identity_denominator':ni,'full_population':{'videos':nv,'identities':ni}}
                for sp,nv,ni in [('train',794,1878),('val',183,486)]})
    diag_ref=save(base/'diagnosis/diagnostics.json',diag)
    policy=copy.deepcopy(s.POLICY)
    policy['metric_name']='DDF1-development-video-person-macro'
    policy_ref=save(base/'policy.json',policy)
    artifacts={'diagnostics':diag_ref,'count_policy':policy_ref,
        'nb1':{'path':str(base/'nb1.json'),'sha256':'3'*64},
        'nt1':{'path':str(base/'nt1.json'),'sha256':'4'*64}}
    ledger=[dict(step=i+1,x0_source=i%24,component='c0',identity=population[0]['key']) for i in range(10000)]
    plan=dict(schema=s.NTR1,pending=[],seed=20260815,T=10000,population=population,
        initial_state_sha256='5'*64,draw_ledger=ledger,draw_ledger_sha256=s.json_sha(ledger),
        candidate_steps=list(range(500,10001,500)),selection='all_8_tune_sources_7_blocks_LF1_min_finite_tie_earlier',
        population_binding={'artifacts':artifacts,'census_sha256':'1'*64,'component_map_sha256':'2'*64,
                            'configuration':{'census':str(base/'projection.jsonl'),'component_map':str(base/'source_components.json')}})
    plan_ref=save(arm.parent/'frozen_plan.json',plan)
    candidates=[dict(step=i,objective=1.,file=f'candidate_{i:06d}.pt') for i in plan['candidate_steps']]
    selected=candidates[0]
    cp=arm/selected['file'];cp.write_bytes(b'ARTIFICIAL CHECKPOINT BYTES, NEVER LOADED')
    cp_ref={'path':str(cp),'sha256':s.digest(cp)}
    sel=dict(schema=s.NTR1,status='complete_training',seed=20260815,variant='canonical',
        plan_sha256=s.json_sha(plan),selected=selected,candidates=candidates,
        initial_state_sha256='5'*64,draw_ledger_sha256=plan['draw_ledger_sha256'])
    sel_ref=save(arm/'selection.json',sel)
    pred=dict(schema=s.NTR1,status='complete_population',plan_sha256=s.json_sha(plan),
              selected=selected,records=records,natural_GT_accessed=False)
    pred_ref=save(arm/'predictions.json',pred)
    freeze_ref=save(base/'execution_freeze.json',dict(source_commit='9'*40,
        gate={'source_commit':'9'*40,'threeway_checkpoint_status':'verified','artifacts':artifacts}))
    return dict(protocol=s.SCHEMA,seed=20260815,variant='canonical',source_commit='9'*40,
        prediction_root=str(arm),output=str(base/'score'),
        inputs=dict(diagnostics=diag_ref,identity_rows=rows_ref,predictions=pred_ref,selection=sel_ref,
                    plan=plan_ref,count_policy=policy_ref,checkpoint=cp_ref,execution_freeze=freeze_ref))


@pytest.fixture
def case(template, tmp_path):
    old=Path(template['prediction_root']).parents[2]
    new=tmp_path/'case'
    shutil.copytree(old,new)
    old_text,new_text=str(old),str(new)
    for path in new.rglob('*.json*'):
        path.write_text(path.read_text().replace(old_text,new_text))
    cfg=json.loads(json.dumps(template).replace(old_text,new_text))
    refresh(cfg)
    return cfg


def refresh(cfg):
    for name in ('identity_rows','count_policy'):
        cfg['inputs'][name]['sha256']=s.digest(cfg['inputs'][name]['path'])
    diag=s.read_json(cfg['inputs']['diagnostics']['path'])
    diag['receipt']['identity_rows_sha256']=cfg['inputs']['identity_rows']['sha256']
    cfg['inputs']['diagnostics']=save(Path(cfg['inputs']['diagnostics']['path']),diag)
    plan=s.read_json(cfg['inputs']['plan']['path'])
    for name in ('diagnostics','count_policy'):
        plan['population_binding']['artifacts'][name]=cfg['inputs'][name]
    cfg['inputs']['plan']=save(Path(cfg['inputs']['plan']['path']),plan)
    plan_sha=s.json_sha(plan)
    for name in ('predictions','selection'):
        d=s.read_json(cfg['inputs'][name]['path']);d['plan_sha256']=plan_sha
        cfg['inputs'][name]=save(Path(cfg['inputs'][name]['path']),d)
    freeze=s.read_json(cfg['inputs']['execution_freeze']['path'])
    freeze['gate']['artifacts']=plan['population_binding']['artifacts']
    cfg['inputs']['execution_freeze']=save(Path(cfg['inputs']['execution_freeze']['path']),freeze)


def mutate(cfg,name,fn,refresh_all=False):
    path=Path(cfg['inputs'][name]['path']);d=s.read_json(path);fn(d)
    cfg['inputs'][name]=save(path,d)
    if refresh_all:refresh(cfg)


def test_full_join_abstention_and_unequal_video_macro(case):
    raw,result=s.score(case)
    assert len(raw)==486 and result['full_population']['videos']==183
    assert raw[0]['count']==0 and raw[0]['raw_prediction_count']==-1 and raw[0]['prediction_status']=='abstain'
    grouped={}
    for r in raw:
        g,c=r['gt_count'],r['count'];grouped.setdefault(r['media_key'],[]).append((abs(c-g)/g,abs(c-g),abs(c-g)<=1,c-g,(c-g)/g,c/g))
    expected=[sum(sum(row[j] for row in rows)/len(rows) for rows in grouped.values())/183 for j in range(6)]
    for key,value in zip(s.POLICY['metrics'],expected):assert result['full_population'][key]==pytest.approx(value)
    assert result['full_population']['nmae'] != pytest.approx(sum(abs(r['count']-r['gt_count'])/r['gt_count'] for r in raw)/486)


def test_cli_spine_emits_bound_artifact(case):
    cfg=Path(case['output']).parent/'cli_config.json';save(cfg,case)
    proc=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/score_temporac_native_predictions.py'),'--config',str(cfg)],
        capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert proc.returncode==0,proc.stderr
    out=Path(case['output']);receipt=s.read_json(out/'receipt.json')
    assert receipt['score_sha256']==s.digest(out/'score.json')
    assert len((out/'scored_rows.jsonl').read_text().splitlines())==486


@pytest.mark.parametrize('value',[-2,-1,1.5,True,float('nan'),float('inf')])
def test_bad_nonabstain_count_rejected(case,value):
    mutate(case,'predictions',lambda d:d['records'][1].update(count=value))
    with pytest.raises((s.ScoreError,ValueError)):s.run(case)
    assert (Path(case['output'])/'failure.json').is_file()
    assert not (Path(case['output'])/'score.json').exists()


@pytest.mark.parametrize('field,value',[('count',0),('status','unknown'),('decoder_invocations',2)])
def test_abstain_contract_rejected(case,field,value):
    mutate(case,'predictions',lambda d:d['records'][0].update({field:value}))
    with pytest.raises(ValueError):s.score(case)


@pytest.mark.parametrize('kind',['missing','duplicate','unknown','partial_run'])
def test_bad_prediction_population_rejected(case,kind):
    def change(d):
        if kind=='missing':d['records'].pop()
        elif kind=='duplicate':d['records'][-1]=d['records'][0]
        elif kind=='unknown':d['records'][-1]['key']='unknown_0'
        else:d['status']='partial_population'
    mutate(case,'predictions',change)
    with pytest.raises(ValueError):s.score(case)


@pytest.mark.parametrize('kind',['plan','selected','not_minimum','checkpoint','policy','diagnostic_rows','diagnostic_inputs','denominator','NPZ','count_components','source'])
def test_cross_bindings_and_denominators_reject(case,kind):
    if kind=='plan':mutate(case,'predictions',lambda d:d.update(plan_sha256='0'*64))
    elif kind=='selected':mutate(case,'predictions',lambda d:d['selected'].update(step=1000))
    elif kind=='not_minimum':mutate(case,'selection',lambda d:d['candidates'][1].update(objective=.1))
    elif kind=='checkpoint':Path(case['inputs']['checkpoint']['path']).write_bytes(b'changed')
    elif kind=='policy':mutate(case,'count_policy',lambda d:d.update(threshold=.4),True)
    elif kind=='diagnostic_rows':
        p=Path(case['inputs']['identity_rows']['path']);p.write_text(p.read_text()+'{}\n')
    elif kind=='diagnostic_inputs':mutate(case,'diagnostics',lambda d:d['receipt']['inputs_sha256'].update({next(iter(d['receipt']['inputs_sha256'])):'0'*64}),True)
    elif kind=='denominator':mutate(case,'diagnostics',lambda d:d['splits']['val'].update(identity_denominator=485),True)
    elif kind=='NPZ':
        p=next((Path(case['prediction_root'])/'predictions').glob('*.npz'));p.write_bytes(b'changed')
    elif kind=='count_components':mutate(case,'predictions',lambda d:d['records'][1].update(count=9))
    elif kind=='source':mutate(case,'execution_freeze',lambda d:d.update(source_commit='0'*40))
    with pytest.raises((s.ScoreError,ValueError)):s.score(case)


def test_no_artificial_gt_zero_coercion(case):
    p=Path(case['inputs']['identity_rows']['path']);rows=[json.loads(x) for x in p.read_text().splitlines()]
    next(r for r in rows if r['split']=='val')['gt_count']=0
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows));refresh(case)
    with pytest.raises(ValueError):s.score(case)


@pytest.mark.parametrize('field,value',[
    ('run_bounds',np.asarray([[50,3]],np.int32)),
    ('component_bounds',np.asarray([[-9,-10]],np.int32)),
    ('event_source_positions',np.asarray([float('inf')],np.float64)),
    ('event_seconds',np.asarray([float('inf')],np.float64))])
def test_malformed_bound_npz_fails_with_receipt(case,field,value):
    p=s.read_json(case['inputs']['predictions']['path']);record=p['records'][1]
    path=Path(case['prediction_root'])/'predictions'/(record['key']+'.npz')
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays[field]=value;np.savez_compressed(path,**arrays)
    record['sha256']=s.digest(path)
    case['inputs']['predictions']=save(Path(case['inputs']['predictions']['path']),p)
    with pytest.raises(ValueError):s.run(case)
    assert (Path(case['output'])/'failure.json').is_file()


def test_truncated_zip_fails_with_receipt(case):
    p=s.read_json(case['inputs']['predictions']['path']);record=p['records'][0]
    path=Path(case['prediction_root'])/'predictions'/(record['key']+'.npz')
    path.write_bytes(path.read_bytes()[:80]);record['sha256']=s.digest(path)
    case['inputs']['predictions']=save(Path(case['inputs']['predictions']['path']),p)
    with pytest.raises(ValueError):s.run(case)
    assert (Path(case['output'])/'failure.json').is_file()

