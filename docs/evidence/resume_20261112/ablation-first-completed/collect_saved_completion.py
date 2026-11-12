"""Read saved ablation artifacts only. No model imports, inference, training or process control."""
from pathlib import Path
import datetime, hashlib, json, math, os, shutil
R = Path('/root/autodl-tmp/TempoRAC')
D = R/'resume20261112/ablation-first-completed'
C = R/'resume20261112/ablation-continuation'
O = R/'runs/frozen_known_source_v1_ablation20261112'
D.mkdir(parents=True, exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(p.read_text())
def canon(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,o): p.write_text(json.dumps(o,indent=2)+'\n')
def copy(p,rel):
    q=D/'raw'/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
contract=read(C/'RUN_CONTRACT.json')
assert sha(C/'RUN_CONTRACT.json')=='dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167'
assert sha(C/'ablation_worker.py')=='d23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144'
specpath=Path(contract['frozen_spec']['path']); manpath=Path(contract['manifest']['path'])
assert sha(specpath)==contract['frozen_spec']['sha256']
assert sha(manpath)==contract['manifest']['sha256']
spec={x['name']:x for x in read(specpath)['runs']}
rows=read(manpath)
complete=sorted(p.parent.name for p in O.glob('*/complete_known_source.json'))
receipt={'schema':'temporac.first_completed_ablation_collection.v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_executor_model':'gpt-6.1-sol','requested_reasoning':'high','actual_backend_attested':False,'semantic_assurance':'same-family provisional','scope':'Saved-output deterministic integrity collection; no performance ranking or component judgment','new_training':False,'new_inference':False,'model_loading':False,'new_tests':False,'process_control':False,'control_files_written':False,'completed_count':len(complete),'completed_names':complete,'arms':[],'checks':[]}
for name in complete:
    d=O/name; comp=read(d/'complete_known_source.json'); prov=read(C/(name+'.provenance.json')); r=spec[name]
    approved=next(x for x in contract['new_training_runs'] if x['name']==name)
    assert prov==comp['provenance']
    assert prov['contract_sha256']==sha(C/'RUN_CONTRACT.json') and prov['worker_sha256']==sha(C/'ablation_worker.py')
    assert prov['source_commit']==contract['training_runtime_commit']=='489dc77b91a21d21215784ff49961edb20dab4fb'
    assert prov['spec_sha256']==sha(specpath) and prov['manifest_sha256']==sha(manpath)
    assert prov['cfg']==r['cfg'] and prov['cfg_sha256']==canon(r['cfg'])==approved['cfg_sha256']
    assert prov['seed']==r['seed']==approved['seed']
    assert sha(Path(prov['source'])/'temporac_cvpr/train.py')=='33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4'
    for k in ['features','train_split','val_split','test_split']: assert prov[k]==r[k]==approved[k]
    cfg=read(d/'config.json'); assert cfg['cfg']==r['cfg'] and cfg['args']['seed']==r['seed'] and not cfg['args']['with_test']
    for k in ['manifest','features','train_split','val_split','test_split']: assert cfg['args'][k]==r[k]
    logs=[json.loads(s) for s in (d/'log.jsonl').read_text().splitlines() if s.strip()]
    assert [x['epoch'] for x in logs]==list(range(approved['epochs'])) and len(logs)==comp['epochs']==50
    assert all(x.get('nonfinite_steps',0)==0 for x in logs)
    assert all(math.isfinite(v) for x in logs for k,v in x.items() if k.startswith('l_'))
    first=read(d/'first_checkpoint_receipt.json'); assert first['provenance']==prov
    hashes={f:{'expected':v,'actual':sha(d/f),'bytes':(d/f).stat().st_size} for f,v in comp['file_sha256'].items()}
    assert all(x['expected']==x['actual'] for x in hashes.values())
    evals=[]
    for job in prov['evaluations']:
        ev=read(d/('eval_'+job['tag']+'.json'))
        expected={f"{x['split']}__{x['video_id']}" for x in rows if x['split']==job['split']}
        assert len(expected)==job['n'] and ev['n']==job['n'] and ev['metrics_tta']['n_pos']==job['n_positive']
        assert set(ev['predictions'])==set(ev['predictions_shift'])==expected
        assert all(math.isfinite(v) for v in [*ev['predictions'].values(),*ev['predictions_shift'].values()])
        assert ev['ckpt']==job['checkpoint']
        evals.append({'tag':job['tag'],'n':ev['n'],'n_positive':ev['metrics_tta']['n_pos'],'checkpoint':ev['ckpt'],'both_views_exact_full_keys':True,'prediction_values_finite':True,'full_keys_sha256':canon(sorted(expected))})
    required={'config.json','log.jsonl','result.json'}|{j['checkpoint'] for j in prov['evaluations']}|{'eval_'+j['tag']+'.json' for j in prov['evaluations']}
    assert set(comp['file_sha256'])==required
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix in {'.json','.jsonl','.log'}:
            assert p.stat().st_size<2000000
            copy(p,Path(name)/p.name)
    copy(C/(name+'.provenance.json'),Path('control')/(name+'.provenance.json'))
    receipt['arms'].append({'name':name,'completed_utc':comp['completed_utc'],'logged_epochs':len(logs),'logged_epoch_range':[logs[0]['epoch'],logs[-1]['epoch']],'contiguous_epochs':True,'nonfinite_steps':0,'loss_fields_finite':True,'config_and_provenance_exact':True,'first_checkpoint_provenance_exact':True,'source_train_file_sha256':sha(Path(prov['source'])/'temporac_cvpr/train.py'),'evaluations':evals,'artifact_hashes':hashes,'completion_receipt_sha256':sha(d/'complete_known_source.json')})
for p in [C/'RUN_CONTRACT.json',C/'ablation_worker.py',C/'LAUNCH_RECEIPT.json',C/'launch_gpu0.json',C/'launch_gpu1.json',C/'worker_gpu0.events.jsonl',C/'worker_gpu1.events.jsonl']:
    copy(p,Path('control')/p.name)
copy(specpath,Path('control')/'frozen_27_spec.json')
workers=[]
for gpu in [0,1]:
    pid=int((C/f'worker_gpu{gpu}.pid').read_text()); events=[json.loads(s) for s in (C/f'worker_gpu{gpu}.events.jsonl').read_text().splitlines() if s.strip()]
    last=next((e for e in reversed(events) if e['status']=='started'),{})
    name=last.get('run'); d=O/name if name else None
    logs=[json.loads(s) for s in (d/'log.jsonl').read_text().splitlines() if s.strip()] if d and (d/'log.jsonl').exists() else []
    child=last.get('pid')
    workers.append({'gpu':gpu,'worker_pid':pid,'worker_alive':Path(f'/proc/{pid}').exists(),'current_arm_from_latest_started_event':name,'phase':last.get('phase'),'child_pid':child,'child_alive':Path(f'/proc/{child}').exists() if child else None,'logged_epochs':len(logs),'last_logged_epoch':logs[-1]['epoch'] if logs else None,'arm_has_completion_receipt':(d/'complete_known_source.json').exists() if d else False,'latest_event_utc':events[-1]['utc']})
receipt['workers']=workers
receipt['checks']=[{'requirement':'Preserve workers and frozen scope','status':'pass','evidence':'Read-only /proc and saved artifacts; collector writes confined to collection directory; no HOLD, start, signal, restart or migration'}, {'requirement':'First completed arms exist and exact50 epochs','status':'pass' if complete else 'pending','evidence':'arms[].logged_epochs and exact range0..49'}, {'requirement':'Config/provenance/source/contract exact','status':'pass' if complete else 'pending','evidence':'arms[].config_and_provenance_exact plus saved raw configuration and provenance'}, {'requirement':'Full fixed populations and prediction keys','status':'pass' if complete else 'pending','evidence':'arms[].evaluations; exact manifest key sets for both views'}, {'requirement':'Artifact SHA integrity','status':'pass' if complete else 'pending','evidence':'arms[].artifact_hashes; all completion-listed artifacts hashed server-only'}, {'requirement':'Component contribution / performance ranking','status':'pending','evidence':'Intentionally no scientific conclusion before paired3seed closure'}]
dump(D/'COMPLETION_RECEIPT.json',receipt)
manifest={str(p.relative_to(D)): {'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='COLLECTION_SHA256.json'}
dump(D/'COLLECTION_SHA256.json',manifest)
print(json.dumps({'observed_utc':receipt['observed_utc'],'completed_count':len(complete),'completed_names':complete,'workers':workers,'local_copy_source':str(D),'raw_file_count':len(list((D/'raw').rglob('*'))),'receipt_sha256':sha(D/'COMPLETION_RECEIPT.json')},indent=2))
