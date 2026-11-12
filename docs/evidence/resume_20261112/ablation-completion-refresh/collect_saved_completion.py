"""Read saved ablation artifacts only. No model imports, inference, training or process control."""
from pathlib import Path
import datetime, hashlib, json, math, os, shutil, subprocess
R = Path('/root/autodl-tmp/TempoRAC')
D = R/'resume20261112/ablation-completion-refresh'
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
all_complete=sorted(p.parent.name for p in O.glob('*/complete_known_source.json'))
excluded=['wo_pyramid_s0']
complete=[n for n in all_complete if n not in excluded]
assert complete==['wo_pyramid_s1','wo_shift_s0'], complete
membership={s: sorted(f"{x['split']}__{x['video_id']}" for x in rows if x['split']==s) for s in ['rc_test','cx_test','ucf_test']}
assert [len(membership[s]) for s in membership]==[152,2403,105]
dump(D/'FROZEN_EVALUATION_MEMBERSHIP.json', {'source_manifest_path':str(manpath),'source_manifest_sha256':sha(manpath),'keys':membership})
receipt={'schema':'temporac.completed_ablation_refresh.v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_executor_model':'gpt-6.1-sol','requested_reasoning':'high','actual_backend_attested':False,'semantic_assurance':'same-family provisional','scope':'Saved-output deterministic integrity collection; no performance ranking or component judgment','new_training':False,'new_inference':False,'model_loading':False,'new_tests':False,'process_control':False,'control_files_written':False,'completed_count':len(complete),'all_completed_count_at_collection':len(all_complete),'all_completed_names':all_complete,'excluded_previously_archived':excluded,'completed_names':complete,'arms':[],'checks':[]}
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
    result=read(d/'result.json')
    evals=[]
    for job in prov['evaluations']:
        ev=read(d/('eval_'+job['tag']+'.json'))
        expected={f"{x['split']}__{x['video_id']}" for x in rows if x['split']==job['split']}
        assert len(expected)==job['n'] and ev['n']==job['n'] and ev['metrics_tta']['n_pos']==job['n_positive']
        assert set(ev['predictions'])==set(ev['predictions_shift'])==expected
        assert all(math.isfinite(v) for v in [*ev['predictions'].values(),*ev['predictions_shift'].values()])
        assert ev['ckpt']==job['checkpoint'] and ev['split']==job['split']
        selected=min(logs, key=lambda z:(z['val_all'][job['target']]['mae_p'],-z['val_all'][job['target']]['obo_p']))
        assert result['targets'][job['target']]['best_epoch']==selected['epoch']
        assert result['targets'][job['target']]['val']==selected['val_all'][job['target']]
        evals.append({'tag':job['tag'],'n':ev['n'],'n_positive':ev['metrics_tta']['n_pos'],'checkpoint':ev['ckpt'],'checkpoint_sha256':hashes[job['checkpoint']]['actual'],'selected_epoch':selected['epoch'],'selection_key':[selected['val_all'][job['target']]['mae_p'],-selected['val_all'][job['target']]['obo_p']],'selection_rule':'min validation (mae_p, -obo_p), first tie retained','metrics_saved':{k:v for k,v in ev.items() if k.startswith('metrics')},'both_views_exact_full_keys':True,'prediction_values_finite':True,'full_keys_sha256':canon(sorted(expected))})
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
    workers[-1]['latest_event_status']=events[-1]['status']
    workers[-1]['worker_cmdline']=(Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace') if Path(f'/proc/{pid}/cmdline').exists() else None)
receipt['workers']=workers
receipt['lane_snapshot_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
proc_snapshot=[]
for pd in Path('/proc').iterdir():
    if not pd.name.isdigit(): continue
    try:
        cmd=(pd/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        if str(R) not in cmd or not ('temporac_cvpr.train' in cmd or 'transrac_native_v1/run.py' in cmd or 'ablation_worker.py' in cmd): continue
        status=(pd/'status').read_text()
        ppid=int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
        proc_snapshot.append({'pid':int(pd.name),'ppid':ppid,'cmdline':cmd,'cwd':os.readlink(pd/'cwd')})
    except (FileNotFoundError,PermissionError,ProcessLookupError): pass
receipt['active_project_processes']=proc_snapshot
p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0
receipt['gpu_snapshot_csv']=p.stdout
p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
assert p.returncode==0
receipt['gpu_compute_processes_csv']=p.stdout
dump(D/'SAVED_METRICS.json',{a['name']:{e['tag']:e['metrics_saved'] for e in a['evaluations']} for a in receipt['arms']})
copy(Path(__file__),Path('collector_source.py'))
dump(D/'SKILL_READ_RECEIPT.json',{'stage':'monitor-experiment','full_read':True,'locked_aris_commit':'f76ec8d6b77792e2242cca049e08d0cbb7d54e19','pinned_source_path':'E:/ICASSP_TempoRAC/aris-work/aris-pinned/skills/skills-codex/monitor-experiment/SKILL.md','installed_copy_path':'E:/ICASSP_TempoRAC/TempoRAC-CVPR/.agents/skills/monitor-experiment/SKILL.md','both_sha256':'66a42b21a7ec5a9c4ee2d4da6babfd533764d6c4188dcc178d5bcbd1d8a31072','scope_override':'Finite saved-evidence integrity collection only, no comparisons, group inference or external notifications'})
receipt['checks']=[{'requirement':'Preserve workers and frozen scope','status':'pass','evidence':'Read-only /proc and saved artifacts; collector writes confined to collection directory; no HOLD, start, signal, restart or migration'}, {'requirement':'New completed arms exist and exact50 epochs','status':'pass' if complete else 'pending','evidence':'arms[].logged_epochs and exact range0..49'}, {'requirement':'Config/provenance/source/contract exact','status':'pass' if complete else 'pending','evidence':'arms[].config_and_provenance_exact plus saved raw configuration and provenance'}, {'requirement':'Full fixed populations and prediction keys','status':'pass' if complete else 'pending','evidence':'arms[].evaluations; exact manifest key sets for both views'}, {'requirement':'Artifact SHA integrity','status':'pass' if complete else 'pending','evidence':'arms[].artifact_hashes; all completion-listed artifacts hashed server-only'}, {'requirement':'Component contribution / performance ranking','status':'pending','evidence':'Intentionally no scientific conclusion before paired3seed closure'}]
receipt['completion_coverage_note']='Two new completed arms beyond prior archive wo_pyramid_s0; no paired3seed inference or single-arm verdict.'
dump(D/'COMPLETION_RECEIPT.json',receipt)
acceptance=['# Acceptance evidence — completion refresh', '', '| Requirement | Status | Evidence |','|---|---|---|']
for c in receipt['checks']:
    acceptance.append('| '+c['requirement']+' | '+c['status']+' | '+c['evidence']+' |')
acceptance += ['| Validation-only checkpoint selection | pass | arms[].evaluations selected_epoch and selection_key verified against all 50 raw log rows and result.json |','| Prior arm excluded | pass | excluded_previously_archived = wo_pyramid_s0; no re-collection |','| Current lanes recorded | pass | lane_snapshot_utc, workers, active_project_processes, GPU CSV |','| Local transfer SHA | pending | LOCAL_TRANSFER_VERIFICATION.json added locally after transfer |','','No model deserialization, training, inference, checkpoint copying or process control occurred.']
(D/'ACCEPTANCE_RESULTS.md').write_text('\n'.join(acceptance)+'\n')
manifest={str(p.relative_to(D)): {'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='COLLECTION_SHA256.json'}
dump(D/'COLLECTION_SHA256.json',manifest)
print(json.dumps({'observed_utc':receipt['observed_utc'],'completed_count':len(complete),'all_completed_count_at_collection':len(all_complete),'all_completed_names':all_complete,'excluded_previously_archived':excluded,'completed_names':complete,'workers':workers,'local_copy_source':str(D),'raw_file_count':len(list((D/'raw').rglob('*'))),'receipt_sha256':sha(D/'COMPLETION_RECEIPT.json')},indent=2))

