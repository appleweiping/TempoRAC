"""Explicit one-shot GPU1 residual after the canonical full RACnet arm. No waiter/retry."""
import argparse, contextlib, datetime, fcntl, hashlib, importlib.util, json, math, os
from pathlib import Path
import shutil, subprocess, sys, time
import xml.etree.ElementTree as ET
sys.dont_write_bytecode = True
R = Path('/root/autodl-tmp/TempoRAC')
A = R/'resume20261112/ablation-continuation'
W = R/'baselines/racnet_native_v1'
D = R/'resume20261113/ablation-after-racnet/runtime'
TRAIN = '489dc77b91a21d21215784ff49961edb20dab4fb'
SOURCE = R/'source_snapshots'/TRAIN
OUTPUTS = R/'runs/frozen_known_source_v1_ablation20261112'
CONTRACT = 'dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167'
WORKER = 'd23bdd79502063ba4488b08c829f28fa92c8b0299f938f79cf7ae3a173d1a144'
HOLD = '8017bcc146e537b408af49f2baa62567c66af6575c5eb905a0678ebbb4a00a40'
HELD = '71cf338ea8ad0fd28c0976bf7832753d883cd842ea85531c11bdcd5ba0cb00ff'
LAUNCH = '858fa67c3efd06c5e1c689393564e38d874b7c76756c0db542e0ac712c4c745e'
RESIDUAL = ['wo_token_s2','wo_joint_cx_s0','wo_joint_cx_s1','wo_joint_cx_s2']
SEQUENCE = ['extract-quva','refresh-inputs','train','evaluate']
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

def require(ok, message):
    if not ok: raise RuntimeError(message)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def live(pid):
    path = Path('/proc',str(pid),'stat')
    return path.exists() and path.read_text().rsplit(')',1)[1].split()[0] != 'Z'
def owned_live(pid,command):
    if not live(pid): return False
    actual = Path('/proc',str(pid),'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    return actual == command

def out(cmd): return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,creationflags=FLAGS,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')).strip()
def sync_dir(path):
    fd = os.open(path,os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
def atomic(path,value):
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); sync_dir(path.parent)
def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj

def racnet_complete(common, finite, launch, pipeline):
    """Only JSON metadata and file stat; no array/checkpoint/model loading."""
    identity = common.identity(W)
    require(launch['candidate_identity'] == identity, 'RAC source differs from canonical launch')
    require(sha(launch['release']['path']) == launch['release']['sha256'], 'RAC release changed')
    release = read(launch['release']['path'])
    runner = Path(launch['command'][2])
    require(sha(runner) == launch['runner_sha256'] == pipeline['runner_sha256'], 'RAC finite runner changed')
    require(release['candidate_identity'] == identity and release['gpu_index'] == launch['gpu_index'] == 1 and release['gpu_uuid'] == launch['gpu_uuid'], 'RAC release identity/GPU mismatch')
    for name in ('immutable_publication_receipt','fresh_review_receipt','current_arm_complete_receipt'):
        ref = release[name]; require(sha(ref['path']) == ref['sha256'], 'RAC approval/prior receipt changed: '+name)
    publication = read(release['immutable_publication_receipt']['path'])
    review = read(release['fresh_review_receipt']['path'])
    require(publication['candidate_identity'] == review['candidate_identity'] == identity and review['verdict'] in ('PASS','PROVISIONAL_PASS'), 'RAC publication/review identity mismatch')
    require(publication['runner_sha256'] == review['runner_sha256'] == launch['runner_sha256'], 'RAC reviewed runner differs')
    require(pipeline['status'] == 'COMPLETE' and pipeline['candidate_identity'] == identity and pipeline['root_release_sha256'] == launch['release']['sha256'], 'RAC pipeline not canonical COMPLETE')
    stages = pipeline['stages']
    require([s['stage'] for s in stages] == SEQUENCE, 'RAC stage order/count mismatch')
    for stage in stages:
        require((stage['status'] == 'COMPLETE' and stage['exit_code'] == 0) or (stage['status'] == 'REUSED_COMPLETE' and stage['exit_code'] is None), 'RAC stage not successful: '+stage['stage'])
    require(finite.feature_complete(W,common) and finite.inputs_complete(W,common) and finite.train_complete(W,common) and finite.evaluation_complete(W,common), 'RAC full-stage receipt qualification failed')
    identity_train = finite.training_identity(W,common)
    train = read(W/'TRAINING_COMPLETE.json'); commit = read(W/'checkpoints/COMMIT.json')
    require(train['actual_steps'] + train['amp_skips'] == train['attempted_steps'] == 9600, 'AMP accounting incomplete')
    require(all(isinstance(train[x],int) and train[x] >= 0 for x in ('actual_steps','amp_skips')), 'Invalid update/overflow counters')
    require(commit['latest']['next_epoch'] == 200, 'RAC latest commit not full200')
    epochs = [read(W/'epochs'/('epoch_%03d.json'%i)) for i in range(200)]
    for i,e in enumerate(epochs):
        require(e['epoch'] == i and e['training_rows'] == 755 and e['attempts'] == 48 and len(e['batches']) == 48, 'RAC epoch metadata incomplete')
        require(e['actual_updates'] + e['amp_skips'] == 48 and e['actual_updates'] == sum(int(b['actual_update']) for b in e['batches']), 'RAC epoch update accounting')
        require((e['validation'] is not None) == (i > 10), 'RAC189 validation eligibility changed')
    require(sum(e['actual_updates'] for e in epochs) == train['actual_steps'] and sum(e['amp_skips'] for e in epochs) == train['amp_skips'], 'RAC cumulative update accounting')
    candidates = [e for e in epochs if e['validation'] is not None]
    require(len(candidates) == 189 and all(math.isfinite(e['validation']['MAE']) and math.isfinite(e['validation']['OBO']) for e in candidates), 'RAC native validation candidates invalid')
    best = min(candidates,key=lambda e:(e['validation']['MAE'],e['epoch']))
    require(train['selection'] == {'epoch':best['epoch'],'value':best['validation']['MAE']} and train['tie_break'] == 'earliest epoch', 'RAC selected checkpoint not native bestMAE')
    require(train['selected'] == commit['best_mae'] and train['selected']['next_epoch'] == best['epoch']+1, 'RAC checkpoint selection mismatch')
    for name in ('best_mae','latest'):
        path = Path(commit[name]['file'])
        require(path.resolve().is_relative_to(W/'checkpoints') and path.is_file() and path.stat().st_size > 0, 'RAC committed payload missing')
    done = read(W/'COMPLETE.json'); predictions = read(W/'PREDICTIONS_357.json')
    expected = read(common.input_path(W,'prediction_inputs'))
    require(done['binding']['identity'] == identity_train and done['binding']['checkpoint_sha256'] == train['selected']['sha256'], 'RAC final checkpoint binding mismatch')
    require(pipeline['final_complete_sha256'] == sha(W/'COMPLETE.json'), 'RAC pipeline final receipt changed')
    require(len(predictions) == len({r['key'] for r in predictions}) == len(expected) == 357 and {r['key'] for r in predictions} == {r['key'] for r in expected}, 'RAC final357 membership mismatch')
    require({p:sum(r['population']==p for r in predictions) for p in ('rc_test','ucf_test','quva_test')} == {'rc_test':152,'ucf_test':105,'quva_test':100}, 'RAC final population mismatch')
    refs = {r['key']:r for r in expected}
    for row in predictions:
        require(row['binding'] == done['binding'] and row['input_sha256'] == refs[row['key']]['sha256'] and not row['nonfinite'] and math.isfinite(row['count']), 'RAC saved prediction identity/nonfinite')
    return dict(candidate_identity=identity,train200=True,attempted_steps=9600,actual_updates=train['actual_steps'],amp_skips=train['amp_skips'],native_validation_candidates=189,predictions=357,selected=train['selected'],selection=train['selection'],final_complete_sha256=sha(W/'COMPLETE.json'))

def ablation_control():
    for name,digest in [('ablation_worker.py',WORKER),('RUN_CONTRACT.json',CONTRACT),('HOLD_AFTER_CURRENT_GPU1',HOLD),('lane_gpu1.json',HELD),('launch_gpu1.json',LAUNCH)]:
        require(sha(A/name)==digest,'Frozen GPU1 control changed: '+name)
    require(out(['git','-C',str(SOURCE),'rev-parse','HEAD']) == TRAIN and not out(['git','-C',str(SOURCE),'status','--porcelain']), 'Original training source changed/dirty')
    require(sha(SOURCE/'temporac_cvpr/train.py') == '33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4','Original train runtime changed')
    c,held,launch = read(A/'RUN_CONTRACT.json'),read(A/'lane_gpu1.json'),read(A/'launch_gpu1.json')
    require(held['status']=='held_after_complete' and held['remaining']==RESIDUAL and held['gpu']==1 and held['contract_sha256']==CONTRACT and held['worker_sha256']==WORKER and held['source_commit']==TRAIN,'Exact four-arm hold mismatch')
    require(not owned_live(int((A/'worker_gpu1.pid').read_text()),launch['command']),'Original GPU1 worker still alive')
    spec,manifest=Path(c['frozen_spec']['path']),Path(c['manifest']['path'])
    require(sha(spec)==c['frozen_spec']['sha256'] and sha(manifest)==c['manifest']['sha256'],'Original spec/manifest changed')
    worker=module('frozen_gpu1_worker',A/'ablation_worker.py'); plans=worker.checked_plans(c,read(spec)); lane=c['lanes']['gpu1']
    require(lane==launch['lane'] and lane[-4:]==RESIDUAL,'Original lane changed')
    # Reuse prior completion evidence; unchanged worker --resume performs its original payload checks.
    for name in lane[:-4]:
        got=read(OUTPUTS/name/'complete_known_source.json'); plan=plans[name]; r=plan['run']
        expected=dict(run=name,source=str(SOURCE),source_commit=TRAIN,worker_sha256=WORKER,contract_sha256=CONTRACT,spec_sha256=c['frozen_spec']['sha256'],manifest_sha256=c['manifest']['sha256'],seed=r['seed'],cfg=r['cfg'],cfg_sha256=worker.cfg_sha(r['cfg']),features=r['features'],train_split=r['train_split'],val_split=r['val_split'],test_split=r['test_split'],evaluations=plan['evaluations'])
        require(got['provenance']==expected and got['epochs']==r['cfg']['epochs'],'Completed prefix receipt mismatch: '+name)
    require(all(not (OUTPUTS/name).exists() for name in RESIDUAL),'Residual partial/output exists; manual recovery required')
    events=[json.loads(x) for x in (A/'worker_gpu1.events.jsonl').read_text().splitlines() if x.strip()]
    require(events[-1]['status']=='held_after_complete' and events[-1]['remaining']==RESIDUAL,'Original GPU1 history changed')
    for e in events:
        if e['status']=='started': require(not owned_live(e['pid'],e['command']),'Original GPU1 child still alive')
    command=[str(R/'env/bin/python'),'-u','-B',str(A/'ablation_worker.py'),'--gpu','1','--source',str(SOURCE),'--commit',TRAIN,'--contract',str(A/'RUN_CONTRACT.json')]
    require(launch['command']==command,'Original worker command changed')
    return dict(pending=RESIDUAL,completed_prefix=lane[:-4],command=command+['--resume'],worker_sha256=WORKER,ablation_contract_sha256=CONTRACT,hold_sha256=HOLD,events_sha256=sha(A/'worker_gpu1.events.jsonl'))

def assess(stack):
    require(os.name!='nt' and R.resolve()==R,'Server-only fixed project root')
    require(not (D/'one_shot.json').exists(),'Existing one-shot claim; no automatic repeat')
    proof=ablation_control()
    launch_path=W/'GPU_PRODUCTION_LAUNCH.json'
    if not launch_path.exists(): return dict(proof,status='WAIT',reason='Canonical RACnet has not launched')
    launch=read(launch_path)
    if owned_live(launch['pid'],launch['command']): return dict(proof,status='WAIT',reason='Canonical RACnet root still alive')
    matches=[]
    for path in (W/'finite_pipeline').glob('*/PIPELINE_COMPLETE.json'):
        value=read(path)
        if value['root_release_sha256']==launch['release']['sha256']: matches.append((path,value))
    require(len(matches)==1,'No unique complete canonical RACnet pipeline; preserve failure/partial state')
    path,pipeline=matches[0]
    require(read(path.parent/'PIPELINE_STATE.json')==pipeline,'Canonical RACnet pipeline state differs from COMPLETE')
    for stage in pipeline['stages']:
        if 'pid' in stage: require(not owned_live(stage['pid'],stage['argv']),'RACnet child still alive')
    for name in ('FINITE_PIPELINE.lock','ACTIVE.lock'):
        lock=stack.enter_context((W/name).open('rb'))
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return dict(proof,status='WAIT',reason='RACnet writer lock held: '+name)
    runner=Path(launch['command'][2]).resolve()
    require(runner.is_relative_to(R) and sha(runner)==launch['runner_sha256'],'Canonical RAC runner source mismatch')
    require(sha(W/'native_common.py')==launch['candidate_identity']['code']['native_common.py'],'Canonical RAC metadata source changed')
    common=module('racnet_gate_metadata',W/'native_common.py')
    finite=module('racnet_finite_receipts',Path(launch['command'][2]))
    facts=racnet_complete(common,finite,launch,pipeline)
    free=shutil.disk_usage(R).free
    require(free>=137438953472,'Shared128GiB storage reserve unmet')
    gpu=out(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits']).split(',')
    require(len(gpu)==3 and int(gpu[0])==1 and gpu[1].strip()==launch['gpu_uuid'],'Physical GPU1 identity mismatch')
    xml=ET.fromstring(out(['nvidia-smi','-i','1','-q','-x'])); require(len(xml.findall('gpu'))==1,'Ambiguous GPU1 resource query')
    processes=xml.findall('.//processes/process_info')
    if int(gpu[2])>=500 or processes: return dict(proof,status='WAIT',reason='GPU1 occupied; no signal/retry',gpu1_memory_mib=int(gpu[2]),gpu1_process_count=len(processes))
    return dict(proof,status='READY',utc=utc(),racnet=facts,racnet_launch_sha256=sha(launch_path),racnet_pipeline_complete={'path':str(path),'sha256':sha(path)},gpu_index=1,gpu_uuid=gpu[1].strip(),gpu1_memory_mib=int(gpu[2]),gpu1_process_count=0,free_bytes=free)

def validate_release(path,proof):
    path=Path(path).resolve(); require(path.is_relative_to(R),'Root release outside project')
    value=read(path)
    require(value['released_by']=='root' and value['authorized_residual']==RESIDUAL and value['gpu_index']==1 and value['gpu_uuid']==proof['gpu_uuid'],'Root release queue/GPU mismatch')
    require(value['entry_sha256']==sha(__file__) and time.time()<value['start_before_epoch'],'Entry identity/startup expiry mismatch')
    require(value['racnet_pipeline_complete']==proof['racnet_pipeline_complete'],'Root release not for canonical complete RACnet')
    return {'path':str(path),'sha256':sha(path)}

def main():
    parser=argparse.ArgumentParser(__doc__); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true'); mode.add_argument('--execute',action='store_true'); parser.add_argument('--release',type=Path)
    args=parser.parse_args(); claimed=False; state={}
    try:
        with contextlib.ExitStack() as stack:
            if args.execute:
                require(args.release is not None,'Explicit root --release required')
                require(os.name!='nt' and R.resolve()==R,'Server-only')
                D.mkdir(parents=True,exist_ok=True); lock=stack.enter_context((D/'successor.lock').open('a')); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            proof=assess(stack)
            if args.check or proof['status']!='READY': print(json.dumps(proof)); return 0
            release=validate_release(args.release,proof)
            state=dict(proof,status='CLAIMED',entry_sha256=sha(__file__),root_release=release,no_automatic_retry=True)
            with (D/'one_shot.json').open('x') as f:
                json.dump(state,f,indent=2); f.flush(); os.fsync(f.fileno())
            sync_dir(D); claimed=True
            archive=D/'HOLD_ORIGINAL_GPU1.archived.json'; require(not archive.exists(),'Original hold archive already exists')
            os.rename(A/'HOLD_AFTER_CURRENT_GPU1',archive); sync_dir(A); sync_dir(D); require(sha(archive)==HOLD,'Archived original hold differs')
            state.update(status='SPAWN_COMMITTED_NO_REPEAT',hold_archive=str(archive)); atomic(D/'one_shot.json',state)
            log=stack.enter_context((D/'worker_gpu1.successor.stdout.log').open('x'))
            process=subprocess.Popen(proof['command'],cwd=SOURCE,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,creationflags=FLAGS)
            state.update(status='WORKER_SPAWNED',worker_pid=process.pid); atomic(D/'one_shot.json',state)
        code=process.wait()  # ordinary finite child lifecycle; never waits for a GPU.
        state.update(status='WORKER_EXITED_OK' if code==0 else 'FAILED_NO_RETRY',worker_returncode=code,finished_utc=utc()); atomic(D/'one_shot.json',state); print(json.dumps(state)); return code
    except Exception as error:
        failure=dict(state,status='FAILED_MANUAL_RECOVERY_ONLY' if claimed else 'BLOCKED',error=repr(error),utc=utc(),no_automatic_retry=True)
        if claimed:
            try: atomic(D/'one_shot.json',failure)
            except Exception: pass
        print(json.dumps(failure)); return 2
if __name__=='__main__': raise SystemExit(main())


