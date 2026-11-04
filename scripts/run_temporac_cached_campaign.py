"""Actor-invoked cached NTR1 first pair, with guarded old-worker handover."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from contextlib import contextmanager

BASE = Path('/mnt/4tssd/lilei/TempoRAC')
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
PROFILE = dict(name='NTR1-R3-cache-reuse-5-4-7', gpu_index=1,
    process_total_cap_bytes=5*1024**3, allocator_reserved_cap_bytes=4*1024**3,
    start_free_min_bytes=7*1024**3, running_free_min_bytes=2*1024**3,
    wall_limit_seconds=86400, gpu_limit_seconds=86400)
CAMPAIGN_LOCK = BASE/'work/ntr1-campaign.lock'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def checked(ref):
    p=Path(ref['path']).resolve()
    if BASE not in p.parents or sha(p)!=ref['sha256']:
        raise ValueError('missing/out-of-root/changed artifact: '+str(p))
    return p


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True));tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def campaign_lock(cfg):
    import fcntl
    CAMPAIGN_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with CAMPAIGN_LOCK.open('a+b') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another NTR1 actor holds the project campaign lock')
        try:
            controller=identity(os.getpid())
            if controller is None:raise RuntimeError('cannot bind campaign lock to controller identity')
            receipt=Path(cfg['run_root'])/('campaign-lock-'+str(controller['pid'])+'-'+str(controller['start_ticks'])+'.json')
            if receipt.exists():raise ValueError('lock acquisition receipt already exists')
            save(receipt,dict(status='acquired',utc=stamp(),lock_path=str(CAMPAIGN_LOCK),controller=controller))
            yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        parts=(p/'stat').read_text().rsplit(')',1)[1].split()
        if parts[0]=='Z':return None
        return dict(pid=pid,start_ticks=int(parts[19]),uid=p.stat().st_uid,
            cwd=os.readlink(p/'cwd'),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().rstrip(),state=parts[0])
    except FileNotFoundError:return None


def same(expected):
    current=identity(expected['pid'])
    if current is None:return None
    for key in ('pid','start_ticks','uid','cwd','command'):
        if current[key]!=expected[key]:raise ValueError('old worker identity changed: '+str(expected['pid']))
    if current['uid']!=os.getuid():raise ValueError('other user process is never controlled')
    return current


def send(expected,sig):
    if same(expected) is not None:os.kill(expected['pid'],sig)


def run_read(argv,cwd=None):
    return subprocess.run(argv,cwd=cwd,check=True,capture_output=True,text=True,creationflags=FLAGS).stdout.strip()


def gpu():
    text=run_read(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'])
    uuid,free=[s.strip() for s in text.split(',')]
    rows=[]
    for line in run_read(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']).splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)==3:rows.append(dict(uuid=fields[0],pid=int(fields[1]),bytes=int(fields[2])*1024**2))
    return dict(uuid=uuid,free_vram=int(free)*1024**2,processes=rows)


def old_has_training(cfg):
    out=Path(cfg['old_worker']['output_root'])
    for p in out.rglob('*'):
        if p.is_file() and (p.name in ('budget.json','steps.jsonl','recovery.pt','selection.json') or p.name.startswith('candidate_')):
            return str(p)
    pids={r['pid'] for r in cfg['old_worker']['processes']}
    if any(r['pid'] in pids for r in gpu()['processes']):return 'old worker owns GPU memory'
    return None


def rollback(cfg):
    resumed=[]
    for row in reversed(cfg['old_worker']['processes']):
        current=same(row)
        if current and current['state'] in ('T','t'):
            send(row,signal.SIGCONT);resumed.append(row['pid'])
    return resumed


def watchdog(cfg_path,state_path,controller):
    cfg=load(cfg_path);until=time.monotonic()+120
    while time.monotonic()<until:
        state=load(state_path)
        if state.get('phase')=='old_terminated':return
        try:alive=same(controller) is not None
        except ValueError:alive=False
        if not alive or state.get('phase')=='handover_failed':
            resumed=rollback(cfg)
            save(Path(cfg['run_root'])/'rollback_watchdog_receipt.json',dict(utc=stamp(),reason='controller_exit_or_failure',resumed=resumed));return
        time.sleep(.5)
    resumed=rollback(cfg)
    save(Path(cfg['run_root'])/'rollback_watchdog_receipt.json',dict(utc=stamp(),reason='stop_window_timeout',resumed=resumed))


def assert_source(cfg):
    code=Path(cfg['code_root']).resolve()
    if BASE not in code.parents or code==BASE/'repo':raise ValueError('new isolated code root required')
    if run_read(['git','rev-parse','HEAD'],code)!=cfg['source_commit'] or run_read(['git','status','--porcelain'],code):
        raise ValueError('cached source HEAD/clean mismatch')


def command(cfg,label,argv,env=None):
    assert_source(cfg)
    root=Path(cfg['run_root']);log=root/'logs'/(label+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    state=root/'sequence_state.json'
    with log.open('x') as stream:
        p=subprocess.Popen(argv,cwd=cfg['code_root'],env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS)
        record=dict(status='running',label=label,pid=p.pid,command=argv,started_utc=stamp(),log=str(log))
        save(state,record);rc=p.wait()
    record.update(status='complete' if rc==0 else 'failed',exit_code=rc,ended_utc=stamp())
    save(root/'logs'/(label+'.receipt.json'),record);save(state,record)
    if rc:raise RuntimeError(label+' failed; preserve logs/budget and stop')


def environment(cfg):
    env=dict(os.environ);root=Path(cfg['run_root'])
    env.update(PYTHONPATH=str(Path(cfg['code_root'])/'src')+':'+str(BASE/'work/native-target-implementation/vendor_pure'),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled')
    for key,name in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('XDG_CACHE_HOME','cache/xdg'),('HF_HOME','cache/hf'),
                     ('TORCH_HOME','cache/torch'),('CUDA_CACHE_PATH','cache/cuda'),('MPLCONFIGDIR','cache/mpl'),
                     ('NUMBA_CACHE_DIR','cache/numba'),('TORCH_EXTENSIONS_DIR','cache/extensions'),('TRITON_CACHE_DIR','cache/triton')]:
        path=root/name;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    return env


def pin_cpu(cfg):
    receipt=load(checked(cfg['cache_affinity_receipt']))
    cpus=receipt['campaign_cpus'];cache=receipt['cache_cpus'];allowed=sorted(os.sched_getaffinity(0))
    if (receipt.get('schema')!='temporac.ntr1.cpu-affinity.v1' or not 1<=len(cpus)<=4
            or cpus!=receipt['allowed_cpus'][:4] or cache!=cpus[:3]
            or not set(cpus)<=set(allowed)):
        raise ValueError('cache/campaign CPU affinity is not one shared set of at most four')
    os.sched_setaffinity(0,cpus)
    if set(os.sched_getaffinity(0))!=set(cpus):raise RuntimeError('campaign CPU affinity not applied')
    save(Path(cfg['run_root'])/'cpu_affinity.json',dict(utc=stamp(),pid=os.getpid(),campaign_cpus=cpus,
        cache_cpus=cache,cache_affinity_receipt=cfg['cache_affinity_receipt']))


def validate_ready(cfg,env):
    ready=load(checked(cfg['readiness_receipt']))
    if (ready['status']!='ready_for_handover' or ready['actual_source_commit']!=cfg['source_commit']
            or ready['natural_train_units']!=1084 or ready['X0_views']!=224 or not ready['review_completed']
            or not ready['review_scope_complete'] or not ready['production_parity_complete']):
        raise ValueError('complete cache, production parity and review required before old-worker handover')
    for field in ('train_cache_manifest','production_parity_receipt','review_receipt','compatibility_receipt'):
        checked(ready[field])
    parity=load(checked(ready['production_parity_receipt']))
    if (parity['schema']!='temporac.ntr1.fast-cache-parity.v1' or parity['status']!='complete'
            or parity['all_bytes_equal'] is not True):
        raise ValueError('actual completed production parity receipt required')
    manifests=load(checked(ready['train_cache_manifest']))
    if (manifests['schema']!='temporac.prepared-cache.manifest.v1' or manifests['status']!='complete'
            or manifests['phase']!='train' or len(manifests['natural']['records'])!=1084 or len(manifests['x0']['records'])!=224):
        raise ValueError('all1084+224 cache records required')
    pair_output=None
    for arm in ('canonical','capacity-control'):
        path=checked(cfg['launchers'][arm]);launcher=load(path)
        if launcher['variant']!=arm:raise ValueError('launcher arm mismatch')
        parent_config=load(checked(launcher['parent_config']))
        if parent_config['seed']!=20260815:raise ValueError('this actor launches only the first seed')
        output=Path(launcher['output']).resolve()
        if Path(cfg['run_root']).resolve() not in output.parents:raise ValueError('cached output outside new run root')
        if pair_output is not None and pair_output!=output:raise ValueError('paired arms need one seed output root')
        pair_output=output
        binding=load(checked(launcher['execution_binding']))
        if binding['actual_source_commit']!=cfg['source_commit']:raise ValueError('execution source mismatch')
        if not (binding['train_cache_manifest']==ready['train_cache_manifest']
                and binding['production_parity_receipt']==ready['production_parity_receipt']):
            raise ValueError('cache/parity binding differs across ready and launchers')
        policy=load(checked(cfg['resource_policy']))
        if policy!=PROFILE:raise ValueError('exact frozen R3 resource profile required')
        lease=load(checked(binding['live_lease']))
        if any(lease.get(k)!=v for k,v in PROFILE.items()):raise ValueError('lease resource policy mismatch')
        if binding['resource_policy_sha256']!=cfg['resource_policy']['sha256']:raise ValueError('lease policy SHA mismatch')
        live=gpu()
        if lease['device_uuid']!=live['uuid'] or live['free_vram']<PROFILE['start_free_min_bytes']:
            raise ValueError('actual GPU1 start free/UUID does not satisfy R3')
        command(cfg,'validate-'+arm,[sys.executable,'-B',str(Path(cfg['code_root'])/'scripts/train_temporac_cached_response.py'),
                'validate-cached','--config',str(path)],dict(env,CUDA_VISIBLE_DEVICES=''))
        report=load(Path(launcher['output'])/arm/'cached_input_validation.json')
        if (report['status']!='complete' or report['kind']!='cached_input_compatibility_validation'
                or report['natural_train_units']!=1084 or report['X0_views']!=224
                or not report['parent_plan_not_recomputed']
                or report['execution_binding_sha256']!=launcher['execution_binding']['sha256']):
            raise ValueError('cached validation output not complete/bound')
        if report['plan_sha256']!=parity['plan_sha256']:raise ValueError('parity and cached runtime plan differ')


def handover(cfg,cfg_path):
    root=Path(cfg['run_root']);state=root/'handover_state.json'
    if old_has_training(cfg):raise ValueError('old worker has GPU/training evidence; never terminate it')
    for row in cfg['old_worker']['processes']:
        cur=same(row)
        if cur and cur['state'] in ('T','t'):raise ValueError('old worker already stopped; ownership unknown')
    controller=identity(os.getpid());save(state,dict(phase='watchdog_start',controller=controller,utc=stamp()))
    with (root/'rollback_watchdog.log').open('x') as f:
        watcher=subprocess.Popen([sys.executable,'-B',__file__,'watch-rollback','--config',str(cfg_path),'--state',str(state),
                    '--controller',json.dumps(controller)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,creationflags=FLAGS)
    if watcher.poll() is not None:raise RuntimeError('rollback watchdog did not survive')
    try:
        save(state,dict(phase='stopping_old',utc=stamp(),watchdog_pid=watcher.pid,old=cfg['old_worker']))
        for row in cfg['old_worker']['processes']:send(row,signal.SIGSTOP)
        time.sleep(.1)
        if old_has_training(cfg):raise ValueError('old worker raced into GPU/training; resume it and abort')
        if gpu()['free_vram']<PROFILE['start_free_min_bytes']:raise ValueError('live free changed before handover')
        for row in reversed(cfg['old_worker']['processes']):
            send(row,signal.SIGTERM);send(row,signal.SIGCONT)
        end=time.monotonic()+5
        while time.monotonic()<end and any(same(row) is not None for row in cfg['old_worker']['processes']):time.sleep(.1)
        if any(same(row) is not None for row in cfg['old_worker']['processes']):raise RuntimeError('old worker did not terminate; no new GPU launch')
        save(state,dict(phase='old_terminated',utc=stamp(),old=cfg['old_worker'],watchdog_pid=watcher.pid,
                        reason='complete cached parity/review/lease; old worker had no GPU/budget/step/checkpoint'))
    except BaseException:
        rollback(cfg);save(state,dict(phase='handover_failed',utc=stamp(),old=cfg['old_worker']));raise


def score_arm(cfg,arm,env):
    launcher=load(checked(cfg['launchers'][arm]));out=Path(launcher['output'])/arm
    selected=load(out/'selection.json')['selected']
    refs=dict(cfg['score_inputs'])
    for key,path in [('predictions',out/'predictions.json'),('selection',out/'selection.json'),
                     ('plan',out.parent/'frozen_plan.json'),('checkpoint',out/selected['file'])]:refs[key]=dict(path=str(path),sha256=sha(path))
    score_cfg=dict(protocol='temporac.native-prediction-score.v1',source_commit=cfg['source_commit'],seed=20260815,
        variant=arm,prediction_root=str(out),output=str(Path(cfg['run_root'])/'scores'/arm),inputs=refs)
    path=Path(cfg['run_root'])/'config'/('score-'+arm+'.json');save(path,score_cfg)
    command(cfg,'score-'+arm,[sys.executable,'-B',str(Path(cfg['code_root'])/'scripts/score_temporac_native_predictions.py'),
        '--config',str(path)],dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'))


def dev_ready(cfg,arm):
    """Resolve late CPU-produced dev bytes only at the prediction boundary."""
    path=Path(cfg['dev_ready_path']).resolve()
    if BASE not in path.parents:raise ValueError('dev-ready path outside server root')
    # Builder publishes dev_ready.json atomically as the immutable manifest itself.
    ref=dict(path=str(path),sha256=sha(path));manifest_path=checked(ref);manifest=load(manifest_path)
    if (manifest.get('schema')!='temporac.prepared-cache.manifest.v1'
            or manifest.get('status')!='complete' or manifest.get('phase')!='dev'
            or len(manifest['natural']['records'])!=486):
        raise ValueError('complete 486-identity dev manifest required before prediction')
    record=dict(utc=stamp(),arm=arm,dev_cache_manifest=ref)
    evidence=Path(cfg['run_root'])/'logs'/('dev-ready-'+arm+'.json')
    if evidence.exists():raise ValueError('refuse to replace dev-ready boundary evidence')
    save(evidence,record)
    return manifest_path,ref['sha256']


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('action',choices=['handover-run','watch-rollback'])
    ap.add_argument('--config',required=True,type=Path);ap.add_argument('--state',type=Path);ap.add_argument('--controller')
    args=ap.parse_args();cfg=load(args.config)
    if args.action=='watch-rollback':watchdog(args.config,args.state,json.loads(args.controller));return
    if cfg.get('schema')!='temporac.ntr1.cached-sequence.v1':raise ValueError('cached sequence schema required')
    root=Path(cfg['run_root']).resolve()
    if BASE not in root.parents or root==Path(cfg['old_worker']['output_root']).resolve():raise ValueError('new separate run root required')
    code=Path(cfg['code_root']).resolve()
    if root==code or code in root.parents or root in code.parents:raise ValueError('run and source roots must be separate')
    if len(cfg['old_worker']['processes'])!=2 or len({x['pid'] for x in cfg['old_worker']['processes']})!=2:
        raise ValueError('exact old driver and child identities required')
    root.mkdir(parents=True,exist_ok=True)
    with campaign_lock(cfg):
        pin_cpu(cfg)
        env=environment(cfg);validate_ready(cfg,env)
        handover(cfg,args.config)
        for arm in ('canonical','capacity-control'):
            launcher=checked(cfg['launchers'][arm]);cli=Path(cfg['code_root'])/'scripts/train_temporac_cached_response.py'
            command(cfg,'train-'+arm,[sys.executable,'-B',str(cli),'train-cached','--config',str(launcher)],env)
            dev,dev_sha=dev_ready(cfg,arm)
            command(cfg,'predict-'+arm,[sys.executable,'-B',str(cli),'predict-cached','--config',str(launcher),
                 '--dev-manifest',str(dev),'--dev-manifest-sha256',dev_sha],env)
            score_arm(cfg,arm,env)
        save(root/'sequence_state.json',dict(status='first_pair_scored_awaiting_parent_diagnosis_and_resource_release',
             source_commit=cfg['source_commit'],seed=20260815,utc=stamp(),scientific_acceptance='not_assigned'))


if __name__=='__main__':main()
