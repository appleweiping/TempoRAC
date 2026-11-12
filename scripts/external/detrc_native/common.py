"""Project-private IO, resource and phase budget interfaces (no model code)."""
from __future__ import annotations
import hashlib,json,os,pathlib,subprocess,time,shutil,contextlib,fcntl
R=pathlib.Path('/root/autodl-tmp/TempoRAC')
PREREQ=R/'external-baselines-20261112/detrc-prerequisites'
WORK=R/'baselines/detrc_native_v1'
BACKBONE=PREREQ/'assets/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth'
BACKBONE_SHA='cc665e2a86857b4ff4873747ae4e98f9cd6de3f72111cf3379954f551cc63c49'
MANIFEST=R/'baselines/transrac-native-assets20261112/inputs/manifest.json'
IDENTITIES=R/'baselines/transrac_native_v1/inputs/IDENTITIES.json'
SCORER=R/'source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr/metrics.py'
SCORER_SHA='fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0'
GIB=1<<30

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))

def atomic_json(path,data):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.pending')
 with temp.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
 fd=os.open(path.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def command(argv,**kw):
 return subprocess.run(argv,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),**kw)

def tree_bytes(path):return sum(p.stat().st_size for p in pathlib.Path(path).rglob('*') if p.is_file() and not p.is_symlink())

def code_identity():
 here=pathlib.Path(__file__).parent
 return {p.name:sha(p) for p in sorted(here.iterdir()) if p.suffix in {'.py','.sh'}}

def contract(work=WORK):return read(work/'RUN_CONTRACT.json')

def binding(work=WORK):return sha(work/'RUN_CONTRACT.json')

def require_release(work,stage,release):
 c=contract(work);r=read(release)
 assert r['contract_sha256']==binding(work),'release/contract mismatch'
 assert stage in r['stages'],'stage not released'
 assert r['code']==code_identity()==c['code'],'released code differs'
 assert c['method']['epochs']==80 and c['method']['batch_size']==64
 return r

@contextlib.contextmanager
def stage_lock(work):
 work.mkdir(parents=True,exist_ok=True)
 with (work/'production.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  yield

def racnet_pending():
 root=R/'baselines/racnet-author-assets20261112'
 control=read(root/'CONTROL.json');state=read(root/'STATE.json')
 total=sum(x['bytes'] for x in control['objects']);assert total==128629792634,total
 covered=0
 for obj in control['objects']:
  intervals=sorted(state['objects'].get(obj['id'],{}).get('coverage',[]));end=0
  for a,b in intervals:
   assert 0<=a<=b<=obj['bytes']
   covered+=max(0,b-max(a,end));end=max(end,b)
 return dict(total_bytes=total,durable_covered_bytes=covered,pending_bytes=total-covered,control_sha256=sha(root/'CONTROL.json'),state_sha256=sha(root/'STATE.json'))

class Capacity:
 def __init__(self,work):
  self.work=pathlib.Path(work);self.used=tree_bytes(work);self.reserve=None;self.refresh=0
 def check(self,additional=0):
  if self.reserve is None or time.monotonic()-self.refresh>=60:
   self.reserve=racnet_pending();self.refresh=time.monotonic()
  limits=contract(self.work)['resources'];free=shutil.disk_usage(self.work).free
  pending=self.reserve['pending_bytes']
  assert self.used+additional<=limits['derived_cap_bytes'],('derived cap',self.used,additional)
  assert free-additional>=limits['free_reserve_bytes']+pending,('free reserve plus pending RACnet author bytes',free,additional,pending)
 def account(self,new,old=0):self.used+=new-old;self.check()

def gpu_gate(work,gpu):
 c=contract(work);limits=c['resources']
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 rows=[x.strip().split(', ') for x in out.strip().splitlines()]
 row=next(x for x in rows if int(x[0])==int(gpu))
 assert int(row[2])<limits['max_idle_used_mib'],('GPU occupied',row)
 assert int(row[3])>=limits['minimum_total_mib'],row
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 assert not any(line.split(',')[0].strip()==row[1] for line in apps.splitlines()),('GPU has compute processes',row)
 Capacity(work).check()
 available=int(next(line.split()[1] for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
 assert available>=limits['min_available_ram_bytes'],available
 return dict(gpu=int(gpu),uuid=row[1],used_mib=int(row[2]),total_mib=int(row[3]),mem_available_bytes=available)

class Budget:
 """Allocation wall time, persisted across attempts; CPU JPEG time separate."""
 def __init__(self,work,stage):self.work=work;self.stage=stage;self.path=work/'BUDGET.json';self.running=None
 def data(self):return read(self.path) if self.path.exists() else {'contract_sha256':binding(self.work),'used_seconds':{}}
 def remaining(self):
  d=self.data();assert d['contract_sha256']==binding(self.work)
  c=contract(self.work)['resources']['gpu_seconds'];used=d['used_seconds']
  return max(0,min(c[self.stage]-used.get(self.stage,0),c['total']-sum(used.values())))
 def run(self,argv,log,env):
  remaining=self.remaining();assert remaining>0,'full-run budget exhausted'
  start=time.monotonic();d=self.data();d['active']={'stage':self.stage,'started_epoch':time.time(),'remaining':remaining};atomic_json(self.path,d)
  try:
   with open(log,'a') as f:
    import signal
    proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    d=self.data();d['active']['pid']=proc.pid;atomic_json(self.path,d)
    try:
     cap=Capacity(self.work)
     dynamic=[self.work/'pickles',self.work/'features'] if self.stage=='extract' else [self.work/'runs'] if self.stage=='train' else [self.work/'predictions']
     base=cap.used-sum(tree_bytes(p) for p in dynamic if p.exists())
     while True:
      left=remaining-(time.monotonic()-start)
      if left<=0:raise subprocess.TimeoutExpired(argv,remaining)
      try:return proc.wait(timeout=min(5,left))
      except subprocess.TimeoutExpired:
       if time.monotonic()-start>=remaining:raise
       cap.used=base+sum(tree_bytes(p) for p in dynamic if p.exists());cap.check()
    except subprocess.TimeoutExpired:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     return 124
    except BaseException:
     if proc.poll() is None:
      os.killpg(proc.pid,signal.SIGTERM)
      try:proc.wait(timeout=5)
      except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     raise
  finally:
   d=self.data();d.pop('active',None);d['used_seconds'][self.stage]=d['used_seconds'].get(self.stage,0)+time.monotonic()-start;atomic_json(self.path,d)

def recover_budget(work):
 p=work/'BUDGET.json'
 if p.exists():
  d=read(p)
  if 'active' in d:
   a=d['active']
   if 'pid' in a:
    try:os.kill(a['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('previous own stage still running; inspect recorded PID/log before resume')
   d.pop('active');charge=min(time.time()-a['started_epoch'],a['remaining'])
   d['used_seconds'][a['stage']]=d['used_seconds'].get(a['stage'],0)+max(0,charge)
   atomic_json(p,d)

def runtime_env(work,gpu=None):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),CUDA_HOME=str(PREREQ/'cuda-12.4'),TMPDIR=str(work/'tmp'),TORCH_EXTENSIONS_DIR=str(PREREQ/'torch-extensions'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(work/'candidate')+':'+str(PREREQ/'vendor/mmaction2-0.24.1')+':'+str(pathlib.Path(__file__).parent))
 (work/'tmp').mkdir(exist_ok=True)
 return e
