"""Released stages only; missing full native features prevent any formal train."""
import argparse,pathlib,os,sys,time
from common import *

def features_ready(work):
 import h5py,numpy as np
 complete=read(work/'FEATURES_COMPLETE.json');assert complete['contract_sha256']==binding(work)
 plan=read(work/'inputs/production_plan.json')['entries']
 for pop in ['rc_train','rc_valid','rc_test','ucf_test','test']:
  expected=[r for r in plan if r['population']==pop]
  with h5py.File(work/'features'/(pop+'.h5'),'r') as f:
   assert set(f)=={r['video_id'] for r in expected},pop
   for r in expected:
    ds=f[r['video_id']];assert ds.shape==(r['native_frames'],2048) and ds.dtype==np.float32 and ds.attrs['backbone_sha256']==BACKBONE_SHA and ds.attrs['source_video_sha256']==r['source_video_sha256'],r['key']
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preflight','rawframes','extract','train','evaluate']);p.add_argument('--work',type=pathlib.Path,default=WORK);p.add_argument('--release',type=pathlib.Path);p.add_argument('--gpu',type=int);a=p.parse_args();work=a.work;assert work.resolve()==WORK.resolve(),"one fixed production root only"
 if a.stage=='prepare':
  from prepare import prepare
  prepare(work);return
 if a.stage=='preflight':
  result={'contract_sha256':binding(work),'rawframes_complete':(work/'RAWFRAMES_COMPLETE.json').exists(),'features_complete':(work/'FEATURES_COMPLETE.json').exists(),'training_released':False,'no_gpu_probe':True};Capacity(work).check();print(json.dumps(result));return
 assert a.release is not None,'root production release file required'
 require_release(work,a.stage,a.release)
 with stage_lock(work):
  if a.stage=='rawframes':
   from rawframes import produce
   produce(work);return
  assert a.gpu is not None,'physical GPU index required'
  recover_budget(work)
  if a.stage in ['train','evaluate']:features_ready(work)
  gpu=gpu_gate(work,a.gpu);atomic_json(work/(a.stage+'_RESOURCE_RECEIPT.json'),dict(contract_sha256=binding(work),**gpu))
  env=runtime_env(work,a.gpu);py=str(PREREQ/'env/bin/python');here=pathlib.Path(__file__).parent
  if a.stage=='extract':
   assert (work/'RAWFRAMES_COMPLETE.json').exists(),'full CPU JPEG production incomplete'
   rc=Budget(work,'extract').run([py,str(here/'produce_features.py')],work/'extract.log',env)
  elif a.stage=='train':
   cfg=work/'candidate/DeTRC/config/project_tsn_full80.py';run=work/'runs/full80_seed0';run.mkdir(parents=True,exist_ok=True)
   args=[py,str(work/'candidate/tools/train.py'),str(cfg),'--validate','--seed','0','--gpu-ids','0']
   latest=run/'durable_latest.pth'
   if latest.exists():args+=['--resume-from',str(latest)]
   rc=Budget(work,'train').run(args,work/'train.log',env)
   if rc==0:
    import torch
    checkpoint=torch.load(latest,map_location='cpu',weights_only=False)
    assert checkpoint['meta']['epoch']==80,'not a complete80epoch run'
    best=pathlib.Path(checkpoint['meta']['hook_msgs']['best_ckpt']);assert best.is_file()
    atomic_json(work/'SELECTED_CHECKPOINT.json',dict(contract_sha256=binding(work),path=str(best),sha256=sha(best),native_validation_OBO=checkpoint['meta']['hook_msgs']['best_score'],epochs_complete=80,selection='native130 OBO strict greater; earliest tie'))
  else:
   selected=read(work/'SELECTED_CHECKPOINT.json');assert selected['contract_sha256']==binding(work) and selected['epochs_complete']==80 and sha(selected['path'])==selected['sha256']
   rc=Budget(work,'evaluate').run([py,str(here/'inference.py'),'--work',str(work),'--checkpoint',selected['path']],work/'inference.log',env)
   if rc==0:
    # The GPU child has exited; scalar labels enter only the separate CPU scorer.
    score_env=runtime_env(work,None)
    with (work/'scoring.log').open('a') as f:rc=command([py,str(here/'score.py')],env=score_env,stdout=f,stderr=subprocess.STDOUT).returncode
  atomic_json(work/(a.stage+'_OUTCOME.json'),dict(contract_sha256=binding(work),exit_code=rc,complete=rc==0,scope='full frozen stage, no reduced substitute',checkpoint_recovery='last completed video or epoch retained; failure and incomplete epoch preserved in log'))
  if rc:raise SystemExit(rc)

if __name__=='__main__':main()
