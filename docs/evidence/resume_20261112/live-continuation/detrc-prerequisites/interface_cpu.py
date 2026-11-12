import pathlib,sys,os,json,hashlib,csv,collections,datetime,logging
B=pathlib.Path('/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites')
C=B/'candidate';sys.path[:0]=[str(C),str(B/'vendor/mmaction2-0.24.1')]
import numpy as np
import torch
import mmcv
from mmcv.runner import EpochBasedRunner
from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook
from mmaction.datasets.pipelines import UntrimmedSampleFrames
from mmaction.models import build_model
import DeTRC.model.DetTRC
from DeTRC.repcount_dataset import load_proposal_file,VideoRecord
from DeTRC.durable_state import runtime_state,restore_runtime,DeTRCDurableStateHook
from DeTRC.model.grid_sample1d import grid_sample1d_cuda
from DeTRC.model.roi_align import Align1D
assert not torch.cuda.is_initialized()
out={}; config=mmcv.Config.fromfile(str(C/'DeTRC/config/repcount_tsn_feature_enc_contrastive.py'))
m=build_model(config.model); assert torch.count_nonzero(m.pos_embed.weight).item()==0
out['model']=dict(type=type(m).__name__,parameter_count=sum(p.numel() for p in m.parameters()),zero_position_table=True,train_position_argument='None',test_pretransform_shape=[2,512,512],author_transformer_permute=[2,0,1],generality='only the fixed zero-table 512x512 default is supported; nonzero table rejected')
# Full existing source CSV populations; no feature H5 or model forwards.
annotations={}
(B/'annotations').mkdir(exist_ok=True)
for split in ['train','valid','test']:
 p=C/'dataset/annotation'/f'{split}_new.csv'
 if split=='train':
  with p.open(newline='') as f:
   reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
  kept=[r for r in rows if r['name'].split('.')[0] not in {'stu3_27','stu4_50'}]
  p=B/'annotations/train_corrected.csv'
  with p.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(kept)
 props=load_proposal_file(str(p));vids=[VideoRecord(x) for x in props]
 annotations[split]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),parsed=len(vids),nonempty=sum(bool(v.gt) for v in vids),empty=[v.id for v in vids if not v.gt])
assert annotations['train']['nonempty']==755,annotations
assert annotations['valid']['nonempty']==130,annotations
assert annotations['test']['nonempty']==151,annotations
out['annotations']=annotations
# Sampler check for every real input length, without decoding a single frame.
manifestp=pathlib.Path('/root/autodl-tmp/TempoRAC/baselines/transrac-native-assets20261112/inputs/manifest.json')
identityp=pathlib.Path('/root/autodl-tmp/TempoRAC/baselines/transrac_native_v1/inputs/IDENTITIES.json')
manifest=json.loads(manifestp.read_text());ids=json.loads(identityp.read_text());ids={r['key']:r for r in ids['entries']}
sampler=UntrimmedSampleFrames(clip_len=1,frame_interval=1,start_index=0)
planned=[]; pops=collections.Counter();frames=collections.Counter()
for row in manifest:
 key=row['population']+'/'+row['video_id'];n=ids[key]['roi_N']
 sampled=sampler(dict(total_frames=n,start_index=1))['frame_inds']
 assert np.array_equal(sampled,np.arange(1,n+1)),key
 pops[row['population']]+=1;frames[row['population']]+=n
 planned.append(dict(key=key,population=row['population'],video_id=row['video_id'],file=row['file'],source_video_sha256=row['source_video_sha256'],full_frames=ids[key]['full_N'],native_frames=n,frame_range=row['frame_range'],raw_frame_dir=str(B/'rawframes'/key),first_jpeg='img_00001.jpg',last_jpeg=f'img_{n:05d}.jpg',planned_h5_key=row['video_id'],feature_shape=[n,2048],dtype='float32'))
(B/'EXTRACTION_INPUT_PLAN.json').write_text(json.dumps(dict(status='plan_only_rawframes_not_built',manifest_sha256=hashlib.sha256(manifestp.read_bytes()).hexdigest(),identities_sha256=hashlib.sha256(identityp.read_bytes()).hexdigest(),entries=planned),indent=2))
for pop in pops:
 (B/'annotations'/f'extract_{pop}.txt').write_text(''.join(f"{r['raw_frame_dir']} {r['native_frames']} 0\n" for r in planned if r['population']==pop))
out['sampler']=dict(rows=len(planned),full_native_frames=sum(frames.values()),populations=dict(pops),frames=dict(frames),interval=1,sampler_start_argument=0,effective_sample_start_index=1,all_exact_1_through_T=True,decoded_frames=0)
# Durable checkpoint serialization/recovery on CPU, no model forward or optimizer step.
checks=B/'checks';checks.mkdir(exist_ok=True)
m.cls_warmup_step=31;m.coef_ce_now=m.coef_ce;m.coef_count_now=m.coef_count
opt=torch.optim.AdamW(m.parameters(),lr=0.0002,weight_decay=0)
runner=EpochBasedRunner(m,optimizer=opt,work_dir=str(checks),logger=logging.getLogger('detrc-check'),max_epochs=80,meta={'hook_msgs':{'best_score':0.4,'best_ckpt':'native_best_marker.pth'}})
runner._epoch=16;runner._iter=100
lr=StepLrUpdaterHook(step=[15,60],gamma=0.1,by_epoch=True);lr.before_run(runner);lr.before_train_epoch(runner)
# Native strict selection: save an improved epoch, retain it on a tie and after
# a later improvement. These are checkpoint-only CPU calls, no validation run.
from mmaction.core import EvalHook
loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.empty(0)))
ev=EvalHook(loader,save_best='OBO',rule='greater');ev.before_run(runner)
ev._save_ckpt(runner,0.5);first=ev.best_ckpt_path
first_sha=hashlib.sha256(pathlib.Path(first).read_bytes()).hexdigest()
runner._epoch=17;ev._save_ckpt(runner,0.5)
assert ev.best_ckpt_path==first
runner._epoch=18;ev._save_ckpt(runner,0.6)
assert pathlib.Path(first).is_file() and ev.best_ckpt_path!=first
out['best_selection']=dict(strict_tie_keeps_first=True,older_best_retained=True,first_best_sha256=first_sha,selected_score=0.6)
hook=DeTRCDurableStateHook();hook.after_train_epoch(runner)
saved=torch.load(checks/'durable_latest.pth',map_location='cpu',weights_only=False)
assert saved['meta']['epoch']==19 and saved['meta']['detrc_runtime']['cls_warmup_step']==31
assert 'optimizer' in saved
m.cls_warmup_step=0;m.coef_ce_now=0;m.coef_count_now=0
runner2=EpochBasedRunner(m,optimizer=opt,work_dir=str(checks),logger=logging.getLogger('detrc-check2'),max_epochs=80,meta={})
runner2.resume(str(checks/'durable_latest.pth'),map_location='cpu');hook.before_run(runner2)
assert m.cls_warmup_step==31 and m.coef_ce_now==config.model.coef_ce
assert runner2.meta['hook_msgs']['best_score']==0.6
lr2=StepLrUpdaterHook(step=[15,60],gamma=0.1,by_epoch=True);lr2.before_run(runner2);lr2.before_train_epoch(runner2)
assert opt.param_groups[0]['lr']==0.0002*0.1
state=runtime_state(m)
expected=(__import__('random').random(),np.random.random(),torch.rand(1))
restore_runtime(m,state)
actual=(__import__('random').random(),np.random.random(),torch.rand(1))
assert expected[0]==actual[0] and expected[1]==actual[1] and torch.equal(expected[2],actual[2])
out['durable']=dict(epoch_roundtrip=runner2.epoch,warmup_step=m.cls_warmup_step,optimizer_restored=True,step_lr_resume=opt.param_groups[0]["lr"],step_lr_initial=opt.param_groups[0]["initial_lr"],best_hook_state_restored=True,python_numpy_torch_rng_roundtrip=True,cuda_rng_tested=False,boundary='epoch boundary; incomplete epoch replays',checkpoint_sha256=hashlib.sha256((checks/'durable_latest.pth').read_bytes()).hexdigest())
out['native_extensions']=dict(roi_align=Align1D.__file__,grid_sample1d=grid_sample1d_cuda.__file__,imports=True,dispatches=0)
out.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_forward_calls=0,optimizer_steps=0,gpu_initialized=torch.cuda.is_initialized())
assert out['gpu_initialized'] is False
(B/'CPU_INTERFACE_CHECKS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
