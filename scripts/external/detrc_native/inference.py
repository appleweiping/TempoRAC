"""Inference consumes feature/length identities only; scalar labels are not opened."""
import pathlib,copy,math,json,hashlib
from common import *

def setup_paths(work):
 import sys
 sys.path[:0]=[str(work/'candidate'),str(PREREQ/'vendor/mmaction2-0.24.1')]

class LabelFreeDataset:
 def __init__(self,rows,work):
  setup_paths(work)
  from DeTRC.repcount_dataset import RepCountDataset
  self.native=object.__new__(RepCountDataset);self.native.clip_len=512;self.native.stride_rate=0.75
  self.rows=sorted(rows,key=lambda r:r['native_frames']);self.work=work
 def structure(self,row,feature):
  import numpy as np
  result=dict(video_name=row['video_id'],origin_snippet_num=row['native_frames'],snippet_num=row['native_frames'],raw_feature=feature,gt_bbox=np.empty((0,3)),video_gt_box=np.empty((0,3)))
  return self.native.prepare_testing_clip(result)
 def item(self,row):
  import torch,h5py,numpy as np
  with h5py.File(row['h5_file'],'r') as f:
   ds=f[row['video_id']];a=ds[:];value_sha=str(ds.attrs['value_sha256'])
  assert a.shape==(row['native_frames'],2048) and a.dtype==np.float32 and np.isfinite(a).all()
  assert hashlib.sha256(a.tobytes()).hexdigest()==value_sha,row['key']
  return self.structure(row,torch.from_numpy(a)),value_sha

def collate(samples):
 return dict(raw_feature=[x['raw_feature'] for x in samples],gt_bbox=[x['gt_bbox'] for x in samples],video_gt_box=[x['video_gt_box'] for x in samples],snippet_num=[x['snippet_num'] for x in samples],video_meta=[dict(video_name=x['video_name'],origin_snippet_num=x['origin_snippet_num']) for x in samples])

def postprocess(pred):
 from DeTRC.utill.temporal_box_producess import softnms_v2
 ori=copy.deepcopy(pred);post=copy.deepcopy(pred);batch=[post]
 softnms_v2(batch,sigma=0.1,top_k=30,score_threshold=0.0001)
 count=lambda classes:sum(float(seg[-1])>0.2 for seg in classes[0])
 return dict(ori_count=count(ori),count=count(batch[0]),ori_segments=[x.tolist() for x in ori],segments=[x.tolist() for x in batch[0]])

def predict(work,checkpoint,populations):
 setup_paths(work)
 import torch
 from mmaction.models import build_model
 import DeTRC.model.DetTRC
 from mmcv import Config
 c=contract(work);checkpoint=pathlib.Path(checkpoint);cs=sha(checkpoint)
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model=build_model(Config.fromfile(str(work/'candidate/DeTRC/config/project_tsn_full80.py')).model)
 model.load_state_dict(saved['state_dict'],strict=True)
 assert torch.count_nonzero(model.pos_embed.weight).item()==0
 model=model.cuda().eval()
 inputs=read(work/'inputs/inference_inputs.json')['entries'];produced=0
 for pop in populations:
  dataset=LabelFreeDataset([r for r in inputs if r['population']==pop],work)
  for start in range(0,len(dataset.rows),64):
   rows=dataset.rows[start:start+64];batch_keys=[r['key'] for r in rows]
   paths=[work/'predictions'/pop/(r['video_id']+'.json') for r in rows]
   identities=[dict(contract_sha256=binding(work),checkpoint_sha256=cs,key=r['key'],batch_keys=batch_keys) for r in rows]
   done=[p.exists() and read(p)['identity']==i for p,i in zip(paths,identities)]
   assert all(not p.exists() or ok for p,ok in zip(paths,done)),'output identity mismatch'
   if all(done):continue
   pairs=[dataset.item(r) for r in rows];samples=[x[0] for x in pairs]
   with torch.no_grad():output=model(return_loss=False,**collate(samples))
   assert len(output)==len(rows)
   for row,path,identity,already,features,result in zip(rows,paths,identities,done,pairs,output):
    # Author GT output is structural empty only; never use it as a scalar target.
    assert len(result[1]['labels'])==0
    if already:continue
    atomic_json(path,dict(identity=identity,population=pop,video_id=row['video_id'],native_frames=row['native_frames'],feature_value_sha256=features[1],**postprocess(result[0])))
    produced+=1
   atomic_json(work/'PREDICTION_PROGRESS.json',dict(contract_sha256=binding(work),checkpoint_sha256=cs,last_batch=batch_keys,new_rows=produced))
 expected=[r for r in inputs if r['population'] in populations]
 assert all((work/'predictions'/r['population']/(r['video_id']+'.json')).exists() for r in expected)
 atomic_json(work/'PREDICTIONS_COMPLETE.json',dict(contract_sha256=binding(work),checkpoint_sha256=cs,populations={p:sum(r['population']==p for r in expected) for p in populations},count=len(expected),all_label_free=True))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--work',type=pathlib.Path,default=WORK);p.add_argument('--checkpoint',required=True);a=p.parse_args()
 predict(a.work,a.checkpoint,['rc_valid','rc_test','ucf_test','test'])
