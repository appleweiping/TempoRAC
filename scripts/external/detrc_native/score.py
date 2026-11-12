"""External scoring only, after all label-free predictions are durable."""
import pathlib,importlib.util,math,collections,copy
from common import *

def clean(x):
 import numpy as np
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
 return x

def score(work=WORK):
 assert sha(SCORER)==SCORER_SHA
 spec=importlib.util.spec_from_file_location('frozen_tempo_metrics',SCORER);metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
 labels=read(work/'inputs/scoring_labels.json')['entries'];inputs=read(work/'inputs/inference_inputs.json')['entries']
 complete=read(work/'PREDICTIONS_COMPLETE.json');assert complete['contract_sha256']==binding(work) and complete['count']==488
 final={};raw={};table=[]
 for r in labels:
  result=read(work/'predictions'/r['population']/(r['video_id']+'.json'))
  assert result['identity']['checkpoint_sha256']==complete['checkpoint_sha256']
  table.append(dict(key=r['key'],population=r['population'],gt=r['count'],pred=result['count'],ori_pred=result['ori_count']))
 for pop,n in {'rc_valid':131,'rc_test':152,'ucf_test':105,'test':100}.items():
  rows=[r for r in table if r['population']==pop];assert len(rows)==n
  final[pop]=metrics.count_metrics([r['pred'] for r in rows],[r['gt'] for r in rows]);raw[pop]=metrics.count_metrics([r['ori_pred'] for r in rows],[r['gt'] for r in rows])
 # Author RC diagnostics use native valid cycles; common scorer keeps CSV scalar.
 from inference import setup_paths
 setup_paths(work)
 import numpy as np
 from DeTRC.repcount_dataset import RepCountDataset,load_proposal_file,VideoRecord
 native={};counterexamples=[]
 for pop,csvname,n in [('rc_valid','valid.csv',130),('rc_test','test.csv',151)]:
  videos=[VideoRecord(x) for x in load_proposal_file(str(work/'inputs'/csvname))];videos=[v for v in videos if v.gt]
  assert len(videos)==n
  from DeTRC.utill.temporal_box_producess import preprocess_groundtruth
  evaluator=object.__new__(RepCountDataset);evaluator.class_num=2;evaluator.soft_nms_sigma=0.1;evaluator.soft_nms_threshold=0.0001;evaluator.best=0;evaluator.video_list=videos
  allgt=evaluator.get_all_gt();native_rows=[]
  for v,gts in zip(videos,allgt):
   prediction=read(work/'predictions'/pop/(v.id+'.json'))
   gt=preprocess_groundtruth([np.asarray(gts).reshape(-1,3)],[v.num_frames])[0]
   native_rows.append([[np.asarray(x,dtype=float).reshape(-1,3) for x in prediction['ori_segments']],gt])
   scalar=next(r['gt'] for r in table if r['key']==pop+'/'+v.id)
   if scalar!=len(v.gt):counterexamples.append(dict(key=pop+'/'+v.id,common_scalar=scalar,native_cycles=len(v.gt)))
  native[pop]=evaluator.evaluate(native_rows)
 assert any(r['key']=='rc_test/stu4_5' and r['common_scalar']==51 and r['native_cycles']==41 for r in counterexamples)
 atomic_json(work/'FINAL_METRICS.json',clean(dict(contract_sha256=binding(work),checkpoint_sha256=complete['checkpoint_sha256'],common=final,common_ori=raw,native_RC=native,native_vs_scalar_counterexamples=counterexamples,complete357=sum(final[p]['n'] for p in ['rc_test','ucf_test','test'])==357,rows=table,scientific_claim='single-seed measured TSN route only; no automatic SOTA/significance conclusion')))

if __name__=='__main__':score()
