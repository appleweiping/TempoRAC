"""One bounded CPU engineering check of new integration interfaces only."""
import os,pathlib,types,threading,importlib.util,copy,collections,hashlib,ast
from common import *

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 from prepare import prepare
 prepare(WORK)
 import torch,numpy as np,mmcv
 from inference import LabelFreeDataset,postprocess,collate,setup_paths
 from rawframes import build_video,complete
 setup_paths(WORK)
 from DeTRC.utill.temporal_box_producess import softnms_v2
 report={'scope':'new integration metadata/JPEG/label-free window/postprocess primitives only; no true video decode, feature/model forward, training or evaluation'}
 plan=read(WORK/'inputs/production_plan.json')['entries'];inputs=read(WORK/'inputs/inference_inputs.json')['entries']
 assert len(plan)==1244 and sum(r['supervised_eligible'] for r in plan)==1243
 assert sum(r['population']=='rc_train' and r['supervised_eligible'] for r in plan)==755
 assert len(inputs)==488 and sum(r['population']!='rc_valid' for r in inputs)==357
 assert not any({'count','bounds','gt','labels'}&set(r) for r in inputs)
 assert all(r['source_end']-r['source_start']+1==r['native_frames'] for r in plan)
 dataset=LabelFreeDataset(inputs,WORK);zero=torch.zeros((1,2048));windows=0
 for row in plan:
  sample=dataset.structure(row,zero.expand(row['native_frames'],2048))
  n=row['native_frames'];clips=max(1,__import__('math').ceil((n-512)/384)+1)
  assert len(sample['raw_feature'])==clips
  assert sum(len(x) for x in sample['gt_bbox'])==0 and sample['video_gt_box'].shape==(0,3)
  assert all(x.shape[0]==min(512,n-384*j) for j,x in enumerate(sample['raw_feature']))
  assert all(x.shape[1]==2048 for x in sample['raw_feature'])
  windows+=clips
 report['metadata']=dict(rows=1244,required=1243,supervised_train=755,label_free=488,final357=True,frames=sum(r['native_frames'] for r in plan),window_structures=windows,no_label_fields=True,all_roi_arithmetic=True)
 # JPEG byte identity against the exact released builder, with an in-memory
 # three-frame fixture. The real dataset videos are never decoded here.
 checks=WORK/'checks';checks.mkdir(exist_ok=True)
 fixture=checks/'jpeg_fixture';fixture.mkdir(exist_ok=True)
 frames=[(np.arange(h*w*3,dtype=np.uint32).reshape(h,w,3)%256).astype(np.uint8) for h,w in [(40,80),(91,33),(48,48)]]
 spec=importlib.util.spec_from_file_location('official_raw_builder',PREREQ/'candidate/tools/data/build_rawframes.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 class Reader:
  def __init__(self,*a,**k):pass
  def __iter__(self):return iter(frames)
 old=mmcv.VideoReader;mmcv.VideoReader=Reader
 native.args=types.SimpleNamespace(out_dir=str(fixture/'official'),use_opencv=True,new_short=256,new_width=0,new_height=0)
 native.lock=threading.Lock();source=fixture/'source_fixture.bin';source.write_bytes(b'engineering fixture; no dataset video')
 try:
  native.extract_frame((str(source),'fixture.mp4',0,None,'rgb',str(fixture/'report.txt')))
  row=dict(key='fixture/roi1..2',file=str(source),source_video_sha256=sha(source),full_frames=3,native_frames=2,source_start=1,source_end=2,raw_dir=str(fixture/'integrated'))
  cap=Capacity(WORK);assert build_video(row,cap)=='built';assert complete(row);assert build_video(row,cap)=='reused'
  assert sha(fixture/'integrated/img_00001.jpg')==sha(fixture/'official/fixture/img_00002.jpg')
  assert sha(fixture/'integrated/img_00002.jpg')==sha(fixture/'official/fixture/img_00003.jpg')
 finally:mmcv.VideoReader=old
 report['jpeg_primitive']=dict(exact_official_byte_equality=True,roi_0based_inclusive_renumbered=True,atomic_video_reuse=True,fixture_frames=3,real_video_decodes=0)
 pred=[np.array([[j*.02,j*.02+.005,.9-j*.002] for j in range(20)],float),np.array([[.40+j*.02,.405+j*.02,.99-j*.001] for j in range(22)],float)]
 reference=[copy.deepcopy(pred)];softnms_v2(reference,sigma=.1,top_k=30,score_threshold=.0001);adapted=postprocess(pred)
 assert all(np.array_equal(np.asarray(adapted['segments'][i]).reshape(-1,3),reference[0][i]) for i in range(2))
 assert sum(len(x) for x in reference[0])==30 and adapted['count']==8
 report['postprocessor']=dict(exact_original=True,all_class_top_k=30,action_count=8,raw_action_count=20)
 # Actual missing features must prevent readiness; no substitute H5 is created.
 from run import features_ready
 try:features_ready(WORK)
 except FileNotFoundError:missing_features_refused=True
 else:raise AssertionError('unexpected feature readiness before production')
 report['readiness']=dict(missing_features_refused=missing_features_refused,formal_training_ready=False,review_pending=True,production_started=False)
 base=(PREREQ/'candidate/tools/data/activitynet/tsn_feature_extraction.py').read_text()
 expected=base.replace('from mmcv.runner import load_checkpoint','from mmcv.runner import load_checkpoint\nfrom feature_state import committed, save_feature').replace('        length = int(length)','        length = int(length)\n        if committed(output_file, length):\n            prog_bar.update()\n            continue').replace('        with open(output_file, "wb") as fout:\n            pickle.dump(feat, fout)','        save_feature(output_file, feat, length)')
 assert expected==(WORK/'candidate/tools/data/activitynet/tsn_feature_extraction.py').read_text()
 for p in pathlib.Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 report.update(extractor_math_unchanged=True,code=code_identity(),contract_sha256=binding(WORK),gpu_initialized=torch.cuda.is_initialized(),model_forwards=0,gpu_dispatches=0)
 assert report['gpu_initialized'] is False
 atomic_json(WORK/'CPU_INTEGRATION_CHECK.json',report)
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
