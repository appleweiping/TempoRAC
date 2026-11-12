"""Freeze full populations and the production contract; no video or model IO."""
import pathlib,shutil,collections,json
from common import *

def prepare(work=WORK):
 work=pathlib.Path(work);work.mkdir(parents=True,exist_ok=True)
 if (work/'RUN_CONTRACT.json').exists():
  assert not (work/'BUDGET.json').exists(),'production contract is immutable once work starts'
 source=PREREQ/'candidate';dest=work/'candidate'
 if not dest.exists():shutil.copytree(source,dest,ignore=shutil.ignore_patterns('build','__pycache__','.git'))
 (work/'inputs').mkdir(exist_ok=True)
 manifest=read(MANIFEST);identity=read(IDENTITIES);lookup={r['key']:r for r in identity['entries']}
 assert sha(MANIFEST)=='61cbbd8c98c7a9764284dd2f032d85d2cfb9403fedabf695fe90271f31fa2333'
 assert sha(IDENTITIES)=='9341a1b74120c5fd12172a1c8af1ecdba32403db32a6392c86a6343ba3e6882d'
 rows=[];labels=[]
 for r in manifest:
  key=r['population']+'/'+r['video_id'];i=lookup[key]
  eligible=not (r['population']=='rc_train' and r['video_id']=='stu6_44')
  roi=r['frame_range'];start,end=(0,i['full_N']-1) if roi is None else map(int,roi)
  assert 0<=start<=end<i['full_N'] and end-start+1==i['roi_N'],key
  rows.append(dict(key=key,population=r['population'],video_id=r['video_id'],file=r['file'],source_video_sha256=r['source_video_sha256'],full_frames=i['full_N'],native_frames=i['roi_N'],source_start=start,source_end=end,supervised_eligible=eligible,raw_dir=str(work/'rawframes'/key),feature_file=str(work/'pickles'/r['population']/(r['video_id']+'.pkl')),h5_file=str(work/'features'/(r['population']+'.h5'))))
  if r['population']!='rc_train':labels.append(dict(key=key,population=r['population'],video_id=r['video_id'],count=r['count']))
 assert len(rows)==1244 and sum(r['supervised_eligible'] for r in rows)==1243
 pops=dict(collections.Counter(r['population'] for r in rows));assert pops=={'rc_train':756,'rc_valid':131,'rc_test':152,'ucf_test':105,'test':100}
 # No count/bounds/native cycle targets are available to the inference process.
 atomic_json(work/'inputs/production_plan.json',dict(entries=rows,required_keys=[r['key'] for r in rows if r['supervised_eligible']],conservative_keys=[r['key'] for r in rows],manifest_sha256=sha(MANIFEST),identities_sha256=sha(IDENTITIES)))
 atomic_json(work/'inputs/inference_inputs.json',dict(entries=[r for r in rows if r['population']!='rc_train']))
 atomic_json(work/'inputs/scoring_labels.json',dict(entries=labels))
 for pop in pops:
  (work/'inputs'/f'extract_{pop}.txt').write_text(''.join(f"{r['raw_dir']} {r['native_frames']} 0\n" for r in rows if r['population']==pop))
 for name,src in [('train.csv',PREREQ/'annotations/train_corrected.csv'),('valid.csv',source/'dataset/annotation/valid_new.csv'),('test.csv',source/'dataset/annotation/test_new.csv')]:shutil.copy2(src,work/'inputs'/name)
 # Candidate-only numerical extractor is unchanged except atomic IO/resume.
 p=dest/'tools/data/activitynet/tsn_feature_extraction.py';text=(source/'tools/data/activitynet/tsn_feature_extraction.py').read_text()
 text=text.replace('from mmcv.runner import load_checkpoint','from mmcv.runner import load_checkpoint\nfrom feature_state import committed, save_feature')
 text=text.replace('        length = int(length)','        length = int(length)\n        if committed(output_file, length):\n            prog_bar.update()\n            continue')
 text=text.replace('        with open(output_file, "wb") as fout:\n            pickle.dump(feat, fout)','        save_feature(output_file, feat, length)')
 p.write_text(text)
 # New paths only; no scientific knobs altered.
 cfg=dest/'DeTRC/config/project_tsn_full80.py';base=(source/'DeTRC/config/project_tsn_full80.py').read_text()
 base=base.replace(str(PREREQ),str(work)).replace("'/annotations/train_corrected.csv'","'/inputs/train.csv'").replace("'/candidate/dataset/annotation/valid_new.csv'","'/inputs/valid.csv'").replace("'/candidate/dataset/annotation/test_new.csv'","'/inputs/test.csv'")
 cfg.write_text(base)
 contract_data=dict(schema='temporac.detrc_native_full80.v1',status='frozen_preparation_not_run_ready',authorization='complete official-code TSN method under existing user authorization; root releases production after one engineering check',identity=dict(original_commit='538abfcd24e0a81b383ca81862ff4360a2389c97',prior_candidate_manifest_sha256=sha(PREREQ/'CANDIDATE_SOURCE_MANIFEST.json'),env_lock_sha256=sha(PREREQ/'ENV_LOCK.json'),backbone_sha256=BACKBONE_SHA,backbone_bytes=97579098),method=dict(route='released TSN/R50 official-code retraining with disclosed compatibility/population adaptations',seed=0,epochs=80,batch_size=64,precision='FP32 no AMP',optimizer='AdamW',lr=0.0002,weight_decay=0,lr_steps=[15,60],lr_gamma=0.1,train_supervised=755,native_selection_rows=130,common_validation_rows=131,selection='post-SoftNMS native validation OBO strict maximum; earliest tie',final_populations={'rc_test':152,'ucf_test':105,'test':100},common_diagnostic='RC131 once on the same selected weight; does not select',clip_len=512,train_stride=128,test_stride=384,frame_interval=1,softnms=dict(sigma=0.1,top_k=30,score_threshold=0.0001),count_threshold=0.2),inputs=dict(plan_sha256=sha(work/'inputs/production_plan.json'),inference_inputs_sha256=sha(work/'inputs/inference_inputs.json'),scoring_labels_sha256=sha(work/'inputs/scoring_labels.json'),default_production_rows=1244,required_rows=1243,frames=954550,feature_fp32_bytes=7819673600,annotation_sha256={n:sha(work/'inputs'/n) for n in ['train.csv','valid.csv','test.csv']}),resources=dict(derived_cap_bytes=64*GIB,free_reserve_bytes=128*GIB,racnet_author_total_bytes=128629792634,racnet_pending_rule='total minus durable STATE coverage; additional to128GiB reserve',cpu_affinity_count=2,cpu_workers=1,nice=10,ionice='2/7',min_available_ram_bytes=32*GIB,minimum_total_mib=24000,max_idle_used_mib=500,gpu_seconds=dict(extract=16*3600,train=30*3600,evaluate=2*3600,total=48*3600),on_failure='preserve completed video/epoch checkpoint; stop incomplete; no automatic sampler/AMP/microbatch/kernel/epoch substitution'),scorer=dict(path=str(SCORER),sha256=SCORER_SHA,primary='positive-count mean(abs(np.round(pred)-gt)/(gt+0.1)); lower better'),recovery='video/epoch boundaries; replay incomplete unit, persistent allocated-time caps across resumes',code=code_identity())
 atomic_json(work/'RUN_CONTRACT.json',contract_data)
 atomic_json(work/'SAFE_MANIFEST.json',dict(contract_sha256=binding(work),code=code_identity(),populations=pops,required_rows=1243,conservative_rows=1244,frames=954550,inputs_identity=contract_data['inputs'],status='no JPEG/features/predictions produced by preparation'))
 return contract_data

if __name__=='__main__':prepare()
