"""Finite V2 continuation: reuse original parity/stu5_3; first-generate only300/800."""
import time
START=time.monotonic()
import copy,datetime,hashlib,importlib.util,json,os,pathlib,signal,sys,traceback
R=pathlib.Path('/root/autodl-tmp/TempoRAC');W=R/'baselines/racnet_native_v1';P=R/'resume20261112/continuation-2244/racnet';D=P/'missing5-implementation';A=P/'missing5-astra';OUT=D/'cpu-continuation-v2';REC=W/'missing5_recovery_v1';CORRECTION='b414c973cf906b69bb1de435cc220c7ee2ecbed818406071506e199c16080ea1'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.tmp');b=(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 with open(tmp,'xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def timeout(*args):raise TimeoutError('remaining561s authorized CPU cap exhausted')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(.001,561-(time.monotonic()-START)))
result={'status':'RUNNING','new_TSM_generator_calls':0,'reused_sentinel':True,'reused_stu5_3':True,'model_constructor_calls':0,'model_forward_calls':0,'GPU_probes':0,'old_author_TSM_array_reads':0,'old_feature_array_reads':0,'old_start_end_array_reads':0,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
 assert not OUT.exists(),'finite continuation already attempted; preserve evidence'
 OUT.mkdir(parents=True);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ.get(k)=='1'
 os.sched_setaffinity(0,{2});assert list(os.sched_getaffinity(0))==[2]
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 assert sha(A/'CORRECTION_SYMMETRY.json')==CORRECTION;correction=load(A/'CORRECTION_SYMMETRY.json');decision=load(D/'AMENDMENT_FIRST_CPU_PRIVATE.json');assert sha(D/'AMENDMENT_FIRST_CPU_PRIVATE.json')==correction['parent_amendment_sha256']
 assert sha(D/'cpu-production/CPU_RESULT.json')==correction['preserved_failure']['cpu_result_sha256'];first=load(D/'cpu-production/CPU_RESULT.json');assert first['status']=='FAIL' and first['parity']=='PASS'
 assert sha(D/'cpu-production/PARITY.json')==correction['inherited_parity']['receipt_sha256'];parity=load(D/'cpu-production/PARITY.json');assert parity['status']=='PASS' and parity['float32_mismatch_count']==parity['nonzero_mask_mismatch_count']==0
 v1=load(W/'RUN_CONTRACT_MISSING5_V1.json');assert sha(W/'RUN_CONTRACT_MISSING5_V1.json')==correction['parent_effective_v1_sha256']
 effective=copy.deepcopy(v1);effective['missing5_recovery']['symmetry_correction']={'correction_sha256':CORRECTION,'parent_effective_v1_sha256':correction['parent_effective_v1_sha256'],'required_checks':correction['generator']['required_checks'],'diagnostic_only_checks':correction['generator']['diagnostic_checks'],'accepted_inherited_artifacts':{'parity_receipt_sha256':correction['inherited_parity']['receipt_sha256'],'stu5_3_file_sha256':correction['reuse_stu5_3']['file_sha256'],'stu5_3_sidecar_sha256':correction['reuse_stu5_3']['sidecar_sha256']}}
 assert not (W/'RUN_CONTRACT_MISSING5_V2.json').exists();dump(W/'RUN_CONTRACT_MISSING5_V2.json',effective);effective_sha=sha(W/'RUN_CONTRACT_MISSING5_V2.json');result['effective_contract_sha256']=effective_sha;result['correction_sha256']=CORRECTION;result['prior_wall_seconds']=first['wall_seconds']
 for n,h in decision['frozen_metadata'].items():assert sha(W/'inputs'/n)==h
 targets={x['key']:x for x in load(W/'inputs/targets.json')};reused=correction['reuse_stu5_3'];p=pathlib.Path(reused['path']);oldside=pathlib.Path(str(p)+'.json');assert sha(p)==reused['file_sha256'] and sha(oldside)==reused['sidecar_sha256'];s=load(oldside);assert not s['checks']['symmetric'] and all(v for k,v in s['checks'].items() if k!='symmetric')
 acceptance={'status':'accepted_by_source_fidelity_correction','key':'rc_train/stu5_3','symmetry':False,'symmetry_role':'diagnostic_only','path':str(p),'file_sha256':reused['file_sha256'],'original_sidecar_sha256':reused['sidecar_sha256'],'original_effective_contract_sha256':correction['parent_effective_v1_sha256'],'effective_contract_sha256':effective_sha,'base_contract_sha256':correction['parent_base_sha256'],'amendment_sha256':correction['parent_amendment_sha256'],'correction_sha256':CORRECTION,'parity_receipt_sha256':correction['inherited_parity']['receipt_sha256'],'preserved_failure_sha256':correction['preserved_failure']['cpu_result_sha256'],'no_matrix_or_sidecar_mutation':True,'original_bounds_and_source_identity_in_sidecar':True};assert not (REC/'STU5_3_ACCEPTANCE_V2.json').exists();dump(REC/'STU5_3_ACCEPTANCE_V2.json',acceptance)
 candidate=(A/'gen_refTSM.import_math_candidate.py').read_bytes();assert hashlib.sha256(candidate).hexdigest()==decision['tsm']['candidate_sha256'];generator=OUT/'gen_refTSM.import_math.py';generator.write_bytes(candidate)
 import numpy as np,torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 def no_model(*args,**kwargs):result['model_constructor_calls']+=1;raise AssertionError('model construction forbidden')
 torch.nn.Module.__init__=no_model
 def no_old_arrays(*args,**kwargs):raise AssertionError('continuation needs no existing array loading')
 np.load=no_old_arrays
 spec=importlib.util.spec_from_file_location('official_remaining2_TSM',generator);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 produced=[]
 for key,n in [('rc_train/val568',300),('rc_valid/stu6_0',800)]:
  t=targets[key];assert t['length']==n and t==next(x for x in decision['missing3_targets'] if x['key']==key);dest=REC/'tsm'/(key.replace('/','__')+'.npy');side=pathlib.Path(str(dest)+'.json')
  if dest.exists() or side.exists():
   assert dest.exists() and side.exists(),'uncommitted existing output; no overwrite'
   q=load(side);assert q['effective_contract_sha256']==effective_sha and q['correction_sha256']==CORRECTION and q['sha256']==sha(dest) and all(q['required_checks'].values()),'existing output conflict'
   produced.append(q);continue
  began=time.monotonic();result['new_TSM_generator_calls']+=1;matrix=module.GenerateTSM().gen_tsm(np.asarray(t['bounds'],dtype=np.int64),n,filter_size=9,smooth_mode='8neighbors_expo')
  required={'shape':matrix.shape==(n,n),'float64':matrix.dtype==np.float64,'finite':bool(np.isfinite(matrix).all()),'range_0_1':bool(np.all((matrix>=0)&(matrix<=1))),'diagonal_one':bool(np.all(np.diag(matrix)==1))};diagnostic={'symmetry':bool(np.array_equal(matrix,matrix.T)),'asymmetric_element_count':int(np.count_nonzero(matrix!=matrix.T)),'max_abs_asymmetry':float(np.max(np.abs(matrix-matrix.T)))}
  with open(dest,'xb') as f:np.save(f,matrix,allow_pickle=False);f.flush();os.fsync(f.fileno())
  q={'key':key,'path':str(dest),'shape':list(matrix.shape),'dtype':str(matrix.dtype),'required_checks':required,'diagnostic':diagnostic,'sha256':sha(dest),'bytes':dest.stat().st_size,'array_payload_sha256':hashlib.sha256(matrix.tobytes(order='C')).hexdigest(),'bounds':t['bounds'],'existing_start_end_target_sha256':t['target_sha256'],'base_contract_sha256':correction['parent_base_sha256'],'amendment_sha256':correction['parent_amendment_sha256'],'effective_contract_sha256':effective_sha,'correction_sha256':CORRECTION,'generator_sha256':decision['tsm']['candidate_sha256'],'official_generator_sha256':decision['tsm']['official_source']['sha256'],'filter_size':9,'smooth_mode':'8neighbors_expo','parity_receipt_sha256':correction['inherited_parity']['receipt_sha256'],'runtime':first['runtime'],'seconds':time.monotonic()-began,'provenance':'first_generated_missing_author_TSM'}
  dump(side,q);produced.append(q);dump(OUT/'TSM_PROGRESS.json',produced);assert all(required.values()),('required TSM check failed',key,required);print(json.dumps({'event':'first_TSM_committed','key':key,'required_checks':required,'diagnostic':diagnostic}),flush=True)
  assert sum(x.stat().st_size for root in [D/'cpu-production',OUT,REC/'tsm'] for x in root.rglob('*') if x.is_file())<=134217728
 assert result['new_TSM_generator_calls']==2 and not torch.cuda.is_initialized()
 for n,h in decision['frozen_metadata'].items():assert sha(W/'inputs'/n)==h
 result.update(status='PASS',new_TSMs=produced,inherited_stu5_3_acceptance_sha256=sha(REC/'STU5_3_ACCEPTANCE_V2.json'),remaining_wall_seconds=time.monotonic()-START,cumulative_CPU_stage_wall_seconds=first['wall_seconds']+time.monotonic()-START,CUDA_initialized=False,parity_receipt_sha256=correction['inherited_parity']['receipt_sha256'],old_failure_sha256=correction['preserved_failure']['cpu_result_sha256'],inherited_one_newly_generated_two=True);assert result['cumulative_CPU_stage_wall_seconds']<600;dump(OUT/'CPU_CONTINUATION_RESULT.json',result);dump(REC/'TSM_COMPLETE_V2.json',result);print(json.dumps({k:v for k,v in result.items() if k!='new_TSMs'},indent=2))
except BaseException as exc:
 result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),remaining_wall_seconds=time.monotonic()-START,traceback=traceback.format_exc());dump(OUT/'CPU_CONTINUATION_RESULT.json',result);print(json.dumps(result,indent=2));raise
finally:signal.setitimer(signal.ITIMER_REAL,0)
