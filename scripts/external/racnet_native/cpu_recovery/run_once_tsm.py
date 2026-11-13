"""One authorized CPU sentinel followed only on PASS by three first reference TSMs."""
import time
START=time.monotonic()
import copy,datetime,hashlib,importlib.util,json,os,pathlib,signal,sys,traceback
R=pathlib.Path('/root/autodl-tmp/TempoRAC');W=R/'baselines/racnet_native_v1';P=R/'resume20261112/continuation-2244/racnet';D=P/'missing5-implementation';A=P/'missing5-astra';OUT=D/'cpu-production';REC=W/'missing5_recovery_v1'
AMENDMENT='c783a2596252f4e7b6a15459b647ac2ef338b4bec29aa9e7145d8e5e2d101733';BASE='77c3d84443fb57415db3b2ed7416b3b16af5f8e8af614c41a05812802e7f8de8'
result={'status':'RUNNING','requested_executor':'gpt-6.1-sol','actual_identity':'unattested','generated_TSM_calls':0,'existing_author_TSM_arrays_read':0,'existing_feature_arrays_read':0,'existing_start_end_arrays_read':0,'new_missing_TSM_count':0,'model_constructor_calls':0,'model_forward_calls':0,'GPU_probes':0,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();tmp=p.with_name('.'+p.name+'.tmp')
 with open(tmp,'xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def timeout(*a):raise TimeoutError('whole authorized CPU stage600s exhausted')
signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(0.001,600-(time.monotonic()-START)))
try:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ.get(k)=='1'
 allowed=sorted(os.sched_getaffinity(0));cpu=next((x for x in allowed if x not in (0,1)),allowed[0]);os.sched_setaffinity(0,{cpu});assert len(os.sched_getaffinity(0))==1
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 assert not OUT.exists(),'one CPU invocation only; preserve prior evidence'
 OUT.mkdir(parents=True);(OUT/'ATTEMPT_STARTED').write_text(result['started_utc'])
 assert sha(A/'DECISION.json')==AMENDMENT
 decision=load(A/'DECISION.json');assert decision['status']=='FINAL_DESIGN_CONDITIONAL_EXECUTION'
 basepath=R/'resume20261112/live-continuation/racnet-contract-astra/RUN_CONTRACT.json';assert sha(basepath)==BASE
 parent=load(basepath);assert decision['base_contract']['sha256']==BASE
 for name,expected in decision['frozen_metadata'].items():assert sha(W/'inputs'/name)==expected,('frozen manifest mismatch',name)
 targets={r['key']:r for r in load(W/'inputs/targets.json')};sentinel=decision['parity']['sentinel'];skey=sentinel['target']['key'];assert skey=='rc_train/stu10_0' and targets[skey]==sentinel['target']
 for t in decision['missing3_targets']:assert targets[t['key']]==t and not t['out_of_range_end']
 orig=pathlib.Path(decision['tsm']['official_source']['path']).read_bytes();candidate=(A/'gen_refTSM.import_math_candidate.py').read_bytes();assert hashlib.sha256(orig).hexdigest()==decision['tsm']['official_source']['sha256'];assert candidate==b'import math\n'+orig and hashlib.sha256(candidate).hexdigest()==decision['tsm']['candidate_sha256']
 generator=OUT/'gen_refTSM.import_math.py';generator.write_bytes(candidate)
 import numpy as np,torch,platform,scipy
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 def no_model(*args,**kwargs):result['model_constructor_calls']+=1;raise AssertionError('no model construction authorized')
 torch.nn.Module.__init__=no_model
 spec=importlib.util.spec_from_file_location('official_missing3_TSM',generator);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 runtime={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'CPU_affinity':list(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),'torch_threads':torch.get_num_threads(),'runtime_receipt_sha256':sha(W/'RUNTIME.json'),'env_spec_sha256':sha(W/'ENV_SPEC.json')};result['runtime']=runtime;result['amendment_sha256']=AMENDMENT;result['base_contract_sha256']=BASE
 source_npz=pathlib.Path(sentinel['author_member']['path']);assert source_npz.is_file() and source_npz.stat().st_size==sentinel['author_member']['bytes']
 np_load=np.load
 def only_sentinel(path,*args,**kwargs):
  assert pathlib.Path(path).resolve()==source_npz.resolve() and result['existing_author_TSM_arrays_read']==0,'old array read outside single sentinel forbidden'
  return np_load(path,*args,**kwargs)
 np.load=only_sentinel
 with np.load(source_npz,allow_pickle=False) as z:
  original=z['gt_tsm'];result['existing_author_TSM_arrays_read']+=1
 payload_sha=hashlib.sha256(original.tobytes(order='C')).hexdigest();assert payload_sha==sentinel['author_member']['arrays']['gt_tsm']['array_payload_sha256']
 began=time.monotonic();result['generated_TSM_calls']+=1;generated=module.GenerateTSM().gen_tsm(np.asarray(targets[skey]['bounds'],dtype=np.int64),250,filter_size=9,smooth_mode='8neighbors_expo');sentinel_seconds=time.monotonic()-began
 # Preserve both full matrices so a negative decision has exact primary evidence.
 np.save(OUT/'sentinel_author_gt_tsm.npy',original,allow_pickle=False);np.save(OUT/'sentinel_generated_gt_tsm.npy',generated,allow_pickle=False)
 shape_ok=original.shape==generated.shape==(250,250);finite=bool(np.isfinite(original).all() and np.isfinite(generated).all());a32=torch.FloatTensor(original).numpy();b32=torch.FloatTensor(generated).numpy();mask_mismatch=int(np.count_nonzero((original!=0)!=(generated!=0))) if shape_ok else None;f32_mismatch=int(np.count_nonzero(a32!=b32)) if shape_ok else None
 parity={'status':'PASS' if shape_ok and finite and mask_mismatch==0 and f32_mismatch==0 else 'FAIL','key':skey,'shape_author':list(original.shape),'shape_generated':list(generated.shape),'finite_both':finite,'nonzero_mask_mismatch_count':mask_mismatch,'float64_exact_equal':bool(np.array_equal(original,generated)),'float64_max_abs_diff':float(np.max(np.abs(original-generated))) if shape_ok and finite else None,'float32_mismatch_count':f32_mismatch,'float32_max_abs_diff':float(np.max(np.abs(a32-b32))) if shape_ok and finite else None,'native_conversion':'torch.FloatTensor then full element exact; rtol0atol0','source_author_TSM_payload_sha256':payload_sha,'generated_array_payload_sha256':hashlib.sha256(generated.tobytes(order='C')).hexdigest(),'official_source_sha256':decision['tsm']['official_source']['sha256'],'candidate_source_sha256':decision['tsm']['candidate_sha256'],'bounds_sha256':hashlib.sha256(json.dumps(targets[skey]['bounds'],separators=(',',':')).encode()).hexdigest(),'generator_parameters':{'filter_size':9,'smooth_mode':'8neighbors_expo'},'seconds':sentinel_seconds,'runtime':runtime,'amendment_sha256':AMENDMENT}
 dump(OUT/'PARITY.json',parity);result['parity_receipt_sha256']=sha(OUT/'PARITY.json');result['parity']=parity['status']
 print(json.dumps({'event':'fixed_sentinel_parity','status':parity['status'],'float32_mismatch_count':f32_mismatch,'float32_max_abs_diff':parity['float32_max_abs_diff'],'nonzero_mask_mismatch_count':mask_mismatch,'float64_exact_equal':parity['float64_exact_equal'],'float64_max_abs_diff':parity['float64_max_abs_diff'],'seconds':sentinel_seconds}),flush=True)
 if parity['status']!='PASS':raise RuntimeError('Fixed full-matrix native-FP32 parity failed; no missing3 TSM or GPU allowed')
 # Effective contract is a versioned child, never a replacement of the parent bytes.
 effective=copy.deepcopy(parent);effective['missing5_recovery']={'base_contract_sha256':BASE,'amendment_sha256':AMENDMENT,'permitted_feature_keys':[r['key'] for r in decision['missing5_inputs']],'permitted_generated_tsm_keys':decision['tsm']['generated_keys'],'allowed_overrides':decision['effective_contract_interface']['allowed_overrides'],'decision_path':str(A/'DECISION.json'),'qualification':decision['qualification']}
 for section in decision['effective_contract_interface']['numerical_sections_exact']:assert effective[section]==parent[section]
 assert not (W/'RUN_CONTRACT_MISSING5_V1.json').exists();dump(W/'RUN_CONTRACT_MISSING5_V1.json',effective);effective_sha=sha(W/'RUN_CONTRACT_MISSING5_V1.json');result['effective_contract_sha256']=effective_sha
 produced=[]
 for t in decision['missing3_targets']:
  n=t['length'];dest=REC/'tsm'/(t['key'].replace('/','__')+'.npy');assert not dest.exists() and not pathlib.Path(str(dest)+'.json').exists(),'existing new TSM must not be regenerated'
  began=time.monotonic();result['generated_TSM_calls']+=1;matrix=module.GenerateTSM().gen_tsm(np.asarray(t['bounds'],dtype=np.int64),n,filter_size=9,smooth_mode='8neighbors_expo')
  checks={'shape':matrix.shape==(n,n),'float64':matrix.dtype==np.float64,'finite':bool(np.isfinite(matrix).all()),'symmetric':bool(np.array_equal(matrix,matrix.T)),'range_0_1':bool(np.all((matrix>=0)&(matrix<=1))),'diagonal_one':bool(np.all(np.diag(matrix)==1))}
  dest.parent.mkdir(parents=True,exist_ok=True)
  with open(dest,'xb') as f:np.save(f,matrix,allow_pickle=False);f.flush();os.fsync(f.fileno())
  row={'key':t['key'],'path':str(dest),'shape':list(matrix.shape),'dtype':str(matrix.dtype),'checks':checks,'sha256':sha(dest),'bytes':dest.stat().st_size,'array_payload_sha256':hashlib.sha256(matrix.tobytes(order='C')).hexdigest(),'bounds':t['bounds'],'existing_start_end_target_sha256':t['target_sha256'],'base_contract_sha256':BASE,'amendment_sha256':AMENDMENT,'effective_contract_sha256':effective_sha,'generator_sha256':decision['tsm']['candidate_sha256'],'official_generator_sha256':decision['tsm']['official_source']['sha256'],'filter_size':9,'smooth_mode':'8neighbors_expo','parity_receipt_sha256':result['parity_receipt_sha256'],'runtime':runtime,'seconds':time.monotonic()-began,'provenance':'first_generated_missing_author_TSM'}
  dump(str(dest)+'.json',row);produced.append(row);result['new_missing_TSM_count']=len(produced);dump(OUT/'TSM_PRODUCTION_PROGRESS.json',produced)
  assert all(checks.values()),('new TSM check failed',t['key'],checks)
  assert sum(p.stat().st_size for root in (OUT,REC/'tsm') for p in root.rglob('*') if p.is_file())<=134217728
 assert result['generated_TSM_calls']==4 and result['new_missing_TSM_count']==3 and not torch.cuda.is_initialized()
 assert sha(basepath)==BASE
 for name,expected in decision['frozen_metadata'].items():assert sha(W/'inputs'/name)==expected
 result.update(status='PASS',new_tsm_receipts=produced,wall_seconds=time.monotonic()-START,CUDA_initialized=False,no_start_end_regeneration=True,no_existing_TSM_or_feature_overwrite=True);dump(OUT/'CPU_RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('new_tsm_receipts',)},indent=2))
except BaseException as exc:
 result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),wall_seconds=time.monotonic()-START,traceback=traceback.format_exc());dump(OUT/'CPU_RESULT.json',result);print(json.dumps(result,indent=2));raise
finally:signal.setitimer(signal.ITIMER_REAL,0)
