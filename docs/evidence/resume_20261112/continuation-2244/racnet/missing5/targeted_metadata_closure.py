"""One targeted metadata closure and new-interface fixture; no old array/model work."""
import builtins,copy,contextlib,io,json,os,pathlib,shutil,sys
R=pathlib.Path('/root/autodl-tmp/TempoRAC');W=R/'baselines/racnet_native_v1';D=R/'resume20261112/continuation-2244/racnet/missing5-implementation';sys.path.insert(0,str(W))
import native_common as nc,native_inputs as ni,native_recovery as nr,run as entry,numpy as np,torch
torch.set_num_threads(1);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
os.sched_setaffinity(0,{2})
counts={'array_load':0,'science':0,'payload_open':0,'payload_hash':0,'model_constructor':0}
def noarray(*a,**kw):counts['array_load']+=1;raise AssertionError('no arrays in metadata closure')
def noscience(*a,**kw):counts['science']+=1;raise AssertionError('no science/GT/decoder in metadata closure')
def nomodel(*a,**kw):counts['model_constructor']+=1;raise AssertionError('no model construction')
np.load=noarray;ni.science=noscience;nc.science=noscience;torch.nn.Module.__init__=nomodel
op0,io0,sha0=builtins.open,io.open,nc.sha
def op(p,*a,**kw):
 if isinstance(p,(str,pathlib.Path)) and pathlib.Path(p).suffix in ('.npy','.npz'):counts['payload_open']+=1;raise AssertionError('payload open forbidden')
 return op0(p,*a,**kw)
def iop(p,*a,**kw):
 if isinstance(p,(str,pathlib.Path)) and pathlib.Path(p).suffix in ('.npy','.npz'):counts['payload_open']+=1;raise AssertionError('payload open forbidden')
 return io0(p,*a,**kw)
def sh(p):
 if pathlib.Path(p).suffix in ('.npy','.npz'):counts['payload_hash']+=1;raise AssertionError('payload hash forbidden')
 return sha0(p)
builtins.open=op;io.open=iop
for m in (nc,ni,nr,entry):m.sha=sh
for p in W.glob('*.py'):compile(p.read_text(),str(p),'exec')
assert not (D/'TARGETED_V2_METADATA_CLOSURE.json').exists(),'one real metadata closure only'
prior={n:nc.sha(W/'inputs'/n) for n in ('targets.json','score_labels.json','quva_raw_inputs.json','inputs.json','READINESS.json')};oldcpu=nc.sha(W/'CPU_CHECK.json')
sys.argv=['run.py','refresh-inputs','--work',str(W)]
with contextlib.redirect_stdout(io.StringIO()):entry.main()
ready=nc.load(nc.input_path(W,'READINESS'));cpu=nc.load(nc.cpu_check_path(W));raw=nc.load(W/'inputs/raw_feature_inputs105.json');pred=nc.load(nc.input_path(W,'prediction_inputs'));rows=nc.load(nc.input_path(W,'inputs'))
assert ready['status']==cpu['status']=='PENDING' and cpu['interfaces_status']=='PASS_CPU_INPUT_PREPARATION'
assert len(ready['missing'])==105 and ready['author_members_available']==1138 and ready['recovered_members_available']==ready['quva_features_available']==0
assert ready['native_author_TSMs']==883 and ready['native_derived_TSMs']==3
assert len(raw)==105 and sum(x['input_T'] for x in raw)==51349 and len(pred)==357 and len(rows)==1243
allowed=set(nr.PRED_FIELDS);assert all(set(x)<=allowed for x in raw) and all(set(x)<=allowed for x in pred)
assert all(nc.sha(W/'inputs'/n)==h for n,h in prior.items()) and nc.sha(W/'CPU_CHECK.json')==oldcpu
assert not nc.load(R/'baselines/racnet-author-assets20261112/STATE.json')['status'].startswith('complete')
# Only new metadata qualification interfaces, synthetic zero-byte placeholder never scientific data.
F=D/'metadata-fixture';assert not F.exists();F.mkdir()
for p in W.glob('*.py'):shutil.copy2(p,F/p.name)
for n in ['ENV_SPEC.json','RUNTIME.json','RUN_CONTRACT_MISSING5_V2.json']:shutil.copy2(W/n,F/n)
shutil.copytree(W/'vendor',F/'vendor')
binding={'candidate_identity':nc.identity(W),'synthetic_fixture':True};nc.savejson(F/'QUVA_BINDING.json',binding)
fake=F/'synthetic-placeholder.npz'
with op0(fake,'wb') as h:h.write(b'')
row=copy.deepcopy(next(x for x in rows if x['key']=='rc_test/stu6_68'));row['path']=str(fake)
member={'key':row['key'],'qualified_key':row['key'],'T':row['input_T'],'path':str(fake),'type':'generated_missing_author_member','arrays':{'img_feature':{'shape':[row['full_N'],37632],'dtype':'float32','CRC_and_finite_verified':True},'length':{'shape':[],'dtype':'int64'}},'asset_errors':[],'scalar_metadata':{'length':row['full_N']},'sha256':'f'*64,'bytes':0,'base_contract_sha256':nc.CONTRACT_SHA,'effective_contract_sha256':nc.EFFECTIVE_CONTRACT_SHA,'amendment_sha256':nc.AMENDMENT_SHA,'correction_sha256':nc.CORRECTION_SHA,'raw_video_sha256':row['raw_video_sha256'],'binding':binding}
nc.savejson(str(fake)+'.json',member);assert nr.qualify_recovered(row,F,None)['qualified_key']==row['key']
member['arrays']['img_feature']['shape'][1]=2048;nc.savejson(str(fake)+'.json',member)
try:nr.qualify_recovered(row,F,None);raise RuntimeError('wrongdimensionaccepted')
except AssertionError:pass
member['arrays']['img_feature']['shape'][1]=37632;member['effective_contract_sha256']='wrong';nc.savejson(str(fake)+'.json',member)
try:nr.qualify_recovered(row,F,None);raise RuntimeError('wrongcontractaccepted')
except AssertionError:pass
# Legacy fullprepare/cpu-check are forbidden on V2; neither may mutate parent evidence.
for action in ['prepare','cpu-check']:
 sys.argv=['run.py',action,'--work',str(W)]
 try:entry.main();raise RuntimeError('legacyconstructionaccepted')
 except AssertionError as e:assert 'V2 inherits' in str(e)
assert not torch.cuda.is_initialized() and all(v==0 for v in counts.values())
result={'status':'PASS','scope':'one actual receipt-only V2 closure plus synthetic new-interface metadata cases; no GPU or old array/model checks','cases':['new actual CLI binds parent/effective/amendment/correction and produces separate CPU/readiness chain','1138author+5derivedpending+100QUVApending;883authorTSM+3derivedTSM','exact105rawrows51349frames and357 label-freepredictionrows','five original manifests/oldCPU bytes unchanged','original failedauthorstate preserved','new generated NPZ receipt accepted;2048-dimension/wrongeffectivecontract rejected','V2 prepare/cpu-check forbidden before any old generation'],'forbidden_call_counts':counts,'CUDA_initialized':False,'candidate_identity':nc.identity(W),'new_readiness_sha256':nc.sha(nc.input_path(W,'READINESS')),'new_CPU_closure_sha256':nc.sha(nc.cpu_check_path(W)),'new_author_recovery_closure_sha256':nc.sha(W/'AUTHOR_ASSET_RECOVERY_CLOSURE.json'),'raw105_manifest_sha256':nc.sha(W/'inputs/raw_feature_inputs105.json'),'inherited_CPU_check_sha256':oldcpu,'frozen_prior_manifest_sha256':prior,'GPU_features_produced':0,'synthetic_placeholder_not_production':True}
nc.savejson(D/'TARGETED_V2_METADATA_CLOSURE.json',result);print(json.dumps(result,indent=2))
