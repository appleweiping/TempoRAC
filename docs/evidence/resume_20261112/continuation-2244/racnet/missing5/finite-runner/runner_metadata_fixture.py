"""New operational helper metadata tests; no releases, subprocesses or models."""
import ast,hashlib,importlib.util,json,pathlib,time
D=pathlib.Path('/root/autodl-tmp/TempoRAC/resume20261112/continuation-2244/racnet/missing5-full-runner-v1');source=D/'run_racnet_missing5_full.py';tree=ast.parse(source.read_text(encoding='utf-8-sig'));compile(tree,str(source),'exec')
assert not any(isinstance(n,ast.While) for n in ast.walk(tree))
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='sleep' for n in ast.walk(tree))
assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen' for n in ast.walk(tree))==1
assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='wait' for n in ast.walk(tree))==1
assert not any(isinstance(n,ast.Import) and any(x.name.split('.')[0] in ('torch','numpy','cv2') for x in n.names) for n in ast.walk(tree))
spec=importlib.util.spec_from_file_location('synthetic_finite_runner',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert m.SEQUENCE==['extract-quva','refresh-inputs','train','evaluate']
F=D/'SYNTHETIC_metadata_fixture';assert not F.exists();F.mkdir()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
class Common:
 def identity(self,w):return {'synthetic_fixture':True}
 def input_path(self,w,n):return w/'inputs'/(n+'.json')
 def cpu_check_path(self,w):return w/'CPU.json'
c=Common();assert not m.feature_complete(F,c) and not m.inputs_complete(F,c) and not m.train_complete(F,c) and not m.evaluation_complete(F,c)
assert not m.train_resume(F,c)
raw=[{'key':'SYNTHETIC_'+str(i),'input_T':493 if i==0 else 489,'path':str(F/'empty'/str(i))} for i in range(105)]
assert sum(x['input_T'] for x in raw)==51349
binding={'candidate_identity':c.identity(F),'inputs_sha256':None};write(c.input_path(F,'unused'),{})
write(F/'inputs/raw_feature_inputs105.json',raw);binding['inputs_sha256']=m.sha(F/'inputs/raw_feature_inputs105.json');videos=[]
for r in raw:
 p=pathlib.Path(r['path']);p.parent.mkdir(exist_ok=True);p.write_bytes(b'');videos.append({'key':r['key'],'T':r['input_T'],'bytes':0,'binding':binding})
write(F/'FEATURES105_COMPLETE.json',{'status':'PASS','frames':51349,'videos':videos,'binding':binding});assert m.feature_complete(F,c)
ready={'status':'PASS','required':1243,'missing':[],'author_members_available':1138,'recovered_members_available':5,'quva_features_available':100,'hashes':{'synthetic':'only'}};write(c.input_path(F,'READINESS'),ready);write(c.cpu_check_path(F),{'status':'PASS','candidate_identity':c.identity(F),'readiness_sha256':m.sha(c.input_path(F,'READINESS'))});assert m.inputs_complete(F,c)
identity=m.training_identity(F,c);p=F/'checkpoints/SYNTHETIC_NOT_A_MODEL.pt';p.parent.mkdir();p.write_bytes(b'SYNTHETIC_METADATA_ONLY');ref={'file':str(p),'sha256':m.sha(p),'next_epoch':50};commit={'identity':identity,'latest':ref,'best_mae':ref};write(F/'checkpoints/COMMIT.json',commit);assert m.train_resume(F,c)
commit['identity']={'wrong':True};write(F/'checkpoints/COMMIT.json',commit)
try:m.train_resume(F,c);raise RuntimeError('wrongtrajectoryaccepted')
except AssertionError:pass
commit['identity']=identity;write(F/'checkpoints/COMMIT.json',commit);write(F/'TRAINING_COMPLETE.json',{'identity':identity,'epochs':200,'attempted_steps':9600,'selected':ref});assert m.train_complete(F,c)
write(c.input_path(F,'prediction_inputs'),{'synthetic':True});value={'status':'COMPLETE','predictions':357,'binding':{'identity':identity,'checkpoint_sha256':ref['sha256'],'prediction_inputs_sha256':m.sha(c.input_path(F,'prediction_inputs'))}}
for field,name in [('prediction_sha256','PREDICTIONS_357.json'),('score_sha256','SCORED_357.json'),('metric_sha256','METRICS_357.json')]:write(F/name,{'SYNTHETIC_NOT_RESULTS':True});value[field]=m.sha(F/name)
write(F/'COMPLETE.json',value);assert m.evaluation_complete(F,c)
try:m.validate_root_release({'released_by':'root','authorized_sequence':m.SEQUENCE,'runner_sha256':'synthetic','start_before_epoch':time.time()-1},c,'synthetic');raise RuntimeError('expiredstartupaccepted')
except AssertionError as e:assert 'expired' in str(e)
result={'status':'PASS','scope':'synthetic operational metadata/static tests only; not model/checkpoint/feature/metric validation','runner_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':['finiteexact4stage sequence;onePopen/waitsite;no while/sleep/modelimports','no-completionroutesselectnewstages','bound105marker/1243CPUqualificationcompletedskip','matchingCOMMITselectsexplicitresume;wrongidentityrejects','200/357completionmarkersacceptedonlywithbindings','expiredrootstartuprejected'],'actual_pipeline_invoked':False,'actual_ownership_receipts_created':0,'actual_GPU_queries':0,'actual_subprocesses':0,'core6_source_mutation':False};write(D/'RUNNER_METADATA_FIXTURE.json',result);print(json.dumps(result,indent=2))
