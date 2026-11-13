"""Server-only metadata fixture. All process/GPU/dispatch interfaces are replaced."""
import contextlib, copy, importlib.util, json, os
from pathlib import Path
import sys, tempfile, types
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('successor_fixture',P/'resume_gpu1_after_racnet.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
real_runner=Path('/root/autodl-tmp/TempoRAC/resume20261112/continuation-2244/racnet/missing5-full-runner-v1/run_racnet_missing5_full.py')
sp=importlib.util.spec_from_file_location('finite_fixture',real_runner); finite=importlib.util.module_from_spec(sp); sp.loader.exec_module(finite)
cases=[]
def save(path,x):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(x))
def reject(name,fn):
    try: fn()
    except (RuntimeError,AssertionError,KeyError): cases.append(name); return
    raise AssertionError('accepted negative '+name)
with tempfile.TemporaryDirectory(prefix='metadata-fixture-',dir=P) as tmp:
    root=Path(tmp); m.R=root; m.W=root/'W'; m.D=root/'D'; m.A=root/'A'; m.OUTPUTS=root/'outputs'; m.SOURCE=root/'source'
    actual_runner=real_runner; real_runner=root/'runner.py'; real_runner.write_bytes(actual_runner.read_bytes())
    W=m.W; W.mkdir(); identity={'contract':'fixture-child','parent_contract':'fixture-parent','code':{'six_core':'fixture'}}
    common=types.SimpleNamespace(identity=lambda w:identity,input_path=lambda w,n:Path(w)/'inputs'/(n+'.json'),cpu_check_path=lambda w:Path(w)/'CPU.json')
    raw=[]; videos=[]
    for i in range(105):
        path=W/'feature-stat'/str(i); path.parent.mkdir(exist_ok=True); path.write_bytes(b'fixture')
        T=51245 if i==0 else 1; row={'key':str(i),'path':str(path),'input_T':T}; raw.append(row)
    save(W/'inputs/raw_feature_inputs105.json',raw)
    binding={'candidate_identity':identity,'inputs_sha256':m.sha(W/'inputs/raw_feature_inputs105.json')}
    for row in raw: videos.append({'key':row['key'],'T':row['input_T'],'binding':binding,'bytes':7})
    save(W/'FEATURES105_COMPLETE.json',{'status':'PASS','frames':51349,'videos':videos,'binding':binding})
    save(W/'inputs/READINESS.json',{'status':'PASS','required':1243,'missing':[],'author_members_available':1138,'recovered_members_available':5,'quva_features_available':100,'hashes':{'fixture':'input'}})
    save(W/'CPU.json',{'status':'PASS','candidate_identity':identity,'readiness_sha256':m.sha(W/'inputs/READINESS.json')})
    runid=finite.training_identity(W,common)
    cp=W/'checkpoints'; cp.mkdir(); (cp/'best.pt').write_bytes(b'NOT A MODEL: metadata fixture only'); (cp/'latest.pt').write_bytes(b'NOT A MODEL')
    selected={'file':str(cp/'best.pt'),'sha256':'saved-metadata-sha-only','next_epoch':12}
    commit={'identity':runid,'best_mae':selected,'latest':{'file':str(cp/'latest.pt'),'sha256':'latest-metadata','next_epoch':200}}
    save(cp/'COMMIT.json',commit)
    training={'identity':runid,'epochs':200,'attempted_steps':9600,'actual_steps':9599,'amp_skips':1,'selected':selected,'selection':{'epoch':11,'value':0.1},'tie_break':'earliest epoch'}
    save(W/'TRAINING_COMPLETE.json',training)
    for i in range(200):
        batches=[{'actual_update':not(i==199 and b==47)} for b in range(48)]
        save(W/'epochs'/('epoch_%03d.json'%i),{'epoch':i,'training_rows':755,'attempts':48,'actual_updates':sum(b['actual_update'] for b in batches),'amp_skips':int(i==199),'batches':batches,'validation':None if i<=10 else {'MAE':0.1 if i==11 else 1.0,'OBO':0.5}})
    expected=[]
    for pop,n in [('rc_test',152),('ucf_test',105),('quva_test',100)]:
        expected.extend({'key':pop+'/'+str(i),'population':pop,'sha256':'input-meta'} for i in range(n))
    save(W/'inputs/prediction_inputs.json',expected)
    finalbinding={'identity':runid,'checkpoint_sha256':selected['sha256'],'prediction_inputs_sha256':m.sha(W/'inputs/prediction_inputs.json')}
    pred=[dict(key=r['key'],population=r['population'],input_sha256=r['sha256'],binding=finalbinding,count=1.,nonfinite=False) for r in expected]
    save(W/'PREDICTIONS_357.json',pred); save(W/'SCORED_357.json',{'fixture':True}); save(W/'METRICS_357.json',{'fixture':True})
    final={'status':'COMPLETE','predictions':357,'binding':finalbinding}
    for field,name in [('prediction_sha256','PREDICTIONS_357.json'),('score_sha256','SCORED_357.json'),('metric_sha256','METRICS_357.json')]: final[field]=m.sha(W/name)
    save(W/'COMPLETE.json',final)
    runner_sha=m.sha(real_runner)
    approval={'candidate_identity':identity,'runner_sha256':runner_sha,'verdict':'PROVISIONAL_PASS'}
    for name in ('publication','review','prior'): save(root/(name+'.json'),approval)
    release={'candidate_identity':identity,'gpu_index':1,'gpu_uuid':'fixtureGPU'}
    for field,name in [('immutable_publication_receipt','publication'),('fresh_review_receipt','review'),('current_arm_complete_receipt','prior')]: release[field]={'path':str(root/(name+'.json')),'sha256':m.sha(root/(name+'.json'))}
    save(root/'release.json',release)
    launch={'candidate_identity':identity,'release':{'path':str(root/'release.json'),'sha256':m.sha(root/'release.json')},'command':['python','-B',str(real_runner)],'runner_sha256':runner_sha,'gpu_index':1,'gpu_uuid':'fixtureGPU','pid':1001}
    pipeline={'status':'COMPLETE','candidate_identity':identity,'runner_sha256':runner_sha,'root_release_sha256':launch['release']['sha256'],'stages':[{'stage':s,'status':'COMPLETE','exit_code':0,'pid':1002,'argv':['fixture']} for s in m.SEQUENCE],'final_complete_sha256':m.sha(W/'COMPLETE.json')}
    check=lambda:m.racnet_complete(common,finite,launch,pipeline)
    facts=check(); assert facts['actual_updates']==9599 and facts['amp_skips']==1; cases.append('full105_1243_200_9600_189_357_with_AMP_skip_PASS')
    pipeline['stages'][1]['exit_code']=1; reject('failed_stage_blocks',check); pipeline['stages'][1]['exit_code']=0
    pipeline['stages'].reverse(); reject('wrong_stage_order_blocks',check); pipeline['stages'].reverse()
    training['actual_steps']=9600; save(W/'TRAINING_COMPLETE.json',training); reject('inconsistent_AMP_accounting_blocks',check); training['actual_steps']=9599; save(W/'TRAINING_COMPLETE.json',training)
    training['selection']['epoch']=12; save(W/'TRAINING_COMPLETE.json',training); reject('wrong_best_MAE_blocks',check); training['selection']['epoch']=11; save(W/'TRAINING_COMPLETE.json',training)
    pred[-1]['key']=pred[0]['key']; save(W/'PREDICTIONS_357.json',pred); final['prediction_sha256']=m.sha(W/'PREDICTIONS_357.json'); save(W/'COMPLETE.json',final); pipeline['final_complete_sha256']=m.sha(W/'COMPLETE.json'); reject('missing_or_duplicate_final_member_blocks',check)
    # assess resource/lifecycle interfaces: fake canonical metadata, locks, process and commands.
    save(W/'GPU_PRODUCTION_LAUNCH.json',launch); folder=W/'finite_pipeline'/'fixture'; save(folder/'PIPELINE_COMPLETE.json',pipeline); save(folder/'PIPELINE_STATE.json',pipeline)
    for name in ('FINITE_PIPELINE.lock','ACTIVE.lock'): (W/name).touch()
    proof={'pending':m.RESIDUAL,'command':['NO_DISPATCH'],'events_sha256':'fixture'}
    original_control=m.ablation_control; original_sha=m.sha
    m.sha=lambda path:'tampered' if Path(path).name=='ablation_worker.py' else original_sha(path)
    reject('worker_source_tamper_blocks',original_control); m.sha=original_sha
    original_assess=m.assess
    (W/'native_common.py').write_text('fixture only')
    identity['code']['native_common.py']=m.sha(W/'native_common.py')
    save(W/'GPU_PRODUCTION_LAUNCH.json',launch)
    m.ablation_control=lambda:copy.deepcopy(proof); m.racnet_complete=lambda *args: facts
    m.module=lambda *args: common if args[0]=='racnet_gate_metadata' else finite
    m.owned_live=lambda pid,cmd:False
    m.shutil.disk_usage=lambda path:types.SimpleNamespace(free=200*1024**3)
    calls=[]
    def fakeout(cmd):
        calls.append(cmd)
        return '<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>' if '-x' in cmd else '1, fixtureGPU, 0'
    m.out=fakeout
    with contextlib.ExitStack() as stack: ready=m.assess(stack)
    assert ready['status']=='READY'; cases.append('fake_resource_free_READY')
    m.owned_live=lambda pid,cmd:True
    with contextlib.ExitStack() as stack: assert m.assess(stack)['status']=='WAIT'
    cases.append('live_root_WAIT_no_signal'); m.owned_live=lambda pid,cmd:False
    m.out=lambda cmd:'<nvidia_smi_log><gpu><processes><process_info/></processes></gpu></nvidia_smi_log>' if '-x' in cmd else '1, fixtureGPU, 20000'
    with contextlib.ExitStack() as stack: assert m.assess(stack)['status']=='WAIT'
    cases.append('occupied_GPU_WAIT_no_signal')
    root_release={'released_by':'root','authorized_residual':m.RESIDUAL,'gpu_index':1,'gpu_uuid':'fixtureGPU','entry_sha256':m.sha(m.__file__),'start_before_epoch':m.time.time()+600,'racnet_pipeline_complete':ready['racnet_pipeline_complete']}
    save(root/'root_release.json',root_release); m.validate_release(root/'root_release.json',ready); cases.append('explicit_root_release_PASS')
    root_release['authorized_residual']=m.RESIDUAL[:-1]; save(root/'root_release.json',root_release); reject('wrong_residual_release_blocks',lambda:m.validate_release(root/'root_release.json',ready)); root_release['authorized_residual']=m.RESIDUAL
    root_release['start_before_epoch']=0; save(root/'root_release.json',root_release); reject('expired_release_blocks',lambda:m.validate_release(root/'root_release.json',ready)); root_release['start_before_epoch']=m.time.time()+600; save(root/'root_release.json',root_release)
    # Exercise atomic hold archive + one-shot using fake dispatch, fixture paths only.
    m.A.mkdir(); (m.A/'HOLD_AFTER_CURRENT_GPU1').write_text('fixture original hold'); m.HOLD=m.sha(m.A/'HOLD_AFTER_CURRENT_GPU1')
    m.assess=lambda stack:copy.deepcopy(ready)
    dispatched=[]
    class FakeProcess:
        pid=900001
        def __init__(self,*args,**kw): dispatched.append(args[0]); assert args[0]==['NO_DISPATCH']
        def wait(self): return 0
    m.subprocess.Popen=FakeProcess; sys.argv=['fixture','--execute','--release',str(root/'root_release.json')]
    assert m.main()==0 and len(dispatched)==1 and (m.D/'HOLD_ORIGINAL_GPU1.archived.json').read_text()=='fixture original hold'; cases.append('atomic_archive_one_fake_dispatch_PASS')
    m.assess=original_assess
    assert m.main()==2 and len(dispatched)==1; cases.append('second_execute_no_dispatch')
    # Durable claim stays retained.
    require=m.require
    reject('durable_one_shot_guard_blocks',lambda:require(not(m.D/'one_shot.json').exists(),'no repeat'))

result={'status':'PASS','cases':cases,'case_count':len(cases),'metadata_only':True,'real_GPU_queries':0,'real_subprocess_dispatches':0,'model_imports':0,'array_or_checkpoint_loads':0,'real_W_mutations':0,'source_sha256':m.sha(P/'resume_gpu1_after_racnet.py')}
(P/'METADATA_FIXTURE.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


