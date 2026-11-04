"""CPU synthetic NTR1 sampler/math/runtime tests; no natural GT/pretrained/GPU."""
from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
import pams.temporac.native_training_runtime_v1 as rt
from pams.temporac.cue import window_schedule
from pams.temporac.nola import nola_reconstruct_torch
from pams.temporac.native_bridge_v1 import reconstruct_native_response


@pytest.fixture(autouse=True)
def deterministic_cpu_backend():
    rt.configure_runtime()


def unit(n=33, kind='natural'):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n,269)).astype(np.float32)
    runs = np.array([[0,n-1]],np.int32)
    windows = window_schedule(runs)
    gates = np.tile([[.15,.25,.6]],(len(windows),1))
    cache = rt.precompute_nola(windows,gates,np.ones(n-1,bool))
    pulse = np.zeros(n-1,np.uint8); pulse[4]=1
    chi=np.zeros(n-1,np.float64); chi[4:20]=1/16
    mask=chi>0
    return rt.PreparedUnit('synthetic',x,runs,windows,gates,cache,pulse,chi,mask,
                           np.array([[4,20]],np.int64),kind)


def test_cycles_paired_complete_exposure_and_resume():
    eligible={'a':[f'a{i}' for i in range(7)],'b':['b1','b2'],'empty':[]}
    s=rt.PairedSampler(eligible,20260815)
    arm1=[s.draw(i) for i in range(140)]
    arm2=[rt.PairedSampler(eligible,20260815).draw(i) for i in range(140)]
    assert arm1==arm2
    assert {x['identity'] for x in arm1[:14]} == set(eligible['a']+eligible['b'])
    for i in range(0,140,2):
        assert {r['component'] for r in arm1[i:i+2]}=={'a','b'}
    for i in range(0,120,24):
        assert {r['x0_source'] for r in arm1[i:i+24]}==set(range(24))
    assert arm1[57:]==[s.draw(i) for i in range(57,140)]
    assert s.counters(57)['component_visits']==dict(__import__('collections').Counter(x['component'] for x in arm1[:57]))


def test_initial_tensor_sha_matched_all_seeds():
    for seed in rt.SEEDS:
        a,b=rt.initial_state(seed),rt.initial_state(seed)
        assert rt.tensor_state_sha(a)==rt.tensor_state_sha(b)
        assert all(torch.equal(a[k],b[k]) for k in a)
    assert rt.tensor_state_sha(rt.initial_state(rt.SEEDS[0])) != rt.tensor_state_sha(rt.initial_state(rt.SEEDS[1]))


@pytest.mark.parametrize('n',[33,216,900])
def test_vector_nola_reference_forward_and_gradient(n):
    rng=np.random.default_rng(5)
    support=np.ones(n-1,bool)
    runs=np.array([[0,n-1]],np.int32)
    if n>200:
        support[100:103]=False
        runs=np.array([[0,100],[103,n-1]],np.int32)
    windows=window_schedule(runs)
    gates=rng.uniform(.1,1,(len(windows),3)); gates/=gates.sum(1,keepdims=True)
    cache=rt.precompute_nola(windows,gates,support)
    z=torch.tensor(rng.normal(0,.5,(n-1,3)),dtype=torch.float64,requires_grad=True)
    p=z.sigmoid()
    fast=rt.vector_fusion(p,cache)
    ref=nola_reconstruct_torch(p,gates,windows,support).response
    np_ref=reconstruct_native_response(p.detach().numpy(),gates,windows,support).response
    assert float((fast-ref).abs().max())<=1e-12
    assert np.max(np.abs(fast.detach().numpy()-np_ref))<=1e-12
    assert float((fast.float()-ref.float()).abs().max())<=2e-7
    weight=torch.linspace(.1,1,n-1,dtype=torch.float64)
    ga=torch.autograd.grad((fast*weight).sum(),z,retain_graph=True)[0]
    gb=torch.autograd.grad((ref*weight).sum(),z)[0]
    nonzero=gb.abs()>1e-12
    assert float(((ga-gb).abs()[nonzero]/gb.abs()[nonzero]).max())<=1e-5
    assert torch.equal(ga[~torch.tensor(support)],torch.zeros_like(ga[~torch.tensor(support)]))


def test_loss_mask_cannot_change_nola_or_gradient_outside_target():
    u=unit()
    other=replace(u,loss_mask=~u.loss_mask)
    assert np.array_equal(u.cache.denominator,other.cache.denominator)
    assert np.array_equal(u.cache.responsibility,other.cache.responsibility)
    for variant in rt.VARIANTS:
        model=rt.TempoRACResponse().float()
        loss=rt.forward_unit(model,u,variant)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


def test_eight_micrographs_gradient_clip_step_equal_total_loss():
    torch.set_num_threads(1)
    state=rt.initial_state(20260815)
    m1,m2=rt.TempoRACResponse(),rt.TempoRACResponse()
    m1.load_state_dict(state); m2.load_state_dict(state)
    o1,o2=rt.make_optimizer(m1),rt.make_optimizer(m2)
    xs=[unit(kind='X0') for _ in range(7)]; natural=unit()
    receipt=rt.optimizer_step(m1,o1,xs,natural,'canonical',1,10000)
    o2.zero_grad(set_to_none=True)
    loss=sum(rt.forward_unit(m2,u,'canonical')*.5/7 for u in xs)+rt.forward_unit(m2,natural,'canonical')*.5
    loss.backward()
    raw=[p.grad.detach().clone() for p in m2.parameters()]
    reference_l2=math.sqrt(sum(float(g.double().square().sum()) for g in raw))
    norm=torch.nn.utils.clip_grad_norm_(m2.parameters(),1.,error_if_nonfinite=True, foreach=False)
    assert abs(float(norm)-reference_l2)<1e-6
    scale=min(1.,1./(reference_l2+1e-6))
    for p,g in zip(m2.parameters(),raw):
        torch.testing.assert_close(p.grad,g*scale,rtol=1e-6,atol=1e-7)
    for a,b in zip(m1.parameters(),m2.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)
    o2.param_groups[0]['lr']=rt.learning_rate(1,10000); o2.step()
    assert receipt.backward_calls==8 and receipt.optimizer_steps==1
    assert abs(receipt.pre_clip_global_norm-float(norm))<1e-6
    for a,b in zip(m1.parameters(),m2.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


def test_schedule_exact_old_10000_and_tie_missing_nan():
    from pams.temporac.runtime import learning_rate as old
    assert rt.horizon({'a':list(range(10001))})==10500
    assert rt.horizon({'a':[1]})==10000
    for step in range(1,10001):
        assert rt.learning_rate(step,10000)==old(step,total_steps=10000,base_lr=1e-3)
    candidates=[{'step':s,'objective':1.} for s in rt.checkpoint_steps(10000)]
    assert rt.select_checkpoint(candidates,10000)['step']==500
    with pytest.raises(ValueError): rt.select_checkpoint(candidates[:-1],10000)
    candidates[0]['objective']=float('nan')
    with pytest.raises(ValueError): rt.select_checkpoint(candidates,10000)
    with pytest.raises(ValueError): rt.horizon({'a':[]})


def plan():
    return rt.FrozenPlan({'T':10000,'seed':20260815,'eligible_by_component':{'a':['id']},
                          'initial_state_sha256':rt.tensor_state_sha(rt.initial_state(20260815))})


def test_exact_resume_optimizer_sampler_wronghash(tmp_path):
    p=plan(); state=rt.initial_state(20260815)
    a=rt.TempoRACResponse(); a.load_state_dict(state); oa=rt.make_optimizer(a)
    xs=[unit(kind='X0') for _ in range(7)]; natural=unit()
    rt.optimizer_step(a,oa,xs,natural,'canonical',1,10000)
    path=tmp_path/'recovery.pt'
    rt.commit_checkpoint(path,a,oa,p,step=1,variant='canonical',candidates=[],cumulative_wall_seconds=12)
    b=rt.TempoRACResponse(); ob=rt.make_optimizer(b)
    cp=rt.resume_exact(path,b,ob,p,'canonical')
    assert cp['step']==1 and cp['cumulative_wall_seconds']==12
    for model,opt in ((a,oa),(b,ob)):
        rt.optimizer_step(model,opt,xs,natural,'canonical',2,10000)
    assert rt.tensor_state_sha(a.state_dict())==rt.tensor_state_sha(b.state_dict())
    with pytest.raises(ValueError): rt.resume_exact(path,b,ob,rt.FrozenPlan(dict(p.payload,seed=20260816)),'canonical')
    with pytest.raises(ValueError): rt.resume_exact(path,b,ob,p,'capacity-control')
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError): rt.resume_exact(path,b,ob,p,'canonical')


def test_r2_admission_shared_gpu_boundary_unknown_lease():
    resource={'lease_id':'shared','device_uuid':'uuid','environment_sha':'a'*64,
              'owned_job_recovery_receipt':'untouched_health_receipt','gpu_index':1,
              'free_vram':8*rt.GIB,'free_disk':100*rt.GIB,
              'other_process_inventory':[{'pid':999,'bytes':2*rt.GIB}]}
    env=rt.memory_envelope(1000,1500)
    assert rt.admission(resource,env)
    with pytest.raises(ValueError): rt.admission(dict(resource,free_vram=8*rt.GIB-1),env)
    with pytest.raises(ValueError): rt.admission(dict(resource,lease_id=None),env)
    threshold=(4*rt.GIB-env['parameter_optimizer_bytes'])//(128*1024)
    assert rt.admission(resource,rt.memory_envelope(threshold,0))
    with pytest.raises(ValueError): rt.admission(resource,rt.memory_envelope(threshold+1,0))


def test_prediction_reference_full_support_no_target_once(monkeypatch):
    u=unit(n=33)
    clock=SimpleNamespace(mode='source_frame_only')
    identity=SimpleNamespace(response_edge_support=u.cache.support,run_bounds=u.run_bounds,
                              response_input=u.response_input,clock=clock)
    routing=SimpleNamespace(local_gates=u.gates,uniform_gates=np.full_like(u.gates,1/3),windows=u.windows)
    monkeypatch.setattr(rt,'compute_native_routing',lambda identity:routing)
    model=rt.TempoRACResponse().float()
    result=list(rt.predict_population(model,[('a',identity)],'canonical'))
    assert len(result)==1 and result[0][1].decoder_invocations==1
    assert result[0][1].support.all() and result[0][1].response.shape==(32,)
    empty=SimpleNamespace(response_edge_support=np.zeros(32,bool),run_bounds=np.empty((0,2),np.int32),clock=clock)
    result=list(rt.predict_population(model,[('b',empty)],'capacity-control'))
    assert result[0][1].status=='abstain' and result[0][1].decoder_invocations==1


def test_production_no_fixture_cap_and_pending_refusal():
    with pytest.raises(ValueError): rt._check_config({'fixture':True})
    with pytest.raises(ValueError): rt._require_train(None)
    with pytest.raises(ValueError): rt.build_training_plan([],{}, {'sources':[], 'views':[]},{})


def test_actual_producer_receipt_interoperability_and_failure_rejection():
    overlay={'schema':'temporac.native-clock-overlay.v1','status':'frozen_complete_census',
             'videos':977,'identities':2364,'records':[{'selection':'raw-retained'}]*977}
    nb={'schema':'temporac.native-learner.v1','status':'validated_population',
        'videos':977,'identities':2364,'records':[{}]*2364,'failures':[]}
    nt=dict(nb,schema='temporac.native-target.nt1',status='complete_population')
    raw={'videos':977,'persons':2364,'records':[{'status':'complete','persons':1,
          'annotation_length':1,'decoded_length':1} for _ in range(977)]}
    raw['records'][0].update(persons=1388,annotation_length=694649,decoded_length=694649)
    diag={'schema':'temporac.native-target-diagnostics.v1','splits':{
        'train':{'identity_denominator':1878,'full_population':{'videos':794}},
        'val':{'identity_denominator':486,'full_population':{'videos':183}}}}
    for name,payload in [('clock_overlay',overlay),('nb1',nb),('nt1',nt),('raw',raw),('diagnostics',diag)]:
        assert rt.qualified_receipt(name,payload)
    assert not rt.qualified_receipt('nb1',dict(nb,status='population_failures'))
    assert not rt.qualified_receipt('clock_overlay',dict(overlay,records=[{'selection':'rejected'}]*977))
    assert not rt.qualified_receipt('nb1',dict(nb,identities=2363))
    assert not rt.qualified_receipt('diagnostics',dict(diag,splits={}))


def test_mixed_traversal_eligibility_filters_loss_only_and_no_valid(monkeypatch):
    pulse=np.zeros(32,np.uint8); pulse[[0,16]]=1
    chi=np.full(32,1/16,np.float64); chi[0]=0;chi[1:16]=1/15
    original_mask=np.ones(32,bool)
    raw={'pulse':pulse,'chi':chi,'edge_mask':original_mask,
         'traversal_bounds':np.array([[0,16],[16,32]],np.int64)}
    monkeypatch.setattr(rt,'lf1_supervision',lambda *a,**k:raw)
    target=SimpleNamespace(metadata={'input_sha256':'a'*64})
    supervised=rt.legal_lf1_supervision(None,target)
    assert supervised['traversal_bounds'].tolist()==[[16,32]]
    assert not supervised['edge_mask'][:16].any() and supervised['edge_mask'][16:].all()
    assert original_mask.all() and raw['traversal_bounds'].tolist()==[[0,16],[16,32]]
    assert supervised['eligibility_ledger'][0]['reason']=='missing_positive_or_negative_LF1_mass'
    assert supervised['eligibility_ledger'][1]['eligible']
    chi[16]=0;chi[17:]=1/15
    supervised=rt.legal_lf1_supervision(None,target)
    assert supervised['traversal_bounds'].shape==(0,2) and not supervised['edge_mask'].any()


def test_admitted_device_sets_live_gpu1_allocator_cap_cpu_mock(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'set_device',lambda n:calls.append(('device',n)))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda n:SimpleNamespace(total_memory=24*rt.GIB))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda f,n:calls.append(('cap',f,n)))
    monkeypatch.setattr(rt,'_resource_check',lambda *a,**k:({'free_vram':10*rt.GIB},{999:rt.GIB}))
    device,baseline=rt.admitted_device(None)
    assert str(device)=='cuda:1' and baseline=={999:rt.GIB}
    assert calls==[('device',1),('cap',5/24,1)]


def test_diagnostic_cross_binding_rejects_changed_component_map(tmp_path):
    import json
    paths={}
    def write(name,data):
        path=tmp_path/(name+'.json'); path.write_text(json.dumps(data))
        paths[name]={'path':str(path),'sha256':rt.file_sha(path)}
        return path
    census=write('census',{})
    components=write('component_map',[])
    overlay=write('clock_overlay',{})
    nb=write('nb1',{'binding':{'census_sha256':rt.file_sha(census),'overlay_manifest_sha256':rt.file_sha(overlay)}})
    nt=write('nt1',{'binding':{'census_sha256':rt.file_sha(census)}})
    inputs={str(p.resolve()):rt.file_sha(p) for p in (census,components,nb,nt)}
    diagnostic=write('diagnostics',{'receipt':{'inputs_sha256':inputs},
                  'same_media_alias_groups':[{'media_keys':['a','b']}]*4})
    write('continuation',{'diagnostics_sha256':rt.file_sha(diagnostic)})
    cfg={'census':str(census),'component_map':str(components),'artifacts':paths}
    rt.verify_evidence_bindings(cfg,[])
    components.write_text('[{"changed":true}]')
    with pytest.raises(ValueError):rt.verify_evidence_bindings(cfg,[])


def test_r2a_harmless_other_growth_and_actual_limits_cpu_mock(monkeypatch,tmp_path):
    import os,shutil
    resource={'lease_id':'shared','device_uuid':'uuid','environment_sha':rt.json_sha(rt.environment_binding()),
        'owned_job_recovery_receipt':'untouched','gpu_index':1,'free_vram':10*rt.GIB,
        'free_disk':100*rt.GIB,'other_process_inventory':[],'disk_path':str(tmp_path)}
    live={'device_uuid':'uuid','free_vram':10*rt.GIB,
          'inventory':[{'pid':999999,'bytes':rt.GIB+1024**2},{'pid':os.getpid(),'bytes':rt.GIB}]}
    monkeypatch.setattr(rt,'_nvidia_query',lambda:live)
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=100*rt.GIB))
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    p=rt.FrozenPlan({'resource':resource,'planned_tensor_bytes':rt.memory_envelope(1000,1500),
                     'required_free_disk':11*rt.GIB})
    checked,_=rt._resource_check(p,starting=False,baseline={999999:rt.GIB})
    assert checked['other_allocation_changes'][999999]==1024**2
    live['free_vram']=2*rt.GIB-1
    with pytest.raises(RuntimeError):rt._resource_check(p,starting=False)
    live['free_vram']=10*rt.GIB;live['inventory'][-1]['bytes']=6*rt.GIB+1
    with pytest.raises(RuntimeError):rt._resource_check(p,starting=False)
    live['inventory'][-1]['bytes']=rt.GIB
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda n:5*rt.GIB+1)
    with pytest.raises(RuntimeError):rt._resource_check(p,starting=False)


def test_exhausted_arm_refused_before_train_or_predict_gpu_admission(monkeypatch,tmp_path):
    import json
    p=rt.FrozenPlan({'cpu_mock':'budget_only'})
    out=tmp_path/'canonical';out.mkdir()
    (out/'budget.json').write_text(json.dumps({'plan_sha256':p.sha256,'status':'complete',
        'wall_seconds':86400.,'gpu_occupation_seconds':86400.}))
    monkeypatch.setattr(rt,'_require_train',lambda p:None)
    monkeypatch.setattr(rt,'admitted_device',lambda p:pytest.fail('exhausted budget reached GPU admission'))
    cfg={'output':str(tmp_path),'variant':'canonical'}
    for call in (lambda:rt._train(cfg,p,{}),lambda:rt._predict(cfg,p)):
        with pytest.raises(RuntimeError,match='exhausted before'):
            call()


@pytest.mark.parametrize('stage',['train','predict'])
def test_setup_resource_failure_is_accounted_before_gpu_cpu_mock(stage,monkeypatch,tmp_path):
    import json
    p=rt.FrozenPlan({'T':10000,'cpu_mock':'setup_failure'})
    out=tmp_path/'canonical';out.mkdir()
    budget_path=out/'budget.json'
    budget_path.write_text(json.dumps({'plan_sha256':p.sha256,'status':'complete',
        'wall_seconds':12.,'gpu_occupation_seconds':12.}))
    candidates=[{'step':s,'objective':1.,'file':f'{s}.pt'} for s in rt.checkpoint_steps(10000)]
    (out/'selection.json').write_text(json.dumps({'status':'complete_training','plan_sha256':p.sha256,
        'candidates':candidates,'selected':candidates[0]}))
    original={'key':'kept','count':2,'status':'partial','sha256':'a'*64}
    (out/'prediction_progress.json').write_text(json.dumps({'records':[original]}))
    monkeypatch.setattr(rt,'_require_train',lambda p:None)
    monkeypatch.setattr(rt,'configure_runtime',lambda:None)
    def fail_admission(plan):
        assert json.loads(budget_path.read_text())['status']=='setup_running'
        raise RuntimeError('resource_incomplete: synthetic setup OOM')
    monkeypatch.setattr(rt,'admitted_device',fail_admission)
    cfg={'output':str(tmp_path),'variant':'canonical'}
    with pytest.raises(RuntimeError,match='synthetic setup OOM'):
        rt._train(cfg,p,{}) if stage=='train' else rt._predict(cfg,p)
    budget=json.loads(budget_path.read_text())
    assert budget['status']=='resource_incomplete' and budget['wall_seconds']>12
    failure=json.loads((out/('failure.json' if stage=='train' else 'prediction_failure.json')).read_text())
    assert failure['status']=='resource_incomplete' and failure['completed_prediction_records']==[original]
