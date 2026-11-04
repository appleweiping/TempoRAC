"""Focused CPU checks for the cached executor; no natural data or GPU work."""
from __future__ import annotations
import ast
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

# Standalone candidate also supports the pre-existing source without modifying it.
CANDIDATE = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('NTR1_BASE_SOURCE', str(CANDIDATE)))
sys.path.insert(0, str(BASE / 'src'))
import pams.temporac
pams.temporac.__path__.insert(0, str(CANDIDATE / 'src/pams/temporac'))
from pams.temporac import native_cached_training_runtime_v1 as runtime
from pams.temporac import native_training_runtime_v1 as core


class CachedRuntimeTest(unittest.TestCase):
    def resource(self):
        return dict(runtime.R3, lease_id='test', device_uuid='test-gpu',
                    environment_sha=core.json_sha(core.environment_binding()),
                    owned_job_recovery_receipt='test-only', free_vram=7 * core.GIB,
                    free_disk=100 * core.GIB, other_process_inventory=[{'pid': 1, 'bytes': core.GIB}],
                    disk_path=tempfile.gettempdir())

    def context(self):
        return runtime.ExecutionContext('binding-A', self.resource(), {},
            {'wall_seconds': 123., 'gpu_occupation_seconds': 120.})

    def plan(self):
        return core.FrozenPlan({'T':10000,'seed':20260815,'pending':[],
            'initial_state_sha256':'initial','eligible_by_component':{'c':['identity']},
            'planned_tensor_bytes':{'B_plan':2494397488},'required_free_disk':1})

    def test_r3_admission_actual_boundaries_and_shared_gpu(self):
        r=self.resource(); env={'B_plan':2494397488}
        self.assertTrue(runtime.admission(r,env))
        with self.assertRaises(ValueError):runtime.admission(dict(r,free_vram=7*core.GIB-1),env)
        self.assertTrue(runtime.admission(dict(r,free_vram=2*core.GIB),env,starting=False))
        with self.assertRaises(ValueError):runtime.admission(dict(r,free_vram=2*core.GIB-1),env,starting=False)
        with self.assertRaises(ValueError):runtime.admission(r,{'B_plan':5*core.GIB+1})
        with self.assertRaises(ValueError):runtime.admission(dict(r,allocator_reserved_cap_bytes=5*core.GIB),env)
        with self.assertRaises(ValueError):runtime.admission(dict(r,lease_id=''),env)

    def test_runtime_checks_actual_own_memory_and_other_growth(self):
        live={'device_uuid':'test-gpu','free_vram':7*core.GIB,'inventory':[{'pid':9,'bytes':8*core.GIB}]}
        with mock.patch.object(core,'_nvidia_query',return_value=live), mock.patch.object(torch.cuda,'is_initialized',return_value=True), mock.patch.object(torch.cuda,'memory_reserved',return_value=4*core.GIB):
            got,_=runtime.resource_check(self.plan(),self.context(),starting=False,baseline={9:1})
            self.assertEqual(got['other_allocation_changes'],{9:8*core.GIB-1})
            live['inventory'].append({'pid':os.getpid(),'bytes':5*core.GIB+1})
            with self.assertRaises(RuntimeError):runtime.resource_check(self.plan(),self.context(),starting=False)
            live['inventory'].pop()
            with mock.patch.object(torch.cuda,'memory_reserved',return_value=4*core.GIB+1):
                with self.assertRaises(RuntimeError):runtime.resource_check(self.plan(),self.context(),starting=False)

    def test_budget_resume_inherits_and_never_resets(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget.json'; plan=self.plan();ctx=self.context()
            b=runtime.Budget(path,plan,ctx,'train');self.assertGreaterEqual(b.elapsed(),123)
            b.record('complete',step=500)
            old=runtime.read_json(path)
            later=runtime.Budget(path,plan,ctx,'predict');self.assertGreaterEqual(later.elapsed(),old['wall_seconds'])
            later.record('complete_prediction')
            with self.assertRaises(ValueError):runtime.Budget(path,plan,replace(ctx,binding_sha256='B'),'train')
            b=runtime.Budget(path,plan,ctx,'predict')
            with self.assertRaises(RuntimeError):runtime.Budget(path,plan,ctx,'train')
            b.prior=86400
            with self.assertRaises(RuntimeError):b.check()

    def test_checkpoint_binds_execution_and_restores_optimizer_rng(self):
        with tempfile.TemporaryDirectory() as d:
            plan=self.plan();ctx=self.context();model=torch.nn.Linear(2,1); opt=core.make_optimizer(model)
            loss=model(torch.tensor([[1.,2.]])).sum();loss.backward();opt.step()
            path=Path(d)/'recovery.pt'
            runtime.commit_checkpoint(path,model,opt,plan,ctx,step=1,variant='canonical',candidates=[],cumulative_wall_seconds=130.)
            saved={k:v.clone() for k,v in model.state_dict().items()}
            torch.manual_seed(91);model2=torch.nn.Linear(2,1);opt2=core.make_optimizer(model2)
            cp=runtime.resume_exact(path,model2,opt2,plan,ctx,'canonical')
            self.assertEqual(cp['step'],1);self.assertEqual(cp['sampler'],plan.sampler.counters(1))
            self.assertTrue(all(torch.equal(saved[k],v) for k,v in model2.state_dict().items()))
            self.assertEqual(opt2.state_dict()['param_groups'],opt.state_dict()['param_groups'])
            self.assertTrue(opt2.state)
            with self.assertRaises(ValueError):runtime.resume_exact(path,model2,opt2,plan,replace(ctx,binding_sha256='wrong'),'canonical')
            path.write_bytes(path.read_bytes()+b'bad')
            with self.assertRaises(ValueError):runtime.resume_exact(path,model2,opt2,plan,ctx,'canonical')

    def test_tune_cached_order_and_exact_objective(self):
        model=torch.nn.Linear(1,1);units={s:tuple(SimpleNamespace(value=float(s*7+b)) for b in range(7)) for s in range(32)}
        calls=[]
        def forward(m,u,v):calls.append(u.value);return torch.tensor(u.value,dtype=torch.float64)
        with mock.patch.object(core,'forward_unit',side_effect=forward), mock.patch.object(core,'prepare_x0_unit',side_effect=lambda s,a:units[s]):
            reference=core.score_x0_tune(model,'canonical',object());calls.clear()
            checks=[];actual=runtime.score_x0_tune_cached(model,'canonical',units,lambda:checks.append(True))
        self.assertEqual(actual,reference)
        self.assertEqual(calls,[float(s*7+b) for s in range(24,32) for b in range(7)])
        self.assertEqual(len(checks),112);self.assertTrue(model.training)

    def test_prediction_no_support_keeps_abstention_and_one_decode(self):
        model=torch.nn.Linear(1,1)
        identity=SimpleNamespace(response_edge_support=np.zeros(4,dtype=bool),run_bounds=np.empty((0,2),np.int32),clock=SimpleNamespace(mode='source_frame_only'))
        with mock.patch.object(core,'compute_native_routing',side_effect=AssertionError('NUDFT called')), mock.patch.object(core,'decode_native_identity',wraps=core.decode_native_identity) as decoder:
            items=list(runtime.predict_population_cached(model,[('unavailable',identity,{})],'canonical'))
        self.assertEqual(len(items),1);self.assertEqual(items[0][1].count,-1)
        self.assertEqual(items[0][1].decoder_invocations,1);self.assertEqual(decoder.call_count,1)

    def test_prediction_matches_original_reference_for_both_arms(self):
        class Model(torch.nn.Linear):
            def cached_responses(self, values, bounds):
                n=len(values)-1
                return SimpleNamespace(edge_probabilities=torch.linspace(.1,.9,n*3).reshape(n,3))
        model=Model(1,1);support=np.ones(8,dtype=bool)
        identity=SimpleNamespace(response_input=np.zeros((9,269),np.float32),response_edge_support=support,
            run_bounds=np.array([[0,8]],np.int32),clock=SimpleNamespace(mode='source_frame_only'))
        arrays={'windows':np.array([[0,0,6],[0,2,8]],np.int64),'local_gates':np.array([[.2,.3,.5],[.6,.3,.1]])}
        for variant in core.VARIANTS:
            gates=arrays['local_gates'] if variant=='canonical' else np.full((2,3),1/3)
            branch=model.cached_responses(identity.response_input,identity.run_bounds).edge_probabilities.numpy()
            expected=core.decode_native_identity(core.reconstruct_native_response(branch.astype(np.float64),gates,[(0,6),(2,8)],support).response,identity)
            actual=list(runtime.predict_population_cached(model,[('key',identity,arrays)],variant))[0][1]
            self.assertEqual(actual.response.tobytes(),expected.response.tobytes())
            self.assertEqual(actual.component_bounds.tobytes(),expected.component_bounds.tobytes())
            self.assertEqual(actual.count,expected.count)

    def test_scientific_core_reused_and_no_replanning(self):
        tree=ast.parse(Path(runtime.__file__).read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden={'_plan','_load_inputs','preprocess_native','compute_native_routing','prepare_x0_unit','prepare_natural_unit'}
        for node in calls:
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
            self.assertNotIn(name,forbidden)
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_cached')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='optimizer_step' for n in ast.walk(train)))
        self.assertEqual(core.make_optimizer(torch.nn.Linear(1,1)).defaults['foreach'],False)

    def test_wrong_parent_sha_fails_before_read(self):
        with self.assertRaises(ValueError):runtime.load_verified_frozen_plan({'sha256':'bad'},{'sha256':'bad'})

    def test_production_parity_rejects_stale_kernel_or_environment(self):
        sources={name:'sha-'+str(i) for i,name in enumerate(runtime.PARITY_SOURCE_FILES)}
        env={'python':'test','numpy':'test','torch':'test'}
        evidence=dict(schema='temporac.ntr1.fast-cache-parity.v1',status='complete',all_bytes_equal=True,
                      plan_sha256='plan',environment=env,new_source_files=dict(sources))
        runtime.verify_production_parity(evidence,'plan',sources,env)
        for name in runtime.PARITY_SOURCE_FILES:
            with self.subTest(name=name):
                bad=dict(evidence,new_source_files=dict(sources,**{name:'stale'}))
                with self.assertRaises(ValueError):runtime.verify_production_parity(bad,'plan',sources,env)
                missing=dict(sources);missing.pop(name)
                with self.assertRaises(ValueError):runtime.verify_production_parity(dict(evidence,new_source_files=missing),'plan',sources,env)
        with self.assertRaises(ValueError):runtime.verify_production_parity(evidence,'plan',sources,{'python':'changed'})
        with self.assertRaises(ValueError):runtime.verify_production_parity(dict(evidence,all_bytes_equal=False),'plan',sources,env)


if __name__=='__main__':unittest.main(verbosity=2)
