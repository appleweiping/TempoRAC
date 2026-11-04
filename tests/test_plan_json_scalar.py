"""Actual _load_inputs metadata path; all identity/target inputs are artificial."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pams.temporac import native_target_diagnostics_v1 as diagnostics

path=Path(__file__).resolve().parents[1]/'src/pams/temporac/native_training_runtime_v1.py'
spec=importlib.util.spec_from_file_location('metadata_fix_runtime',path)
runtime=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=runtime
spec.loader.exec_module(runtime)


@pytest.mark.parametrize('dtype,bounds,want',[(np.int64,[[0,17]],18),(np.int32,[[0,17]],18),(np.int64,[],0)])
def test_actual_load_inputs_emits_json_integer(monkeypatch,tmp_path,dtype,bounds,want):
    cfg={'census':str(tmp_path/'census.jsonl'),'component_map':str(tmp_path/'components.json'),
         'bridge_root':str(tmp_path/'bridge'),'target_root':str(tmp_path/'target')}
    Path(cfg['component_map']).write_text('[]')
    row={'media_key':'a'*64,'persons':[{'slot':0}],'split':'train','length':18}
    key=row['media_key']+'_0'
    for root,suffixes in [('bridge_root',['.npz','.json','.derived.npz']),('target_root',['.npz','.json'])]:
        base=Path(cfg[root])/'identities/train';base.mkdir(parents=True)
        for suffix in suffixes:(base/(key+suffix)).write_bytes(b'artificial existence only')
    monkeypatch.setattr(runtime,'_module_from_script',lambda _:SimpleNamespace(read_census=lambda _:[row]))
    monkeypatch.setattr(diagnostics,'source_membership',lambda *_:({row['media_key']:'c0'},[[{'split':'train'}]]*4))
    identity=SimpleNamespace(run_bounds=np.asarray(bounds,dtype=dtype).reshape(-1,2),status='artificial')
    target=SimpleNamespace(traversal_bounds=np.asarray([[0,17]],dtype=np.int64))
    monkeypatch.setattr(runtime,'_identity_record',lambda *_:(identity,target,{}))
    monkeypatch.setattr(runtime,'legal_lf1_supervision',lambda *_:{'traversal_bounds':[[0,17]],'eligibility_ledger':[]})
    _,population,_,_,_=runtime._load_inputs(cfg)
    assert population[0]['max_run']==want
    assert type(population[0]['max_run']) is int
    assert json.loads(json.dumps(population))[0]['max_run']==want
