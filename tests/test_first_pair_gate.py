"""CPU fake orchestration proves the first-pair gate actually blocks."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('first_pair_driver',ROOT/'orchestration/run_sequence.py')
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def setup_driver(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'W',tmp_path)
    (tmp_path/'execution_freeze.json').write_text(json.dumps({'source_commit':'9'*40}))
    calls=[]
    def run(stage,seed,variant):
        calls.append((stage,seed,variant))
        out=tmp_path/'runs'/str(seed);out.mkdir(parents=True,exist_ok=True)
        arm=out/variant;arm.mkdir(exist_ok=True)
        if stage in ('plan','validate'):
            (out/(stage+'.json')).write_text(json.dumps({'plan_sha256':'x','pending':[]}))
        if stage=='train':(arm/'selection.json').write_text('{}')
        if stage=='predict':(arm/'predictions.json').write_text(json.dumps({'status':'complete_population','records':[{}]*486}))
    def score(seed,variant):
        calls.append(('score',seed,variant))
        out=tmp_path/'scores'/str(seed)/variant;out.mkdir(parents=True,exist_ok=True)
        arm=tmp_path/'runs'/str(seed)/variant
        (out/'score.json').write_text(json.dumps({'status':'complete_population','seed':seed,'variant':variant,
            'source_commit':'9'*40,'full_population':{'videos':183,'identities':486},
            'receipt':{'inputs_sha256':{str((arm/(k+'.json')).resolve()):d.ref(arm/(k+'.json'))['sha256']
                                       for k in ('selection','predictions')}}}))
    monkeypatch.setattr(d,'run',run)
    monkeypatch.setattr(d,'score_arm',score)
    return calls


def test_default_stops_after_complete_first_pair(monkeypatch,tmp_path):
    calls=setup_driver(monkeypatch,tmp_path)
    monkeypatch.setattr(sys,'argv',['run_sequence.py'])
    d.main()
    assert {seed for _,seed,_ in calls}=={20260815}
    assert calls[-1]==('score',20260815,'capacity-control')
    state=json.loads((tmp_path/'campaign_state.json').read_text())
    assert state['status']=='first_pair_scored_awaiting_parent_diagnosis_and_resource_release'


def test_remaining_without_actual_score_resource_release_refused(monkeypatch,tmp_path):
    calls=setup_driver(monkeypatch,tmp_path)
    release=tmp_path/'release.json'
    release.write_text(json.dumps({'decision':'continue_remaining_four','source_commit':'9'*40}))
    monkeypatch.setattr(sys,'argv',['run_sequence.py','--remaining-release',str(release)])
    with pytest.raises(KeyError):d.main()
    assert calls==[]


def test_positive_parent_release_runs_only_remaining_four(monkeypatch,tmp_path):
    calls=setup_driver(monkeypatch,tmp_path)
    monkeypatch.setattr(sys,'argv',['run_sequence.py']);d.main();calls.clear()
    resource=tmp_path/'resource.json';resource.write_text('parent resource arrangement')
    release={'decision':'continue_remaining_four','source_commit':'9'*40,
             'resource_arrangement':d.ref(resource),'first_pair':{}}
    for variant in ('canonical','capacity-control'):
        arm=tmp_path/'runs'/'20260815'/variant
        release['first_pair'][variant]={'selection':d.ref(arm/'selection.json'),
            'predictions':d.ref(arm/'predictions.json'),'score':d.ref(tmp_path/'scores'/'20260815'/variant/'score.json')}
    path=tmp_path/'release.json';path.write_text(json.dumps(release))
    monkeypatch.setattr(sys,'argv',['run_sequence.py','--remaining-release',str(path)]);d.main()
    assert {seed for _,seed,_ in calls}=={20260816,20260817}
    assert sum(stage=='score' for stage,_,_ in calls)==4


@pytest.mark.parametrize('kind',['source','changed_scored_input'])
def test_stale_scored_release_blocks_remaining(monkeypatch,tmp_path,kind):
    calls=setup_driver(monkeypatch,tmp_path)
    monkeypatch.setattr(sys,'argv',['run_sequence.py']);d.main();calls.clear()
    resource=tmp_path/'resource.json';resource.write_text('parent resource arrangement')
    release={'decision':'continue_remaining_four','source_commit':'9'*40,
             'resource_arrangement':d.ref(resource),'first_pair':{}}
    for variant in ('canonical','capacity-control'):
        arm=tmp_path/'runs'/'20260815'/variant;score=tmp_path/'scores'/'20260815'/variant/'score.json'
        if kind=='source':
            report=json.loads(score.read_text());report['source_commit']='8'*40;score.write_text(json.dumps(report))
        else:(arm/'selection.json').write_text('{"changed_after_score":true}')
        release['first_pair'][variant]={'selection':d.ref(arm/'selection.json'),
            'predictions':d.ref(arm/'predictions.json'),'score':d.ref(score)}
    path=tmp_path/'release.json';path.write_text(json.dumps(release))
    monkeypatch.setattr(sys,'argv',['run_sequence.py','--remaining-release',str(path)])
    with pytest.raises(ValueError):d.main()
    assert calls==[]
