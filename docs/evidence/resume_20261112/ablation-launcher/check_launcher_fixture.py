"""Server CPU engineering fixture: actual21 mappings and synthetic finite queue transitions."""
import argparse
import copy
import json
from pathlib import Path
from ablation_worker import checked_plans, run_lane, validate_residual, sha, CONTRACT_SHA

ap = argparse.ArgumentParser()
ap.add_argument('--contract', type=Path, required=True)
ap.add_argument('--spec', type=Path, required=True)
ap.add_argument('--work', type=Path, required=True)
args = ap.parse_args()
assert not args.work.exists()
assert sha(args.contract) == CONTRACT_SHA
contract, spec = json.loads(args.contract.read_text()), json.loads(args.spec.read_text())
assert sha(args.spec) == contract['frozen_spec']['sha256']
plans = checked_plans(contract, spec)
assert len(contract['lanes']['gpu0']) == 12 and len(contract['lanes']['gpu1']) == 9
for name, plan in plans.items():
    jobs, cfg = plan['evaluations'], plan['run']['cfg']
    if name.startswith('wo_joint_rc_'):
        assert [(j['tag'], j['checkpoint'], j['n']) for j in jobs] == [('rc_test', 'best.pt', 152)]
        assert cfg['epochs'] == 80
    elif name.startswith('wo_joint_cx_'):
        assert [(j['tag'], j['checkpoint'], j['n']) for j in jobs] == [('cx_test', 'best.pt', 2403)]
        assert cfg['epochs'] == 80
    else:
        assert {j['tag'] for j in jobs} == {'rc_test', 'cx_test', 'ucf_test'}
        assert all(j['checkpoint'].startswith('best_') for j in jobs)
        assert cfg['epochs'] == (80 if name.startswith('wo_balance_') else 50)
    assert all(j['features'] == plan['run']['features'] and j['aux_features'] == cfg.get('aux_features', []) for j in jobs)
    assert all(j['tag'] != 'quva' for j in jobs)
    if name.startswith('wo_depth_'):
        assert plan['run']['features'].endswith('/UNION_stmae_p4') and not cfg.get('aux_features')
    if name.startswith('wo_shift_'):
        assert cfg['shift_prob'] == 0 and cfg['shift_eval']
    if name.startswith('wo_pyramid_'):
        assert cfg['rates'] == [4]
bad = copy.deepcopy(contract)
bad['new_training_runs'][0]['cfg_sha256'] = 'changed'
try:
    checked_plans(bad, spec)
except AssertionError:
    pass
else:
    raise AssertionError('Changed cfg binding must reject')

names = ['engineering_arm_a', 'engineering_arm_b', 'engineering_arm_c']
closed, order, events = set(), [], []
hold = [False]
def execute(name):
    # PLACEHOLDER engineering phases only; no training model, scientific inputs or metrics.
    order.append((name, 'train'))
    hold[0] = name == names[0]  # manual request arrives during training
    order.extend([(name, 'frozen_eval_1'), (name, 'frozen_eval_2'), (name, 'complete_receipt')])
    closed.add(name)
def held():
    assert order[-1][1] == 'complete_receipt', 'Hold cannot be checked mid arm'
    return hold[0]
def event(status, **extra):
    events.append({'status': status, **extra})
remaining = run_lane(names, closed.__contains__, execute, held, event)
assert remaining == names[1:] and closed == {names[0]}
assert events[-1] == {'status': 'held_after_complete', 'run': names[0], 'remaining': names[1:]}
hold[0] = False  # explicit operator clear/resume, never a timer or automatic retry
assert run_lane(names, closed.__contains__, execute, held, event) == []
assert [n for n, phase in order if phase == 'train'] == names
assert events[-1]['status'] == 'queue_complete'
failure_events, tried = [], []
def fail(name):
    tried.append(name)
    raise RuntimeError('injected engineering child failure')
try:
    run_lane(names, lambda n: False, fail, lambda: (_ for _ in ()).throw(AssertionError('No hold after failed arm')),
             lambda status, **kw: failure_events.append({'status': status, **kw}))
except RuntimeError:
    pass
else:
    raise AssertionError('Child failure must propagate')
assert tried == [names[0]] and failure_events[-1]['remaining'] == names
# A resumed lane can be killed after another complete receipt but before its state advances.
class AbruptStop(BaseException):
    pass
closed2, ran2, saved = {names[0]}, [], names[1:]
assert validate_residual(names, saved, closed2.__contains__) == names[1:]
def durable_then_stop(name):
    ran2.append(name)
    closed2.add(name)  # completed train, evaluations and receipt; prior held state remains stale
    raise AbruptStop()
try:
    run_lane(names, closed2.__contains__, durable_then_stop, lambda: False, event)
except AbruptStop:
    pass
else:
    raise AssertionError('Abrupt engineering stop must propagate')
assert saved == names[1:] and closed2 == set(names[:2])
assert validate_residual(names, saved, closed2.__contains__) == names[2:]
def finish(name):
    ran2.append(name)
    closed2.add(name)
assert run_lane(names, closed2.__contains__, finish, lambda: False, event) == []
assert ran2 == names[1:]  # the durable arm is not repeated
for invalid_saved, invalid_closed in [(names[2:], {names[0]}), (names[1:], {names[0], names[2]})]:
    try:
        validate_residual(names, invalid_saved, invalid_closed.__contains__)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing/holed completion cannot advance residual')
args.work.mkdir(parents=True)
report = {'scope': 'server CPU engineering configuration/queue fixture; no scientific result', 'passed': True,
          'mapped_runs': len(plans), 'mapped_epochs': sum(p['run']['cfg']['epochs'] for p in plans.values()),
          'plans': {n: {'epochs': p['run']['cfg']['epochs'], 'evaluations': p['evaluations']} for n, p in plans.items()},
          'cfg_drift_rejected': True, 'hold_after_all_evaluation_and_receipt_only': True,
          'explicit_residual_resume_skips_closed_prefix': True, 'failure_stops_before_successor': True,
          'held_resume_durable_progress_abrupt_stop_stale_state_resume': True,
          'missing_or_out_of_order_completion_rejected': True,
          'synthetic_event_trace': events, 'synthetic_failure_trace': failure_events,
          'no_quva_ablation': True, 'no_model_or_checkpoint_fixture_run': True}
(args.work / 'ENGINEERING_REPORT.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ['plans', 'synthetic_event_trace', 'synthetic_failure_trace']}, indent=2))
