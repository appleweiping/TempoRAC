"""Finite exact21-arm lanes. Server only; manual hold is read after complete arms."""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time

CONTRACT_SHA = 'dc60e3d493e00b8aa20c600661b6991bfeed14319f65e32aea543ace92a8a167'
TRAIN_SHA = '489dc77b91a21d21215784ff49961edb20dab4fb'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()

def cfg_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(obj, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def checked_plans(contract, spec):
    """Map only contract roles; cfg feature roots remain the exact frozen per-arm roots."""
    runs = {r['name']: r for r in spec['runs']}
    assert len(contract['new_training_runs']) == 21
    plans = {}
    for approved in contract['new_training_runs']:
        name = approved['name']
        r = runs[name]
        assert cfg_sha(r['cfg']) == approved['cfg_sha256'], (name, 'cfg digest')
        for k in ['seed', 'features', 'train_split', 'val_split', 'test_split']:
            assert r[k] == approved[k], (name, k)
        assert r['cfg']['epochs'] == approved['epochs'] in [50, 80]
        assert not r['cfg'].get('init_from') and approved['fresh_initialization']
        vals, tests = r['val_split'].split(','), r['test_split'].split(',')
        assert len(vals) == len(tests)
        jobs = []
        for val, test in zip(vals, tests):
            if len(vals) > 1:
                rule = contract['evaluation']['union_arms'][test]
                ckpt, n, n_pos = rule['checkpoint'], rule['n'], rule['n_positive']
            else:
                rule = contract['evaluation'][{'rc_test': 'single_rc', 'cx_test': 'single_cx'}[test]]
                assert rule['select_role'] == val and rule['test_role'] == test
                ckpt, n, n_pos = rule['checkpoint'], rule['test'], rule['n_positive']
            assert ckpt == (f'best_{val}.pt' if len(vals) > 1 else 'best.pt')
            jobs.append({'target': val, 'split': test, 'tag': test, 'checkpoint': ckpt,
                         'n': n, 'n_positive': n_pos, 'features': r['features'],
                         'aux_features': r['cfg'].get('aux_features', [])})
        plans[name] = {'run': r, 'evaluations': jobs}
    lane_names = contract['lanes']['gpu0'] + contract['lanes']['gpu1']
    assert len(lane_names) == len(set(lane_names)) == len(plans) == 21
    assert set(lane_names) == set(plans)
    assert sum(p['run']['cfg']['epochs'] for p in plans.values()) == 1320
    assert all(j['tag'] != 'quva' for p in plans.values() for j in p['evaluations'])
    return plans

def run_lane(names, complete, run_one, hold_requested, event):
    """Only complete arms advance or release a lane; no hold check inside run_one."""
    completed = [complete(n) for n in names]
    prefix = next((i for i, done in enumerate(completed) if not done), len(names))
    assert all(not done for done in completed[prefix:]), 'Completed outputs must be a lane prefix'
    for i, name in enumerate(names[prefix:], prefix):
        try:
            run_one(name)
            assert complete(name), 'No successor before durable complete receipt'
        except Exception as exc:
            event('failed', run=name, remaining=names[i:], error=repr(exc))
            raise
        remaining = names[i + 1:]
        if remaining and hold_requested():
            event('held_after_complete', run=name, remaining=remaining)
            return remaining
    event('queue_complete', remaining=[])
    return []

def validate_residual(names, saved, complete):
    """A hard stop may leave the held/failed state behind durable completed arms."""
    assert len(saved) <= len(names) and saved == names[len(names) - len(saved):], 'Saved residual is not a lane suffix'
    done = [complete(n) for n in names]  # validates provenance and hashes of every skipped arm
    prefix = next((i for i, value in enumerate(done) if not value), len(names))
    assert all(not value for value in done[prefix:]), 'Completed outputs must be a lane prefix'
    pending = names[prefix:]
    assert len(pending) <= len(saved) and pending == saved[len(saved) - len(pending):], 'Residual may only advance past verified complete arms'
    return pending

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = Path('/root/autodl-tmp/TempoRAC')
    control = root / 'resume20261112/ablation-continuation'
    outputs = root / 'runs/frozen_known_source_v1_ablation20261112'
    assert root.resolve() == root and args.source.resolve().is_relative_to(root)
    assert sha(args.contract) == CONTRACT_SHA
    contract = json.loads(args.contract.read_text())
    assert args.commit == contract['training_runtime_commit'] == TRAIN_SHA
    cmd_output = lambda cmd: subprocess.check_output(cmd, text=True, creationflags=FLAGS).strip()
    assert cmd_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD']) == TRAIN_SHA
    assert not cmd_output(['git', '-C', str(args.source), 'status', '--porcelain'])
    assert sha(args.source / 'temporac_cvpr/train.py') == '33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4'
    spec_path = Path(contract['frozen_spec']['path'])
    assert sha(spec_path) == contract['frozen_spec']['sha256']
    assert sha(Path(contract['manifest']['path'])) == contract['manifest']['sha256']
    spec = json.loads(spec_path.read_text())
    plans = checked_plans(contract, spec)
    lane = contract['lanes'][f'gpu{args.gpu}']
    assert args.resume or not any((outputs / n).exists() for n in lane), 'Existing lane requires explicit --resume'
    control.mkdir(parents=True, exist_ok=True)
    worker_pidfile = control / f'worker_gpu{args.gpu}.pid'
    assert not (worker_pidfile.exists() and Path(f'/proc/{worker_pidfile.read_text().strip()}').exists()), 'Lane worker still alive'
    worker_pidfile.write_text(str(os.getpid()))
    state_path = control / f'lane_gpu{args.gpu}.json'
    hold_path = control / f'HOLD_AFTER_CURRENT_GPU{args.gpu}'
    worker_sha = sha(Path(__file__))
    py = str(root / 'env/bin/python')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(args.source),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               CUDA_CACHE_PATH=str(root / '.cache/cuda'), XDG_CACHE_HOME=str(root / '.cache'),
               TORCH_HOME=str(root / '.cache/torch'), HF_HOME=str(root / '.cache/huggingface'), TMPDIR=str(root / 'tmp'))
    rows = json.loads(Path(contract['manifest']['path']).read_text())
    expected = {s: {f"{x['split']}__{x['video_id']}" for x in rows if x['split'] == s}
                for s in ['rc_test', 'cx_test', 'ucf_test']}

    def event(status, **extra):
        obj = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'worker_pid': os.getpid(),
               'gpu': args.gpu, 'status': status, **extra}
        with (control / f'worker_gpu{args.gpu}.events.jsonl').open('a') as f:
            f.write(json.dumps(obj) + '\n')
        if status in {'held_after_complete', 'queue_complete', 'failed'}:
            write_json(state_path, {**obj, 'contract_sha256': CONTRACT_SHA, 'worker_sha256': worker_sha,
                                    'source_commit': TRAIN_SHA})
        print(json.dumps(obj), flush=True)

    def provenance(name):
        r = plans[name]['run']
        return {'run': name, 'source': str(args.source), 'source_commit': TRAIN_SHA, 'worker_sha256': worker_sha,
                'contract_sha256': CONTRACT_SHA, 'spec_sha256': contract['frozen_spec']['sha256'],
                'manifest_sha256': contract['manifest']['sha256'], 'seed': r['seed'],
                'cfg': r['cfg'], 'cfg_sha256': cfg_sha(r['cfg']), 'features': r['features'],
                'train_split': r['train_split'], 'val_split': r['val_split'], 'test_split': r['test_split'],
                'evaluations': plans[name]['evaluations']}

    def complete(name):
        path = outputs / name / 'complete_known_source.json'
        if not path.exists():
            return False
        got = json.loads(path.read_text())
        assert got['provenance'] == provenance(name)
        for filename, digest in got['file_sha256'].items():
            assert sha(outputs / name / filename) == digest, (name, filename, 'completed file changed')
        return True

    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text())
        assert state['contract_sha256'] == CONTRACT_SHA and state['worker_sha256'] == worker_sha
        validate_residual(lane, state['remaining'], complete)

    def execute(command, path, name, phase):
        if path.exists():
            k = 1
            while path.with_name(f'{path.stem}.attempt{k}{path.suffix}').exists():
                k += 1
            path = path.with_name(f'{path.stem}.attempt{k}{path.suffix}')
        with path.open('x') as log:
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 cwd=str(args.source), env=env, creationflags=FLAGS)
            if phase == 'train':
                (outputs / name / 'pid').write_text(str(p.pid))
            event('started', run=name, phase=phase, pid=p.pid, command=command)
            while p.poll() is None:
                d = outputs / name
                if phase == 'train' and (d / 'latest.pt').is_file() and not (d / 'first_checkpoint_receipt.json').exists():
                    import torch
                    raw = (d / 'latest.pt').read_bytes()
                    ck = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
                    r = plans[name]['run']
                    assert ck['cfg'] == r['cfg'] and ck['protocol']['seed'] == r['seed']
                    assert ck['protocol']['manifest'][0]['sha256'] == contract['manifest']['sha256']
                    proof = {'epoch': ck['epoch'], 'latest_sha256': hashlib.sha256(raw).hexdigest(),
                             'keys': sorted(ck), 'provenance': provenance(name),
                             'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    (d / 'first_checkpoint_receipt.json').write_text(json.dumps(proof, indent=2))
                    event('first_durable_checkpoint', run=name, epoch=ck['epoch'], latest_sha256=proof['latest_sha256'])
                    del ck, raw
                time.sleep(3)
            code = p.wait()
        event('exited', run=name, phase=phase, pid=p.pid, returncode=code)
        if code:
            raise RuntimeError((name, phase, code))

    def run_one(name):
        plan, d = plans[name], outputs / name
        r = plan['run']
        prov_path = control / f'{name}.provenance.json'
        if d.exists():
            assert args.resume and json.loads(prov_path.read_text()) == provenance(name)
            pidfile = d / 'pid'
            assert not (pidfile.exists() and Path(f'/proc/{pidfile.read_text().strip()}').exists()), 'Trainer PID alive'
        else:
            d.mkdir(parents=True)
            with prov_path.open('x') as f:
                f.write(json.dumps(provenance(name), indent=2))
        mem = cmd_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
        pids = cmd_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
        assert int(mem) < 500 and not pids, ('GPU owned by another task', mem, pids)
        if not (d / 'result.json').exists():
            command = [py, '-u', '-B', '-m', 'temporac_cvpr.train', '--manifest', *r['manifest'], '--features', r['features'],
                       '--out', str(d), '--seed', str(r['seed']), '--cfg', json.dumps(r['cfg']),
                       '--train-split', r['train_split'], '--val-split', r['val_split'], '--test-split', r['test_split']]
            if (d / 'latest.pt').exists():
                assert args.resume
                command.append('--resume')
            execute(command, d / 'stdout.log', name, 'train')
        logs = [json.loads(s) for s in (d / 'log.jsonl').read_text().splitlines() if s.strip()]
        assert [x['epoch'] for x in logs] == list(range(r['cfg']['epochs']))
        assert all(x.get('nonfinite_steps', 0) == 0 for x in logs)
        assert all(math.isfinite(v) for x in logs for k, v in x.items() if k.startswith('l_'))
        got = json.loads((d / 'config.json').read_text())
        assert got['cfg'] == r['cfg'] and got['args']['seed'] == r['seed'] and not got['args']['with_test']
        for key in ['manifest', 'features', 'train_split', 'val_split', 'test_split']:
            assert got['args'][key] == r[key]
        assert (d / 'latest.pt').is_file() and (d / 'first_checkpoint_receipt.json').is_file()
        files = ['config.json', 'log.jsonl', 'result.json']
        for job in plan['evaluations']:
            tag, ckpt = job['tag'], job['checkpoint']
            assert (d / ckpt).is_file()
            if not (d / f'eval_{tag}.json').exists():
                command = [py, '-u', '-B', '-m', 'temporac_cvpr.eval_saved', '--run', str(d), '--manifest', *r['manifest'],
                           '--features', job['features'], '--split', job['split'], '--tag', tag, '--ckpt', ckpt]
                execute(command, d / f'eval_{tag}.log', name, 'eval_' + tag)
            ev = json.loads((d / f'eval_{tag}.json').read_text())
            assert ev['n'] == job['n'] and ev['metrics_tta']['n_pos'] == job['n_positive']
            assert set(ev['predictions']) == set(ev['predictions_shift']) == expected[job['split']]
            assert all(math.isfinite(v) for v in [*ev['predictions'].values(), *ev['predictions_shift'].values()])
            assert ev['ckpt'] == ckpt
            files += [ckpt, f'eval_{tag}.json']
        receipt = {'provenance': provenance(name), 'epochs': len(logs),
                   'file_sha256': {f: sha(d / f) for f in set(files)},
                   'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        assert not (d / 'complete_known_source.json').exists()
        write_json(d / 'complete_known_source.json', receipt)
        event('run_complete', run=name, epochs=len(logs), populations={j['tag']: j['n'] for j in plan['evaluations']})

    run_lane(lane, complete, run_one, hold_path.exists, event)

if __name__ == '__main__':
    main()
