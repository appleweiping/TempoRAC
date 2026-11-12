"""Finite known-source-v1 continuation. Server only; root authorizes each worker launch.

GPU0: --gpu 0 --runs full_s0 full_s2
GPU1: --gpu 1 --runs full_s1 (root launches only after successful full RepNet run).
Required: --source <immutable clean Git snapshot> --commit <frozen SHA>.
No scheduler, retry, protocol change, historical overwrite, or release of other24 runs.
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

R = Path('/root/autodl-tmp/TempoRAC')
OLD = R / 'resume20261111analysis/known_source_v1'
C = R / 'resume20261112/continuation'
GROUP = 'frozen_known_source_v1_resume20261112'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, choices=[0, 1], required=True)
ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--commit', required=True)
ap.add_argument('--resume', action='store_true', help='explicitly continue existing durable latest.pt only')
args = ap.parse_args()
assert R.resolve() == R and args.source.resolve().is_relative_to(R)
assert args.runs == ({0: ['full_s0', 'full_s2'], 1: ['full_s1']}[args.gpu])
assert subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True,
                               creationflags=FLAGS).strip() == args.commit
assert not subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain'], text=True,
                                   creationflags=FLAGS).strip(), 'Source snapshot must be clean'
assert (args.source / 'temporac_cvpr/train.py').is_file()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pf = json.loads((OLD / 'preflight.json').read_text())
spec = json.loads((OLD / 'frozen_27_spec.json').read_text())
assert sha(Path(pf['derived_manifest'])) == pf['derived_sha256'] == '96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2'
assert sha(OLD / 'frozen_27_spec.json') == pf['spec_sha256']
assert set(args.runs) <= set(spec['initial_release'])
allruns = {r['name']: r for r in spec['runs']}
C.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(args.source),
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONDONTWRITEBYTECODE='1',
           TMPDIR=str(R / 'tmp'))
py = str(R / 'env/bin/python')


def event(status, **extra):
    row = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'worker_pid': os.getpid(),
           'gpu': args.gpu, 'status': status, **extra}
    with (C / f'worker_gpu{args.gpu}.events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def execute(cmd, log, name, phase, d):
    # Each invocation has its own log; a manual resume does not erase the first attempt.
    if log.exists():
        attempt = 1
        while log.with_name(f'{log.stem}.attempt{attempt}{log.suffix}').exists():
            attempt += 1
        log = log.with_name(f'{log.stem}.attempt{attempt}{log.suffix}')
    with log.open('x') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             cwd=str(args.source), env=env, creationflags=FLAGS)
        if phase == 'train':
            (d / 'pid').write_text(str(p.pid))
        event('started', run=name, phase=phase, pid=p.pid, command=cmd, log=str(log))
        while p.poll() is None:
            latest = d / 'latest.pt'
            first = d / 'first_checkpoint_receipt.json'
            if phase == 'train' and latest.is_file() and not first.exists():
                raw = latest.read_bytes()  # atomic file replacement keeps this observed generation intact
                ck = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
                assert ck['epoch'] >= 0 and ck['cfg']['epochs'] == 50
                assert ck['protocol']['manifest'][0]['sha256'] == pf['derived_sha256']
                assert all(k in ck for k in ['model', 'optimizer', 'scheduler', 'scaler', 'rng', 'bests'])
                proof = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         'epoch': ck['epoch'], 'latest_sha256': hashlib.sha256(raw).hexdigest(),
                         'keys': sorted(ck), 'manifest_sha256': pf['derived_sha256'], 'source_commit': args.commit,
                         'cuda_rng_devices': len(ck['rng']['cuda']), 'best_targets': sorted(ck['bests'])}
                first.write_text(json.dumps(proof, indent=2))
                event('first_durable_checkpoint', run=name, **proof)
                del raw, ck
            time.sleep(3)
        code = p.wait()
    event('exited', run=name, phase=phase, pid=p.pid, returncode=code)
    if code:
        raise RuntimeError((name, phase, code))  # no automatic retry or successor after failure


for name in args.runs:
    r = allruns[name]
    assert r['cfg']['epochs'] == 50
    d = R / 'runs' / GROUP / name
    provenance = {'run': name, 'source_commit': args.commit, 'source': str(args.source),
                  'train_sha256': sha(args.source / 'temporac_cvpr/train.py'),
                  'spec_sha256': pf['spec_sha256'], 'manifest_sha256': pf['derived_sha256'],
                  'cfg': r['cfg'], 'seed': r['seed']}
    if d.exists():
        assert args.resume, ('Existing output; explicit --resume required', str(d))
        assert json.loads((C / f'{name}.provenance.json').read_text()) == provenance
        pidfile = d / 'pid'
        assert not (pidfile.exists() and Path(f'/proc/{pidfile.read_text().strip()}').exists()), 'Trainer PID still alive'
        if (d / 'complete_known_source.json').exists():
            event('skip_completed', run=name)
            continue
    else:
        d.mkdir(parents=True)
        with (C / f'{name}.provenance.json').open('x') as f:
            f.write(json.dumps(provenance, indent=2))
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used',
                                   '--format=csv,noheader,nounits'], text=True, creationflags=FLAGS).strip()
    assert int(gpu) < 500, ('GPU acquired by another task', args.gpu, gpu)
    compute_pids = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                            '--format=csv,noheader,nounits'], text=True, creationflags=FLAGS).strip()
    assert not compute_pids, ('Another CUDA process owns this GPU', args.gpu, compute_pids)
    if not (d / 'result.json').exists():
        cmd = [py, '-u', '-B', '-m', 'temporac_cvpr.train', '--manifest', *r['manifest'],
               '--features', r['features'], '--out', str(d), '--seed', str(r['seed']), '--cfg', json.dumps(r['cfg']),
               '--train-split', r['train_split'], '--val-split', r['val_split'], '--test-split', r['test_split']]
        if (d / 'latest.pt').exists():
            assert args.resume
            cmd.append('--resume')
        execute(cmd, d / 'stdout.log', name, 'train', d)
    logs = [json.loads(s) for s in (d / 'log.jsonl').read_text().splitlines() if s.strip()]
    assert [x['epoch'] for x in logs] == list(range(50))
    assert all(x.get('nonfinite_steps', 0) == 0 for x in logs)
    assert all(math.isfinite(v) for x in logs for k, v in x.items() if k.startswith('l_'))
    got = json.loads((d / 'config.json').read_text())
    assert got['cfg'] == r['cfg'] and got['args']['seed'] == r['seed'] and not got['args']['with_test']
    assert (d / 'latest.pt').is_file() and (d / 'first_checkpoint_receipt.json').is_file()
    for val, tag in [('rc_valid', 'rc_test'), ('cx_val', 'cx_test'), ('ucf_dev', 'ucf_test'), ('rc_valid', 'quva')]:
        assert (d / f'best_{val}.pt').is_file()
        if (d / f'eval_{tag}.json').exists():
            continue  # reuse completed saved-checkpoint evaluation; population checked below
        command = [py, '-u', '-B', '-m', 'temporac_cvpr.eval_saved', '--run', str(d), '--manifest',
                   *(r['manifest'] if tag != 'quva' else [str(R / 'data/QUVA/manifest.json')]),
                   '--features', r['features'] if tag != 'quva' else str(R / 'features/QUVA_stmae_b8_p4'),
                   '--split', tag if tag != 'quva' else 'test', '--tag', tag, '--ckpt', f'best_{val}.pt']
        if tag == 'quva':
            command += ['--aux', str(R / 'features/QUVA_stmae_p4'), '--domain', 'RepCountA']
        execute(command, d / f'eval_{tag}.log', name, 'eval_' + tag, d)
    expect = {'rc_test': 152, 'cx_test': 2403, 'ucf_test': 105, 'quva': 100}
    for tag, n in expect.items():
        ev = json.loads((d / f'eval_{tag}.json').read_text())
        assert ev['n'] == n and len(ev['predictions']) == n and len(ev['predictions_shift']) == n
        assert ev['predictions'].keys() == ev['predictions_shift'].keys()
        assert all(math.isfinite(v) for v in [*ev['predictions'].values(), *ev['predictions_shift'].values()])
    receipt = {**provenance, 'epochs': len(logs), 'gpu': args.gpu, 'eval_populations': expect,
               'checkpoints': {p.name: sha(p) for p in d.glob('best*.pt')},
               'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (d / 'complete_known_source.json').write_text(json.dumps(receipt, indent=2))
    event('run_complete', run=name, receipt=receipt)
event('queue_complete', runs=args.runs)
