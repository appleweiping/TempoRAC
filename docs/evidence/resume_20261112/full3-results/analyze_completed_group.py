"""Extend saved-artifact/seed summaries for the frozen corrected three-run group.

Server only. Reuses immutable temporac_cvpr.metrics.count_metrics; no model import,
inference, selection, bootstrap, training or shared state mutation.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
root, out = args.root.resolve(), args.out.resolve()
assert root == Path('/root/autodl-tmp/TempoRAC') and out.is_relative_to(root)
assert not out.exists(), 'Preserve any previous group closure; do not overwrite'
TRAIN_SHA = '489dc77b91a21d21215784ff49961edb20dab4fb'
PUB_SHA = '49b30a320916fe9f2913f5671f9b7d38734859db'
source = root / 'source_snapshots' / TRAIN_SHA
sys.path.insert(0, str(source))
from temporac_cvpr.metrics import count_metrics

control = root / 'resume20261112/continuation'
group = root / 'runs/frozen_known_source_v1_resume20261112'
old = root / 'resume20261111analysis/known_source_v1'
flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
def command(cmd):
    return subprocess.check_output(cmd, text=True, creationflags=flags).strip()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()
provenance, small_files = {}, {}
def record(path, archive_name=None):
    path = Path(path)
    digest = sha(path)
    provenance[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest}
    if archive_name:
        assert path.suffix != '.pt'
        small_files[archive_name] = path
    return digest
def read(path, archive_name=None):
    record(path, archive_name)
    return json.loads(path.read_text())
def finite_json(values):
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in values.items()}
def aggregate(scores):
    result = {}
    for metric in scores[0]:
        values = [s[metric] for s in scores if s[metric] is not None]
        if metric in {'n', 'n_pos'}:
            assert len(set(values)) == 1
            result[metric] = values[0]
        else:
            result[metric] = {'mean': float(np.mean(values)) if values else None,
                              'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                              'n_seeds': len(values)}
    return result

assert command(['git', '-C', str(source), 'rev-parse', 'HEAD']) == TRAIN_SHA
assert not command(['git', '-C', str(source), 'status', '--porcelain'])
publication = root / 'source_snapshots' / PUB_SHA
assert command(['git', '-C', str(publication), 'rev-parse', 'HEAD']) == PUB_SHA
assert not command(['git', '-C', str(publication), 'status', '--porcelain'])
assert record(source / 'temporac_cvpr/train.py') == '33f8ed0e56142c40f8b37421cb45fd3282cda8aa7959e5fa0b2d793950bb61b4'
record(source / 'temporac_cvpr/metrics.py', 'source/metrics.py')
assert record(control / 'continuation_worker.py', 'source/continuation_worker.py') == 'e8844834850bfdb749d6240d82c2541e9c82d796891fe84dfebba6d98fd009d4'
sync = read(root / 'source_sync.json', 'inputs/global_source_sync_observed.json')
assert sync['commit'] == sync['github_head_verified'] == PUB_SHA
assert sync['active_training_commit'] == TRAIN_SHA
spec = read(old / 'frozen_27_spec.json', 'inputs/frozen_27_spec.json')
assert record(old / 'frozen_27_spec.json') == '5e6750b0ff6a894be85c171e5ae31c150e9cb5288a3cb9ab4df8f0f5984babe4'
assert set(spec['initial_release']) == {'full_s0', 'full_s1', 'full_s2'}
byrun = {r['name']: r for r in spec['runs']}
manifest_path = root / 'data/UNION_known_source_v1/manifest_dev.json'
manifest = read(manifest_path, 'inputs/corrected_manifest_dev.json')
assert record(manifest_path) == '96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2'
original_path = root / 'data/UNION/manifest_dev.json'
original = read(original_path)
assert record(original_path) == 'bd171452197b74435e6fecfd19332f3e0e7c428e48b1f3a8aeb2b77c35894a13'
held_out = {'rc_valid', 'cx_val', 'ucf_dev', 'rc_test', 'cx_test', 'ucf_test'}
assert [x for x in manifest if x['split'] in held_out] == [x for x in original if x['split'] in held_out]
quva_path = root / 'data/QUVA/manifest.json'
quva = read(quva_path, 'inputs/quva_manifest.json')
assert record(quva_path) == 'bcf2d2964ab2d5ec36c8f6624b856f719a5de3d910294a436c165c36d80120a1'
train_counts = Counter(x['split'] for x in manifest if x['split'] in {'rc_train', 'cx_train', 'ucf_train'})
assert train_counts == {'rc_train': 756, 'cx_train': 4222, 'ucf_train': 331}
assert train_counts['rc_train'] * 4 + train_counts['ucf_train'] * 4 + train_counts['cx_train'] == 8570
workers = {}
for gpu, expected_runs in [(0, ['full_s0', 'full_s2']), (1, ['full_s1'])]:
    launch = read(control / f'launch_gpu{gpu}.json', f'control/launch_gpu{gpu}.json')
    event_path = control / f'worker_gpu{gpu}.events.jsonl'
    record(event_path, f'control/worker_gpu{gpu}.events.jsonl')
    events = [json.loads(s) for s in event_path.read_text().splitlines() if s.strip()]
    assert events[-1]['status'] == 'queue_complete' and events[-1]['runs'] == expected_runs
    assert not Path(f"/proc/{launch['worker_pid']}").exists()
    assert all(e['returncode'] == 0 for e in events if e['status'] == 'exited')
    workers[str(gpu)] = {'pid': launch['worker_pid'], 'absent': True,
                         'queue_complete_utc': events[-1]['utc'], 'events': events}

benchmarks = {'rc_test': ('RepCount-A', 'rc_valid', 152), 'cx_test': ('Countix', 'cx_val', 2403),
              'ucf_test': ('UCFRep', 'ucf_dev', 105), 'quva': ('QUVA', 'rc_valid', 100)}
seed_results, per_video, closure = {}, [], {}
for seed in [0, 1, 2]:
    name = f'full_s{seed}'
    d, r = group / name, byrun[name]
    completion = read(d / 'complete_known_source.json', f'{name}/complete_known_source.json')
    cfg = read(d / 'config.json', f'{name}/config.json')
    run_provenance = read(control / f'{name}.provenance.json', f'{name}/run_provenance.json')
    assert cfg['cfg'] == r['cfg'] == run_provenance['cfg']
    assert cfg['args']['seed'] == r['seed'] == run_provenance['seed'] == seed
    assert cfg['cfg']['epochs'] == completion['epochs'] == 50
    for field in ['manifest', 'features', 'train_split', 'val_split', 'test_split']:
        assert cfg['args'][field] == r[field], (name, field)
    assert not cfg['args']['with_test']
    assert completion['source_commit'] == run_provenance['source_commit'] == TRAIN_SHA
    assert completion['manifest_sha256'] == run_provenance['manifest_sha256'] == sync['data_manifest_sha256']
    assert completion['eval_populations'] == {'rc_test': 152, 'cx_test': 2403, 'ucf_test': 105, 'quva': 100}
    log_path = d / 'log.jsonl'
    record(log_path, f'{name}/log.jsonl')
    logs = [json.loads(s) for s in log_path.read_text().splitlines() if s.strip()]
    assert [x['epoch'] for x in logs] == list(range(50))
    assert all(x.get('nonfinite_steps', 0) == 0 for x in logs)
    assert all(math.isfinite(v) for x in logs for k, v in x.items() if k.startswith('l_'))
    pid = int((d / 'pid').read_text())
    assert not Path(f'/proc/{pid}').exists()
    assert (d / 'latest.pt').is_file()
    for filename, digest in completion['checkpoints'].items():
        assert record(d / filename) == digest
    for pattern in ['result*.json', 'predictions*.json', 'eval_*.log', 'stdout.log', 'first_checkpoint_receipt.json']:
        for p in d.glob(pattern):
            record(p, f'{name}/{p.name}')
    record(d / 'pid', f'{name}/pid')
    for target in ['rc_valid', 'cx_val', 'ucf_dev']:
        selected = min(logs, key=lambda x: (x['val_all'][target]['mae_p'], -x['val_all'][target]['obo_p']))['epoch']
        assert json.loads((d / f'result_{target}.json').read_text())['best_epoch'] == selected
    closure[name] = {'seed': seed, 'epochs': 50, 'finite_losses': True, 'trainer_pid': pid, 'trainer_absent': True,
                     'latest_exists': True, 'latest_bytes': (d / 'latest.pt').stat().st_size,
                     'completed_utc': completion['completed_utc'], 'validation_selection_checked': True,
                     'training_logged_seconds': float(sum(x['time'] for x in logs)), 'completion_receipt': completion}
    seed_results[name] = {}
    for tag, (label, target, n) in benchmarks.items():
        ev = read(d / f'eval_{tag}.json', f'{name}/eval_{tag}.json')
        rows = [x for x in (quva if tag == 'quva' else manifest) if x['split'] == ('test' if tag == 'quva' else tag)]
        bykey = {f"{x['split']}__{x['video_id']}": x for x in rows}
        keys = sorted(bykey)
        assert len(rows) == len(keys) == ev['n'] == n
        assert sorted(ev['predictions']) == sorted(ev['predictions_shift']) == keys
        assert ev['ckpt'] == f'best_{target}.pt'
        assert ev['best_epoch'] == json.loads((d / f'result_{target}.json').read_text())['best_epoch']
        gt = np.array([bykey[k]['count'] for k in keys], dtype=np.float64)
        assert np.isfinite(gt).all() and (gt >= 0).all()
        orig = np.array([ev['predictions'][k] for k in keys], dtype=np.float64)
        shift = np.array([ev['predictions_shift'][k] for k in keys], dtype=np.float64)
        views = {'original': orig, 'shift': shift, 'two_view_mean': (orig + shift) * .5}
        view_results = {}
        for view, pred in views.items():
            assert np.isfinite(pred).all()
            metrics = count_metrics(pred, gt)
            stored = ev[{'original': 'metrics', 'shift': 'metrics_shift', 'two_view_mean': 'metrics_tta'}[view]]
            assert stored.keys() == metrics.keys()
            assert all(abs(metrics[k] - stored[k]) < 1e-10 for k in metrics), (name, tag, view, 'stored metric mismatch')
            zero = gt == 0
            zero_rows = [{'key': k, 'gt': 0.0, 'prediction_continuous': float(p), 'prediction_rounded': float(np.round(p)),
                          'absolute_error': float(abs(np.round(p)))} for k, p, z in zip(keys, pred, zero) if z]
            zero_metrics = finite_json(count_metrics(pred[zero], gt[zero])) if zero.any() else None
            view_results[view] = {'metrics': finite_json(metrics), 'zero_count_rows': zero_rows, 'zero_metrics': zero_metrics}
        for i, key in enumerate(keys):
            e = float(abs(np.round(views['two_view_mean'][i]) - gt[i]))
            per_video.append({'run': name, 'seed': seed, 'benchmark': tag, 'key': key,
                              'video_id': bykey[key]['video_id'], 'gt': float(gt[i]),
                              'original': float(orig[i]), 'shift': float(shift[i]),
                              'two_view_mean': float(views['two_view_mean'][i]), 'two_view_rounded': float(np.round(views['two_view_mean'][i])),
                              'absolute_error': e, 'positive_count_nmae': e / (float(gt[i]) + .1) if gt[i] > 0 else None,
                              'zero_count': bool(gt[i] == 0), 'known_ambiguous_label': tag == 'rc_test' and bykey[key]['video_id'] in {'stu4_5', 'stu4_27'}})
        seed_results[name][tag] = {'dataset': label, 'best_epoch': ev['best_epoch'], 'checkpoint': ev['ckpt'], 'views': view_results}

summary = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'scope': 'fixed corrected-training seeds0,1,2; complete saved original evaluations; group closure, not scientific acceptance',
           'evaluation_type': 'real_gt; inherited dataset labels with documented conflicts',
           'training_commit': TRAIN_SHA, 'publication_commit_observed': PUB_SHA, 'manifest_sha256': record(manifest_path),
           'primary': 'positive-count abs(np.round(pred)-gt)/(gt+0.1), lower better; fixed mean of original and shift counts',
           'companion': 'positive-count OBO: abs(np.round(pred)-gt)<=1',
           'aggregation': 'unweighted mean of three per-seed metrics; sample SD with ddof=1; no seed ensemble or significance claim',
           'seeds': [0, 1, 2], 'seed_results': seed_results, 'benchmarks': {}, 'run_closure': closure,
           'global_sync_observation': {'sha256': record(root / 'source_sync.json'), 'actual_publication_head_clean': PUB_SHA,
               'actual_training_head_clean': TRAIN_SHA, 'reported_scientific_group_complete': sync['scientific_group_complete'],
               'reported_completed_corrected_seeds': sync['completed_corrected_seeds'],
               'stale_fields': ['scientific_group_complete', 'completed_corrected_seeds', 'active_scope'], 'shared_file_modified': False},
           'limitations': ['Historical test-informed method selection persists.',
               'Known-source-disjoint training does not resolve inherited evaluation-label conflicts or unknown overlap.',
               'All already-run seeds, views, domains, zero-count and disputed records retained.',
               'Historical contaminated-protocol runs are not pooled with this group.',
               'No ablation contribution, external-method fairness, SOTA, six-domain coverage or final-paper acceptance follows from closure.']}
for tag, (label, _, n) in benchmarks.items():
    summary['benchmarks'][tag] = {'dataset': label, 'n_all': n, 'views': {}}
    for view in ['original', 'shift', 'two_view_mean']:
        scores = [seed_results[f'full_s{i}'][tag]['views'][view]['metrics'] for i in [0, 1, 2]]
        zs = [seed_results[f'full_s{i}'][tag]['views'][view]['zero_metrics'] for i in [0, 1, 2]]
        summary['benchmarks'][tag]['views'][view] = {'aggregate': aggregate(scores),
            'zero_aggregate': aggregate(zs) if all(z is not None for z in zs) else None}
out.mkdir(parents=True)
record(Path(__file__), 'source/analyze_completed_group.py')
(out / 'group_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
(out / 'raw_prediction_provenance.json').write_text(json.dumps(provenance, indent=2))
with (out / 'per_video_all_seeds.jsonl').open('w') as f:
    for row in per_video:
        f.write(json.dumps(row, allow_nan=False) + '\n')
lines = ['# Frozen corrected-training group closure', '',
         'All three predeclared seeds completed50 epochs and fixed original-population evaluations. No model execution was performed for this report.', '',
         '| Dataset | All/positive N | Primary NMAE mean ± sample SD | Positive OBO mean ± sample SD |',
         '|---|---:|---:|---:|']
for tag, bench in summary['benchmarks'].items():
    a = bench['views']['two_view_mean']['aggregate']
    lines.append(f"| {bench['dataset']} | {a['n']}/{a['n_pos']} | {a['mae_p']['mean']:.9f} ± {a['mae_p']['sample_sd']:.9f} | {a['obo_p']['mean']:.9f} ± {a['obo_p']['sample_sd']:.9f} |")
lines += ['', 'The primary inference is fixed per-seed mean of original and shifted counts, followed by NumPy rounding. SD uses ddof=1 across the three fitted seeds. No ensemble, significance or SOTA claim.', '',
          'Full seed-level/all-population/zero-count/original/shift/fixed-mean metrics and selected epochs are in group_summary.json. Every saved prediction and negative/disputed row remains in the evidence and per_video_all_seeds.jsonl.', '',
          'Shared source_sync.json still records completedseeds[0] and group_complete=false from08:57UTC. Actual publication49b30a3 and training489dc77 snapshots are clean; this closure records the newer actual completion without modifying shared state.', '',
          'Inherited test-informed method choice, evaluation-label conflicts, missing corrected ablations/external comparison/coverage remain unresolved. This is execution/statistical closure only.']
(out / 'GROUP_REPORT.md').write_text('\n'.join(lines) + '\n')
artifact_hashes = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.iterdir() if p.is_file()}
(out / 'ARTIFACT_SHA256.json').write_text(json.dumps(artifact_hashes, indent=2))
with zipfile.ZipFile(out / 'completed_group_evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for name, path in small_files.items():
        z.write(path, name)
    for p in out.iterdir():
        if p.is_file() and p.suffix != '.zip':
            z.write(p, 'analysis/' + p.name)
archive = out / 'completed_group_evidence.zip'
receipt = {'completed_utc': summary['created_utc'], 'out': str(out), 'n_seeds': 3,
           'primary': {tag: bench['views']['two_view_mean']['aggregate'] for tag, bench in summary['benchmarks'].items()},
           'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive), 'raw_small_files': len(small_files),
           'per_video_rows': len(per_video), 'workers_absent': True,
           'gpu_compute_apps': command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits']),
           'shared_state_modified': False, 'new_model_execution': False}
(out / 'COMPLETION_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
