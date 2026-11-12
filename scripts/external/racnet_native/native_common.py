"""Persistence and identity plumbing; no RACnet mathematics live here."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

ROOT = Path('/root/autodl-tmp/TempoRAC')
CONTRACT_SHA = '77c3d84443fb57415db3b2ed7416b3b16af5f8e8af614c41a05812802e7f8de8'
PACKET = ROOT / 'resume20261112/live-continuation/racnet-contract-astra'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def syncdir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        syncdir(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def savejson(path, value):
    atomic(path, lambda f: f.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()))


def journal(path, value):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def contract():
    path = PACKET / 'RUN_CONTRACT.json'
    assert sha(path) == CONTRACT_SHA, 'frozen contract changed'
    return load(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def science(work):
    src = Path(work) / 'vendor'
    return (module('racnet_model', src / 'model.py'),
            module('racnet_gt', src / 'dataset/gen_gt.py'),
            module('racnet_counts', src / 'dataset/cal_count.py'))


def identity(work):
    work = Path(work)
    return {'contract': CONTRACT_SHA,
            'code': {p.name: sha(p) for p in sorted(work.glob('*.py'))},
            'env_spec': sha(work / 'ENV_SPEC.json'),
            'runtime': sha(work / 'RUNTIME.json') if (work / 'RUNTIME.json').exists() else None,
            'science': {str(p.relative_to(work)): sha(p) for p in sorted((work / 'vendor').rglob('*.py'))}}


def seed(value):
    import numpy as np
    import torch
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    random.seed(value)


def rng_state(cuda=True):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if cuda else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def runtime_flags():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return {'matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
            'cudnn_benchmark': False, 'cudnn_deterministic': False, 'deterministic_algorithms': False}


def gpu_gate(work, receipt, mode):
    """Only explicit root ownership receipts permit CUDA initialization."""
    import shutil
    import subprocess
    work = Path(work)
    r = load(receipt)
    assert r['released_by'] == 'root' and r['mode'] == mode
    assert r['contract_sha256'] == CONTRACT_SHA and r['candidate_identity'] == identity(work)
    assert r['authorized_gpu_count'] == 1
    prior = r['current_arm_complete_receipt']
    assert sha(prior['path']) == prior['sha256'], 'prior full-arm completion receipt missing/stale'
    assert load(prior['path'])['status'] in ('COMPLETE', 'complete', 'completed', 'held_after_complete')
    assert r['gates']['source_runtime'] == r['gates']['CPU_interfaces'] == r['gates']['lane'] == r['gates']['resources'] == 'PASS'
    if mode != 'extract-quva':
        assert r['gates']['all1243_inputs'] == 'PASS'
    pub = r['immutable_publication_receipt']
    assert sha(pub['path']) == pub['sha256'] and load(pub['path'])['candidate_identity'] == identity(work)
    assert r['fresh_review_status'] in ('PASS', 'PROVISIONAL_PASS')
    review = r['fresh_review_receipt']
    assert sha(review['path']) == review['sha256']
    reviewed = load(review['path'])
    assert reviewed['candidate_identity'] == identity(work) and reviewed['verdict'] in ('PASS', 'PROVISIONAL_PASS')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible == str(r['gpu_index']) and ',' not in visible, 'exactly one released lane required'
    assert time.time() < r['expires_epoch'], 'ownership receipt expired'
    # This query is read-only and happens only at a future explicit formal launch.
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                       check=True, capture_output=True, text=True, creationflags=flags)
    assert not any(line.split(',')[0].strip() == r['gpu_uuid'] for line in q.stdout.splitlines()), 'lane occupied'
    free = shutil.disk_usage(ROOT).free
    assert free - int(r['pending_author_bytes']) - int(r['additional_bytes_reserved']) >= 137438953472
    assert int(r['additional_bytes_reserved']) <= 34359738368
    storage_guard(work)
    return r


def storage_guard(work):
    """The additional32GiB cap and free-volume reserve are never score dependent."""
    import shutil
    used = sum(p.stat().st_size for p in Path(work).rglob('*') if p.is_file())
    assert used <= 34359738368, ('candidate32GiB cap exceeded', used)
    assert shutil.disk_usage(ROOT).free >= 137438953472, 'shared128GiB free reserve'
    return used


def timing_summary(rows, elapsed_seconds, reused_keys, total_field, inference_field=None):
    """Retained production durations and this invocation's wall time are distinct."""
    keys = [r['key'] for r in rows]
    reused_keys = set(reused_keys)
    assert len(keys) == len(set(keys)) and reused_keys <= set(keys)
    assert math.isfinite(elapsed_seconds) and elapsed_seconds >= 0
    result = {'retained_output_count': len(rows), 'reused_output_count_current_invocation': len(reused_keys),
              'generated_output_count_current_invocation': len(rows)-len(reused_keys),
              'current_invocation_elapsed_seconds': elapsed_seconds,
              'current_invocation_scope': 'function entry through final summary creation; final summary file writes excluded',
              'retained_duration_scope': 'sum of each completed output original production duration across invocations; excludes abandoned partial work'}
    fields = [('total', total_field)] + ([('inference', inference_field)] if inference_field else [])
    for label, field in fields:
        for row in rows:
            assert math.isfinite(row[field]) and row[field] >= 0
        result['retained_output_%s_seconds_sum' % label] = sum(r[field] for r in rows)
        result['reused_output_%s_seconds_sum' % label] = sum(r[field] for r in rows if r['key'] in reused_keys)
        result['generated_output_%s_seconds_sum' % label] = sum(r[field] for r in rows if r['key'] not in reused_keys)
    return result
