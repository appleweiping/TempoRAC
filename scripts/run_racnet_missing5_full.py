"""Root-invoked finite105 -> receipt refresh -> train200 -> evaluate357.
No GPU waiter, scheduler, scientific budget extension or automatic retry.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path('/root/autodl-tmp/TempoRAC')
WORK = ROOT / 'baselines/racnet_native_v1'
SEQUENCE = ['extract-quva', 'refresh-inputs', 'train', 'evaluate']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with open(temp, 'x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def metadata_module():
    spec = importlib.util.spec_from_file_location('racnet_finite_metadata', WORK / 'native_common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib only; never constructs/imports a model.
    return module


def training_identity(work, common):
    ready = load(common.input_path(work, 'READINESS'))
    return {'candidate': common.identity(work), 'readiness': sha(common.input_path(work, 'READINESS')),
            'input_hashes': ready['hashes']}


def feature_complete(work, common):
    path = work / 'FEATURES105_COMPLETE.json'
    if not path.exists():
        return False
    value = load(path)
    assert value['status'] == 'PASS' and value['frames'] == 51349 and len(value['videos']) == 105
    assert value['binding']['candidate_identity'] == common.identity(work)
    assert value['binding']['inputs_sha256'] == sha(work / 'inputs/raw_feature_inputs105.json')
    raw = {r['key']: r for r in load(work / 'inputs/raw_feature_inputs105.json')}
    assert {r['key'] for r in value['videos']} == set(raw)
    assert sum(r['T'] for r in value['videos']) == 51349
    for row in value['videos']:
        dest = Path(raw[row['key']]['path'])
        assert row['binding'] == value['binding'] and row['T'] == raw[row['key']]['input_T']
        assert dest.is_file() and dest.stat().st_size == row['bytes']
    return True


def inputs_complete(work, common):
    ready_path, cpu_path = common.input_path(work, 'READINESS'), common.cpu_check_path(work)
    if not ready_path.exists() or not cpu_path.exists():
        return False
    ready, cpu = load(ready_path), load(cpu_path)
    if ready['status'] != 'PASS' or cpu['status'] != 'PASS':
        return False
    assert ready['required'] == 1243 and not ready['missing']
    assert ready['author_members_available'] == 1138 and ready['recovered_members_available'] == 5 and ready['quva_features_available'] == 100
    assert cpu['candidate_identity'] == common.identity(work) and cpu['readiness_sha256'] == sha(ready_path)
    return True


def train_complete(work, common):
    path = work / 'TRAINING_COMPLETE.json'
    if not path.exists():
        return False
    complete, commit = load(path), load(work / 'checkpoints/COMMIT.json')
    assert complete['identity'] == training_identity(work, common) == commit['identity']
    assert complete['epochs'] == 200 and complete['attempted_steps'] == 9600
    assert complete['selected'] == commit['best_mae'] and Path(complete['selected']['file']).is_file()
    return True


def train_resume(work, common):
    path = work / 'checkpoints/COMMIT.json'
    if not path.exists():
        assert not (work / 'PROGRESS.json').exists(), 'training progress without commit; preserve for operator'
        return False
    commit = load(path)
    assert commit['identity'] == training_identity(work, common), 'existing trajectory identity mismatch'
    latest = commit['latest']
    assert isinstance(latest['next_epoch'], int) and 0 <= latest['next_epoch'] <= 200
    assert Path(latest['file']).is_file() and Path(latest['file']).stat().st_size > 0
    # The unchanged train --resume entry verifies payload SHA and restores all state/RNG.
    return True


def evaluation_complete(work, common):
    path = work / 'COMPLETE.json'
    if not path.exists():
        return False
    value, training = load(path), load(work / 'TRAINING_COMPLETE.json')
    assert value['status'] == 'COMPLETE' and value['predictions'] == 357
    assert value['binding']['identity'] == training_identity(work, common)
    assert value['binding']['checkpoint_sha256'] == training['selected']['sha256']
    assert value['binding']['prediction_inputs_sha256'] == sha(common.input_path(work, 'prediction_inputs'))
    for field, name in [('prediction_sha256', 'PREDICTIONS_357.json'), ('score_sha256', 'SCORED_357.json'), ('metric_sha256', 'METRICS_357.json')]:
        assert sha(work / name) == value[field]
    return True


def validate_root_release(release, common, runner_sha):
    assert release['released_by'] == 'root' and release['authorized_sequence'] == SEQUENCE
    assert release['runner_sha256'] == runner_sha
    assert time.time() < release['start_before_epoch'], 'root startup release expired'
    assert release['candidate_identity'] == common.identity(WORK)
    assert release['contract_sha256'] == common.CONTRACT_SHA
    assert release['effective_contract_sha256'] == common.EFFECTIVE_CONTRACT_SHA
    assert release['amendment_sha256'] == common.AMENDMENT_SHA and release['correction_sha256'] == common.CORRECTION_SHA
    assert release['authorized_gpu_count'] == 1 and int(release['pending_author_bytes']) == 0
    assert all(release['gates'][k] == 'PASS' for k in ('source_runtime', 'CPU_interfaces', 'lane', 'resources'))
    for key in ('current_arm_complete_receipt', 'immutable_publication_receipt', 'fresh_review_receipt'):
        ref = release[key]
        assert sha(ref['path']) == ref['sha256']
    assert load(release['current_arm_complete_receipt']['path'])['status'] in ('COMPLETE', 'complete', 'completed', 'held_after_complete')
    publication = load(release['immutable_publication_receipt']['path'])
    review = load(release['fresh_review_receipt']['path'])
    assert publication['candidate_identity'] == review['candidate_identity'] == common.identity(WORK)
    assert publication['runner_sha256'] == review['runner_sha256'] == runner_sha
    assert release['fresh_review_status'] in ('PASS', 'PROVISIONAL_PASS') and review['verdict'] in ('PASS', 'PROVISIONAL_PASS')
    return release


def ownership(release, common, mode, folder):
    assert release['candidate_identity'] == common.identity(WORK), 'source changed during finite sequence'
    ready = inputs_complete(WORK, common)
    if mode != 'extract-quva':
        assert ready, 'all1243 feature qualification required before train/evaluate'
    value = dict(release)
    value.update(mode=mode, expires_epoch=time.time()+600,
                 pipeline_authorization='fixed four-command continuation; expiry is only per-stage startup gate',
                 current_free_bytes=__import__('shutil').disk_usage(ROOT).free,
                 current_candidate_bytes=common.storage_guard(WORK))
    value['gates'] = dict(release['gates'], all1243_inputs='PASS' if ready else 'PENDING')
    path = folder / ('OWNERSHIP_' + mode + '.json')
    save(path, value)
    return path


def run_pipeline(release_path):
    assert ROOT.is_dir() and WORK.is_dir(), 'server-only fixed project root'
    release_path = Path(release_path).resolve()
    assert release_path.is_relative_to(ROOT)
    common = metadata_module()
    release_sha, runner_sha = sha(release_path), sha(__file__)
    release = validate_root_release(load(release_path), common, runner_sha)
    folder = WORK / 'finite_pipeline' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    state = {'status': 'RUNNING', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'root_release_path': str(release_path), 'root_release_sha256': release_sha,
             'runner_sha256': runner_sha, 'candidate_identity': release['candidate_identity'], 'stages': []}
    save(folder / 'PIPELINE_STATE.json', state)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        for action in SEQUENCE:  # exactly four bounded stages; no poll/retry loop.
            assert sha(release_path) == release_sha and common.identity(WORK) == release['candidate_identity']
            done = (feature_complete(WORK, common) if action == 'extract-quva' else
                    inputs_complete(WORK, common) if action == 'refresh-inputs' else
                    train_complete(WORK, common) if action == 'train' else evaluation_complete(WORK, common))
            if done:
                state['stages'].append({'stage': action, 'status': 'REUSED_COMPLETE', 'exit_code': None})
                save(folder / 'PIPELINE_STATE.json', state)
                continue
            if action in ('refresh-inputs', 'train', 'evaluate'):
                assert feature_complete(WORK, common), 'full105 production prerequisite'
            if action in ('train', 'evaluate'):
                assert inputs_complete(WORK, common)
            if action == 'evaluate':
                assert train_complete(WORK, common)
            cmd = [str(ROOT/'envs/racnet-native310/bin/python'), '-B', str(WORK/'run.py'), action, '--work', str(WORK)]
            if action != 'refresh-inputs':
                own = ownership(release, common, action, folder)
                cmd += ['--ownership-receipt', str(own)]
            if action == 'train' and train_resume(WORK, common):
                cmd.append('--resume')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       TMPDIR=str(WORK/'tmp'), XDG_CACHE_HOME=str(WORK/'cache'), MPLCONFIGDIR=str(WORK/'cache/matplotlib'),
                       CUDA_VISIBLE_DEVICES='' if action == 'refresh-inputs' else str(release['gpu_index']))
            log = folder / (action+'.log')
            row = {'stage': action, 'status': 'RUNNING', 'argv': cmd, 'log': str(log),
                   'started_epoch': time.time(), 'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES']}
            state['stages'].append(row)
            save(folder / 'PIPELINE_STATE.json', state)
            with open(log, 'xb') as handle:
                process = subprocess.Popen(cmd, cwd=WORK, env=env, stdout=handle, stderr=subprocess.STDOUT, creationflags=flags)
                row['pid'] = process.pid
                save(folder / 'PIPELINE_STATE.json', state)
                code = process.wait()  # ordinary finite child lifecycle, never waits for a GPU.
            row.update(exit_code=code, ended_epoch=time.time(), status='COMPLETE' if code == 0 else 'FAILED')
            save(folder / 'PIPELINE_STATE.json', state)
            assert code == 0, ('stage failed; no automatic retry', action, code)
            verified = (feature_complete(WORK, common) if action == 'extract-quva' else
                        inputs_complete(WORK, common) if action == 'refresh-inputs' else
                        train_complete(WORK, common) if action == 'train' else evaluation_complete(WORK, common))
            assert verified, ('stage exited without qualified completion', action)
        state.update(status='COMPLETE', ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     final_complete_sha256=sha(WORK/'COMPLETE.json'))
        save(folder / 'PIPELINE_STATE.json', state)
        save(folder / 'PIPELINE_COMPLETE.json', state)
        print(json.dumps({'status': 'COMPLETE', 'receipt': str(folder/'PIPELINE_COMPLETE.json')}))
    except BaseException as exc:
        state.update(status='FAILED_NO_AUTOMATIC_RETRY', error_type=type(exc).__name__, error=str(exc), ended_epoch=time.time())
        save(folder / 'PIPELINE_STATE.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True, help='actual root finite-sequence release JSON')
    args = parser.parse_args()
    assert ROOT.is_dir() and WORK.is_dir(), 'server-only pre-existing project/work root'
    with open(WORK/'FINITE_PIPELINE.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_pipeline(args.release)


if __name__ == '__main__':
    main()
