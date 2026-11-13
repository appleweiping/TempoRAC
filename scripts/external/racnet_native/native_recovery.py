"""Versioned missing-author input qualification and full105 raw-RGB production.
No GT/TSM generation occurs here. The three first TSMs and fixed parity are reused.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import os
import time
import uuid
from native_common import (ROOT, CONTRACT_SHA, EFFECTIVE_CONTRACT_SHA, AMENDMENT_SHA,
                           CORRECTION_SHA, atomic, contract, digest, gpu_gate, identity,
                           input_path, cpu_check_path, load, savejson, sha, storage_guard,
                           syncdir, timing_summary)
P = ROOT / 'resume20261112/continuation-2244/racnet'
AMENDMENT = P / 'missing5-astra/DECISION.json'
CORRECTION = P / 'missing5-astra/CORRECTION_SYMMETRY.json'
ORIGINAL_CPU_SHA = '2cc4e5cebcf7f1d68b195b1fb56c65e23564ee58d985f0848f1448f55afe43ff'
ORIGINAL_READY_SHA = '80c5a53bc79d93577d128d9e92ddd223adbf2967e48aedc974ebc8dad26d9d03'
PRED_FIELDS = ('key', 'population', 'full_N', 'input_T', 'raw_video', 'raw_video_sha256', 'slice', 'path', 'sha256')


def recovery_sources(work):
    work = Path(work)
    assert sha(AMENDMENT) == AMENDMENT_SHA and sha(CORRECTION) == CORRECTION_SHA
    amendment, correction = load(AMENDMENT), load(CORRECTION)
    assert sha(work / 'RUN_CONTRACT_MISSING5_V2.json') == EFFECTIVE_CONTRACT_SHA
    assert sha(work / 'RUN_CONTRACT_MISSING5_V1.json') == correction['parent_effective_v1_sha256']
    assert sha(work / 'CPU_CHECK.json') == ORIGINAL_CPU_SHA
    assert sha(work / 'inputs/READINESS.json') == ORIGINAL_READY_SHA
    prior = load(work / 'CPU_CHECK.json')
    assert len(prior['passed']) == 8 and prior['no_model_forward'] and not prior['cuda_initialized']
    current = identity(work)
    assert prior['candidate_identity']['contract'] == CONTRACT_SHA
    for name in ('env_spec', 'runtime', 'science'):
        assert prior['candidate_identity'][name] == current[name], ('old CPU not reusable', name)
    for name, expected in amendment['frozen_metadata'].items():
        assert sha(work / 'inputs' / name) == expected, ('frozen metadata changed', name)
    complete = load(work / 'missing5_recovery_v1/TSM_COMPLETE_V2.json')
    assert complete['status'] == 'PASS' and complete['effective_contract_sha256'] == EFFECTIVE_CONTRACT_SHA
    assert complete['correction_sha256'] == CORRECTION_SHA and complete['new_TSM_generator_calls'] == 2
    assert complete['cumulative_CPU_stage_wall_seconds'] < 600
    assert complete['parity_receipt_sha256'] == correction['inherited_parity']['receipt_sha256']
    assert sha(P / 'missing5-implementation/cpu-production/PARITY.json') == complete['parity_receipt_sha256']
    assert sha(P / 'missing5-implementation/cpu-production/CPU_RESULT.json') == correction['preserved_failure']['cpu_result_sha256']
    targets = {r['key']: r for r in load(work / 'inputs/targets.json')}
    tsms = {}
    for expected in amendment['missing3_targets']:
        key, n = expected['key'], expected['length']
        assert targets[key] == expected
        path = work / 'missing5_recovery_v1/tsm' / (key.replace('/', '__') + '.npy')
        side = Path(str(path) + '.json')
        row = load(side)
        assert row['key'] == key and row['shape'] == [n, n] and row['dtype'] == 'float64'
        assert path.is_file() and path.stat().st_size == row['bytes']
        assert row['bounds'] == expected['bounds'] and row['existing_start_end_target_sha256'] == expected['target_sha256']
        assert row['generator_sha256'] == amendment['tsm']['candidate_sha256']
        assert row['parity_receipt_sha256'] == complete['parity_receipt_sha256']
        if key == 'rc_train/stu5_3':
            accept_path = work / 'missing5_recovery_v1/STU5_3_ACCEPTANCE_V2.json'
            accept = load(accept_path)
            assert sha(side) == correction['reuse_stu5_3']['sidecar_sha256']
            assert row['sha256'] == correction['reuse_stu5_3']['file_sha256']
            assert accept['effective_contract_sha256'] == EFFECTIVE_CONTRACT_SHA and accept['correction_sha256'] == CORRECTION_SHA
            assert accept['file_sha256'] == row['sha256'] and accept['symmetry'] is False
            assert all(v for k, v in row['checks'].items() if k != 'symmetric')
            acceptance = {'path': str(accept_path), 'sha256': sha(accept_path)}
        else:
            assert row['effective_contract_sha256'] == EFFECTIVE_CONTRACT_SHA and row['correction_sha256'] == CORRECTION_SHA
            assert all(row['required_checks'].values())
            acceptance = None
        tsms[key] = {'path': str(path), 'sha256': row['sha256'], 'sidecar': str(side),
                     'sidecar_sha256': sha(side), 'acceptance_v2': acceptance, 'shape': [n, n]}
    return amendment, correction, prior, tsms


def required_raw_inputs(work, amendment):
    work = Path(work)
    raw = load(work / 'inputs/quva_raw_inputs.json')
    assert len(raw) == 100 and sum(r['input_T'] for r in raw) == 48030
    combined = deepcopy(raw)
    for row in amendment['missing5_inputs']:
        item = dict(row)
        item['path'] = str(work / 'missing5_recovery_v1/features' / (row['key'].replace('/', '__') + '.npz'))
        combined.append(item)
    combined.sort(key=lambda r: r['key'])
    assert len(combined) == 105 and sum(r['input_T'] for r in combined) == 51349
    assert len({r['key'] for r in combined}) == 105
    return combined


def validate_member(row, member):
    n, pop = row['full_N'], row['population']
    assert not member['asset_errors']
    arrays = member['arrays']
    fk = 'img_feature' if 'img_feature' in arrays else 'frames'
    assert arrays[fk]['shape'] == [n, 37632] and arrays[fk]['CRC_and_finite_verified'] is True
    if 'frames' in arrays and 'img_feature' in arrays:
        assert all(arrays['frames'][k] == arrays['img_feature'][k] for k in ('shape', 'dtype', 'array_payload_sha256'))
    if pop.startswith('rc_'):
        assert member['scalar_metadata']['length'] == n and arrays['length']['shape'] == []
        assert arrays['length']['dtype'].startswith(('int', 'uint'))
    if pop in ('rc_train', 'rc_valid'):
        assert arrays['gt_tsm']['shape'] == [n, n] and arrays['gt_tsm']['CRC_and_finite_verified'] is True
    path = Path(member['path'])
    assert path.is_file() and path.stat().st_size == member['bytes']


def qualify_recovered(row, work, tsm):
    marker = Path(row['path'] + '.json')
    if not marker.exists():
        return None
    member = load(marker)
    assert member['qualified_key'] == row['key'] and member['type'] == 'generated_missing_author_member'
    assert member['base_contract_sha256'] == CONTRACT_SHA and member['effective_contract_sha256'] == EFFECTIVE_CONTRACT_SHA
    assert member['amendment_sha256'] == AMENDMENT_SHA and member['correction_sha256'] == CORRECTION_SHA
    assert member['binding'] == load(Path(work) / 'QUVA_BINDING.json')
    assert member['binding']['candidate_identity'] == identity(work)
    assert member['raw_video_sha256'] == row['raw_video_sha256']
    if tsm is not None:
        assert member['tsm_receipt_sha256'] == tsm['sidecar_sha256']
    validate_member(row, member)
    return member


def refresh_recovery(work):
    """Receipt-only new chain; original CPU/readiness/targets remain untouched."""
    import torch
    from native_inputs import member_index, qualify_quva
    work = Path(work)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    amendment, correction, prior, tsms = recovery_sources(work)
    c = contract()
    rows = load(work / 'inputs/inputs.json')
    old_ready = load(work / 'inputs/READINESS.json')
    assert len(rows) == 1243 and digest([r['key'] for r in rows]) == old_ready['order_sha256']
    members, snapshots = member_index(Path(c['paths']['author_assets']))
    asset = Path(c['paths']['author_assets'])
    state, control, failure = load(asset / 'STATE.json'), load(asset / 'CONTROL.json'), load(asset / 'FAILURE_003.json')
    assert state['status'] == 'stopped_on_failure' and failure['error'] == 'Missing native755/validation/test assets'
    assert not (asset / 'COMPLETION.json').exists(), 'never relabel original author completion'
    assert len(control['objects']) == 61 and sum(o['bytes'] for o in control['objects']) == 128629792634
    assert all(state['objects'][o['id']]['verified'] and state['objects'][o['id']]['sha256'] == o['sha256'] for o in control['objects'])
    missing5 = {r['key'] for r in amendment['missing5_inputs']}
    required = {r['key'] for r in rows if r['population'] != 'quva_test'}
    assert required - set(members) == missing5 and len(required & set(members)) == 1138
    raw105 = required_raw_inputs(work, amendment)
    rawpath = work / 'inputs/raw_feature_inputs105.json'
    if rawpath.exists():
        assert load(rawpath) == raw105
    else:
        savejson(rawpath, raw105)
    raw_by_key = {r['key']: r for r in raw105}
    updated, missing, generated = deepcopy(rows), [], []
    author_inventory = []
    for item in updated:
        key, pop = item['key'], item['population']
        if key in missing5:
            item['path'] = raw_by_key[key]['path']
            item['source_type'] = 'generated_missing_author_member'
            item['tsm_provenance'] = tsms.get(key)
            member = qualify_recovered(item, work, tsms.get(key))
            if member is None:
                missing.append({'key': key, 'reason': 'missing-author raw-RGB feature production pending'})
            else:
                item.update(sha256=member['sha256'], derived_member_receipt=member)
            generated.append({'key': key, 'path': item['path'], 'available': member is not None,
                              'receipt_sha256': sha(item['path'] + '.json') if member else None})
        elif pop == 'quva_test':
            if not qualify_quva(item, work):
                missing.append({'key': key, 'reason': 'complete official QUVA features pending'})
        else:
            member = members[key]
            validate_member(item, member)
            if 'sha256' in item:
                assert item['sha256'] == member['sha256'] and item['path'] == member['path']
            item.update(path=member['path'], sha256=member['sha256'], crc_finite_receipt=member)
            author_inventory.append({'key': key, 'path': member['path'], 'sha256': member['sha256']})
    prediction = [{k: r[k] for k in PRED_FIELDS if k in r} for r in updated if r['population'].endswith('test')]
    assert len(prediction) == 357 and len(author_inventory) == 1138
    assert sum(r['population'] in ('rc_train', 'rc_valid') and r['key'] not in missing5 for r in updated) == 883
    assert dict(Counter(r['population'] for r in updated)) == old_ready['counts']
    savejson(input_path(work, 'inputs'), updated)
    savejson(input_path(work, 'prediction_inputs'), prediction)
    hashes = dict(old_ready['hashes'])
    for name in ('inputs', 'prediction_inputs'):
        hashes[name] = sha(input_path(work, name))
    ready = deepcopy(old_ready)
    ready.update(status='PENDING' if missing else 'PASS', missing=missing, hashes=hashes,
                 author_objects_verified=61, author_members_available=1138,
                 recovered_members_available=sum(r['available'] for r in generated),
                 quva_features_available=sum(r['population'] == 'quva_test' and 'sha256' in r for r in updated),
                 native_author_TSMs=883, native_derived_TSMs=3, index_snapshots=snapshots,
                 parent_readiness_sha256=ORIGINAL_READY_SHA, parent_CPU_check_sha256=ORIGINAL_CPU_SHA,
                 base_contract_sha256=CONTRACT_SHA, effective_contract_sha256=EFFECTIVE_CONTRACT_SHA,
                 amendment_sha256=AMENDMENT_SHA, correction_sha256=CORRECTION_SHA,
                 raw105_manifest_sha256=sha(rawpath), raw105_videos=105, raw105_frames=51349,
                 author_original_status=state['status'], no_model_forward=True,
                 no_existing_start_end_or_TSM_reconstruction=True, targeted_closure='missing5 V2 receipt-only chain')
    assert ready['status'] != 'PASS' or ready['recovered_members_available'] == 5 and ready['quva_features_available'] == 100
    savejson(input_path(work, 'READINESS'), ready)
    closure = {'status': 'CPU_PARITY_AND_TSM_COMPLETE_FEATURES_PENDING' if missing else 'ALL1243_INPUTS_PASS',
               'base_contract_sha256': CONTRACT_SHA, 'effective_contract_sha256': EFFECTIVE_CONTRACT_SHA,
               'amendment_sha256': AMENDMENT_SHA, 'correction_sha256': CORRECTION_SHA,
               'originalSTATE_sha256': sha(asset / 'STATE.json'), 'originalFAILURE_003_sha256': sha(asset / 'FAILURE_003.json'),
               'original_author_status': state['status'], 'author_objects': control['objects'],
               'archive_bytes': 128629792634, 'physical_member_receipts': len(members), 'index_snapshots': snapshots,
               'required_author_inventory': author_inventory, 'exact_missing5': sorted(missing5),
               'generated5_provenance_inventory': generated, 'TSM3_receipts': tsms,
               'TSM_completion_sha256': sha(work / 'missing5_recovery_v1/TSM_COMPLETE_V2.json'),
               'parity_receipt_sha256': correction['inherited_parity']['receipt_sha256'],
               'unchanged_author_TSM_count': 883, 'all1243_order_sha256': old_ready['order_sha256'],
               'frozen_metadata': amendment['frozen_metadata'], 'no_original_author_relabel_or_mutation': True}
    savejson(work / 'AUTHOR_ASSET_RECOVERY_CLOSURE.json', closure)
    cpu = {'status': ready['status'], 'interfaces_status': 'PASS_CPU_INPUT_PREPARATION',
           'candidate_identity': identity(work), 'readiness_sha256': sha(input_path(work, 'READINESS')),
           'inherited_CPU_check': {'path': str(work / 'CPU_CHECK.json'), 'sha256': ORIGINAL_CPU_SHA,
                                   'candidate_identity': prior['candidate_identity'], 'passed_old_primitives': prior['passed']},
           'inherited_readiness_sha256': ORIGINAL_READY_SHA,
           'targeted_checks': ['immutable parent/effective/amendment/correction chain', 'fixed parity receipt reuse',
                               'one retained and two first-generated TSM receipts', '1138+5+100 exact membership',
                               '105 raw inputs without labels', 'unchanged targets/scorer/QUVA manifests'],
           'recovery_closure_sha256': sha(work / 'AUTHOR_ASSET_RECOVERY_CLOSURE.json'),
           'no_model_forward': True, 'cuda_initialized': False, 'old_GT_or_TSM_generation_calls': 0}
    savejson(cpu_check_path(work), cpu)
    storage_guard(work)
    return ready


def extract_required105(work, receipt):
    """Same frozen backbone/single-frame math; only the five absent NPZs are new."""
    import numpy as np
    import torch
    import cv2
    import zipfile
    from native_common import runtime_flags
    from native_extract import backbone
    work = Path(work)
    invocation_start = time.monotonic()
    gpu_gate(work, receipt, 'extract-quva')
    amendment, _, _, tsms = recovery_sources(work)
    rows = load(work / 'inputs/raw_feature_inputs105.json')
    assert rows == required_raw_inputs(work, amendment) and len(rows) == 105
    runtime_flags()
    model, strict = backbone(work)
    model.eval().to('cuda')
    binding = {'candidate_identity': identity(work),
               'backbone': {k: strict[k] for k in ('source_sha256', 'base_config_sha256', 'weights_sha256')},
               'inputs_sha256': sha(work / 'inputs/raw_feature_inputs105.json'),
               'mode': 'official-single-frame-transpose0321-fp32-minus127.5-div127.5', 'videos': 105, 'frames': 51349}
    if (work / 'QUVA_BINDING.json').exists():
        assert load(work / 'QUVA_BINDING.json') == binding
    else:
        savejson(work / 'QUVA_BINDING.json', binding)
    done, reused = [], set()
    for row in rows:
        storage_guard(work)
        dest = Path(row['path'])
        marker = Path(str(dest) + '.json')
        if marker.exists():
            old = load(marker)
            assert old['binding'] == binding and old['raw_video_sha256'] == row['raw_video_sha256']
            assert sha(dest) == old['sha256'] and dest.stat().st_size == old['bytes']
            assert old['T'] == row['input_T']
            done.append(old)
            reused.add(row['key'])
            continue
        assert not dest.exists(), ('uncommitted final output; preserve for operator', row['key'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.parent / ('.' + dest.name + '.' + uuid.uuid4().hex + '.features.npy')
        f = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(row['input_T'], 37632))
        cap = cv2.VideoCapture(row['raw_video'])
        assert cap.isOpened(), row['key']
        count, started = 0, time.monotonic()
        try:
            with torch.no_grad():
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    assert count < row['input_T']
                    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_LINEAR)
                    x = torch.FloatTensor(np.ascontiguousarray(rgb.transpose(2, 1, 0)))
                    x -= 127.5
                    x /= 127.5
                    out = model(x[None, :, None].to('cuda'))
                    assert tuple(out.shape) == (1, 768, 1, 7, 7)
                    flat = out[0, :, 0].contiguous().view(37632).cpu().numpy()
                    assert np.isfinite(flat).all()
                    f[count] = flat
                    count += 1
            assert count == row['input_T']
            f.flush()
            with open(temp, 'rb') as handle:
                os.fsync(handle.fileno())
            n = row['input_T']
            if row['population'] == 'quva_test':
                del f
                os.replace(temp, dest)
                syncdir(dest.parent)
                result = {'key': row['key'], 'T': n, 'shape': [n, 37632], 'dtype': 'float32', 'finite': True,
                          'sha256': sha(dest), 'bytes': dest.stat().st_size, 'seconds': time.monotonic()-started,
                          'raw_video_sha256': row['raw_video_sha256'], 'binding': binding}
            else:
                arrays = {'img_feature': {'shape': [n, 37632], 'dtype': 'float32', 'CRC_and_finite_verified': True},
                          'length': {'shape': [], 'dtype': 'int64'}}
                data = {'img_feature': f, 'length': np.asarray(n, dtype=np.int64)}
                tsm = tsms.get(row['key'])
                if tsm:
                    # Only the three new TSMs are read; no generator or old start/end array is called.
                    data['gt_tsm'] = np.load(tsm['path'], allow_pickle=False)
                    assert data['gt_tsm'].shape == (n, n)
                    arrays['gt_tsm'] = {'shape': [n, n], 'dtype': 'float64', 'CRC_and_finite_verified': True}
                atomic(dest, lambda handle: np.savez(handle, **data))
                with zipfile.ZipFile(dest) as archive:
                    assert archive.testzip() is None, ('new recovered NPZ CRC failure', row['key'])
                result = {'key': row['key'], 'qualified_key': row['key'], 'T': n, 'path': str(dest),
                          'type': 'generated_missing_author_member', 'arrays': arrays, 'asset_errors': [],
                          'scalar_metadata': {'length': n}, 'sha256': sha(dest), 'bytes': dest.stat().st_size,
                          'base_contract_sha256': CONTRACT_SHA, 'effective_contract_sha256': EFFECTIVE_CONTRACT_SHA,
                          'amendment_sha256': AMENDMENT_SHA, 'correction_sha256': CORRECTION_SHA,
                          'raw_video_sha256': row['raw_video_sha256'], 'binding': binding,
                          'tsm_receipt_sha256': tsm['sidecar_sha256'] if tsm else None,
                          'tsm_acceptance_v2': tsm['acceptance_v2'] if tsm else None,
                          'parity_receipt_sha256': load(CORRECTION)['inherited_parity']['receipt_sha256'] if tsm else None,
                          'seconds': time.monotonic()-started}
                del data, f
            savejson(marker, result)
            if temp.exists():
                temp.unlink()  # Only a successful new output's redundant temporary feature array.
            done.append(result)
            savejson(work / 'FEATURES105_PROGRESS.json', {'videos': len(done), 'frames': sum(x['T'] for x in done), 'binding': binding})
        finally:
            cap.release()
    assert len(done) == 105 and sum(x['T'] for x in done) == 51349
    timing = timing_summary(done, time.monotonic()-invocation_start, reused, total_field='seconds')
    savejson(work / 'FEATURES105_COMPLETE.json', {'status': 'PASS', 'binding': binding, 'videos': done, 'frames': 51349, 'timing': timing})
