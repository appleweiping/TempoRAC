"""Frozen membership adapters. Supervision never enters PredictionInputs."""
from collections import Counter
import contextlib
import io
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from native_common import ROOT, PACKET, contract, load, sha, digest, savejson, atomic, science


def feature_array(npz):
    keys = npz.files
    key = 'img_feature' if 'img_feature' in keys else 'frames'
    f = npz[key]
    if 'frames' in keys and 'img_feature' in keys:
        other = npz['frames']
        assert f.dtype == other.dtype and f.shape == other.shape and np.array_equal(f, other), 'feature alias mismatch'
    assert f.ndim == 2 and f.shape[1] == 37632
    return f


def member_index(asset):
    out, snapshots = {}, []
    for p in sorted((asset / 'indices').glob('members_*.jsonl')):
        # Worker appends only complete verified records; ignore an uncommitted tail.
        raw = p.read_bytes()
        lines = raw.splitlines(keepends=True)
        committed = b''.join(x for x in lines if x.endswith(b'\n'))
        snapshots.append({'path': str(p), 'committed_sha256': __import__('hashlib').sha256(committed).hexdigest()})
        for line in committed.splitlines():
            m = __import__('json').loads(line)
            k = m.get('qualified_key')
            if not k:
                continue
            if k in out:
                assert out[k]['sha256'] == m['sha256'], ('duplicate member mismatch', k)
            out[k] = m
    return out, snapshots


def quva_extraction_inputs(prediction_rows):
    """Extraction identity depends on raw videos/destinations, never generated features."""
    fields = ('key', 'population', 'full_N', 'input_T', 'raw_video', 'raw_video_sha256', 'path')
    return [{k: r[k] for k in fields} for r in prediction_rows if r['population'] == 'quva_test']


def qualify_quva(item, work):
    receipt = Path(item['path'] + '.json')
    if not receipt.exists():
        return False
    q = load(receipt)
    assert q['raw_video_sha256'] == item['raw_video_sha256'] and q['T'] == item['full_N']
    assert q['shape'] == [item['full_N'], 37632] and q['finite'] is True
    assert q['binding'] == load(Path(work) / 'QUVA_BINDING.json')
    assert Path(item['path']).exists() and Path(item['path']).stat().st_size == q['bytes']
    item['sha256'] = q['sha256']
    return True


def prepare(work):
    work = Path(work)
    c = contract()
    mp = Path(c['inputs']['inherited_manifest'])
    ip = Path(c['inputs']['inherited_decoded_identity'])
    assert sha(mp) == c['inputs']['inherited_manifest_sha256']
    assert sha(ip) == c['inputs']['inherited_decoded_identity_sha256']
    evidence = load(PACKET / 'SOURCE_SCHEMA_EVIDENCE.json')
    identities = {r['key']: r for r in load(ip)['entries']}
    ucf = {r['key']: r for r in evidence['ucf']['rows']}
    assert len(ucf) == 105 and sum(r['crop_needed'] for r in ucf.values()) == 88
    members, index_snapshot = member_index(Path(c['paths']['author_assets']))
    _, gt, decoder = science(work)
    csvdir = mp.parent
    with contextlib.redirect_stdout(io.StringIO()):
        csv_labels = {pop: gt.get_labels_dict(str(csvdir / (split + '.csv')))
                      for pop, split in [('rc_train', 'train'), ('rc_valid', 'valid'), ('rc_test', 'test')]}
    csv_sha = {p.name: sha(p) for p in sorted(csvdir.glob('*.csv'))}
    allrows, targets, score_rows, prediction_rows, missing = [], [], [], [], []
    for source in load(mp):
        pop = source['population']
        if source['dataset'] == 'QUVA':
            assert pop == 'test'
            pop = 'quva_test'
        if pop not in c['inputs']['populations'] or (pop == 'rc_train' and source['video_id'] == 'stu6_44'):
            continue
        oldkey = source['population'] + '/' + source['video_id']
        key = pop + '/' + source['video_id']
        ident = identities[oldkey]
        n = ident['full_N']
        item = {'key': key, 'population': pop, 'full_N': n, 'input_T': n,
                'raw_video': source['file'], 'raw_video_sha256': source['source_video_sha256']}
        score = {'key': key, 'population': pop, 'dataset': source['dataset'], 'scalar': source['count']}
        if pop.startswith('rc_'):
            labels = csv_labels[pop][source['video_id'] + '.npz']
            assert labels.tolist() == [int(float(x)) for x in source['bounds']], ('CSV bounds mismatch', key)
            assert source['count'] is not None
            with contextlib.redirect_stdout(io.StringIO()):
                s, e = gt.GenerateGT().gen_gt(labels, n, 'start')
            s, e = np.asarray(s, np.float32), np.asarray(e, np.float32)
            assert s.shape == e.shape == (n,) and np.isfinite(s).all() and np.isfinite(e).all()
            score['native_gt'] = float(decoder.count_peaks_scaling(torch.from_numpy(s[None]), pro=0.2)[0])
            score['native_minus_scalar'] = score['native_gt'] - source['count']
            if pop in ('rc_train', 'rc_valid'):
                assert all(0 <= x < n for x in labels[::2]), ('invalid start', key)
                targetfile = work / 'inputs/targets' / (key.replace('/', '__') + '.npz')
                if targetfile.exists():
                    with np.load(targetfile, allow_pickle=False) as prior:
                        assert np.array_equal(prior['start'], s) and np.array_equal(prior['end'], e)
                else:
                    atomic(targetfile, lambda f, s=s, e=e: np.savez(f, start=s, end=e))
                targets.append({'key': key, 'target_path': str(targetfile), 'target_sha256': sha(targetfile),
                                'bounds': labels.tolist(), 'length': n,
                                'out_of_range_end': [int(x) for x in labels[1::2] if x < 0 or x >= n]})
        else:
            score['native_gt'] = source['count']
        member = members.get(key)
        if pop == 'ucf_test':
            u = ucf[key]
            assert n == u['full_N'] and source['count'] == u['embedded_count'] == u['frozen_count']
            assert source['frame_range'] == u['roi_inclusive']
            item['slice'] = u['slice_python']
            item['input_T'] = u['ROI_N']
            assert member and member['sha256'] == u['author_npz_sha256']
        if pop == 'quva_test':
            item['path'] = str(work / 'quva_features' / (source['video_id'] + '.npy'))
            if not qualify_quva(item, work):
                missing.append({'key': key, 'reason': 'complete official QUVA features pending'})
        elif member is None:
            missing.append({'key': key, 'reason': 'author member pending'})
        else:
            assert not member['asset_errors'], (key, member['asset_errors'])
            arrays = member['arrays']
            fk = 'img_feature' if 'img_feature' in arrays else 'frames'
            assert arrays[fk]['shape'] == [n, 37632]
            assert arrays[fk]['CRC_and_finite_verified'] is True
            if 'frames' in arrays and 'img_feature' in arrays:
                assert arrays['frames']['shape'] == arrays['img_feature']['shape']
                assert arrays['frames']['dtype'] == arrays['img_feature']['dtype']
                assert arrays['frames']['array_payload_sha256'] == arrays['img_feature']['array_payload_sha256']
            if pop.startswith('rc_'):
                assert member['scalar_metadata']['length'] == n
                assert arrays['length']['shape'] == [] and arrays['length']['dtype'].startswith(('int', 'uint'))
            if pop in ('rc_train', 'rc_valid'):
                assert arrays['gt_tsm']['shape'] == [n, n] and arrays['gt_tsm']['CRC_and_finite_verified'] is True
            path = Path(member['path'])
            assert path.is_file() and path.stat().st_size == member['bytes']
            item.update(path=str(path), sha256=member['sha256'], crc_finite_receipt=member)
        allrows.append(item)
        if pop.endswith('test'):
            # Deliberately construct a new whitelist: no count, bounds or label/TSM metadata.
            prediction_rows.append({k: item[k] for k in ('key', 'population', 'full_N', 'input_T', 'raw_video', 'raw_video_sha256', 'slice', 'path', 'sha256') if k in item})
            score_rows.append(score)
    allrows.sort(key=lambda r: r['key'])
    prediction_rows.sort(key=lambda r: r['key'])
    targets.sort(key=lambda r: r['key'])
    counts = dict(Counter(r['population'] for r in allrows))
    assert counts == {k: v for k, v in c['inputs']['populations'].items() if k not in ('total_required', 'final_total')}
    assert len(targets) == 886 and len(prediction_rows) == len(score_rows) == 357
    assert sum(r['input_T'] for r in prediction_rows if r['population'] == 'quva_test') == 48030
    assert len({r['key'] for r in allrows}) == 1243
    exceptional = [r['key'] for r in targets if r['out_of_range_end']]
    assert exceptional == sorted(c['inputs']['exceptional_tsm_required'])
    output = {'inputs': allrows, 'targets': targets, 'prediction_inputs': prediction_rows,
              'score_labels': sorted(score_rows, key=lambda r: r['key'])}
    hashes = {}
    for name, rows in output.items():
        savejson(work / 'inputs' / (name + '.json'), rows)
        hashes[name] = sha(work / 'inputs' / (name + '.json'))
    savejson(work / 'inputs/quva_raw_inputs.json', quva_extraction_inputs(prediction_rows))
    state = load(Path(c['paths']['author_assets']) / 'STATE.json')
    verified = sum(o.get('verified', False) for o in state['objects'].values())
    ready = not missing and verified == 61
    receipt = {'status': 'PASS' if ready else 'PENDING', 'counts': counts,
               'required': 1243, 'native_targets': 886, 'ucf_roi': {'crop': 88, 'full': 17},
               'exceptional_endpoints': exceptional, 'author_objects_verified': verified,
               'author_members_available': sum('crc_finite_receipt' in r for r in allrows),
               'quva_features_available': sum(r['population'] == 'quva_test' and 'sha256' in r for r in allrows),
               'missing': missing, 'hashes': hashes, 'index_snapshots': index_snapshot, 'csv_sha256': csv_sha,
               'order_sha256': digest([r['key'] for r in allrows]),
               'reused_crc_finite': True, 'media_decoded_again': False, 'no_model_forward': True}
    savejson(work / 'inputs/READINESS.json', receipt)
    return receipt


def refresh_inputs(work):
    """Refresh feature receipts while inheriting already checked supervision bytes."""
    from copy import deepcopy
    import os
    from native_common import identity
    work = Path(work)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU receipt refresh only'
    c = contract()
    previous = load(work / 'CPU_CHECK.json')
    current = identity(work)
    assert len(previous['passed']) == 8 and previous['no_model_forward'] and not previous['cuda_initialized']
    for field in ('contract', 'env_spec', 'runtime', 'science'):
        assert current[field] == previous['candidate_identity'][field], ('previous check not reusable', field)
    before_sha = sha(work / 'inputs/READINESS.json')
    assert previous['readiness_sha256'] == before_sha, 'inherited CPU/input receipt mismatch'
    ready = load(work / 'inputs/READINESS.json')
    for name, expected in ready['hashes'].items():
        assert sha(work / 'inputs' / (name + '.json')) == expected, ('inherited manifest changed', name)
    rows = load(work / 'inputs/inputs.json')
    targets = load(work / 'inputs/targets.json')
    labels = load(work / 'inputs/score_labels.json')
    raw_quva = load(work / 'inputs/quva_raw_inputs.json')
    frozen = {name: sha(work / 'inputs' / (name + '.json'))
              for name in ('targets', 'score_labels', 'quva_raw_inputs')}
    assert len(targets) == ready['native_targets'] == 886
    assert len(labels) == 357 and sum(r['population'] == 'rc_test' for r in labels) == 152
    assert len(raw_quva) == 100 and sum(r['input_T'] for r in raw_quva) == 48030
    assert len(rows) == 1243 and digest([r['key'] for r in rows]) == ready['order_sha256']
    counts = dict(Counter(r['population'] for r in rows))
    assert counts == ready['counts'] == {k: v for k, v in c['inputs']['populations'].items()
                                       if k not in ('total_required', 'final_total')}
    members, snapshots = member_index(Path(c['paths']['author_assets']))
    state = load(Path(c['paths']['author_assets']) / 'STATE.json')
    verified = sum(o.get('verified', False) for o in state['objects'].values())
    assert verified == 61 and state['status'].lower() in ('complete', 'completed'), 'formal author completion required'
    updated, missing = deepcopy(rows), []
    for item in updated:
        key, pop, n = item['key'], item['population'], item['full_N']
        if pop == 'quva_test':
            if not qualify_quva(item, work):
                assert 'sha256' not in item, ('previously qualified QUVA receipt missing', key)
                missing.append({'key': key, 'reason': 'complete official QUVA features pending'})
            continue
        member = members.get(key)
        assert member is not None, ('completed author input missing', key)
        assert not member['asset_errors'], (key, member['asset_errors'])
        arrays = member['arrays']
        fk = 'img_feature' if 'img_feature' in arrays else 'frames'
        assert arrays[fk]['shape'] == [n, 37632] and arrays[fk]['CRC_and_finite_verified'] is True
        if 'frames' in arrays and 'img_feature' in arrays:
            assert arrays['frames']['shape'] == arrays['img_feature']['shape']
            assert arrays['frames']['dtype'] == arrays['img_feature']['dtype']
            assert arrays['frames']['array_payload_sha256'] == arrays['img_feature']['array_payload_sha256']
        if pop.startswith('rc_'):
            assert member['scalar_metadata']['length'] == n
            assert arrays['length']['shape'] == [] and arrays['length']['dtype'].startswith(('int', 'uint'))
        if pop in ('rc_train', 'rc_valid'):
            assert arrays['gt_tsm']['shape'] == [n, n] and arrays['gt_tsm']['CRC_and_finite_verified'] is True
        if 'sha256' in item:
            assert item['sha256'] == member['sha256'] and item['path'] == member['path'], ('qualified author input changed', key)
        path = Path(member['path'])
        assert path.is_file() and path.stat().st_size == member['bytes']
        item.update(path=str(path), sha256=member['sha256'], crc_finite_receipt=member)
    prediction = [{k: r[k] for k in ('key', 'population', 'full_N', 'input_T', 'raw_video',
                  'raw_video_sha256', 'slice', 'path', 'sha256') if k in r}
                  for r in updated if r['population'].endswith('test')]
    assert len(prediction) == 357 and quva_extraction_inputs(prediction) == raw_quva
    assert [r['key'] for r in prediction] == [r['key'] for r in labels]
    # Archive only small prior manifests/receipts; no feature or target payload opens.
    inherited = work / 'input_refresh_history' / before_sha
    if not inherited.exists():
        for name in ('inputs/inputs.json', 'inputs/prediction_inputs.json', 'inputs/READINESS.json', 'CPU_CHECK.json'):
            payload = (work / name).read_bytes()
            atomic(inherited / name, lambda f, b=payload: f.write(b))
    prior_cpu_sha = sha(work / 'CPU_CHECK.json')
    hashes = dict(ready['hashes'])
    for name, value in (('inputs', updated), ('prediction_inputs', prediction)):
        savejson(work / 'inputs' / (name + '.json'), value)
        hashes[name] = sha(work / 'inputs' / (name + '.json'))
    ready.update(status='PENDING' if missing else 'PASS', missing=missing, hashes=hashes,
                 author_objects_verified=verified, author_members_available=sum('crc_finite_receipt' in r for r in updated),
                 quva_features_available=sum(r['population'] == 'quva_test' and 'sha256' in r for r in updated),
                 index_snapshots=snapshots,
                 inherited_supervision={'cpu_check_sha256': prior_cpu_sha, 'readiness_sha256': before_sha,
                                        'candidate_identity': previous['candidate_identity'], 'frozen_manifest_sha256': frozen},
                 reused_crc_finite=True, media_decoded_again=False, no_model_forward=True)
    for name, expected in frozen.items():
        assert sha(work / 'inputs' / (name + '.json')) == expected, ('frozen manifest changed', name)
    savejson(work / 'inputs/READINESS.json', ready)
    savejson(work / 'INPUT_REFRESH.json', {'status': ready['status'], 'candidate_identity': current,
             'inherited_cpu_check_sha256': prior_cpu_sha, 'inherited_readiness_sha256': before_sha,
             'readiness_sha256': sha(work / 'inputs/READINESS.json'), 'frozen_manifest_sha256': frozen,
             'native_targets_inherited': 886, 'rc_test_native_gt_inherited': 152,
             'target_payloads_opened': 0, 'GT_or_decoder_calls': 0, 'feature_payloads_opened': 0,
             'CRC_or_finite_checks_repeated': 0})
    return ready


class SupervisedInputs(torch.utils.data.Dataset):
    def __init__(self, work, population):
        work = Path(work)
        assert population in ('rc_train', 'rc_valid')
        self.rows = [r for r in load(work / 'inputs/inputs.json') if r['population'] == population]
        self.targets = {r['key']: r for r in load(work / 'inputs/targets.json')}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with np.load(r['path'], allow_pickle=False) as data:
            f = torch.FloatTensor(feature_array(data))
            assert int(data['length']) == len(f) == r['input_T']
            tsm = torch.FloatTensor(data['gt_tsm'])
            assert tsm.shape == (len(f), len(f))
        with np.load(self.targets[r['key']]['target_path'], allow_pickle=False) as targets:
            s, e = torch.FloatTensor(targets['start']), torch.FloatTensor(targets['end'])
        return f, tsm, (s, e), r['key']


def supervised_collate(rows):
    """Exact native zero padding; masks derived only from feature lengths."""
    f, t, labels, keys = zip(*rows)
    lengths = torch.tensor([len(x) for x in f])
    features = pad_sequence(f, batch_first=True)
    maxlen = features.shape[1]
    masks = torch.arange(maxlen)[None, :] < lengths[:, None]
    s = pad_sequence([x[0] for x in labels], batch_first=True, padding_value=-100)
    e = pad_sequence([x[1] for x in labels], batch_first=True, padding_value=-100)
    tm = masks[:, :, None] & masks[:, None, :]
    tsm = torch.stack([torch.nn.functional.pad(x, (0, maxlen-len(x), 0, maxlen-len(x))) for x in t])
    return {'features': features, 'target_s': s*masks, 'target_e': e*masks,
            'gt_tsm': tsm, 'masks': masks, 'tsm_masks': tm, 'keys': keys, 'lengths': lengths.tolist()}


class PredictionInputs(torch.utils.data.Dataset):
    def __init__(self, rows):
        allowed = {'key', 'population', 'full_N', 'input_T', 'raw_video', 'raw_video_sha256', 'slice', 'path', 'sha256'}
        assert all(set(r) <= allowed for r in rows), 'prediction metadata contains supervision'
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        if r['population'] == 'quva_test':
            data = np.load(r['path'], allow_pickle=False)
        else:
            with np.load(r['path'], allow_pickle=False) as npz:
                data = feature_array(npz)
            if 'slice' in r:
                data = data[slice(*r['slice'])]
        assert data.shape == (r['input_T'], 37632)
        return torch.FloatTensor(data), r['key']
