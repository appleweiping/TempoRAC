"""Read-only NT1/NB1 diagnostics. No producer, model or label selection imports.

The proposed supplied-track metrics are not certified official MultiRep scores.
Only this evaluator layer accepts original object mappings/count/period labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from .native_bridge_v1 import load_native_feature, native_clock, native_mapping

EXPECTED = {'train': {'videos': 794, 'persons': 1878}, 'val': {'videos': 183, 'persons': 486}}
SCHEMA = 'temporac.native-target-diagnostics.v1'
METRIC_REFERENCE = 'FULL-v1-supplied-track-proposed-video-person-macro'
SEED = 20261104
TARGET_ARRAYS = {'target_valid', 'pulse', 'chi', 'traversal_bounds', 'seams', 'run_id', 'branch_id'}
GEOMETRY = {'v12_geometry', 'v15_geometry'}


class DiagnosticError(ValueError):
    """Whole-population integrity failure, never a reason to filter an identity."""


def require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'invalid integer {name}')
    return value


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f'invalid finite {name}')
    require(value > 0 if positive else value >= 0, f'invalid {name}')
    return float(value)


def binary(value, size, name):
    a = np.asarray(value)
    require(a.shape == (size,) and a.dtype == np.bool_, f'{name} requires full-domain bool')
    return a


def load_census(path, *, _expected=EXPECTED):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    actual = {s: {'videos': 0, 'persons': 0} for s in EXPECTED}
    keys, paths = set(), set()
    for r in rows:
        require(r['split'] in EXPECTED, 'only train/val census permitted')
        rel = PurePosixPath(r['media_path'])
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in r['media_path']
                and rel.parts[0] == r['split'] and rel.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'),
                'invalid split-relative media path')
        require(r['media_key'] == hashlib.sha256(r['media_path'].encode()).hexdigest(), 'census routing key mismatch')
        require(r['media_key'] not in keys and r['media_path'] not in paths, 'duplicate census row')
        keys.add(r['media_key'])
        paths.add(r['media_path'])
        integer(r['length'], 'source length', 1)
        require(r['persons'] and [p['slot'] for p in r['persons']] == list(range(len(r['persons']))), 'incomplete slots')
        actual[r['split']]['videos'] += 1
        actual[r['split']]['persons'] += len(r['persons'])
    require(actual == _expected, f'full census mismatch: {actual}')
    if _expected == EXPECTED:
        require((sum(r['length'] for r in rows), sum(r['length'] * len(r['persons']) for r in rows))
                == (695625, 1758034), 'frozen frame census mismatch')
    return rows


def source_membership(census, components):
    by_id = {str(PurePosixPath(r['media_path']).parent): r['media_key'] for r in census}
    result = {}
    component_ids = set()
    for c in components:
        require(isinstance(c['id'], str) and c['id'] not in component_ids, 'duplicate source component')
        component_ids.add(c['id'])
        require(len(c['members']) == len(set(c['members'])), 'duplicate component members')
        for member in c['members']:
            if member in by_id:
                key = by_id[member]
                require(key not in result, 'overlapping source components')
                result[key] = c['id']
    require(set(result) == {r['media_key'] for r in census}, 'missing source component mapping')
    aliases = defaultdict(list)
    for r in census:
        aliases[r['media_sha256']].append(r)
    groups = [g for g in aliases.values() if len(g) > 1]
    for group in groups:
        require(len({result[r['media_key']] for r in group}) == 1, 'same-media aliases split across source components')
    return result, groups


def _ledger(path, census, status):
    data = read_json(path)
    require(data['status'] in status and not data.get('failures'), 'population manifest incomplete/failed')
    want = {(r['split'], r['media_key'], p['slot']) for r in census for p in r['persons']}
    records = data['records']
    actual = [(r['split'], r['media_key'], r['slot']) for r in records]
    require(len(actual) == len(set(actual)) and set(actual) == want, 'population ledger census mismatch')
    require(data['videos'] == len(census) and data['identities'] == len(want), 'population manifest totals mismatch')
    return data, dict(zip(actual, records, strict=True))


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def validate_derived(feature, derived, clock_mode):
    """Existing NB1 full derived schema and support/clock domains; no target production."""
    n = int(feature.source_length[0])
    e = n - 1
    specs = {
        'response_input': (np.float32, (n, 269)), 'joint_usable': (np.bool_, (n, 17)),
        'frame_derived_valid': (np.bool_, (n,)), 'observed_edge_support': (np.bool_, (e,)),
        'response_edge_support': (np.bool_, (e,)), 'edge_lengths': (np.float64, (e,)),
        'direction': (np.float64, (n, 66)), 'direction_mask': (np.bool_, (n, 33)),
        'run_bounds': (np.int32, None), 'short_runs': (np.int32, None),
    }
    clock = native_clock(feature)
    mapping = native_mapping(feature, clock)
    specs.update({'map_' + k: (v.dtype, v.shape) for k, v in mapping.items()})
    require(set(derived) == set(specs), 'NB1 derived schema member mismatch')
    for name, (dtype, shape) in specs.items():
        a = derived[name]
        require(a.dtype == np.dtype(dtype) and (a.shape == shape if shape is not None else
                a.ndim == 2 and a.shape[1:] == (2,)), f'NB1 derived dtype/shape mismatch: {name}')
        if a.dtype.kind == 'f' and name not in ('map_seconds', 'map_q'):
            require(np.isfinite(a).all(), f'NB1 nonfinite derived: {name}')
    require(clock_mode == clock.mode, 'NB1 clock mode/feature mismatch')
    for name, expected in mapping.items():
        actual = derived['map_' + name]
        equal = np.array_equal(actual, expected, equal_nan=True) if actual.dtype.kind == 'f' else np.array_equal(actual, expected)
        require(equal, f'NB1 source mapping mismatch: {name}')
    require(not np.any(derived['joint_usable'] & ~feature.joint_observed), 'NB1 usable joint lacks observation')
    require(not np.any(derived['frame_derived_valid'] & ~feature.frame_source_ok), 'NB1 derived frame lacks source support')
    obs, response = derived['observed_edge_support'], derived['response_edge_support']
    valid = derived['frame_derived_valid']
    require(not np.any(obs & ~(valid[:-1] & valid[1:] & clock.edge_time_support)), 'NB1 observed edge lacks frame/clock support')
    unions = {}
    for name in ('run_bounds', 'short_runs'):
        union = np.zeros(e, bool)
        previous = 0
        for a, b in derived[name].tolist():
            require(0 <= a < b <= e and a >= previous, f'NB1 invalid {name} domain/order')
            require((b - a + 1 >= 17) == (name == 'run_bounds'), f'NB1 invalid {name} run length')
            previous = b
            union[a:b] = True
        unions[name] = union
    require(not np.any(unions['run_bounds'] & unions['short_runs'])
            and np.array_equal(response, unions['run_bounds'])
            and np.array_equal(obs, unions['run_bounds'] | unions['short_runs']), 'NB1 run/support mismatch')
    require(np.all(derived['edge_lengths'] >= 0) and not np.any(derived['edge_lengths'][~obs]), 'NB1 edge length domain mismatch')
    require(not np.any(derived['direction'][~np.repeat(derived['direction_mask'], 2, axis=1)]), 'NB1 unsupported direction payload')


def load_identity(bridge_root, target_root, video, slot, nb_ledger, nt_ledger, receipt,
                  *, nb_binding=None, nt_binding=None):
    """Validate immutable numerical artifacts without calling either producer."""
    stem = f"{video['media_key']}_{slot}"
    nb = Path(bridge_root) / 'identities' / video['split'] / stem
    nt = Path(target_root) / 'identities' / video['split'] / stem
    nbmeta, meta = read_json(nb.with_suffix('.json')), read_json(nt.with_suffix('.json'))
    for path, sha in ((nb.with_suffix('.npz'), nbmeta['feature_sha256']),
                      (nb.with_suffix('.derived.npz'), nbmeta['derived_sha256']),
                      (nt.with_suffix('.npz'), meta['npz_sha256'])):
        require(digest(path) == sha, f'artifact hash mismatch: {path}')
    for path in (nb.with_suffix('.npz'), nb.with_suffix('.json'), nb.with_suffix('.derived.npz'),
                 nt.with_suffix('.npz'), nt.with_suffix('.json')):
        receipt[str(path.resolve())] = digest(path)
    native_feature = load_native_feature(nb.with_suffix('.npz'))
    feature, derived, target = native_feature.as_arrays(), _npz(nb.with_suffix('.derived.npz')), _npz(nt.with_suffix('.npz'))
    n = video['length']
    e = n - 1
    require(nbmeta['schema'] == 'temporac.native-learner.v1' and meta['schema'] == 'temporac.native-target.nt1', 'artifact schema mismatch')
    validate_derived(native_feature, derived, nbmeta['summary']['clock_mode'])
    if nb_binding is not None:
        require(all(nbmeta['binding'].get(k) == v for k, v in nb_binding.items()), 'NB1 population binding mismatch')
    if nt_binding is not None:
        require(meta['config_sha256'] == nt_binding['config_sha256']
                and meta['producer_sha256'] == json_digest(nt_binding['producer'])
                and meta['availability_sha256'] == json_digest(nt_binding['availability']), 'NT1 population binding mismatch')
    require(feature['source_length'].tolist() == [n] and feature['slot'].tolist() == [slot]
            and feature['opaque_key'].tobytes().hex() == video['media_key']
            and np.array_equal(feature['source_frame_index'], np.arange(n)), 'NB1 census identity mismatch')
    require(meta['source_length'] == n and meta['slot'] == slot and meta['opaque_key'] == video['media_key'], 'NT1 census identity mismatch')
    require(meta['input_sha256'] == nbmeta['feature_sha256']
            and meta['bridge_sha256'] == json_digest(nbmeta['binding'])
            and meta['raw_sha256'] == json_digest({k: v for k, v in nbmeta['binding'].items() if k.startswith('raw_')}), 'NT1 NB1 binding mismatch')
    require(set(target) == TARGET_ARRAYS, 'NT1 numerical schema mismatch')
    m = binary(target['target_valid'], e, 'target')
    obs = binary(derived['observed_edge_support'], e, 'observed')
    response = binary(derived['response_edge_support'], e, 'response')
    require(not np.any(response & ~obs) and not np.any(m & ~response), 'support funnel violation')
    p, chi, bounds = target['pulse'], target['chi'], target['traversal_bounds']
    require(p.dtype == np.uint8 and p.shape == (e,) and np.isin(p, (0, 1)).all(), 'nonbinary pulse')
    require(chi.dtype == np.float64 and chi.shape == (e,) and np.isfinite(chi).all() and np.all(chi >= 0), 'invalid chi')
    require(bounds.dtype == np.int64 and bounds.ndim == 2 and bounds.shape[1:] == (2,), 'invalid bounds')
    k = len(bounds)
    require(target['seams'].dtype == np.float64 and target['seams'].shape == (k,)
            and target['run_id'].dtype == np.int64 and target['run_id'].shape == (k,)
            and target['branch_id'].dtype.kind == 'U' and target['branch_id'].shape == (k,), 'invalid traversal provenance shape')
    require(meta['hybrid_mode'] == 'unavailable' and all(b in GEOMETRY for b in target['branch_id']),
            'G-NT1 requires separately frozen pure geometry artifacts; hybrid/mixed forbidden')
    require(len(meta['traversals']) == k, 'missing traversal ledger')
    union, training = np.zeros(e, bool), np.zeros(e, bool)
    previous = 0
    for j, (left, right) in enumerate(bounds.tolist()):
        require(0 <= left < right <= e and left >= previous, 'bounds outside/overlap/order')
        previous = right
        union[left:right] = True
        require(target['seams'][j] == left and p[left] == 1 and p[left:right].sum() == 1
                and abs(chi[left:right].sum() - 1) <= 1e-12, 'invalid pulse/unit mass')
        tr = meta['traversals'][j]
        require(tr['bounds'] == [left, right] and tr['branch_id'] == target['branch_id'][j]
                and tr['run_id'] == target['run_id'][j] and tr['teacher_sha256'] is None, 'traversal provenance mismatch')
        run_id = int(target['run_id'][j])
        require(0 <= run_id < len(derived['run_bounds']), 'invalid run ID')
        a, b = derived['run_bounds'][run_id]
        require(a <= left < right <= b, 'traversal crosses executable run')
        w, pulse = chi[left:right], p[left:right]
        training[left:right] = bool(np.sum(w * pulse) > 0 and np.sum(w * (1 - pulse)) > 0)
    require(np.array_equal(union, m) and not np.any(p[~m]) and not np.any(chi[~m]), 'target mask/payload mismatch')
    require(meta['status'] == ('certified' if k else 'rejected')
            and meta['source'] == ('geometry' if k else 'rejected'), 'target source/status mismatch')
    require(nt_ledger['K'] == k and nt_ledger['attempts'] == meta['attempts']
            and nt_ledger['source'] == meta['source'] and nb_ledger['clock_mode'] == nbmeta['summary']['clock_mode'], 'population/artifact ledger mismatch')
    mode = nbmeta['summary']['clock_mode']
    require(mode in ('pts', 'source_frame_only'), 'unknown clock mode')
    seconds = derived['map_seconds']
    require(seconds.shape == (n,), 'clock domain mismatch')
    require(not np.isinf(seconds).any(), 'infinite clock seconds')
    dt = np.diff(seconds)
    # Nonfinite clock payload is explicit missing time, not a nonfinite prediction.
    known = np.isfinite(dt) & (dt > 0) if mode == 'pts' else np.zeros(e, bool)
    reasons = list(meta.get('input_reasons', [])) + [str(a['reason']) for a in meta['attempts'] if a['status'] != 'certified']
    support = {}
    masks = {'observed': obs, 'response': response, 'target': m, 'training': training}
    for name, mask in masks.items():
        supported_frames = np.zeros(n, bool)
        supported_frames[:-1] |= mask
        supported_frames[1:] |= mask
        support[name] = {'edges': int(mask.sum()), 'source_edges': e,
                         'edge_fraction': float(mask.mean()) if e else 0.,
                         'frames_incident_to_edges': int(supported_frames.sum()), 'source_frames': n,
                         'frame_fraction': float(supported_frames.mean()),
                         'known_seconds': float(dt[mask & known].sum()) if known.any() else None,
                         'unknown_time_edges': int(np.sum(mask & ~known)),
                         'identity_supported': bool(mask.any())}
    return {'split': video['split'], 'media_key': video['media_key'], 'media_path': video['media_path'],
            'slot': slot, 'source_length': n, 'baseline': 'G-NT1', 'count': int(p.sum()),
            'abstain': k == 0, 'K': k, 'branch': '+'.join(sorted(set(target['branch_id'].tolist()))) or 'rejected',
            'clock_mode': mode, 'status': meta['status'], 'input_status': meta['input_status'],
            'reasons': reasons, 'attempts': meta['attempts'], 'traversals': meta['traversals'],
            'pulse_positions': np.flatnonzero(p).tolist(), 'support': support,
            '_masks': masks}


def unit_diagnostics(periods, pulses, masks, n, endpoint_policy='unresolved'):
    require(endpoint_policy in ('unresolved', 'e_minus_s_v1'), 'unknown endpoint policy')
    require(isinstance(periods, list), 'GT periods must be explicit list')
    validated = []
    for pair in periods:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2, 'GT period needs two raw endpoints')
        s, e = (number(v, 'period endpoint') for v in pair)
        validated.append((s, e))
    if endpoint_policy == 'unresolved':
        return {'status': 'unavailable_endpoint_unresolved', 'raw_periods': periods}
    require(all(0 <= s < e <= n for s, e in validated), 'GT period outside raw source domain')
    rows = []
    pulses = np.asarray(pulses, float)
    for s, e in validated:
        coverage = {}
        for name, mask in masks.items():
            overlap = np.maximum(0, np.minimum(np.arange(n - 1) + 1, e) - np.maximum(np.arange(n - 1), s))
            length = float(overlap[np.asarray(mask)].sum())
            # Classify geometric coverage without rounding away a positive gap.
            supported = np.asarray(mask)[overlap > 0]
            has_support = bool(supported.any())
            full = bool(e <= n - 1 and supported.all())
            coverage[name] = {'fraction': length / (e - s), 'fully_supported': full,
                              'partially_supported': has_support and not full, 'unsupported': not has_support}
        count = int(np.sum((pulses >= s) & (pulses < e)))
        rows.append({'period': [s, e], 'pulse_count': count, 'pulse_bin': '0' if count == 0 else '1' if count == 1 else '>=2', 'support': coverage})
    inside = np.zeros(len(pulses), bool)
    for s, e in validated:
        inside |= (pulses >= s) & (pulses < e)
    return {'status': 'available', 'endpoint_policy': endpoint_policy, 'periods': rows,
            'pulses_outside_gt_union': int((~inside).sum()), 'total_pulses': len(pulses),
            'outside_pulse_fraction': float((~inside).mean()) if len(pulses) else None}


def macro_metrics(rows):
    """Mean people within video then videos; repeated bootstrap videos stay copies."""
    if not rows:
        return {'status': 'undefined_empty_population', 'videos': 0, 'identities': 0}
    videos = defaultdict(list)
    fields = ('nmae', 'raw_mae', 'obo', 'signed_error', 'relative_signed_error', 'count_ratio')
    for r in rows:
        g = number(r['gt_count'], 'GT count', positive=True)
        c = number(r['count'], 'prediction count')
        d = c - g
        videos[r.get('_bootstrap_video', r['media_key'])].append((abs(d) / g, abs(d), float(abs(d) <= 1), d, d / g, c / g))
    values = np.asarray([np.mean(v, axis=0) for v in videos.values()])
    return {'status': 'available', 'reference': METRIC_REFERENCE, 'videos': len(videos),
            'identities': len(rows), **dict(zip(fields, map(float, values.mean(axis=0)), strict=True))}


def cluster_bootstrap(rows, replicates=0, seed=SEED):
    integer(replicates, 'bootstrap replicates')
    require(seed == SEED, 'bootstrap seed fixed by diagnostic protocol')
    if not replicates:
        return {'status': 'not_requested', 'replicates': 0, 'seed': seed}
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_component']].append(row)
    names = sorted(groups)
    require(names, 'empty bootstrap population')
    rng = np.random.default_rng(seed)
    draws, values = [], []
    for _ in range(replicates):
        selected = rng.integers(0, len(names), size=len(names))
        draws.append(selected.tolist())
        sample = []
        for copy, idx in enumerate(selected):
            sample.extend(dict(r, _bootstrap_video=(copy, r['media_key'])) for r in groups[names[idx]])
        values.append(macro_metrics(sample))
    fields = ('nmae', 'raw_mae', 'obo', 'signed_error', 'relative_signed_error', 'count_ratio')
    return {'status': 'available_proposed_cluster_percentile', 'seed': seed, 'replicates': replicates,
            'clusters': names, 'draw_indices': draws, 'replicate_metrics': values,
            'percentile95': {f: list(map(float, np.quantile([v[f] for v in values], [.025, .975]))) for f in fields}}


def distribution(values):
    a = np.asarray([v for v in values if v is not None], float)
    return {'known': len(a), 'unknown': len(values) - len(a), 'sum_known': float(a.sum()) if len(a) else None,
            'min': float(a.min()) if len(a) else None, 'median': float(np.median(a)) if len(a) else None,
            'p95': float(np.quantile(a, .95)) if len(a) else None, 'max': float(a.max()) if len(a) else None}


def summarize(rows):
    stages = ('observed', 'response', 'target', 'training')
    result = {'full_population': macro_metrics(rows), 'certified_conditional': macro_metrics([r for r in rows if not r['abstain']]),
              'identity_denominator': len(rows), 'abstained': sum(r['abstain'] for r in rows),
              'count_period_disagreements': sum(r['count_period_disagreement'] for r in rows), 'support_funnel': {}}
    for stage in stages:
        s = [r['support'][stage] for r in rows]
        all_periods = [p for r in rows for p in r['unit_diagnostics'].get('periods', [])]
        full = [p for p in all_periods if p['support'][stage]['fully_supported']]
        partial = [p for p in all_periods if p['support'][stage]['partially_supported']]
        result['support_funnel'][stage] = {
            'supported_identities': sum(x['identity_supported'] for x in s), 'identity_denominator': len(rows),
            'edges': sum(x['edges'] for x in s), 'source_edges': sum(x['source_edges'] for x in s),
            'frames_incident_to_edges': sum(x['frames_incident_to_edges'] for x in s), 'source_frames': sum(x['source_frames'] for x in s),
            'seconds': distribution([x['known_seconds'] for x in s]),
            'unknown_time_edges': sum(x['unknown_time_edges'] for x in s),
            'edge_fraction_distribution': distribution([x['edge_fraction'] for x in s]),
            'frame_fraction_distribution': distribution([x['frame_fraction'] for x in s]),
            'conditional_on_stage_supported': {
                'identities': sum(x['identity_supported'] for x in s),
                'edge_fraction_distribution': distribution([x['edge_fraction'] for x in s if x['identity_supported']]),
                'frame_fraction_distribution': distribution([x['frame_fraction'] for x in s if x['identity_supported']]),
                'seconds': distribution([x['known_seconds'] for x in s if x['identity_supported']])},
            'event_status': 'available' if all(r['unit_diagnostics']['status'] == 'available' for r in rows) else 'unavailable_endpoint_unresolved',
            'period_denominator': len(all_periods), 'fully_supported_periods': len(full), 'partially_supported_periods': len(partial),
            'unsupported_periods': len(all_periods) - len(full) - len(partial),
            'fully_supported_fraction': len(full) / len(all_periods) if all_periods else None,
            'partially_supported_fraction': len(partial) / len(all_periods) if all_periods else None,
            'all_period_pulse_bins': dict(Counter(p['pulse_bin'] for p in all_periods)),
            'fully_supported_pulse_bins': dict(Counter(p['pulse_bin'] for p in full)),
            'partially_supported_pulse_bins': dict(Counter(p['pulse_bin'] for p in partial))}
    events_available = all(r['unit_diagnostics']['status'] == 'available' for r in rows)
    outside = sum(r['unit_diagnostics'].get('pulses_outside_gt_union', 0) for r in rows)
    pulses = sum(r['count'] for r in rows)
    result['unit_totals'] = {'status': 'available' if events_available else 'unavailable_endpoint_unresolved',
                             'outside_pulses': outside if events_available else None,
                             'outside_pulse_fraction': outside / pulses if events_available and pulses else None,
                             'all_pulses': sum(r['count'] for r in rows),
                             'raw_period_counts': distribution([r['gt_period_count'] for r in rows])}
    result['K_distribution'] = distribution([r['K'] for r in rows])
    result['source_length_distribution'] = distribution([r['source_length'] for r in rows])
    result['error_distributions'] = {field: distribution([r[field] for r in rows])
                                   for field in ('count', 'gt_count', 'signed_error', 'absolute_error', 'relative_error', 'count_ratio')}
    result['status_counts'] = dict(Counter(r.get('status', 'fixture_unspecified') for r in rows))
    result['input_status_counts'] = dict(Counter(r.get('input_status', 'fixture_unspecified') for r in rows))
    return result


def evaluate_rows(rows, *, endpoint_policy='unresolved', bootstrap_replicates=0):
    """Internal fixture-friendly numerical entry; production census enforced in CLI."""
    require(bool(rows), 'empty evaluation population')
    for r in rows:
        g = number(r['gt_count'], 'GT count', positive=True)
        c = number(r['count'], 'prediction count')
        r['gt_period_count'] = len(r['gt_periods'])
        r['count_period_disagreement'] = g != len(r['gt_periods'])
        r['signed_error'], r['absolute_error'] = c - g, abs(c - g)
        r['relative_error'], r['count_ratio'], r['obo'] = abs(c - g) / g, c / g, abs(c - g) <= 1
        r['unit_diagnostics'] = unit_diagnostics(r['gt_periods'], r['pulse_positions'], r.pop('_masks'), r['source_length'], endpoint_policy)
    report = {'schema': SCHEMA, 'metric_reference': METRIC_REFERENCE, 'official_metric_status': 'not_certified',
              'period_ap': 'unavailable_no_legal_period_decoder', 'endpoint_policy': endpoint_policy,
              'target_quality_acceptance': 'not_evaluated_no_tolerance_frozen', 'splits': {}, 'groups': {}}
    for split in ('train', 'val'):
        selected = [r for r in rows if r['split'] == split]
        if selected:
            report['splits'][split] = summarize(selected)
            report['splits'][split]['bootstrap'] = cluster_bootstrap(selected, bootstrap_replicates)
    for key in ('split', 'source_component', 'branch', 'clock_mode', 'reason'):
        groups = defaultdict(list)
        for r in rows:
            for v in sorted(set(r['reasons'])) or ['none'] if key == 'reason' else [r[key]]:
                groups[str(v)].append(r)
        report['groups'][key] = {}
        for name, group in sorted(groups.items()):
            report['groups'][key][name] = summarize(group)
            report['groups'][key][name]['by_split'] = {
                split: summarize([r for r in group if r['split'] == split])
                for split in sorted({r['split'] for r in group})}
    return report


def run(config, *, _expected=EXPECTED):
    """All paths and protocol choices explicit; no debug caps in production config."""
    fields = {'protocol', 'inputs', 'annotation_root', 'bridge_root', 'target_root', 'output',
              'count_field', 'periods_field', 'endpoint_policy', 'bootstrap_replicates'}
    require(set(config) == fields and config['protocol'] == SCHEMA, 'exact frozen diagnostic config required')
    require(config['endpoint_policy'] in ('unresolved', 'e_minus_s_v1'), 'explicit endpoint_policy required')
    integer(config['bootstrap_replicates'], 'bootstrap replicates')
    paths = {k: Path(config[k]).resolve() for k in ('annotation_root', 'bridge_root', 'target_root', 'output')}
    inputs = config['inputs']
    require(set(inputs) == {'census', 'provenance', 'source_components', 'nb1_manifest', 'nt1_manifest'}, 'exact input manifests required')
    receipt = {}
    for item in inputs.values():
        require(set(item) == {'path', 'sha256'}, 'input path and frozen SHA required')
        path = Path(item['path']).resolve()
        require(digest(path) == item['sha256'], f'frozen input hash mismatch: {path}')
        receipt[str(path)] = item['sha256']
    import os
    if os.name == 'nt':
        require(all(p.drive.upper() == 'E:' for p in [*paths.values(), *(Path(v['path']).resolve() for v in inputs.values())]), 'E-only paths required')
    out = paths['output']
    require(not out.exists(), 'output must be new; previous raw evidence retained')
    for p in list(paths.values())[:-1]:
        require(out != p and not out.is_relative_to(p) and not p.is_relative_to(out), 'output must be separate from all input roots')
    census_path = inputs['census']['path']
    census = load_census(census_path, _expected=_expected)
    members, aliases = source_membership(census, read_json(inputs['source_components']['path']))
    if _expected == EXPECTED:
        require(len(aliases) == 4 and all(all(r['split'] == 'train' for r in g) for g in aliases), 'frozen four train alias groups mismatch')
    nbman, nbl = _ledger(inputs['nb1_manifest']['path'], census, {'validated_population'})
    ntman, ntl = _ledger(inputs['nt1_manifest']['path'], census, {'complete_population'})
    require(nbman['binding']['census_sha256'] == digest(census_path)
            and ntman['binding']['census_sha256'] == digest(census_path), 'population manifest census SHA mismatch')
    provenance = read_json(inputs['provenance']['path'])
    require(provenance['schema'] == 'temporac.bbox-provenance.v1', 'invalid original mapping provenance')
    require(provenance.get('projection_sha256') == digest(census_path), 'original provenance projection/census SHA mismatch')
    mapping = {r['media_key']: r for r in provenance['records']}
    require(len(mapping) == len(provenance['records']) == len(census)
            and set(mapping) == {r['media_key'] for r in census}, 'incomplete original object mapping census')
    rows = []
    # Validate complete artifacts before opening any GT label file.
    for v in census:
        for p in v['persons']:
            identity = (v['split'], v['media_key'], p['slot'])
            row = load_identity(paths['bridge_root'], paths['target_root'], v, p['slot'], nbl[identity], ntl[identity], receipt,
                                nb_binding=nbman['binding'], nt_binding=ntman['binding'])
            row['source_component'] = members[v['media_key']]
            rows.append(row)
    by_video = defaultdict(list)
    for row in rows:
        by_video[row['media_key']].append(row)
    for v in census:
        rel = PurePosixPath(v['media_path']).with_suffix('.json')
        path = (paths['annotation_root'] / Path(*rel.parts)).resolve(strict=True)
        require(path.is_relative_to(paths['annotation_root'] / v['split']), 'annotation path escapes train/val')
        prov = mapping[v['media_key']]
        require(digest(path) == prov['annotation_sha256'], 'original annotation SHA mismatch')
        receipt[str(path)] = digest(path)
        annotation = read_json(path)
        objects = {k for k in annotation if k.startswith('object')}
        object_map = prov['source_object_to_slot']
        require(set(object_map) == objects and len(object_map) == len(v['persons'])
                and sorted(object_map.values()) == list(range(len(v['persons']))), 'missing/duplicate original object mapping')
        require(annotation['length'] == v['length'], 'original annotation length mismatch')
        slot_object = {slot: obj for obj, slot in object_map.items()}
        for row in by_video[v['media_key']]:
            obj = slot_object[row['slot']]
            label = annotation[obj]
            row.update(original_object=obj, gt_count=label[config['count_field']], gt_periods=label[config['periods_field']])
    report = evaluate_rows(rows, endpoint_policy=config['endpoint_policy'], bootstrap_replicates=config['bootstrap_replicates'])
    report['same_media_alias_groups'] = []
    for group in aliases:
        keys = [v['media_key'] for v in group]
        raw = [{k: r[k] for k in ('media_key', 'slot', 'original_object', 'gt_count', 'gt_periods', 'count')}
               for key in keys for r in by_video[key]]
        labels = [[(r['original_object'], r['gt_count'], r['gt_periods']) for r in by_video[key]] for key in keys]
        report['same_media_alias_groups'].append({'media_keys': keys, 'label_conflict': any(x != labels[0] for x in labels[1:]), 'rows': raw})
    require(all(digest(p) == sha for p, sha in receipt.items()), 'input changed during evaluation')
    report['receipt'] = {'config_sha256': json_digest(config), 'inputs_sha256': receipt,
                         'code_sha256': digest(__file__), 'gt_access': 'train_and_val_evaluator_only',
                         'producer_invoked': False, 'natural_labels_feed_producer': False}
    cli = Path(__file__).resolve().parents[3] / 'scripts/evaluate_temporac_native_targets.py'
    report['receipt']['cli_sha256'] = digest(cli)
    out.mkdir(parents=True)
    raw_path = out / 'identity_rows.jsonl'
    raw_path.write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
    report['receipt']['identity_rows_sha256'] = digest(raw_path)
    report_path = out / 'diagnostics.json'
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'receipt.json').write_text(json.dumps({'report_sha256': digest(report_path), 'rows_sha256': digest(raw_path),
                                               'config': config, 'inputs_sha256': receipt}, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    return report
