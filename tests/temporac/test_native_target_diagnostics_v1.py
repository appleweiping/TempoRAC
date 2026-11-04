"""Synthetic CPU counterexamples; no natural annotation/model/producer imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pams.temporac import native_target_diagnostics_v1 as d
from pams.temporac.native_bridge_v1 import NativeFeatureV1, preprocess_native


def sample(key='a', *, count=1, gt=2, n=7, pulses=(0,), periods=None, component='A', split='train'):
    mask = np.zeros(n - 1, bool)
    mask[:4] = True
    masks = {'observed': np.ones(n - 1, bool), 'response': np.ones(n - 1, bool),
             'target': mask, 'training': mask}
    support = {name: {'edges': int(m.sum()), 'source_edges': n - 1, 'edge_fraction': float(m.mean()),
                       'frames_incident_to_edges': int(m.sum()) + int(m.any()), 'source_frames': n,
                       'frame_fraction': (int(m.sum()) + int(m.any())) / n,
                       'known_seconds': None, 'unknown_time_edges': int(m.sum()),
                       'identity_supported': bool(m.any())} for name, m in masks.items()}
    return {'media_key': key, 'split': split, 'source_component': component,
            'count': count, 'gt_count': gt, 'source_length': n, 'pulse_positions': list(pulses),
            'gt_periods': [[0, 2], [2, 6]] if periods is None else periods,
            '_masks': masks, 'support': support, 'abstain': count == 0, 'K': count,
            'branch': 'v12_geometry' if count else 'rejected', 'clock_mode': 'source_frame_only',
            'reasons': ['synthetic_rejection'] if count == 0 else []}


def test_unequal_people_video_macro_not_identity_pool():
    rows = [sample('one', count=0, gt=10), sample('many', count=2, gt=2),
            sample('many', count=2, gt=2), sample('many', count=2, gt=2)]
    score = d.macro_metrics(rows)
    assert score['nmae'] == .5  # identity pooling would .25
    assert score['raw_mae'] == 5
    assert score['obo'] == .5
    assert score['signed_error'] == -5
    assert score['relative_signed_error'] == -.5
    assert score['count_ratio'] == .5


def test_partial_count_not_coverage_inflated_and_rejected_rows_retained():
    rows = [sample('partial', count=1, gt=4), sample('failed', count=0, gt=4, pulses=())]
    report = d.evaluate_rows(rows, endpoint_policy='e_minus_s_v1')
    full = report['splits']['train']['full_population']
    conditional = report['splits']['train']['certified_conditional']
    assert full['nmae'] == .875 and conditional['nmae'] == .75
    assert full['identities'] == 2 and conditional['identities'] == 1
    assert rows[0]['count'] == 1 and rows[1]['abstain']
    assert rows[0]['count_period_disagreement']


def test_all_failed_conditional_is_undefined_not_perfect():
    rows = [sample('a', count=0, gt=10, pulses=()), sample('b', count=0, gt=1, pulses=())]
    report = d.evaluate_rows(rows)
    score = report['splits']['train']
    assert score['full_population']['nmae'] == 1
    assert score['full_population']['obo'] == .5  # standard OBO does not imply identity success
    assert score['certified_conditional']['status'] == 'undefined_empty_population'
    assert score['unit_totals']['outside_pulses'] is None


def test_units_boundary_zero_N_pulse_bins_support_and_outside_union():
    masks = {name: np.array([1, 1, 1, 0, 0, 0], bool) for name in ('observed', 'response', 'target', 'training')}
    units = d.unit_diagnostics([[0, 1], [1, 4], [4, 7]], [0, 1, 2, 6], masks, 7, 'e_minus_s_v1')
    assert [p['pulse_bin'] for p in units['periods']] == ['1', '>=2', '1']
    assert units['periods'][0]['support']['target']['fully_supported']
    assert units['periods'][1]['support']['target']['partially_supported']
    assert units['periods'][2]['support']['target']['unsupported']
    last = d.unit_diagnostics([[0, 7]], [0, 6], {k: np.ones(6, bool) for k in masks}, 7, 'e_minus_s_v1')
    assert last['periods'][0]['support']['target']['fraction'] == 6 / 7
    assert not last['periods'][0]['support']['target']['fully_supported']  # no fake N sample
    outside = d.unit_diagnostics([[1, 2], [4, 5]], [0, 1, 2, 4, 6], masks, 7, 'e_minus_s_v1')
    assert outside['pulses_outside_gt_union'] == 3 and outside['outside_pulse_fraction'] == .6
    assert d.unit_diagnostics([[2, 3]], [], masks, 7, 'e_minus_s_v1')['periods'][0]['pulse_bin'] == '0'


def test_unresolved_endpoint_never_guesses_inclusive_semantic():
    masks = {k: np.ones(6, bool) for k in ('observed', 'response', 'target', 'training')}
    assert d.unit_diagnostics([[0, 0]], [0], masks, 7)['status'] == 'unavailable_endpoint_unresolved'
    with pytest.raises(d.DiagnosticError):
        d.unit_diagnostics([[0, 0]], [0], masks, 7, 'e_minus_s_v1')


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True, None])
def test_invalid_GT_count_fails_whole_population(bad):
    with pytest.raises(d.DiagnosticError):
        d.evaluate_rows([sample('good'), sample('invalid', gt=bad)])


def test_empty_GT_periods_retained_disagreement_and_pulses_outside():
    rows = [sample(periods=[])]
    result = d.evaluate_rows(rows, endpoint_policy='e_minus_s_v1')
    assert rows[0]['count_period_disagreement']
    assert rows[0]['unit_diagnostics']['pulses_outside_gt_union'] == 1
    assert result['splits']['train']['support_funnel']['target']['fully_supported_fraction'] is None


def test_empty_population_cannot_be_evaluated_as_success():
    with pytest.raises(d.DiagnosticError, match='empty'):
        d.evaluate_rows([])


def test_known_zero_duration_distinct_from_unknown_clock(tmp_path):
    cfg, expected = fixture_bundle(tmp_path)
    census = d.load_census(cfg['inputs']['census']['path'], _expected=expected)
    v = census[0]
    base = Path(cfg['bridge_root']) / 'identities' / 'train' / f"{v['media_key']}_0"
    feature_arrays = d._npz(base.with_suffix('.npz'))
    feature_arrays.update(pts=np.arange(17, dtype=np.int64), pts_present=np.ones(17, bool),
                          time_base=np.array([1, 10], np.int64))
    feature = NativeFeatureV1(**feature_arrays)
    np.savez(base.with_suffix('.npz'), **feature_arrays)
    np.savez(base.with_suffix('.derived.npz'), **derived_arrays(preprocess_native(feature)))
    meta = d.read_json(base.with_suffix('.json'))
    meta['summary']['clock_mode'] = 'pts'
    meta['feature_sha256'] = d.digest(base.with_suffix('.npz'))
    meta['derived_sha256'] = d.digest(base.with_suffix('.derived.npz'))
    write_json(base.with_suffix('.json'), meta)
    target = Path(cfg['target_root']) / 'identities' / 'train' / f"{v['media_key']}_0.npz"
    target_meta = d.read_json(target.with_suffix('.json'))
    target_meta['input_sha256'] = meta['feature_sha256']
    write_json(target.with_suffix('.json'), target_meta)
    nbman = d.read_json(cfg['inputs']['nb1_manifest']['path'])
    ntman = d.read_json(cfg['inputs']['nt1_manifest']['path'])
    nbrow = dict(nbman['records'][0], clock_mode='pts')
    row = d.load_identity(cfg['bridge_root'], cfg['target_root'], v, 0, nbrow, ntman['records'][0], {})
    assert row['support']['target']['known_seconds'] == pytest.approx(.6)
    assert row['support']['target']['unknown_time_edges'] == 0
    arrays = d._npz(target)
    arrays.update(target_valid=np.zeros(16, bool), pulse=np.zeros(16, np.uint8), chi=np.zeros(16, np.float64),
                  traversal_bounds=np.empty((0, 2), np.int64), seams=np.empty(0, np.float64),
                  run_id=np.empty(0, np.int64), branch_id=np.array([], dtype='<U16'))
    np.savez(target, **arrays)
    tmeta = d.read_json(target.with_suffix('.json'))
    tmeta.update(npz_sha256=d.digest(target), traversals=[], status='rejected', source='rejected')
    write_json(target.with_suffix('.json'), tmeta)
    ntrow = dict(ntman['records'][0], K=0, source='rejected')
    rejected = d.load_identity(cfg['bridge_root'], cfg['target_root'], v, 0, nbrow, ntrow, {})
    assert rejected['count'] == 0 and rejected['abstain']
    assert rejected['support']['target']['known_seconds'] == 0.
    assert rejected['support']['target']['unknown_time_edges'] == 0


def test_cluster_resampling_whole_sources_deterministic_unequal_video_clusters():
    rows = [sample('a', count=0, gt=2, component='A'), sample('b', count=0, gt=2, component='A'),
            sample('c', count=2, gt=2, component='B')]
    boot = d.cluster_bootstrap(rows, 40)
    assert boot == d.cluster_bootstrap(rows, 40)
    assert boot['percentile95']['nmae'] == [0., 1.]
    for draw, score in zip(boot['draw_indices'], boot['replicate_metrics'], strict=True):
        assert score['videos'] == sum(2 if i == 0 else 1 for i in draw)
        expected = sum(2 if i == 0 else 0 for i in draw) / score['videos']
        assert score['nmae'] == expected


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')


def derived_arrays(identity):
    """Actual NB1 synthetic CPU arrays, using the existing writer's field set."""
    g = identity.geometry_features
    arrays = {'response_input': identity.response_input, 'joint_usable': identity.joint_mask,
              'frame_derived_valid': identity.frame_valid, 'observed_edge_support': identity.observed_edge_support,
              'response_edge_support': identity.response_edge_support, 'edge_lengths': g.edge_lengths,
              'direction': g.direction, 'direction_mask': g.direction_element_mask,
              'run_bounds': identity.run_bounds, 'short_runs': identity.short_runs}
    arrays.update({'map_' + k: v for k, v in identity.mapping.items()})
    return arrays


def fixture_bundle(tmp_path, *, aliases=False):
    census, prov, components, nbrows, ntrows = [], [], [], [], []
    nb_binding = {'census_sha256': '', 'source': {'synthetic': '1' * 64}}
    nt_binding = {'census_sha256': '', 'producer': {'synthetic': '2' * 64}, 'config_sha256': '3' * 64,
                  'availability': {'hybrid_mode': 'unavailable'}}
    for index, split in enumerate(('train', 'val')):
        media = f'{split}/video{index}/video.mp4'
        key = hashlib.sha256(media.encode()).hexdigest()
        v = {'media_key': key, 'media_path': media, 'split': split, 'length': 17,
             'persons': [{'slot': 0}], 'media_sha256': ('9' if aliases else str(index + 1)) * 64}
        census.append(v)
        ann = tmp_path / 'annotations' / Path(media).with_suffix('.json')
        write_json(ann, {'length': 17, 'object4': {'count': 2 + index, 'period': [[0, 2], [2, 17]]}})
        prov.append({'media_key': key, 'annotation_sha256': d.digest(ann), 'source_object_to_slot': {'object4': 0}})
        components.append({'id': f'C{index}', 'members': [f'{split}/video{index}']})
        nbrows.append({'media_key': key, 'split': split, 'slot': 0, 'clock_mode': 'source_frame_only'})
        ntrows.append({'media_key': key, 'split': split, 'slot': 0, 'K': 2, 'source': 'geometry', 'attempts': []})
    cp = tmp_path / 'census.jsonl'
    cp.write_text(''.join(json.dumps(v) + '\n' for v in census), encoding='utf-8')
    nb_binding['census_sha256'] = nt_binding['census_sha256'] = d.digest(cp)
    for v in census:
        key, split = v['media_key'], v['split']
        nb = tmp_path / 'bridge' / 'identities' / split / f'{key}_0'
        nt = tmp_path / 'target' / 'identities' / split / f'{key}_0'
        nb.parent.mkdir(parents=True, exist_ok=True)
        nt.parent.mkdir(parents=True, exist_ok=True)
        raw = np.ones((17, 17, 3), np.float32)
        raw[:, :, 0] = np.arange(17) % 5
        raw[:, :, 1] = np.arange(17) // 5
        raw[:, :, 0] += np.arange(17, dtype=np.float32)[:, None] * .01
        feature = NativeFeatureV1(np.frombuffer(bytes.fromhex(key), np.uint8), np.array([0], np.int64),
                    np.array([17], np.int64), np.arange(17, dtype=np.int64), raw, np.ones((17, 17), bool),
                    np.ones(17, bool), np.zeros(17, np.int64), np.zeros(17, bool), np.zeros(2, np.int64))
        identity = preprocess_native(feature)
        assert identity.response_edge_support.all()
        np.savez(nb.with_suffix('.npz'), **feature.as_arrays())
        np.savez(nb.with_suffix('.derived.npz'), **derived_arrays(identity))
        binding = dict(nb_binding, raw_npz_sha256='4' * 64, raw_json_sha256='5' * 64, slot=0)
        nbmeta = {'schema': 'temporac.native-learner.v1', 'feature_sha256': d.digest(nb.with_suffix('.npz')),
                  'derived_sha256': d.digest(nb.with_suffix('.derived.npz')), 'binding': binding,
                  'summary': {'clock_mode': 'source_frame_only'}}
        write_json(nb.with_suffix('.json'), nbmeta)
        np.savez(nt.with_suffix('.npz'), target_valid=np.arange(16) < 6,
                 pulse=np.array([1, 0, 0, 1, 0, 0] + [0] * 10, np.uint8),
                 chi=np.array([1/3] * 6 + [0.] * 10, np.float64), traversal_bounds=np.array([[0, 3], [3, 6]], np.int64),
                 seams=np.array([0., 3.]), run_id=np.array([0, 0], np.int64), branch_id=np.array(['v12_geometry', 'v15_geometry']))
        write_json(nt.with_suffix('.json'), {'schema': 'temporac.native-target.nt1', 'npz_sha256': d.digest(nt.with_suffix('.npz')),
            'source_length': 17, 'slot': 0, 'opaque_key': key, 'input_sha256': nbmeta['feature_sha256'],
            'bridge_sha256': d.json_digest(binding), 'raw_sha256': d.json_digest({k: v for k, v in binding.items() if k.startswith('raw_')}),
            'producer_sha256': d.json_digest(nt_binding['producer']), 'config_sha256': nt_binding['config_sha256'],
            'availability_sha256': d.json_digest(nt_binding['availability']), 'hybrid_mode': 'unavailable',
            'status': 'certified', 'source': 'geometry', 'input_status': 'complete', 'attempts': [], 'input_reasons': [],
            'traversals': [{'bounds': [a, b], 'run_id': 0, 'branch_id': branch, 'teacher_sha256': None}
                           for a, b, branch in ((0, 3, 'v12_geometry'), (3, 6, 'v15_geometry'))]})
    if aliases:
        components = [{'id': 'shared', 'members': [c['members'][0] for c in components]}]
    files = {'census': cp, 'provenance': tmp_path / 'provenance.json', 'source_components': tmp_path / 'components.json',
             'nb1_manifest': tmp_path / 'nb1.json', 'nt1_manifest': tmp_path / 'nt1.json'}
    write_json(files['provenance'], {'schema': 'temporac.bbox-provenance.v1', 'projection_sha256': d.digest(cp), 'records': prov})
    write_json(files['source_components'], components)
    for name, status, records, binding in (('nb1_manifest', 'validated_population', nbrows, nb_binding),
                                          ('nt1_manifest', 'complete_population', ntrows, nt_binding)):
        write_json(files[name], {'status': status, 'records': records, 'videos': 2, 'identities': 2, 'failures': [], 'binding': binding})
    cfg = {'protocol': d.SCHEMA, 'inputs': {k: {'path': str(p), 'sha256': d.digest(p)} for k, p in files.items()},
           'annotation_root': str(tmp_path / 'annotations'), 'bridge_root': str(tmp_path / 'bridge'),
           'target_root': str(tmp_path / 'target'), 'output': str(tmp_path / 'results'), 'count_field': 'count',
           'periods_field': 'period', 'endpoint_policy': 'e_minus_s_v1', 'bootstrap_replicates': 5}
    expected = {s: {'videos': 1, 'persons': 1} for s in ('train', 'val')}
    return cfg, expected


def test_full_isolated_fixture_outputs_receipts_alias_conflict_and_unknown_seconds(tmp_path):
    cfg, expected = fixture_bundle(tmp_path, aliases=True)
    report = d.run(cfg, _expected=expected)
    assert report['same_media_alias_groups'][0]['label_conflict']
    assert len(report['same_media_alias_groups'][0]['rows']) == 2
    assert report['splits']['val']['count_period_disagreements'] == 1
    assert report['splits']['train']['support_funnel']['target']['seconds']['unknown'] == 1
    raw = Path(cfg['output']) / 'identity_rows.jsonl'
    assert d.digest(raw) == report['receipt']['identity_rows_sha256']
    receipt = d.read_json(Path(cfg['output']) / 'receipt.json')
    assert receipt['report_sha256'] == d.digest(Path(cfg['output']) / 'diagnostics.json')
    assert len(raw.read_text().splitlines()) == 2
    assert not any(name in sys.modules for name in ('torch', 'scipy', 'pams.temporac.native_target_nt1'))
    with pytest.raises(d.DiagnosticError, match='new'):
        d.run(cfg, _expected=expected)


def test_production_cli_cannot_override_full_population(tmp_path):
    cfg, _ = fixture_bundle(tmp_path)
    with pytest.raises(d.DiagnosticError, match='full census'):
        d.run(cfg)
    cfg['debug_cap'] = 1
    with pytest.raises(d.DiagnosticError, match='exact frozen'):
        d.run(cfg)
    spec = importlib.util.spec_from_file_location('diagnostic_cli', Path(__file__).resolve().parents[2] / 'scripts/evaluate_temporac_native_targets.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(SystemExit) as error:
        cli.main(['--expected', '1'])
    assert error.value.code == 2


@pytest.mark.parametrize('mutation', ['missing_npz', 'NaN_pulse', 'hybrid', 'wrong_mapping', 'annotation_changed', 'missing_ledger'])
def test_integrity_failure_never_filters_rows_or_emits_results(tmp_path, mutation):
    cfg, expected = fixture_bundle(tmp_path)
    cp = d.load_census(cfg['inputs']['census']['path'], _expected=expected)[0]
    target = Path(cfg['target_root']) / 'identities' / 'train' / f"{cp['media_key']}_0.npz"
    if mutation == 'missing_npz':
        target.unlink()
    elif mutation in ('NaN_pulse', 'hybrid'):
        arrays = d._npz(target)
        arrays['pulse'] = np.full(6, np.nan) if mutation == 'NaN_pulse' else arrays['pulse']
        if mutation == 'hybrid':
            arrays['branch_id'] = np.array(['v9_hybrid', 'v15_geometry'])
        np.savez(target, **arrays)
        meta = d.read_json(target.with_suffix('.json'))
        meta['npz_sha256'] = d.digest(target)
        write_json(target.with_suffix('.json'), meta)
    elif mutation == 'annotation_changed':
        write_json(Path(cfg['annotation_root']) / Path(cp['media_path']).with_suffix('.json'), {'length': 7})
    else:
        which = 'provenance' if mutation == 'wrong_mapping' else 'nt1_manifest'
        item = cfg['inputs'][which]
        data = d.read_json(item['path'])
        if mutation == 'wrong_mapping':
            data['records'][0]['source_object_to_slot'] = {'object99': 0}
        else:
            data['records'].pop()
        write_json(Path(item['path']), data)
        item['sha256'] = d.digest(item['path'])
    with pytest.raises((ValueError, OSError)):
        d.run(cfg, _expected=expected)
    assert not Path(cfg['output']).exists()


def test_source_mapping_missing_and_alias_cluster_split_fail(tmp_path):
    cfg, expected = fixture_bundle(tmp_path, aliases=True)
    census = d.load_census(cfg['inputs']['census']['path'], _expected=expected)
    with pytest.raises(d.DiagnosticError, match='missing source'):
        d.source_membership(census, [])
    components = [{'id': f'C{i}', 'members': [str(Path(v['media_path']).parent).replace('\\', '/')]} for i, v in enumerate(census)]
    with pytest.raises(d.DiagnosticError, match='aliases split'):
        d.source_membership(census, components)


@pytest.mark.parametrize('binding', ['wrong', 'missing'])
def test_original_projection_binding_fails_before_any_GT_or_metrics(tmp_path, monkeypatch, binding):
    cfg, expected = fixture_bundle(tmp_path)
    item = cfg['inputs']['provenance']
    provenance = d.read_json(item['path'])
    if binding == 'wrong':
        provenance['projection_sha256'] = '0' * 64
    else:
        del provenance['projection_sha256']
    write_json(Path(item['path']), provenance)
    item['sha256'] = d.digest(item['path'])  # self-consistent own hash is insufficient
    reader = d.read_json

    def no_GT(path):
        assert not Path(path).is_relative_to(Path(cfg['annotation_root']))
        return reader(path)

    def no_metrics(*args, **kwargs):
        pytest.fail('metrics must not run on mismatched provenance')

    monkeypatch.setattr(d, 'read_json', no_GT)
    monkeypatch.setattr(d, 'macro_metrics', no_metrics)
    with pytest.raises(d.DiagnosticError, match='projection/census SHA'):
        d.run(cfg, _expected=expected)
    assert not Path(cfg['output']).exists()


@pytest.mark.parametrize('mutation', ['feature_missing', 'feature_extra', 'feature_shape', 'feature_dtype',
                                    'feature_observation', 'feature_PTS', 'derived_missing', 'derived_extra',
                                    'derived_shape', 'derived_dtype', 'derived_clock', 'derived_runs'])
def test_complete_NB1_schema_and_domains_reject_invalid_artifact(tmp_path, mutation):
    cfg, expected = fixture_bundle(tmp_path)
    v = d.load_census(cfg['inputs']['census']['path'], _expected=expected)[0]
    nb = Path(cfg['bridge_root']) / 'identities' / 'train' / f"{v['media_key']}_0"
    is_feature = mutation.startswith('feature')
    path = nb.with_suffix('.npz' if is_feature else '.derived.npz')
    arrays = d._npz(path)
    if mutation == 'feature_missing':
        del arrays['xyconf_raw']
    elif mutation == 'feature_extra':
        arrays['unexpected'] = np.zeros(1)
    elif mutation == 'feature_shape':
        arrays['xyconf_raw'] = arrays['xyconf_raw'][:-1]
    elif mutation == 'feature_dtype':
        arrays['slot'] = arrays['slot'].astype(np.int32)
    elif mutation == 'feature_observation':
        arrays['xyconf_raw'][0, 0, 0] = np.nan
    elif mutation == 'feature_PTS':
        arrays['pts_present'][0] = True  # invalid zero rational time base
    elif mutation == 'derived_missing':
        del arrays['direction']
    elif mutation == 'derived_extra':
        arrays['unexpected'] = np.zeros(1)
    elif mutation == 'derived_shape':
        arrays['response_input'] = arrays['response_input'][:, :-1]
    elif mutation == 'derived_dtype':
        arrays['direction'] = arrays['direction'].astype(np.float32)
    elif mutation == 'derived_clock':
        arrays['map_seconds'][0] = 0.  # source-frame-only has no seconds
    else:
        arrays['run_bounds'][0, 1] = 17  # outside actual edge domain
    np.savez(path, **arrays)
    nbmeta = d.read_json(nb.with_suffix('.json'))
    nbmeta['feature_sha256' if is_feature else 'derived_sha256'] = d.digest(path)
    write_json(nb.with_suffix('.json'), nbmeta)
    if is_feature:
        target = Path(cfg['target_root']) / 'identities' / 'train' / f"{v['media_key']}_0.json"
        tmeta = d.read_json(target)
        tmeta['input_sha256'] = nbmeta['feature_sha256']
        write_json(target, tmeta)
    with pytest.raises((ValueError, KeyError)):
        d.run(cfg, _expected=expected)
    assert not Path(cfg['output']).exists()
