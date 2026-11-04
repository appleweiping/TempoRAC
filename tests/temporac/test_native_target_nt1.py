"""Synthetic NT1 CPU engineering evidence; no natural GT or teacher."""
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pams.temporac.native_target_nt1 as nt
from pams.temporac.certify import CertificateTrack
from pams.temporac.certify_target_adaptive_v6 import certify_target_adaptive_v6
from pams.temporac.certify_target_adaptive_v7 import certify_target_adaptive_v7
from pams.temporac.full_chain_landmarks_candidate_v1 import full_chain_integer_landmarks
from pams.temporac.full_chain_landmarks_candidate_v2 import full_chain_integer_landmarks_v2
from pams.temporac.multi_cycle_dtw_template_candidate_v1 import multi_cycle_dtw_reconstruction
from pams.temporac.phase_closure_candidate_v1 import close_phase_on_landmarks


def identity(n=216, runs=None, zero=False):
    t = np.arange(n) % 20
    angle = 2 * np.pi * t / 20
    geometry = np.tile(np.column_stack((np.cos(angle), np.sin(angle))), (1, 33)) * .2
    if zero:
        geometry[:] = geometry[0]
    teacher = np.column_stack((geometry, geometry, np.full((n, 17), .9), np.zeros((n, 66))))
    runs = np.asarray([[0, n - 1]] if runs is None else runs, np.int32)
    support = np.zeros(n - 1, bool)
    for left, r in runs:
        support[left:r] = True
    gf = SimpleNamespace(geometry=geometry, teacher_input=teacher,
                         coordinate_mask=np.ones((n, 66), bool), direction_element_mask=np.ones((n, 33), bool))
    feature = SimpleNamespace(source_length=np.array([n]), slot=np.array([0]), opaque_key=np.arange(32, dtype=np.uint8))
    return SimpleNamespace(feature=feature, geometry_features=gf, joint_mask=np.ones((n, 17), bool),
                           response_edge_support=support, run_bounds=runs)


def policy():
    return nt.NativeTargetPolicy(**{k: str(i) * 64 for i, k in enumerate(nt.NativeTargetPolicy.__dataclass_fields__)
                                  if k.endswith('sha256')})


@pytest.mark.parametrize('n', [216, 900])
def test_native_full_domain_and_start_pulses(n):
    i = identity(n)
    target = nt.produce_native_target(i, policy())
    assert target.target_valid.shape == (n - 1,)
    assert len(target.traversal_bounds) >= 2
    for left, r in target.traversal_bounds:
        assert target.pulse[left] == 1 and target.pulse[left:r].sum() == 1
        assert abs(target.chi[left:r].sum() - 1) < 1e-12
    assert target.metadata['source'] == 'geometry'
    assert all(x['teacher_sha256'] is None for x in target.metadata['traversals'])
    assert all(x['branch_id'] == 'v12_geometry' for x in target.metadata['attempts'])
    assert nt.geometry_count(target)['count'] == len(target.traversal_bounds)


@pytest.mark.parametrize('branch,selector,certifier', [
    ('v12_geometry', full_chain_integer_landmarks, certify_target_adaptive_v6),
    ('v15_geometry', full_chain_integer_landmarks_v2, certify_target_adaptive_v7)])
def test_exact_original_numerical_parity(branch, selector, certifier):
    i = identity()
    g = i.geometry_features
    continuous = g.teacher_input[:, :149]
    mask = np.ones_like(continuous, np.uint8)
    result, track = nt.numerical_branch(g.geometry, continuous, mask, g.coordinate_mask,
                                       i.response_edge_support, branch)
    bounds = np.array([[0, 216]], np.int32)
    points = selector(g.geometry, bounds).landmarks
    reconstruction, _ = multi_cycle_dtw_reconstruction(continuous, mask, g.geometry, points)
    phase = close_phase_on_landmarks(np.tile([[1., 0.]], (216, 1)), points, geometry=g.geometry)
    direct = certifier(CertificateTrack(g.geometry, continuous, mask, phase, reconstruction, bounds), 'natural')
    assert direct.certified and result.certified
    for name in ('phase', 'reconstruction'):
        np.testing.assert_array_equal(getattr(track, name), locals()[name])
    for name in ('pulse', 'chi', 'target_mask', 'traversal_bounds', 'seams', 'canonical_phase', 'winding', 'reasons'):
        np.testing.assert_array_equal(getattr(result, name), getattr(direct, name))


def test_multirun_preserves_success_and_unavailable_fallback(monkeypatch):
    original = nt.numerical_branch
    def branch(*args, **kwargs):
        if args[5] == 'v12_geometry':
            raise nt.NativeTargetError('forced first branch rejection')
        return original(*args, **kwargs)
    monkeypatch.setattr(nt, 'numerical_branch', branch)
    i = identity(216, [[0, 99], [102, 215]])
    target = nt.produce_native_target(i, policy())
    assert len(target.traversal_bounds) >= 4
    for run in (0, 1):
        assert [r['branch_id'] for r in target.metadata['attempts'] if r['run_id'] == run] == list(nt.BRANCHES)
        assert [r['status'] for r in target.metadata['attempts'] if r['run_id'] == run] == ['rejected', 'unavailable', 'certified']
    assert not target.target_valid[99:102].any()
    assert not target.pulse[99:102].any()
    assert set(target.run_id) == {0, 1}


def test_zero_motion_short_run_and_response_loss_reject():
    for i in (identity(zero=True), identity(16)):
        t = nt.produce_native_target(i, policy())
        assert not t.target_valid.any() and not t.pulse.any() and not t.chi.any()
    i = identity()
    t = nt.produce_native_target(i, policy())
    support = i.response_edge_support.copy()
    support[int(t.traversal_bounds[0, 0])] = False
    with pytest.raises(ValueError, match='support loss'):
        nt.validate_native_target(t, response_support=support)
    with pytest.raises(ValueError, match='input SHA'):
        nt.lf1_supervision(i, t, expected_input_sha256='f' * 64)


def test_strict_exact_zero_arc_and_invalid_payload():
    g = identity().geometry_features.geometry.copy()
    g[1] = g[0]
    with pytest.raises(ValueError, match='zero'):
        nt._strict_traversals(g, np.ones_like(g, bool), np.ones(215, bool), np.array([0, 20, 40]))
    t = nt.produce_native_target(identity(), policy())
    for changes in ({'pulse': np.full(215, 2, np.uint8)}, {'chi': np.zeros(215)},
                    {'traversal_bounds': np.array([[0, 20], [19, 40]], np.int64)},
                    {'target_valid': np.ones(215, bool)}):
        with pytest.raises(ValueError):
            replace(t, **changes)


def test_save_resume_tamper(tmp_path):
    t = nt.produce_native_target(identity(), policy())
    path = tmp_path / 'target.npz'
    nt.save_native_target(path, t)
    loaded = nt.load_native_target(path, expected_input_sha256=policy().input_sha256)
    np.testing.assert_array_equal(loaded.pulse, t.pulse)
    meta = json.loads(path.with_suffix('.json').read_text())
    meta['input_sha256'] = 'f' * 64
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='input SHA'):
        nt.load_native_target(path, expected_input_sha256=policy().input_sha256)


def test_hybrid_geometry_override_and_outside_numeric():
    i = identity()
    g = i.geometry_features
    angle = np.linspace(0, 19, 216)
    phase = np.column_stack((np.cos(angle), np.sin(angle)))
    result, track = nt.numerical_branch(g.geometry, g.teacher_input[:, :149], np.ones((216, 149), np.uint8),
                                       g.coordinate_mask, i.response_edge_support, 'v9_hybrid', phase)
    assert result.certified
    for left, r in result.traversal_bounds:
        expected = close_phase_on_landmarks(phase, np.unique(result.traversal_bounds), geometry=g.geometry)
        np.testing.assert_array_equal(track.phase[left:r + 1], expected[left:r + 1])
    phase[-1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        nt.numerical_branch(g.geometry, g.teacher_input[:, :149], np.ones((216, 149), np.uint8),
                            g.coordinate_mask, i.response_edge_support, 'v9_hybrid', phase)


def test_terminal_n_minus_one_and_mixed_source(monkeypatch):
    # Actual original kernel accepts last observed sample N-1 as right endpoint.
    i = identity(201)
    t = nt.produce_native_target(i, policy())
    assert t.traversal_bounds[-1, 1] == 200
    assert len(t.pulse) == 200
    assert t.pulse[-1] == 0
    original = nt.numerical_branch
    def route(*args, **kwargs):
        if args[5] == 'v12_geometry' and args[0].shape[0] == 114:
            raise ValueError('exercise hybrid for second run')
        return original(*args, **kwargs)
    monkeypatch.setattr(nt, 'numerical_branch', route)
    i = identity(216, [[0, 99], [102, 215]])
    p = replace(policy(), hybrid_mode='available')
    artifact = {'phase': np.tile([[1., 0.]], (216, 1)), 'input_sha256': p.input_sha256,
                'availability_sha256': p.availability_sha256, 'teacher_sha256': 'a' * 64,
                'generation_code_sha256': 'b' * 64, 'generation_config_sha256': 'c' * 64,
                'selection_source': 'synthetic fixture only', 'opaque_key': i.feature.opaque_key.tobytes().hex(),
                'slot': 0, 'run_bounds': i.run_bounds, 'channel_definition': 'NB1-215',
                'static_definition': 'arc_static_features-v1'}
    with pytest.raises(ValueError, match='production_hybrid_intake_unavailable'):
        nt.produce_native_target(i, p, artifact)
    # Fixture-only injected phase intake exercises source bookkeeping; not a production artifact.
    monkeypatch.setattr(nt, '_phase_artifact', lambda *args: artifact['phase'])
    t = nt.produce_native_target(i, p, artifact)
    assert t.metadata['source'] == 'mixed'
    with pytest.raises(ValueError, match='hybrid'):
        nt.geometry_count(t)


def test_cli_pending_no_partial_target(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('target_cli', Path(__file__).resolve().parents[2] / 'scripts/build_temporac_native_targets.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    census = tmp_path / 'census.jsonl'
    census.write_text('{}')
    row = {'split': 'train', 'media_key': 'synthetic', 'length': 216, 'persons': [{'slot': 0}]}
    monkeypatch.setattr(cli, 'bridge_script', lambda: SimpleNamespace(read_census=lambda *a, **k: [row]))
    cfg = {'bridge_root': str(tmp_path / 'bridge'), 'census': str(census),
           'output': str(tmp_path / 'out'), 'hybrid_mode': 'unavailable'}
    assert cli.execute(cfg, 'plan') == 0
    assert cli.execute(cfg, 'build') == 2
    assert not (tmp_path / 'out/identities').exists()
    assert (tmp_path / 'out/population_pending.json').exists()


def test_real_nb1_pts_gap_integration():
    from pams.temporac.native_bridge_v1 import NativeFeatureV1, preprocess_native
    n = 216
    base = np.column_stack((np.linspace(-.8, .8, 17), np.linspace(.2, 1, 17)))
    base[11], base[12] = (-.2, 0), (.2, 0)
    xy = np.zeros((n, 17, 3), np.float32)
    xy[:, :, :2] = base
    xy[:, 7:11, 0] += (.15 * np.sin(2 * np.pi * np.arange(n) / 20))[:, None]
    xy[:, 13:17, 1] += (.12 * np.cos(2 * np.pi * np.arange(n) / 20))[:, None]
    xy[:, :, 2] = .9
    pts = np.arange(n, dtype=np.int64) * 40
    pts[100:] += 200  # >=5 median increments: separate, never compressed.
    f = NativeFeatureV1(np.arange(32, dtype=np.uint8), np.array([0], np.int64),
                         np.array([n], np.int64), np.arange(n, dtype=np.int64), xy,
                         np.ones((n, 17), bool), np.ones(n, bool), pts,
                         np.ones(n, bool), np.array([1, 1000], np.int64))
    i = preprocess_native(f)
    t = nt.produce_native_target(i, policy())
    assert len(i.run_bounds) == 2
    assert not t.target_valid[99]
    assert all(not (left <= 99 < right) for left, right in t.traversal_bounds)
    adapted = nt.lf1_supervision(i, t, expected_input_sha256=policy().input_sha256)
    assert adapted['edge_mask'].shape == (215,)


def test_cli_actual_nb1_full_fixture_build_validate_resume(tmp_path, monkeypatch):
    from pams.temporac.native_bridge_v1 import NativeFeatureV1
    spec = importlib.util.spec_from_file_location('target_cli_fixture', Path(__file__).resolve().parents[2] / 'scripts/build_temporac_native_targets.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    n = 216
    census = tmp_path / 'census.jsonl'
    census.write_text('{}')
    row = {'split': 'train', 'media_key': np.arange(32, dtype=np.uint8).tobytes().hex(), 'length': n, 'persons': [{'slot': 0}]}
    monkeypatch.setattr(cli, 'bridge_script', lambda: SimpleNamespace(read_census=lambda *a, **k: [row]))
    cfg = {'bridge_root': str(tmp_path / 'bridge'), 'census': str(census),
           'output': str(tmp_path / 'out'), 'hybrid_mode': 'unavailable'}
    stem = tmp_path / ('bridge/identities/train/' + row['media_key'] + '_0')
    stem.parent.mkdir(parents=True)
    xy = np.zeros((n, 17, 3), np.float32)
    # Explicit upstream raw failure: every identity remains full-E rejected, never negative supervision.
    f = NativeFeatureV1(np.arange(32, dtype=np.uint8), np.array([0], np.int64),
                         np.array([n], np.int64), np.arange(n, dtype=np.int64), xy,
                         np.zeros((n, 17), bool), np.zeros(n, bool), np.zeros(n, np.int64),
                         np.zeros(n, bool), np.array([1, 1000], np.int64))
    np.savez(stem.with_suffix('.npz'), **f.as_arrays())
    np.savez(stem.with_suffix('.derived.npz'), fixture=np.zeros(1))
    receipt = {'feature_sha256': nt.file_sha(stem.with_suffix('.npz')),
               'derived_sha256': nt.file_sha(stem.with_suffix('.derived.npz')),
               'binding': {'raw_npz_sha256': 'a' * 64, 'raw_json_sha256': 'b' * 64}, 'raw_status': 'fixture_failure'}
    stem.with_suffix('.json').write_text(json.dumps(receipt))
    assert cli.execute(cfg, 'build') == 0
    assert cli.execute(cfg, 'validate') == 0
    assert cli.execute(cfg, 'diagnose-inputs') == 0
    target_path = tmp_path / ('out/identities/train/' + row['media_key'] + '_0.npz')
    target = nt.load_native_target(target_path)
    assert target.pulse.shape == (215,) and not target.target_valid.any()
    assert cli.execute(cfg, 'build') == 0
    receipt['binding']['raw_npz_sha256'] = 'c' * 64
    stem.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='resume'):
        cli.execute(cfg, 'build')
