"""Label-free full-native NT1 plan/build/validate/diagnose-inputs; no training or labels."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np  # noqa: E402

from pams.temporac.native_bridge_v1 import load_native_feature, preprocess_native  # noqa: E402
from pams.temporac.native_target_nt1 import (  # noqa: E402
    SCHEMA,
    NativeTargetPolicy,
    file_sha,
    geometry_count,
    json_sha,
    load_native_target,
    produce_native_target,
    save_native_target,
    validate_native_target,
)

EXPECTED = {'train': {'videos': 794, 'persons': 1878}, 'val': {'videos': 183, 'persons': 486}}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def bridge_script():
    spec = importlib.util.spec_from_file_location('nb1_builder_readonly',
                                                 ROOT / 'scripts/build_temporac_native_learner_bridge.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def producer_binding():
    # Bind the original numerical kernels and their local transitive dependencies.
    names = ('native_target_nt1', 'certify', 'certify_adaptive_v2', 'certify_adaptive_v3',
             'certify_adaptive_v4', 'certify_adaptive_v5', 'certify_target_adaptive_v4',
             'certify_target_adaptive_v6', 'certify_target_adaptive_v7',
             'full_chain_landmarks_candidate_v1', 'full_chain_landmarks_candidate_v2',
             'multi_cycle_dtw_template_candidate_v1', 'dtw_phase_candidate_v1',
             'periodic_template_candidate_v1', 'phase_closure_candidate_v1',
             'native_bridge_v1', 'native_time_cue_v1', 'preprocess', 'quadrature', 'types',
             'hashio', 'receipts', 'contract', 'x0')
    return {'cli': file_sha(__file__), **{n: file_sha(ROOT / f'src/pams/temporac/{n}.py') for n in names}}


def target_summary(identity, target):
    response, observed, valid = identity.response_edge_support, identity.observed_edge_support, target.target_valid
    dt = np.diff(identity.clock.seconds)
    known = identity.clock.edge_time_support if identity.clock.mode == 'pts' else np.zeros(len(valid), bool)
    training = np.zeros(len(valid), bool)
    ineligible = []
    for left, r in target.traversal_bounds:
        w, p = target.chi[left:r], target.pulse[left:r]
        ok = np.sum(w * p) > 0 and np.sum(w * (1 - p)) > 0
        if ok:
            training[left:r] = True
        else:
            ineligible.append([int(left), int(r), 'missing_positive_or_negative_mass'])
    return {
        'source_length': len(valid) + 1, 'status': target.metadata['status'],
        'source': target.metadata['source'], 'K': len(target.traversal_bounds),
        'input_status': identity.status, 'input_reasons': list(identity.reasons),
        'short_runs': identity.short_runs.tolist(),
        'clock_mode': identity.clock.mode, 'attempts': target.metadata['attempts'],
        'training_ineligible': ineligible,
        'support': {name: {'edges': int(mask.sum()),
                           'known_seconds': float(dt[mask & known].sum()) if known.any() else None,
                           'unknown_time_edges': int((mask & ~known).sum()),
                           'source_edge_fraction': float(mask.mean()) if mask.size else 0.0}
                    for name, mask in (('observed', observed), ('response', response),
                                       ('target', valid), ('training', training))},
        'direct_pseudo_event_baseline': geometry_count(target),
        'true_unit_quality': 'unavailable_requires_isolated_evaluator',
    }


def execute(cfg, command, *, expected=EXPECTED):
    if set(cfg) != {'bridge_root', 'census', 'output', 'hybrid_mode'}:
        raise ValueError('config requires exactly bridge_root,census,output,hybrid_mode; no caps')
    if cfg['hybrid_mode'] != 'unavailable':
        raise ValueError('first production build requires hybrid unavailable; freeze availability separately')
    for name in ('bridge_root', 'census', 'output'):
        p = Path(cfg[name]).resolve()
        if os.name == 'nt' and p.drive.upper() != 'E:':
            raise ValueError('all project inputs/outputs must remain on E:')
    out, bridge = Path(cfg['output']).resolve(), Path(cfg['bridge_root']).resolve()
    if bridge == out or bridge in out.parents or out in bridge.parents:
        raise ValueError('target output must be separate from NB1 input')
    rows = bridge_script().read_census(cfg['census'], expected=expected)
    population = [(r, p) for r in rows for p in r['persons']]
    binding = {'config_sha256': json_sha(cfg), 'census_sha256': file_sha(cfg['census']),
               'producer': producer_binding(), 'availability': {'hybrid_mode': 'unavailable'}}
    missing = []
    for row, p in population:
        stem = bridge / 'identities' / row['split'] / f"{row['media_key']}_{p['slot']}"
        if not all(stem.with_suffix(s).is_file() for s in ('.npz', '.json', '.derived.npz')):
            missing.append({'media_key': row['media_key'], 'slot': p['slot']})
    if command == 'plan':
        atomic_json(out / 'plan.json', {'schema': SCHEMA, 'status': 'prepared_not_built',
                    'binding': binding, 'videos': len(rows), 'identities': len(population), 'missing': missing})
        return 0
    if missing:
        atomic_json(out / 'population_pending.json', {'status': 'pending_complete_NB1', 'missing': missing,
                    'videos': len(rows), 'identities': len(population), 'binding': binding})
        return 2
    ledger, errors = [], []
    for row, p in population:
        slot = p['slot']
        stem = bridge / 'identities' / row['split'] / f"{row['media_key']}_{slot}"
        path = stem.with_suffix('.npz')
        receipt = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
        if receipt['feature_sha256'] != file_sha(path) or receipt['derived_sha256'] != file_sha(stem.with_suffix('.derived.npz')):
            raise ValueError('NB1 input hash mismatch; stop build')
        feature = load_native_feature(path)
        identity = preprocess_native(feature)
        if (int(feature.slot[0]) != slot or int(feature.source_length[0]) != row['length']
                or feature.opaque_key.tobytes().hex() != row['media_key']):
            raise ValueError('NB1 census identity mismatch; stop build')
        policy = NativeTargetPolicy(input_sha256=file_sha(path),
                    raw_sha256=json_sha({k: v for k, v in receipt['binding'].items() if k.startswith('raw_')}),
                    bridge_sha256=json_sha(receipt['binding']), producer_sha256=json_sha(binding['producer']),
                    config_sha256=binding['config_sha256'], availability_sha256=json_sha(binding['availability']))
        dest = out / 'identities' / row['split'] / f"{row['media_key']}_{slot}.npz"
        if command == 'diagnose-inputs':
            ledger.append({'media_key': row['media_key'], 'split': row['split'], 'slot': slot,
                           'source_length': row['length'], 'response_edges': int(identity.response_edge_support.sum()),
                           'observed_edges': int(identity.observed_edge_support.sum()), 'runs': identity.run_bounds.tolist(),
                           'status': identity.status, 'reasons': list(identity.reasons)})
            continue
        try:
            if dest.with_suffix('.json').exists():
                target = load_native_target(dest, expected_input_sha256=policy.input_sha256)
                if any(target.metadata[k] != v for k, v in policy.__dict__.items()):
                    raise ValueError('resume provenance/config/code changed; choose new output')
                validate_native_target(target, response_support=identity.response_edge_support)
            elif command == 'validate':
                raise ValueError('missing full-population target artifact')
            else:
                target = produce_native_target(identity, policy)
                save_native_target(dest, target)
            ledger.append({'media_key': row['media_key'], 'split': row['split'], 'slot': slot,
                           **target_summary(identity, target)})
        except (ValueError, OSError, KeyError) as exc:
            # Integrity errors are not certificate rejection or unavailable hybrid.
            errors.append({'media_key': row['media_key'], 'slot': slot, 'error': str(exc)})
            raise ValueError('target artifact integrity failure: ' + str(exc)) from exc
    result = {'schema': SCHEMA, 'status': 'diagnosed_inputs' if command == 'diagnose-inputs' else 'complete_population',
              'videos': len(rows), 'identities': len(ledger), 'records': ledger, 'failures': errors,
              'status_counts': dict(Counter(x['status'] for x in ledger)), 'binding': binding,
              'natural_GT_accessed': False, 'target_quality_acceptance': 'pending_isolated_evaluator'}
    atomic_json(out / ('input_diagnostics.json' if command == 'diagnose-inputs' else
                      'validation.json' if command == 'validate' else 'population_ledger.json'), result)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'build', 'validate', 'diagnose-inputs'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return execute(json.loads(args.config.read_text(encoding='utf-8-sig')), args.command)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
