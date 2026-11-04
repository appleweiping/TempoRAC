"""Isolated DDF1 development scoring for completed NTR1 predictions.

The numerical metric implementation is the already reviewed macro_metrics.
Only this evaluator reads the previous diagnostic's GT-bearing identity rows.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
import zipfile

import numpy as np

from pams.temporac.native_target_diagnostics_v1 import macro_metrics

SCHEMA = 'temporac.native-prediction-score.v1'
NTR1 = 'temporac.native-training.ntr1'
INPUTS = {'diagnostics', 'identity_rows', 'predictions', 'selection', 'plan',
          'count_policy', 'checkpoint', 'execution_freeze'}
POLICY = {
    'schema': 'temporac.ntr1.count-policy.v1',
    'split': 'val', 'threshold': 0.5,
    'failure_policy': 'abstain_minus_one_to_zero_full_population',
    'aggregation': 'mean_people_within_video_then_mean_videos',
    'population': {'videos': 183, 'identities': 486},
    'metrics': ['nmae', 'raw_mae', 'obo', 'signed_error', 'relative_signed_error', 'count_ratio'],
    'official_metric_status': 'not_certified',
    'period_ap': 'unavailable_no_legal_period_decoder', 'bootstrap_replicates': 0,
}


class ScoreError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ScoreError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    def bad(value):
        raise ScoreError('nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad)


def item_equal(left, right):
    return (Path(left['path']).resolve() == Path(right['path']).resolve()
            and left['sha256'] == right['sha256'])


def project_path(value):
    path = Path(value).resolve()
    if path.drive:
        require(path.drive.upper() == 'E:', 'score artifacts must stay on E:')
    else:
        base = Path('/mnt/4tssd/lilei/TempoRAC')
        require(path == base or base in path.parents, 'score artifacts outside verified server root')
    return path


def _input_files(config):
    require(set(config) == {'protocol', 'inputs', 'prediction_root', 'output',
                            'seed', 'variant', 'source_commit'}, 'exact score config required')
    require(config['protocol'] == SCHEMA and set(config['inputs']) == INPUTS, 'score schema/input set mismatch')
    require(config['seed'] in (20260815, 20260816, 20260817) and type(config['seed']) is int,
            'invalid frozen seed')
    require(config['variant'] in ('canonical', 'capacity-control'), 'invalid NTR1 variant')
    require(isinstance(config['source_commit'], str) and re.fullmatch('[0-9a-f]{40}', config['source_commit']),
            'source commit required')
    root = project_path(config['prediction_root'])
    paths, hashes = {}, {}
    for key, ref in config['inputs'].items():
        require(set(ref) == {'path', 'sha256'}, 'exact path/SHA input required: ' + key)
        path = project_path(ref['path'])
        require(path.is_file() and digest(path) == ref['sha256'], 'input missing/changed: ' + key)
        paths[key], hashes[str(path)] = path, ref['sha256']
    require(paths['predictions'] == root / 'predictions.json'
            and paths['selection'] == root / 'selection.json'
            and paths['plan'] == root.parent / 'frozen_plan.json', 'NTR1 output layout mismatch')
    return root, paths, hashes


def _bindings(config, paths):
    data = {k: read_json(v) for k, v in paths.items() if k not in ('identity_rows', 'checkpoint')}
    policy, freeze, plan, pred, sel, diag = (data[k] for k in
        ('count_policy', 'execution_freeze', 'plan', 'predictions', 'selection', 'diagnostics'))
    require(all(policy.get(k) == v for k, v in POLICY.items()), 'frozen DDF1 policy fields differ')
    name = policy.get('metric_name', '')
    require(isinstance(name, str) and ('development' in name.lower() or 'nonofficial' in name.lower()),
            'metric name must explicitly identify development/nonofficial scope')
    require(freeze['source_commit'] == config['source_commit']
            and freeze['gate']['source_commit'] == config['source_commit'], 'execution source binding mismatch')
    require(freeze['gate']['threeway_checkpoint_status'] == 'verified', 'execution checkpoint not verified')
    require(plan['schema'] == NTR1 and not plan['pending'], 'unqualified NTR1 plan')
    plan_sha = json_sha(plan)
    require(pred['schema'] == NTR1 and pred['status'] == 'complete_population'
            and pred['natural_GT_accessed'] is False, 'prediction is not complete feature-only output')
    require(sel['schema'] == NTR1 and sel['status'] == 'complete_training', 'selection is not complete training')
    require(sel['plan_sha256'] == pred['plan_sha256'] == plan_sha, 'prediction/selection/plan mismatch')
    require(sel['seed'] == plan['seed'] == config['seed'] and sel['variant'] == config['variant'], 'seed/arm mismatch')
    require(sel['initial_state_sha256'] == plan['initial_state_sha256']
            and sel['draw_ledger_sha256'] == plan['draw_ledger_sha256'], 'paired selection bindings mismatch')
    require(json_sha(plan['draw_ledger']) == plan['draw_ledger_sha256'], 'draw ledger content mismatch')
    total = plan['T']
    require(type(total) is int and total >= 10000 and total % 500 == 0, 'full fixed horizon required')
    require(len(plan['draw_ledger']) == total, 'draw ledger must cover full horizon')
    steps = list(range(500, total + 1, 500))
    require(plan['candidate_steps'] == steps and [c['step'] for c in sel['candidates']] == steps,
            'full candidate checkpoint set required')
    require(all(type(c['objective']) in (int, float) and math.isfinite(c['objective']) for c in sel['candidates']),
            'nonfinite selection objective')
    chosen = min(sel['candidates'], key=lambda c: (c['objective'], c['step']))
    require(sel['selected'] == pred['selected'] == chosen, 'selected checkpoint is not frozen X0 minimum')
    require(plan['selection'] == 'all_8_tune_sources_7_blocks_LF1_min_finite_tie_earlier', 'non-X0 selection policy')
    require(Path(chosen['file']).name == chosen['file']
            and paths['checkpoint'] == paths['selection'].parent / chosen['file'], 'checkpoint byte reference mismatch')
    binding = plan['population_binding']
    for key in ('diagnostics', 'count_policy'):
        require(item_equal(binding['artifacts'][key], config['inputs'][key])
                and item_equal(freeze['gate']['artifacts'][key], config['inputs'][key]),
                'execution/plan artifact mismatch: ' + key)
    require(diag['schema'] == 'temporac.native-target-diagnostics.v1'
            and diag['official_metric_status'] == 'not_certified', 'unrecognized full diagnostic')
    require(diag['receipt']['identity_rows_sha256'] == config['inputs']['identity_rows']['sha256'],
            'diagnostic row content mismatch')
    require(len(diag['same_media_alias_groups']) == 4, 'full diagnostic alias evidence missing')
    for split, videos, identities in [('train', 794, 1878), ('val', 183, 486)]:
        item = diag['splits'][split]
        require(item['identity_denominator'] == identities
                and item['full_population']['videos'] == videos
                and item['full_population']['identities'] == identities, 'full diagnostic denominator mismatch')
    cfg = binding['configuration']
    req = {str(Path(cfg['census']).resolve()): binding['census_sha256'],
           str(Path(cfg['component_map']).resolve()): binding['component_map_sha256']}
    for key in ('nb1', 'nt1'):
        ref = binding['artifacts'][key]
        req[str(Path(ref['path']).resolve())] = ref['sha256']
        require(item_equal(ref, freeze['gate']['artifacts'][key]), 'execution NB1/NT1 mismatch')
    require(all(diag['receipt']['inputs_sha256'].get(p) == h for p, h in req.items()),
            'diagnostic does not describe plan population inputs')
    return data, plan_sha


def _join(config, root, paths, data, hashes):
    # GT-bearing rows are read only after immutable model/selection/policy checks above.
    rows = [json.loads(line) for line in paths['identity_rows'].read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    def index(population, *, plan=False):
        result = {}
        for r in population:
            require(r['split'] in ('train', 'val'), 'unknown diagnostic/plan split')
            require(type(r['slot']) is int and r['slot'] >= 0, 'invalid identity slot')
            key = r['media_key'] + '_' + str(r['slot'])
            require(key not in result and (not plan or r['key'] == key), 'duplicate/mismatched population identity')
            result[key] = r
        for split, videos, identities in [('train', 794, 1878), ('val', 183, 486)]:
            selected = [r for r in result.values() if r['split'] == split]
            require(len(selected) == identities and len({r['media_key'] for r in selected}) == videos,
                    'complete diagnostic/plan population required')
        return result
    diagnostic, population = index(rows), index(data['plan']['population'], plan=True)
    require(set(diagnostic) == set(population) and all(
        (diagnostic[k]['split'], diagnostic[k]['source_length']) == (population[k]['split'], population[k]['N'])
        for k in diagnostic), 'diagnostic/plan identity or source-length mismatch')
    val = {k: r for k, r in diagnostic.items() if r['split'] == 'val'}
    predictions = data['predictions']['records']
    require(len(predictions) == 486 and len({p['key'] for p in predictions}) == 486
            and {p['key'] for p in predictions} == set(val), 'missing/duplicate/unknown prediction identities')
    scored = []
    for p in predictions:
        key = p['key']
        r = val[key]
        require(type(p['count']) is int and p['status'] in ('complete', 'partial', 'abstain'), 'invalid count/status')
        require((p['status'] == 'abstain' and p['count'] == -1)
                or (p['status'] != 'abstain' and p['count'] >= 0), 'invalid count for prediction status')
        require(type(p['decoder_invocations']) is int and p['decoder_invocations'] == 1, 'decoder invocation mismatch')
        require(p['period_ap_status'] == 'unavailable_no_period_decoder', 'unexpected period decoder claim')
        path = root / 'predictions' / (key + '.npz')
        require(path.is_file() and digest(path) == p['sha256'], 'prediction NPZ missing/changed: ' + key)
        hashes[str(path)] = p['sha256']
        with np.load(path, allow_pickle=False) as arrays:
            require(set(arrays.files) == {'response', 'support', 'run_bounds', 'component_bounds',
                                         'event_source_positions', 'event_seconds'}, 'prediction NPZ schema mismatch')
            n = r['source_length']
            response, support, bounds = arrays['response'], arrays['support'], arrays['component_bounds']
            runs, positions, seconds = arrays['run_bounds'], arrays['event_source_positions'], arrays['event_seconds']
            require(response.shape == support.shape == (n - 1,) and support.dtype == np.bool_
                    and response.dtype == np.dtype('<f4')
                    and np.isfinite(response).all() and np.all((response >= 0) & (response <= 1)),
                    'prediction response/support mismatch')
            require(bounds.ndim == runs.ndim == 2 and bounds.shape[1] == runs.shape[1] == 2
                    and bounds.dtype == runs.dtype == np.dtype('<i4'), 'invalid predicted bound shape/dtype')
            count = 0 if p['status'] == 'abstain' else p['count']
            require(len(bounds) == count and positions.shape == seconds.shape == (count,)
                    and positions.dtype == seconds.dtype == np.dtype('<f8'), 'prediction count/component mismatch')
            require(np.isfinite(positions).all() and not np.isinf(seconds).any(), 'invalid event coordinates')
            owned = np.zeros(n - 1, dtype=bool)
            previous = -1
            for start, stop in runs.tolist():
                require(0 <= start < stop <= n - 1 and start > previous, 'invalid run bound range/order')
                owned[start:stop] = True
                previous = stop
            require(np.array_equal(owned, support) and not np.any(response[~support]), 'run/support mismatch')
            previous = -1
            for (start, stop), position in zip(bounds.tolist(), positions.tolist()):
                require(0 <= start < stop <= n - 1 and start > previous
                        and any(a <= start < stop <= b for a, b in runs.tolist()), 'invalid component bound range/order')
                require(start + .5 <= position <= stop - .5 and position - .5 == int(position - .5),
                        'event source position outside its component')
                previous = stop
            expected_status = 'complete' if support.size and support.all() else 'partial' if support.any() else 'abstain'
            require(p['status'] == expected_status, 'prediction status/support mismatch')
        scored.append({'media_key': r['media_key'], 'slot': r['slot'], 'source_component': r['source_component'],
                       'gt_count': r['gt_count'], 'count': count, 'raw_prediction_count': p['count'],
                       'prediction_status': p['status'], 'prediction_sha256': p['sha256']})
    return scored


def score(config):
    root, paths, hashes = _input_files(config)
    data, plan_sha = _bindings(config, paths)
    rows = _join(config, root, paths, data, hashes)
    result = macro_metrics(rows)
    require(result['videos'] == 183 and result['identities'] == 486, 'scored denominator mismatch')
    require(all(digest(p) == h for p, h in hashes.items()), 'input changed during scoring')
    return rows, {'schema': SCHEMA, 'status': 'complete_population', 'metric_name': data['count_policy']['metric_name'],
        'seed': config['seed'], 'variant': config['variant'], 'source_commit': config['source_commit'],
        'plan_sha256': plan_sha, 'selected': data['selection']['selected'], 'full_population': result,
        'prediction_status_counts': dict(Counter(r['prediction_status'] for r in rows)),
        'official_metric_status': 'not_certified', 'period_ap': 'unavailable_no_legal_period_decoder',
        'bootstrap_replicates': 0, 'scientific_acceptance': 'not_assigned',
        'receipt': {'config_sha256': json_sha(config), 'inputs_sha256': hashes,
                    'scorer_sha256': digest(__file__), 'metric_core_sha256': digest(Path(macro_metrics.__code__.co_filename)),
                    'GT_source': 'completed_diagnostic_identity_rows_val_only',
                    'original_annotations_opened': False, 'model_invoked': False, 'GPU_used': False}}


def run(config):
    out = project_path(config['output'])
    require(not out.exists(), 'output exists; preserve prior scoring evidence')
    inputs = [Path(x['path']).resolve() for x in config['inputs'].values()]
    require(all(out != p and out not in p.parents and p not in out.parents for p in inputs),
            'output overlaps immutable input')
    try:
        rows, result = score(config)
    except (ValueError, KeyError, OSError, TypeError, zipfile.BadZipFile, EOFError) as exc:
        out.mkdir(parents=True)
        (out / 'failure.json').write_text(json.dumps({'schema': SCHEMA, 'status': 'failed',
            'error': str(exc), 'config_sha256': json_sha(config), 'inputs': config['inputs'],
            'partial_score_written': False}, sort_keys=True, indent=2))
        if isinstance(exc, (zipfile.BadZipFile, EOFError)):
            raise ScoreError('invalid prediction archive: ' + str(exc)) from exc
        raise
    out.mkdir(parents=True)
    raw = out / 'scored_rows.jsonl'
    raw.write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in rows))
    result['receipt']['scored_rows_sha256'] = digest(raw)
    report = out / 'score.json'
    report.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    (out / 'receipt.json').write_text(json.dumps({'schema': SCHEMA, 'score_sha256': digest(report),
        'scored_rows_sha256': digest(raw), 'inputs': config['inputs']}, indent=2, sort_keys=True))
    return result
