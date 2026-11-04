"""Explicit cached NTR1 execution. Frozen scientific functions live in NTR1 core.

This module never calls _plan, _load_inputs, preprocess_native or NUDFT. R3 is
an execution context; the byte-verified parent scientific plan is immutable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from pams.temporac import native_training_runtime_v1 as core

SCHEMA = core.SCHEMA
BINDING_SCHEMA = 'temporac.ntr1.cached-execution-binding.v1'
LAUNCHER_SCHEMA = 'temporac.ntr1.cached-launcher.v1'
PARENT_SOURCE = '591b4659c5c94ed7b9538ca8b1dd4fc1bdede375'
PARENT_FILE_SHA = '41405fdacb7648ab66b0d456b842085e60497967d63693c4cd3616752a96e382'
PARENT_PLAN_SHA = '8d50706a22bcfc47206b37ae399c56c2dbf70c99640c573d49ff20efa11259c3'
PARENT_CONFIG_SHA = '8e8911c1c2c7b0b774fb7b38f288b1f046b61d4e6c21eb5197df381d9ceeb3f7'
PARENT_INIT_SHA = '52ea59cc8f9889a2f050bb502468ba8c2940c017e105953105a666530a9ef388'
PARENT_DRAW_SHA = '06b435616da5aace9260e06be57528dc6ef85c6c7fc253fd4d324e141dcc2014'
R3 = {'name': 'NTR1-R3-cache-reuse-5-4-7', 'gpu_index': 1,
      'process_total_cap_bytes': 5 * core.GIB,
      'allocator_reserved_cap_bytes': 4 * core.GIB,
      'start_free_min_bytes': 7 * core.GIB,
      'running_free_min_bytes': 2 * core.GIB,
      'wall_limit_seconds': 86400, 'gpu_limit_seconds': 86400}
NEW_SOURCE_FILES = ('src/pams/temporac/native_cached_training_runtime_v1.py',
                    'src/pams/temporac/native_prepared_cache_v1.py',
                    'src/pams/temporac/cache_build_v1.py',
                    'src/pams/temporac/native_fast_cue_v1.py',
                    'scripts/train_temporac_cached_response.py',
                    'scripts/prepare_temporac_cache.py',
                    'scripts/run_temporac_cached_campaign.py')
PARITY_SOURCE_FILES = ('src/pams/temporac/native_fast_cue_v1.py',
                       'src/pams/temporac/native_prepared_cache_v1.py',
                       'src/pams/temporac/cache_build_v1.py')


def verify_production_parity(evidence, plan_sha256, actual_sources, environment):
    """The measured numerical kernel must be the actual deployed kernel."""
    if (evidence.get('schema') != 'temporac.ntr1.fast-cache-parity.v1'
            or evidence.get('status') != 'complete'
            or evidence.get('all_bytes_equal') is not True
            or evidence.get('plan_sha256') != plan_sha256
            or evidence.get('environment') != environment):
        raise ValueError('production parity incomplete or plan/environment differs')
    measured = evidence.get('new_source_files', {})
    if any(name not in actual_sources or measured.get(name) != actual_sources[name]
           for name in PARITY_SOURCE_FILES):
        raise ValueError('production parity kernel source differs from execution')


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def checked_ref(ref):
    if set(ref) != {'path', 'sha256'}:
        raise ValueError('explicit path/SHA reference required')
    path = core._checked_path(ref['path'])
    if core.file_sha(path) != ref['sha256']:
        raise ValueError('missing/changed referenced artifact: ' + str(path))
    return path


def load_verified_frozen_plan(plan_ref, validate_ref):
    if plan_ref['sha256'] != PARENT_FILE_SHA or validate_ref['sha256'] != PARENT_FILE_SHA:
        raise ValueError('wrong parent plan/validate byte identity')
    document = read_json(checked_ref(plan_ref))
    validated = read_json(checked_ref(validate_ref))
    if document != validated or document.get('status') != 'CPU_plan_complete_not_trained':
        raise ValueError('parent plan/validate not complete and identical')
    payload = {k: v for k, v in document.items() if k not in ('status', 'plan_sha256')}
    if document.get('plan_sha256') != PARENT_PLAN_SHA or core.json_sha(payload) != PARENT_PLAN_SHA:
        raise ValueError('parent scientific plan payload SHA mismatch')
    plan = core.FrozenPlan(payload)
    if (payload['schema'] != SCHEMA or payload['pending'] or payload['seed'] != 20260815
            or plan.T != 10000 or payload['C'] != 23 or max(payload['n_c'].values()) != 248
            or payload['initial_state_sha256'] != PARENT_INIT_SHA
            or payload['draw_ledger_sha256'] != PARENT_DRAW_SHA
            or core.json_sha(payload['draw_ledger']) != PARENT_DRAW_SHA):
        raise ValueError('parent frozen scientific invariants differ')
    population = payload['population']
    if (Counter(r['split'] for r in population) != {'train': 1878, 'val': 486}
            or len({r['key'] for r in population}) != 2364
            or sum(r['N'] for r in population) != 1758034
            or any(len({r['media_key'] for r in population if r['split'] == split}) != count
                   for split, count in (('train', 794), ('val', 183)))):
        raise ValueError('parent full population differs')
    eligible = {key for values in payload['eligible_by_component'].values() for key in values}
    if len(eligible) != 1084 or set(payload['targets']) != {r['key'] for r in population}:
        raise ValueError('parent eligible/full target population mismatch')
    if eligible != {r['key'] for r in population if r['split'] == 'train' and payload['targets'][r['key']]['eligible']}:
        raise ValueError('parent eligibility ledger mismatch')
    if len(payload['draw_ledger']) != plan.T or any(
            row != plan.sampler.draw(i) for i, row in enumerate(payload['draw_ledger'])):
        raise ValueError('parent paired draw sequence differs')
    if dict(sorted(Counter(r['identity'] for r in payload['draw_ledger']).items())) != payload['exposures']:
        raise ValueError('parent exposure ledger differs')
    return plan


def admission(resource, envelope, *, starting=True):
    if any(resource.get(key) != value for key, value in R3.items()):
        raise ValueError('explicit exact R3 resource policy required')
    required = ('lease_id', 'device_uuid', 'environment_sha', 'owned_job_recovery_receipt',
                'free_vram', 'free_disk', 'other_process_inventory', 'disk_path')
    if any(resource.get(k) is None for k in required) or not resource['lease_id'] or not resource['owned_job_recovery_receipt']:
        raise ValueError('unknown/pending R3 shared-resource lease')
    if envelope['B_plan'] > resource['process_total_cap_bytes']:
        raise ValueError('planning envelope exceeds R3 process cap')
    threshold = resource['start_free_min_bytes'] if starting else resource['running_free_min_bytes']
    if resource['free_vram'] < threshold:
        raise ValueError('actual free VRAM below R3 headroom')
    return True


@dataclass(frozen=True)
class ExecutionContext:
    binding_sha256: str
    resource: dict
    train_manifest: dict
    prior_consumption: dict


def resource_check(plan, context, *, starting, baseline=None):
    resource = context.resource
    live = core._nvidia_query()
    if resource['device_uuid'] != live['device_uuid']:
        raise RuntimeError('resource_incomplete: GPU UUID changed')
    admission(dict(resource, free_vram=live['free_vram']), plan.payload['planned_tensor_bytes'], starting=starting)
    if resource['environment_sha'] != core.json_sha(core.environment_binding()):
        raise RuntimeError('resource_incomplete: R3 environment differs')
    own = sum(r['bytes'] for r in live['inventory'] if r['pid'] == os.getpid())
    if own > resource['process_total_cap_bytes']:
        raise RuntimeError('resource_incomplete: NVML process total exceeds R3 5GiB')
    other = {r['pid']: r['bytes'] for r in live['inventory'] if r['pid'] != os.getpid()}
    live['other_allocation_changes'] = {} if baseline is None else {
        pid: value - baseline.get(pid, 0) for pid, value in other.items() if value != baseline.get(pid, 0)}
    if torch.cuda.is_initialized() and torch.cuda.memory_reserved(1) > resource['allocator_reserved_cap_bytes']:
        raise RuntimeError('resource_incomplete: allocator reserved exceeds R3 4GiB')
    if starting and shutil.disk_usage(resource['disk_path']).free < plan.payload['required_free_disk']:
        raise RuntimeError('resource_incomplete: disk reserve below parent two peaks plus 10GiB')
    return live, other


def require_train(plan, context):
    if plan.payload['pending']:
        raise ValueError('pending parent scientific plan')
    admission(context.resource, plan.payload['planned_tensor_bytes'])
    if context.resource['free_disk'] < plan.payload['required_free_disk']:
        raise ValueError('R3 lease disk inventory insufficient')


def admitted_device(plan, context):
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        raise RuntimeError('GPU1 unavailable; no CPU model execution fallback')
    _, baseline = resource_check(plan, context, starting=True)
    torch.cuda.set_device(1)
    torch.cuda.set_per_process_memory_fraction(
        context.resource['allocator_reserved_cap_bytes'] / torch.cuda.get_device_properties(1).total_memory, 1)
    return torch.device('cuda:1'), baseline


def current_commit():
    result = subprocess.run(['git', '-C', str(core.ROOT), 'rev-parse', 'HEAD'],
                            check=True, text=True, capture_output=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return result.stdout.strip()


def load_execution(launcher):
    expected = {'schema', 'parent_plan', 'parent_validate', 'parent_config', 'execution_binding', 'output', 'variant'}
    if set(launcher) != expected or launcher['schema'] != LAUNCHER_SCHEMA or launcher['variant'] not in core.VARIANTS:
        raise ValueError('exact cached launcher schema required')
    plan = load_verified_frozen_plan(launcher['parent_plan'], launcher['parent_validate'])
    if launcher['parent_config']['sha256'] != PARENT_CONFIG_SHA:
        raise ValueError('parent config SHA mismatch')
    cfg = read_json(checked_ref(launcher['parent_config']))
    core._check_config(cfg)
    expected_cfg = {k: v for k, v in cfg.items() if k not in ('variant', 'output')}
    if expected_cfg != plan.payload['population_binding']['configuration']:
        raise ValueError('parent config/scientific plan mismatch')
    bound, pending = core._artifact_receipts(cfg)
    if pending or bound != plan.payload['population_binding']['artifacts']:
        raise ValueError('original eight artifact qualifications/bindings changed')
    core.verify_evidence_bindings(cfg, plan.payload['population'])
    binding_path = checked_ref(launcher['execution_binding'])
    binding = read_json(binding_path)
    if (binding['schema'] != BINDING_SCHEMA or binding['parent_source_commit'] != PARENT_SOURCE
            or binding['parent_plan_file_sha256'] != PARENT_FILE_SHA
            or binding['parent_validate_file_sha256'] != PARENT_FILE_SHA
            or binding['parent_plan_payload_sha256'] != plan.sha256
            or binding['parent_config_sha256'] != PARENT_CONFIG_SHA
            or binding['actual_source_commit'] != current_commit()
            or binding['environment'] != core.environment_binding()):
        raise ValueError('execution binding does not describe actual code/environment/parent')
    launcher_payload = {k: v for k, v in launcher.items() if k != 'execution_binding'}
    if binding['launcher_config_sha256'] != core.json_sha(launcher_payload):
        raise ValueError('launcher execution fields changed')
    expected_sources = dict(plan.payload['population_binding']['sources'])
    actual_sources = binding['actual_source_file_hashes']
    if set(actual_sources) != set(expected_sources) | set(NEW_SOURCE_FILES):
        raise ValueError('execution source file inventory incomplete')
    for relative, sha in actual_sources.items():
        path = (core.ROOT / relative).resolve()
        if core.ROOT not in path.parents or core.file_sha(path) != sha:
            raise ValueError('actual source changed: ' + relative)
        if relative in expected_sources and sha != expected_sources[relative]:
            raise ValueError('frozen scientific/core source changed: ' + relative)
    checked_ref(binding['train_cache_manifest'])
    lease = read_json(checked_ref(binding['live_lease']))
    if binding['resource_policy_sha256'] != core.json_sha(R3):
        raise ValueError('resource policy fingerprint differs')
    admission(lease, plan.payload['planned_tensor_bytes'])
    for name in ('compatibility_receipt', 'production_parity_receipt'):
        evidence = read_json(checked_ref(binding[name]))
        if evidence.get('status') != 'complete' or evidence.get('plan_sha256') != plan.sha256:
            raise ValueError('cached compatibility/production parity not complete: ' + name)
        if name == 'production_parity_receipt':
            verify_production_parity(evidence, plan.sha256, actual_sources, binding['environment'])
    prior = read_json(checked_ref(binding['prior_budget_receipt']))
    if (prior.get('status') != 'complete' or prior.get('plan_sha256') != plan.sha256
            or prior.get('variant') != launcher['variant']):
        raise ValueError('prior cumulative budget receipt mismatch')
    for key in ('wall_seconds', 'gpu_occupation_seconds'):
        if type(prior.get(key)) not in (int, float) or not math.isfinite(prior[key]) or prior[key] < 0:
            raise ValueError('invalid prior budget consumption')
    if prior.get('source_budget') is not None:
        old = read_json(checked_ref(prior['source_budget']))
        if old.get('plan_sha256') != plan.sha256 or old.get('status') in ('running', 'prediction_running', 'setup_running'):
            raise ValueError('prior source budget requires actor reconciliation')
        if any(prior[k] < old[k] for k in ('wall_seconds', 'gpu_occupation_seconds')):
            raise ValueError('prior budget consumption cannot decrease')
    elif not prior.get('reason') or prior['wall_seconds'] or prior['gpu_occupation_seconds']:
        raise ValueError('no prior GPU budget needs explicit zero-consumption reason')
    cfg = dict(cfg, output=str(core._checked_path(launcher['output'])), variant=launcher['variant'])
    core._check_config(cfg)
    return cfg, plan, ExecutionContext(launcher['execution_binding']['sha256'], lease, binding['train_cache_manifest'], prior)

class Budget:
    """One cumulative arm ledger, including setup and all later stages."""
    def __init__(self, path, plan, context, stage):
        self.path, self.plan, self.context, self.stage = Path(path), plan, context, stage
        self.start = time.monotonic()
        self.value = read_json(path) if self.path.exists() else self.fields() | {
            'status': 'new', 'wall_seconds': context.prior_consumption['wall_seconds'],
            'gpu_occupation_seconds': context.prior_consumption['gpu_occupation_seconds'], 'step': 0}
        self.verify(self.value)
        if self.value['status'] in ('running', 'prediction_running', 'setup_running'):
            raise RuntimeError('resource_incomplete: prior launcher budget requires reconciliation')
        self.prior = max(self.value['wall_seconds'], self.value.get('gpu_occupation_seconds', 0.))
        if not math.isfinite(self.prior) or self.prior < 0 or self.prior >= 86400:
            raise RuntimeError('resource_incomplete: invalid or exhausted cumulative arm budget')
        self.record('setup_running')

    def fields(self):
        return {'plan_sha256': self.plan.sha256,
                'execution_binding_sha256': self.context.binding_sha256}

    def verify(self, value):
        if any(value.get(k) != v for k, v in self.fields().items()):
            raise ValueError('budget belongs to another plan/execution binding')

    def elapsed(self):
        return self.prior + time.monotonic() - self.start

    def check(self):
        if self.elapsed() >= 86400:
            raise RuntimeError('resource_incomplete: cumulative arm wall/GPU budget exceeded')

    def record(self, status, **fields):
        self.value.update(self.fields() | fields | {'status': status, 'stage': self.stage,
            'wall_seconds': self.elapsed(), 'gpu_occupation_seconds': self.elapsed(), 'pid': os.getpid()})
        core.atomic_json(self.path, self.value)


def commit_checkpoint(path, model, optimizer, plan, context, *, step, variant, candidates,
                      cumulative_wall_seconds, recovery=True):
    """Original optimizer/RNG/sampler payload plus explicit new execution identity."""
    if variant not in core.VARIANTS or not 0 <= step <= plan.T:
        raise ValueError('invalid checkpoint arm/step')
    expected = [s for s in core.checkpoint_steps(plan.T) if s <= step]
    if [c['step'] for c in candidates] != expected:
        raise ValueError('candidate history incomplete at recovery commit')
    value = {'schema': SCHEMA, 'plan_sha256': plan.sha256,
             'execution_binding_sha256': context.binding_sha256,
             'initial_state_sha256': plan.payload['initial_state_sha256'], 'variant': variant,
             'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
             'rng': core._rng_state(), 'sampler': plan.sampler.counters(step),
             'candidates': candidates, 'cumulative_wall_seconds': cumulative_wall_seconds,
             'recovery_only': recovery}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)
    core.atomic_json(path.with_suffix('.receipt.json'), {'schema': SCHEMA, 'sha256': core.file_sha(path),
        'plan_sha256': plan.sha256, 'execution_binding_sha256': context.binding_sha256,
        'step': step, 'variant': variant, 'recovery_only': recovery})


def resume_exact(path, model, optimizer, plan, context, variant):
    receipt = read_json(Path(path).with_suffix('.receipt.json'))
    if (receipt.get('plan_sha256') != plan.sha256
            or receipt.get('execution_binding_sha256') != context.binding_sha256):
        raise ValueError('checkpoint receipt belongs to another execution; no implicit source migration')
    # The core verifies bytes, initial state, optimizer, RNG, sampler and candidate history.
    cp = core.resume_exact(path, model, optimizer, plan, variant)
    if (cp.get('execution_binding_sha256') != context.binding_sha256
            or cp['step'] != receipt['step'] or cp['variant'] != receipt['variant']
            or cp['recovery_only'] != receipt['recovery_only']):
        raise ValueError('checkpoint payload/receipt execution identity mismatch')
    return cp


@torch.no_grad()
def score_x0_tune_cached(model, variant, x0_units, monitor):
    was_training = model.training
    model.eval()
    try:
        scores = []
        for source in range(24, 32):
            values = []
            for unit in x0_units[source]:
                monitor()
                values.append(float(core.forward_unit(model, unit, variant)))
                monitor()
            scores.append(sum(values) / 7)
        result = sum(scores) / 8
        if not math.isfinite(result):
            raise ValueError('nonfinite complete X0 tune objective')
        return result
    finally:
        model.train(was_training)


def train_cached(cfg, plan, context, units, x0_units, budget):
    core.configure_runtime()
    device, baseline = admitted_device(plan, context)
    state = core.initial_state(cfg['seed'])
    if core.tensor_state_sha(state) != plan.payload['initial_state_sha256']:
        raise ValueError('paired initial tensor SHA mismatch')
    model = core.TempoRACResponse().float()
    model.load_state_dict(state, strict=True)
    model.to(device)
    optimizer = core.make_optimizer(model)
    out = Path(cfg['output']) / cfg['variant']
    recovery = out / 'recovery.pt'
    start, candidates = 0, []
    if recovery.exists():
        cp = resume_exact(recovery, model, optimizer, plan, context, cfg['variant'])
        start, candidates = cp['step'], cp['candidates']
        budget.prior = max(budget.prior, cp['cumulative_wall_seconds'])
    def monitor():
        budget.check()
        live, _ = resource_check(plan, context, starting=False, baseline=baseline)
        return live
    budget.record('running', step=start)
    for step in range(start + 1, plan.T + 1):
        live = monitor()
        draw = plan.payload['draw_ledger'][step - 1]
        receipt = core.optimizer_step(model, optimizer, x0_units[draw['x0_source']],
            units[draw['identity']], cfg['variant'], step, plan.T)
        monitor()
        with (out / 'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(asdict(receipt) | {'draw': draw, 'wall_seconds': budget.elapsed(),
                'plan_sha256': plan.sha256, 'execution_binding_sha256': context.binding_sha256,
                'peak_allocated': torch.cuda.max_memory_allocated(1),
                'peak_reserved': torch.cuda.max_memory_reserved(1),
                'gpu_process_inventory': live['inventory']}, sort_keys=True) + '\n')
        if step % 500 == 0:
            objective = score_x0_tune_cached(model, cfg['variant'], x0_units, monitor)
            candidates.append({'step': step, 'objective': objective, 'file': f'candidate_{step:06d}.pt'})
            commit_checkpoint(out / candidates[-1]['file'], model, optimizer, plan, context,
                step=step, variant=cfg['variant'], candidates=candidates,
                cumulative_wall_seconds=budget.elapsed(), recovery=False)
        commit_checkpoint(recovery, model, optimizer, plan, context, step=step,
            variant=cfg['variant'], candidates=candidates, cumulative_wall_seconds=budget.elapsed())
        budget.record('running', step=step)
    selected = core.select_checkpoint(candidates, plan.T)
    monitor()
    core.atomic_json(out / 'selection.json', {'schema': SCHEMA, 'status': 'complete_training',
        'plan_sha256': plan.sha256, 'execution_binding_sha256': context.binding_sha256,
        'variant': cfg['variant'], 'seed': cfg['seed'], 'selected': selected, 'candidates': candidates,
        'initial_state_sha256': plan.payload['initial_state_sha256'],
        'draw_ledger_sha256': plan.payload['draw_ledger_sha256']})
    budget.record('complete', step=plan.T)
    return 0


@torch.no_grad()
def predict_population_cached(model, identities, variant):
    """Same reference reconstruction and once-only decoder; no target/cue API."""
    if variant not in core.VARIANTS:
        raise ValueError('unknown prediction variant')
    model.eval()
    device = next(model.parameters()).device
    for key, identity, arrays in identities:
        support = identity.response_edge_support
        if support.any():
            windows = tuple((int(row[1]), int(row[2])) for row in arrays['windows'])
            gates = arrays['local_gates'] if variant == 'canonical' else np.full_like(arrays['local_gates'], 1 / 3)
            values = torch.tensor(identity.response_input, device=device, dtype=torch.float32)
            branch = model.cached_responses(values, identity.run_bounds).edge_probabilities.cpu().numpy()
            response = core.reconstruct_native_response(branch.astype(np.float64), gates, windows, support).response
        else:
            response = np.zeros(len(support), np.float64)
        yield key, core.decode_native_identity(response, identity)


def predict_cached(cfg, plan, context, dev_manifest, budget):
    from pams.temporac.cache_build_v1 import iter_dev_cache
    dev_path = checked_ref(dev_manifest)
    out = Path(cfg['output']) / cfg['variant']
    selection = read_json(out / 'selection.json')
    if (selection.get('plan_sha256') != plan.sha256
            or selection.get('execution_binding_sha256') != context.binding_sha256
            or selection['status'] != 'complete_training'):
        raise ValueError('predict requires complete selection from this plan/execution')
    selected = core.select_checkpoint(selection['candidates'], plan.T)
    if selected != selection['selected']:
        raise ValueError('selection minimum/tie mismatch')
    expected_keys = {r['key'] for r in plan.payload['population'] if r['split'] == 'val'}
    prediction_binding = {'plan_sha256': plan.sha256, 'execution_binding_sha256': context.binding_sha256,
                          'dev_cache_manifest_sha256': dev_manifest['sha256'], 'selected': selected}
    progress_path = out / 'prediction_progress.json'
    predictions = []
    if progress_path.exists():
        progress = read_json(progress_path)
        if any(progress.get(k) != v for k, v in prediction_binding.items()):
            raise ValueError('prediction progress belongs to different execution/dev cache/selection')
        predictions = progress['records']
        keys = [r['key'] for r in predictions]
        if len(set(keys)) != len(keys) or not set(keys) <= expected_keys:
            raise ValueError('duplicate/unknown completed prediction identity')
        for record in predictions:
            if core.file_sha(out / 'predictions' / (record['key'] + '.npz')) != record['sha256']:
                raise ValueError('completed prediction bytes changed')
    completed = {r['key'] for r in predictions}
    # Iterate/verify all records before admitting GPU, preserving feature-only API.
    identities = list(iter_dev_cache(cfg, plan.payload, dev_path))
    if len(identities) != 486 or {key for key, _, _ in identities} != expected_keys:
        raise ValueError('complete exact 486-identity dev cache required before prediction')
    core.configure_runtime()
    device, baseline = admitted_device(plan, context)
    model = core.TempoRACResponse().float().to(device)
    optimizer = core.make_optimizer(model)
    cp = resume_exact(out / selected['file'], model, optimizer, plan, context, cfg['variant'])
    if cp['step'] != selected['step'] or cp['recovery_only']:
        raise ValueError('selected candidate checkpoint step/type mismatch')
    def monitor():
        budget.check()
        live, _ = resource_check(plan, context, starting=False, baseline=baseline)
        budget.record('prediction_running', current_inventory=live['inventory'])
    def pending():
        for item in identities:
            if item[0] not in completed:
                monitor()
                yield item
    for key, pred in predict_population_cached(model, pending(), cfg['variant']):
        monitor()
        path = out / 'predictions' / (key + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, response=pred.response, support=pred.support,
                run_bounds=pred.run_bounds, component_bounds=pred.component_bounds,
                event_source_positions=pred.event_source_positions, event_seconds=pred.event_seconds)
        os.replace(temporary, path)
        predictions.append({'key': key, 'count': pred.count, 'status': pred.status,
            'decoder_invocations': pred.decoder_invocations, 'sha256': core.file_sha(path),
            'period_ap_status': pred.period_ap_status})
        core.atomic_json(progress_path, prediction_binding | {'schema': SCHEMA,
            'status': 'partial_population', 'records': predictions})
    if {p['key'] for p in predictions} != expected_keys or len(predictions) != 486:
        raise ValueError('complete exact full dev prediction required')
    monitor()
    core.atomic_json(out / 'predictions.json', prediction_binding | {'schema': SCHEMA,
        'status': 'complete_population', 'records': predictions, 'natural_GT_accessed': False})
    budget.record('complete_prediction')
    return 0


def execute(launcher, command, *, dev_manifest=None):
    if command not in ('validate-cached', 'train-cached', 'predict-cached'):
        raise ValueError('unknown explicit cached command')
    cfg, plan, context = load_execution(launcher)
    require_train(plan, context)
    from pams.temporac.cache_build_v1 import load_train_cache
    # Prediction never opens natural supervision or the train cache.
    if command == 'predict-cached':
        units, x0_units = {}, {}
    else:
        units, x0_units = load_train_cache(cfg, plan.payload, checked_ref(context.train_manifest))
    eligible = {key for values in plan.payload['eligible_by_component'].values() for key in values}
    if command != 'predict-cached' and (set(units) != eligible or set(x0_units) != set(range(32)) or any(len(x0_units[s]) != 7 for s in range(32))):
        raise ValueError('train-ready cache must contain exact 1084 natural identities and 224 X0 views')
    if command != 'predict-cached' and any(u.kind != 'natural' or u.key != key for key, u in units.items()):
        raise ValueError('natural prepared unit identity/type mismatch')
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'frozen_plan.json'
    if frozen.exists():
        if read_json(frozen) != plan.payload:
            raise ValueError('existing frozen scientific plan differs')
    else:
        core.atomic_json(frozen, plan.payload)
        core.atomic_json(output / 'paired_draw_ledger.json', plan.payload['draw_ledger'])
    out = output / cfg['variant']
    out.mkdir(parents=True, exist_ok=True)
    binding_record = {'plan_sha256': plan.sha256, 'execution_binding_sha256': context.binding_sha256}
    if command == 'validate-cached':
        core.atomic_json(out / 'cached_input_validation.json', binding_record | {'schema': BINDING_SCHEMA,
            'status': 'complete', 'kind': 'cached_input_compatibility_validation',
            'parent_plan_not_recomputed': True, 'natural_train_units': len(units), 'X0_views': 224})
        return 0
    budget = Budget(out / 'budget.json', plan, context, command)
    try:
        if command == 'train-cached':
            return train_cached(cfg, plan, context, units, x0_units, budget)
        if dev_manifest is None:
            raise ValueError('predict-cached requires immutable dev manifest path/SHA')
        return predict_cached(cfg, plan, context, dev_manifest, budget)
    except BaseException as exc:
        budget.record('resource_incomplete')
        progress_path = out / 'prediction_progress.json'
        progress = read_json(progress_path) if progress_path.exists() else {}
        core.atomic_json(out / ('prediction_failure.json' if command == 'predict-cached' else 'failure.json'),
            binding_record | {'schema': SCHEMA, 'status': 'resource_incomplete', 'stage': command,
            'error': str(exc), 'wall_seconds': budget.elapsed(), 'last_recovery': str(out / 'recovery.pt'),
            'progress_manifest': str(progress_path), 'completed_prediction_records': progress.get('records', [])})
        raise
