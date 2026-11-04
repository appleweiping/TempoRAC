"""NTR1: full native identities, paired cycles, vectorized detached NOLA and LF1.

The training sum changes floating-point operation order relative to the scalar
Neumaier reference. Prediction always uses the NB1 float64 reference followed by
its single float32 quantization / single identity decoder call.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from pams.temporac.response import TempoRACResponse
from pams.temporac.objective_lf1 import traversal_identity_loss
from pams.temporac.native_bridge_v1 import (load_native_feature, preprocess_native,
    reconstruct_native_response, decode_native_identity, window_taper)
from pams.temporac.native_time_cue_v1 import compute_native_routing
from pams.temporac.native_target_nt1 import load_native_target, lf1_supervision
from pams.temporac.native_x0_adapter_v1 import (X0AnalyticAdapterNTR1, validate_x0_bank,
    file_sha, json_sha, atomic_json)

SCHEMA = 'temporac.native-training.ntr1'
SEEDS = (20260815, 20260816, 20260817)
VARIANTS = ('canonical', 'capacity-control')
GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = ('raw', 'decode', 'clock_overlay', 'nb1', 'nt1', 'diagnostics', 'count_policy', 'continuation')
CONFIG_KEYS = {'census', 'bridge_root', 'target_root', 'component_map', 'x0_root',
               'artifacts', 'output', 'seed', 'variant', 'resource'}


def _readonly(value, dtype=None):
    a = np.array(value, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


def seed_for(seed, purpose, *parts):
    if seed not in SEEDS:
        raise ValueError('NTR1 seed outside frozen three-seed matrix')
    data = json.dumps([SCHEMA, 'temporac.ntr1.paired', seed, purpose, *parts],
                      separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'little')


def cycle_item(values, index, *, seed, purpose, key=''):
    values = tuple(sorted(values))
    if not values or type(index) is not int or index < 0:
        raise ValueError('cycle requires nonempty population and nonnegative integer visit')
    cycle, offset = divmod(index, len(values))
    rng = np.random.Generator(np.random.PCG64(seed_for(seed, purpose, key, cycle)))
    return values[int(rng.permutation(len(values))[offset])]


@dataclass(frozen=True)
class PairedSampler:
    eligible_by_component: dict
    seed: int

    def __post_init__(self):
        if self.seed not in SEEDS or not any(self.eligible_by_component.values()):
            raise ValueError('paired sampler needs frozen seed and eligible natural family')
        ids = [i for v in self.eligible_by_component.values() for i in v]
        if len(ids) != len(set(ids)):
            raise ValueError('identity cannot have multiple component votes')

    @property
    def components(self):
        return tuple(sorted(c for c, ids in self.eligible_by_component.items() if ids))

    def draw(self, step):
        """Zero-based step; component visits need no mutable sampler RNG state."""
        component = cycle_item(self.components, step, seed=self.seed, purpose='component')
        # Exactly one visit per component in every complete component cycle.
        visit = step // len(self.components)
        identity = cycle_item(self.eligible_by_component[component], visit,
                              seed=self.seed, purpose='identity', key=component)
        source = cycle_item(tuple(range(24)), step, seed=self.seed, purpose='X0')
        return {'step': step + 1, 'x0_source': source, 'component': component, 'identity': identity}

    def counters(self, completed_steps):
        cycle, remainder = divmod(completed_steps, len(self.components))
        counts = {c: cycle for c in self.components}
        for j in range(remainder):
            counts[cycle_item(self.components, cycle * len(self.components) + j,
                              seed=self.seed, purpose='component')] += 1
        return {'completed_steps': completed_steps, 'component_visits': dict(sorted(counts.items()))}


def horizon(eligible):
    values = [len(v) for v in eligible.values() if v]
    if not values:
        raise ValueError('C=0: natural family unavailable')
    return 500 * math.ceil(max(10000, len(values) * max(values)) / 500)


def learning_rate(step, total_steps):
    if (type(step) is not int or type(total_steps) is not int or total_steps < 10000
            or total_steps % 500 or not 1 <= step <= total_steps):
        raise ValueError('invalid full NTR1 horizon/step')
    warmup = math.ceil(.05 * total_steps)
    if step <= warmup:
        return float(1e-3 * step / warmup)
    phase = math.pi * (step - warmup) / (total_steps - warmup)
    return float(1e-3 * (.1 + .9 * (1 + math.cos(phase)) / 2))


def checkpoint_steps(total_steps):
    learning_rate(1, total_steps)
    return tuple(range(500, total_steps + 1, 500))


def select_checkpoint(candidates, total_steps):
    if tuple(c['step'] for c in candidates) != checkpoint_steps(total_steps):
        raise ValueError('missing/extra/out-of-order selection candidates')
    if any(not math.isfinite(c['objective']) for c in candidates):
        raise ValueError('NaN/nonfinite tune candidate')
    return min(candidates, key=lambda c: (c['objective'], c['step']))


def tensor_state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode('ascii') + b'\0' + a.dtype.str.encode() + b'\0')
        h.update(json.dumps(list(a.shape)).encode() + b'\0' + a.tobytes())
    return h.hexdigest()


def initial_state(seed):
    """One CPU state per seed, no variant in the RNG preimage."""
    model = TempoRACResponse().cpu().float()
    generator = torch.Generator(device='cpu').manual_seed(seed_for(seed, 'init'))
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters(), key=lambda kv: kv[0].encode('ascii')):
            if not name.isascii():
                raise ValueError('initialization requires ASCII parameter names')
            owner = model.get_submodule(name.rsplit('.', 1)[0])
            leaf = name.rsplit('.', 1)[1]
            if isinstance(owner, torch.nn.LayerNorm):
                parameter.fill_(1 if leaf == 'weight' else 0)
            else:
                if isinstance(owner, torch.nn.Linear):
                    fan_in = owner.in_features
                elif isinstance(owner, torch.nn.Conv1d):
                    fan_in = owner.in_channels * owner.kernel_size[0] // owner.groups
                else:
                    raise ValueError('unfrozen initializer: ' + name)
                bound = 1 / math.sqrt(fan_in)
                parameter.uniform_(-bound, bound, generator=generator)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@dataclass(frozen=True)
class NOLACache:
    denominator: np.ndarray
    responsibility: np.ndarray
    support: np.ndarray


def precompute_nola(windows, gates, support):
    """Full response support only; no target argument exists on this interface."""
    m = np.asarray(support)
    a = np.asarray(gates, np.float64)
    if (m.ndim != 1 or m.dtype != np.bool_ or a.shape != (len(windows), 3)
            or not np.isfinite(a).all() or np.any(a < 0)
            or not np.allclose(a.sum(1), 1, atol=1e-12, rtol=0)):
        raise ValueError('NOLA cache support/gates invalid')
    d = np.zeros(len(m), np.float64)
    correction = np.zeros_like(d)
    numerator = np.zeros((len(m), 3), np.float64)
    ncorrection = np.zeros_like(numerator)
    for window, gate in zip(windows, a):
        start, stop = (window.edge_start, window.edge_stop) if hasattr(window, 'edge_start') else window
        if not 0 <= start < stop <= len(m):
            raise ValueError('NOLA window outside full edge domain')
        h = window_taper(stop - start) * m[start:stop]
        for total, corr, value in ((d[start:stop], correction[start:stop], h),
                (numerator[start:stop], ncorrection[start:stop], h[:, None] * gate)):
            updated = total + value
            corr += np.where(np.abs(total) >= np.abs(value), (total - updated) + value,
                             (value - updated) + total)
            total[:] = updated
    d += correction
    numerator += ncorrection
    if np.any(d[m] < 1e-3) or np.any(d[~m] != 0):
        raise ValueError('NOLA denominator below 1e-3 on response support')
    pi = np.zeros_like(numerator)
    pi[m] = numerator[m] / d[m, None]
    return NOLACache(_readonly(d), _readonly(pi), _readonly(m))


def vector_fusion(probabilities, cache):
    if probabilities.shape != cache.responsibility.shape:
        raise ValueError('full-domain branch probability shape required')
    pi = torch.tensor(cache.responsibility, device=probabilities.device, dtype=torch.float64)
    return (probabilities.double() * pi).sum(1)


@dataclass(frozen=True)
class PreparedUnit:
    key: str
    response_input: np.ndarray
    run_bounds: np.ndarray
    windows: tuple
    gates: np.ndarray
    cache: NOLACache
    pulse: np.ndarray
    chi: np.ndarray
    loss_mask: np.ndarray
    traversal_bounds: np.ndarray
    kind: str
    identity: object = None


def legal_lf1_supervision(identity, target):
    """Filter only complete legal LF1 traversals; preserve original NT1 payload."""
    raw = lf1_supervision(identity, target, expected_input_sha256=target.metadata['input_sha256'])
    bounds, ledger = [], []
    mask = np.zeros_like(raw['edge_mask'], dtype=bool)
    for index, (left, stop) in enumerate(raw['traversal_bounds']):
        w, pulse = raw['chi'][left:stop], raw['pulse'][left:stop]
        wp, wn = float((w * pulse).sum()), float((w * (1 - pulse)).sum())
        ok = math.isfinite(wp) and math.isfinite(wn) and wp > 0 and wn > 0
        ledger.append({'index': index, 'bounds': [int(left), int(stop)], 'eligible': ok,
                       'positive_mass': wp, 'negative_mass': wn,
                       'reason': None if ok else 'missing_positive_or_negative_LF1_mass'})
        if ok:
            bounds.append([left, stop])
            mask[left:stop] = True
    # Soft-pi/head mass is deliberately NOT used to exclude traversals. Any
    # numerical responsibility underflow still raises inside the LF1 criterion.
    return dict(raw, edge_mask=mask, traversal_bounds=np.array(bounds, np.int64).reshape(-1, 2),
                eligibility_ledger=ledger)


def prepare_natural_unit(identity, target):
    routing = compute_native_routing(identity)  # FULL NB1 response support, before LF1.
    cache = precompute_nola(routing.windows, routing.local_gates, identity.response_edge_support)
    supervision = legal_lf1_supervision(identity, target)
    if not len(supervision['traversal_bounds']):
        raise ValueError('identity has no legal complete LF1 traversal')
    key = identity.feature.opaque_key.tobytes().hex() + '_' + str(int(identity.feature.slot[0]))
    return PreparedUnit(key, identity.response_input, identity.run_bounds, routing.windows,
        routing.local_gates, cache, supervision['pulse'], supervision['chi'],
        supervision['edge_mask'], supervision['traversal_bounds'], 'natural', identity)


def prepare_x0_unit(source, adapter=None):
    adapter = adapter or X0AnalyticAdapterNTR1()
    out = []
    for unit in adapter.source(source):
        cache = precompute_nola(unit.windows, unit.gates, unit.support)
        if not np.allclose(cache.responsibility, unit.responsibility, atol=1e-12, rtol=0):
            raise ValueError('analytic edge responsibility / full NOLA mismatch')
        out.append(PreparedUnit(f'U{source:04d}_B{unit.block_index}', unit.response_input,
            unit.run_bounds, unit.windows, unit.gates, cache, unit.pulse, unit.chi,
            unit.target_mask.astype(bool), unit.traversal_bounds, 'X0'))
    return tuple(out)


def forward_unit(model, unit, variant):
    if variant not in VARIANTS:
        raise ValueError('unknown NTR1 variant')
    device = next(model.parameters()).device
    values = torch.tensor(unit.response_input, device=device, dtype=torch.float32)
    branch = model.cached_responses(values, unit.run_bounds).edge_probabilities
    if variant == 'capacity-control':
        fused = branch.mean(1)  # probability mean, never logits mean.
        pi = np.full_like(unit.cache.responsibility, 1 / 3)
    else:
        fused = vector_fusion(branch, unit.cache).float()
        pi = unit.cache.responsibility
    def t(a):
        return torch.tensor(a, device=device)
    return traversal_identity_loss(branch, fused, t(unit.pulse), t(unit.loss_mask), t(unit.chi),
        t(pi), unit.traversal_bounds, capacity_control=variant == 'capacity-control').total


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                             weight_decay=1e-4, foreach=False, fused=False)


@dataclass(frozen=True)
class StepReceipt:
    step: int
    loss: float
    pre_clip_global_norm: float
    learning_rate: float
    backward_calls: int = 8
    optimizer_steps: int = 1


def optimizer_step(model, optimizer, x0_units, natural_unit, variant, step, total_steps):
    if len(x0_units) != 7 or natural_unit.kind != 'natural' or any(u.kind != 'X0' for u in x0_units):
        raise ValueError('one whole natural identity plus all seven X0 blocks required')
    lr = learning_rate(step, total_steps)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for unit, weight in [(u, .5 / 7) for u in x0_units] + [(natural_unit, .5)]:
        loss = forward_unit(model, unit, variant) * weight
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite LF1; step not committed')
        loss.backward()
        total += float(loss.detach())
        del loss
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
    optimizer.step()
    return StepReceipt(step, total, float(norm), lr)


@torch.no_grad()
def score_x0_tune(model, variant, adapter):
    was_training = model.training
    model.eval()
    try:
        scores = [sum(float(forward_unit(model, unit, variant)) for unit in
                      prepare_x0_unit(source, adapter)) / 7 for source in range(24, 32)]
        result = sum(scores) / 8
        if not math.isfinite(result):
            raise ValueError('nonfinite complete X0 tune objective')
        return result
    finally:
        model.train(was_training)


@dataclass(frozen=True)
class FrozenPlan:
    payload: dict

    @property
    def sha256(self):
        return json_sha(self.payload)

    @property
    def T(self):
        return self.payload['T']

    @property
    def sampler(self):
        return PairedSampler(self.payload['eligible_by_component'], self.payload['seed'])


def paired_draw(plan, seed, step):
    if seed != plan.payload['seed']:
        raise ValueError('paired draw seed differs from frozen plan')
    return plan.sampler.draw(step)


def memory_envelope(max_N, max_x0_N, parameter_count=55235):
    if min(max_N, max_x0_N) < 0:
        raise ValueError('negative full identity length')
    state_bytes = parameter_count * 4 * 4
    return {'parameter_optimizer_bytes': state_bytes,
            'B_plan': 2 * GIB + 128 * 1024 * max(max_N, max_x0_N) + state_bytes,
            'planning_only_not_workspace_upper_bound': True,
            'tensor_lifetime': [
                {'name': 'response_input', 'shape': [max_N, 269], 'dtype': 'float32', 'lifetime': 'one identity'},
                {'name': 'run_activations', 'shape': [max_N, 64, 8], 'dtype': 'float32',
                 'lifetime': 'all runs of one identity retained through backward'},
                {'name': 'pi', 'shape': [max_N - 1, 3], 'dtype': 'float64', 'lifetime': 'one unit'},
                {'name': 'optimizer_parameter_gradient', 'shape': [parameter_count, 4],
                 'dtype': 'float32', 'lifetime': 'whole job'}]}


def admission(resource, envelope, *, starting=True):
    required = ('lease_id', 'device_uuid', 'environment_sha', 'owned_job_recovery_receipt',
                'free_vram', 'free_disk', 'other_process_inventory', 'gpu_index')
    if any(resource.get(k) is None for k in required) or not resource['lease_id']:
        raise ValueError('unknown/pending shared-resource lease')
    if resource['gpu_index'] != 1:
        raise ValueError('NTR1 only authorizes GPU1')
    if envelope['B_plan'] > 6 * GIB:
        raise ValueError('CPU memory envelope exceeds frozen 6GiB cap; no truncation permitted')
    if resource['free_vram'] < (8 if starting else 2) * GIB:
        raise ValueError('actual free VRAM below R2 headroom')
    # The presence of other PIDs is explicitly allowed; SIGSTOP is not free memory.
    return True


def build_training_plan(population, targets, x0_manifest, config):
    """Population must retain every full native identity including ineligible ones."""
    counts = Counter(r['split'] for r in population)
    videos = {split: len({r['media_key'] for r in population if r['split'] == split})
              for split in ('train', 'val')}
    if counts != {'train': 1878, 'val': 486} or videos != {'train': 794, 'val': 183}:
        raise ValueError('production requires complete frozen 794/183 videos and 2364 identities')
    if sum(r['N'] for r in population) != 1758034 or len({r['key'] for r in population}) != 2364:
        raise ValueError('production full person-frame/opaque identity census mismatch')
    if set(targets) != {r['key'] for r in population}:
        raise ValueError('target index must include all ineligible identities')
    if (len(x0_manifest['sources']) != 40 or len(x0_manifest['views']) != 224
            or [r['split'] for r in x0_manifest['sources']] != ['train'] * 24 + ['tune'] * 8 + ['heldout'] * 8):
        raise ValueError('X0 requires exact 40/24/8/8 and 224 views')
    eligible = {}
    for row in population:
        if row['split'] == 'train':
            eligible.setdefault(row['component'], [])
            if targets[row['key']]['eligible']:
                eligible[row['component']].append(row['key'])
    eligible = {c: sorted(ids) for c, ids in sorted(eligible.items())}
    total_steps = horizon(eligible)
    sampler = PairedSampler(eligible, config['seed'])
    ledger = [sampler.draw(s) for s in range(total_steps)]
    exposures = dict(sorted(Counter(r['identity'] for r in ledger).items()))
    natural_rows = {r['key']: r['N'] for r in population}
    xrows = {(r['source_id'], r['block']): r['arrays']['response_input']['shape'][0]
             for r in x0_manifest['views']}
    max_train_N = max(r['N'] for r in population if r['split'] == 'train')
    max_dev_N = max(r['N'] for r in population if r['split'] == 'val')
    max_N = max(max_train_N, max_dev_N)
    max_run = max((r['max_run'] for r in population if r['split'] == 'train'), default=0)
    xmax = max(xrows.values())
    memory = memory_envelope(max_N, xmax)
    cache_bytes = sum((r['N'] - 1) * (8 * 4 + 1) for r in population)
    scheduled_rows = sum(natural_rows[r['identity']] + sum(xrows[r['x0_source'], b] for b in range(7))
                         for r in ledger)
    tune_rows = sum(xrows[s, b] for s in range(24, 32) for b in range(7))
    checkpoint_bytes = memory['parameter_optimizer_bytes'] + 1024 * 1024
    disk_required = 2 * (checkpoint_bytes * (len(checkpoint_steps(total_steps)) + 1)
                        + cache_bytes + sum(r['N'] * 64 for r in population)) + 10 * GIB
    payload = {'schema': SCHEMA, 'seed': config['seed'], 'T': total_steps,
        'C': len(sampler.components), 'n_c': {c: len(ids) for c, ids in eligible.items()},
        'eligible_by_component': eligible, 'population': population, 'targets': targets,
        'population_binding': config['population_binding'], 'max_N': max_N, 'max_run': max_run,
        'max_train_identity_N': max_train_N, 'max_dev_identity_N': max_dev_N,
        'max_X0_block_N': xmax, 'planned_tensor_bytes': memory, 'B_plan': memory['B_plan'],
        'exposures': exposures, 'draw_ledger': ledger, 'draw_ledger_sha256': json_sha(ledger),
        'scheduled_rows': scheduled_rows, 'tune_rows_per_candidate': tune_rows,
        'dev_prediction_rows': sum(r['N'] for r in population if r['split'] == 'val'),
        'cache_bytes': cache_bytes, 'required_free_disk': disk_required,
        'selection': 'all_8_tune_sources_7_blocks_LF1_min_finite_tie_earlier',
        'candidate_steps': list(checkpoint_steps(total_steps)), 'initial_state_sha256': tensor_state_sha(initial_state(config['seed'])),
        'floating_point': 'vectorized_training_probability_NOLA_not_bit_identical; NB1_reference_prediction',
        'pending': config.get('pending', []), 'resource': config['resource']}
    return FrozenPlan(payload)


def source_binding():
    names = ('native_training_runtime_v1', 'native_x0_adapter_v1', 'native_bridge_v1',
             'native_time_cue_v1', 'native_target_nt1', 'objective_lf1', 'response', 'preprocess',
             'cue', 'nola', 'quadrature', 'decode', 'native_target_diagnostics_v1')
    paths = [ROOT / f'src/pams/temporac/{n}.py' for n in names]
    paths += [ROOT / 'scripts/train_temporac_native_response.py']
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def environment_binding():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}


def configure_runtime():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    # CPU engineering receipts use native convolution: the available Torch2.0.1
    # oneDNN backend produced corrupted depthwise backward gradients in this suite.
    # This CPU backend selection does not modify CUDA/cuDNN kernels.
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(min(4, max(1, int(os.environ.get('OMP_NUM_THREADS', '4')))))


def _rng_state():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_rng(rng):
    torch.set_rng_state(rng['torch'].cpu())
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in rng['cuda']])


def commit_checkpoint(path, model, optimizer, plan, *, step, variant, candidates,
                      cumulative_wall_seconds=0., recovery=True):
    if variant not in VARIANTS or not 0 <= step <= plan.T:
        raise ValueError('invalid checkpoint arm/step')
    expected = [s for s in checkpoint_steps(plan.T) if s <= step]
    if [c['step'] for c in candidates] != expected:
        raise ValueError('candidate history incomplete at recovery commit')
    value = {'schema': SCHEMA, 'plan_sha256': plan.sha256,
             'initial_state_sha256': plan.payload['initial_state_sha256'], 'variant': variant,
             'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
             'rng': _rng_state(), 'sampler': plan.sampler.counters(step),
             'candidates': candidates, 'cumulative_wall_seconds': cumulative_wall_seconds,
             'recovery_only': recovery}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)
    atomic_json(path.with_suffix('.receipt.json'), {'schema': SCHEMA, 'sha256': file_sha(path),
                'plan_sha256': plan.sha256, 'step': step, 'variant': variant, 'recovery_only': recovery})


def resume_exact(path, model, optimizer, plan, variant):
    receipt = json.loads(Path(path).with_suffix('.receipt.json').read_text(encoding='utf-8'))
    if receipt['sha256'] != file_sha(path):
        raise ValueError('checkpoint payload SHA changed')
    cp = torch.load(path, map_location=next(model.parameters()).device)
    if (cp['schema'] != SCHEMA or cp['plan_sha256'] != plan.sha256 or cp['variant'] != variant
            or cp['initial_state_sha256'] != plan.payload['initial_state_sha256']
            or cp['sampler'] != plan.sampler.counters(cp['step'])):
        raise ValueError('exact resume input/code/config/initial/sampler binding mismatch')
    expected = [s for s in checkpoint_steps(plan.T) if s <= cp['step']]
    if [c['step'] for c in cp['candidates']] != expected:
        raise ValueError('resume selection candidates incomplete')
    model.load_state_dict(cp['model'], strict=True)
    optimizer.load_state_dict(cp['optimizer'])
    _restore_rng(cp['rng'])
    return cp


@torch.no_grad()
def predict_population(model, identities, variant):
    """Every supplied identity, including abstentions; no target API on this path."""
    if variant not in VARIANTS:
        raise ValueError('unknown prediction variant')
    model.eval()
    device = next(model.parameters()).device
    for key, identity in identities:
        support = identity.response_edge_support
        if support.any():
            routing = compute_native_routing(identity)
            values = torch.tensor(identity.response_input, device=device, dtype=torch.float32)
            branch = model.cached_responses(values, identity.run_bounds).edge_probabilities.cpu().numpy()
            gates = routing.local_gates if variant == 'canonical' else routing.uniform_gates
            response = reconstruct_native_response(branch.astype(np.float64), gates,
                                                    routing.windows, support).response
        else:
            response = np.zeros(len(support), np.float64)
        yield key, decode_native_identity(response, identity)


def _module_from_script(name):
    spec = importlib.util.spec_from_file_location('ntr1_readonly_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checked_path(path):
    p = Path(path).resolve()
    if os.name == 'nt':
        if p.drive.upper() != 'E:':
            raise ValueError('NTR1 project artifacts and caches must remain on E:')
    elif not (str(p) == '/mnt/4tssd/lilei/TempoRAC' or str(p).startswith('/mnt/4tssd/lilei/TempoRAC/')):
        raise ValueError('NTR1 artifacts must remain in verified server project root')
    return p


def _check_config(cfg):
    if set(cfg) != CONFIG_KEYS or cfg['seed'] not in SEEDS or cfg['variant'] not in VARIANTS:
        raise ValueError('exact NTR1 config schema required; no caps/fixtures/overrides')
    for key in ('census', 'bridge_root', 'target_root', 'component_map', 'x0_root', 'output'):
        _checked_path(cfg[key])
    out = Path(cfg['output']).resolve()
    for key in ('bridge_root', 'target_root', 'x0_root'):
        source = Path(cfg[key]).resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError('training output must be separate from immutable inputs')
    if set(cfg['artifacts']) != set(RECEIPTS):
        raise ValueError('all full-input / diagnostic / count / continuation receipts required')
    for artifact in cfg['artifacts'].values():
        if set(artifact) != {'path', 'sha256'}:
            raise ValueError('each receipt requires explicit path and expected SHA256')
        _checked_path(artifact['path'])


def qualified_receipt(name, payload):
    """Consume actual frozen producer formats, retaining full population gates."""
    if name == 'raw':
        rows = payload.get('records', [])
        return (payload.get('videos') == 977 and payload.get('persons') == 2364
                and len(rows) == 977 and all(r.get('status') == 'complete' for r in rows)
                and sum(r.get('persons', 0) for r in rows) == 2364
                and sum(r.get('annotation_length', 0) for r in rows) == 695625
                and all(r.get('annotation_length') == r.get('decoded_length') for r in rows))
    if name == 'decode':
        return (payload.get('schema_version') == 'temporac.decode-audit.v1'
                and payload.get('status') == 'complete' and payload.get('completed') == 977
                and payload.get('decoded') == 977 and payload.get('findings_count') == 0)
    if name == 'clock_overlay':
        return (payload.get('schema') == 'temporac.native-clock-overlay.v1'
                and payload.get('status') == 'frozen_complete_census'
                and payload.get('videos') == 977 and payload.get('identities') == 2364
                and len(payload.get('records', [])) == 977
                and not any(r.get('selection') == 'rejected' for r in payload['records']))
    if name in ('nb1', 'nt1'):
        schema = 'temporac.native-learner.v1' if name == 'nb1' else 'temporac.native-target.nt1'
        terminal = 'validated_population' if name == 'nb1' else 'complete_population'
        return (payload.get('schema') == schema and payload.get('status') == terminal
                and payload.get('videos') == 977 and payload.get('identities') == 2364
                and len(payload.get('records', [])) == 2364 and not payload.get('failures'))
    if name == 'diagnostics':
        return (payload.get('schema') == 'temporac.native-target-diagnostics.v1'
                and all(payload.get('splits', {}).get(split, {}).get('identity_denominator') == ids
                        and payload['splits'][split].get('full_population', {}).get('videos') == videos
                        for split, ids, videos in (('train', 1878, 794), ('val', 486, 183))))
    if name == 'continuation':
        return payload.get('decision') == 'continue_full_method_test' and bool(payload.get('reason'))
    if name == 'count_policy':
        return (bool(payload.get('metric_name')) and payload.get('split') == 'val'
                and bool(payload.get('failure_policy')) and payload.get('threshold') == .5)
    return False


def _artifact_receipts(cfg):
    bound, pending = {}, []
    for name, artifact in cfg['artifacts'].items():
        path = Path(artifact['path'])
        if not path.is_file():
            pending.append(name + '_receipt_missing')
            continue
        if file_sha(path) != artifact['sha256']:
            raise ValueError('frozen receipt changed: ' + name)
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        bound[name] = artifact
        if not qualified_receipt(name, payload):
            pending.append(name + '_not_qualified_full_population')
    return bound, pending


def _identity_record(cfg, row, person):
    stem = f"{row['media_key']}_{person['slot']}"
    base = Path(cfg['bridge_root']) / 'identities' / row['split'] / stem
    feature_path = base.with_suffix('.npz')
    receipt = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
    if (receipt['feature_sha256'] != file_sha(feature_path)
            or receipt['derived_sha256'] != file_sha(base.with_suffix('.derived.npz'))):
        raise ValueError('NB1 feature/derived artifact changed')
    feature = load_native_feature(feature_path)
    if (int(feature.source_length[0]) != row['length'] or int(feature.slot[0]) != person['slot']
            or feature.opaque_key.tobytes().hex() != row['media_key']):
        raise ValueError('opaque feature identity / source frame census mismatch')
    identity = preprocess_native(feature)
    builder = _module_from_script('build_temporac_native_learner_bridge')
    with np.load(base.with_suffix('.derived.npz'), allow_pickle=False) as derived:
        expected = builder.derived_arrays(identity)
        if set(derived.files) != set(expected):
            raise ValueError('NB1 derived schema mismatch')
        for key, value in expected.items():
            if not np.array_equal(derived[key], value, equal_nan=value.dtype.kind in 'fc'):
                raise ValueError('NB1 derived numerical mismatch: ' + key)
    target_path = Path(cfg['target_root']) / 'identities' / row['split'] / (stem + '.npz')
    target = load_native_target(target_path, expected_input_sha256=file_sha(feature_path))
    if target.metadata['bridge_sha256'] != json_sha(receipt['binding']):
        raise ValueError('NT1 target bound to different NB1/overlay provenance')
    lf1_supervision(identity, target, expected_input_sha256=file_sha(feature_path))
    return identity, target, {'feature_sha256': file_sha(feature_path),
        'bridge_receipt_sha256': file_sha(base.with_suffix('.json')),
        'derived_sha256': file_sha(base.with_suffix('.derived.npz')),
        'target_sha256': file_sha(target_path), 'target_receipt_sha256': file_sha(target_path.with_suffix('.json'))}


def _load_inputs(cfg, *, retain_units=False):
    census = _module_from_script('build_temporac_native_learner_bridge').read_census(cfg['census'])
    from pams.temporac.native_target_diagnostics_v1 import source_membership
    components, aliases = source_membership(census,
        json.loads(Path(cfg['component_map']).read_text(encoding='utf-8-sig')))
    if len(aliases) != 4 or any(any(v['split'] != 'train' for v in group) for group in aliases):
        raise ValueError('frozen four train same-media alias groups must remain in components')
    population, targets, units, pending = [], {}, {}, []
    for row in census:
        for person in row['persons']:
            key = f"{row['media_key']}_{person['slot']}"
            paths = [Path(cfg['bridge_root']) / 'identities' / row['split'] / (key + suffix)
                     for suffix in ('.npz', '.json', '.derived.npz')]
            paths += [Path(cfg['target_root']) / 'identities' / row['split'] / (key + suffix)
                      for suffix in ('.npz', '.json')]
            if any(not p.is_file() for p in paths):
                pending.append(key)
                continue
            identity, target, binding = _identity_record(cfg, row, person)
            supervision = legal_lf1_supervision(identity, target)
            eligible = len(supervision['traversal_bounds']) > 0
            population.append({'key': key, 'media_key': row['media_key'], 'slot': person['slot'],
                'split': row['split'], 'N': row['length'],
                'max_run': max((b - a + 1 for a, b in identity.run_bounds), default=0),
                'component': components[row['media_key']] if row['split'] == 'train' else None,
                'status': identity.status, 'bindings': binding})
            targets[key] = {'eligible': eligible, 'traversals': len(target.traversal_bounds),
                            'eligible_traversals': len(supervision['traversal_bounds']),
                            'traversal_eligibility': supervision['eligibility_ledger'],
                            'reason': None if eligible else 'no_complete_positive_and_negative_LF1_traversal'}
            if retain_units and row['split'] == 'train' and eligible:
                units[key] = prepare_natural_unit(identity, target)
    return census, population, targets, units, pending


def verify_evidence_bindings(cfg, population):
    """Cross-check producer/diagnostic bindings already emitted by frozen stages."""
    def payload(name):
        return json.loads(Path(cfg['artifacts'][name]['path']).read_text(encoding='utf-8-sig'))
    nb, nt, diag = payload('nb1'), payload('nt1'), payload('diagnostics')
    census_sha = file_sha(cfg['census'])
    if (nb['binding']['census_sha256'] != census_sha or nt['binding']['census_sha256'] != census_sha
            or nb['binding'].get('overlay_manifest_sha256') != cfg['artifacts']['clock_overlay']['sha256']):
        raise ValueError('NB1/NT1 census or overlay binding mismatch')
    inputs = diag['receipt']['inputs_sha256']
    required = {str(Path(cfg['census']).resolve()): census_sha,
                str(Path(cfg['component_map']).resolve()): file_sha(cfg['component_map'])}
    for name in ('nb1', 'nt1'):
        item = cfg['artifacts'][name]
        required[str(Path(item['path']).resolve())] = item['sha256']
    for row in population:
        key, split, b = row['key'], row['split'], row['bindings']
        for root, suffix, field in (('bridge_root', '.npz', 'feature_sha256'),
                ('bridge_root', '.json', 'bridge_receipt_sha256'),
                ('bridge_root', '.derived.npz', 'derived_sha256'),
                ('target_root', '.npz', 'target_sha256'),
                ('target_root', '.json', 'target_receipt_sha256')):
            required[str((Path(cfg[root]) / 'identities' / split / (key + suffix)).resolve())] = b[field]
    if any(inputs.get(path) != sha for path, sha in required.items()):
        raise ValueError('complete diagnostic report belongs to different loaded artifacts/components')
    continuation = payload('continuation')
    if continuation.get('diagnostics_sha256') != cfg['artifacts']['diagnostics']['sha256']:
        raise ValueError('continuation decision does not bind this complete diagnostic report')
    aliases = diag.get('same_media_alias_groups', [])
    if len(aliases) != 4 or any(len(g['media_keys']) < 2 for g in aliases):
        raise ValueError('diagnostic report must preserve four same-media train alias groups')


def _plan(cfg, *, retain_units=False):
    _check_config(cfg)
    bound, pending = _artifact_receipts(cfg)
    _, population, targets, units, missing = _load_inputs(cfg, retain_units=retain_units)
    pending += ['full_NB1_NT1_population_missing'] if missing else []
    if not (Path(cfg['x0_root']) / 'manifest.json').is_file():
        pending.append('X0_bank_missing')
        bank = None
    else:
        bank = validate_x0_bank(cfg['x0_root'])
        if bank.get('fresh_process_byte_check') != 'complete':
            pending.append('X0_two_fresh_process_byte_check')
    binding = {'census_sha256': file_sha(cfg['census']), 'component_map_sha256': file_sha(cfg['component_map']),
               'sources': source_binding(), 'artifacts': bound, 'environment': environment_binding(),
               'x0_manifest_sha256': file_sha(Path(cfg['x0_root']) / 'manifest.json') if bank else None,
               'configuration': {k: v for k, v in cfg.items() if k not in ('variant', 'output')}}
    if missing or bank is None:
        return None, {'schema': SCHEMA, 'status': 'pending_complete_inputs',
                      'binding': binding, 'pending': pending, 'missing_identities': missing,
                      'videos_required': {'train': 794, 'val': 183}, 'identities_required': 2364}, units
    if not pending:
        verify_evidence_bindings(cfg, population)
    plan = build_training_plan(population, targets, bank, dict(cfg, population_binding=binding, pending=pending))
    result = dict(plan.payload, plan_sha256=plan.sha256,
                  status='pending_qualification' if pending else 'CPU_plan_complete_not_trained')
    return plan, result, units


def _nvidia_query():
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    def query(args):
        return subprocess.run(['nvidia-smi', *args], check=True, capture_output=True, text=True,
                              creationflags=flags).stdout.strip().splitlines()
    gpu = query(['-i', '1', '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits'])
    uuid, total, free = [p.strip() for p in gpu[0].split(',')]
    procs = query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'])
    inventory = []
    for line in procs:
        fields = [p.strip() for p in line.split(',')]
        if len(fields) == 3 and fields[0] == uuid:
            inventory.append({'pid': int(fields[1]), 'bytes': int(fields[2]) * 1024 ** 2})
    return {'device_uuid': uuid, 'total_vram': int(total) * 1024 ** 2,
            'free_vram': int(free) * 1024 ** 2, 'inventory': inventory}


def _resource_check(plan, *, starting, baseline=None):
    resource = plan.payload['resource']
    live = _nvidia_query()
    if resource['device_uuid'] != live['device_uuid']:
        raise RuntimeError('resource_incomplete: GPU UUID changed')
    try:
        admission(dict(resource, free_vram=live['free_vram']), plan.payload['planned_tensor_bytes'], starting=starting)
    except ValueError as exc:
        raise RuntimeError('resource_incomplete: ' + str(exc)) from exc
    if resource['environment_sha'] != json_sha(environment_binding()):
        raise RuntimeError('resource_incomplete: environment binding differs')
    own = sum(r['bytes'] for r in live['inventory'] if r['pid'] == os.getpid())
    if own > 6 * GIB:
        raise RuntimeError('resource_incomplete: NVML process total exceeds 6GiB')
    other = {r['pid']: r['bytes'] for r in live['inventory'] if r['pid'] != os.getpid()}
    # R2a: sharing changes are recorded, not a memory-stability requirement.
    live['other_allocation_changes'] = {} if baseline is None else {
        pid: value - baseline.get(pid, 0) for pid, value in other.items()
        if value != baseline.get(pid, 0)}
    if torch.cuda.is_initialized() and torch.cuda.memory_reserved(1) > 5 * GIB:
        raise RuntimeError('resource_incomplete: allocator reserved exceeds 5GiB')
    import shutil
    free_disk = shutil.disk_usage(Path(resource['disk_path'])).free
    if starting and free_disk < plan.payload['required_free_disk']:
        raise RuntimeError('resource_incomplete: disk reserve below two peaks plus 10GiB')
    return live, other


def _require_train(plan):
    if plan is None or plan.payload['pending']:
        raise ValueError('train refuses unqualified/pending full-production plan')
    resource = plan.payload['resource']
    if resource.get('wall_limit_seconds') != 24 * 3600 or resource.get('gpu_limit_seconds') != 24 * 3600:
        raise ValueError('each arm must retain frozen 24h wall/GPU occupation cap')
    admission(resource, plan.payload['planned_tensor_bytes'])
    if not resource.get('owned_job_recovery_receipt'):
        raise ValueError('owned-job recovery handoff pending')
    if resource['free_disk'] < plan.payload['required_free_disk']:
        raise ValueError('frozen disk inventory insufficient')


def admitted_device(plan):
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        raise RuntimeError('GPU1 unavailable; no CPU model execution fallback')
    _, baseline = _resource_check(plan, starting=True)
    torch.cuda.set_device(1)
    torch.cuda.set_per_process_memory_fraction(5 * GIB / torch.cuda.get_device_properties(1).total_memory, 1)
    return torch.device('cuda:1'), baseline


def _train_impl(cfg, plan, units):
    begin = time.monotonic()
    _require_train(plan)
    out = Path(cfg['output']) / cfg['variant']
    budget_path = out / 'budget.json'
    setup_prior = 0.
    if budget_path.exists():
        prior = json.loads(budget_path.read_text(encoding='utf-8'))
        if prior['plan_sha256'] != plan.sha256 or prior['status'] in ('running', 'prediction_running'):
            raise RuntimeError('resource_incomplete: prior launcher budget unreconciled before GPU admission')
        setup_prior = max(prior['wall_seconds'], prior.get('gpu_occupation_seconds', 0.))
    if setup_prior >= 24 * 3600:
        raise RuntimeError('resource_incomplete: arm budget exhausted before GPU admission')
    configure_runtime()
    device, baseline = admitted_device(plan)
    state = initial_state(cfg['seed'])
    if tensor_state_sha(state) != plan.payload['initial_state_sha256']:
        raise ValueError('paired initial tensor SHA mismatch')
    model = TempoRACResponse().float()
    model.load_state_dict(state, strict=True)
    model.to(device)
    optimizer = make_optimizer(model)
    adapter = X0AnalyticAdapterNTR1(cfg['x0_root'])
    out = Path(cfg['output']) / cfg['variant']
    out.mkdir(parents=True, exist_ok=True)
    recovery = out / 'recovery.pt'
    start, prior_wall, candidates = 0, setup_prior, []
    if recovery.exists():
        cp = resume_exact(recovery, model, optimizer, plan, cfg['variant'])
        start, candidates = cp['step'], cp['candidates']
        prior_wall = max(prior_wall, cp['cumulative_wall_seconds'])
    budget_path = out / 'budget.json'
    if budget_path.exists():
        budget = json.loads(budget_path.read_text(encoding='utf-8'))
        if budget['plan_sha256'] != plan.sha256 or budget['status'] in ('running', 'prediction_running'):
            raise RuntimeError('resource_incomplete: unaccounted previous launcher; reconcile cumulative budget first')
        prior_wall = max(prior_wall, budget['wall_seconds'])
    step = start
    def elapsed():
        return prior_wall + time.monotonic() - begin
    def record_budget(status):
        atomic_json(budget_path, {'plan_sha256': plan.sha256, 'status': status, 'wall_seconds': elapsed(),
                                 'gpu_occupation_seconds': elapsed(), 'step': step, 'pid': os.getpid()})
    record_budget('running')
    try:
        for step in range(start + 1, plan.T + 1):
            if elapsed() >= 24 * 3600:
                raise RuntimeError('resource_incomplete: cumulative arm wall/GPU budget exceeded')
            live, _ = _resource_check(plan, starting=False, baseline=baseline)
            draw = plan.payload['draw_ledger'][step - 1]
            receipt = optimizer_step(model, optimizer, prepare_x0_unit(draw['x0_source'], adapter),
                units[draw['identity']], cfg['variant'], step, plan.T)
            with (out / 'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(asdict(receipt), draw=draw, wall_seconds=elapsed(),
                    peak_allocated=torch.cuda.max_memory_allocated(1),
                    peak_reserved=torch.cuda.max_memory_reserved(1),
                    gpu_process_inventory=live['inventory']), sort_keys=True) + '\n')
            if step % 500 == 0:
                objective = score_x0_tune(model, cfg['variant'], adapter)
                candidates.append({'step': step, 'objective': objective, 'file': f'candidate_{step:06d}.pt'})
                commit_checkpoint(out / candidates[-1]['file'], model, optimizer, plan, step=step,
                    variant=cfg['variant'], candidates=candidates, cumulative_wall_seconds=elapsed(), recovery=False)
            # Independent recovery snapshots never become additional tune candidates.
            commit_checkpoint(recovery, model, optimizer, plan, step=step, variant=cfg['variant'],
                              candidates=candidates, cumulative_wall_seconds=elapsed())
            record_budget('running')
        selected = select_checkpoint(candidates, plan.T)
        _resource_check(plan, starting=False, baseline=baseline)
        if elapsed() > 24 * 3600:
            raise RuntimeError('resource_incomplete: tune/completion exceeded cumulative budget')
        atomic_json(out / 'selection.json', {'schema': SCHEMA, 'status': 'complete_training',
                    'plan_sha256': plan.sha256, 'variant': cfg['variant'], 'seed': cfg['seed'],
                    'selected': selected, 'candidates': candidates,
                    'initial_state_sha256': plan.payload['initial_state_sha256'],
                    'draw_ledger_sha256': plan.payload['draw_ledger_sha256']})
        record_budget('complete')
    except BaseException as exc:
        # Last successful recovery is authoritative; never save partly updated
        # optimizer state as if a completed selection step had committed.
        record_budget('resource_incomplete' if isinstance(exc, (RuntimeError, torch.cuda.OutOfMemoryError)) else 'failed')
        atomic_json(out / 'failure.json', {'schema': SCHEMA, 'status': 'resource_incomplete',
                    'error': str(exc), 'last_recovery': str(recovery), 'wall_seconds': elapsed()})
        raise
    return 0


def _predict_impl(cfg, plan):
    begin = time.monotonic()
    _require_train(plan)
    out = Path(cfg['output']) / cfg['variant']
    budget_path = out / 'budget.json'
    budget = json.loads(budget_path.read_text(encoding='utf-8'))
    if budget['plan_sha256'] != plan.sha256 or budget['status'] in ('running', 'prediction_running'):
        raise RuntimeError('resource_incomplete: prediction budget unreconciled before GPU admission')
    prior_wall = max(budget['wall_seconds'], budget.get('gpu_occupation_seconds', 0.))
    if prior_wall >= 24 * 3600:
        raise RuntimeError('resource_incomplete: arm budget exhausted before prediction GPU admission')
    configure_runtime()
    selection = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
    if selection['plan_sha256'] != plan.sha256 or selection['status'] != 'complete_training':
        raise ValueError('predict requires fixed complete-training checkpoint selection')
    selected = select_checkpoint(selection['candidates'], plan.T)
    if selected != selection['selected']:
        raise ValueError('selection minimum/tie mismatch')
    device, baseline = admitted_device(plan)
    model = TempoRACResponse().float().to(device)
    optimizer = make_optimizer(model)
    cp = resume_exact(out / selected['file'], model, optimizer, plan, cfg['variant'])
    if cp['step'] != selected['step']:
        raise ValueError('selected checkpoint step mismatch')
    census = _module_from_script('build_temporac_native_learner_bridge').read_census(cfg['census'])
    def identities():
        for row in census:
            if row['split'] == 'val':
                for p in row['persons']:
                    key = f"{row['media_key']}_{p['slot']}"
                    # Prediction loads feature-only NB1; targets are not opened here.
                    feature = load_native_feature(Path(cfg['bridge_root']) / 'identities' / 'val' / (key + '.npz'))
                    if key not in completed_keys:
                        yield key, preprocess_native(feature)
    def monitor_prediction():
        elapsed = prior_wall + time.monotonic() - begin
        if elapsed >= 24 * 3600:
            raise RuntimeError('resource_incomplete: cumulative arm prediction budget exceeded')
        live, _ = _resource_check(plan, starting=False, baseline=baseline)
        atomic_json(budget_path, dict(budget, current_inventory=live['inventory'], status='prediction_running', wall_seconds=elapsed,
                                     gpu_occupation_seconds=elapsed))
    def monitored_identities():
        for item in identities():
            monitor_prediction()
            yield item
    progress_path = out / 'prediction_progress.json'
    predictions = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if progress['plan_sha256'] != plan.sha256 or progress['selected'] != selected:
            raise ValueError('prediction progress belongs to different inputs/selected checkpoint')
        predictions = progress['records']
        if len({r['key'] for r in predictions}) != len(predictions):
            raise ValueError('duplicate prediction progress identity')
        for record in predictions:
            if file_sha(out / 'predictions' / (record['key'] + '.npz')) != record['sha256']:
                raise ValueError('completed prediction bytes changed')
    completed_keys = {r['key'] for r in predictions}
    try:
        for key, pred in predict_population(model, monitored_identities(), cfg['variant']):
            monitor_prediction()
            path = out / 'predictions' / (key + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as stream:
                np.savez_compressed(stream, response=pred.response, support=pred.support,
                    run_bounds=pred.run_bounds, component_bounds=pred.component_bounds,
                    event_source_positions=pred.event_source_positions, event_seconds=pred.event_seconds)
            predictions.append({'key': key, 'count': pred.count, 'status': pred.status,
                    'decoder_invocations': pred.decoder_invocations, 'sha256': file_sha(path),
                    'period_ap_status': pred.period_ap_status})
            atomic_json(progress_path, {'schema': SCHEMA, 'status': 'partial_population',
                'plan_sha256': plan.sha256, 'selected': selected, 'records': predictions})
        if len(predictions) != 486 or len({p['key'].rsplit('_', 1)[0] for p in predictions}) != 183:
            raise ValueError('complete 183-video/486-identity dev prediction required')
        monitor_prediction()
        atomic_json(budget_path, dict(budget, status='complete_prediction',
            wall_seconds=prior_wall + time.monotonic() - begin,
            gpu_occupation_seconds=prior_wall + time.monotonic() - begin))
        atomic_json(out / 'predictions.json', {'schema': SCHEMA, 'status': 'complete_population',
                    'plan_sha256': plan.sha256, 'selected': selected, 'records': predictions,
                    'natural_GT_accessed': False})
        return 0
    except BaseException as exc:
        elapsed = prior_wall + time.monotonic() - begin
        atomic_json(budget_path, dict(budget, status='resource_incomplete', wall_seconds=elapsed,
                                     gpu_occupation_seconds=elapsed))
        atomic_json(out / 'prediction_failure.json', {'schema': SCHEMA, 'status': 'resource_incomplete',
                    'error': str(exc), 'wall_seconds': elapsed, 'selected': selected,
                    'completed_prediction_records': predictions, 'progress_manifest': str(progress_path)})
        raise


def _accounted_stage(stage, implementation, cfg, plan, *args):
    """Account setup, including admission/context/model/checkpoint failures."""
    _require_train(plan)
    out = Path(cfg['output']) / cfg['variant']
    out.mkdir(parents=True, exist_ok=True)
    budget_path = out / 'budget.json'
    previous = json.loads(budget_path.read_text(encoding='utf-8')) if budget_path.exists() else {
        'plan_sha256': plan.sha256, 'status': 'new', 'wall_seconds': 0., 'gpu_occupation_seconds': 0.}
    if previous['plan_sha256'] != plan.sha256 or previous['status'] in ('running', 'prediction_running', 'setup_running'):
        raise RuntimeError('resource_incomplete: unaccounted previous launcher before GPU admission')
    prior = max(previous['wall_seconds'], previous.get('gpu_occupation_seconds', 0.))
    if prior >= 24 * 3600:
        raise RuntimeError('resource_incomplete: arm budget exhausted before GPU admission')
    begin = time.monotonic()
    atomic_json(budget_path, dict(previous, status='setup_running', stage=stage, pid=os.getpid()))
    try:
        result = implementation(cfg, plan, *args)
        latest = json.loads(budget_path.read_text(encoding='utf-8'))
        elapsed = max(latest['wall_seconds'], prior + time.monotonic() - begin)
        atomic_json(budget_path, dict(latest, wall_seconds=elapsed, gpu_occupation_seconds=elapsed))
        return result
    except BaseException as exc:
        latest = json.loads(budget_path.read_text(encoding='utf-8'))
        elapsed = max(latest['wall_seconds'], prior + time.monotonic() - begin)
        atomic_json(budget_path, dict(latest, status='resource_incomplete', wall_seconds=elapsed,
                                     gpu_occupation_seconds=elapsed))
        progress_path = out / 'prediction_progress.json'
        progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
        atomic_json(out / ('prediction_failure.json' if stage == 'predict' else 'failure.json'),
            {'schema': SCHEMA, 'status': 'resource_incomplete', 'stage': stage, 'error': str(exc),
             'wall_seconds': elapsed, 'last_recovery': str(out / 'recovery.pt'),
             'progress_manifest': str(progress_path), 'completed_prediction_records': progress.get('records', [])})
        raise


def _train(cfg, plan, units):
    return _accounted_stage('train', _train_impl, cfg, plan, units)


def _predict(cfg, plan):
    return _accounted_stage('predict', _predict_impl, cfg, plan)


def execute(cfg, command):
    if command not in ('plan', 'validate', 'train', 'predict'):
        raise ValueError('unknown explicit NTR1 command')
    plan, result, units = _plan(cfg, retain_units=command == 'train')
    output = _checked_path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    if command in ('plan', 'validate'):
        atomic_json(output / (command + '.json'), result)
        return 0 if plan is not None and not plan.payload['pending'] else 2
    _require_train(plan)
    frozen = output / 'frozen_plan.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf-8')) != plan.payload:
            raise ValueError('existing frozen plan differs; exact resume refuses altered inputs')
    else:
        atomic_json(frozen, plan.payload)
        atomic_json(output / 'paired_draw_ledger.json', plan.payload['draw_ledger'])
    return _train(cfg, plan, units) if command == 'train' else _predict(cfg, plan)
