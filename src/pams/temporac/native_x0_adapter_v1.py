"""NTR1 analytic X0, independent of natural targets, teachers and checkpoints.

q is synthetic_tick, never seconds/Hz. Analytic clean coordinates, phase and
responsibilities remain target-side and never enter the 269 encoder channels.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from pams.temporac import x0
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.cue import window_schedule

SCHEMA = 'temporac.x0-analytic.ntr1'
ARRAYS = ('response_input', 'run_bounds', 'support', 'q', 'pulse', 'chi',
          'target_mask', 'traversal_bounds', 'responsibility', 'gates', 'trim_bounds')


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


@dataclass(frozen=True)
class AnalyticX0Unit:
    source_id: int
    block_index: int
    response_input: np.ndarray
    run_bounds: np.ndarray
    support: np.ndarray
    q: np.ndarray
    pulse: np.ndarray
    chi: np.ndarray
    target_mask: np.ndarray
    traversal_bounds: np.ndarray
    responsibility: np.ndarray
    gates: np.ndarray
    trim_bounds: np.ndarray
    clock_unit: str = 'synthetic_tick'

    @property
    def windows(self):
        return window_schedule(self.run_bounds)

    def arrays(self):
        return {name: getattr(self, name) for name in ARRAYS}

    def __post_init__(self):
        n = len(self.response_input)
        e = n - 1
        if (self.response_input.dtype != np.float32 or self.response_input.shape != (n, 269)
                or self.q.dtype != np.float64 or self.q.shape != (n,)
                or not np.all(np.diff(self.q) == 1)
                or self.clock_unit != 'synthetic_tick'):
            raise ValueError('X0 requires unchanged 269-channel rows and synthetic integer-tick clock')
        if (self.support.dtype != np.bool_ or self.support.shape != (e,) or not self.support.all()
                or self.target_mask.shape != (e,) or self.pulse.shape != (e,)
                or self.chi.shape != (e,) or self.traversal_bounds.shape != (10, 2)
                or self.responsibility.shape != (e, 3)
                or self.gates.shape != (len(self.windows), 3)):
            raise ValueError('X0 full-domain / ten-traversal schema mismatch')
        union = np.zeros(e, bool)
        for left, stop in self.traversal_bounds:
            if not 0 <= left < stop <= e or union[left:stop].any():
                raise ValueError('X0 traversal bounds/trim invalid')
            union[left:stop] = True
            if (self.pulse[left] != 1 or self.pulse[left:stop].sum() != 1
                    or abs(self.chi[left:stop].sum() - 1) > 1e-12):
                raise ValueError('X0 start pulse / chi normalization invalid')
        if (not np.array_equal(union, self.target_mask.astype(bool))
                or self.pulse.sum() != 10 or np.any(self.pulse[~union]) or np.any(self.chi[~union])
                or not np.isfinite(self.responsibility).all() or np.any(self.responsibility <= 0)
                or not np.allclose(self.responsibility.sum(1), 1, rtol=0, atol=1e-12)):
            raise ValueError('X0 guards or analytic responsibilities invalid')
        for value in self.arrays().values():
            value.setflags(write=False)


class X0AnalyticAdapterNTR1:
    def __init__(self, bank_root=None):
        self.bank_root = None if bank_root is None else Path(bank_root)

    def view(self, source_id, block_index):
        if self.bank_root is not None:
            with np.load(self.bank_root / f'U{source_id:04d}_B{block_index}.npz', allow_pickle=False) as f:
                if set(f.files) != set(ARRAYS):
                    raise ValueError('X0 bank array schema mismatch')
                return AnalyticX0Unit(source_id, block_index, **{k: f[k] for k in ARRAYS})
        view = x0.generate_view(source_id, block_index, resampler='linear', offset=0)
        identity = preprocess_identity(view.motion, np.ones(len(view.clock), np.uint8), view.clock)
        first, stop = identity.trim_bounds
        bounds = view.traversal_bounds.astype(np.int64) - first
        # All frozen X0 views have valid guard rows. Refuse any preprocessing change
        # that would invalidate the original analytic window/edge responsibility.
        starts = np.array([w.edge_start for w in window_schedule(identity.run_bounds)])
        if not np.array_equal(starts, view.window_start - first):
            raise ValueError('X0 trim altered frozen analytic window schedule')
        return AnalyticX0Unit(source_id, block_index,
                np.array(identity.response_input, copy=True), np.array(identity.run_bounds, copy=True),
                np.ones(stop - first - 1, bool), view.clock[first:stop].astype(np.float64),
                np.array(view.pulse[first:stop-1], copy=True), np.array(view.chi[first:stop-1], copy=True),
                np.array(view.target_mask[first:stop-1], copy=True), bounds,
                np.array(view.edge_responsibility[first:stop-1], copy=True),
                np.array(view.p_star, copy=True), np.array([first, stop], np.int64))

    def source(self, source_id):
        return tuple(self.view(source_id, b) for b in range(7))


def source_binding():
    from pams.temporac import preprocess
    return {'x0': file_sha(x0.__file__), 'preprocess': file_sha(preprocess.__file__),
            'adapter': file_sha(__file__)}


def build_x0_bank(root, generator_contract_sha256):
    """Build once; existing artifacts are validated, never replaced.

    Reproducibility across two fresh processes is established by independently
    comparing `sources` and every raw-array digest, not ZIP timestamps.
    """
    root = Path(root)
    if (root / 'manifest.json').exists():
        return validate_x0_bank(root, generator_contract_sha256=generator_contract_sha256)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('incomplete/nonempty X0 bank: preserve it and choose a new output')
    sources = json.loads(x0.x0_manifest(generator_contract_sha256))
    adapter = X0AnalyticAdapterNTR1()
    views = []
    for source in range(32):
        for block in range(7):
            unit = adapter.view(source, block)
            name = f'U{source:04d}_B{block}.npz'
            with (root / name).open('wb') as f:
                np.savez_compressed(f, **unit.arrays())
            views.append({'source_id': source, 'block': block, 'file': name,
                          'npz_sha256': file_sha(root / name),
                          'arrays': {k: {'sha256': array_sha(v), 'shape': list(v.shape),
                                         'dtype': v.dtype.str} for k, v in unit.arrays().items()}})
    manifest = {'schema': SCHEMA, 'sources': sources, 'views': views,
                'generator_contract_sha256': generator_contract_sha256,
                'source_binding': source_binding(),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'scipy': scipy.__version__},
                'reference_byte_compatibility': 'not_claimed_without_reference',
                'fresh_process_byte_check': 'pending'}
    atomic_json(root / 'manifest.json', manifest)
    return manifest


def validate_x0_bank(root, *, generator_contract_sha256=None):
    root = Path(root)
    m = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (m['schema'] != SCHEMA or m['source_binding'] != source_binding()
            or len(m['sources']) != 40 or len(m['views']) != 224
            or (generator_contract_sha256 is not None
                and m['generator_contract_sha256'] != generator_contract_sha256)):
        raise ValueError('X0 bank provenance/population mismatch')
    for i, row in enumerate(m['sources']):
        if (row['id'] != f'U{i:04d}' or row['split'] != x0.source_split(i)
                or row['source_key_hex'] != x0.source_key(i).hex()
                or row['generator_contract_sha256'] != m['generator_contract_sha256']
                or not 0 <= row['accepted_attempt'] < 64):
            raise ValueError('X0 source identity/split/generation mismatch')
    expected = {(s, b) for s in range(32) for b in range(7)}
    if {(r['source_id'], r['block']) for r in m['views']} != expected:
        raise ValueError('X0 complete train/tune view coverage required')
    adapter = X0AnalyticAdapterNTR1(root)
    for row in m['views']:
        name = f"U{row['source_id']:04d}_B{row['block']}.npz"
        if row['file'] != name or file_sha(root / name) != row['npz_sha256']:
            raise ValueError('X0 frozen array file changed')
        unit = adapter.view(row['source_id'], row['block'])
        actual = {k: {'sha256': array_sha(v), 'shape': list(v.shape), 'dtype': v.dtype.str}
                  for k, v in unit.arrays().items()}
        if actual != row['arrays']:
            raise ValueError('X0 frozen raw-array bytes changed')
    return m
