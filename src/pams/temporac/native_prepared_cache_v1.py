"""Persistent feature-only NTR1 preparation. Never imports or accepts target APIs."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from pams.temporac.native_bridge_v1 import load_native_feature, native_clock
from pams.temporac.preprocess import BONES, _geometry_masks, validate_run_bounds
from pams.temporac.native_fast_cue_v1 import compute_local_routing

SCHEMA = 'temporac.prepared-cache.v1'
FEATURE_ARRAYS = {'windows', 'local_gates', 'denominator', 'responsibility', 'support'}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_receipt(value):
    a = np.ascontiguousarray(value)
    if a.dtype.hasobject:
        raise ValueError('object/pickle arrays forbidden')
    return {'dtype': a.dtype.str, 'shape': list(a.shape), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def kernel_binding():
    root = Path(__file__).parent
    names = ('native_prepared_cache_v1.py', 'native_fast_cue_v1.py', 'native_bridge_v1.py',
             'native_time_cue_v1.py', 'cache_build_v1.py', 'cue.py', 'quadrature.py', 'preprocess.py', 'native_training_runtime_v1.py')
    return {'files': {n: file_sha(root / n) for n in names},
            'python': platform.python_version(), 'numpy': np.__version__}


@contextmanager
def unit_lock(path):
    """Kernel-held advisory lock; released on process exit, no stale PID heuristic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class UnitStore:
    def __init__(self, root, role):
        if role not in ('natural', 'supervision', 'x0'):
            raise ValueError('unknown cache role')
        self.root, self.role = Path(root), role
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, identity, binding):
        return json_sha({'schema': SCHEMA, 'role': self.role, 'identity': identity, 'binding': binding})

    def load(self, identity, binding):
        key = self.key(identity, binding)
        marker = self.root / (key + '.json')
        if not marker.exists():
            return None
        receipt = json.loads(marker.read_text(encoding='utf-8'))
        expected = {'schema': SCHEMA, 'role': self.role, 'identity': identity, 'binding': binding, 'key': key}
        if any(receipt.get(k) != v for k,v in expected.items()):
            raise ValueError('cache receipt binding mismatch')
        path = self.root / (key + '.npz')
        if file_sha(path) != receipt['payload_sha256']:
            raise ValueError('committed cache payload changed')
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        if {k: array_receipt(v) for k,v in arrays.items()} != receipt['arrays']:
            raise ValueError('cache arrays changed')
        if self.role == 'natural' and set(arrays) != FEATURE_ARRAYS:
            raise ValueError('natural cache must be feature-only')
        for a in arrays.values():
            a.setflags(write=False)
        return arrays, receipt

    def write(self, identity, binding, arrays, metadata=None):
        if self.role == 'natural' and set(arrays) != FEATURE_ARRAYS:
            raise ValueError('natural cache must be feature-only')
        key = self.key(identity, binding)
        with unit_lock(self.root / (key + '.lock')):
            hit = self.load(identity, binding)
            if hit is not None:
                return hit[1]
            payload = self.root / (key + '.npz')
            tmp = payload.with_name(payload.name + f'.tmp.{os.getpid()}')
            descriptions = {k: array_receipt(v) for k,v in arrays.items()}
            with tmp.open('wb') as f:
                np.savez_compressed(f, **arrays)
                f.flush(); os.fsync(f.fileno())
            with np.load(tmp, allow_pickle=False) as f:
                if {k: array_receipt(f[k]) for k in f.files} != descriptions:
                    raise ValueError('cache write roundtrip mismatch')
            os.replace(tmp, payload)
            receipt = {'schema': SCHEMA, 'role': self.role, 'identity': identity, 'binding': binding,
                       'key': key, 'payload_sha256': file_sha(payload), 'arrays': descriptions,
                       'bytes': payload.stat().st_size, 'metadata': metadata or {}}
            atomic_json(self.root / (key + '.json'), receipt)
            return receipt


@dataclass(frozen=True)
class RoutingGeometry:
    geometry: np.ndarray
    edge_coordinate_mask: np.ndarray
    edge_lengths: np.ndarray


@dataclass(frozen=True)
class NativePreparedView:
    """Declared lightweight feature interface consumed by routing and decoder."""
    feature: object
    clock: object
    response_input: np.ndarray
    run_bounds: np.ndarray
    response_edge_support: np.ndarray
    observed_edge_support: np.ndarray
    geometry_features: RoutingGeometry
    mapping: dict
    status: str


def read_derived(feature_path, derived_path, *, expected_N=None, expected_key=None, for_routing=True):
    """Restore exact routing geometry; never call preprocessing/target APIs."""
    feature = load_native_feature(feature_path)
    n = int(feature.source_length[0])
    key = feature.opaque_key.tobytes().hex() + '_' + str(int(feature.slot[0]))
    if (expected_N is not None and n != expected_N) or (expected_key is not None and key != expected_key):
        raise ValueError('feature identity mismatch')
    clock = native_clock(feature)
    with np.load(derived_path, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    e = max(0, n - 1)
    specs = {'response_input': ('<f4', (n,269)), 'joint_usable': ('|b1',(n,17)),
             'frame_derived_valid': ('|b1',(n,)), 'observed_edge_support': ('|b1',(e,)),
             'response_edge_support': ('|b1',(e,)), 'edge_lengths': ('<f8',(e,)),
             'direction': ('<f8',(n,66)), 'direction_mask': ('|b1',(n,33)),
             'run_bounds': ('<i4',None), 'short_runs': ('<i4',None),
             'map_source_frame_index': ('<i8',(n,)), 'map_q': ('<f8',(n,)),
             'map_seconds': ('<f8',(n,)), 'map_segment': ('<i4',(n,))}
    for name in ('left_source_index','right_source_index'):
        specs['map_'+name] = ('<i8',(n,17))
    for name in ('left_weight','right_weight','gap_span'):
        specs['map_'+name] = ('<f8',(n,17))
    specs['map_support_kind'] = ('<U12',(n,17))
    specs['map_supported'] = ('|b1',(n,17))
    if set(a) != set(specs):
        raise ValueError('derived schema mismatch')
    for name,(dtype,shape) in specs.items():
        if a[name].dtype.str != dtype or (shape is not None and a[name].shape != shape):
            raise ValueError('derived array shape/dtype mismatch: '+name)
        a[name].setflags(write=False)
    if a['short_runs'].ndim != 2 or a['short_runs'].shape[1:] != (2,):
        raise ValueError('short-run schema mismatch')
    if not np.array_equal(a['map_source_frame_index'],feature.source_frame_index):
        raise ValueError('source-row mapping mismatch')
    if np.any(a['response_edge_support'] & ~a['observed_edge_support']):
        raise ValueError('response support outside observed support')
    bounds = validate_run_bounds(a['run_bounds'], e)
    support = np.zeros(e,bool)
    for start,stop in bounds:
        support[start:stop] = True
    if not np.array_equal(support,a['response_edge_support']):
        raise ValueError('derived support/run mismatch')
    if not np.array_equal(a['map_q'],clock.q,equal_nan=True) or not np.array_equal(a['map_seconds'],clock.seconds,equal_nan=True):
        raise ValueError('derived clock mismatch')
    if not for_routing:
        return NativePreparedView(feature,clock,a['response_input'],bounds,a['response_edge_support'],
            a['observed_edge_support'],None,{k[4:]:v for k,v in a.items() if k.startswith('map_')},
            'complete' if support.size and support.all() else 'partial' if support.any() else 'unusable')
    pose = a['response_input'][:,:34].astype(np.float64)
    coordinates = pose.reshape(n,17,2)
    bones = np.zeros((n,32),np.float64)
    mask = a['joint_usable']
    for index,(source,target) in enumerate(BONES):
        ok = mask[:,source] & mask[:,target]
        bones[ok,2*index:2*index+2] = coordinates[ok,target] - coordinates[ok,source]
    geometry = np.ascontiguousarray(np.concatenate((pose,bones),axis=1))
    _,coordinate_mask = _geometry_masks(mask)
    edge_coordinate_mask = (coordinate_mask[:-1] & coordinate_mask[1:]) & a['observed_edge_support'][:,None]
    for v in (geometry,edge_coordinate_mask): v.setflags(write=False)
    return NativePreparedView(feature,clock,a['response_input'],bounds,a['response_edge_support'],
        a['observed_edge_support'],RoutingGeometry(geometry,edge_coordinate_mask,a['edge_lengths']),
        {k[4:]:v for k,v in a.items() if k.startswith('map_')},
        'complete' if support.size and support.all() else 'partial' if support.any() else 'unusable')


def natural_binding(row, kernel):
    # Target bindings intentionally excluded from feature-only identity.
    return {'kernel': kernel, 'N': row['N'], 'source': {k:row['bindings'][k] for k in
        ('feature_sha256','bridge_receipt_sha256','derived_sha256')}}


def prepare_natural_arrays(identity, progress=None):
    from pams.temporac.native_training_runtime_v1 import precompute_nola
    t = time.perf_counter()
    routing = compute_local_routing(identity,progress=progress)
    cue_seconds = time.perf_counter()-t
    t = time.perf_counter()
    cache = precompute_nola(routing.windows,routing.local_gates,identity.response_edge_support)
    return {'windows':np.asarray([(w.run_index,w.edge_start,w.edge_stop) for w in routing.windows],np.int64).reshape(-1,3),
            'local_gates':routing.local_gates,'denominator':cache.denominator,
            'responsibility':cache.responsibility,'support':cache.support}, {'cue_seconds':cue_seconds,'nola_seconds':time.perf_counter()-t}


def cached_natural_unit(key, identity, arrays, supervision):
    from pams.temporac.cue import Window
    from pams.temporac.native_training_runtime_v1 import PreparedUnit,NOLACache
    return PreparedUnit(key,identity.response_input,identity.run_bounds,
        tuple(Window(*map(int,w)) for w in arrays['windows']),arrays['local_gates'],
        NOLACache(arrays['denominator'],arrays['responsibility'],arrays['support']),
        supervision['pulse'],supervision['chi'],supervision['loss_mask'],
        supervision['traversal_bounds'],'natural',None)

