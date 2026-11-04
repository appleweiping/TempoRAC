"""NT1: count-blind native targets, strict per-run original numerical certificates.

No annotation, evaluator, model or natural FeatureRecord enters this module.
The certificate establishes pseudo-event geometry, not true repetition units.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .certify import CertificateTrack
from .certify_adaptive_v5 import stable_pair_integer_landmarks_v5
from .certify_target_adaptive_v4 import certify_target_adaptive_v4
from .certify_target_adaptive_v6 import certify_target_adaptive_v6
from .certify_target_adaptive_v7 import certify_target_adaptive_v7
from .full_chain_landmarks_candidate_v1 import full_chain_integer_landmarks
from .full_chain_landmarks_candidate_v2 import full_chain_integer_landmarks_v2
from .multi_cycle_dtw_template_candidate_v1 import multi_cycle_dtw_reconstruction
from .periodic_template_candidate_v1 import periodic_template_reconstruction
from .phase_closure_candidate_v1 import close_phase_on_landmarks

SCHEMA = 'temporac.native-target.nt1'
BRANCHES = ('v12_geometry', 'v9_hybrid', 'v15_geometry')
ARRAYS = ('target_valid', 'pulse', 'chi', 'traversal_bounds', 'seams', 'run_id', 'branch_id')


class NativeTargetError(ValueError):
    """Invalid target contract or producer integrity."""


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _frozen(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class NativeTargetPolicy:
    input_sha256: str
    raw_sha256: str
    bridge_sha256: str
    producer_sha256: str
    config_sha256: str
    availability_sha256: str
    hybrid_mode: str = 'unavailable'

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if k.endswith('sha256') and not _sha(v):
                raise NativeTargetError('actual SHA256 required: ' + k)
        if self.hybrid_mode not in ('unavailable', 'available'):
            raise NativeTargetError('invalid hybrid_mode')


@dataclass(frozen=True)
class NativeTargetNT1:
    target_valid: np.ndarray
    pulse: np.ndarray
    chi: np.ndarray
    traversal_bounds: np.ndarray
    seams: np.ndarray
    run_id: np.ndarray
    branch_id: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_native_target(self)
        for name in ARRAYS:
            object.__setattr__(self, name, _frozen(getattr(self, name)))


def validate_native_target(target, *, expected_input_sha256=None, response_support=None):
    m, p, chi = (np.asarray(getattr(target, k)) for k in ARRAYS[:3])
    e = m.size
    if m.dtype != np.bool_ or m.shape != (e,) or p.shape != (e,) or chi.shape != (e,):
        raise NativeTargetError('target must use complete one-dimensional edge domain')
    if p.dtype != np.uint8 or not np.isin(p, (0, 1)).all():
        raise NativeTargetError('pulse must be uint8 binary')
    if chi.dtype != np.float64 or not np.isfinite(chi).all() or np.any(chi < 0):
        raise NativeTargetError('chi must be finite nonnegative float64')
    b = np.asarray(target.traversal_bounds)
    if b.dtype != np.int64 or b.ndim != 2 or b.shape[1:] != (2,):
        raise NativeTargetError('bounds must be int64 [K,2]')
    k = len(b)
    if (target.seams.shape != (k,) or target.seams.dtype != np.float64
            or target.run_id.shape != (k,) or target.run_id.dtype != np.int64
            or target.branch_id.shape != (k,) or target.branch_id.dtype.kind != 'U'):
        raise NativeTargetError('traversal provenance arrays mismatch')
    union = np.zeros(e, bool)
    previous = 0
    for j, (left, r) in enumerate(b.tolist()):
        if not 0 <= left < r <= e or left < previous:
            raise NativeTargetError('bounds overlap, are unordered or outside full edge domain')
        previous = r
        union[left:r] = True
        if (target.seams[j] != left or p[left] != 1 or np.sum(p[left:r]) != 1
                or abs(float(np.sum(chi[left:r], dtype=np.float64)) - 1) > 1e-12):
            raise NativeTargetError('each traversal requires one start pulse and normalized chi')
        if target.run_id[j] < 0 or target.branch_id[j] not in BRANCHES:
            raise NativeTargetError('invalid per-traversal provenance')
    if not np.array_equal(m, union) or np.any(p[~m]) or np.any(chi[~m]):
        raise NativeTargetError('target mask must equal exact traversal union; outside payload zero')
    meta = target.metadata
    if meta.get('schema') != SCHEMA or meta.get('source_length') != e + 1:
        raise NativeTargetError('schema or full N mismatch')
    for name in NativeTargetPolicy.__dataclass_fields__:
        if name.endswith('sha256') and not _sha(meta.get(name)):
            raise NativeTargetError('missing target binding: ' + name)
    if expected_input_sha256 is not None and meta['input_sha256'] != expected_input_sha256:
        raise NativeTargetError('input SHA mismatch')
    provenance = meta.get('traversals', [])
    if len(provenance) != k:
        raise NativeTargetError('traversal ledger mismatch')
    for j, row in enumerate(provenance):
        branch = str(target.branch_id[j])
        if (row.get('branch_id') != branch or row.get('run_id') != int(target.run_id[j])
                or row.get('bounds') != b[j].tolist()):
            raise NativeTargetError('traversal ledger binding mismatch')
        teacher = row.get('teacher_sha256')
        if (branch == 'v9_hybrid' and not _sha(teacher)) or (branch != 'v9_hybrid' and teacher is not None):
            raise NativeTargetError('invalid teacher provenance')
    sources = set(str(x) for x in target.branch_id)
    source = ('rejected' if not sources else 'hybrid' if sources == {'v9_hybrid'} else
              'geometry' if 'v9_hybrid' not in sources else 'mixed')
    if meta.get('source') != source:
        raise NativeTargetError('identity source provenance mismatch')
    if response_support is not None:
        support = np.asarray(response_support)
        if support.shape != m.shape or support.dtype != np.bool_:
            raise NativeTargetError('response support must be full-domain bool')
        if np.any(m & ~support):
            raise NativeTargetError('entire traversal ineligible: response support loss')
    return target


def _phase_artifact(identity, policy, artifact):
    if policy.hybrid_mode == 'unavailable':
        if artifact is not None:
            raise NativeTargetError('phase supplied while hybrid unavailable')
        return None
    # Production availability is frozen unavailable in this delivery. Hash strings
    # supplied by a caller do not attest checkpoint/phase artifacts. The internal
    # numerical_branch accepts deterministic mock phase for CPU parity only.
    raise NativeTargetError('production_hybrid_intake_unavailable_requires_frozen_artifact_validation')


def _strict_traversals(geometry, coordinate_mask, response, landmarks):
    if len(landmarks) < 3:
        raise NativeTargetError('insufficient_landmarks')
    for left, r in zip(landmarks[:-1], landmarks[1:], strict=True):
        common = coordinate_mask[left:r] & coordinate_mask[left + 1:r + 1]
        delta = geometry[left + 1:r + 1] - geometry[left:r]
        masked_arc = np.sqrt(np.sum(delta * delta * common, axis=1)
                             / np.maximum(1, np.sum(common, axis=1)))
        if not np.all(response[left:r]):
            raise NativeTargetError('traversal_response_support_loss')
        if np.any(np.sum(common, axis=1) < 32):
            raise NativeTargetError('traversal_insufficient_coordinates')
        if np.any(~np.isfinite(masked_arc)) or np.any(masked_arc <= 0):
            raise NativeTargetError('traversal_zero_or_nonfinite_masked_arc')
        if np.any(np.linalg.norm(delta, axis=1) <= 0):
            raise NativeTargetError('nonpositive_interpolation_denominator')


def numerical_branch(geometry, continuous, continuous_mask, coordinate_mask, response,
                     branch_id, phase=None):
    """Original per-run numerical kernel plus NT1 strict qualification; no provenance fiction."""
    geometry = np.asarray(geometry, np.float64)
    n = len(geometry)
    bounds = np.asarray([[0, n]], np.int32)
    selector = {'v12_geometry': full_chain_integer_landmarks,
                'v9_hybrid': stable_pair_integer_landmarks_v5,
                'v15_geometry': full_chain_integer_landmarks_v2}[branch_id]
    landmarks = selector(geometry, bounds).landmarks
    _strict_traversals(geometry, coordinate_mask, response, landmarks)
    seed = np.tile([[1., 0.]], (n, 1)) if phase is None else phase
    closed = close_phase_on_landmarks(seed, landmarks, geometry=geometry)
    if branch_id == 'v9_hybrid':
        reconstruction = periodic_template_reconstruction(continuous, continuous_mask,
                                                          geometry, landmarks)
    else:
        reconstruction, coords = multi_cycle_dtw_reconstruction(continuous, continuous_mask,
                                                               geometry, landmarks)
        if any(np.any(np.diff(c) <= 0) for c in coords):
            raise NativeTargetError('nonpositive_DTW_interpolation_denominator')
    track = CertificateTrack(geometry=geometry, continuous=continuous,
                             continuous_mask=continuous_mask, phase=closed,
                             reconstruction=reconstruction, run_bounds=bounds, provenance=None)
    certifier = {'v12_geometry': certify_target_adaptive_v6,
                 'v9_hybrid': certify_target_adaptive_v4,
                 'v15_geometry': certify_target_adaptive_v7}[branch_id]
    return certifier(track, 'natural'), track


def produce_native_target(identity, policy, optional_phase_artifact=None):
    """First complete certificate per NB1 executable run; rejected runs remain in ledger."""
    phase = _phase_artifact(identity, policy, optional_phase_artifact)
    n = int(identity.feature.source_length[0])
    e = n - 1
    target_valid, pulse, chi = np.zeros(e, bool), np.zeros(e, np.uint8), np.zeros(e, np.float64)
    bounds, seams, runs, branches, attempts, traversals = [], [], [], [], [], []
    gf = identity.geometry_features
    continuous = np.asarray(gf.teacher_input[:, :149], np.float64)
    cmask = np.concatenate((gf.coordinate_mask,
                            np.repeat(gf.direction_element_mask, 2, axis=1), identity.joint_mask), axis=1)
    for run_id, (a_raw, b_raw) in enumerate(identity.run_bounds):
        a, b = int(a_raw), int(b_raw)
        for branch in BRANCHES:
            row = {'run_id': run_id, 'run_bounds': [a, b], 'branch_id': branch}
            if branch == 'v9_hybrid' and phase is None:
                attempts.append(dict(row, status='unavailable', reason='hybrid_mode_unavailable'))
                continue
            try:
                cert, _track = numerical_branch(
                    gf.geometry[a:b + 1], continuous[a:b + 1], cmask[a:b + 1],
                    gf.coordinate_mask[a:b + 1], identity.response_edge_support[a:b], branch,
                    None if branch != 'v9_hybrid' else phase[a:b + 1])
                if not cert.certified:
                    attempts.append(dict(row, status='rejected', reason='certificate',
                                         certificate_reasons=cert.reasons.tolist()))
                    continue
                target_valid[a:b] = cert.target_mask.astype(bool)
                pulse[a:b], chi[a:b] = cert.pulse, cert.chi
                for j, (left, right) in enumerate(cert.traversal_bounds):
                    left, r = a + int(left), a + int(right)
                    bounds.append([left, r])
                    seams.append(a + float(cert.seams[j]))
                    runs.append(run_id)
                    branches.append(branch)
                    traversals.append({'run_id': run_id, 'bounds': [left, r], 'branch_id': branch,
                                       'teacher_sha256': None if branch != 'v9_hybrid' else
                                       optional_phase_artifact['teacher_sha256']})
                attempts.append(dict(row, status='certified', reason=None,
                                     traversal_count=len(cert.traversal_bounds)))
                break
            except (ValueError, FloatingPointError, ArithmeticError) as exc:
                attempts.append(dict(row, status='rejected', reason=str(exc),
                                     exception_type=type(exc).__name__))
    sources = set(branches)
    source = ('rejected' if not sources else 'hybrid' if sources == {'v9_hybrid'} else
              'geometry' if 'v9_hybrid' not in sources else 'mixed')
    metadata = dict(policy.__dict__, schema=SCHEMA, source_length=n,
                    opaque_key=identity.feature.opaque_key.tobytes().hex(), slot=int(identity.feature.slot[0]),
                    source=source, status='certified' if bounds else 'rejected',
                    input_status=getattr(identity, 'status', 'synthetic_fixture'),
                    input_reasons=list(getattr(identity, 'reasons', ())),
                    short_runs=np.asarray(getattr(identity, 'short_runs', np.empty((0, 2), np.int32))).tolist(),
                    attempts=attempts, traversals=traversals)
    result = NativeTargetNT1(target_valid, pulse, chi, np.asarray(bounds, np.int64).reshape(-1, 2),
                             np.asarray(seams, np.float64), np.asarray(runs, np.int64),
                             np.asarray(branches, dtype='<U16'), metadata)
    return validate_native_target(result, response_support=identity.response_edge_support)


def save_native_target(path, target):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_native_target(target)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **{k: getattr(target, k) for k in ARRAYS})
    os.replace(tmp, path)
    meta = dict(target.metadata, npz_sha256=file_sha(path))
    sidecar = path.with_suffix('.json')
    tmp = sidecar.with_name(sidecar.name + '.tmp')
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    os.replace(tmp, sidecar)


def load_native_target(path, *, expected_input_sha256=None):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta.pop('npz_sha256') != file_sha(path):
        raise NativeTargetError('target payload SHA mismatch')
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != set(ARRAYS):
            raise NativeTargetError('target NPZ schema mismatch')
        result = NativeTargetNT1(**{k: values[k] for k in ARRAYS}, metadata=meta)
    return validate_native_target(result, expected_input_sha256=expected_input_sha256)


def lf1_supervision(identity, target, *, expected_input_sha256):
    """Explicit adapter; whole unsupported traversals cannot be clipped to qualify."""
    validate_native_target(target, expected_input_sha256=expected_input_sha256,
                           response_support=identity.response_edge_support)
    return {'pulse': target.pulse, 'chi': target.chi,
            'edge_mask': identity.response_edge_support & target.target_valid,
            'traversal_bounds': target.traversal_bounds}


def geometry_count(target):
    """G-NT1 only: certified pseudo-events, zero on full rejection, no coverage scaling."""
    if any(x == 'v9_hybrid' for x in target.branch_id):
        raise NativeTargetError('G-NT1 cannot consume hybrid target')
    return {'baseline': 'G-NT1', 'count': int(target.pulse.sum()),
            'abstain': not bool(target.traversal_bounds.size), 'true_unit_status': 'unvalidated'}
