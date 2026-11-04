"""NB1 full-native, feature-only CPU interface; independent of fixed320 archives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.temporac.decode import connected_components, quantize_response
from pams.temporac.preprocess import (
    BONES,
    DIRECTION_NORM_FLOOR,
    JOINT_COUNT,
    GeometryFeatures,
    PreprocessError,
    _geometry_masks,
    _masked_norm,
    collapse_duplicate_clocks,
    mask_to_half_open_runs,
    validate_run_bounds,
)
from pams.temporac.quadrature import neumaier_sum
from pams.temporac.timebase_audit import build_timebase

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SCHEMA = "temporac.native-learner.v1"
TARGET_SCHEMA = "temporac.native-target.v1"


def freeze(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def binary(a, shape, name):
    a = np.asarray(a)
    if a.shape != shape or a.dtype.kind not in "bu" or not np.isin(a, (0, 1)).all():
        raise ValueError(f"{name}: expected binary {shape}")
    return freeze(a.astype(bool))


@dataclass(frozen=True)
class NativeFeatureV1:
    opaque_key: np.ndarray
    slot: np.ndarray
    source_length: np.ndarray
    source_frame_index: np.ndarray
    xyconf_raw: np.ndarray
    joint_observed: np.ndarray
    frame_source_ok: np.ndarray
    pts: np.ndarray
    pts_present: np.ndarray
    time_base: np.ndarray

    def __post_init__(self):
        n = np.asarray(self.xyconf_raw).shape[0]
        if n < 1:
            raise ValueError("native identity requires >=1 original rows")
        specs = {
            "opaque_key": (np.uint8, (32,)),
            "slot": (np.int64, (1,)),
            "source_length": (np.int64, (1,)),
            "source_frame_index": (np.int64, (n,)),
            "xyconf_raw": (np.float32, (n, 17, 3)),
            "pts": (np.int64, (n,)),
            "time_base": (np.int64, (2,)),
        }
        for name, (dtype, shape) in specs.items():
            a = np.asarray(getattr(self, name))
            if a.dtype != np.dtype(dtype) or a.shape != shape:
                raise ValueError(f"{name}: dtype/shape mismatch")
            object.__setattr__(self, name, freeze(a))
        for name, shape in (
            ("joint_observed", (n, 17)),
            ("frame_source_ok", (n,)),
            ("pts_present", (n,)),
        ):
            object.__setattr__(self, name, binary(getattr(self, name), shape, name))
        if (
            self.source_length[0] != n
            or self.slot[0] < 0
            or not np.array_equal(self.source_frame_index, np.arange(n))
        ):
            raise ValueError("native source identity/order mismatch")
        if np.any(self.joint_observed & ~np.isfinite(self.xyconf_raw).all(axis=2)):
            raise ValueError("observed joint must be finite")
        if np.any(self.xyconf_raw[~self.joint_observed] != 0):
            raise ValueError("unobserved payload must be zero")
        if self.pts_present.any() and np.any(self.time_base <= 0):
            raise ValueError("real PTS requires positive rational time_base")

    def as_arrays(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_native_feature(path) -> NativeFeatureV1:
    with np.load(path, allow_pickle=False) as a:
        if set(a.files) != {f.name for f in fields(NativeFeatureV1)}:
            raise ValueError("feature-only schema member mismatch")
        return NativeFeatureV1(**{k: a[k] for k in a.files})


def project_raw_identity(arrays, meta, slot):
    """Trusted projection only; raw schema validation is the caller's prerequisite."""
    n = int(meta["annotation_length"])
    aligned = n == meta["decoded_length"]
    raw_usable = aligned and meta["status"] == "complete"
    key = bytes.fromhex(meta["media_key"])
    if len(key) != 32 or slot not in arrays["slot"]:
        raise ValueError("raw identity mismatch")
    ok = (
        (arrays["decoded_ok"] & arrays["bbox_valid"][slot] & arrays["pose_returned"][slot])
        if raw_usable
        else np.zeros(n, bool)
    )
    observed = arrays["joint_observed"][slot].copy() if aligned else np.zeros((n, 17), bool)
    raw = arrays["xyconf_raw"][slot].copy() if aligned else np.zeros((n, 17, 3), np.float32)
    clocks = meta.get("clocks", {})
    return NativeFeatureV1(
        np.frombuffer(key, np.uint8),
        np.array([slot], np.int64),
        np.array([n], np.int64),
        np.arange(n, dtype=np.int64),
        raw,
        observed,
        ok,
        arrays["pts"].copy() if aligned else np.zeros(n, np.int64),
        arrays["pts_present"].copy() if aligned else np.zeros(n, bool),
        np.array([clocks.get("time_base_num") or 0, clocks.get("time_base_den") or 0], np.int64),
    )


@dataclass(frozen=True)
class NativeClockV1:
    mode: str
    q: np.ndarray
    seconds: np.ndarray
    present: np.ndarray
    edge_time_support: np.ndarray
    segments: tuple[tuple[int, int], ...]
    tau: float | None
    anchor_pts: int | None
    reasons: tuple[str, ...]


def native_clock(feature):
    n = int(feature.source_length[0])
    present = feature.pts_present
    reasons = []
    if not present.any():
        q = np.arange(n, dtype=np.float64)
        seconds = np.full(n, np.nan)
        valid = np.ones(n, bool)
        mode, tau, anchor = "source_frame_only", None, None
    else:
        mode = "pts"
        valid = present.copy()
        anchor = int(feature.pts[np.flatnonzero(present)[0]])
        num, den = map(int, feature.time_base)
        # Python integer subtraction precedes any float conversion, including diff.
        differences = [int(feature.pts[j + 1]) - int(feature.pts[j]) for j in range(n - 1)]
        positive = [d for j, d in enumerate(differences) if present[j] and present[j + 1] and d > 0]
        seconds = np.full(n, np.nan)
        seconds[present] = [
            float(int(feature.pts[j]) - anchor) * num / den for j in np.flatnonzero(present)
        ]
        tau = float(np.median(np.asarray(positive, np.float64))) * num / den if positive else None
        q = seconds / tau if tau is not None and tau > 0 else np.full(n, np.nan)
        if tau is None:
            valid[:] = False
            reasons.append("clock_invalid")
        if not present.all():
            reasons.append("missing_pts")
    dq = np.diff(q)
    edge = valid[:-1] & valid[1:] & np.isfinite(dq) & (dq > 0) & (dq < 5)
    if np.any(valid[:-1] & valid[1:] & (dq <= 0)):
        reasons.append("nonpositive_pts_step")
    if np.any(dq >= 5):
        reasons.append("clock_gap_ge5")
    # Mapping uses strict increasing segments, split at every unsupported clock edge.
    segments = []
    start = 0
    while start < n:
        if not valid[start]:
            start += 1
            continue
        stop = start + 1
        while stop < n and edge[stop - 1]:
            stop += 1
        segments.append((start, stop))
        start = stop
    return NativeClockV1(
        mode,
        freeze(q),
        freeze(seconds),
        freeze(valid),
        freeze(edge),
        tuple(segments),
        tau,
        anchor,
        tuple(reasons),
    )


def native_mapping(feature, clock):
    n = int(feature.source_length[0])
    mapping = {
        "source_frame_index": np.arange(n),
        "q": clock.q,
        "seconds": clock.seconds,
        "segment": np.full(n, -1, np.int32),
    }
    for name in ("left_source_index", "right_source_index"):
        mapping[name] = np.full((n, 17), -1, np.int64)
    for name in ("left_weight", "right_weight", "gap_span"):
        mapping[name] = np.zeros((n, 17), np.float64)
    mapping["support_kind"] = np.full((n, 17), "unsupported", dtype="<U12")
    mapping["supported"] = np.zeros((n, 17), bool)
    for seg, (start, stop) in enumerate(clock.segments):
        mapping["segment"][start:stop] = seg
        if stop - start < 2:
            continue
        for joint in range(17):
            observed = feature.joint_observed[start:stop, joint]
            if not observed.any():
                continue
            m = build_timebase(
                clock.q[start:stop],
                clock.q[start:stop],
                observed,
                clock_unit="tau" if clock.mode == "pts" else "source_frames",
                max_interpolation_gap=0,
            )
            for name in ("left_source_index", "right_source_index"):
                mapping[name][start:stop, joint] = getattr(m, name) + start
            for name in ("left_weight", "right_weight", "gap_span", "support_kind", "supported"):
                mapping[name][start:stop, joint] = getattr(m, name)
    return {k: freeze(v) for k, v in mapping.items()}


@dataclass(frozen=True)
class NativeIdentityV1:
    feature: NativeFeatureV1
    clock: NativeClockV1
    motion: np.ndarray
    joint_mask: np.ndarray
    frame_valid: np.ndarray
    geometry_features: GeometryFeatures
    observed_edge_support: np.ndarray
    response_edge_support: np.ndarray
    short_runs: np.ndarray
    mapping: dict
    status: str
    reasons: tuple[str, ...]
    scale: float | None

    @property
    def run_bounds(self):
        return self.geometry_features.run_bounds

    @property
    def response_input(self):
        return self.geometry_features.response_input

    @property
    def edge_lengths(self):
        return self.geometry_features.edge_lengths


def preprocess_native(feature, clock_meta=None) -> NativeIdentityV1:
    """Preserve N original rows; clock_meta is optional terminal-boundary metadata only."""
    if clock_meta is not None and set(clock_meta) - {"terminal_seconds", "terminal_verified"}:
        raise ValueError("clock_meta cannot override PTS/tau")
    n = int(feature.source_length[0])
    clock = native_clock(feature)
    mapping = native_mapping(feature, clock)
    raw = feature.xyconf_raw.copy()
    raw[~feature.joint_observed] = 0
    reasons = list(clock.reasons)
    scale = None
    try:
        track = collapse_duplicate_clocks(raw, feature.frame_source_ok, feature.source_frame_index)
        motion, mask, frame_valid, scale = (
            track.motion,
            track.joint_mask,
            track.frame_valid,
            track.scale,
        )
    except PreprocessError as exc:
        reasons.append(exc.reason)
        motion = np.zeros((n, 17, 3), np.float32)
        mask = np.zeros((n, 17), bool)
        frame_valid = np.zeros(n, bool)
    _, coordinate_mask = _geometry_masks(mask)
    common = np.count_nonzero(coordinate_mask[:-1] & coordinate_mask[1:], axis=1)
    edge = frame_valid[:-1] & frame_valid[1:] & (common >= 32) & clock.edge_time_support
    all_runs = mask_to_half_open_runs(edge)
    short = np.asarray([r for r in all_runs.tolist() if r[1] - r[0] + 1 < 17], np.int32).reshape(
        -1, 2
    )
    runs = np.asarray([r for r in all_runs.tolist() if r[1] - r[0] + 1 >= 17], np.int32).reshape(
        -1, 2
    )
    response_support = np.zeros(max(0, n - 1), bool)
    for start, stop in runs:
        response_support[start:stop] = True
    if short.size:
        reasons.append("short_run")
    # Build with all observed runs so lengths/directions remain traceable, then build
    # executable lag channels separately; no lag crosses any failed/short run.
    geometry = build_native_geometry_features(motion, mask, all_runs)
    executable = build_native_geometry_features(motion, mask, runs)
    geometry = GeometryFeatures(
        geometry.pose,
        geometry.bones,
        geometry.geometry,
        geometry.element_mask,
        geometry.coordinate_mask,
        geometry.edge_coordinate_mask,
        geometry.edge_lengths,
        geometry.direction,
        geometry.direction_element_mask,
        geometry.teacher_input,
        executable.response_input,
        executable.run_bounds,
    )
    status = (
        "complete"
        if response_support.size and response_support.all()
        else "partial"
        if response_support.any()
        else "unusable"
    )
    if status == "unusable":
        reasons.append("no_executable_support")
    return NativeIdentityV1(
        feature,
        clock,
        freeze(motion),
        freeze(mask),
        freeze(frame_valid),
        geometry,
        freeze(edge),
        freeze(response_support),
        freeze(short),
        mapping,
        status,
        tuple(reasons),
        scale,
    )


def build_native_geometry_features(
    normalized_motion: ArrayLike,
    joint_mask: ArrayLike,
    run_bounds: ArrayLike,
) -> GeometryFeatures:
    """NB1 copy of F4: sole arithmetic extension allows finite exact-zero edges."""

    motion = np.asarray(normalized_motion, dtype=np.float64)
    raw_mask = np.asarray(joint_mask)
    if motion.ndim != 3 or motion.shape[1:] != (JOINT_COUNT, 3):
        raise PreprocessError("SOURCE_SCHEMA", "normalized_motion must be [T,17,3]")
    sample_count = motion.shape[0]
    edge_count = sample_count - 1
    if sample_count < 1:
        raise PreprocessError("TOO_SHORT", "native geometry requires at least one sample")
    if raw_mask.shape != (sample_count, JOINT_COUNT) or raw_mask.dtype.kind not in "bu":
        raise PreprocessError("SOURCE_SCHEMA", "joint_mask must be binary [T,17]")
    if not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise PreprocessError("SOURCE_SCHEMA", "joint_mask must be binary")
    mask = np.asarray(raw_mask, dtype=np.bool_)
    if not np.all(np.isfinite(motion[mask])):
        raise PreprocessError("NORMALIZATION", "valid normalized joints must be finite")
    bounds = validate_run_bounds(run_bounds, edge_count)

    coordinates = np.zeros((sample_count, JOINT_COUNT, 2), dtype=np.float64)
    coordinates[mask] = motion[:, :, :2][mask]
    confidence = np.zeros((sample_count, JOINT_COUNT), dtype=np.float64)
    confidence[mask] = motion[:, :, 2][mask]
    pose = np.ascontiguousarray(coordinates.reshape(sample_count, 34))
    bones = np.zeros((sample_count, 32), dtype=np.float64)
    for index, (source, target) in enumerate(BONES):
        supported = mask[:, source] & mask[:, target]
        bones[supported, 2 * index : 2 * index + 2] = (
            coordinates[supported, target] - coordinates[supported, source]
        )
    geometry = np.ascontiguousarray(np.concatenate((pose, bones), axis=1))
    element_mask, coordinate_mask = _geometry_masks(mask)

    edge_coordinate_mask = np.zeros((edge_count, 66), dtype=np.bool_)
    edge_lengths = np.zeros(edge_count, dtype=np.float64)
    for start, stop in bounds.tolist():
        for edge in range(start, stop):
            edge_mask = coordinate_mask[edge] & coordinate_mask[edge + 1]
            edge_coordinate_mask[edge] = edge_mask
            dimension_count = int(np.count_nonzero(edge_mask))
            if dimension_count < 32:
                raise PreprocessError(
                    "GEOMETRY_LENGTH", f"edge {edge} has fewer than 32 geometry coordinates"
                )
            delta = geometry[edge + 1] - geometry[edge]
            squared_sum = neumaier_sum(float(value * value) for value in delta[edge_mask])
            scaled = np.float64(66.0 / dimension_count) * np.float64(squared_sum)
            length = float(np.sqrt(scaled)) if scaled >= 0.0 else float("nan")
            if not np.isfinite(squared_sum) or not np.isfinite(length) or length < 0.0:
                raise PreprocessError("GEOMETRY_LENGTH", f"edge {edge} has invalid masked length")
            edge_lengths[edge] = length

    direction = np.zeros((sample_count, 66), dtype=np.float64)
    direction_element_mask = np.zeros((sample_count, 33), dtype=np.bool_)
    for start, stop in bounds.tolist():
        # Edge run [start, stop) owns sample run [start, stop].
        for sample in range(start + 1, stop):
            common_elements = (
                element_mask[sample - 1] & element_mask[sample] & element_mask[sample + 1]
            )
            common_coordinates = np.repeat(common_elements, 2)
            if not np.any(common_coordinates):
                continue
            incoming = geometry[sample] - geometry[sample - 1]
            outgoing = geometry[sample + 1] - geometry[sample]
            incoming_norm = _masked_norm(incoming, common_coordinates)
            outgoing_norm = _masked_norm(outgoing, common_coordinates)
            if (
                not np.isfinite(incoming_norm)
                or not np.isfinite(outgoing_norm)
                or incoming_norm <= DIRECTION_NORM_FLOOR
                or outgoing_norm <= DIRECTION_NORM_FLOOR
            ):
                continue
            chord_sum = np.zeros(66, dtype=np.float64)
            chord_sum[common_coordinates] = incoming[common_coordinates] / np.float64(
                incoming_norm
            ) + outgoing[common_coordinates] / np.float64(outgoing_norm)
            chord_sum_norm = _masked_norm(chord_sum, common_coordinates)
            if not np.isfinite(chord_sum_norm) or chord_sum_norm <= DIRECTION_NORM_FLOOR:
                continue
            direction[sample, common_coordinates] = chord_sum[common_coordinates] / np.float64(
                chord_sum_norm
            )
            direction_element_mask[sample] = common_elements

    teacher_input_f64 = np.concatenate(
        (
            geometry,
            direction,
            confidence,
            element_mask.astype(np.float64),
            direction_element_mask.astype(np.float64),
        ),
        axis=1,
    )
    if teacher_input_f64.shape != (sample_count, 215):
        raise AssertionError("teacher channel construction did not produce 215 channels")

    response_input_f64 = np.zeros((sample_count, 269), dtype=np.float64)
    response_input_f64[:, 0:66] = geometry
    response_input_f64[:, 66:83] = confidence
    response_input_f64[:, 185:202] = mask.astype(np.float64)
    response_input_f64[:, 202:218] = element_mask[:, 17:33].astype(np.float64)
    for lag, value_slice, mask_slice in (
        (4, slice(83, 117), slice(218, 235)),
        (2, slice(117, 151), slice(235, 252)),
        (1, slice(151, 185), slice(252, 269)),
    ):
        for start, stop in bounds.tolist():
            for sample in range(start + lag, stop + 1):
                response_input_f64[sample, value_slice] = pose[sample - lag]
                response_input_f64[sample, mask_slice] = mask[sample - lag].astype(np.float64)
    if not np.all(np.isfinite(teacher_input_f64)) or not np.all(np.isfinite(response_input_f64)):
        raise PreprocessError("NORMALIZATION", "model input contains a nonfinite value")

    return GeometryFeatures(
        pose=pose,
        bones=np.ascontiguousarray(bones),
        geometry=geometry,
        element_mask=element_mask,
        coordinate_mask=coordinate_mask,
        edge_coordinate_mask=np.ascontiguousarray(edge_coordinate_mask),
        edge_lengths=np.ascontiguousarray(edge_lengths),
        direction=np.ascontiguousarray(direction),
        direction_element_mask=np.ascontiguousarray(direction_element_mask),
        teacher_input=np.ascontiguousarray(teacher_input_f64.astype(np.dtype("<f4"))),
        response_input=np.ascontiguousarray(response_input_f64.astype(np.dtype("<f4"))),
        run_bounds=bounds,
    )


class NOLAError(ValueError):
    """Invalid masked native reconstruction."""


@runtime_checkable
class EdgeWindow(Protocol):
    edge_start: int
    edge_stop: int


@dataclass(frozen=True, slots=True)
class NOLAResult:
    numerator: Float64Array
    denominator: Float64Array
    response: Float64Array
    window_response: tuple[Float64Array, ...]

    def __post_init__(self) -> None:
        self.numerator.setflags(write=False)
        self.denominator.setflags(write=False)
        self.response.setflags(write=False)
        for value in self.window_response:
            value.setflags(write=False)


def _window_interval(window: EdgeWindow | Sequence[int]) -> tuple[int, int]:
    if isinstance(window, EdgeWindow):
        start, stop = int(window.edge_start), int(window.edge_stop)
    else:
        if len(window) != 2:
            raise NOLAError("window tuples must contain (edge_start, edge_stop)")
        start, stop = int(window[0]), int(window[1])
    if not (0 <= start < stop):
        raise NOLAError("windows must be nonempty half-open edge intervals")
    return start, stop


def _validated_windows(
    windows: Sequence[EdgeWindow | Sequence[int]], edge_count: int
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    previous_start = -1
    for window in windows:
        start, stop = _window_interval(window)
        if stop > edge_count:
            raise NOLAError("window exceeds the response edge sequence")
        if start < previous_start:
            raise NOLAError("NOLA windows must follow window-start order")
        previous_start = start
        result.append((start, stop))
    return tuple(result)


def _binary_edge_mask(mask: ArrayLike, edge_count: int) -> NDArray[np.bool_]:
    raw = np.asarray(mask)
    if raw.shape != (edge_count,) or raw.dtype.kind not in "bu":
        raise NOLAError("edge_mask must be binary shape [E]")
    if not np.all((raw == 0) | (raw == 1)):
        raise NOLAError("edge_mask must be binary")
    return np.asarray(raw, dtype=np.bool_)


def window_taper(edge_slots: int) -> Float64Array:
    """Return the exact strictly-positive F13 taper for local edge slots."""

    if not (1 <= edge_slots <= 127):
        raise NOLAError("a local window must contain between 1 and 127 edge slots")
    slot = np.arange(edge_slots, dtype=np.float64)
    taper = np.float64(1e-3) + np.float64(1.0 - 1e-3) * np.sin(
        np.pi * (slot + np.float64(0.5)) / np.float64(127.0)
    ) ** np.float64(2.0)
    if not np.all(np.isfinite(taper)) or not np.all(taper >= np.float64(1e-3)):
        raise NOLAError("NOLA taper is not strictly positive")
    return np.ascontiguousarray(taper)


def _validate_numpy_probabilities(
    branch_probabilities: ArrayLike, gates: ArrayLike
) -> tuple[Float64Array, Float64Array]:
    probabilities = np.asarray(branch_probabilities, dtype=np.float64)
    gate_array = np.asarray(gates, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise NOLAError("branch_probabilities must have shape [E,3]")
    if gate_array.ndim != 2 or gate_array.shape[1] != 3:
        raise NOLAError("gates must have shape [W,3]")
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise NOLAError("branch probabilities must be finite values in [0,1]")
    if not np.all(np.isfinite(gate_array)) or np.any(gate_array < 0.0):
        raise NOLAError("gates must be finite and nonnegative")
    if not np.allclose(np.sum(gate_array, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise NOLAError("each routing gate must sum to one")
    return probabilities, gate_array


def probability_fusion(branch_probabilities: ArrayLike, gate: ArrayLike) -> Float64Array:
    """Fuse sigmoid probabilities (never logits) with one three-way gate."""

    probabilities = np.asarray(branch_probabilities, dtype=np.float64)
    gate_array = np.asarray(gate, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise NOLAError("branch_probabilities must have shape [N,3]")
    if gate_array.shape != (3,) or not np.all(np.isfinite(gate_array)):
        raise NOLAError("gate must be finite shape [3]")
    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.all(np.isfinite(probabilities))
    ):
        raise NOLAError("branch probabilities must be finite values in [0,1]")
    if np.any(gate_array < 0.0) or not np.isclose(np.sum(gate_array), 1.0, rtol=0.0, atol=1e-12):
        raise NOLAError("gate must be nonnegative and sum to one")
    return np.ascontiguousarray(probabilities @ gate_array, dtype=np.float64)


def fuse_window_probabilities(
    branch_probabilities: ArrayLike,
    gates: ArrayLike,
    windows: Sequence[EdgeWindow | Sequence[int]],
) -> tuple[Float64Array, ...]:
    """Slice cached run-level probabilities and fuse every window once."""

    probabilities, gate_array = _validate_numpy_probabilities(branch_probabilities, gates)
    intervals = _validated_windows(windows, probabilities.shape[0])
    if gate_array.shape[0] != len(intervals):
        raise NOLAError("gate row count must equal window count")
    return tuple(
        probability_fusion(probabilities[start:stop], gate_array[index])
        for index, (start, stop) in enumerate(intervals)
    )


def _neumaier_add_array(total: Float64Array, correction: Float64Array, value: Float64Array) -> None:
    candidate = total + value
    use_total = np.abs(total) >= np.abs(value)
    correction += np.where(
        use_total,
        (total - candidate) + value,
        (value - candidate) + total,
    )
    total[:] = candidate


def reconstruct_native_response(
    branch_probabilities: ArrayLike,
    gates: ArrayLike,
    windows: Sequence[EdgeWindow | Sequence[int]],
    edge_mask: ArrayLike,
) -> NOLAResult:
    """Fuse F12 and reconstruct F13 with no epsilon, clamp, or fallback."""

    probabilities, gate_array = _validate_numpy_probabilities(branch_probabilities, gates)
    edge_count = probabilities.shape[0]
    intervals = _validated_windows(windows, edge_count)
    if gate_array.shape[0] != len(intervals):
        raise NOLAError("gate row count must equal window count")
    valid = _binary_edge_mask(edge_mask, edge_count)
    numerator = np.zeros(edge_count, dtype=np.float64)
    numerator_correction = np.zeros(edge_count, dtype=np.float64)
    denominator = np.zeros(edge_count, dtype=np.float64)
    denominator_correction = np.zeros(edge_count, dtype=np.float64)
    fused_windows: list[Float64Array] = []

    for window_index, (start, stop) in enumerate(intervals):
        fused = probability_fusion(probabilities[start:stop], gate_array[window_index])
        fused_windows.append(fused)
        taper = window_taper(stop - start)
        window_valid = valid[start:stop].astype(np.float64)
        numerator_value = taper * window_valid * fused
        denominator_value = taper * window_valid
        _neumaier_add_array(
            numerator[start:stop], numerator_correction[start:stop], numerator_value
        )
        _neumaier_add_array(
            denominator[start:stop], denominator_correction[start:stop], denominator_value
        )

    numerator += numerator_correction
    denominator += denominator_correction
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise NOLAError("NOLA numerator or denominator is nonfinite")
    if np.any(denominator[valid] < np.float64(1e-3)):
        raise NOLAError("every valid edge requires exact NOLA denominator >= 1e-3")
    if np.any(denominator[~valid] != 0.0) or np.any(numerator[~valid] != 0.0):
        raise NOLAError("masked NOLA edges must remain exact zero")
    response = np.zeros(edge_count, dtype=np.float64)
    # This division is executed only after the strictly-positive proof.  There
    # is intentionally no epsilon in the denominator.
    response[valid] = numerator[valid] / denominator[valid]
    if (
        not np.all(np.isfinite(response))
        or np.any(response[valid] < 0.0)
        or np.any(response[valid] > 1.0)
    ):
        raise NOLAError("reconstructed response is not a finite probability")
    return NOLAResult(
        numerator=np.ascontiguousarray(numerator),
        denominator=np.ascontiguousarray(denominator),
        response=np.ascontiguousarray(response),
        window_response=tuple(fused_windows),
    )


@dataclass(frozen=True)
class NativePredictionV1:
    response: np.ndarray
    support: np.ndarray
    run_bounds: np.ndarray
    component_bounds: np.ndarray
    component_source_ranges: np.ndarray
    event_source_positions: np.ndarray
    event_seconds: np.ndarray
    count: int
    status: str
    decoder_invocations: int = 1
    period_ap_status: str = "unavailable_no_period_decoder"


def decode_native_identity(response, identity) -> NativePredictionV1:
    """One full-identity component call, even with no executable support."""
    support = identity.response_edge_support
    r = quantize_response(response)
    if r.shape != support.shape or np.any((r < 0) | (r > 1)):
        raise ValueError("native response must be full-length probabilities")
    r[~support] = 0
    if support.any():
        components = connected_components(r, support, identity.run_bounds)
    else:
        components = connected_components(
            np.empty(0, dtype="<f4"), np.empty(0, np.uint8), np.empty((0, 2), np.int32)
        )
    positions = components.location.astype(np.float64) + 0.5
    seconds = np.full(positions.size, np.nan)
    if identity.clock.mode == "pts":
        for j, edge in enumerate(components.location):
            if identity.clock.edge_time_support[edge]:
                seconds[j] = (identity.clock.seconds[edge] + identity.clock.seconds[edge + 1]) / 2
    return NativePredictionV1(
        freeze(r),
        support,
        identity.run_bounds,
        components.bounds,
        freeze(components.bounds.astype(np.float64)),
        freeze(positions),
        freeze(seconds),
        components.count if support.any() else -1,
        "complete" if support.size and support.all() else "partial" if support.any() else "abstain",
    )


def source_boundary_seconds(identity, boundary, clock_meta=None):
    """Discrete source frame boundaries [0,N]; endpoint N is never clamped."""
    n = int(identity.feature.source_length[0])
    if type(boundary) is not int or not 0 <= boundary <= n:
        raise ValueError("source boundary outside [0,N]")
    if identity.clock.mode != "pts":
        return None
    if boundary < n:
        return float(identity.clock.seconds[boundary]) if identity.clock.present[boundary] else None
    if clock_meta and clock_meta.get("terminal_verified") is True:
        value = clock_meta.get("terminal_seconds")
        if (
            value is not None
            and np.isfinite(value)
            and identity.clock.present[-1]
            and value > identity.clock.seconds[-1]
        ):
            return float(value)
        raise ValueError("verified terminal_seconds must follow last timestamp")
    return None


def sample_position_seconds(identity, position, *, segment):
    """Map sample centers only inside the named clock segment; never bridge a gap."""
    if type(segment) is not int or not 0 <= segment < len(identity.clock.segments):
        raise ValueError("unknown clock segment")
    start, stop = identity.clock.segments[segment]
    if not np.isfinite(position) or not start <= position <= stop - 1:
        raise ValueError("position outside named sample-center segment")
    if identity.clock.mode != "pts":
        return None
    left, right = int(np.floor(position)), int(np.ceil(position))
    return float(
        identity.clock.seconds[left]
        + (position - left) * (identity.clock.seconds[right] - identity.clock.seconds[left])
    )


@dataclass(frozen=True)
class NativeTargetV1:
    target_valid: np.ndarray
    pulse: np.ndarray
    chi: np.ndarray
    traversal_bounds: np.ndarray
    branch_provenance: str
    producer: str
    producer_sha256: str
    traversal_provenance: tuple[str, ...] | None = None

    def __post_init__(self):
        e = np.asarray(self.pulse).size
        object.__setattr__(self, "target_valid", binary(self.target_valid, (e,), "target_valid"))
        for name in ("pulse", "chi"):
            a = np.asarray(getattr(self, name), np.float64)
            if a.shape != (e,) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
                raise ValueError("target values must be full-length finite [0,1]")
            if np.any(a[~self.target_valid] != 0):
                raise ValueError("unsupported target payload must be zero")
            object.__setattr__(self, name, freeze(a))
        b = np.asarray(self.traversal_bounds)
        if b.ndim != 2 or b.shape[1:] != (2,) or b.dtype.kind not in "iu":
            raise ValueError("traversal bounds must be explicit integer [K,2]")
        previous = 0
        for start, stop in b.tolist():
            if (
                not 0 <= start < stop <= e
                or start < previous
                or not self.target_valid[start:stop].all()
            ):
                raise ValueError("traversal crosses uncertified target support")
            previous = stop
        object.__setattr__(self, "traversal_bounds", freeze(b.astype(np.int64)))
        sources = ("full-chain geometry", "stable-pair hybrid")
        if self.branch_provenance not in (*sources, "mixed", "rejected"):
            raise ValueError("unknown branch provenance")
        provenance = self.traversal_provenance
        if provenance is None:
            if self.branch_provenance == "mixed":
                raise ValueError("mixed summary requires per-traversal provenance")
            provenance = (self.branch_provenance,) * len(b)
        if not isinstance(provenance, (list, tuple)) or len(provenance) != len(b):
            raise ValueError("traversal provenance must match traversal bounds one-to-one")
        if any(source not in sources for source in provenance):
            raise ValueError("traversal provenance must name a certified source branch")
        unique = set(provenance)
        summary = "mixed" if len(unique) > 1 else next(iter(unique), self.branch_provenance)
        if self.branch_provenance != summary or (
            self.branch_provenance == "mixed" and len(unique) < 2
        ):
            raise ValueError("branch provenance summary disagrees with traversals")
        object.__setattr__(self, "traversal_provenance", tuple(provenance))
        if (
            not self.producer
            or len(self.producer_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.producer_sha256)
        ):
            raise ValueError("target producer/hash required")
        if self.branch_provenance == "rejected" and (self.target_valid.any() or b.size):
            raise ValueError("rejected target cannot supervise")


def load_native_target(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        set(meta) - {"traversal_provenance"}
        != {"schema", "branch_provenance", "producer", "producer_sha256", "npz_sha256"}
        or meta["schema"] != TARGET_SCHEMA
    ):
        raise ValueError("native target sidecar mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["npz_sha256"]:
        raise ValueError("target payload hash mismatch")
    with np.load(path, allow_pickle=False) as a:
        if set(a.files) != {"target_valid", "pulse", "chi", "traversal_bounds"}:
            raise ValueError("native target members mismatch")
        return NativeTargetV1(
            **{k: a[k] for k in a.files},
            **{k: meta[k] for k in ("branch_provenance", "producer", "producer_sha256")},
            traversal_provenance=meta.get("traversal_provenance"),
        )


def native_supervision(identity, target):
    if target.target_valid.shape != identity.response_edge_support.shape:
        raise ValueError("target must use complete native source edge domain")
    supervised = identity.response_edge_support & target.target_valid
    eligible, reasons = [], []
    for start, stop in target.traversal_bounds:
        m = supervised[start:stop]
        chi = target.chi[start:stop]
        pulse = target.pulse[start:stop]
        wp = neumaier_sum(chi[m] * pulse[m])
        wn = neumaier_sum(chi[m] * (1 - pulse[m]))
        if not identity.response_edge_support[start:stop].all():
            reason = "response_support_gap"
        elif wp <= 0 or wn <= 0:
            reason = "missing_positive_or_negative_weight"
        else:
            reason = None
        ok = reason is None
        if not ok:
            supervised[start:stop] = False
        eligible.append(ok)
        reasons.append(reason)
    # These are raw masked masses; no new loss normalization is introduced.
    positive_weights = np.zeros(supervised.size, np.float64)
    negative_weights = np.zeros(supervised.size, np.float64)
    positive_weights[supervised] = target.chi[supervised] * target.pulse[supervised]
    negative_weights[supervised] = target.chi[supervised] * (1 - target.pulse[supervised])
    # Training must reduce by complete identity, never by number of windows/runs.
    return {
        "supervised": freeze(supervised),
        "positive_weights": freeze(positive_weights),
        "negative_weights": freeze(negative_weights),
        "traversal_eligible": tuple(eligible),
        "traversal_reasons": tuple(reasons),
        "reduction_unit": "complete_identity_source_edges",
    }
