"""NTR1 exact ordered frequency-axis reduction. No changed time reduction order."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
Float64Array = NDArray[np.float64]
from pams.temporac.cue import FREQUENCIES, PERIOD_COUNT
from pams.temporac.quadrature import neumaier_sum


def frequency_ordered_sum(terms):
    terms = np.asarray(terms, dtype=np.float64)
    if terms.ndim != 2:
        raise ValueError('frequency terms require [frequencies, time]')
    total = np.zeros(terms.shape[0], dtype=np.float64)
    correction = np.zeros_like(total)
    for index in range(terms.shape[1]):
        value = terms[:, index]
        candidate = total + value
        correction = correction + np.where(np.abs(total) >= np.abs(value),
            (total - candidate) + value, (value - candidate) + total)
        total = candidate
    return total + correction


def coordinate_spectrum_ordered(
    velocity: Float64Array,
    midpoint: Float64Array,
    weight: Float64Array,
) -> Float64Array | None:
    if velocity.size < 3:
        return None
    span = float(midpoint[-1] - midpoint[0])
    total_weight = neumaier_sum(weight)
    if not np.isfinite(span) or span <= 0.0 or not np.isfinite(total_weight) or total_weight <= 0.0:
        return None
    center = np.float64((midpoint[0] + midpoint[-1]) / 2.0)
    half_span = np.float64(span / 2.0)
    z = (midpoint - center) / half_span
    s0 = total_weight
    s1 = neumaier_sum(weight * z)
    s2 = neumaier_sum(weight * z * z)
    t0 = neumaier_sum(weight * velocity)
    t1 = neumaier_sum(weight * z * velocity)
    determinant = np.float64(s0 * s2 - s1 * s1)
    if (
        s0 <= 0.0
        or s2 <= 0.0
        or not np.isfinite(determinant)
        or determinant < np.float64(1e-12) * np.float64(s0) * np.float64(s2)
    ):
        return None
    alpha = np.float64((t0 * s2 - t1 * s1) / determinant)
    beta = np.float64((t1 * s0 - t0 * s1) / determinant)
    residual = velocity - alpha - beta * z
    power = neumaier_sum(weight * residual * residual)
    denominator = np.float64(s0 * power)
    if not np.isfinite(denominator) or denominator <= np.float64(1e-12):
        return None

    spectrum = np.empty(PERIOD_COUNT, dtype=np.float64)
    real_terms, imaginary_terms = [], []
    for frequency in FREQUENCIES:
        angle = np.float64(2.0 * np.pi) * frequency * midpoint
        real_terms.append(weight * residual * np.cos(angle))
        imaginary_terms.append(-weight * residual * np.sin(angle))
    real = frequency_ordered_sum(np.stack(real_terms))
    imaginary = frequency_ordered_sum(np.stack(imaginary_terms))
    for frequency_index in range(PERIOD_COUNT):
        r, i = real[frequency_index], imaginary[frequency_index]
        spectrum[frequency_index] = np.float64((r * r + i * i) / denominator)
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        return None
    return spectrum

from dataclasses import dataclass
from pams.temporac.native_time_cue_v1 import shared_native_quadrature
from pams.temporac.cue import (LOG_FREQUENCIES, CueError, WindowCue, _failure_cue,
    Window, window_schedule, _coverage, _run_reference, _gate_for_cue)
from pams.temporac.quadrature import QuadratureError, neumaier_sum_array
from pams.temporac.preprocess import validate_run_bounds

def compute_native_window_cue_ordered(
    geometry: ArrayLike,
    coordinate_edge_mask: ArrayLike,
    clocks: ArrayLike,
    edge_mask: ArrayLike,
    window: Window,
) -> WindowCue:
    """Compute F6 and reliability on one shared irregular-clock quadrature."""

    g = np.asarray(geometry, dtype=np.float64)
    coordinate_mask_raw = np.asarray(coordinate_edge_mask)
    q_raw = np.asarray(clocks)
    base_mask_raw = np.asarray(edge_mask)
    if g.ndim != 2 or g.shape[1] != 66 or g.shape[0] < 2:
        raise CueError("geometry must have shape [T,66]")
    edge_count = g.shape[0] - 1
    if coordinate_mask_raw.shape != (edge_count, 66) or coordinate_mask_raw.dtype.kind not in "bu":
        raise CueError("coordinate_edge_mask must be binary [E,66]")
    if not np.all((coordinate_mask_raw == 0) | (coordinate_mask_raw == 1)):
        raise CueError("coordinate_edge_mask must be binary")
    if q_raw.shape != (g.shape[0],) or q_raw.dtype.kind not in "fiu":
        raise CueError("native clocks must be numeric shape [T]")
    if base_mask_raw.shape != (edge_count,) or base_mask_raw.dtype.kind not in "bu":
        raise CueError("edge_mask must be binary shape [E]")
    if not np.all((base_mask_raw == 0) | (base_mask_raw == 1)):
        raise CueError("edge_mask must be binary")
    if not (0 <= window.edge_start < window.edge_stop <= edge_count):
        raise CueError("window is outside the edge sequence")
    if not np.all(np.isfinite(g)):
        raise CueError("geometry must be finite")

    selected_q = np.asarray(q_raw[window.edge_start : window.edge_stop + 1], np.float64)
    if not np.isfinite(selected_q).all() or np.any(np.diff(selected_q) <= 0):
        return _failure_cue(window, edge_count, "native_clock_invalid")
    if 2 * float(np.max(FREQUENCIES)) * float(np.max(np.diff(selected_q))) >= 1:
        return _failure_cue(window, edge_count, "clock_grid_unsupported")

    coordinate_mask = np.asarray(coordinate_mask_raw, dtype=np.bool_)
    base_mask = np.asarray(base_mask_raw, dtype=np.bool_)
    try:
        quadrature = shared_native_quadrature(
            q_raw,
            base_mask,
            edge_start=window.edge_start,
            edge_stop=window.edge_stop,
        )
    except QuadratureError as error:
        valid_edges = int(np.count_nonzero(base_mask[window.edge_start : window.edge_stop]))
        return _failure_cue(window, edge_count, str(error), valid_edges=valid_edges)

    indices = quadrature.edge_indices
    valid_edge_count = int(indices.size)
    span = quadrature.span
    if valid_edge_count < 16 or span < 16.0:
        return _failure_cue(
            window,
            edge_count,
            "cue requires 16 base-valid edges and midpoint span 16",
            valid_edges=valid_edge_count,
            span=span,
        )
    clocks_f64 = np.asarray(q_raw, dtype=np.float64)
    clock_delta = np.diff(clocks_f64)
    geometry_delta = np.diff(g, axis=0)
    valid_spectra: list[Float64Array] = []
    for coordinate in range(66):
        subset = indices[coordinate_mask[indices, coordinate]]
        if subset.size < 3:
            continue
        weight = quadrature.weights[subset]
        midpoint = quadrature.midpoints[subset]
        if neumaier_sum(weight) <= 0.0 or float(midpoint[-1] - midpoint[0]) <= 0.0:
            continue
        velocity = geometry_delta[subset, coordinate] / clock_delta[subset]
        spectrum = coordinate_spectrum_ordered(velocity, midpoint, weight)
        if spectrum is not None:
            valid_spectra.append(spectrum)

    valid_coordinate_count = len(valid_spectra)
    if valid_coordinate_count < 16:
        failed = _failure_cue(
            window,
            edge_count,
            "cue requires at least 16 valid coordinate spectra",
            valid_edges=valid_edge_count,
            span=span,
        )
        return WindowCue(
            window=failed.window,
            spectrum=failed.spectrum,
            distribution=failed.distribution,
            u=failed.u,
            gamma=failed.gamma,
            valid_coordinate_count=valid_coordinate_count,
            valid_edge_count=failed.valid_edge_count,
            midpoint_span=failed.midpoint_span,
            quadrature_weights=quadrature.weights.copy(),
            failure=failed.failure,
        )

    average_spectrum = neumaier_sum_array(np.stack(valid_spectra, axis=0)) / np.float64(
        valid_coordinate_count
    )
    stabilized = average_spectrum + np.float64(1e-8)
    normalizer = neumaier_sum(stabilized)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return _failure_cue(
            window,
            edge_count,
            "spectral probability denominator is invalid",
            valid_edges=valid_edge_count,
            span=span,
        )
    distribution = stabilized / np.float64(normalizer)
    u = neumaier_sum(distribution * LOG_FREQUENCIES)
    entropy_term = neumaier_sum(distribution * np.log(distribution + np.float64(1e-8)))
    entropy_confidence = float(
        np.clip(1.0 + entropy_term / np.log(np.float64(PERIOD_COUNT)), 0.0, 1.0)
    )
    gamma = float(
        min(1.0, valid_edge_count / 32.0)
        * min(1.0, span / 128.0)
        * (valid_coordinate_count / 66.0)
        * entropy_confidence
    )
    if not np.isfinite(u) or not np.isfinite(gamma):
        return _failure_cue(
            window,
            edge_count,
            "cue statistic is nonfinite",
            valid_edges=valid_edge_count,
            span=span,
        )
    if gamma < 0.15:
        gamma = 0.0
    return WindowCue(
        window=window,
        spectrum=np.ascontiguousarray(average_spectrum),
        distribution=np.ascontiguousarray(distribution),
        u=u,
        gamma=gamma,
        valid_coordinate_count=valid_coordinate_count,
        valid_edge_count=valid_edge_count,
        midpoint_span=span,
        quadrature_weights=quadrature.weights.copy(),
        failure=None,
    )

@dataclass(frozen=True)
class LocalRouting:
    windows: tuple
    local_gates: np.ndarray


def compute_local_routing(identity, progress=None):
    """Only the unchanged local-gate prefix of the reference routing path."""
    g = identity.geometry_features
    edge_count = len(identity.response_edge_support)
    bounds = validate_run_bounds(identity.run_bounds, edge_count)
    windows = window_schedule(bounds)
    cues = []
    for index, window in enumerate(windows):
        cues.append(compute_native_window_cue_ordered(g.geometry, g.edge_coordinate_mask,
            identity.clock.q, identity.response_edge_support, window))
        if progress is not None:
            progress(index + 1, len(windows))
    coverage = _coverage(windows, edge_count)
    references = []
    for run_index in range(bounds.shape[0]):
        indices = [i for i,w in enumerate(windows) if w.run_index == run_index]
        references.append(_run_reference(run_index, indices, cues, g.edge_lengths, coverage))
    local = np.empty((len(windows), 3), dtype=np.float64)
    for index, cue in enumerate(cues):
        local[index] = _gate_for_cue(cue, references[cue.window.run_index])
    local.setflags(write=False)
    return LocalRouting(windows, local)
