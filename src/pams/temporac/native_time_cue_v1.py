"""NB1 float64-clock wrapper; frozen old integer APIs remain untouched."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.temporac.cue import (
    FREQUENCIES,
    LOG_FREQUENCIES,
    PERIOD_COUNT,
    UNIFORM_GATE,
    CueError,
    RoutingResult,
    RunReference,
    Window,
    WindowCue,
    _coordinate_spectrum,
    _coverage,
    _failure_cue,
    _gate_for_cue,
    _run_reference,
    window_schedule,
)
from pams.temporac.preprocess import validate_run_bounds
from pams.temporac.quadrature import (
    EdgeQuadrature,
    QuadratureError,
    neumaier_sum,
    neumaier_sum_array,
)

Float64Array = NDArray[np.float64]


def shared_native_quadrature(clocks, edge_mask, *, edge_start=0, edge_stop=None):
    q = np.asarray(clocks, np.float64)
    e = q.size - 1
    mask = np.asarray(edge_mask, bool)
    stop = e if edge_stop is None else edge_stop
    if q.ndim != 1 or mask.shape != (e,) or not 0 <= edge_start < stop <= e:
        raise QuadratureError("invalid native quadrature shape/interval")
    indices = np.flatnonzero(mask & (np.arange(e) >= edge_start) & (np.arange(e) < stop))
    if indices.size < 3:
        raise QuadratureError("shared quadrature requires at least three midpoints")
    if (
        np.any(~np.isfinite(q[indices]))
        or np.any(~np.isfinite(q[indices + 1]))
        or np.any(q[indices + 1] <= q[indices])
    ):
        raise QuadratureError("native selected edge clocks must increase")
    midpoint = (q[:-1] + q[1:]) / 2
    x = midpoint[indices]
    spacing = np.diff(x)
    if not np.all(spacing > 0):
        raise QuadratureError("native midpoint sequence must increase")
    span = float(x[-1] - x[0])
    trapezoid = np.empty(indices.size, np.float64)
    trapezoid[0], trapezoid[-1] = spacing[0] / 2, spacing[-1] / 2
    trapezoid[1:-1] = (x[2:] - x[:-2]) / 2
    taper = np.sin(np.pi * (x - x[0]) / span) ** 2
    taper[0] = taper[-1] = 0
    weights = np.zeros(e, np.float64)
    weights[indices] = trapezoid * taper
    if not np.isfinite(weights).all() or neumaier_sum(weights) <= 0:
        raise QuadratureError("invalid native quadrature denominator")
    return EdgeQuadrature(midpoint, weights, indices, span)


def compute_native_window_cue(
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
        spectrum = _coordinate_spectrum(velocity, midpoint, weight)
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


def _compute_native_routing(
    geometry: ArrayLike,
    coordinate_edge_mask: ArrayLike,
    clocks: ArrayLike,
    edge_mask: ArrayLike,
    edge_lengths: ArrayLike,
    run_bounds: ArrayLike,
) -> RoutingResult:
    """Compute local/global/uniform/blocked/shuffled gates with per-run reset."""

    g = np.asarray(geometry, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 66 or g.shape[0] < 2:
        raise CueError("geometry must have shape [T,66]")
    edge_count = g.shape[0] - 1
    bounds = validate_run_bounds(run_bounds, edge_count)
    lengths = np.asarray(edge_lengths, dtype=np.float64)
    if lengths.shape != (edge_count,) or not np.all(np.isfinite(lengths)):
        raise CueError("edge_lengths must be finite shape [E]")
    windows = window_schedule(bounds)
    cues = tuple(
        compute_native_window_cue(
            geometry,
            coordinate_edge_mask,
            clocks,
            edge_mask,
            window,
        )
        for window in windows
    )
    coverage = _coverage(windows, edge_count)
    references_list: list[RunReference] = []
    indices_by_run: list[list[int]] = []
    for run_index in range(bounds.shape[0]):
        indices = [index for index, window in enumerate(windows) if window.run_index == run_index]
        indices_by_run.append(indices)
        references_list.append(_run_reference(run_index, indices, cues, lengths, coverage))
    references = tuple(references_list)

    local = np.empty((len(windows), 3), dtype=np.float64)
    for index, cue in enumerate(cues):
        local[index] = _gate_for_cue(cue, references[cue.window.run_index])

    global_gates = np.empty_like(local)
    for run_index, (start, stop) in enumerate(bounds.tolist()):
        full_window = Window(run_index, start, stop)
        # The full-run cue is an intervention and not appended to the frozen
        # local window/reference receipt.
        full_cue = compute_native_window_cue(
            geometry,
            coordinate_edge_mask,
            clocks,
            edge_mask,
            full_window,
        )
        gate = _gate_for_cue(full_cue, references[run_index])
        global_gates[indices_by_run[run_index]] = gate

    uniform = np.tile(UNIFORM_GATE, (len(windows), 1))
    # Blocking the cue is definitionally the exact uniform byte path.
    blocked = uniform.copy()
    shuffled = local.copy()
    for indices in indices_by_run:
        reliable = [index for index in indices if cues[index].reliable]
        if reliable:
            shifted = np.roll(local[reliable], shift=1, axis=0)
            shuffled[reliable] = shifted
    if not np.array_equal(blocked, uniform):
        raise AssertionError("blocked and uniform routing must be bit-identical")
    return RoutingResult(
        windows=windows,
        cues=cues,
        references=references,
        local_gates=np.ascontiguousarray(local),
        global_gates=np.ascontiguousarray(global_gates),
        uniform_gates=np.ascontiguousarray(uniform),
        blocked_gates=np.ascontiguousarray(blocked),
        shuffled_gates=np.ascontiguousarray(shuffled),
        coverage=np.ascontiguousarray(coverage),
    )


def compute_native_routing(identity) -> RoutingResult:
    g = identity.geometry_features
    if identity.response_edge_support.size == 0:
        empty = np.empty((0, 3), np.float64)
        return RoutingResult(
            (),
            (),
            (),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            np.empty(0, np.int32),
        )
    return _compute_native_routing(
        g.geometry,
        g.edge_coordinate_mask,
        identity.clock.q,
        identity.response_edge_support,
        g.edge_lengths,
        identity.run_bounds,
    )
