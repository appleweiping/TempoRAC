"""Additive TempoRAC inference ablations; the frozen model graph is untouched."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.temporac.decode import DecodeResult, decode_identity
from pams.temporac.nola import nola_reconstruct, probability_fusion

Float64Array = NDArray[np.float64]


class AblationError(ValueError):
    """Raised when an ablation input would make variants incomparable."""


@dataclass(frozen=True, slots=True)
class AblationOutput:
    name: str
    count: int
    response: Float64Array
    decode: DecodeResult

    def __post_init__(self) -> None:
        copied = np.ascontiguousarray(self.response, dtype=np.float64)
        copied.setflags(write=False)
        object.__setattr__(self, "response", copied)


def _validate_inputs(
    branch_probabilities: ArrayLike,
    local_gates: ArrayLike,
    windows: ArrayLike,
    edge_mask: ArrayLike,
    run_bounds: ArrayLike,
) -> tuple[
    Float64Array,
    Float64Array,
    NDArray[np.int32],
    NDArray[np.uint8],
    NDArray[np.int32],
]:
    probabilities = np.asarray(branch_probabilities, dtype=np.float64)
    gates = np.asarray(local_gates, dtype=np.float64)
    window_table = np.asarray(windows)
    mask = np.asarray(edge_mask)
    bounds = np.asarray(run_bounds)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise AblationError("branch_probabilities must have shape [E,3]")
    if gates.ndim != 2 or gates.shape != (len(window_table), 3):
        raise AblationError("local_gates must have shape [W,3]")
    if (
        window_table.ndim != 2
        or window_table.shape[1:] != (2,)
        or window_table.dtype.kind not in "iu"
    ):
        raise AblationError("windows must be an integer [W,2] table")
    if mask.shape != (len(probabilities),) or mask.dtype.kind not in "bu":
        raise AblationError("edge_mask must be binary shape [E]")
    if bounds.ndim != 2 or bounds.shape[1:] != (2,) or bounds.dtype.kind not in "iu":
        raise AblationError("run_bounds must be an integer [R,2] table")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise AblationError("branch probabilities must be finite values in [0,1]")
    if not np.all(np.isfinite(gates)) or np.any(gates < 0.0):
        raise AblationError("gates must be finite and nonnegative")
    if not np.allclose(np.sum(gates, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise AblationError("each gate must sum to one")
    if not np.all((mask == 0) | (mask == 1)):
        raise AblationError("edge_mask must be binary")
    return (
        np.ascontiguousarray(probabilities),
        np.ascontiguousarray(gates),
        np.ascontiguousarray(window_table, dtype=np.int32),
        np.ascontiguousarray(mask, dtype=np.uint8),
        np.ascontiguousarray(bounds, dtype=np.int32),
    )


def boxcar_overlap_add(
    branch_probabilities: ArrayLike,
    gates: ArrayLike,
    windows: ArrayLike,
    edge_mask: ArrayLike,
) -> Float64Array:
    """Normalized overlap-add with unit weights, isolating the NOLA taper."""

    probabilities = np.asarray(branch_probabilities, dtype=np.float64)
    gate_array = np.asarray(gates, dtype=np.float64)
    window_table = np.asarray(windows, dtype=np.int64)
    mask = np.asarray(edge_mask, dtype=np.bool_)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise AblationError("branch_probabilities must have shape [E,3]")
    if gate_array.shape != (len(window_table), 3):
        raise AblationError("gates must have shape [W,3]")
    if mask.shape != (len(probabilities),):
        raise AblationError("edge_mask must have shape [E]")
    numerator = np.zeros(len(probabilities), dtype=np.float64)
    denominator = np.zeros(len(probabilities), dtype=np.float64)
    previous_start = -1
    for index, pair in enumerate(window_table.tolist()):
        start, stop = int(pair[0]), int(pair[1])
        if not (0 <= start < stop <= len(probabilities)) or start < previous_start:
            raise AblationError("windows must be ordered nonempty half-open intervals")
        previous_start = start
        fused = probability_fusion(probabilities[start:stop], gate_array[index])
        valid = mask[start:stop].astype(np.float64)
        numerator[start:stop] += valid * fused
        denominator[start:stop] += valid
    if np.any(denominator[mask] <= 0.0):
        raise AblationError("every valid edge must be covered by at least one window")
    response = np.zeros(len(probabilities), dtype=np.float64)
    response[mask] = numerator[mask] / denominator[mask]
    if not np.all(np.isfinite(response)) or np.any((response < 0.0) | (response > 1.0)):
        raise AblationError("boxcar overlap-add produced a non-probability")
    result = np.ascontiguousarray(response)
    result.setflags(write=False)
    return result


def _hard_gates(gates: Float64Array) -> Float64Array:
    hard = np.zeros_like(gates)
    hard[np.arange(len(gates)), np.argmax(gates, axis=1)] = 1.0
    return hard


def _constant_gates(rows: int, expert_index: int | None) -> Float64Array:
    if expert_index is None:
        return np.full((rows, 3), 1.0 / 3.0, dtype=np.float64)
    result = np.zeros((rows, 3), dtype=np.float64)
    result[:, expert_index] = 1.0
    return result


def run_ablation_suite(
    branch_probabilities: ArrayLike,
    local_gates: ArrayLike,
    windows: ArrayLike,
    edge_mask: ArrayLike,
    run_bounds: ArrayLike,
    *,
    global_gates: ArrayLike | None = None,
) -> Mapping[str, AblationOutput]:
    """Run same-checkpoint causal ablations and decode each response once."""

    probabilities, local, window_table, mask, bounds = _validate_inputs(
        branch_probabilities, local_gates, windows, edge_mask, run_bounds
    )
    gate_sets: dict[str, Float64Array] = {
        "full_local_soft_nola": local,
        "without_local_tempo_uniform": _constant_gates(len(local), None),
        "without_soft_routing_hard": _hard_gates(local),
        # TempoRACResponse.branch_order is (slow, medium, fast).
        "single_medium_expert": _constant_gates(len(local), 1),
    }
    if global_gates is not None:
        global_array = np.asarray(global_gates, dtype=np.float64)
        if global_array.shape != local.shape:
            raise AblationError("global_gates must have the same shape as local_gates")
        gate_sets["global_tempo_soft_nola"] = global_array

    responses: dict[str, Float64Array] = {
        name: nola_reconstruct(probabilities, gates, window_table, mask).response
        for name, gates in gate_sets.items()
    }
    responses["without_nola_taper_boxcar"] = boxcar_overlap_add(
        probabilities, local, window_table, mask
    )
    outputs: dict[str, AblationOutput] = {}
    for name, response in responses.items():
        decoded = decode_identity(response, mask, bounds)
        outputs[name] = AblationOutput(name, decoded.count, response, decoded)
    return outputs


__all__ = [
    "AblationError",
    "AblationOutput",
    "boxcar_overlap_add",
    "run_ablation_suite",
]
