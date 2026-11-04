"""Additive CPU timebase audit; never alters frozen resampling/training paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "temporac.timebase-audit.v1"


def _clock(value: Any, name: str) -> np.ndarray:
    clock = np.asarray(value, dtype=np.float64)
    if clock.ndim != 1 or clock.size < 2 or not np.all(np.isfinite(clock)):
        raise ValueError(f"{name} must contain at least two finite clocks")
    if np.any(np.diff(clock) <= 0):
        raise ValueError(f"{name} must be strictly increasing (duplicates are ambiguous)")
    return clock


def _freeze(value: np.ndarray) -> np.ndarray:
    """Copy to bytes-backed storage, preserving the provenance after construction."""
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class TimebaseMap:
    """Use build_timebase to construct a validated map for one support channel.

    Indices address original source rows, including rows whose support is false.
    gap_span is the distance between bracketing observed anchors for interpolation,
    or the distance to the held endpoint for extrapolation; observed rows have zero.
    missing_source_rows counts rows strictly between the anchors (zero for a held
    endpoint), not the unknown number of missing physical frames in a clock gap.
    Values can exist even where supported is false; they must not be treated as GT.
    """

    clock_unit: str
    max_interpolation_gap: float
    source_clock: np.ndarray
    output_physical_clock: np.ndarray
    source_observed: np.ndarray
    left_source_index: np.ndarray
    right_source_index: np.ndarray
    left_weight: np.ndarray
    right_weight: np.ndarray
    support_kind: np.ndarray
    supported: np.ndarray
    gap_span: np.ndarray
    missing_source_rows: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
        return result

    def row_to_clock(self, row: Any) -> np.ndarray:
        """Convert continuous sample-center coordinates; reject out-of-range rows."""
        return _coordinate(row, np.arange(self.output_physical_clock.size),
                           self.output_physical_clock)

    def clock_to_row(self, clock: Any) -> np.ndarray:
        """Convert physical time to continuous output row; never silently clip."""
        return _coordinate(clock, self.output_physical_clock,
                           np.arange(self.output_physical_clock.size))

    def resample_values(self, values: Any) -> np.ndarray:
        """Apply recorded linear/endpoint-hold weights without modifying source.

        All observed values must be finite; missing source values are ignored.
        No geometry repair or perturbation is applied to a stationary trajectory.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 1 or values.shape[0] != self.source_clock.size:
            raise ValueError("values must have source rows on the first axis")
        if not np.all(np.isfinite(values[self.source_observed])):
            raise ValueError("observed source values must be finite")
        shape = (self.output_physical_clock.size,) + (1,) * (values.ndim - 1)
        left = values[self.left_source_index]
        return left + self.right_weight.reshape(shape) * (values[self.right_source_index] - left)


def _coordinate(value: Any, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < source[0]) or np.any(values > source[-1]):
        raise ValueError("coordinate is nonfinite or outside the output sample-center domain")
    return np.asarray(np.interp(values, source, target))


def build_timebase(
    source_clock: Any,
    output_physical_clock: Any,
    source_observed: Any,
    *,
    clock_unit: str,
    max_interpolation_gap: float,
) -> TimebaseMap:
    """Build explicit coordinate/support provenance without resampling the clocks.

    No gap threshold is inferred. Interpolated rows are supported only if their
    complete observed-anchor span <= max_interpolation_gap. Extrapolation uses
    endpoint holding for numeric output and is always unsupported. Exact observed
    times use exact float equality (no hidden snapping tolerance).
    """
    source = _clock(source_clock, "source_clock")
    output = _clock(output_physical_clock, "output_physical_clock")
    observed = np.asarray(source_observed)
    if observed.shape != source.shape or not np.all(np.isin(observed, (0, 1))):
        raise ValueError("source_observed must be an explicit binary mask matching source_clock")
    observed = observed.astype(bool)
    if not np.any(observed):
        raise ValueError("at least one observed source row is required")
    if not isinstance(clock_unit, str) or not clock_unit.strip():
        raise ValueError("clock_unit must be explicit and nonempty")
    threshold = float(max_interpolation_gap)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("max_interpolation_gap must be finite and nonnegative")

    anchors = np.flatnonzero(observed)
    anchor_clock = source[anchors]
    insertion = np.searchsorted(anchor_clock, output, side="left")
    right = anchors[np.clip(insertion, 0, anchors.size - 1)]
    left = anchors[np.clip(insertion - 1, 0, anchors.size - 1)]
    exact = output == source[right]
    left[exact] = right[exact]
    extrapolated = (output < anchor_clock[0]) | (output > anchor_clock[-1])
    interpolated = ~exact & ~extrapolated
    span = source[right] - source[left]
    weight = np.zeros(output.size, dtype=np.float64)
    weight[interpolated] = (output[interpolated] - source[left[interpolated]]) / span[interpolated]
    gap = span.copy()
    gap[extrapolated] = np.abs(output[extrapolated] - source[left[extrapolated]])
    kind = np.full(output.size, "interpolated", dtype="<U12")
    kind[exact] = "observed"
    kind[extrapolated] = "extrapolated"
    supported = exact | (interpolated & (span <= threshold))
    missing_rows = np.maximum(right - left - 1, 0)
    arrays = (source, output, observed, left, right, 1.0 - weight, weight,
              kind, supported, gap, missing_rows)
    return TimebaseMap(clock_unit, threshold, *(_freeze(value) for value in arrays))


def audit_legacy_npz(path: str | Path) -> dict[str, Any]:
    """Diagnose a legacy feature NPZ without reconstructing destroyed provenance.

    Hash receipts and source_length cannot recover source clocks or masks.
    An index-shaped clock is a warning, not proof that an archive came from v2.
    """
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        clock = archive["sampled_frame_indices"] if "sampled_frame_indices" in keys else None
        mask = archive["frame_mask"] if "frame_mask" in keys else None
        index_clock = bool(
            clock is not None and clock.ndim == 1
            and np.array_equal(clock, np.arange(clock.size))
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "input_format": "legacy_or_unversioned_feature_npz",
            "members": keys,
            "missing_provenance_fields": sorted({
                "source_clock", "output_physical_clock", "source_observed",
                "left_source_index", "right_source_index", "left_weight", "right_weight",
                "support_kind", "supported", "gap_span", "clock_unit", "schema_version",
            } - set(keys)),
            "index_shaped_clock": index_clock,
            "all_frames_marked_observed": bool(mask is not None and np.all(mask == 1)),
            "coordinate_provenance_verified": False,
            "recoverable_from_this_audit": False,
            "status": "requires_original_source_and_versioned_timebase",
            "reason": "Legacy feature fields cannot certify physical clocks or original support.",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="legacy feature .npz (read only)")
    args = parser.parse_args(argv)
    try:
        report = audit_legacy_npz(args.archive)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"cannot audit archive: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 2  # Missing coordinate provenance is an audit finding, not a PASS.


if __name__ == "__main__":
    raise SystemExit(main())
