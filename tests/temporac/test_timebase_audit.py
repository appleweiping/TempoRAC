from __future__ import annotations

import json

import numpy as np
import pytest

from pams.temporac.timebase_audit import (
    SCHEMA_VERSION,
    audit_legacy_npz,
    build_timebase,
    main,
)


def _map(source, output, mask, gap=3.0):
    return build_timebase(source, output, mask, clock_unit="seconds", max_interpolation_gap=gap)


def test_irregular_clock_retains_original_indices_weights_and_support():
    source = np.array([10.0, 11.0, 12.0, 20.0, 21.0])
    mapping = _map(source, [9.0, 10.0, 11.0, 12.0, 16.0, 20.0, 22.0], [1, 1, 0, 1, 1])
    np.testing.assert_array_equal(mapping.source_clock, source)
    np.testing.assert_array_equal(mapping.left_source_index, [0, 0, 1, 1, 1, 3, 4])
    np.testing.assert_array_equal(mapping.right_source_index, [0, 0, 1, 3, 3, 3, 4])
    np.testing.assert_allclose(mapping.right_weight, [0, 0, 0, 1 / 9, 5 / 9, 0, 0])
    np.testing.assert_array_equal(mapping.support_kind,
                                  ["extrapolated", "observed", "observed", "interpolated",
                                   "interpolated", "observed", "extrapolated"])
    np.testing.assert_array_equal(mapping.supported, [0, 1, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(mapping.gap_span, [1, 0, 0, 9, 9, 0, 1])
    np.testing.assert_array_equal(mapping.missing_source_rows, [0, 0, 0, 1, 1, 0, 0])
    values = source.copy()
    values[2] = np.nan  # The unavailable source row cannot leak into interpolation.
    np.testing.assert_allclose(mapping.resample_values(values), [10, 10, 11, 12, 16, 20, 21])
    source[:] = -1
    assert mapping.source_clock[0] == 10
    with pytest.raises(ValueError):
        mapping.source_clock.setflags(write=True)


@pytest.mark.parametrize("source,output", [
    ([0, 1, 1], [0, 1]), ([0, 2, 1], [0, 1]), ([0, np.nan], [0, 1]),
    ([0, 1], [0, 0]), ([0, 1], [1, 0]), ([0, 1], [0, np.inf]),
])
def test_duplicate_nonmonotonic_and_nonfinite_clocks_rejected(source, output):
    with pytest.raises(ValueError):
        _map(source, output, np.ones(len(source)))


def test_support_threshold_is_explicit_and_interpolation_is_never_observation():
    low = _map([0, 2, 8], [0, 1, 2, 5, 8], [1, 1, 1], gap=2)
    np.testing.assert_array_equal(low.supported, [1, 1, 1, 0, 1])
    assert low.support_kind[1] == "interpolated"
    high = _map([0, 2, 8], [0, 1, 2, 5, 8], [1, 1, 1], gap=6)
    assert high.supported.all()
    assert np.count_nonzero(high.support_kind == "observed") == 3
    for gap in [-1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            _map([0, 1], [0, 1], [1, 1], gap=gap)


def test_all_stationary_pose_is_exactly_stationary_without_repair():
    mapping = _map([0, 1, 8, 10], [-1, 0, 2, 4, 8, 11], [1, 0, 1, 1])
    pose = np.full((4, 17, 3), 0.7)
    before = pose.copy()
    result = mapping.resample_values(pose)
    np.testing.assert_array_equal(result, np.full((6, 17, 3), 0.7))
    np.testing.assert_array_equal(pose, before)
    np.testing.assert_array_equal(np.diff(result, axis=0), 0)


def test_boundaries_and_half_open_intervals_roundtrip_on_irregular_grid():
    mapping = _map([5, 6, 14], [5, 7, 12, 14], [1, 1, 1])
    rows = np.array([0, 0.5, 1, 1.75, 2, 3])
    np.testing.assert_allclose(mapping.clock_to_row(mapping.row_to_clock(rows)), rows)
    # Shared endpoint remains shared, without +/-1 rounding or inclusivity changes.
    intervals = [np.array([5, 8]), np.array([8, 14])]
    for interval in intervals:
        np.testing.assert_allclose(mapping.row_to_clock(mapping.clock_to_row(interval)), interval)
    for coordinate in [-0.01, 3.01, np.nan]:
        with pytest.raises(ValueError):
            mapping.row_to_clock(coordinate)
    for coordinate in [4.99, 14.01, np.inf]:
        with pytest.raises(ValueError):
            mapping.clock_to_row(coordinate)


def test_missing_and_single_anchor_support_are_not_fabricated():
    with pytest.raises(ValueError, match="observed"):
        _map([0, 1, 2], [0, 1, 2], [0, 0, 0])
    mapping = _map([0, 1, 2], [0, 1, 2], [0, 1, 0])
    np.testing.assert_array_equal(mapping.supported, [0, 1, 0])
    np.testing.assert_array_equal(mapping.resample_values([np.nan, 7, np.nan]), [7, 7, 7])
    with pytest.raises(ValueError, match="finite"):
        mapping.resample_values([0, np.nan, 0])
    with pytest.raises(ValueError, match="binary"):
        _map([0, 1], [0, 1], [1, 2])


def test_json_retains_both_clocks_and_support_provenance():
    payload = _map([10, 20], [10, 15, 20], [1, 1]).as_dict()
    payload = json.loads(json.dumps(payload, allow_nan=False))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["clock_unit"] == "seconds"
    assert payload["source_clock"] == [10, 20]
    assert payload["output_physical_clock"] == [10, 15, 20]
    assert payload["source_observed"] == [True, True]
    assert payload["support_kind"] == ["observed", "interpolated", "observed"]
    assert payload["supported"] == [True, False, True]


def test_legacy_cli_reports_irrecoverable_coordinates_without_modifying_archive(tmp_path, capsys):
    path = tmp_path / "legacy-v2.npz"
    np.savez(path, sampled_frame_indices=np.arange(4), frame_mask=np.ones(4), source_length=[640])
    before = path.read_bytes()
    report = audit_legacy_npz(path)
    assert report["index_shaped_clock"]
    assert not report["recoverable_from_this_audit"]
    assert not report["coordinate_provenance_verified"]
    assert main([str(path)]) == 2
    assert json.loads(capsys.readouterr().out) == report
    assert path.read_bytes() == before
