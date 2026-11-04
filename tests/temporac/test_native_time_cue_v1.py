"""Float clock NUDFT equivalence, support and conservative grid tests."""

from dataclasses import replace

import numpy as np
import pytest

from pams.temporac.cue import Window, compute_routing, compute_window_cue
from pams.temporac.native_bridge_v1 import preprocess_native
from pams.temporac.native_time_cue_v1 import compute_native_routing, compute_native_window_cue
from tests.temporac.test_native_bridge_v1 import feature


def test_old_complete_geometry_routing_equivalence():
    identity = preprocess_native(feature(300))
    g = identity.geometry_features
    old = compute_routing(
        g.geometry,
        g.edge_coordinate_mask,
        np.arange(300, dtype=np.int64),
        identity.response_edge_support,
        g.edge_lengths,
        identity.run_bounds,
    )
    new = compute_native_routing(identity)
    np.testing.assert_array_equal(new.local_gates, old.local_gates)
    for a, b in zip(new.cues, old.cues, strict=True):
        np.testing.assert_array_equal(a.spectrum, b.spectrum)
        assert a.gamma == b.gamma


def test_known_sine_float_clock_and_grid_guard():
    q = np.arange(300, dtype=np.float64)
    g = np.sin(2 * np.pi * q[:, None] / 19 + np.arange(66)[None, :] / 7)
    mask = np.ones((299, 66), bool)
    edges = np.ones(299, bool)
    w = Window(0, 0, 127)
    old = compute_window_cue(g, mask, q.astype(np.int64), edges, w)
    new = compute_native_window_cue(g, mask, q, edges, w)
    np.testing.assert_array_equal(old.spectrum, new.spectrum)
    assert new.gamma > 0
    bad = q.copy()
    bad[50:] += 1.5
    cue = compute_native_window_cue(g, mask, bad, edges, w)
    assert cue.gamma == 0 and cue.failure == "clock_grid_unsupported"
    with pytest.raises(ValueError, match="integer"):
        compute_window_cue(g, mask, q, edges, w)


def test_physical_rational_unit_conversion_and_identity_independence():
    f = feature(300, pts=np.arange(300, dtype=np.int64) * 1000)
    a = preprocess_native(f)
    b = preprocess_native(replace(f, pts=f.pts * 2, time_base=np.array([1, 2000], np.int64)))
    np.testing.assert_array_equal(a.clock.q, b.clock.q)
    ra = compute_native_routing(a)
    rb = compute_native_routing(b)
    np.testing.assert_array_equal(ra.local_gates, rb.local_gates)
    for x, y in zip(ra.cues, rb.cues, strict=True):
        np.testing.assert_array_equal(x.spectrum, y.spectrum)
    compute_native_routing(preprocess_native(feature(216, stationary=True)))
    np.testing.assert_array_equal(ra.local_gates, compute_native_routing(a).local_gates)


def test_static_uniform_and_gap_windows():
    r = preprocess_native(feature(216, stationary=True))
    routing = compute_native_routing(r)
    assert all(c.gamma == 0 for c in routing.cues)
    np.testing.assert_array_equal(routing.local_gates, np.full((len(routing.windows), 3), 1 / 3))
    f = feature(300)
    ok = f.frame_source_ok.copy()
    ok[149] = False
    r = preprocess_native(replace(f, frame_source_ok=ok))
    routing = compute_native_routing(r)
    assert all(r.response_edge_support[w.edge_start : w.edge_stop].all() for w in routing.windows)
    assert not routing.coverage[148:150].any()
