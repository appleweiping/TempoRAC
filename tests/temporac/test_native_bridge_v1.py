"""Synthetic CPU NB1 engineering tests; no trained models or benchmark labels."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pams.temporac.native_bridge_v1 as nb
from pams.temporac.cue import window_schedule
from pams.temporac.preprocess import _geometry_masks, preprocess_identity


def feature(n=216, *, pts=None, present=None, stationary=False):
    base = np.stack((np.linspace(-0.8, 0.8, 17), np.linspace(0.2, 1, 17)), axis=1)
    base[11], base[12], base[5], base[6] = (-0.2, 0), (0.2, 0), (-0.3, 0.8), (0.3, 0.8)
    x = np.zeros((n, 17, 3), np.float32)
    x[:, :, :2] = base
    t = np.arange(n)
    if not stationary:
        x[:, 7:11, 0] += (0.15 * np.sin(2 * np.pi * t / 19))[:, None].astype(np.float32)
        x[:, 13:17, 1] += (0.12 * np.cos(2 * np.pi * t / 19))[:, None].astype(np.float32)
    x[:, :, 2] = 0.9
    return nb.NativeFeatureV1(
        np.arange(32, dtype=np.uint8),
        np.array([0], np.int64),
        np.array([n], np.int64),
        np.arange(n, dtype=np.int64),
        x,
        np.ones((n, 17), bool),
        np.ones(n, bool),
        np.zeros(n, np.int64) if pts is None else np.array(pts, np.int64),
        np.zeros(n, bool)
        if present is None and pts is None
        else np.ones(n, bool)
        if present is None
        else np.array(present, bool),
        np.array([1, 1000], np.int64),
    )


def script_module():
    spec = importlib.util.spec_from_file_location(
        "native_learner_builder",
        Path(__file__).resolve().parents[2] / "scripts/build_temporac_native_learner_bridge.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("n", [216, 900])
def test_full_native_and_legacy_equivalence(n):
    f = feature(n)
    r = nb.preprocess_native(f)
    old = preprocess_identity(f.xyconf_raw, np.ones(n, np.uint8), np.arange(n, dtype=np.int64))
    assert r.response_input.shape == (n, 269)
    assert r.run_bounds.tolist() == [[0, n - 1]]
    np.testing.assert_array_equal(r.response_input, old.response_input)
    np.testing.assert_array_equal(r.edge_lengths, old.edge_lengths)
    assert np.array_equal(r.feature.source_frame_index, np.arange(n))
    assert r.clock.mode == "source_frame_only"
    with pytest.raises(ValueError):
        r.feature.xyconf_raw.setflags(write=True)


def test_missing_216_two_frames_no_crossing_lag_windows_nola(monkeypatch):
    f = feature()
    ok = f.frame_source_ok.copy()
    ok[100:102] = False
    r = nb.preprocess_native(replace(f, frame_source_ok=ok))
    assert r.response_input.shape == (216, 269)
    assert r.run_bounds.tolist() == [[0, 99], [102, 215]]
    assert not r.response_edge_support[99:102].any()
    assert not r.response_input[102:106, 218:235].any()
    windows = window_schedule(r.run_bounds)
    for w in windows:
        assert r.response_edge_support[w.edge_start : w.edge_stop].all()
    reconstructed = nb.reconstruct_native_response(
        np.full((215, 3), 0.7), np.full((len(windows), 3), 1 / 3), windows, r.response_edge_support
    )
    assert (reconstructed.denominator[r.response_edge_support] >= 1e-3).all()
    assert not reconstructed.numerator[~r.response_edge_support].any()
    assert not reconstructed.denominator[~r.response_edge_support].any()
    called = []
    original = nb.connected_components

    def counted(*a):
        called.append(1)
        return original(*a)

    monkeypatch.setattr(nb, "connected_components", counted)
    p = nb.decode_native_identity(reconstructed.response, r)
    assert len(called) == 1 and p.count == 2 and p.status == "partial"
    assert p.component_source_ranges.tolist() == [[0.0, 99.0], [102.0, 215.0]]
    assert p.period_ap_status.startswith("unavailable")


@pytest.mark.parametrize("n,eligible", [(16, False), (17, True)])
def test_short_run_boundary(n, eligible):
    r = nb.preprocess_native(feature(n))
    assert r.response_edge_support.any() == eligible
    assert bool(r.short_runs.size) == (not eligible)


def test_stationary_zero_motion_and_failed_decode_once(monkeypatch):
    f = feature(stationary=True)
    r = nb.preprocess_native(f)
    assert r.response_edge_support.all() and not r.edge_lengths.any()
    assert not r.geometry_features.direction_element_mask.any()
    failed = nb.preprocess_native(replace(f, frame_source_ok=np.zeros(216, bool)))
    called = []
    original = nb.connected_components

    def counted(*a):
        called.append(a[0].size)
        return original(*a)

    monkeypatch.setattr(nb, "connected_components", counted)
    p = nb.decode_native_identity(np.zeros(215), failed)
    assert called == [0] and p.count == -1 and p.status == "abstain"
    assert p.response.shape == (215,) and not p.support.any()


def test_confidence_observation_and_hip_thresholds():
    f = feature(20)
    x = f.xyconf_raw.copy()
    x[:, 0, 2] = 0.19
    r = nb.preprocess_native(replace(f, xyconf_raw=x))
    assert f.joint_observed[:, 0].all() and not r.joint_mask[:, 0].any()
    x = f.xyconf_raw.copy()
    x[:, :, 2] = 0.19
    x[:, :7, 2] = 0.9
    x[:, 11, 2] = 0.9
    assert nb.preprocess_native(replace(f, xyconf_raw=x)).frame_valid.all()
    x[:, 11, 2] = 0.19
    x[:, 7, 2] = 0.9
    assert not nb.preprocess_native(replace(f, xyconf_raw=x)).frame_valid.any()
    x = f.xyconf_raw.copy()
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(f, xyconf_raw=x)
    obs = f.joint_observed.copy()
    obs[:, 0] = False
    x = f.xyconf_raw.copy()
    x[:, 0] = 0
    r = nb.preprocess_native(replace(f, xyconf_raw=x, joint_observed=obs))
    assert (r.mapping["support_kind"][:, 0] == "unsupported").all()
    assert not r.mapping["supported"][:, 0].any()


def test_geometry_coordinate_gate_uses_actual_32_threshold():
    # Shared geometry counts occur in even dimensions; explicitly exercise below/at32.
    candidates = []
    for bits in range(1 << 17):
        if bits.bit_count() < 8 or not bits & (1 << 11 | 1 << 12):
            continue
        mask = np.array([bool(bits & (1 << j)) for j in range(17)])
        count = int(_geometry_masks(mask[None])[1].sum())
        if count in (30, 32) and count not in [v[0] for v in candidates]:
            candidates.append((count, mask))
        if len(candidates) == 2:
            break
    for count, mask in candidates:
        f = feature(20)
        obs = np.tile(mask, (20, 1))
        x = f.xyconf_raw.copy()
        x[~obs] = 0
        r = nb.preprocess_native(replace(f, xyconf_raw=x, joint_observed=obs))
        assert r.observed_edge_support.all() == (count >= 32)
    assert sorted(x[0] for x in candidates) == [30, 32]
    # Odd31 cannot be generated by pairs of geometry coordinates.


def test_pts_large_integer_missing_nonpositive_and_gap():
    ticks = 2**60 + np.arange(216, dtype=np.int64) * 1000
    r = nb.preprocess_native(feature(pts=ticks))
    np.testing.assert_array_equal(r.clock.q, np.arange(216))
    assert r.clock.tau == 1.0
    assert nb.source_boundary_seconds(r, 0) == 0.0
    assert nb.source_boundary_seconds(r, 215) == 215.0
    assert nb.source_boundary_seconds(r, 216) is None
    assert (
        nb.source_boundary_seconds(r, 216, {"terminal_verified": True, "terminal_seconds": 216.0})
        == 216.0
    )
    assert nb.sample_position_seconds(r, 1.5, segment=0) == 1.5
    for delta, valid in [(0, False), (-1, False), (4999, True), (5000, False)]:
        pts = np.arange(216, dtype=np.int64) * 1000
        pts[101:] += delta - 1000
        rr = nb.preprocess_native(feature(pts=pts))
        assert rr.clock.edge_time_support[100] == valid
    present = np.ones(216, bool)
    present[100] = False
    rr = nb.preprocess_native(feature(pts=ticks, present=present))
    assert not rr.clock.edge_time_support[99:101].any()
    assert rr.clock.mode == "pts"
    with pytest.raises(ValueError, match="outside"):
        nb.sample_position_seconds(rr, 100.0, segment=0)
    no = nb.preprocess_native(feature(pts=np.zeros(216, np.int64)))
    assert "clock_invalid" in no.reasons and no.status == "unusable"


def test_mapping_candidate_never_enters_model_and_features_strict(tmp_path):
    f = feature(30)
    obs = f.joint_observed.copy()
    obs[10, 0] = False
    x = f.xyconf_raw.copy()
    x[10, 0] = 0
    f = replace(f, xyconf_raw=x, joint_observed=obs)
    r = nb.preprocess_native(f)
    assert r.mapping["support_kind"][10, 0] == "interpolated"
    assert not r.mapping["supported"][10, 0]
    assert r.response_input[10, 0] == 0 and r.response_input[10, 185] == 0
    p = tmp_path / "feature.npz"
    np.savez(p, **f.as_arrays())
    loaded = nb.load_native_feature(p)
    np.testing.assert_array_equal(loaded.xyconf_raw, x)
    np.savez(p, **f.as_arrays(), count=np.array([1]))
    with pytest.raises(ValueError, match="member"):
        nb.load_native_feature(p)


def test_target_mask_certified_intervals_and_weights():
    r = nb.preprocess_native(feature(30))
    valid = np.zeros(29, bool)
    valid[4:20] = True
    chi = np.zeros(29)
    chi[4:12] = 1
    pulse = np.zeros(29)
    pulse[4:8] = 1
    target = nb.NativeTargetV1(
        valid, pulse, chi, np.array([[4, 20]], np.int64), "stable-pair hybrid", "fixture", "a" * 64
    )
    s = nb.native_supervision(r, target)
    np.testing.assert_array_equal(s["supervised"], valid)
    assert s["traversal_eligible"] == (True,)
    # Sole positive pulse lies on a response-unsupported edge: reject traversal.
    ok = r.feature.frame_source_ok.copy()
    ok[5] = False
    gap = nb.preprocess_native(replace(r.feature, frame_source_ok=ok))
    only_positive = np.zeros(29)
    only_positive[4] = 1
    assert nb.native_supervision(gap, replace(target, pulse=only_positive))[
        "traversal_eligible"
    ] == (False,)
    t = replace(target, chi=np.zeros(29))
    assert nb.native_supervision(r, t)["traversal_eligible"] == (False,)
    with pytest.raises(ValueError, match="uncertified"):
        replace(target, traversal_bounds=np.array([[0, 20]], np.int64))


@pytest.mark.parametrize("corrupt_member", ["feature", "derived"])
@pytest.mark.parametrize("corruption", [b"PK\x03\x04broken", b"appended corruption"])
def test_cli_real_artifacts_resume_validation_census(tmp_path, corrupt_member, corruption):
    m = script_module()
    raw = m.raw_module()
    row = {
        "schema": raw.PROJECTION,
        "split": "train",
        "media_key": "a" * 64,
        "media_path": "train/x/x.mp4",
        "media_sha256": "b" * 64,
        "width": 8,
        "height": 6,
        "length": 30,
        "persons": [{"slot": 0, "bbox_xyxy": [[0, 0, 8, 6]] * 30, "bbox_present": [True] * 30}],
    }
    arrays = raw.empty_raw(row)
    arrays["source_frame_index"] = np.arange(30, dtype=np.int64)
    arrays["decoded_ok"] = np.ones(30, bool)
    arrays["pts"] = np.zeros(30, np.int64)
    arrays["pts_present"] = np.zeros(30, bool)
    arrays["best_effort_timestamp"] = np.zeros(30, np.int64)
    arrays["best_effort_present"] = np.zeros(30, bool)
    arrays["xyconf_raw"][0] = feature(30).xyconf_raw
    arrays["joint_observed"][:] = True
    arrays["pose_returned"][:] = True
    arrays["bbox_valid"][:] = True
    meta = {
        "schema": raw.SCHEMA,
        "annotation_length": 30,
        "decoded_length": 30,
        "status": "complete",
        "slots": [{"slot": 0, "status": "complete"}],
        "media_key": "a" * 64,
        "media_sha256": "b" * 64,
        "clocks": {},
    }
    raw.save_raw(tmp_path / "input/raw/train" / ("a" * 64), arrays, meta)
    second_row = dict(row, media_key="c" * 64, media_path="train/y/y.mp4")
    second_meta = dict(meta, media_key="c" * 64)
    raw.save_raw(tmp_path / "input/raw/train" / ("c" * 64), arrays, second_meta)
    census = tmp_path / "census.jsonl"
    census.write_text(json.dumps(row) + "\n" + json.dumps(second_row) + "\n")
    cfg = {
        "raw_root": str(tmp_path / "input"),
        "census": str(census),
        "output": str(tmp_path / "output"),
    }
    expected = {"train": {"videos": 2, "persons": 2}, "val": {"videos": 0, "persons": 0}}
    assert m.execute(cfg, "plan", expected=expected) == 0
    assert m.execute(cfg, "build", expected=expected) == 0
    assert m.execute(cfg, "validate", expected=expected) == 0
    assert m.execute(cfg, "build", expected=expected) == 0
    ledger = json.loads((tmp_path / "output/validation.json").read_text())
    assert ledger["identities"] == 2 and ledger["records"][0]["source_length"] == 30
    with pytest.raises(ValueError, match="census"):
        m.read_census(census)
    feature_path = tmp_path / "output/identities/train" / (("a" * 64) + "_0.npz")
    damaged = (
        feature_path if corrupt_member == "feature" else feature_path.with_suffix(".derived.npz")
    )
    damaged.write_bytes(
        corruption if corruption.startswith(b"PK") else damaged.read_bytes() + corruption
    )
    for command in ("validate", "build"):
        assert m.execute(cfg, command, expected=expected) == 2
        report = "validation.json" if command == "validate" else "population_ledger.json"
        ledger = json.loads((tmp_path / "output" / report).read_text())
        assert ledger["identities"] == 2 and len(ledger["records"]) == 2
        assert ledger["records"][0]["status"] == "artifact_failure"
        assert "learner artifact hash mismatch" in ledger["records"][0]["error"]
        assert ledger["records"][1]["status"] == "complete"


@pytest.mark.parametrize("n", [216, 900])
def test_old_feature_record_still_rejects_native_n(n):
    from pams.temporac.types import ContractError, FeatureRecord

    f = feature(n)
    with pytest.raises(ContractError, match="shape"):
        FeatureRecord(
            np.ones(n, np.uint8),
            f.slot,
            f.xyconf_raw,
            f.opaque_key,
            np.ones(1, np.uint8),
            f.source_frame_index,
            f.source_length,
        )


def test_float32_threshold_bit_boundary_preserves_old_semantics():
    f = feature(30)
    for score in [
        np.nextafter(np.float32(0.2), np.float32(0)),
        np.float32(0.2),
        np.nextafter(np.float32(0.2), np.float32(1)),
    ]:
        x = f.xyconf_raw.copy()
        x[:, 0, 2] = score
        native = nb.preprocess_native(replace(f, xyconf_raw=x))
        legacy = preprocess_identity(x, np.ones(30, np.uint8), np.arange(30, dtype=np.int64))
        expected = bool(np.float64(score) > np.float64(0.20))
        assert native.joint_mask[:, 0].all() == expected
        np.testing.assert_array_equal(native.joint_mask, legacy.joint_mask)
    assert np.float64(np.float32(0.2)) > np.float64(0.20)


def test_single_row_retained_without_executing_model():
    from pams.temporac.native_time_cue_v1 import compute_native_routing

    r = nb.preprocess_native(feature(1))
    assert r.response_input.shape == (1, 269) and r.status == "unusable"
    assert not compute_native_routing(r).windows
    p = nb.decode_native_identity(np.empty(0), r)
    assert p.count == -1 and p.status == "abstain" and p.decoder_invocations == 1


def test_failed_raw_and_unresolved_alignment_keep_all_fixed_slots():
    m = script_module()
    raw = m.raw_module()
    row = {
        "schema": raw.PROJECTION,
        "split": "train",
        "media_key": "a" * 64,
        "media_path": "train/x/x.mp4",
        "media_sha256": "b" * 64,
        "width": 8,
        "height": 6,
        "length": 20,
        "persons": [
            {"slot": s, "bbox_xyxy": [[0, 0, 8, 6]] * 20, "bbox_present": [True] * 20}
            for s in range(2)
        ],
    }
    for status, d in [("failed", 20), ("alignment_unresolved", 19)]:
        arrays = raw.empty_raw(row)
        for name in ("source_frame_index", "pts", "best_effort_timestamp"):
            arrays[name] = np.arange(d, dtype=np.int64)
        for name in ("decoded_ok", "pts_present", "best_effort_present"):
            arrays[name] = np.zeros(d, bool)
        meta = {
            "annotation_length": 20,
            "decoded_length": d,
            "status": status,
            "media_key": "a" * 64,
            "clocks": {},
        }
        results = [nb.preprocess_native(nb.project_raw_identity(arrays, meta, s)) for s in range(2)]
        assert len(results) == 2
        assert all(r.status == "unusable" and r.response_input.shape == (20, 269) for r in results)


def test_cli_pending_complete_cache_does_not_build_partial(tmp_path):
    m = script_module()
    raw = m.raw_module()
    row = {
        "schema": raw.PROJECTION,
        "split": "train",
        "media_key": "a" * 64,
        "media_path": "train/x/x.mp4",
        "media_sha256": "b" * 64,
        "width": 8,
        "height": 6,
        "length": 20,
        "persons": [{"slot": 0, "bbox_xyxy": [[0, 0, 8, 6]] * 20, "bbox_present": [True] * 20}],
    }
    census = tmp_path / "census.jsonl"
    census.write_text(json.dumps(row) + "\n")
    cfg = {
        "raw_root": str(tmp_path / "input"),
        "census": str(census),
        "output": str(tmp_path / "output"),
    }
    expected = {"train": {"videos": 1, "persons": 1}, "val": {"videos": 0, "persons": 0}}
    assert m.execute(cfg, "build", expected=expected) == 2
    assert not (tmp_path / "output/identities").exists()
    assert json.loads((tmp_path / "output/population_pending.json").read_text())["identities"] == 1


def test_failed_raw_retains_real_clock_and_returned_observation_payload():
    f = feature(20, pts=np.arange(20, dtype=np.int64) * 1000)
    arrays = {
        "slot": np.array([0]),
        "decoded_ok": np.ones(20, bool),
        "bbox_valid": np.ones((1, 20), bool),
        "pose_returned": np.ones((1, 20), bool),
        "joint_observed": f.joint_observed[None],
        "xyconf_raw": f.xyconf_raw[None],
        "pts": f.pts,
        "pts_present": f.pts_present,
    }
    meta = {
        "annotation_length": 20,
        "decoded_length": 20,
        "status": "failed",
        "media_key": "a" * 64,
        "clocks": {"time_base_num": 1, "time_base_den": 1000},
    }
    projected = nb.project_raw_identity(arrays, meta, 0)
    r = nb.preprocess_native(projected)
    assert r.status == "unusable" and r.clock.mode == "pts"
    np.testing.assert_array_equal(projected.pts, f.pts)
    np.testing.assert_array_equal(projected.xyconf_raw, f.xyconf_raw)
    assert projected.joint_observed.all() and not projected.frame_source_ok.any()


def test_native_target_loader_exact_members_and_hash(tmp_path):
    import hashlib

    p = tmp_path / "target.npz"
    np.savez(
        p,
        target_valid=np.ones(3, bool),
        pulse=np.array([0.0, 1.0, 0.0]),
        chi=np.ones(3),
        traversal_bounds=np.array([[0, 3]], np.int64),
    )
    meta = {
        "schema": nb.TARGET_SCHEMA,
        "branch_provenance": "full-chain geometry",
        "producer": "fixture",
        "producer_sha256": "a" * 64,
        "npz_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    p.with_suffix(".json").write_text(json.dumps(meta))
    assert nb.load_native_target(p).traversal_bounds.tolist() == [[0, 3]]
    p.write_bytes(p.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="hash"):
        nb.load_native_target(p)


@pytest.mark.parametrize("pulse_kind", ["positive_at_10", "constant_point_one"])
def test_response_gap_disqualifies_whole_traversal_preserving_other_traversals(pulse_kind):
    f = feature(80)
    source_ok = f.frame_source_ok.copy()
    source_ok[40] = False
    identity = nb.preprocess_native(replace(f, frame_source_ok=source_ok))
    assert not identity.response_edge_support[39:41].any()
    valid = np.zeros(79, bool)
    valid[10:79] = True
    chi = np.zeros(79)
    chi[valid] = 1 / 50
    pulse = np.zeros(79)
    if pulse_kind == "positive_at_10":
        pulse[[10, 60]] = 1
    else:
        pulse[valid] = 0.10
    target = nb.NativeTargetV1(
        valid,
        pulse,
        chi,
        np.array([[10, 60], [60, 79]], np.int64),
        "full-chain geometry",
        "fixture",
        "a" * 64,
    )
    supervision = nb.native_supervision(identity, target)
    assert supervision["traversal_eligible"] == (False, True)
    assert supervision["traversal_reasons"] == ("response_support_gap", None)
    assert not supervision["supervised"][10:60].any()
    assert not supervision["positive_weights"][10:60].any()
    assert not supervision["negative_weights"][10:60].any()
    assert supervision["supervised"][60:79].all()
    np.testing.assert_array_equal(supervision["positive_weights"][60:79], chi[60:79] * pulse[60:79])
    np.testing.assert_array_equal(
        supervision["negative_weights"][60:79], chi[60:79] * (1 - pulse[60:79])
    )


def test_native_target_per_traversal_provenance_and_mixed_loader(tmp_path):
    import hashlib

    valid = np.ones(10, bool)
    pulse = np.resize(np.array([0.0, 1.0]), 10)
    bounds = np.array([[0, 5], [5, 10]], np.int64)
    sources = ("full-chain geometry", "stable-pair hybrid")
    target = nb.NativeTargetV1(
        valid, pulse, np.ones(10), bounds, "mixed", "fixture", "a" * 64, sources
    )
    assert target.branch_provenance == "mixed" and target.traversal_provenance == sources
    single = nb.NativeTargetV1(
        valid, pulse, np.ones(10), bounds, "full-chain geometry", "fixture", "a" * 64
    )
    assert single.traversal_provenance == ("full-chain geometry",) * 2
    for bad, message in [
        (None, "per-traversal"),
        ((sources[0],), "one-to-one"),
        ((sources[0], sources[0]), "summary"),
        ((sources[0], "rejected"), "certified"),
    ]:
        with pytest.raises(ValueError, match=message):
            replace(target, traversal_provenance=bad)
    with pytest.raises(ValueError, match="summary"):
        replace(target, branch_provenance="full-chain geometry")
    p = tmp_path / "mixed.npz"
    np.savez(p, target_valid=valid, pulse=pulse, chi=np.ones(10), traversal_bounds=bounds)
    meta = {
        "schema": nb.TARGET_SCHEMA,
        "branch_provenance": "mixed",
        "traversal_provenance": list(sources),
        "producer": "fixture",
        "producer_sha256": "a" * 64,
        "npz_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    p.with_suffix(".json").write_text(json.dumps(meta))
    loaded = nb.load_native_target(p)
    assert loaded.traversal_provenance == sources and loaded.branch_provenance == "mixed"
    meta["branch_provenance"] = "full-chain geometry"
    p.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="summary"):
        nb.load_native_target(p)
