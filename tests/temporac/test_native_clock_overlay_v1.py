"""Synthetic CPU clocks and complete fixture populations; no natural labels/models."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pams.temporac.native_clock_overlay_v1 import (
    ROOT,
    build_clock_overlay,
    canonical,
    load_overlay,
    readonly_nb1,
    resolve_clock,
    save_overlay,
    sha,
)


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, *, pts=None, old_pts=None, n=40, persons=2, name="x"):
    nb = readonly_nb1()
    raw = nb.raw_module()
    media_path = f"train/{name}/{name}.mp4"
    key = sha(media_path.encode())
    row = dict(
        schema=raw.PROJECTION,
        split="train",
        media_key=key,
        media_path=media_path,
        media_sha256="b" * 64,
        width=8,
        height=6,
        length=n,
        persons=[
            dict(slot=s, bbox_xyxy=[[0, 0, 8, 6]] * n, bbox_present=[True] * n)
            for s in range(persons)
        ],
    )
    a = raw.empty_raw(row)
    clocks, _ = raw.attach_clock(a, row, n)
    xy = np.stack((np.linspace(-0.8, 0.8, 17), np.linspace(0.2, 1, 17)), axis=1).astype(np.float32)
    xy[11], xy[12], xy[5], xy[6] = (-0.2, 0), (0.2, 0), (-0.3, 0.8), (0.3, 0.8)
    a["xyconf_raw"][..., :2] = xy
    a["xyconf_raw"][..., 2] = 0.9
    a["pose_returned"][:] = a["joint_observed"][:] = True
    a["reason"][:] = 0
    if old_pts is not None:
        a["pts"][:] = [0 if p is None else p for p in old_pts]
        a["pts_present"][:] = [p is not None for p in old_pts]
        clocks.update(time_base_num=1, time_base_den=1000)
    meta = dict(
        schema=raw.SCHEMA,
        status="complete",
        annotation_length=n,
        decoded_length=n,
        width=8,
        height=6,
        media_key=key,
        split="train",
        media_sha256="b" * 64,
        binding={
            "media_sha256": "b" * 64,
            "script_sha256": sha(
                (ROOT / "scripts/build_temporac_native_pose_cache.py").read_bytes()
            ),
        },
        decoder={"backend": "FFMPEG", "name": "opencv", "version": "4.11.0"},
        slots=[{"slot": s, "status": "complete"} for s in range(persons)],
        clocks=clocks,
        errors=[],
    )
    base = tmp_path / "rawroot/raw/train" / key
    raw.save_raw(base, a, meta)
    points = list(range(n)) if pts is None else pts
    csv_path = tmp_path / "decode/clocks" / (name + ".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "frame_index,stream_index,width,height,pts,best_effort_timestamp\n"
        + "".join(f"{i},0,8,6,{'' if p is None else p},{i}\n" for i, p in enumerate(points)),
        encoding="utf-8",
    )
    stat = dict(st_dev=1, st_ino=2, st_size=99, st_mtime_ns=4)
    decode = dict(
        path=media_path,
        split="train",
        status="decoded",
        prior_sha256="b" * 64,
        prior_bytes=99,
        stat_before=stat,
        stat_after=stat.copy(),
        annotation={"length": n, "width": 8, "height": 6},
        stream={"index": 0, "width": 8, "height": 6, "time_base": "1/1000"},
        decode={"returncode": 0, "parse_errors": [], "decoded_frames": n, "dimensions": {"8x6": n}},
        findings=[],
        frame_clock_csv=f"clocks/{name}.csv",
    )
    return row, a, meta, base, decode, csv_path


def overlay_fixture(data):
    _, _, _, base, decode, csv_path = data
    return build_clock_overlay(
        base.with_suffix(".json"), base.with_suffix(".npz"), canonical(decode) + b"\n", csv_path
    )


def test_attach_nonclock_bytes_idempotence_and_storage(tmp_path):
    data = fixture(tmp_path)
    row, a, meta, base, decode, csv_path = data
    original_bytes = base.with_suffix(".npz").read_bytes(), base.with_suffix(".json").read_bytes()
    overlay = overlay_fixture(data)
    f = readonly_nb1().project_raw_identity(a, meta, 0)
    resolved, provenance = resolve_clock(f, overlay)
    assert provenance["selection"] == "overlay-attached"
    assert "no decode-time media rehash" in provenance["media_sha_assurance"]
    for key, value in f.as_arrays().items():
        if key not in ("pts", "pts_present", "time_base"):
            assert resolved.as_arrays()[key].tobytes() == value.tobytes()
    again, _ = resolve_clock(resolved, overlay)
    assert all(
        again.as_arrays()[k].tobytes() == v.tobytes() for k, v in resolved.as_arrays().items()
    )
    assert original_bytes == (
        base.with_suffix(".npz").read_bytes(),
        base.with_suffix(".json").read_bytes(),
    )
    assert overlay.metadata["decode_row_sha256"] == sha(canonical(decode) + b"\n")
    assert overlay.metadata["clock_csv_sha256"] == sha(csv_path.read_bytes())
    dest = tmp_path / "saved/overlay"
    save_overlay(dest, overlay)
    save_overlay(dest, overlay)
    assert load_overlay(dest).content_sha256 == overlay.content_sha256
    with pytest.raises(ValueError):
        overlay.pts.setflags(write=True)
    with pytest.raises(ValueError, match="identity"):
        resolve_clock(replace(f, opaque_key=np.zeros(32, np.uint8)), overlay)


@pytest.mark.parametrize(
    "kind",
    ["all_missing", "partial_missing", "duplicate", "regression", "large_integer", "raw_complete"],
)
def test_six_clock_classes_and_nb1_support(tmp_path, kind):
    pts = list(range(40))
    old = None
    if kind == "all_missing":
        pts = [None] * 40
    elif kind == "partial_missing":
        pts[20] = None
    elif kind == "duplicate":
        pts[20] = pts[19]
    elif kind == "regression":
        pts[20] = pts[19] - 1
    elif kind == "large_integer":
        pts = [2**60 + i for i in pts]
    else:
        old = pts
    data = fixture(tmp_path, pts=pts, old_pts=old)
    overlay = overlay_fixture(data)
    _, a, meta, _, _, _ = data
    f, _ = resolve_clock(readonly_nb1().project_raw_identity(a, meta, 0), overlay)
    identity = readonly_nb1().preprocess_native(f)
    assert f.pts.tolist() == [0 if p is None else p for p in pts]
    assert f.pts_present.tolist() == [p is not None for p in pts]
    if kind == "all_missing":
        assert identity.clock.mode == "source_frame_only"
        assert overlay.metadata["selection"] == "unavailable"
    else:
        assert identity.clock.mode == "pts"
        if kind in ("partial_missing", "duplicate", "regression"):
            assert not identity.clock.edge_time_support[19]
            assert len(identity.clock.segments) >= 2
        else:
            assert identity.clock.edge_time_support.all()
    if kind == "raw_complete":
        assert overlay.metadata["selection"] == "raw-retained"


@pytest.mark.parametrize(
    "damage",
    [
        "sha",
        "stat",
        "stream",
        "dimensions",
        "length",
        "parse",
        "rawfailed",
        "old_pts",
        "old_unit",
        "csvindex",
        "csvstream",
        "csvdimension",
        "csvbadint",
        "overflow",
        "timebase",
        "rawhash",
    ],
)
def test_binding_corruptions_reject_without_fallback(tmp_path, damage):
    data = fixture(tmp_path, old_pts=list(range(40)))
    _, _, meta, base, decode, csv_path = data
    if damage == "sha":
        decode["prior_sha256"] = "c" * 64
    elif damage == "stat":
        decode["stat_after"]["st_mtime_ns"] += 1
    elif damage == "stream":
        decode["stream"]["index"] = 1
    elif damage == "dimensions":
        decode["stream"]["height"] = 7
    elif damage == "length":
        decode["decode"]["decoded_frames"] -= 1
    elif damage == "parse":
        decode["decode"]["parse_errors"] = ["bad"]
    elif damage == "timebase":
        decode["stream"]["time_base"] = "1/0"
    elif damage.startswith("csv") or damage == "overflow":
        text = csv_path.read_text()
        replacement = {
            "csvindex": "1,0,8,6,0,0",
            "csvstream": "0,1,8,6,0,0",
            "csvdimension": "0,0,9,6,0,0",
            "csvbadint": "0,0,8,6,0.0,0",
            "overflow": f"0,0,8,6,{2**63},0",
        }[damage]
        csv_path.write_text(text.replace("0,0,8,6,0,0", replacement))
    else:
        if damage == "rawfailed":
            meta["status"] = "failed"
        elif damage == "old_pts":
            csv_path.write_text(csv_path.read_text().replace("0,0,8,6,0,0", "0,0,8,6,3,0"))
        elif damage == "old_unit":
            meta["clocks"]["time_base_den"] = 999
        elif damage == "rawhash":
            meta["npz_sha256"] = "f" * 64
        base.with_suffix(".json").write_bytes(canonical(meta))
    with pytest.raises((ValueError, ZeroDivisionError)):
        overlay_fixture(data)


def population(tmp_path):
    data = [fixture(tmp_path, name=name) for name in ("x", "y")]
    census = tmp_path / "census.jsonl"
    census.write_bytes(b"".join(canonical(d[0]) + b"\n" for d in data))
    decode_root = tmp_path / "decode"
    (decode_root / "decode.jsonl").write_bytes(b"".join(canonical(d[4]) + b"\n" for d in data))
    (decode_root / "status.json").write_bytes(
        canonical(
            {
                "status": "complete",
                "completed": 2,
                "script_sha256": sha((ROOT / "scripts/audit_temporac_decode.py").read_bytes()),
            }
        )
    )
    inventory = tmp_path / "media.jsonl"
    inventory.write_bytes(
        b"".join(
            canonical(
                dict(
                    path=d[0]["media_path"],
                    split="train",
                    sha256="b" * 64,
                    status="hashed",
                    bytes=99,
                )
            )
            + b"\n"
            for d in data
        )
    )
    cfg = dict(
        raw_root=str(tmp_path / "rawroot"),
        census=str(census),
        decode_root=str(decode_root),
        media_inventory=str(inventory),
        output=str(tmp_path / "overlay"),
    )
    expected = {"train": {"videos": 2, "persons": 4}, "val": {"videos": 0, "persons": 0}}
    return data, cfg, expected


def test_plan_spine(tmp_path):
    _, cfg, expected = population(tmp_path)
    assert (
        script("build_temporac_native_clock_overlay").execute(cfg, "plan", expected=expected) == 0
    )
    assert json.loads((Path(cfg["output"]) / "plan.json").read_bytes())["identities"] == 4


def test_complete_census_resume_validate_and_old_artifact_invalidation(tmp_path):
    data, cfg, expected = population(tmp_path)
    oc = script("build_temporac_native_clock_overlay")
    qb = script("build_temporac_qualified_native_bridge")
    for command in ("build", "build", "validate"):
        assert oc.execute(cfg, command, expected=expected) == 0
    manifest = Path(cfg["output"]) / "overlay_manifest.json"
    qcfg = dict(
        raw_root=cfg["raw_root"],
        census=cfg["census"],
        overlay_manifest=str(manifest),
        output=str(tmp_path / "qualified"),
    )
    for command in ("plan", "build", "build", "validate"):
        assert qb.execute(qcfg, command, expected=expected) == 0
    receipt = json.loads((tmp_path / "qualified/validation.json").read_bytes())
    assert receipt["identities"] == 4
    assert receipt["clock_qualification_counts"] == {"qualified": 4}
    assert receipt["clock_mode_counts"] == {"pts": 4}
    key = data[0][0]["media_key"]
    sidecar = tmp_path / "qualified/identities/train" / f"{key}_0.json"
    prior = json.loads(sidecar.read_bytes())
    # Old source-frame fallback artifacts cannot pass the new build binding.
    prior["binding"].pop("overlay_manifest_sha256")
    sidecar.write_bytes(canonical(prior))
    assert qb.execute(qcfg, "build", expected=expected) == 2
    failed = json.loads((tmp_path / "qualified/population_ledger.json").read_bytes())
    assert failed["identities"] == 4
    assert failed["status_counts"]["artifact_failure"] == 2
    assert failed["clock_qualification_counts"]["qualified"] == 2
    assert "resume binding mismatch" in failed["failures"][0]["error"]


def test_full_census_pending_and_rejected_slots_retained(tmp_path):
    data, cfg, expected = population(tmp_path)
    oc = script("build_temporac_native_clock_overlay")
    status = tmp_path / "decode/status.json"
    status.write_bytes(canonical({"status": "running", "completed": 1}))
    assert oc.execute(cfg, "build", expected=expected) == 2
    assert not (tmp_path / "overlay/videos").exists()
    assert not (tmp_path / "overlay/overlay_manifest.json").exists()
    status.write_bytes(
        canonical(
            {
                "status": "complete",
                "completed": 2,
                "script_sha256": sha((ROOT / "scripts/audit_temporac_decode.py").read_bytes()),
            }
        )
    )
    data[0][5].write_text(data[0][5].read_text().replace("0,0,8,6,0,0", "1,0,8,6,0,0"))
    assert oc.execute(cfg, "build", expected=expected) == 2
    qcfg = dict(
        raw_root=cfg["raw_root"],
        census=cfg["census"],
        overlay_manifest=str(tmp_path / "overlay/overlay_manifest.json"),
        output=str(tmp_path / "qualified"),
    )
    assert (
        script("build_temporac_qualified_native_bridge").execute(qcfg, "build", expected=expected)
        == 2
    )
    ledger = json.loads((tmp_path / "qualified/population_ledger.json").read_bytes())
    assert ledger["identities"] == 4 and ledger["status_counts"]["artifact_failure"] == 2
    assert not (
        tmp_path / "qualified/identities/train" / (data[0][0]["media_key"] + "_0.npz")
    ).exists()


def test_manifest_change_invalidates_qualified_build(tmp_path):
    _, cfg, expected = population(tmp_path)
    oc = script("build_temporac_native_clock_overlay")
    qb = script("build_temporac_qualified_native_bridge")
    assert oc.execute(cfg, "build", expected=expected) == 0
    manifest = tmp_path / "overlay/overlay_manifest.json"
    qcfg = dict(
        raw_root=cfg["raw_root"],
        census=cfg["census"],
        overlay_manifest=str(manifest),
        output=str(tmp_path / "qualified"),
    )
    assert qb.execute(qcfg, "build", expected=expected) == 0
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    assert qb.execute(qcfg, "validate", expected=expected) == 2
    with pytest.raises(ValueError, match="census"):
        qb.execute(qcfg, "plan")


def test_raw_partial_pts_backfill_and_rational_noop(tmp_path):
    data = fixture(tmp_path, old_pts=[i if i % 2 == 0 else None for i in range(40)])
    overlay = overlay_fixture(data)
    resolved, _ = resolve_clock(readonly_nb1().project_raw_identity(data[1], data[2], 0), overlay)
    assert resolved.pts_present.all() and resolved.pts.tolist() == list(range(40))
    full = fixture(tmp_path, old_pts=list(range(40)), name="full")
    full[2]["clocks"].update(time_base_num=2, time_base_den=2000)
    full[3].with_suffix(".json").write_bytes(canonical(full[2]))
    original = readonly_nb1().project_raw_identity(full[1], full[2], 0)
    result, _ = resolve_clock(original, overlay_fixture(full))
    assert result.time_base.tobytes() == original.time_base.tobytes()


def test_unverified_decoder_version_rejected(tmp_path):
    data = fixture(tmp_path)
    data[2]["decoder"]["version"] = "4.10.0"
    data[3].with_suffix(".json").write_bytes(canonical(data[2]))
    with pytest.raises(ValueError, match="presentation order"):
        overlay_fixture(data)


def test_read_set_changes_rejected(tmp_path, monkeypatch):
    import pams.temporac.native_clock_overlay_v1 as module

    data = fixture(tmp_path)
    original = module.stable_bytes
    reads = []

    def changing(path):
        result = original(path)
        if Path(path) == data[5]:
            reads.append(1)
            if len(reads) == 1:
                data[5].write_bytes(result + b"\n")
        return result

    monkeypatch.setattr(module, "stable_bytes", changing)
    with pytest.raises(ValueError, match="changed during overlay"):
        overlay_fixture(data)


def test_duplicate_decode_rows_rejected(tmp_path):
    _, cfg, expected = population(tmp_path)
    path = tmp_path / "decode/decode.jsonl"
    path.write_bytes(path.read_bytes() + path.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="multiple decode clocks"):
        script("build_temporac_native_clock_overlay").execute(cfg, "build", expected=expected)
    assert not (tmp_path / "overlay/overlay_manifest.json").exists()
