"""CPU contract tests; synthetic frontend outputs are fixtures, never experiment results."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "native_pose_builder",
    Path(__file__).resolve().parents[2] / "scripts/build_temporac_native_pose_cache.py",
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def row(n=3, p=2):
    return dict(
        schema=native.PROJECTION,
        split="train",
        media_key="a" * 64,
        media_path="train/video/video.mp4",
        media_sha256="b" * 64,
        width=8,
        height=6,
        length=n,
        persons=[
            dict(slot=i, bbox_xyxy=[[i, 0, i + 3, 5] for _ in range(n)], bbox_present=[True] * n)
            for i in range(p)
        ],
    )


def frames(n):
    return [np.full((6, 8, 3), i, dtype=np.uint8) for i in range(n)]


def fixture_frontend(images, boxes):
    return [
        dict(
            labels=np.arange(17),
            keypoints=np.full((17, 2), int(im[0, 0, 0]) * 10 + box[0]),
            scores=np.full(17, -0.1),
        )
        for im, box in zip(images, boxes, strict=True)
    ]


def test_native_multi_frame_slot_and_low_score():
    r = row(7, 3)
    arrays, meta = native.extract_video(r, frames(7), fixture_frontend, 4)
    assert arrays["xyconf_raw"].shape == (3, 7, 17, 3)
    assert arrays["joint_observed"].all()
    for f in range(7):
        for p in range(3):
            assert arrays["xyconf_raw"][p, f, 0, 0] == 10 * f + p
    assert (arrays["xyconf_raw"][..., 2] < 0).all()  # Raw confidence is never clipped.
    assert meta["status"] == "complete"
    assert native.validate_raw(arrays, meta, r)["bridge"]["admissible"] is False
    assert not arrays["pts_present"].any()


def test_missing_invalid_partial_outside_bbox_and_zero_failure():
    r = row()
    r["persons"][0]["bbox_present"][0] = False
    r["persons"][0]["bbox_xyxy"][0] = [0] * 4
    r["persons"][0]["bbox_xyxy"][1] = [4, 4, 4, 4]
    r["persons"][0]["bbox_xyxy"][2] = [-1, -1, 2, 3]
    arrays, meta = native.extract_video(r, frames(3), fixture_frontend, 2)
    assert not arrays["pose_returned"][0, 0]
    assert not arrays["pose_returned"][0, 1]
    assert arrays["pose_returned"][0, 2]
    assert native.xyxy_to_xywh([-1, -1, 2, 3]) == [-1, -1, 3, 4]
    native.validate_raw(arrays, meta, r)

    def failed(*args):
        raise RuntimeError("fixture frontend failure")

    failed_arrays, failed_meta = native.extract_video(row(), frames(3), failed, 3)
    assert failed_meta["status"] == "failed"
    assert not failed_arrays["joint_observed"].any()
    assert len(failed_meta["slots"]) == 2
    native.validate_raw(failed_arrays, failed_meta, row())


def test_nonfinite_joint_keeps_other_joints_and_labels():
    def frontend(images, boxes):
        result = fixture_frontend(images, boxes)
        result[0]["keypoints"] = result[0]["keypoints"].astype(float)
        result[0]["keypoints"][4, 0] = np.nan
        return result

    a, m = native.extract_video(row(1, 1), frames(1), frontend, 1)
    assert a["pose_returned"][0, 0]
    assert a["joint_invalid"][0, 0, 4]
    assert not a["joint_observed"][0, 0, 4]
    assert a["joint_observed"].sum() == 16
    assert not a["xyconf_raw"][0, 0, 4].any()
    native.validate_raw(a, m, row(1, 1))
    pose = fixture_frontend(frames(1), [[0, 0, 2, 2]])[0]
    pose["labels"] = np.arange(17)[::-1]
    with pytest.raises(ValueError, match="COCO"):
        native.assign_pose(native.empty_raw(row(1, 1)), 0, 0, pose)


def test_length_mismatch_preserves_two_ledgers():
    r = row(5, 2)
    a, m = native.extract_video(r, frames(3), fixture_frontend, 4)
    assert m["status"] == "alignment_unresolved"
    assert a["xyconf_raw"].shape == (2, 5, 17, 3)
    assert len(a["source_frame_index"]) == 3
    assert not a["joint_observed"].any()
    native.validate_raw(a, m, r)
    a, m = native.extract_video(r, frames(8), fixture_frontend, 4)
    assert len(a["source_frame_index"]) == 8
    assert a["bbox_xyxy"].shape == (2, 5, 4)
    assert not a["pose_returned"].any()
    native.validate_raw(a, m, r)


def test_preserves_missing_duplicate_pts_and_best_effort(tmp_path):
    (tmp_path / "clocks").mkdir()
    with (tmp_path / "clocks/0000.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_index", "stream_index", "width", "height", "pts", "best_effort_timestamp"]
        )
        writer.writerows([[0, 0, 8, 6, 100, 100], [1, 0, 8, 6, "", 101], [2, 0, 8, 6, 100, 102]])
    r = row()
    audit = dict(
        status="decoded",
        prior_sha256=r["media_sha256"],
        decode={"decoded_frames": 3},
        frame_clock_csv="clocks/0000.csv",
        stream={"time_base": "1/24000"},
    )
    a, m = native.extract_video(r, frames(3), fixture_frontend, 2, audit, tmp_path)
    assert a["pts"].tolist() == [100, 0, 100]
    assert a["pts_present"].tolist() == [True, False, True]
    assert a["best_effort_timestamp"].tolist() == [100, 101, 102]
    assert m["clocks"]["time_base_den"] == 24000
    native.validate_raw(a, m, r)


def test_projection_whitelist_and_numeric_object_order(tmp_path):
    root = tmp_path / "source"
    (root / "train/v").mkdir(parents=True)
    media = root / "train/v/v.mp4"
    media.write_bytes(b"fixture media, not decoded")
    # Trusted source may contain labels; their values cannot escape the projection.
    raw = dict(
        width=8,
        height=6,
        length=1,
        video_name="v",
        object10={"bbox": [[1, 0, 3, 5]], "count": 999, "period": [[3, 100]], "periodicity": []},
        object2={"bbox": [None], "count": 888, "period": [], "periodicity": []},
    )
    media.with_suffix(".json").write_text(json.dumps(raw))
    manifest = tmp_path / "media.jsonl"
    manifest.write_text(
        json.dumps(
            dict(
                split="train",
                path="train/v/v.mp4",
                sha256=native.digest_file(media),
                status="hashed",
            )
        )
        + "\n"
    )
    expected = {"train": {"videos": 1, "persons": 2}, "val": {"videos": 0, "persons": 0}}
    rows, prov = native.project_bbox_only(manifest, root, expected=expected)
    native.reject_privileged(rows)
    assert prov["records"][0]["source_object_to_slot"] == {"object2": 0, "object10": 1}
    assert rows[0]["persons"][0]["bbox_present"] == [False]
    native.validate_projection(rows, expected=expected)
    for key in ("count", "period", "periodicity", "density", "boundary", "extra"):
        bad = copy.deepcopy(rows[0])
        bad[key] = 0
        with pytest.raises(ValueError):
            native.validate_projection_record(bad)
    bad = copy.deepcopy(rows[0])
    bad["persons"][0]["count"] = 99
    with pytest.raises(ValueError):
        native.validate_projection_record(bad)
    bad = copy.deepcopy(rows[0])
    bad["split"] = "test"
    with pytest.raises(ValueError):
        native.validate_projection_record(bad)
    with pytest.raises(ValueError, match="census"):
        native.census(rows)


def test_atomic_raw_hash_and_resume_binding(tmp_path):
    r = row()
    a, m = native.extract_video(r, frames(3), fixture_frontend, 3)
    native.save_raw(tmp_path / "raw", a, m)
    b, n = native.load_raw(tmp_path / "raw")
    native.validate_raw(b, n, r)
    binding = native.binding_for(r, {"batch_size": 3, "device": "cpu"}, native.ASSETS, None)
    different = native.binding_for(r, {"batch_size": 4, "device": "cpu"}, native.ASSETS, None)
    assert binding != different
    assert not list(tmp_path.glob("*.tmp"))
    target = tmp_path / "raw.npz"
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        native.load_raw(tmp_path / "raw")


def test_decode_empty_and_dimension_failure_keeps_slots():
    a, m = native.extract_video(row(), [], fixture_frontend, 2)
    assert len(m["slots"]) == 2 and m["decoded_length"] == 0
    assert not a["pose_returned"].any()
    native.validate_raw(a, m, row())
    a, m = native.extract_video(row(1), [np.zeros((4, 4, 3), dtype=np.uint8)], fixture_frontend, 2)
    assert m["status"] == "alignment_unresolved"
    assert not a["joint_observed"].any()


def test_plan_cli_is_honest(tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"output": str(tmp_path / "out")}))
    assert native.main(["plan", "--config", str(config)]) == 0
    assert (
        json.loads((tmp_path / "out/plan.json").read_text())["status"] == "prepared_not_extracted"
    )


def test_video_initialization_failure_has_all_failed_slots(tmp_path, monkeypatch):
    r = row(2, 2)
    root = tmp_path / "source"
    (root / "train/video").mkdir(parents=True)
    media = root / r["media_path"]
    media.write_bytes(b"fixture media")
    r["media_sha256"] = native.digest_file(media)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setattr(native, "verify_assets", lambda snapshot: native.ASSETS)

    def unavailable(*args):
        raise RuntimeError("fixture model unavailable")

    monkeypatch.setattr(native, "ViTPoseFrontend", unavailable)
    result = native.extract_native(
        [r], None, tmp_path / "snapshot", tmp_path / "output", "cpu", 2, source_root=root
    )
    assert len(result) == 1
    a, m = native.load_raw(tmp_path / "output/raw/train" / r["media_key"])
    assert len(m["slots"]) == 2
    assert all(s["status"] == "failed" for s in m["slots"])
    assert not a["joint_observed"].any()
    native.validate_raw(a, m, r)
    ledger = json.loads((tmp_path / "output/population_ledger.json").read_text())
    assert ledger["records"][0]["status"] == "alignment_unresolved"


def test_extract_resume_reuses_only_matching_complete_artifact(tmp_path, monkeypatch):
    r = row(2, 2)
    root = tmp_path / "source"
    (root / "train/video").mkdir(parents=True)
    media = root / r["media_path"]
    media.write_bytes(b"fixture media")
    r["media_sha256"] = native.digest_file(media)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setattr(native, "verify_assets", lambda snapshot: native.ASSETS)
    monkeypatch.setattr(native, "opencv_frames", lambda path, info: iter(frames(2)))

    class FixtureFrontend:
        environment = {"fixture": True}

        def __init__(self, *args):
            pass

        def __call__(self, images, boxes):
            return fixture_frontend(images, boxes)

    monkeypatch.setattr(native, "ViTPoseFrontend", FixtureFrontend)
    output = tmp_path / "output"
    first = native.extract_native([r], None, tmp_path, output, "cpu", 2, source_root=root)
    assert first[0]["status"] == "complete"

    def must_not_initialize(*args):
        raise AssertionError("resume must not initialize model")

    monkeypatch.setattr(native, "ViTPoseFrontend", must_not_initialize)
    second = native.extract_native(
        [r], None, tmp_path, output, "cpu", 2, source_root=root, resume=True
    )
    assert second[0]["status"] == "reused_complete"
    with pytest.raises(ValueError, match="binding mismatch"):
        native.extract_native([r], None, tmp_path, output, "cpu", 3, source_root=root, resume=True)


def test_resume_rejects_changed_media_and_retries_failed_new_batch(tmp_path, monkeypatch):
    r = row(2, 2)
    root = tmp_path / "source"
    (root / "train/video").mkdir(parents=True)
    media = root / r["media_path"]
    media.write_bytes(b"original fixture media")
    r["media_sha256"] = native.digest_file(media)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setattr(native, "verify_assets", lambda snapshot: native.ASSETS)
    monkeypatch.setattr(native, "opencv_frames", lambda path, info: iter(frames(2)))

    class Frontend:
        environment = {"fixture": True}

        def __init__(self, *args):
            pass

        def __call__(self, images, boxes):
            return fixture_frontend(images, boxes)

    monkeypatch.setattr(native, "ViTPoseFrontend", Frontend)
    output = tmp_path / "complete"
    native.extract_native([r], None, tmp_path, output, "cpu", 2, source_root=root)
    completed_npz = output / "raw/train" / (r["media_key"] + ".npz")
    before = native.digest_file(completed_npz)
    media.write_bytes(b"changed fixture media")
    with pytest.raises(ValueError, match="media path/SHA changed"):
        native.extract_native([r], None, tmp_path, output, "cpu", 2, source_root=root, resume=True)
    assert native.digest_file(completed_npz) == before
    media.write_bytes(b"original fixture media")

    class FailingFrontend(Frontend):
        def __call__(self, images, boxes):
            raise RuntimeError("fixture oversized batch failure")

    monkeypatch.setattr(native, "ViTPoseFrontend", FailingFrontend)
    failed_output = tmp_path / "failed"
    failed = native.extract_native([r], None, tmp_path, failed_output, "cpu", 2, source_root=root)
    assert failed[0]["status"] == "failed"
    old_attempts = {
        p: p.read_bytes() for p in (failed_output / "attempts" / r["media_key"]).glob("*.json")
    }
    monkeypatch.setattr(native, "ViTPoseFrontend", Frontend)
    retry = native.extract_native(
        [r], None, tmp_path, failed_output, "cpu", 1, source_root=root, resume=True
    )
    assert retry[0]["status"] == "complete"
    assert all(p.read_bytes() == data for p, data in old_attempts.items())
    assert len(list((failed_output / "attempts" / r["media_key"]).glob("*.json"))) == 2
