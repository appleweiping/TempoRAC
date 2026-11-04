"""Independent bbox-only native raw pose cache; no fixed-320 learner bridge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path, PurePosixPath

SCHEMA = "temporac.native-pose.v1"
PROJECTION = "temporac.bbox-only.v1"
EXPECTED = {"train": {"videos": 794, "persons": 1878}, "val": {"videos": 183, "persons": 486}}
ASSETS = {
    "model.safetensors": "85375373893ddd3641f3912821073e53f5435f9e966e1dca59d004454bfe4fdf",
    "config.json": "72fc2f0fc29eef43e87aa0ddd4c1269c5226ae157a99d0057019dd40129fed7c",
    "preprocessor_config.json": "9b11cadc98c30b968a70cc1658ce1fbd74b721b0f21402a2ff8bc1dc9d1474a0",
    "README.md": "5e5c36e4905746e856f2913910fcfe4cb3965e771857154e7075c3d06e6bf98e",
}
FIELDS = {
    "schema",
    "split",
    "media_key",
    "media_path",
    "media_sha256",
    "width",
    "height",
    "length",
    "persons",
}
PERSON_FIELDS = {"slot", "bbox_xyxy", "bbox_present"}
PRIVILEGED = {
    "count",
    "count_gt",
    "period",
    "periods",
    "periodicity",
    "density",
    "boundary",
    "boundaries",
}
REASONS = {
    0: "pose_returned",
    1: "bbox_absent",
    2: "bbox_invalid",
    3: "frontend_failed",
    4: "alignment_unresolved",
    5: "decode_failed",
    6: "pending",
}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def read_jsonl(path, running=False):
    if running and (path is None or not Path(path).exists()):
        return []
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if running and not line.endswith("\n"):
                    break
                raise
    return rows


def reject_privileged(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in PRIVILEGED:
                raise ValueError("privileged projection field: " + key)
            reject_privileged(item)
    elif isinstance(value, list):
        for item in value:
            reject_privileged(item)


def relative_media(value, split):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if (
        split not in ("train", "val")
        or path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != split
        or path.suffix.lower() != ".mp4"
    ):
        raise ValueError("media must stay in explicit train/val split")
    return path.as_posix()


def check_sha(value):
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ValueError("invalid SHA256")


def bbox_valid(box, width, height):
    return (
        all(math.isfinite(x) for x in box)
        and box[2] > box[0]
        and box[3] > box[1]
        and min(box[2], width) > max(box[0], 0)
        and min(box[3], height) > max(box[1], 0)
    )


def xyxy_to_xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def validate_projection_record(row):
    reject_privileged(row)
    if set(row) != FIELDS or row["schema"] != PROJECTION:
        raise ValueError("projection exact whitelist mismatch")
    relative_media(row["media_path"], row["split"])
    check_sha(row["media_key"])
    check_sha(row["media_sha256"])
    for key in ("length", "width", "height"):
        if type(row[key]) is not int or row[key] <= 0:
            raise ValueError("positive integer dimensions/length required")
    if not isinstance(row["persons"], list) or not row["persons"]:
        raise ValueError("all supplied identities required")
    for slot, person in enumerate(row["persons"]):
        if (
            set(person) != PERSON_FIELDS
            or type(person["slot"]) is not int
            or person["slot"] != slot
        ):
            raise ValueError("fixed contiguous slots required")
        if (
            len(person["bbox_xyxy"]) != row["length"]
            or len(person["bbox_present"]) != row["length"]
        ):
            raise ValueError("bbox length mismatch; no padding/trimming")
        for box, present in zip(person["bbox_xyxy"], person["bbox_present"], strict=True):
            if (
                type(present) is not bool
                or not isinstance(box, list)
                or len(box) != 4
                or not all(type(x) in (int, float) and math.isfinite(x) for x in box)
            ):
                raise ValueError("finite bbox and explicit boolean support required")
            if not present and any(box):
                raise ValueError("absent bbox payload must be zero")
    return row


def census(records, expected=EXPECTED):
    actual = {s: {"videos": 0, "persons": 0} for s in ("train", "val")}
    keys, paths = set(), set()
    for row in records:
        validate_projection_record(row)
        if row["media_key"] in keys or row["media_path"] in paths:
            raise ValueError("duplicate routing identity/path")
        keys.add(row["media_key"])
        paths.add(row["media_path"])
        actual[row["split"]]["videos"] += 1
        actual[row["split"]]["persons"] += len(row["persons"])
    if actual != expected:
        raise ValueError(f"full census mismatch: {actual} != {expected}")
    return actual


def project_bbox_only(
    media_manifest, annotation_root, splits=("train", "val"), *, expected=EXPECTED
):
    if tuple(splits) != ("train", "val"):
        raise ValueError("complete train/val only")
    root = Path(annotation_root).resolve(strict=True)
    records, provenance, seen = [], [], set()
    for item in read_jsonl(media_manifest):
        rel = relative_media(item["path"], item["split"])
        check_sha(item.get("sha256"))
        if item.get("status") != "hashed" or rel in seen:
            raise ValueError("unhashed/duplicate media row")
        seen.add(rel)
        media = (root / rel).resolve(strict=True)
        if not media.is_relative_to(root / item["split"]):
            raise ValueError("source path escapes split")
        annotation = media.with_suffix(".json")
        raw = read_json(annotation)  # Only trusted projector reads source annotations.
        objects = [k for k in raw if k.startswith("object")]
        if any(re.fullmatch(r"object\d+", k) is None for k in objects):
            raise ValueError("unrecognized object identity")
        objects.sort(key=lambda k: int(k[6:]))
        people = []
        for slot, obj in enumerate(objects):
            people.append(
                {
                    "slot": slot,
                    "bbox_xyxy": [[0, 0, 0, 0] if b is None else list(b) for b in raw[obj]["bbox"]],
                    "bbox_present": [b is not None for b in raw[obj]["bbox"]],
                }
            )
        row = dict(
            schema=PROJECTION,
            split=item["split"],
            media_key=hashlib.sha256(rel.encode()).hexdigest(),
            media_path=rel,
            media_sha256=item["sha256"],
            width=raw["width"],
            height=raw["height"],
            length=raw["length"],
            persons=people,
        )
        validate_projection_record(row)
        records.append(row)
        provenance.append(
            {
                "media_key": row["media_key"],
                "annotation_sha256": digest_file(annotation),
                "source_object_to_slot": {obj: slot for slot, obj in enumerate(objects)},
            }
        )
    census(records, expected)
    aliases = {}
    for row in records:
        aliases.setdefault(row["media_sha256"], []).append(row["media_key"])
    return records, {
        "schema": "temporac.bbox-provenance.v1",
        "records": provenance,
        "same_media_aliases": [v for v in aliases.values() if len(v) > 1],
    }


def decode_records(audit_root):
    if audit_root is None:
        return {}
    path = Path(audit_root)
    if path.is_dir():
        path = path / "decode.jsonl"
    return {relative_media(r["path"], r["split"]): r for r in read_jsonl(path, running=True)}


def validate_projection(projection, decode_audit=None, *, expected=EXPECTED):
    population = census(projection, expected)
    audit = decode_records(decode_audit)
    status = dict(available=0, pending=0, alignment_unresolved=0)
    for row in projection:
        item = audit.get(row["media_path"])
        if item is None or item.get("status") != "decoded":
            status["pending"] += 1
            continue
        if item.get("prior_sha256") != row["media_sha256"]:
            raise ValueError("decode audit SHA mismatch")
        status["available"] += 1
        status["alignment_unresolved"] += item["decode"]["decoded_frames"] != row["length"]
    return {"population": population, "decode_audit": status, "training_bridge": "not_implemented"}


def verify_assets(snapshot):
    actual = {name: digest_file(Path(snapshot) / name) for name in ASSETS}
    if actual != ASSETS:
        raise ValueError("offline asset SHA mismatch")
    return actual


def empty_raw(row):
    import numpy as np

    p, n = len(row["persons"]), row["length"]
    boxes = np.asarray([x["bbox_xyxy"] for x in row["persons"]], dtype=np.float64)
    present = np.asarray([x["bbox_present"] for x in row["persons"]], dtype=bool)
    valid = np.asarray(
        [
            [
                present[i, f] and bbox_valid(boxes[i, f], row["width"], row["height"])
                for f in range(n)
            ]
            for i in range(p)
        ],
        dtype=bool,
    )
    reason = np.full((p, n), 6, dtype=np.uint8)
    reason[~present], reason[present & ~valid] = 1, 2
    return dict(
        slot=np.arange(p, dtype=np.int64),
        bbox_xyxy=boxes,
        bbox_present=present,
        bbox_valid=valid,
        xyconf_raw=np.zeros((p, n, 17, 3), dtype=np.float32),
        pose_returned=np.zeros((p, n), dtype=bool),
        joint_observed=np.zeros((p, n, 17), dtype=bool),
        joint_invalid=np.zeros((p, n, 17), dtype=bool),
        reason=reason,
    )


def assign_pose(arrays, frame, slot, pose):
    import numpy as np

    labels, xy, scores = (np.asarray(pose[k]) for k in ("labels", "keypoints", "scores"))
    if (
        labels.shape != (17,)
        or not np.array_equal(labels, np.arange(17))
        or xy.shape != (17, 2)
        or scores.shape != (17,)
    ):
        raise ValueError("expected exact ordered COCO0..16; no thresholding")
    value = np.concatenate((xy, scores[:, None]), axis=1).astype(np.float32)
    finite = np.isfinite(value).all(axis=1)
    value[~finite] = 0
    arrays["xyconf_raw"][slot, frame] = value
    arrays["pose_returned"][slot, frame] = True
    arrays["joint_observed"][slot, frame] = finite
    arrays["joint_invalid"][slot, frame] = ~finite
    arrays["reason"][slot, frame] = 0


def attach_clock(arrays, row, d, audit_item=None, audit_root=None):
    import numpy as np

    arrays.update(
        source_frame_index=np.arange(d, dtype=np.int64),
        decoded_ok=np.ones(d, dtype=bool),
        pts=np.zeros(d, dtype=np.int64),
        pts_present=np.zeros(d, dtype=bool),
        best_effort_timestamp=np.zeros(d, dtype=np.int64),
        best_effort_present=np.zeros(d, dtype=bool),
    )
    meta = dict(
        clock_unit="source_frame",
        clock_status="source_frame_only",
        time_base_num=None,
        time_base_den=None,
        decode_audit_status="unavailable",
        clock_sidecar_status="unavailable",
    )
    if audit_item is not None:
        if audit_item.get("prior_sha256") != row["media_sha256"]:
            raise ValueError("decode audit SHA mismatch")
        meta["decode_audit_status"] = audit_item.get("status", "unknown")
        if audit_item.get("status") == "decoded":
            if audit_item["decode"]["decoded_frames"] != d:
                meta["clock_sidecar_status"] = "decoder_count_disagreement"
            else:
                base = Path(audit_root)
                if base.is_file():
                    base = base.parent
                rel = PurePosixPath(audit_item["frame_clock_csv"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("clock path escapes audit root")
                path = base / rel
                if path.is_file():
                    with path.open(newline="", encoding="utf-8") as stream:
                        rows = list(csv.DictReader(stream))
                    if len(rows) != d or [int(x["frame_index"]) for x in rows] != list(range(d)):
                        raise ValueError("clock CSV must retain exact native frame order")
                    for field, mask in (
                        ("pts", "pts_present"),
                        ("best_effort_timestamp", "best_effort_present"),
                    ):
                        for i, item in enumerate(rows):
                            raw = item.get(field)
                            if raw not in (None, "", "N/A"):
                                arrays[field][i], arrays[mask][i] = int(raw), True
                    meta.update(
                        clock_sidecar_status="loaded",
                        clock_sidecar_sha256=digest_file(path),
                        stream=audit_item.get("stream", {}),
                    )
                    try:
                        tb = Fraction(meta["stream"]["time_base"])
                        if tb <= 0:
                            raise ValueError("nonpositive time_base")
                        meta.update(time_base_num=tb.numerator, time_base_den=tb.denominator)
                        if arrays["pts_present"].any():
                            meta["clock_status"] = "pts_preserved"
                    except (KeyError, ValueError, ZeroDivisionError):
                        pass
    mismatch = d != row["length"] or meta["clock_sidecar_status"] == "decoder_count_disagreement"
    if mismatch:
        invalidate_alignment(arrays)
    return meta, mismatch


def invalidate_alignment(arrays):
    for key in ("xyconf_raw", "joint_observed", "joint_invalid", "pose_returned"):
        arrays[key][:] = 0
    arrays["reason"][:] = 4


class ViTPoseFrontend:
    def __init__(self, snapshot, device):
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        torch.set_num_threads(1)
        self.torch, self.device = torch, torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            str(snapshot), local_files_only=True, use_fast=False
        )
        self.model = (
            VitPoseForPoseEstimation.from_pretrained(
                str(snapshot), local_files_only=True, torch_dtype=torch.float32
            )
            .float()
            .eval()
            .to(self.device)
        )
        self.environment = dict(
            torch=torch.__version__,
            python=platform.python_version(),
            device=str(self.device),
            precision="float32",
            autocast=False,
        )

    def __call__(self, images, boxes):
        import numpy as np

        grouped = [np.asarray([xyxy_to_xywh(box)], dtype=np.float32) for box in boxes]
        inputs = self.processor(images, boxes=grouped, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_pose_estimation(outputs, boxes=grouped, threshold=None)
        if len(result) != len(images) or any(len(x) != 1 for x in result):
            raise ValueError("frontend crop ownership mismatch")
        return [
            {k: x[0][k].detach().cpu().numpy() for k in ("labels", "keypoints", "scores")}
            for x in result
        ]


def opencv_frames(path, info):
    import cv2

    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("OpenCV failed to open media")
    info.update(
        name="opencv",
        version=cv2.__version__,
        backend=cap.getBackendName(),
        fps_metadata=cap.get(cv2.CAP_PROP_FPS),
        threads=1,
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def extract_video(row, frames, frontend, batch_size, audit_item=None, audit_root=None):
    arrays = empty_raw(row)
    errors, images, boxes, owners = [], [], [], []
    d, dimensions_ok = 0, True

    def flush():
        if not images:
            return
        try:
            result = frontend(images, boxes)
            if len(result) != len(owners):
                raise ValueError("batch output count mismatch")
            for (frame, slot), pose in zip(owners, result, strict=True):
                assign_pose(arrays, frame, slot, pose)
        except Exception as exc:
            for frame, slot in owners:
                for key in ("xyconf_raw", "joint_observed", "joint_invalid", "pose_returned"):
                    arrays[key][slot, frame] = 0
                arrays["reason"][slot, frame] = 3
            errors.append(
                dict(
                    stage="frontend",
                    ownership=[list(x) for x in owners],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        images.clear()
        boxes.clear()
        owners.clear()

    try:
        for frame_index, image in enumerate(frames):
            d += 1
            if image.shape[:2] != (row["height"], row["width"]):
                dimensions_ok = False
                continue
            if frame_index >= row["length"]:
                continue
            for slot in range(len(row["persons"])):
                if arrays["bbox_valid"][slot, frame_index]:
                    images.append(image)
                    boxes.append(arrays["bbox_xyxy"][slot, frame_index].tolist())
                    owners.append((frame_index, slot))
                    if len(images) == batch_size:
                        flush()
        flush()
    except Exception as exc:
        errors.append(dict(stage="decode", error=f"{type(exc).__name__}: {exc}"))
        flush()
    clocks, mismatch = attach_clock(arrays, row, d, audit_item, audit_root)
    if not dimensions_ok:
        mismatch = True
        invalidate_alignment(arrays)
    status = "alignment_unresolved" if mismatch else ("failed" if errors or d == 0 else "complete")
    slots = [
        dict(
            slot=i,
            status="complete"
            if status == "complete" and arrays["pose_returned"][i].any()
            else "failed",
            pose_frames=int(arrays["pose_returned"][i].sum()),
        )
        for i in range(len(row["persons"]))
    ]
    return arrays, dict(
        schema=SCHEMA,
        media_key=row["media_key"],
        split=row["split"],
        media_sha256=row["media_sha256"],
        width=row["width"],
        height=row["height"],
        annotation_length=row["length"],
        decoded_length=d,
        status=status,
        slots=slots,
        clocks=clocks,
        reason_codes=REASONS,
        errors=errors,
        annotation_axis="L expected bbox/pose rows",
        decode_axis="D decoded clock rows",
        supervision="externally supervised ViTPose + supplied GT bbox/identity assisted; count-label-free frontend",
        joint_observed_meaning="finite xy/score returned, not visibility",
        training_bridge="not_implemented",
    )


def audit_bridge(arrays, meta):
    reasons = ["versioned_320_bridge_not_implemented"]
    if meta["annotation_length"] != 320:
        reasons.append("native_length_not_320")
    if meta["status"] != "complete":
        reasons.append("raw_not_complete")
    if not arrays["joint_observed"].all():
        reasons.append("missing_joint_support")
    return dict(admissible=False, reasons=reasons, scope="audit only; frozen learner unchanged")


def validate_raw(arrays, meta, row=None):
    import numpy as np

    p, l, d = len(meta["slots"]), meta["annotation_length"], meta["decoded_length"]  # noqa: E741
    specs = {
        "slot": ((p,), np.int64),
        "bbox_xyxy": ((p, l, 4), np.float64),
        "xyconf_raw": ((p, l, 17, 3), np.float32),
        "bbox_present": ((p, l), np.bool_),
        "bbox_valid": ((p, l), np.bool_),
        "pose_returned": ((p, l), np.bool_),
        "joint_observed": ((p, l, 17), np.bool_),
        "joint_invalid": ((p, l, 17), np.bool_),
        "reason": ((p, l), np.uint8),
        "source_frame_index": ((d,), np.int64),
        "decoded_ok": ((d,), np.bool_),
        "pts": ((d,), np.int64),
        "pts_present": ((d,), np.bool_),
        "best_effort_timestamp": ((d,), np.int64),
        "best_effort_present": ((d,), np.bool_),
    }
    if meta.get("schema") != SCHEMA or set(arrays) != set(specs):
        raise ValueError("raw schema/member mismatch")
    for key, (shape, dtype) in specs.items():
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError("raw shape/dtype mismatch: " + key)
    if not np.array_equal(arrays["slot"], np.arange(p)) or not np.array_equal(
        arrays["source_frame_index"], np.arange(d)
    ):
        raise ValueError("identity/frame order mismatch")
    observed = arrays["joint_observed"]
    if not np.isfinite(arrays["xyconf_raw"]).all() or not np.isfinite(arrays["bbox_xyxy"]).all():
        raise ValueError("nonfinite numeric payload")
    if np.any(arrays["xyconf_raw"][~observed] != 0) or np.any(observed & arrays["joint_invalid"]):
        raise ValueError("invalid joint payload/mask mismatch")
    if np.any(observed & ~arrays["pose_returned"][..., None]) or np.any(
        arrays["pose_returned"] & ~arrays["bbox_valid"]
    ):
        raise ValueError("ownership/support violation")
    if l != d and (
        meta["status"] != "alignment_unresolved" or observed.any() or arrays["pose_returned"].any()
    ):
        raise ValueError("unresolved alignment must have no observed pose")
    if row is not None:
        validate_projection_record(row)
        if (meta["media_key"], meta["media_sha256"], p, l) != (
            row["media_key"],
            row["media_sha256"],
            len(row["persons"]),
            row["length"],
        ):
            raise ValueError("raw/projection identity mismatch")
        if not np.array_equal(
            arrays["bbox_xyxy"],
            np.asarray([x["bbox_xyxy"] for x in row["persons"]], dtype=np.float64),
        ):
            raise ValueError("raw bbox altered")
    return dict(
        status=meta["status"],
        persons=p,
        annotation_length=l,
        decoded_length=d,
        bridge=audit_bridge(arrays, meta),
    )


def source_commit():
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    return p.stdout.strip() if p.returncode == 0 else "unavailable"


def binding_for(row, cfg, assets, audit):
    return dict(
        projection_sha256=digest_json(row),
        media_sha256=row["media_sha256"],
        assets=assets,
        effective_config=dict(
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            precision="float32",
            threads=1,
            processor_use_fast=False,
            postprocess_threshold=None,
            interpolation="none",
        ),
        decode_audit_sha256=None if audit is None else digest_json(audit),
        script_sha256=digest_file(__file__),
    )


def save_raw(base, arrays, meta):
    import numpy as np

    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    target = base.with_suffix(".npz")
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, target)
    meta["npz_sha256"] = digest_file(target)
    write_json(base.with_suffix(".json"), meta)


def load_raw(base):
    import numpy as np

    base = Path(base)
    meta = read_json(base.with_suffix(".json"))
    if meta["npz_sha256"] != digest_file(base.with_suffix(".npz")):
        raise ValueError("raw NPZ hash mismatch")
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return arrays, meta


def extract_native(
    projection, decode_audit, snapshot, output, device, batch_size, *, source_root, resume=False
):
    if (
        type(batch_size) is not int
        or batch_size < 1
        or (device != "cpu" and re.fullmatch(r"cuda:\d+", device) is None)
    ):
        raise ValueError("positive batch_size and explicit cpu/cuda:N required")
    if os.environ.get("OMP_NUM_THREADS") != "1" or os.environ.get("MKL_NUM_THREADS") != "1":
        raise ValueError("explicit OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 required")
    root, output = Path(source_root).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("output must be outside source tree")
    ledger = [
        dict(
            media_key=r["media_key"],
            split=r["split"],
            status="pending",
            slots=[dict(slot=p["slot"], status="pending") for p in r["persons"]],
        )
        for r in projection
    ]
    write_json(output / "population_ledger.json", dict(schema=SCHEMA, records=ledger))
    assets, audit = verify_assets(snapshot), decode_records(decode_audit)
    frontend, summaries = None, []
    initialization_error = None
    for video_index, row in enumerate(projection):
        validate_projection_record(row)
        base = output / "raw" / row["split"] / row["media_key"]
        item = audit.get(row["media_path"])
        binding = binding_for(row, dict(batch_size=batch_size, device=device), assets, item)
        media = (root / row["media_path"]).resolve(strict=True)
        if (
            not media.is_relative_to(root / row["split"])
            or digest_file(media) != row["media_sha256"]
        ):
            raise ValueError("media path/SHA changed since projection")
        if resume and base.with_suffix(".json").is_file():
            arrays, meta = load_raw(base)
            validate_raw(arrays, meta, row)
            if meta["status"] == "complete":
                if meta["binding"] != binding:
                    raise ValueError("resume artifact binding mismatch")
                summaries.append(
                    dict(
                        media_key=row["media_key"],
                        status="reused_complete",
                        persons=len(row["persons"]),
                    )
                )
                ledger[video_index].update(status=meta["status"], slots=meta["slots"])
                write_json(output / "population_ledger.json", dict(schema=SCHEMA, records=ledger))
                continue
        started = time.monotonic()
        decoder = {}
        try:
            if initialization_error is not None:
                raise RuntimeError(initialization_error)
            if frontend is None:
                try:
                    frontend = ViTPoseFrontend(snapshot, device)
                except Exception as exc:
                    initialization_error = f"{type(exc).__name__}: {exc}"
                    raise
            arrays, meta = extract_video(
                row, opencv_frames(media, decoder), frontend, batch_size, item, decode_audit
            )
        except Exception as exc:
            # Preserve every slot even if the decoder/model cannot be entered.
            arrays = empty_raw(row)
            clocks, _ = attach_clock(arrays, row, 0)
            arrays["reason"][:] = 5
            meta = dict(
                schema=SCHEMA,
                media_key=row["media_key"],
                split=row["split"],
                media_sha256=row["media_sha256"],
                width=row["width"],
                height=row["height"],
                annotation_length=row["length"],
                decoded_length=0,
                status="alignment_unresolved",
                slots=[
                    dict(slot=p["slot"], status="failed", pose_frames=0) for p in row["persons"]
                ],
                clocks=clocks,
                reason_codes=REASONS,
                errors=[
                    dict(stage="video_preflight_or_clock", error=f"{type(exc).__name__}: {exc}")
                ],
                decode_axis="zero rows observed by this failed attempt; not a claim that media has zero frames",
                training_bridge="not_implemented",
            )
        meta.update(
            binding=binding,
            decoder=decoder,
            environment=frontend.environment
            if frontend
            else dict(device=device, initialization="failed"),
            source_commit=source_commit(),
            command=sys.argv,
            elapsed_seconds=time.monotonic() - started,
        )
        validate_raw(arrays, meta, row)
        write_json(output / "attempts" / row["media_key"] / (str(time.time_ns()) + ".json"), meta)
        save_raw(base, arrays, meta)
        summaries.append(
            dict(media_key=row["media_key"], status=meta["status"], persons=len(row["persons"]))
        )
        ledger[video_index].update(status=meta["status"], slots=meta["slots"])
        write_json(output / "population_ledger.json", dict(schema=SCHEMA, records=ledger))
        write_json(
            output / "extraction_progress.json",
            dict(schema=SCHEMA, completed_records=len(summaries), records=summaries),
        )
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["plan", "project", "validate", "extract", "validate_raw"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    out = Path(cfg["output"])
    if args.command == "plan":
        write_json(
            out / "plan.json",
            dict(
                schema=SCHEMA,
                status="prepared_not_extracted",
                expected=EXPECTED,
                training_bridge="not_implemented",
            ),
        )
        return 0
    source = Path(cfg["source_root"]).resolve(strict=True)
    if out.resolve().is_relative_to(source):
        raise ValueError("output must stay outside source tree")
    path = out / "projection.jsonl"
    if args.command == "project":
        rows, provenance = project_bbox_only(cfg["media_manifest"], source)
        out.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        os.replace(tmp, path)
        provenance.update(
            media_manifest_sha256=digest_file(cfg["media_manifest"]),
            projection_sha256=digest_file(path),
            script_sha256=digest_file(__file__),
            source_commit=source_commit(),
        )
        write_json(out / "projection_provenance.json", provenance)
        report = validate_projection(rows, cfg.get("decode_audit"))
        write_json(out / "projection_validation.json", report)
    else:
        rows = read_jsonl(path)
        if read_json(out / "projection_provenance.json")["projection_sha256"] != digest_file(path):
            raise ValueError("projection hash mismatch")
        report = validate_projection(rows, cfg.get("decode_audit"))
        if args.command == "validate":
            write_json(out / "projection_validation.json", report)
        elif args.command == "extract":
            if args.device is None:
                parser.error("extract requires explicit --device cpu or cuda:N")
            report = extract_native(
                rows,
                cfg.get("decode_audit"),
                cfg["snapshot"],
                out,
                args.device,
                cfg["batch_size"],
                source_root=source,
                resume=args.resume,
            )
        else:
            report = []
            for row in rows:
                arrays, meta = load_raw(out / "raw" / row["split"] / row["media_key"])
                report.append(validate_raw(arrays, meta, row))
            write_json(
                out / "raw_validation.json",
                dict(videos=len(report), persons=sum(x["persons"] for x in report), records=report),
            )
    print(
        json.dumps(report if isinstance(report, dict) else dict(videos=len(report)), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
