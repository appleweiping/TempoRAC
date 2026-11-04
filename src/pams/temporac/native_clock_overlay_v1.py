"""Immutable true-PTS view; original native pose bytes are never rewritten."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from pams.temporac.native_bridge_v1 import NativeFeatureV1, freeze

SCHEMA = "temporac.native-clock-overlay.v1"
LIMITATION = "prior inventory SHA copied by decode; stable stat only, no decode-time media rehash or pixel-equivalence proof"
ROOT = Path(__file__).resolve().parents[3]
STREAM_CONTRACT = {
    "opencv_version": "4.11.0",
    "opencv_source": "https://github.com/opencv/opencv/blob/4.11.0/modules/videoio/src/cap_ffmpeg_impl.hpp#L1164-L1310",
    "opencv_source_sha256": "f10f6987757d3da421e31cd529f99a4cd2bb955f771e116855644a61cc365354",
    "ffmpeg_stream_specifier": "https://github.com/FFmpeg/FFmpeg/blob/n6.1/doc/fftools-common-opts.texi#L30-L45",
    "selection": "default non-raw OpenCV first video stream equals ffprobe v:0; no unique-video-stream claim",
}


def readonly_nb1():
    spec = importlib.util.spec_from_file_location(
        "nb1_clock_readonly", ROOT / "scripts/build_temporac_native_learner_bridge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def stable_bytes(path):
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(data) != after.st_size:
        raise ValueError("input changed during read: " + str(path))
    return data


def integer(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?\d+", value):
        raise ValueError("invalid integer: " + name)
    result = int(value)
    if not -(2**63) <= result < 2**63:
        raise ValueError("int64 overflow: " + name)
    return result


def rational(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+/\d+", value):
        raise ValueError("invalid rational time_base")
    num, den = (integer(x, "time_base") for x in value.split("/"))
    if num <= 0 or den <= 0:
        raise ValueError("nonpositive time_base")
    return Fraction(num, den)


@dataclass(frozen=True)
class OverlayV1:
    metadata: dict
    source_frame_index: np.ndarray
    pts: np.ndarray
    pts_present: np.ndarray

    def __post_init__(self):
        # Canonical roundtrip detaches metadata from caller-owned dictionaries.
        object.__setattr__(self, "metadata", json.loads(canonical(self.metadata)))
        n = self.metadata["N"]
        for name, dtype in (
            ("source_frame_index", np.int64),
            ("pts", np.int64),
            ("pts_present", np.bool_),
        ):
            a = np.asarray(getattr(self, name))
            if a.shape != (n,) or a.dtype != np.dtype(dtype):
                raise ValueError("overlay dtype/shape mismatch")
            object.__setattr__(self, name, freeze(a))
        if self.metadata["schema"] != SCHEMA or not np.array_equal(
            self.source_frame_index, np.arange(n)
        ):
            raise ValueError("overlay schema/source index mismatch")
        if np.any(self.pts[~self.pts_present] != 0):
            raise ValueError("missing PTS payload must be zero")
        rational(self.metadata["time_base"])

    def arrays(self):
        return {name: getattr(self, name) for name in ("source_frame_index", "pts", "pts_present")}

    @property
    def content_sha256(self):
        return sha(canonical(self.metadata) + b"".join(a.tobytes() for a in self.arrays().values()))


def build_clock_overlay(raw_json, raw_npz, decode_row_bytes, clock_csv) -> OverlayV1:
    """Bind exact input bytes; accept actual PTS defects, reject binding defects."""
    if not isinstance(decode_row_bytes, bytes):
        raise ValueError("decode row must be exact JSONL bytes")
    jb, nb, cb = (stable_bytes(p) for p in (raw_json, raw_npz, clock_csv))
    meta, row = json.loads(jb), json.loads(decode_row_bytes)
    if meta.get("npz_sha256") != sha(nb):
        raise ValueError("raw NPZ hash mismatch")
    with np.load(io.BytesIO(nb), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    readonly_nb1().raw_module().validate_raw(arrays, meta)
    n = meta["annotation_length"]
    if meta["status"] != "complete" or n != meta["decoded_length"] or n <= 0:
        raise ValueError("raw must be complete and L=D=N")
    if meta.get("errors") or not arrays["decoded_ok"].all():
        raise ValueError("raw decode errors")
    if (
        meta.get("decoder", {}).get("backend") != "FFMPEG"
        or meta["decoder"].get("name") != "opencv"
        or meta["decoder"].get("version") != STREAM_CONTRACT["opencv_version"]
        or meta["binding"].get("script_sha256")
        != sha(stable_bytes(ROOT / "scripts/build_temporac_native_pose_cache.py"))
    ):
        raise ValueError("native presentation order unconfirmed")
    if (
        row.get("status") != "decoded"
        or row["decode"]["returncode"] != 0
        or row["decode"]["parse_errors"]
    ):
        raise ValueError("decode failed or parse errors")
    allowed = {
        "pts_missing_or_not_strictly_monotone",
        "best_effort_timestamp_missing_or_not_strictly_monotone",
    }
    if set(row.get("findings", [])) - allowed:
        raise ValueError("decode binding findings")
    stat = row.get("stat_before")
    if (
        not isinstance(stat, dict)
        or set(stat) != {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
        or stat != row.get("stat_after")
        or stat["st_size"] != row["prior_bytes"]
    ):
        raise ValueError("decode media stat unstable")
    media_sha = meta["media_sha256"]
    if (
        row["prior_sha256"] != media_sha
        or meta["binding"]["media_sha256"] != media_sha
        or len(media_sha) != 64
        or any(c not in "0123456789abcdef" for c in media_sha)
    ):
        raise ValueError("media SHA mismatch")
    raw_path = row["path"]
    if (
        row["split"] != meta["split"]
        or Path(raw_path).parts[0] != meta["split"]
        or sha(raw_path.encode()) != meta["media_key"]
    ):
        raise ValueError("decode/raw media key mismatch")
    stream = row["stream"]
    stream_index = stream["index"]
    if type(stream_index) is not int or stream_index < 0:
        raise ValueError("invalid selected stream")
    tb = rational(stream["time_base"])
    annotation = {"length": n, "width": meta["width"], "height": meta["height"]}
    if (
        row["annotation"] != annotation
        or row["decode"]["decoded_frames"] != n
        or row["decode"]["dimensions"] != {f"{meta['width']}x{meta['height']}": n}
        or (stream["width"], stream["height"]) != (meta["width"], meta["height"])
    ):
        raise ValueError("decode/raw dimension or length mismatch")
    old_stream = meta.get("clocks", {}).get("stream")
    if old_stream and (
        old_stream.get("index") != stream_index
        or rational(old_stream["time_base"]) != tb
        or (old_stream.get("width"), old_stream.get("height")) != (meta["width"], meta["height"])
    ):
        raise ValueError("raw/decode selected stream conflict")
    reader = csv.DictReader(io.StringIO(cb.decode("utf-8"), newline=""))
    required = {"frame_index", "stream_index", "width", "height", "pts"}
    if (
        not reader.fieldnames
        or not required <= set(reader.fieldnames)
        or len(set(reader.fieldnames)) != len(reader.fieldnames)
    ):
        raise ValueError("clock CSV missing or duplicate columns")
    records = list(reader)
    if len(records) != n:
        raise ValueError("clock CSV row count mismatch")
    pts, present = np.zeros(n, np.int64), np.zeros(n, bool)
    for i, r in enumerate(records):
        if (
            None in r
            or integer(r["frame_index"], "frame_index") != i
            or integer(r["stream_index"], "stream_index") != stream_index
            or integer(r["width"], "width") != meta["width"]
            or integer(r["height"], "height") != meta["height"]
        ):
            raise ValueError("clock CSV source index/stream/dimension mismatch")
        if r["pts"] not in ("", "N/A"):
            pts[i], present[i] = integer(r["pts"], "pts"), True
    old_present = arrays["pts_present"]
    old_tb = meta.get("clocks", {})
    if old_present.any() and (
        old_tb.get("time_base_num", 0) <= 0
        or old_tb.get("time_base_den", 0) <= 0
        or Fraction(old_tb["time_base_num"], old_tb["time_base_den"]) != tb
        or not present[old_present].all()
        or not np.array_equal(arrays["pts"][old_present], pts[old_present])
    ):
        raise ValueError("original present PTS or rational unit conflict")
    selection = (
        "raw-retained"
        if old_present.all()
        else "overlay-attached"
        if present.any()
        else "unavailable"
    )
    metadata = dict(
        schema=SCHEMA,
        media_key=meta["media_key"],
        split=meta["split"],
        N=n,
        raw_json_sha256=sha(jb),
        raw_npz_sha256=sha(nb),
        decode_row_sha256=sha(decode_row_bytes),
        clock_csv_sha256=sha(cb),
        media_sha256=media_sha,
        stream_index=stream_index,
        time_base=f"{tb.numerator}/{tb.denominator}",
        original_clock=meta.get("clocks", {}),
        original_pts_sha256=sha(arrays["pts"].tobytes()),
        original_pts_present_sha256=sha(old_present.tobytes()),
        selection=selection,
        generator_sha256=sha(stable_bytes(__file__)),
        media_sha_assurance=LIMITATION,
        presentation_order="native continuous OpenCV FFMPEG / audit selected v:0 presentation rows",
        stream_selection_contract=STREAM_CONTRACT,
        raw_generator_sha256=meta["binding"]["script_sha256"],
        decode_generator_sha256=sha(stable_bytes(ROOT / "scripts/audit_temporac_decode.py")),
    )
    for path, original in ((raw_json, jb), (raw_npz, nb), (clock_csv, cb)):
        if stable_bytes(path) != original:
            raise ValueError("clock inputs changed during overlay construction")
    return OverlayV1(metadata, arrays["source_frame_index"], pts, present)


def resolve_clock(feature: NativeFeatureV1, overlay: OverlayV1):
    """Only clock fields change; a binding conflict raises, never falls back."""
    m = overlay.metadata
    if (
        feature.opaque_key.tobytes().hex() != m["media_key"]
        or int(feature.source_length[0]) != m["N"]
        or not np.array_equal(feature.source_frame_index, overlay.source_frame_index)
    ):
        raise ValueError("feature/overlay identity binding mismatch")
    original = sha(feature.pts.tobytes()), sha(feature.pts_present.tobytes())
    final = np.array_equal(feature.pts, overlay.pts) and np.array_equal(
        feature.pts_present, overlay.pts_present
    )
    if original != (m["original_pts_sha256"], m["original_pts_present_sha256"]) and not final:
        raise ValueError("feature original clock binding mismatch")
    tb = rational(m["time_base"])
    if feature.pts_present.any() and Fraction(*map(int, feature.time_base)) != tb:
        raise ValueError("feature rational unit conflict")
    resolved = (
        replace(feature)
        if m["selection"] == "raw-retained"
        else replace(
            feature,
            pts=overlay.pts,
            pts_present=overlay.pts_present,
            time_base=np.array([tb.numerator, tb.denominator], np.int64),
        )
    )
    return resolved, dict(
        selection=m["selection"],
        overlay_content_sha256=overlay.content_sha256,
        original_clock=m["original_clock"],
        original_raw_json_sha256=m["raw_json_sha256"],
        original_raw_npz_sha256=m["raw_npz_sha256"],
        media_sha_assurance=LIMITATION,
    )


def save_overlay(base, overlay):
    helper = readonly_nb1()
    base = Path(base)
    jp, npz = base.with_suffix(".json"), base.with_suffix(".npz")
    if jp.exists():
        loaded = load_overlay(base)
        if loaded.content_sha256 != overlay.content_sha256:
            raise ValueError("immutable overlay changed; choose new output")
        return
    if npz.exists():
        raise ValueError("orphan overlay payload; choose new output")
    helper.atomic_npz(npz, overlay.arrays())
    helper.atomic_json(
        jp,
        dict(
            metadata=overlay.metadata,
            npz_sha256=sha(stable_bytes(npz)),
            content_sha256=overlay.content_sha256,
        ),
    )


def load_overlay(base):
    base = Path(base)
    receipt = json.loads(stable_bytes(base.with_suffix(".json")))
    payload = stable_bytes(base.with_suffix(".npz"))
    if sha(payload) != receipt["npz_sha256"]:
        raise ValueError("overlay payload hash mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {"source_frame_index", "pts", "pts_present"}:
            raise ValueError("overlay payload schema mismatch")
        result = OverlayV1(receipt["metadata"], **{k: archive[k] for k in archive.files})
    if result.content_sha256 != receipt["content_sha256"]:
        raise ValueError("overlay content mismatch")
    return result
