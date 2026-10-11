"""Export v44 validation count labels from JSON without loading executable pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_CACHE_NAME = re.compile(r"v(?P<video>\d+)-p(?P<person>\d+)\.npz\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")

    cache_identities: set[tuple[int, int]] = set()
    for path in args.cache_dir.glob("v*-p*.npz"):
        match = _CACHE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected cache filename: {path.name}")
        cache_identities.add((int(match.group("video")), int(match.group("person"))))
    rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    video_index = 0
    for sample_dir in sorted(path for path in args.split_dir.iterdir() if path.is_dir()):
        annotation_path = sample_dir / f"{sample_dir.name}.json"
        alphapose_path = sample_dir / "alphapose" / "alphapose-results.json"
        if not annotation_path.is_file() or not alphapose_path.is_file():
            continue
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(annotation, dict):
            raise TypeError(f"annotation must be an object: {annotation_path}")
        object_keys = sorted(key for key in annotation if key.startswith("object"))
        for person_index, object_key in enumerate(object_keys):
            identity = (video_index, person_index)
            if identity not in cache_identities:
                continue
            payload = annotation[object_key]
            if not isinstance(payload, dict):
                raise TypeError(f"annotation object must be a mapping: {annotation_path}")
            raw_count = payload.get("count")
            if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                raise TypeError(f"count must be numeric: {annotation_path}:{object_key}")
            count = int(raw_count)
            if float(raw_count) != float(count) or count <= 0:
                raise ValueError(
                    f"count must be a positive integer: {annotation_path}:{object_key}"
                )
            rows.append(
                {
                    "video_index": video_index,
                    "person_index": person_index,
                    "count_gt": count,
                }
            )
        source_receipts.append(
            {
                "video_index": video_index,
                "sample": sample_dir.name,
                "annotation_sha256": _sha256(annotation_path),
                "alphapose_sha256": _sha256(alphapose_path),
            }
        )
        video_index += 1

    observed = {(row["video_index"], row["person_index"]) for row in rows}
    missing = sorted(cache_identities - observed)
    extra = sorted(observed - cache_identities)
    if (missing or extra) and not args.allow_missing:
        affected_videos = {video for video, _ in missing + extra}
        sources = [
            receipt
            for receipt in source_receipts
            if receipt["video_index"] in affected_videos
        ]
        raise ValueError(
            "JSON/cache identity mismatch: "
            f"missing={missing[:5]} extra={extra[:5]} sources={sources}"
        )
    output = {
        "schema": "temporac.v44-json-label-export.v1",
        "scientific_status": (
            "EVALUATOR_ONLY_PARTIAL_IDENTITY_DIAGNOSTIC"
            if missing or extra
            else "EVALUATOR_ONLY_SUPPLIED_TRACK_PARTIAL_CACHE_DIAGNOSTIC"
        ),
        "identity_count": len(rows),
        "video_count": video_index,
        "missing_cache_identities": [list(identity) for identity in missing],
        "extra_json_identities": [list(identity) for identity in extra],
        "labels": rows,
        "source_receipts": source_receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        key: value
        for key, value in output.items()
        if key not in {"labels", "source_receipts"}
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
