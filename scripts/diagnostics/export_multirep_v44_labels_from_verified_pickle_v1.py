"""One-purpose evaluator label export from the hash-frozen v44 validation pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_VAL_SHA256 = "4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251"
_CACHE_NAME = re.compile(r"v(?P<video>\d+)-p(?P<person>\d+)\.npz\Z")


def _read_verified(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError("source must be a regular non-symlink file")
    payload = path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != EXPECTED_VAL_SHA256:
        raise ValueError(f"unexpected v44 validation SHA-256: {observed}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")

    cache_identities: set[tuple[int, int]] = set()
    for path in args.cache_dir.glob("v*-p*.npz"):
        match = _CACHE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected cache filename: {path.name}")
        cache_identities.add((int(match.group("video")), int(match.group("person"))))

    # Explicitly authorized, hash-frozen, one-purpose trusted boundary.  No object from
    # the decoded payload is retained after scalar label extraction.
    decoded = pickle.loads(_read_verified(args.source))  # noqa: S301
    if not isinstance(decoded, dict) or set(decoded) != {"val"}:
        raise TypeError("v44 validation pickle must contain the sole key 'val'")
    samples = decoded["val"]
    if not isinstance(samples, list) or len(samples) != 51:
        raise TypeError("v44 validation payload must contain exactly 51 video rows")

    labels: list[dict[str, int]] = []
    for video_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError("each validation video must be a mapping")
        person_mask = np.asarray(sample.get("person_mask"))
        count_gt = np.asarray(sample.get("count_gt"))
        if (
            person_mask.ndim != 1
            or person_mask.dtype != np.dtype(np.bool_)
            or count_gt.shape != person_mask.shape
            or count_gt.dtype.kind != "f"
        ):
            raise TypeError("person_mask/count_gt schema differs from audited v44")
        for person_index in np.flatnonzero(person_mask).tolist():
            identity = (video_index, int(person_index))
            raw_count = float(count_gt[person_index])
            count = int(raw_count)
            if raw_count != float(count) or count <= 0:
                raise ValueError("every valid v44 identity requires a positive integer count")
            labels.append(
                {
                    "video_index": video_index,
                    "person_index": int(person_index),
                    "count_gt": count,
                }
            )
    observed = {(row["video_index"], row["person_index"]) for row in labels}
    if observed != cache_identities:
        raise ValueError("verified pickle identity population differs from safe NPZ cache")
    output: dict[str, Any] = {
        "schema": "temporac.v44-verified-pickle-label-export.v1",
        "scientific_status": "EVALUATOR_ONLY_SUPPLIED_TRACK_PARTIAL_CACHE_DIAGNOSTIC",
        "source_sha256": EXPECTED_VAL_SHA256,
        "video_count": len(samples),
        "identity_count": len(labels),
        "labels": labels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "labels"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
