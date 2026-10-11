"""Reconstruct ignored historical TempoRAC operator evidence from frozen reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def describe(label: str, value: Any) -> None:
    payload = canonical(value)
    print(label, len(payload), hashlib.sha256(payload).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", required=True, type=Path)
    args = parser.parse_args()

    v2 = json.loads((args.v2 / "replay_witness.json").read_bytes())
    aggregates = v2["aggregate_roots"]
    aggregates["components"] = aggregates.pop("decoded_components")
    for record in aggregates.values():
        record["order"] = (
            "direct raw C-order row-array concatenation in frozen Cartesian order"
        )
    common = {
        "actual_inactive_b255": v2["actual_inactive_b255"],
        "all_rows_validated": {
            "association_rule": (
                "truth/component adjacency iff half-open edge sets intersect"
            ),
            "decoder_invocations_per_row": 1,
            "flat_amplitude_one_plateaus": 6912,
            "flat_amplitude_one_rows": 2304,
            "nola_rule": (
                "every covering window receives the same desired float32 response"
            ),
            "response_mask_component_nola_decoder_association": True,
            "row_count": 10368,
        },
        "candidate_aggregate_preimages": aggregates,
        "candidate_manifest_bytes": v2["candidate_manifest_bytes"],
        "candidate_manifest_sha256": v2["candidate_manifest_sha256"],
        "candidate_status": "CANDIDATE_PENDING_FRESH_REVIEW",
    }
    describe("witness-min", common)
    optional = {
        "normative_roots": v2["normative_roots"],
        "row_witnesses": v2["row_witnesses"],
        "schema": "temporac.operator-replay-witness.v1",
        "validated_row_count": 10368,
    }
    keys = tuple(optional)
    for mask in range(1 << len(keys)):
        value = dict(common)
        suffix = []
        for index, key in enumerate(keys):
            if mask & (1 << index):
                value[key] = optional[key]
                suffix.append(key)
        payload = canonical(value)
        if 3850 <= len(payload) <= 3950:
            describe("witness+" + ",".join(suffix), value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
