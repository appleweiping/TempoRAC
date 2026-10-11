"""Train-only, count-blind sweep of the natural return-RMS threshold.

This diagnostic does not alter the frozen v3 implementation.  It temporarily
binds the module-level threshold while evaluating the same landmark routine,
then restores the original value before exit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from smoke_temporac_pseudo_targets_v7 import _feature

from pams.temporac import full_chain_landmarks_candidate_v3 as landmarks_v3
from pams.temporac.campaign_runtime_candidate_v1 import adapted_feature_receipt
from pams.temporac.preprocess import preprocess_identity
from pams.warp_phase.packing import trusted_load_source_pickle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=(0.075, 0.08, 0.085, 0.09, 0.10),
    )
    args = parser.parse_args()
    source_path = (
        args.repository
        / "data/counting_multirep_skeleton_pose_expanded_v44_len320"
        / f"{args.split}.pkl"
    )
    decoded, source_receipt = trusted_load_source_pickle(
        source_path, repository_root=args.repository, split=args.split
    )
    prepared_rows: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for video_index, sample in enumerate(decoded[args.split]):
        for person_index in range(len(sample["person_mask"])):
            if not bool(sample["person_mask"][person_index]):
                continue
            feature, receipt, digest = _feature(
                sample,
                split=args.split,
                source_sha256=source_receipt.sha256,
                video_index=video_index,
                person_index=person_index,
            )
            adapted, _receipt, _sha, _resampling = adapted_feature_receipt(
                feature,
                receipt,
                expected_source_receipt_sha256=digest,
            )
            prepared = preprocess_identity(
                adapted.motion,
                adapted.frame_mask,
                adapted.sampled_frame_indices,
            )
            bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)
            bounds[:, 1] += 1
            prepared_rows.append(
                (video_index, person_index, prepared.geometry_features.geometry, bounds)
            )

    original = landmarks_v3.MAX_RETURN_RMS
    results: list[dict[str, object]] = []
    try:
        for threshold in args.thresholds:
            landmarks_v3.MAX_RETURN_RMS = threshold
            available: list[list[int]] = []
            periods: list[int] = []
            for video_index, person_index, geometry, bounds in prepared_rows:
                result = landmarks_v3.full_chain_integer_landmarks_v3(geometry, bounds)
                if result.landmarks.size >= 3:
                    available.append([video_index, person_index])
                    if result.estimated_period is not None:
                        periods.append(result.estimated_period)
            results.append(
                {
                    "available": len(available),
                    "available_identities": available,
                    "maximum_period": max(periods, default=None),
                    "minimum_period": min(periods, default=None),
                    "threshold": threshold,
                }
            )
    finally:
        landmarks_v3.MAX_RETURN_RMS = original

    print(
        json.dumps(
            {
                "identities": len(prepared_rows),
                "results": results,
                "schema": "temporac.return-threshold-sweep.v18",
                "split": args.split,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
