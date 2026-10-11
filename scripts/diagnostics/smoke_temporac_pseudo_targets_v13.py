"""Run the frozen v17 cascade with a diagnostic return-RMS threshold."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import smoke_temporac_pseudo_targets_v12 as frozen_smoke

from pams.temporac import full_chain_landmarks_candidate_v3 as landmarks_v3


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--return-threshold", required=True, type=float)
    args, remaining = parser.parse_known_args()
    if not 0.075 <= args.return_threshold <= 0.10:
        parser.error("--return-threshold must be in [0.075, 0.10]")
    original = landmarks_v3.MAX_RETURN_RMS
    original_infer = frozen_smoke.infer_natural_pseudo_target_v17
    threshold = args.return_threshold

    def infer_with_frozen_fallback(*args: Any, **kwargs: Any) -> Any:
        landmarks_v3.MAX_RETURN_RMS = original
        established = original_infer(*args, **kwargs)
        if established.target.certified:
            return established
        landmarks_v3.MAX_RETURN_RMS = threshold
        return original_infer(*args, **kwargs)

    try:
        frozen_smoke.infer_natural_pseudo_target_v17 = infer_with_frozen_fallback
        sys.argv = [sys.argv[0], *remaining]
        return frozen_smoke.main()
    finally:
        frozen_smoke.infer_natural_pseudo_target_v17 = original_infer
        landmarks_v3.MAX_RETURN_RMS = original


if __name__ == "__main__":
    raise SystemExit(main())
