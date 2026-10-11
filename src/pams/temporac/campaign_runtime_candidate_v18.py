"""v17 runtime with a conservative train-selected landmark fallback."""

from __future__ import annotations

import torch

from pams.temporac.campaign_runtime_candidate_v1 import NaturalPseudoTarget
from pams.temporac.campaign_runtime_candidate_v17 import infer_natural_pseudo_target_v17
from pams.temporac.campaign_runtime_candidate_v18_landmarks import (
    infer_natural_pseudo_target_v18_landmarks,
)
from pams.temporac.teacher import TempoRACTeacher
from pams.temporac.types import FeatureRecord


def infer_natural_pseudo_target_v18(
    model: TempoRACTeacher,
    source_feature: FeatureRecord,
    source_receipt_bytes: bytes,
    *,
    expected_source_receipt_sha256: str,
    selected_teacher_sha256: str,
    device: torch.device,
) -> NaturalPseudoTarget:
    """Keep every v17 certification and apply v18 only after abstention."""

    established = infer_natural_pseudo_target_v17(
        model,
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        selected_teacher_sha256=selected_teacher_sha256,
        device=device,
    )
    if established.target.certified:
        return established
    return infer_natural_pseudo_target_v18_landmarks(
        model,
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        selected_teacher_sha256=selected_teacher_sha256,
        device=device,
    )


__all__ = ["infer_natural_pseudo_target_v18"]
