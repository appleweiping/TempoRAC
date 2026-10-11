"""Diagnostic v8 certificate with the v3 landmark selector.

The candidate clones the already reviewed v7 certificate bytecode into an
isolated globals mapping and replaces only its natural landmark callable.  It
does not mutate v7 and preserves the frozen X0 path exactly.  A self-contained
source copy is still required before normative promotion.
"""

from __future__ import annotations

from types import FunctionType
from typing import cast

import pams.temporac.certify as frozen
from pams.temporac.certify_target_adaptive_v7 import (
    _certify_once_adaptive_v7,
    certify_target_adaptive_v7,
)
from pams.temporac.full_chain_landmarks_candidate_v3 import (
    full_chain_integer_landmarks_v3,
)

_once_globals = dict(_certify_once_adaptive_v7.__globals__)
_once_globals["full_chain_integer_landmarks_v2"] = full_chain_integer_landmarks_v3
_certify_once_adaptive_v8 = cast(
    object,
    FunctionType(
        _certify_once_adaptive_v7.__code__,
        _once_globals,
        "_certify_once_adaptive_v8",
        _certify_once_adaptive_v7.__defaults__,
        _certify_once_adaptive_v7.__closure__,
    ),
)

_certificate_globals = dict(certify_target_adaptive_v7.__globals__)
_certificate_globals["_certify_once_adaptive_v7"] = _certify_once_adaptive_v8
_certify_target_adaptive_v8_impl = FunctionType(
    certify_target_adaptive_v7.__code__,
    _certificate_globals,
    "_certify_target_adaptive_v8_impl",
    certify_target_adaptive_v7.__defaults__,
    certify_target_adaptive_v7.__closure__,
)


def certify_target_adaptive_v8(
    track: frozen.CertificateTrack,
    source_kind: frozen.SourceKind,
) -> frozen.CertifiedTarget:
    """Run the v7 certificate control flow with v3 natural landmarks."""

    return cast(frozen.CertifiedTarget, _certify_target_adaptive_v8_impl(track, source_kind))


__all__ = ["certify_target_adaptive_v8"]
