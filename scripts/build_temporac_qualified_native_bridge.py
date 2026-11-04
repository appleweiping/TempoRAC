"""Compose checked NB1 helpers with a frozen clock manifest; never launches training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from pams.temporac.native_clock_overlay_v1 import (  # noqa: E402
    SCHEMA as OVERLAY_SCHEMA,
)
from pams.temporac.native_clock_overlay_v1 import (  # noqa: E402
    canonical,
    load_overlay,
    readonly_nb1,
    resolve_clock,
    sha,
    stable_bytes,
)

NB = readonly_nb1()
EXPECTED = NB.EXPECTED


def source_binding():
    return {
        **NB.source_binding(),
        **{
            name: sha(stable_bytes(ROOT / name))
            for name in (
                "src/pams/temporac/native_clock_overlay_v1.py",
                "scripts/build_temporac_native_clock_overlay.py",
                "scripts/build_temporac_qualified_native_bridge.py",
            )
        },
    }


def same(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and (np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b))
    )


def execute(cfg, command, *, expected=EXPECTED):
    if set(cfg) != {"raw_root", "census", "overlay_manifest", "output"}:
        raise ValueError("config requires raw_root,census,overlay_manifest,output; no caps")
    paths = {k: Path(v).resolve() for k, v in cfg.items()}
    if os.name == "nt" and any(p.drive.upper() != "E:" for p in paths.values()):
        raise ValueError("all project paths must remain on E:")
    out = paths["output"]
    for name, p in paths.items():
        if name != "output" and (out == p or out in p.parents or p in out.parents):
            raise ValueError("qualified output must be separate from inputs")
    if out == paths["overlay_manifest"].parent or paths["overlay_manifest"].parent in out.parents:
        raise ValueError("qualified output must be separate from overlays")
    rows = NB.read_census(paths["census"], expected=expected)
    census_sha = sha(stable_bytes(paths["census"]))
    manifest_bytes = (
        stable_bytes(paths["overlay_manifest"]) if paths["overlay_manifest"].exists() else None
    )
    base = dict(
        config=cfg,
        census_sha256=census_sha,
        source=source_binding(),
        overlay_manifest_sha256=None if manifest_bytes is None else sha(manifest_bytes),
        build_kind="temporac.qualified-native-bridge.v1",
    )
    base["build_id"] = sha(canonical(base))
    if command == "plan":
        NB.atomic_json(
            out / "plan.json",
            dict(
                schema=NB.SCHEMA,
                status="prepared_not_built",
                binding=base,
                videos=len(rows),
                identities=sum(len(r["persons"]) for r in rows),
            ),
        )
        return 0
    if manifest_bytes is None:
        raise ValueError("complete frozen clock manifest required")
    manifest = json.loads(manifest_bytes)
    if (
        manifest["schema"] != OVERLAY_SCHEMA
        or manifest["status"] != "frozen_complete_census"
        or manifest["binding"]["census_sha256"] != census_sha
        or manifest["videos"] != len(rows)
        or manifest["identities"] != sum(len(r["persons"]) for r in rows)
    ):
        raise ValueError("frozen overlay population binding mismatch")
    entries = {r["media_key"]: r for r in manifest["records"]}
    if len(entries) != len(rows) or set(entries) != {r["media_key"] for r in rows}:
        raise ValueError("overlay full census membership mismatch")
    for row in rows:
        entry = entries[row["media_key"]]
        if entry["split"] != row["split"] or entry["slots"] != [p["slot"] for p in row["persons"]]:
            raise ValueError("overlay identity membership mismatch")
    missing = [
        r["media_key"]
        for r in rows
        if not all(
            (paths["raw_root"] / "raw" / r["split"] / r["media_key"]).with_suffix(s).is_file()
            for s in (".json", ".npz")
        )
    ]
    if missing:
        NB.atomic_json(
            out / "population_pending.json",
            dict(
                schema=NB.SCHEMA,
                status="pending_complete_raw",
                missing_raw=missing,
                videos=len(rows),
                identities=sum(len(r["persons"]) for r in rows),
            ),
        )
        return 2
    ledger, errors = [], []
    for row in rows:
        try:
            entry = entries[row["media_key"]]
            if entry["selection"] == "rejected":
                raise ValueError("clock overlay rejected: " + entry["error"])
            overlay_path = (paths["overlay_manifest"].parent / entry["overlay"]).resolve()
            if not overlay_path.is_relative_to(paths["overlay_manifest"].parent):
                raise ValueError("overlay path escapes manifest root")
            overlay = load_overlay(overlay_path)
            if (
                overlay.content_sha256 != entry["content_sha256"]
                or overlay.metadata != entry["binding"]
            ):
                raise ValueError("overlay manifest/content binding mismatch")
            if (
                overlay.metadata["generator_sha256"]
                != base["source"]["src/pams/temporac/native_clock_overlay_v1.py"]
            ):
                raise ValueError("overlay generator source changed")
            rawbase = paths["raw_root"] / "raw" / row["split"] / row["media_key"]
            raw_snapshot = {s: sha(stable_bytes(rawbase.with_suffix(s))) for s in (".json", ".npz")}
            arrays, meta = NB.raw_module().load_raw(rawbase)
            NB.raw_module().validate_raw(arrays, meta, row)
            rawbinding = dict(
                base,
                raw_json_sha256=sha(stable_bytes(rawbase.with_suffix(".json"))),
                raw_npz_sha256=sha(stable_bytes(rawbase.with_suffix(".npz"))),
                overlay_content_sha256=overlay.content_sha256,
                overlay_source_binding=overlay.metadata,
            )
            if (
                rawbinding["raw_json_sha256"] != overlay.metadata["raw_json_sha256"]
                or rawbinding["raw_npz_sha256"] != overlay.metadata["raw_npz_sha256"]
                or raw_snapshot
                != {".json": rawbinding["raw_json_sha256"], ".npz": rawbinding["raw_npz_sha256"]}
            ):
                raise ValueError("raw/overlay file binding mismatch")
            for person in row["persons"]:
                slot = person["slot"]
                feature, provenance = resolve_clock(
                    NB.project_raw_identity(arrays, meta, slot), overlay
                )
                identity = NB.preprocess_native(feature)
                info = NB.summary(identity)
                info["raw_status"] = meta["status"]
                stem = out / "identities" / row["split"] / f"{row['media_key']}_{slot}"
                fp, dp, jp = (
                    stem.with_suffix(".npz"),
                    stem.with_suffix(".derived.npz"),
                    stem.with_suffix(".json"),
                )
                binding = dict(rawbinding, slot=slot, clock_provenance=provenance)
                derived = NB.derived_arrays(identity)
                if jp.exists():
                    prior = json.loads(stable_bytes(jp))
                    if prior["binding"] != binding:
                        raise ValueError("resume binding mismatch; choose new output")
                    if (
                        sha(stable_bytes(fp)) != prior["feature_sha256"]
                        or sha(stable_bytes(dp)) != prior["derived_sha256"]
                    ):
                        raise ValueError("learner artifact hash mismatch")
                    loaded = NB.load_native_feature(fp)
                    if not all(
                        same(v, loaded.as_arrays()[k]) for k, v in feature.as_arrays().items()
                    ):
                        raise ValueError("resolved feature mismatch")
                    with np.load(dp, allow_pickle=False) as archive:
                        if set(archive.files) != set(derived) or not all(
                            same(archive[k], v) for k, v in derived.items()
                        ):
                            raise ValueError("derived artifact mismatch")
                    if prior["schema"] != NB.SCHEMA or prior["summary"] != info:
                        raise ValueError("learner sidecar mismatch")
                elif command == "validate":
                    raise ValueError("missing qualified learner identity")
                elif fp.exists() or dp.exists():
                    raise ValueError("orphan learner artifacts; choose new output")
                else:
                    NB.atomic_npz(fp, feature.as_arrays())
                    NB.atomic_npz(dp, derived)
                    NB.atomic_json(
                        jp,
                        dict(
                            schema=NB.SCHEMA,
                            binding=binding,
                            raw_status=meta["status"],
                            feature_sha256=sha(stable_bytes(fp)),
                            derived_sha256=sha(stable_bytes(dp)),
                            summary=info,
                        ),
                    )
                ledger.append(
                    dict(
                        media_key=row["media_key"],
                        split=row["split"],
                        slot=slot,
                        clock_qualification="qualified",
                        overlay_selection=entry["selection"],
                        **info,
                    )
                )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(dict(media_key=row["media_key"], error=error))
            existing = {r["slot"] for r in ledger if r["media_key"] == row["media_key"]}
            for person in row["persons"]:
                if person["slot"] not in existing:
                    ledger.append(
                        dict(
                            media_key=row["media_key"],
                            split=row["split"],
                            slot=person["slot"],
                            status="artifact_failure",
                            clock_qualification="rejected",
                            error=error,
                        )
                    )
    aggregate = dict(
        schema=NB.SCHEMA,
        status="population_failures" if errors else "validated_population",
        videos=len(rows),
        identities=len(ledger),
        records=ledger,
        failures=errors,
        binding=base,
        status_counts=dict(Counter(r["status"] for r in ledger)),
        clock_mode_counts=dict(Counter(r.get("clock_mode", "unavailable") for r in ledger)),
        reason_counts=dict(Counter(x for r in ledger for x in r.get("reasons", []))),
        clock_qualification_counts=dict(Counter(r["clock_qualification"] for r in ledger)),
        target_support="unavailable_no_target_producer",
        support_distributions_by_clock_mode={},
    )
    for mode in aggregate["clock_mode_counts"]:
        selected = [r for r in ledger if r.get("clock_mode", "unavailable") == mode]
        aggregate["support_distributions_by_clock_mode"][mode] = {}
        for field in (
            "source_length",
            "observed_joint_count",
            "usable_joint_count",
            "source_supported_frames",
            "derived_valid_frames",
            "clock_supported_edges",
            "geometry_supported_edges",
            "response_supported_edges",
            "response_supported_seconds",
            "run_count",
        ):
            values = np.array([r[field] for r in selected if r.get(field) is not None], np.float64)
            aggregate["support_distributions_by_clock_mode"][mode][field] = (
                None
                if not len(values)
                else dict(
                    min=float(values.min()),
                    median=float(np.median(values)),
                    p95=float(np.quantile(values, 0.95)),
                    max=float(values.max()),
                    total=float(values.sum()),
                )
            )
    NB.atomic_json(
        out / ("validation.json" if command == "validate" else "population_ledger.json"), aggregate
    )
    return 2 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "build", "validate"))
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return execute(json.loads(stable_bytes(args.config)), args.command)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
