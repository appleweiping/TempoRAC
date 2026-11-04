"""Build independent full-native learner artifacts; never launches training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

from pams.temporac.native_bridge_v1 import (  # noqa: E402
    SCHEMA,
    load_native_feature,
    preprocess_native,
    project_raw_identity,
)

EXPECTED = {"train": {"videos": 794, "persons": 1878}, "val": {"videos": 183, "persons": 486}}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def raw_module():
    spec = importlib.util.spec_from_file_location(
        "native_raw_readonly", ROOT / "scripts/build_temporac_native_pose_cache.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_census(path, expected=EXPECTED):
    rows = [
        json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()
    ]
    raw = raw_module()
    for row in rows:
        raw.validate_projection_record(row)
    raw.census(rows, expected=expected)
    if expected == EXPECTED:
        native_frames = sum(r["length"] for r in rows)
        person_frames = sum(r["length"] * len(r["persons"]) for r in rows)
        if (native_frames, person_frames) != (695625, 1758034):
            raise ValueError("frozen full native frame census mismatch")
    if len({r["media_key"] for r in rows}) != len(rows):
        raise ValueError("duplicate media census key")
    return rows


def source_binding():
    paths = [
        Path(__file__),
        ROOT / "src/pams/temporac/native_bridge_v1.py",
        ROOT / "src/pams/temporac/native_time_cue_v1.py",
    ]
    paths += [
        ROOT / f"src/pams/temporac/{name}.py"
        for name in ("preprocess", "cue", "quadrature", "timebase_audit", "decode")
    ]
    paths.append(ROOT / "scripts/build_temporac_native_pose_cache.py")
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def derived_arrays(identity):
    g = identity.geometry_features
    result = {
        "response_input": identity.response_input,
        "joint_usable": identity.joint_mask,
        "frame_derived_valid": identity.frame_valid,
        "observed_edge_support": identity.observed_edge_support,
        "response_edge_support": identity.response_edge_support,
        "edge_lengths": g.edge_lengths,
        "direction": g.direction,
        "direction_mask": g.direction_element_mask,
        "run_bounds": identity.run_bounds,
        "short_runs": identity.short_runs,
    }
    result.update({"map_" + k: v for k, v in identity.mapping.items()})
    return result


def summary(identity):
    support = identity.response_edge_support
    seconds = identity.clock.seconds
    duration = float(np.sum(np.diff(seconds)[support])) if identity.clock.mode == "pts" else None
    return {
        "status": identity.status,
        "reasons": list(identity.reasons),
        "clock_mode": identity.clock.mode,
        "source_length": int(identity.feature.source_length[0]),
        "observed_joint_count": int(identity.feature.joint_observed.sum()),
        "usable_joint_count": int(identity.joint_mask.sum()),
        "source_supported_frames": int(identity.feature.frame_source_ok.sum()),
        "derived_valid_frames": int(identity.frame_valid.sum()),
        "response_supported_edges": int(support.sum()),
        "clock_supported_edges": int(identity.clock.edge_time_support.sum()),
        "geometry_supported_edges": int(identity.observed_edge_support.sum()),
        "response_supported_seconds": duration,
        "tau_seconds": identity.clock.tau,
        "run_count": len(identity.run_bounds),
        "short_run_count": len(identity.short_runs),
        "target_support": "unavailable_no_target_producer",
        "period_ap": "unavailable_no_period_decoder",
    }


def check_config(cfg):
    if set(cfg) != {"raw_root", "census", "output"}:
        raise ValueError("config requires exactly raw_root,census,output (no caps)")
    paths = [Path(cfg[k]).resolve() for k in cfg]
    if os.name == "nt" and any(p.drive.upper() != "E:" for p in paths):
        raise ValueError("all inputs/outputs must remain on E:")
    raw, census, out = (Path(cfg[k]).resolve() for k in ("raw_root", "census", "output"))
    if raw == out or raw in out.parents or out in raw.parents:
        raise ValueError("output must be separate from raw cache")


def execute(cfg, command, *, expected=EXPECTED):
    check_config(cfg)
    rows = read_census(cfg["census"], expected=expected)
    out = Path(cfg["output"])
    base_binding = {
        "config": cfg,
        "census_sha256": digest(cfg["census"]),
        "source": source_binding(),
    }
    raw = raw_module()
    missing = [
        r["media_key"]
        for r in rows
        if not all(
            (Path(cfg["raw_root"]) / "raw" / r["split"] / r["media_key"]).with_suffix(s).is_file()
            for s in (".json", ".npz")
        )
    ]
    if command == "plan":
        atomic_json(
            out / "plan.json",
            {
                "schema": SCHEMA,
                "status": "prepared_not_built",
                "binding": base_binding,
                "videos": len(rows),
                "identities": sum(len(r["persons"]) for r in rows),
                "missing_raw": missing,
            },
        )
        return 0
    # Never write a partial production build while the complete raw cache generates.
    if missing:
        atomic_json(
            out / "population_pending.json",
            {
                "schema": SCHEMA,
                "status": "pending_complete_raw",
                "videos": len(rows),
                "identities": sum(len(r["persons"]) for r in rows),
                "missing_raw": missing,
            },
        )
        return 2
    ledger, errors = [], []
    for row in rows:
        rawbase = Path(cfg["raw_root"]) / "raw" / row["split"] / row["media_key"]
        try:
            arrays, meta = raw.load_raw(rawbase)
            raw.validate_raw(arrays, meta, row)
            rawbinding = dict(
                base_binding,
                raw_npz_sha256=digest(rawbase.with_suffix(".npz")),
                raw_json_sha256=digest(rawbase.with_suffix(".json")),
            )
            for person in row["persons"]:
                slot = person["slot"]
                stem = out / "identities" / row["split"] / f"{row['media_key']}_{slot}"
                feature_path = stem.with_suffix(".npz")
                sidecar = stem.with_suffix(".json")
                derived_path = stem.with_suffix(".derived.npz")
                binding = dict(rawbinding, slot=slot)
                feature = project_raw_identity(arrays, meta, slot)
                identity = preprocess_native(feature)
                info = summary(identity)
                info["raw_status"] = meta["status"]
                if sidecar.exists():
                    prior = json.loads(sidecar.read_text(encoding="utf-8"))
                    if prior["binding"] != binding:
                        raise ValueError("resume binding mismatch; choose a new output directory")
                    if (
                        digest(feature_path) != prior["feature_sha256"]
                        or digest(derived_path) != prior["derived_sha256"]
                    ):
                        raise ValueError("learner artifact hash mismatch")
                    loaded = load_native_feature(feature_path)
                    for key, value in feature.as_arrays().items():
                        if not np.array_equal(value, loaded.as_arrays()[key], equal_nan=True):
                            raise ValueError("feature/raw projection mismatch")
                    with np.load(derived_path, allow_pickle=False) as derived:
                        expected_arrays = derived_arrays(identity)
                        if set(derived.files) != set(expected_arrays):
                            raise ValueError("derived schema mismatch")
                        for key, value in expected_arrays.items():
                            same = (
                                np.array_equal(derived[key], value, equal_nan=True)
                                if value.dtype.kind in "fc"
                                else np.array_equal(derived[key], value)
                            )
                            if not same:
                                raise ValueError("derived artifact mismatch: " + key)
                    if prior["summary"] != info:
                        raise ValueError("derived summary mismatch")
                elif command == "validate":
                    raise ValueError("missing learner identity artifact")
                else:
                    atomic_npz(feature_path, feature.as_arrays())
                    atomic_npz(derived_path, derived_arrays(identity))
                    atomic_json(
                        sidecar,
                        {
                            "schema": SCHEMA,
                            "binding": binding,
                            "raw_status": meta["status"],
                            "feature_sha256": digest(feature_path),
                            "derived_sha256": digest(derived_path),
                            "summary": info,
                        },
                    )
                ledger.append(
                    dict(media_key=row["media_key"], split=row["split"], slot=slot, **info)
                )
        except (ValueError, OSError, KeyError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"media_key": row["media_key"], "error": error})
            existing = {r["slot"] for r in ledger if r["media_key"] == row["media_key"]}
            for p in row["persons"]:
                if p["slot"] not in existing:
                    ledger.append(
                        {
                            "media_key": row["media_key"],
                            "split": row["split"],
                            "slot": p["slot"],
                            "status": "artifact_failure",
                            "error": error,
                        }
                    )
    distributions = {}
    for mode in sorted({r.get("clock_mode", "unavailable") for r in ledger}):
        selected = [r for r in ledger if r.get("clock_mode", "unavailable") == mode]
        distributions[mode] = {}
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
        ):
            values = np.array([r[field] for r in selected if r.get(field) is not None], np.float64)
            distributions[mode][field] = (
                None
                if not values.size
                else {
                    "identities": int(values.size),
                    "sum": float(values.sum()),
                    "min": float(values.min()),
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(values.max()),
                }
            )
    aggregate = {
        "schema": SCHEMA,
        "status": "validated_population" if not errors else "population_failures",
        "videos": len(rows),
        "identities": len(ledger),
        "records": ledger,
        "failures": errors,
        "status_counts": dict(Counter(x["status"] for x in ledger)),
        "support_distributions_by_clock_mode": distributions,
        "clock_mode_counts": dict(Counter(x.get("clock_mode", "unavailable") for x in ledger)),
        "reason_counts": dict(Counter(reason for x in ledger for reason in x.get("reasons", []))),
        "binding": base_binding,
        "target_support": "unavailable_no_target_producer",
    }
    atomic_json(
        out / ("validation.json" if command == "validate" else "population_ledger.json"), aggregate
    )
    return 0 if not errors else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "build", "validate"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    try:
        return execute(cfg, args.command)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
