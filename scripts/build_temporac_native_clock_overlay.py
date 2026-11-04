"""Freeze one complete native clock census; no partial population build or training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pams.temporac.native_bridge_v1 import native_clock  # noqa: E402
from pams.temporac.native_clock_overlay_v1 import (  # noqa: E402
    LIMITATION,
    SCHEMA,
    build_clock_overlay,
    readonly_nb1,
    resolve_clock,
    save_overlay,
    sha,
    stable_bytes,
)

NB = readonly_nb1()
EXPECTED = NB.EXPECTED


def config_paths(cfg):
    if set(cfg) != {"raw_root", "census", "decode_root", "media_inventory", "output"}:
        raise ValueError(
            "config requires raw_root,census,decode_root,media_inventory,output; no caps"
        )
    paths = {k: Path(v).resolve() for k, v in cfg.items()}
    if os.name == "nt" and any(p.drive.upper() != "E:" for p in paths.values()):
        raise ValueError("all project paths must remain on E:")
    out = paths["output"]
    for k, p in paths.items():
        if k != "output" and (out == p or out in p.parents or p in out.parents):
            raise ValueError("overlay output must be separate from inputs")
    return paths


def read_decode(path):
    # splitlines(keepends=True) preserves the exact historical JSONL row bytes.
    data = stable_bytes(path)
    rows = {}
    for line in data.splitlines(keepends=True):
        if not line.strip():
            raise ValueError("blank decode JSONL row")
        item = json.loads(line)
        if item["path"] in rows:
            raise ValueError("multiple decode clocks for one media path")
        rows[item["path"]] = (item, line)
    return rows, sha(data)


def execute(cfg, command, *, expected=EXPECTED):
    paths = config_paths(cfg)
    rows = NB.read_census(paths["census"], expected=expected)
    out = paths["output"]
    census_bytes = stable_bytes(paths["census"])
    binding = {
        "config": cfg,
        "census_sha256": sha(census_bytes),
        "source": {
            "overlay": sha(stable_bytes(ROOT / "src/pams/temporac/native_clock_overlay_v1.py")),
            "cli": sha(stable_bytes(__file__)),
            **NB.source_binding(),
        },
    }
    status_path = paths["decode_root"] / "status.json"
    status = json.loads(stable_bytes(status_path)) if status_path.exists() else {}
    missing = []
    for r in rows:
        base = paths["raw_root"] / "raw" / r["split"] / r["media_key"]
        if not all(base.with_suffix(s).is_file() for s in (".json", ".npz")):
            missing.append(r["media_key"])
    if command == "plan":
        NB.atomic_json(
            out / "plan.json",
            dict(
                schema=SCHEMA,
                status="prepared_not_built",
                binding=binding,
                videos=len(rows),
                identities=sum(len(r["persons"]) for r in rows),
                missing_raw=missing,
                decode_status=status.get("status", "unavailable"),
            ),
        )
        return 0
    if (
        missing
        or status.get("status") not in ("complete", "complete_with_findings")
        or status.get("completed") != len(rows)
    ):
        NB.atomic_json(
            out / "population_pending.json",
            dict(
                schema=SCHEMA,
                status="pending_complete_raw_and_decode",
                missing_raw=missing,
                decode_status=status.get("status"),
                videos=len(rows),
                identities=sum(len(r["persons"]) for r in rows),
            ),
        )
        return 2
    if status.get("script_sha256") != sha(stable_bytes(ROOT / "scripts/audit_temporac_decode.py")):
        raise ValueError("decode generator source binding mismatch")
    decoded, decode_sha = read_decode(paths["decode_root"] / "decode.jsonl")
    inventory_bytes = stable_bytes(paths["media_inventory"])
    inventory = {}
    for line in inventory_bytes.splitlines():
        r = json.loads(line)
        if r["path"] in inventory:
            raise ValueError("duplicate media inventory path")
        inventory[r["path"]] = r
    member_paths = {r["media_path"] for r in rows}
    if set(decoded) != member_paths or set(inventory) != member_paths:
        raise ValueError("complete decode/inventory census membership mismatch")
    binding.update(
        decode_jsonl_sha256=decode_sha,
        decode_status_sha256=sha(stable_bytes(status_path)),
        media_inventory_sha256=sha(inventory_bytes),
    )
    # Preflight ALL raw completion before any per-video artifact is frozen.
    for r in rows:
        base = paths["raw_root"] / "raw" / r["split"] / r["media_key"]
        m = json.loads(stable_bytes(base.with_suffix(".json")))
        if m.get("status") not in ("complete", "failed", "alignment_unresolved", "decode_failed"):
            raise ValueError("unfinished raw population")
    records = []
    for r in rows:
        result = {
            "media_key": r["media_key"],
            "split": r["split"],
            "slots": [p["slot"] for p in r["persons"]],
        }
        try:
            inventory_row = inventory[r["media_path"]]
            if (
                inventory_row.get("status") != "hashed"
                or inventory_row["sha256"] != r["media_sha256"]
                or inventory_row["split"] != r["split"]
            ):
                raise ValueError("frozen inventory/projection media mismatch")
            base = paths["raw_root"] / "raw" / r["split"] / r["media_key"]
            arrays, meta = NB.raw_module().load_raw(base)
            NB.raw_module().validate_raw(arrays, meta, r)
            item, row_bytes = decoded[r["media_path"]]
            if inventory_row["bytes"] != item["prior_bytes"]:
                raise ValueError("inventory/decode byte-size mismatch")
            csv_path = (paths["decode_root"] / item["frame_clock_csv"]).resolve()
            if not csv_path.is_relative_to(paths["decode_root"]):
                raise ValueError("clock CSV escapes decode root")
            overlay = build_clock_overlay(
                base.with_suffix(".json"), base.with_suffix(".npz"), row_bytes, csv_path
            )
            clock = native_clock(
                resolve_clock(NB.project_raw_identity(arrays, meta, 0), overlay)[0]
            )
            dest = out / "videos" / r["split"] / r["media_key"]
            if command == "validate":
                from pams.temporac.native_clock_overlay_v1 import load_overlay

                if load_overlay(dest).content_sha256 != overlay.content_sha256:
                    raise ValueError("overlay changed during validation")
            else:
                save_overlay(dest, overlay)
            result.update(
                selection=overlay.metadata["selection"],
                overlay=str(dest.relative_to(out).as_posix()),
                content_sha256=overlay.content_sha256,
                binding=overlay.metadata,
                pts_frames=int(overlay.pts_present.sum()),
                source_frames=overlay.metadata["N"],
                adjacent_time_edges=int(clock.edge_time_support.sum())
                if clock.mode == "pts"
                else 0,
                adjacent_known_seconds=float(
                    sum(
                        clock.seconds[j + 1] - clock.seconds[j]
                        for j in range(len(clock.edge_time_support))
                        if clock.mode == "pts" and clock.edge_time_support[j]
                    )
                ),
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result.update(selection="rejected", error=f"{type(exc).__name__}: {exc}")
        records.append(result)
    manifest = dict(
        schema=SCHEMA,
        status="frozen_complete_census",
        binding=binding,
        records=records,
        videos=len(rows),
        identities=sum(len(r["persons"]) for r in rows),
        selection_counts=dict(Counter(r["selection"] for r in records)),
        media_sha_assurance=LIMITATION,
    )
    manifest_path = out / "overlay_manifest.json"
    if manifest_path.exists():
        if json.loads(stable_bytes(manifest_path)) != manifest:
            raise ValueError("immutable full manifest changed; choose a new output")
    elif command == "validate":
        raise ValueError("missing frozen full manifest")
    else:
        NB.atomic_json(manifest_path, manifest)
    NB.atomic_json(
        out / "validation.json",
        dict(
            manifest_sha256=sha(stable_bytes(manifest_path)),
            status="population_failures"
            if any(r["selection"] == "rejected" for r in records)
            else "validated_population",
        ),
    )
    return 2 if any(r["selection"] == "rejected" for r in records) else 0


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
