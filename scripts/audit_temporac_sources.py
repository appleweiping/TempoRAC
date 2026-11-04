"""Label-free source path/byte provenance. Never opens composite media or labels.

Filename tokens are provenance candidates, not verified synthesis recipes. Byte
canonical IDs identify exact encodings only, not latent recordings after edits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def composite_inventory(inventory: dict, source: str, expected: dict) -> list[dict]:
    matches = [s for s in inventory["sources"] if s["root"] == source
               and s["kind"] == "multirep_larger_split_tree"]
    if len(matches) != 1:
        raise ValueError("expected exactly one selected composite inventory root")
    records = []
    for split, count in expected.items():
        entry = matches[0]["splits"][split]
        names = entry["entry_names"]
        if entry["entry_count"] != count or len(names) != count or len(set(names)) != count:
            raise ValueError(f"population mismatch: {split}")
        for name in sorted(names):
            tokens = name.split("+")
            if any(not x or "/" in x or "\\" in x or x in (".", "..") for x in tokens):
                raise ValueError(f"invalid composite identity: {name}")
            records.append({"id": split + "/" + name, "split": split, "name": name,
                            "tokens": tokens})
    return records


def source_paths(roots: list[Path], tokens: set[str]) -> tuple[dict, list]:
    """Exact case-sensitive stems; preserve lexical and resolved paths."""
    paths = defaultdict(list)
    root_receipts = []
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"source root missing: {root}")
        media = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
        root_receipts.append({"root": str(root), "media_files": len(media),
                              "matched_files": sum(p.stem in tokens for p in media)})
        for path in media:
            if path.stem in tokens:
                paths[path.stem].append({"path": str(path), "realpath": str(path.resolve(strict=True))})
    return {token: sorted(paths[token], key=lambda r: r["path"]) for token in sorted(tokens)}, root_receipts


def token_graph(paths: dict, hashes: dict) -> tuple[dict, dict]:
    tokens, canonical = {}, defaultdict(lambda: {"tokens": [], "realpaths": []})
    for token, candidates in paths.items():
        realpaths = sorted({r["realpath"] for r in candidates})
        values = sorted({hashes[p]["sha256"] for p in realpaths if hashes[p]["status"] == "hashed"})
        failed = [p for p in realpaths if hashes[p]["status"] != "hashed"]
        state = "missing" if not realpaths else ("read_error" if failed else
                ("resolved_byte_identity" if len(values) == 1 else "ambiguous_distinct_bytes"))
        tokens[token] = {"paths": candidates, "realpaths": realpaths, "byte_ids": values,
                         "state": state, "canonical_id": values[0] if state == "resolved_byte_identity" else None}
        for value in values:
            canonical[value]["tokens"].append(token)
            canonical[value]["realpaths"].extend(p for p in realpaths
                                                 if hashes[p].get("sha256") == value)
    for entry in canonical.values():
        entry["realpaths"] = sorted(set(entry["realpaths"]))
    return tokens, dict(canonical)


def memberships(composites: list[dict], tokens: dict) -> tuple[list, dict]:
    exposed_tokens = {t for r in composites if r["split"] in ("train", "val") for t in r["tokens"]}
    # All possible byte identities of ambiguous exposed names are conservative
    # exposure edges; no selected candidate may intersect any of them.
    exposed_bytes = {h for t in exposed_tokens for h in tokens[t]["byte_ids"]}
    rows = []
    for comp in composites:
        row = dict(comp)
        row["canonical_ids"] = sorted({tokens[t]["canonical_id"] for t in row["tokens"]
                                        if tokens[t]["canonical_id"] is not None})
        row["unresolved_tokens"] = sorted({t for t in row["tokens"] if tokens[t]["canonical_id"] is None})
        row["same_token_exposure"] = sorted(set(row["tokens"]) & exposed_tokens)
        row["possible_byte_exposure"] = sorted({h for t in row["tokens"] for h in tokens[t]["byte_ids"]} & exposed_bytes)
        if comp["split"] != "test":
            row["classification"] = "current_development_exposure"
        elif row["same_token_exposure"] or row["possible_byte_exposure"]:
            row["classification"] = "overlap"
        elif row["unresolved_tokens"]:
            row["classification"] = "unknown"
        else:
            row["classification"] = "B_candidate"
        rows.append(row)
    return rows, {"tokens": sorted(exposed_tokens), "possible_byte_ids": sorted(exposed_bytes),
                  "history_complete": False, "scope": "current full train and validation only"}


def components(rows: list[dict], tokens: dict) -> list[dict]:
    """Conservative shared-source graph includes ambiguous possibilities."""
    parent = {r["id"]: r["id"] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for row in rows:
        keys = {"token:" + t for t in row["tokens"]}
        keys.update("sha256:" + h for t in row["tokens"] for h in tokens[t]["byte_ids"])
        for key in sorted(keys):
            if key in first:
                a, b = sorted((find(row["id"]), find(first[key])))
                parent[b] = a
            else:
                first[key] = row["id"]
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row["id"])
    return [{"id": k, "members": sorted(v)} for k, v in sorted(groups.items())]


def reuse_composite_audit(status_path: Path, media_path: Path, composites: list[dict], source: str) -> dict:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    media = [json.loads(line) for line in media_path.read_text(encoding="utf-8").splitlines() if line]
    expected = {r["id"] for r in composites if r["split"] in ("train", "val")}
    actual = [str(PurePosixPath(r["path"]).parent) for r in media]
    if (status["status"] != "complete" or status["source"] != source or status["failures"]
            or status["completed"] != len(expected) or len(media) != len(expected)
            or len(set(actual)) != len(actual) or set(actual) != expected
            or any(r["status"] != "hashed" or r["split"] != PurePosixPath(r["path"]).parts[0] for r in media)):
        raise ValueError("existing composite audit does not cover exact current train/val population")
    return {"status_path": str(status_path), "status_sha256": sha256(status_path),
            "media_path": str(media_path), "media_sha256": sha256(media_path),
            "completed": len(media), "exact_byte_duplicates": status["exact_byte_duplicates"],
            "scope": "receipt reuse only; no composite media opened"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory-only", action="store_true", help="path-set spine; no bytes or membership claims")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    output = args.output.resolve()
    roots = [Path(p).resolve(strict=True) for p in config["source_roots"]]
    source = config["composite_root"]
    if any(output.is_relative_to(p) for p in [*roots, Path(source).resolve(strict=True)]):
        parser.error("output must be outside all source trees")
    output.mkdir(parents=True, exist_ok=False)
    inventory_path = Path(config["inventory"])
    composites = composite_inventory(json.loads(inventory_path.read_text(encoding="utf-8-sig")), source, config["expected"])
    if not args.inventory_only:
        live = {split: sorted(p.name for p in (Path(source) / split).iterdir() if p.is_dir()) for split in config["expected"]}
        for split in live:
            if live[split] != sorted(r["name"] for r in composites if r["split"] == split):
                raise ValueError("current composite directory membership differs from bound inventory")
    all_tokens = {t for r in composites for t in r["tokens"]}
    paths, root_receipts = source_paths(roots, all_tokens)
    base = {"schema_version": "temporac.source-inventory.v1", "pid": os.getpid(),
            "started_unix": time.time(), "script_sha256": sha256(Path(__file__)),
            "config_sha256": sha256(args.config), "config_path": str(args.config.resolve()),
            "inventory_sha256": sha256(inventory_path), "config": config,
            "source_roots": root_receipts, "token_count": len(all_tokens),
            "status": "inventory_only" if args.inventory_only else "running", "completed": 0}
    write_json(output / "source_paths.json", paths)
    write_json(output / "status.json", base)
    if args.inventory_only:
        print(json.dumps({"status": base["status"], "tokens": len(paths)}), flush=True)
        return 0
    reuse = reuse_composite_audit(Path(config["composite_audit_status"]), Path(config["composite_audit_media"]), composites, source)
    write_json(output / "composite_audit_reused.json", reuse)
    hashes = {}
    unique = sorted({r["realpath"] for candidates in paths.values() for r in candidates})
    base["expected_source_files"] = len(unique)
    with (output / "source_media.jsonl").open("x", encoding="utf-8") as stream:
        for path in unique:
            record = {"realpath": path}
            try:
                p = Path(path)
                before = p.stat()
                digest = sha256(p)
                after = p.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("source changed while hashing")
                record.update(status="hashed", sha256=digest, bytes=before.st_size, mtime_ns=before.st_mtime_ns)
            except (OSError, ValueError) as exc:
                record.update(status="failed", error=str(exc))
            hashes[path] = record
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            base["completed"] += 1
            if base["completed"] % 20 == 0:
                write_json(output / "status.json", base)
                print(json.dumps({"hashed": base["completed"], "total": len(unique)}), flush=True)
    graph, canonical = token_graph(paths, hashes)
    rows, exposure = memberships(composites, graph)
    write_json(output / "canonical_graph.json", {"tokens": graph, "canonical_byte_nodes": canonical,
               "edge_semantics": "composite->token by filename proxy; token->path exact stem; path->byte node by SHA256"})
    write_json(output / "membership.json", {"A": [r for r in rows if r["split"] == "test"],
                "development": [r for r in rows if r["split"] != "test"], "exposure": exposure})
    write_json(output / "source_components.json", components(rows, graph))
    base.update(status="complete" if all(r["status"] == "hashed" for r in hashes.values()) else "complete_with_errors",
        elapsed_seconds=round(time.time() - base["started_unix"], 3),
        source_state_counts=dict(Counter(t["state"] for t in graph.values())),
        test_classifications=dict(Counter(r["classification"] for r in rows if r["split"] == "test")),
        canonical_byte_nodes=len(canonical),
        distinct_token_alias_groups=[{"sha256": h, **n} for h, n in canonical.items() if len(n["tokens"]) > 1],
        limitations=["Filename decomposition is not verified against a synthesis recipe.",
            "SHA256 deduplicates exact bytes only; transcoded/cropped same recordings may differ.",
            "Historical pilot/v4x and other development exposures are not fully reconciled; B is candidate only.",
            "No test annotations, counts, periods, metrics, models or GPUs accessed.",
            "Official release identity remains a separate verification obligation."])
    base["artifacts"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "status.json"}
    write_json(output / "status.json", base)
    print(json.dumps({k: base[k] for k in ("status", "completed", "test_classifications", "source_state_counts")}), flush=True)
    return 0 if base["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
