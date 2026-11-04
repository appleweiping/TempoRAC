"""Read-only, full train/validation media inventory; never reads annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, required=True)
    parser.add_argument("--expected-val", type=int, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(source):
        parser.error("output must be outside the read-only source tree")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    paths = {split: sorted((source / split).rglob("*.mp4")) for split in ("train", "val")}
    expected = {"train": args.expected_train, "val": args.expected_val}
    actual = {split: len(items) for split, items in paths.items()}
    if actual != expected:
        write_json(output / "status.json", {"status": "failed_population_check", "expected": expected, "actual": actual})
        return 2
    status = {
        "schema_version": "temporac.media-inventory.v1",
        "status": "running",
        "source": str(source),
        "expected": expected,
        "completed": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "complete train and validation MP4 byte integrity; no annotation reads; test untouched",
    }
    write_json(output / "status.json", status)
    seen: dict[str, list[dict]] = {}
    failures = []
    total_bytes = 0
    with (output / "media.jsonl").open("x", encoding="utf-8") as stream:
        for split, items in paths.items():
            for path in items:
                record = {"split": split, "path": str(path.relative_to(source))}
                before = path.stat()
                try:
                    digest = hashlib.sha256()
                    with path.open("rb") as media:
                        header = media.read(32)
                        digest.update(header)
                        for chunk in iter(lambda: media.read(1024 * 1024), b""):
                            digest.update(chunk)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("source changed while reading")
                    # ISO base media begins with a box whose type is ftyp. This
                    # is a container-header check, not full decoding validation.
                    if before.st_size < 32 or header[4:8] != b"ftyp":
                        raise ValueError("missing expected MP4 ftyp header")
                    record.update(bytes=before.st_size, sha256=digest.hexdigest(), status="hashed")
                    total_bytes += before.st_size
                    seen.setdefault(digest.hexdigest(), []).append({"split": split, "path": record["path"]})
                except (OSError, ValueError) as exc:
                    record.update(status="failed", error=str(exc))
                    failures.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                status["completed"] += 1
                if status["completed"] % 25 == 0:
                    write_json(output / "status.json", status)
    duplicates = [members for members in seen.values() if len(members) > 1]
    cross_split_duplicates = [members for members in duplicates if len({item["split"] for item in members}) > 1]
    status.update(
        status="complete" if not failures else "complete_with_errors",
        elapsed_seconds=round(time.time() - started, 3),
        total_bytes=total_bytes,
        failures=failures,
        exact_byte_duplicates=duplicates,
        exact_byte_cross_split_duplicates=cross_split_duplicates,
        limitations=["No codec/full-frame decoding performed.", "Different composites may share source videos without identical bytes; source-video leakage still requires provenance audit.", "No test media or annotation labels were read.", "No model efficacy or formal experiment result is produced."],
    )
    write_json(output / "status.json", status)
    print(json.dumps({key: status[key] for key in ("status", "completed", "total_bytes", "elapsed_seconds")}), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
