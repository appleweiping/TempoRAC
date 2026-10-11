"""Strict structural precheck for the additive A005 Round-10 pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_a005_round10_candidate_v1 import (
    NEW_CLOSURE,
    NEW_TIME,
    NEW_TOKEN,
    NEW_UNIQUENESS,
    OLD_CLOSURE,
    OLD_TIME,
    OLD_TOKEN,
    OLD_UNIQUENESS,
)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    _validate_text(raw, path)
    return raw, json.loads(raw, object_pairs_hook=_strict_pairs)


def _validate_text(raw: bytes, path: Path) -> None:
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw:
        raise ValueError(f"non-canonical text bytes: {path}")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"terminal LF invariant failed: {path}")


def _transform(value: str) -> str:
    return (
        value.replace(OLD_UNIQUENESS.decode(), NEW_UNIQUENESS.decode())
        .replace(OLD_CLOSURE.decode(), NEW_CLOSURE.decode())
        .replace(OLD_TIME.decode(), NEW_TIME.decode())
        .replace(OLD_TOKEN.decode(), NEW_TOKEN.decode())
    )


def _compare(old: Any, new: Any, path: str, changed: list[str]) -> None:
    if type(old) is not type(new):
        raise ValueError(f"type changed at {path}")
    if isinstance(old, dict):
        if tuple(old) != tuple(new):
            raise ValueError(f"object keys/order changed at {path}")
        for key in old:
            _compare(old[key], new[key], f"{path}/{key}", changed)
        return
    if isinstance(old, list):
        if len(old) != len(new):
            raise ValueError(f"array length changed at {path}")
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            _compare(left, right, f"{path}/{index}", changed)
        return
    if isinstance(old, str):
        if old != new:
            if _transform(old) != new:
                raise ValueError(f"unapproved string change at {path}")
            changed.append(path)
        return
    if old != new:
        raise ValueError(f"non-string value changed at {path}")


def _appendix(markdown: bytes) -> bytes:
    start_marker = b"~~~json\n"
    end_marker = b"\n~~~\n"
    start = markdown.index(start_marker) + len(start_marker)
    end = markdown.index(end_marker, start)
    return markdown[start:end] + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-json", required=True, type=Path)
    parser.add_argument("--new-json", required=True, type=Path)
    parser.add_argument("--new-md", required=True, type=Path)
    args = parser.parse_args()
    _old_raw, old = _load(args.old_json)
    new_raw, new = _load(args.new_json)
    markdown = args.new_md.read_bytes()
    _validate_text(markdown, args.new_md)
    if _appendix(markdown) != new_raw:
        raise ValueError("Markdown appendix differs from standalone JSON")
    changed: list[str] = []
    _compare(old, new, "", changed)
    if not changed:
        raise ValueError("candidate has no approved changes")
    for stale in (OLD_UNIQUENESS, OLD_CLOSURE, OLD_TIME, OLD_TOKEN):
        if stale in new_raw or stale in markdown:
            raise ValueError(f"stale Round-9 bytes remain: {stale!r}")
    for required in (NEW_UNIQUENESS, NEW_CLOSURE, NEW_TIME, NEW_TOKEN):
        if required not in new_raw or required not in markdown:
            raise ValueError(f"required Round-10 bytes missing: {required!r}")
    print(
        json.dumps(
            {
                "changed_json_string_paths": len(changed),
                "json_bytes": len(new_raw),
                "json_sha256": _sha(new_raw),
                "markdown_bytes": len(markdown),
                "markdown_sha256": _sha(markdown),
                "result": "PASS",
                "schema": "temporac.a005-round10-author-precheck.v1",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
