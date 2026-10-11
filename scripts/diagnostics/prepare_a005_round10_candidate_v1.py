"""Create an additive A005 Round-10 candidate from the reviewed Round-9 bytes."""

from __future__ import annotations

import argparse
from pathlib import Path

OLD_TIME = b"2026-08-19T19:54:24+08:00"
NEW_TIME = b"2026-08-22T17:08:41+08:00"
OLD_TOKEN = b"20260819_195424"
NEW_TOKEN = b"20260822_170841"
OLD_UNIQUENESS = (
    b"expanded outcomes are unique by that K,current stage,action,token."
)
NEW_UNIQUENESS = (
    b"expanded rows are unique by observable (K,current stage) across all emitter "
    b"metadata; action and mapped_token_or_null are outputs of the uniquely selected "
    b"row and never selector inputs."
)
OLD_CLOSURE = (
    b"Runtime audit selection uses only event,arity,exact type vector,current stage "
    b"and outcome. Emitter is deterministic least-site static provenance,never "
    b"callback authority; the seven-key raw row requires no nonexistent field."
)
NEW_CLOSURE = (
    b"Runtime audit selection uses only event,arity,exact type vector and current "
    b"stage. The observable definition key K=(event,arity,exact type vector) selects "
    b"one definition and observable (K,current stage) selects one expanded row; "
    b"action,mapped_token_or_null and outcome are outputs of that row and never "
    b"selector inputs. Emitter is deterministic least-site static provenance,never "
    b"callback authority; the seven-key raw row requires no nonexistent field."
)


def _appendix(markdown: bytes) -> bytes:
    start_marker = b"~~~json\n"
    end_marker = b"\n~~~\n"
    start = markdown.index(start_marker) + len(start_marker)
    end = markdown.index(end_marker, start)
    return markdown[start:end] + b"\n"


def _replace_exact(raw: bytes, old: bytes, new: bytes, *, count: int) -> bytes:
    actual = raw.count(old)
    if actual != count:
        raise ValueError(f"replacement count mismatch: expected {count}, got {actual}")
    return raw.replace(old, new)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--source-md", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    source_json = args.source_json.read_bytes()
    source_md = args.source_md.read_bytes()
    if _appendix(source_md) != source_json:
        raise ValueError("source Markdown appendix differs from source JSON")
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Round-10 output already exists")

    output_json = _replace_exact(
        source_json, OLD_UNIQUENESS, NEW_UNIQUENESS, count=1
    )
    output_json = _replace_exact(output_json, OLD_CLOSURE, NEW_CLOSURE, count=1)
    output_json = output_json.replace(OLD_TIME, NEW_TIME).replace(OLD_TOKEN, NEW_TOKEN)

    output_md = _replace_exact(source_md, OLD_UNIQUENESS, NEW_UNIQUENESS, count=1)
    output_md = _replace_exact(output_md, OLD_CLOSURE, NEW_CLOSURE, count=1)
    output_md = output_md.replace(OLD_TIME, NEW_TIME).replace(OLD_TOKEN, NEW_TOKEN)
    if _appendix(output_md) != output_json:
        raise ValueError("generated Markdown appendix differs from generated JSON")

    args.output_json.write_bytes(output_json)
    args.output_md.write_bytes(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
