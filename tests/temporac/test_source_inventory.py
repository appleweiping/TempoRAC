"""Mechanical source-provenance checks, without labels or media decoding."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("source_inventory", Path(__file__).parents[2] / "scripts/audit_temporac_sources.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_exact_stem_case_and_all_real_paths(tmp_path):
    for name, content in [("a/Foo.mp4", b"same"), ("b/Foo.mp4", b"same"), ("c/foo.mp4", b"different")]:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
    paths, receipts = audit.source_paths([tmp_path], {"Foo", "FOO"})
    assert len(paths["Foo"]) == 2
    assert paths["FOO"] == []
    assert receipts[0]["media_files"] == 3


def test_aliases_ambiguity_missing_and_read_failure():
    paths = {"a": [{"path": "/1", "realpath": "/1"}],
             "alias": [{"path": "/2", "realpath": "/2"}],
             "amb": [{"path": "/1", "realpath": "/1"}, {"path": "/3", "realpath": "/3"}],
             "missing": [], "bad": [{"path": "/4", "realpath": "/4"}]}
    hashes = {"/1": {"status": "hashed", "sha256": "H1"},
              "/2": {"status": "hashed", "sha256": "H1"},
              "/3": {"status": "hashed", "sha256": "H2"}, "/4": {"status": "failed"}}
    tokens, nodes = audit.token_graph(paths, hashes)
    assert tokens["a"]["canonical_id"] == tokens["alias"]["canonical_id"]
    assert tokens["amb"]["state"] == "ambiguous_distinct_bytes"
    assert tokens["amb"]["canonical_id"] is None
    assert tokens["missing"]["state"] == "missing"
    assert tokens["bad"]["state"] == "read_error"
    assert set(nodes["H1"]["tokens"]) == {"a", "alias", "amb"}


def test_membership_retains_all_and_conservative_exposure_edges():
    tokens = {"train": {"canonical_id": None, "byte_ids": ["H1", "H2"]},
              "alias": {"canonical_id": "H1", "byte_ids": ["H1"]},
              "new": {"canonical_id": "H3", "byte_ids": ["H3"]},
              "missing": {"canonical_id": None, "byte_ids": []}}
    comps = [{"id": "train/x", "split": "train", "tokens": ["train"]},
             {"id": "test/alias", "split": "test", "tokens": ["alias"]},
             {"id": "test/new", "split": "test", "tokens": ["new"]},
             {"id": "test/missing", "split": "test", "tokens": ["missing"]},
             {"id": "test/same", "split": "test", "tokens": ["train", "missing"]}]
    rows, exposure = audit.memberships(comps, tokens)
    assert len(rows) == len(comps)
    assert [r["classification"] for r in rows[1:]] == ["overlap", "B_candidate", "unknown", "overlap"]
    assert exposure["history_complete"] is False
    groups = audit.components(rows, tokens)
    joined = next(g for g in groups if "train/x" in g["members"])
    assert "test/alias" in joined["members"]


def test_population_and_receipt_reuse(tmp_path):
    source = "/composites"
    inv = {"sources": [{"root": source, "kind": "multirep_larger_split_tree", "splits": {
        "train": {"entry_count": 1, "entry_names": ["a+b"]},
        "val": {"entry_count": 1, "entry_names": ["c+d"]},
        "test": {"entry_count": 1, "entry_names": ["e+f"]}}}]}
    comps = audit.composite_inventory(inv, source, {"train": 1, "val": 1, "test": 1})
    status = tmp_path / "status.json"
    media = tmp_path / "media.jsonl"
    status.write_text(json.dumps({"status": "complete", "source": source, "failures": [], "completed": 2, "exact_byte_duplicates": []}))
    rows = [{"path": r["id"] + "/video.mp4", "split": r["split"], "status": "hashed"} for r in comps[:2]]
    media.write_text("\n".join(json.dumps(r) for r in rows))
    assert audit.reuse_composite_audit(status, media, comps, source)["completed"] == 2
    media.write_text("\n".join(json.dumps(rows[0]) for _ in range(2)))
    with pytest.raises(ValueError, match="exact current"):
        audit.reuse_composite_audit(status, media, comps, source)
    with pytest.raises(ValueError, match="population mismatch"):
        audit.composite_inventory(inv, source, {"train": 2})


def test_cli_spine_to_complete_membership_without_opening_labels(tmp_path, monkeypatch):
    originals, source, output = tmp_path / "originals", tmp_path / "composites", tmp_path / "output"
    originals.mkdir()
    for token, content in [("seen", b"encoding1"), ("alias", b"encoding1"), ("novel", b"encoding2")]:
        (originals / (token + ".mp4")).write_bytes(content)
    names = {"train": ["seen"], "val": ["seen+seen"], "test": ["alias", "novel", "missing"]}
    for split, values in names.items():
        for name in values:
            folder = source / split / name
            folder.mkdir(parents=True)
            # Deliberately invalid annotation text: successful execution cannot
            # depend on parsing this file. Composite media need not be opened.
            (folder / (name + ".json")).write_text("NOT JSON: LABEL ACCESS IS FORBIDDEN")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"sources": [{"root": str(source), "kind": "multirep_larger_split_tree",
        "splits": {s: {"entry_count": len(ns), "entry_names": ns} for s, ns in names.items()}}]}))
    status, media = tmp_path / "prior_status.json", tmp_path / "prior_media.jsonl"
    status.write_text(json.dumps({"status": "complete", "source": str(source), "failures": [], "completed": 2, "exact_byte_duplicates": []}))
    media.write_text("\n".join(json.dumps({"split": s, "path": f"{s}/{n}/{n}.mp4", "status": "hashed"})
                               for s in ("train", "val") for n in names[s]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"inventory": str(inventory), "source_roots": [str(originals)],
        "composite_root": str(source), "expected": {s: len(ns) for s, ns in names.items()},
        "composite_audit_status": str(status), "composite_audit_media": str(media)}))
    monkeypatch.setattr("sys.argv", ["audit", "--config", str(config), "--output", str(output)])
    assert audit.main() == 0
    result = json.loads((output / "membership.json").read_text())
    assert len(result["A"]) == 3
    assert {r["name"]: r["classification"] for r in result["A"]} == {
        "alias": "overlap", "novel": "B_candidate", "missing": "unknown"}
    assert len(result["development"]) == 2
