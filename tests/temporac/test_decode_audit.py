"""CPU interface tests; synthetic receipts never stand in for natural-media audit."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts" / "audit_temporac_decode.py"
spec = importlib.util.spec_from_file_location("decode_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def make_inventory(tmp_path, *, count=1):
    source, inventory = tmp_path / "source", tmp_path / "inventory"
    inventory.mkdir()
    (source / "train").mkdir(parents=True)
    (source / "val").mkdir()
    rows = []
    for i in range(count):
        path = source / "train" / f"clip{i}.mp4"
        path.write_bytes(b"synthetic fixture")
        path.with_suffix(".json").write_text(json.dumps(dict(length=3, width=16, height=8, object0={"count": "DO_NOT_EXPORT"})))
        rows.append(dict(path=f"train/clip{i}.mp4", split="train", bytes=path.stat().st_size, sha256="a" * 64, status="hashed"))
    (inventory / "media.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (inventory / "status.json").write_text(json.dumps(dict(status="complete", source=str(source.resolve()), completed=count)))
    return source, inventory, rows


def outputs(tmp_path):
    output = tmp_path / "output"
    for part in ("clocks", "logs"):
        (output / part).mkdir(parents=True)
    return output


def test_f0_empty_fixture_end_to_end(tmp_path, monkeypatch):
    source, inventory, _ = make_inventory(tmp_path, count=0)
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="ffprobe version fixture", stderr=""))
    output = tmp_path / "output"
    assert audit.main(["--source", str(source), "--inventory", str(inventory), "--output", str(output),
                       "--ffprobe", "fixture", "--expected-train", "0", "--expected-val", "0"]) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "complete" and status["completed"] == 0
    assert (output / "decode.jsonl").is_file()


def test_inventory_rejects_test_and_current_population_drift(tmp_path):
    source, inventory, rows = make_inventory(tmp_path)
    assert len(audit.load_inventory(source.resolve(), inventory, {"train": 1, "val": 0})) == 1
    (source / "val" / "new.mp4").write_bytes(b"extra")
    with pytest.raises(ValueError, match="membership"):
        audit.load_inventory(source.resolve(), inventory, {"train": 1, "val": 0})
    rows[0]["split"] = "test"
    (inventory / "media.jsonl").write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="forbidden"):
        audit.load_inventory(source.resolve(), inventory, {"train": 1, "val": 0})


def test_annotation_exports_only_three_technical_fields(tmp_path):
    source, _, _ = make_inventory(tmp_path)
    assert audit.read_technical_annotation(source / "train" / "clip0.json") == dict(length=3, width=16, height=8)


def test_clock_missing_duplicate_reverse_and_no_synthetic_gap():
    clock = audit.ClockStats()
    for tick in [0, 2, None, 2, 1, 5]:
        clock.add(tick)
    assert clock.result() == dict(valid=5, missing=1, duplicates=1, reversed=1, first_tick=0, last_tick=5,
                                  min_adjacent_delta_tick=-1, max_adjacent_delta_tick=4)


class FakeProcess:
    def __init__(self, lines, returncode=0):
        self.stdout = io.StringIO(lines)
        self.returncode = returncode

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def frame_lines(ticks=(10, 12, 14)):
    return "".join(f"frame|media_type=video|stream_index=0|width=16|height=8|pts={tick}|pkt_dts=N/A|best_effort_timestamp={tick}|pkt_duration=2\n" for tick in ticks)


def test_decode_streams_all_frames_raw_ticks_and_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *a, **k: FakeProcess(frame_lines()))
    result = audit.decode_frames("fixture", Path("unused"), tmp_path / "clock.csv", tmp_path / "stderr")
    assert result["decoded_frames"] == 3 and result["dimensions"] == {"16x8": 3}
    assert result["clocks"]["pkt_dts"]["missing"] == 3
    assert "0,0,16,8,10,,10,2" in (tmp_path / "clock.csv").read_text()
    assert result["clocks"]["pts"]["min_adjacent_delta_tick"] == 2


def metadata():
    return dict(width=16, height=8, time_base="1/20", nb_frames="3", avg_frame_rate="10/1", r_frame_rate="10/1")


def test_full_receipt_physical_clock_and_prior_hash_limitation(tmp_path, monkeypatch):
    source, _, rows = make_inventory(tmp_path)
    monkeypatch.setattr(audit, "probe_metadata", lambda *a: metadata())
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *a, **k: FakeProcess(frame_lines()))
    result = audit.audit_one(rows[0], source, outputs(tmp_path), 0, "fixture")
    assert result["status"] == "decoded" and result["findings"] == []
    assert result["physical_clock"]["min_adjacent_delta_seconds"] == 0.1
    assert result["physical_clock"]["observed_span_fps"] == 10.0
    assert result["prior_sha256"] == "a" * 64 and "no_current_rehash" in result["sha_binding"]
    assert result["stat_before"] == result["stat_after"]
    assert "DO_NOT_EXPORT" not in json.dumps(result)


def test_corrupt_annotation_does_not_prevent_full_decode(tmp_path, monkeypatch):
    source, _, rows = make_inventory(tmp_path)
    (source / "train" / "clip0.json").write_text("invalid json")
    monkeypatch.setattr(audit, "probe_metadata", lambda *a: metadata())
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *a, **k: FakeProcess(frame_lines()))
    result = audit.audit_one(rows[0], source, outputs(tmp_path), 0, "fixture")
    assert result["decode"]["decoded_frames"] == 3
    assert "annotation_unavailable_or_invalid" in result["findings"]


def test_nonmonotone_decode_exit_and_warnings_preserved(tmp_path, monkeypatch):
    source, _, rows = make_inventory(tmp_path)
    monkeypatch.setattr(audit, "probe_metadata", lambda *a: metadata())

    def popen(*args, **kwargs):
        kwargs["stderr"].write(b"fixture decode warning")
        return FakeProcess(frame_lines((10, 10, 9)), 7)

    monkeypatch.setattr(audit.subprocess, "Popen", popen)
    output = outputs(tmp_path)
    result = audit.audit_one(rows[0], source, output, 0, "fixture")
    assert set(result["findings"]) == {"decoder_nonzero_exit", "pts_missing_or_not_strictly_monotone",
                                       "best_effort_timestamp_missing_or_not_strictly_monotone", "decode_stderr_nonempty"}
    assert (output / result["decode_stderr"]).read_text() == "fixture decode warning"
    assert result["physical_clock"]["observed_span_fps"] is None


def test_size_mismatch_stat_change_and_count_mismatch_retained(tmp_path, monkeypatch):
    source, _, rows = make_inventory(tmp_path)
    rows[0]["bytes"] += 1
    monkeypatch.setattr(audit, "probe_metadata", lambda *a: metadata())

    def popen(*args, **kwargs):
        (source / rows[0]["path"]).write_bytes(b"changed media while decoding")
        return FakeProcess(frame_lines((10, 12)))

    monkeypatch.setattr(audit.subprocess, "Popen", popen)
    result = audit.audit_one(rows[0], source, outputs(tmp_path), 0, "fixture")
    assert {"prior_inventory_byte_size_mismatch", "source_stat_changed", "annotation_frame_count_mismatch",
            "stream_declared_frame_count_mismatch"}.issubset(result["findings"])


def test_missing_metadata_still_decodes_without_invented_clock(tmp_path, monkeypatch):
    source, _, rows = make_inventory(tmp_path)

    def failed(*args):
        raise ValueError("bad metadata")

    monkeypatch.setattr(audit, "probe_metadata", failed)
    monkeypatch.setattr(audit.subprocess, "Popen", lambda *a, **k: FakeProcess(frame_lines()))
    result = audit.audit_one(rows[0], source, outputs(tmp_path), 0, "fixture")
    assert result["decode"]["decoded_frames"] == 3 and "physical_clock" not in result
    assert "invalid_or_missing_stream_time_base" in result["findings"]


def test_cli_does_not_count_zero_frame_or_partial_failed_decode_as_success(tmp_path, monkeypatch):
    source, inventory, _ = make_inventory(tmp_path, count=3)
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="ffprobe version fixture", stderr=""))
    monkeypatch.setattr(audit, "probe_metadata", lambda *a: metadata())

    def popen(command, **kwargs):
        if command[-1].endswith("clip0.mp4"):
            return FakeProcess("", 0)
        if command[-1].endswith("clip1.mp4"):
            return FakeProcess(frame_lines((10,)), 7)
        return FakeProcess(frame_lines(), 0)

    monkeypatch.setattr(audit.subprocess, "Popen", popen)
    output = tmp_path / "output"
    assert audit.main(["--source", str(source), "--inventory", str(inventory), "--output", str(output),
                       "--ffprobe", "fixture", "--expected-train", "3", "--expected-val", "0"]) == 1
    status = json.loads((output / "status.json").read_text())
    rows = [json.loads(line) for line in (output / "decode.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["failed", "failed", "decoded"]
    assert "zero_decoded_frames" in rows[0]["findings"]
    assert rows[1]["decode"]["decoded_frames"] == 1 and "decoder_nonzero_exit" in rows[1]["findings"]
    assert status["completed"] == 3 and status["decoded"] == 1 and status["findings_count"] == 2
    assert status["status"] == "complete_with_findings"
