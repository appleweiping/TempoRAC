"""Serial CPU full-frame audit; only train/val technical annotation fields."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path

SCHEMA = "temporac.decode-audit.v1"
FRAME_FIELDS = ("pts", "pkt_dts", "best_effort_timestamp", "pkt_duration")
STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def snapshot(path: Path) -> dict:
    stat = path.stat()
    return {key: getattr(stat, key) for key in STAT_FIELDS}


def run_options() -> dict:
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def load_inventory(source: Path, inventory: Path, expected: dict) -> list[dict]:
    status = json.loads((inventory / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or Path(status["source"]).resolve() != source:
        raise ValueError("prior byte inventory must be complete and refer to this source")
    records = [json.loads(line) for line in (inventory / "media.jsonl").read_text(encoding="utf-8").splitlines() if line]
    counts = {split: 0 for split in expected}
    seen = set()
    for item in records:
        relative = Path(item["path"])
        split = item.get("split")
        if split not in counts or relative.is_absolute() or ".." in relative.parts or relative.parts[0] != split:
            raise ValueError("inventory contains invalid or forbidden split/path")
        path = (source / relative).resolve(strict=True)
        if not path.is_relative_to(source / split) or relative.suffix.lower() != ".mp4":
            raise ValueError("inventory path escapes its train/val split or is not MP4")
        sha = item.get("sha256", "")
        if item.get("status") != "hashed" or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("inventory row is not a successfully hashed media record")
        if item["path"] in seen:
            raise ValueError("duplicate inventory path")
        seen.add(item["path"])
        counts[split] += 1
    if counts != expected or status.get("completed") != len(records):
        raise ValueError(f"population mismatch: expected {expected}, actual {counts}")
    current = {path.relative_to(source).as_posix() for split in expected for path in (source / split).rglob("*.mp4")}
    if seen != current:
        raise ValueError("current train/val MP4 membership differs from prior inventory")
    return records


def read_technical_annotation(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {key: data[key] for key in ("length", "width", "height")}
    if any(type(value) is not int or value <= 0 for value in result.values()):
        raise ValueError("length/width/height must be positive integers")
    return result


class ClockStats:
    def __init__(self) -> None:
        self.missing = 0
        self.duplicates = 0
        self.reversed = 0
        self.first = None
        self.last = None
        self.previous = None
        self.minimum_delta = None
        self.maximum_delta = None
        self.valid = 0

    def add(self, value: int | None) -> None:
        if value is None:
            self.missing += 1
            # Do not count a gap across a missing timestamp as a frame interval.
            self.previous = None
            return
        self.valid += 1
        if self.first is None:
            self.first = value
        if self.last is not None:
            self.duplicates += int(value == self.last)
            self.reversed += int(value < self.last)
        if self.previous is not None:
            delta = value - self.previous
            self.minimum_delta = delta if self.minimum_delta is None else min(self.minimum_delta, delta)
            self.maximum_delta = delta if self.maximum_delta is None else max(self.maximum_delta, delta)
        self.last = self.previous = value

    def result(self) -> dict:
        return dict(valid=self.valid, missing=self.missing, duplicates=self.duplicates, reversed=self.reversed,
                    first_tick=self.first, last_tick=self.last,
                    min_adjacent_delta_tick=self.minimum_delta, max_adjacent_delta_tick=self.maximum_delta)


def integer(value: str | None) -> int | None:
    return None if value in (None, "", "N/A") else int(value)


def probe_metadata(ffprobe: str, path: Path, stderr: Path) -> dict:
    command = [ffprobe, "-v", "warning", "-threads", "1", "-select_streams", "v:0", "-show_streams",
               "-show_entries", "stream=index,codec_name,width,height,time_base,r_frame_rate,avg_frame_rate,nb_frames,duration,start_time",
               "-of", "json", str(path)]
    with stderr.open("wb") as log:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, check=False, **run_options())
    if result.returncode:
        raise ValueError(f"metadata probe exit {result.returncode}; see stderr log")
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("exactly one selected v:0 stream required")
    return streams[0]


def decode_frames(ffprobe: str, path: Path, clock_path: Path, stderr: Path) -> dict:
    command = [ffprobe, "-v", "warning", "-threads", "1", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "frame=media_type,stream_index,width,height,pts,pkt_dts,best_effort_timestamp,pkt_duration",
               "-of", "compact=p=1:nk=0", str(path)]
    clocks = {key: ClockStats() for key in FRAME_FIELDS[:3]}
    dimensions = {}
    count = 0
    parse_errors = []
    with stderr.open("wb") as log, clock_path.open("x", newline="", encoding="utf-8") as clock:
        writer = csv.writer(clock)
        writer.writerow(("frame_index", "stream_index", "width", "height", *FRAME_FIELDS))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                                   errors="replace", **run_options())
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if not line.startswith("frame|"):
                    continue
                fields = dict(part.split("=", 1) for part in line.rstrip().split("|")[1:] if "=" in part)
                if fields.get("media_type") != "video":
                    continue
                count += 1
                try:
                    width, height = int(fields["width"]), int(fields["height"])
                    ticks = {key: integer(fields.get(key)) for key in FRAME_FIELDS}
                    dimensions[f"{width}x{height}"] = dimensions.get(f"{width}x{height}", 0) + 1
                    for key, values in clocks.items():
                        values.add(ticks[key])
                    writer.writerow((count - 1, fields.get("stream_index", ""), width, height,
                                     *("" if ticks[key] is None else ticks[key] for key in FRAME_FIELDS)))
                except (KeyError, ValueError) as exc:
                    parse_errors.append({"frame_index": count - 1, "error": str(exc), "raw": line.rstrip()})
                    for values in clocks.values():
                        values.add(None)
                    writer.writerow((count - 1, fields.get("stream_index", ""), fields.get("width", ""),
                                     fields.get("height", ""), *(fields.get(key, "") for key in FRAME_FIELDS)))
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
    return dict(decoded_frames=count, dimensions=dimensions, clocks={key: values.result() for key, values in clocks.items()},
                returncode=returncode, parse_errors=parse_errors)


def audit_one(item: dict, source: Path, output: Path, index: int, ffprobe: str) -> dict:
    path = source / item["path"]
    record = dict(split=item["split"], path=item["path"], prior_sha256=item["sha256"], prior_bytes=item["bytes"],
                  sha_binding="prior_inventory_sha_plus_current_size_and_stable_stat_only; no_current_rehash",
                  findings=[], status="failed")
    prefix = f"{index:04d}"
    metadata_log, decode_log = (output / "logs" / f"{prefix}.{stage}.stderr" for stage in ("metadata", "decode"))
    clock_path = output / "clocks" / f"{prefix}.csv"
    record.update(metadata_stderr=str(metadata_log.relative_to(output)), decode_stderr=str(decode_log.relative_to(output)),
                  frame_clock_csv=str(clock_path.relative_to(output)))
    started = time.monotonic()
    try:
        record["stat_before"] = snapshot(path)
        if record["stat_before"]["st_size"] != item["bytes"]:
            record["findings"].append("prior_inventory_byte_size_mismatch")
        try:
            record["annotation"] = read_technical_annotation(path.with_suffix(".json"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record["annotation_error"] = str(exc)
            record["findings"].append("annotation_unavailable_or_invalid")
        try:
            metadata = record["stream"] = probe_metadata(ffprobe, path, metadata_log)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            metadata = {}
            record["metadata_error"] = str(exc)
            record["findings"].append("stream_metadata_unavailable_or_invalid")
        decoded = record["decode"] = decode_frames(ffprobe, path, clock_path, decode_log)
        if decoded["returncode"]:
            record["findings"].append("decoder_nonzero_exit")
        if not decoded["decoded_frames"]:
            record["findings"].append("zero_decoded_frames")
        if decoded["parse_errors"]:
            record["findings"].append("decoded_frame_parse_errors")
        annotation = record.get("annotation")
        if annotation is not None and decoded["decoded_frames"] != annotation["length"]:
            record["findings"].append("annotation_frame_count_mismatch")
        if annotation is not None and decoded["dimensions"] != {f"{annotation['width']}x{annotation['height']}": decoded["decoded_frames"]}:
            record["findings"].append("annotation_or_variable_dimension_mismatch")
        declared = integer(metadata.get("nb_frames"))
        if declared is not None and declared != decoded["decoded_frames"]:
            record["findings"].append("stream_declared_frame_count_mismatch")
        for key in ("pts", "best_effort_timestamp"):
            if any(decoded["clocks"][key][field] for field in ("missing", "duplicates", "reversed")):
                record["findings"].append(f"{key}_missing_or_not_strictly_monotone")
        try:
            time_base = Fraction(metadata["time_base"])
            if time_base <= 0:
                raise ValueError("nonpositive time_base")
            clock = decoded["clocks"]["best_effort_timestamp"]
            minimum, maximum = clock["min_adjacent_delta_tick"], clock["max_adjacent_delta_tick"]
            record["physical_clock"] = dict(unit="seconds", source="decoded best_effort_timestamp * stream time_base",
                                            time_base=metadata["time_base"],
                                            min_adjacent_delta_seconds=None if minimum is None else float(minimum * time_base),
                                            max_adjacent_delta_seconds=None if maximum is None else float(maximum * time_base),
                                            observed_span_fps=None if clock["missing"] or clock["duplicates"] or clock["reversed"] or clock["valid"] < 2 else
                                            float(Fraction(clock["valid"] - 1, clock["last_tick"] - clock["first_tick"]) / time_base))
        except (KeyError, ValueError, ZeroDivisionError):
            record["findings"].append("invalid_or_missing_stream_time_base")
        record["status"] = "decoded" if decoded["returncode"] == 0 and decoded["decoded_frames"] > 0 and not decoded["parse_errors"] else "failed"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["findings"].append("audit_exception")
    finally:
        try:
            record["stat_after"] = snapshot(path)
            if record.get("stat_before") != record["stat_after"]:
                record["findings"].append("source_stat_changed")
                record["status"] = "failed"
        except OSError as exc:
            record["findings"].append("source_stat_after_unavailable")
            record["status"] = "failed"
            record["stat_after_error"] = str(exc)
        for stage, log in (("metadata", metadata_log), ("decode", decode_log)):
            if log.exists() and log.stat().st_size:
                record["findings"].append(f"{stage}_stderr_nonempty")
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--expected-train", type=int, default=794)
    parser.add_argument("--expected-val", type=int, default=183)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(source) or output.exists():
        parser.error("output must be new and outside read-only source")
    expected = {"train": args.expected_train, "val": args.expected_val}
    try:
        records = load_inventory(source, args.inventory, expected)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    try:
        version = subprocess.run([args.ffprobe, "-version"], capture_output=True, text=True, check=False, **run_options())
        if version.returncode or not version.stdout.startswith("ffprobe version"):
            raise ValueError("ffprobe backend unavailable")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True)
    (output / "clocks").mkdir()
    (output / "logs").mkdir()
    started = time.time()
    status = dict(schema_version=SCHEMA, status="running", source=str(source), inventory=str(args.inventory.resolve()),
                  expected=expected, completed=0, decoded=0, findings_count=0, findings=[], started_unix=started,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), ffprobe=args.ffprobe,
                  ffprobe_version=version.stdout, ffprobe_version_stderr=version.stderr,
                  cpu_only=True, workers=1, decoder_threads=1,
                  limitations=["Prior media SHA copied without rehash; same-size changes since hash pass are not excluded.",
                               "Best-effort timestamps may be reconstructed by decoder; raw PTS/DTS retained separately.",
                               "No test media/labels, person labels, training, or model metrics."])
    write_json(output / "status.json", status)
    with (output / "decode.jsonl").open("x", encoding="utf-8") as stream:
        for index, item in enumerate(records):
            record = audit_one(item, source, output, index, args.ffprobe)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            status["completed"] += 1
            status["decoded"] += int(record["status"] == "decoded")
            if record["findings"]:
                status["findings"].append({"path": item["path"], "findings": record["findings"]})
                status["findings_count"] += 1
            status["elapsed_seconds"] = round(time.time() - started, 3)
            write_json(output / "status.json", status)
    status["elapsed_seconds"] = round(time.time() - started, 3)
    status["status"] = "complete_with_findings" if status["findings_count"] else "complete"
    write_json(output / "status.json", status)
    print(json.dumps({key: status[key] for key in ("status", "completed", "decoded", "findings_count", "elapsed_seconds")}), flush=True)
    return int(bool(status["findings_count"]))


if __name__ == "__main__":
    raise SystemExit(main())
