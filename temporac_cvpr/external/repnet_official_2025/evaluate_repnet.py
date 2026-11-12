"""Full-population RepNet adapter. Inference core is verbatim author code.
All execution is server-only. No training or automatic resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path("/root/autodl-tmp/TempoRAC")
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def under_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Path outside authorized project root: {p}")
    return p


def preflight(cfg):
    if sys.platform != "linux" or not HERE.is_relative_to(ROOT):
        raise RuntimeError("This adapter executes only on the authorized Linux server.")
    records, inputs = [], {}
    seen = set()
    for pop in cfg["populations"]:
        manifest = under_root(pop["manifest"])
        all_rows = json.loads(manifest.read_text())
        rows = [r for r in all_rows if r["dataset"] == pop["dataset"] and r["split"] == pop["split"]]
        if len(rows) != pop["n"]:
            raise ValueError(f"Population mismatch: {pop['name']} {len(rows)} != {pop['n']}")
        inputs[str(manifest)] = sha(manifest)
        for r in rows:
            key = (r["dataset"], r["split"], r["video_id"])
            if key in seen:
                raise ValueError(f"Duplicate record key: {key}")
            seen.add(key)
            file = under_root(r["file"])
            count = float(r["count"])
            if not math.isfinite(count) or count < 0:
                raise ValueError(f"Invalid count: {key}")
            if not file.is_file() or file.stat().st_size != r["bytes"]:
                raise ValueError(f"Missing/size-mismatched media: {file}")
            actual = sha(file)
            if actual != r["sha256"]:
                raise ValueError(f"Media SHA mismatch: {file}")
            key_name = f"{r.get('fsplit', r['split'])}__{r['video_id']}.json"
            meta_path = under_root(Path(pop["decode_metadata_dir"]) / key_name)
            meta = json.loads(meta_path.read_text())
            if meta["dataset"] != r["dataset"] or meta["video_id"] != r["video_id"]:
                raise ValueError(f"Decode metadata identity mismatch: {meta_path}")
            expected_full = int(meta.get("frames_full_video", meta["len_reader"]))
            if expected_full <= 0 or expected_full != int(meta["len_reader"]):
                raise ValueError(f"Invalid frozen full decode length: {meta_path}")
            if "frame_range" in r and meta.get("frame_range_requested") != r["frame_range"]:
                raise ValueError(f"Frozen crop differs from manifest: {meta_path}")
            inputs[str(meta_path)] = sha(meta_path)
            records.append(dict(r, population=pop["name"],
                                expected_full_frames=expected_full,
                                expected_input_frames=int(meta["frames_decoded"]),
                                decode_metadata_file=str(meta_path),
                                decode_metadata_sha256=sha(meta_path),
                                ambiguous_input=(pop["name"] == "RepCountA"
                                  and file.stem in cfg["ambiguous_repcount_video_ids"])))
    expected = sum(p["n"] for p in cfg["populations"])
    if len(records) != expected:
        raise ValueError("Total population mismatch")
    for name, expected_sha in cfg["checkpoint_sha256"].items():
        asset = under_root(Path(cfg["checkpoint_dir"]) / name)
        if sha(asset) != expected_sha:
            raise ValueError(f"Checkpoint SHA mismatch: {asset}")
    for name, expected_sha in cfg["source_sha256"].items():
        if sha(HERE / name) != expected_sha:
            raise ValueError(f"Source SHA mismatch: {name}")
    return records, dict(manifests=inputs, sources=cfg["source_sha256"],
                         weights=cfg["checkpoint_sha256"],
                         count=expected, ambiguous_records=[
                             dict(dataset=r["dataset"], video_id=r["video_id"], file=r["file"])
                             for r in records if r["ambiguous_input"]])


def gpu_check(cfg):
    runtime = cfg["runtime"]
    if runtime["gpu"] != 1:
        raise RuntimeError("GPU0 reserved for TempoRAC; only GPU1 is authorized.")
    gate = cfg["execution_authorization"]
    receipt_path = under_root(gate["resume_receipt"])
    if sha(receipt_path) != gate["resume_receipt_sha256"]:
        raise RuntimeError("Resume authorization receipt differs from reviewed contract.")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("user_instruction") != "继续吧"
            or "official RepNet full357" not in receipt.get("scope", "")
            or receipt.get("supersedes_pause") != "resume-20261112/user_pause_receipt.json"
            or receipt.get("automatic_restart_after_future_pause") is not False):
        raise RuntimeError("Explicit user resume does not authorize this evaluation.")
    pause_path = under_root(gate["latest_pause_receipt"])
    if sha(pause_path) != gate["latest_pause_sha256"]:
        raise RuntimeError("Latest historical pause differs from authorization.")
    pause = json.loads(pause_path.read_text())
    if (pause.get("remaining_owned_live_pids") != [] or
            receipt_time(receipt["observed_utc"]) <= receipt_time(pause["observed_utc"])):
        raise RuntimeError("Explicit resume must postdate the recorded stopped pause.")
    for stop in gate["stop_receipts"]:
        stop_path = under_root(stop["path"])
        if sha(stop_path) != stop["sha256"]:
            raise RuntimeError("Historical user-stop evidence differs.")
        stopped = json.loads(stop_path.read_text())
        if stopped.get("remaining_alive") or stopped["runs_preserved"]["full_s1"]["complete"]:
            raise RuntimeError("Stopped training must remain recorded as incomplete.")
    for pid in gate["stopped_pids"]:
        if Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"Stopped process PID remains present: {pid}")
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if ("temporac_cvpr" in command and "/frozen_known_source_v1/full_s1" in command
                or "known_source_worker.py --gpu 1" in command):
            raise RuntimeError(f"Predecessor process remains: {proc.name}")
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,name",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout
    rows = [line.split(",") for line in output.strip().splitlines()]
    matching = [r for r in rows if int(r[0].strip()) == runtime["gpu"]]
    if len(matching) != 1 or int(matching[0][1].strip()) < runtime["min_gpu_free_mb"]:
        raise RuntimeError("Insufficient free GPU memory; no model was imported.")
    uuid = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    if any(line.split(",")[0].strip() == uuid for line in apps.splitlines()):
        raise RuntimeError("GPU1 has another compute owner; no model was imported.")
    return dict(gpu_snapshot=output, compute_apps=apps,
                authorization_basis="explicit_user_resume_after_user_stop_not_training_completion",
                resume_receipt_sha256=sha(receipt_path), stop_receipts=gate["stop_receipts"])


def receipt_time(value):
    # PowerShell receipts use 100 ns (7-digit) fractions; Python 3.10 accepts
    # microseconds. Normalize only that excess precision, preserving raw files.
    return datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", value).replace("Z", "+00:00"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adopt_predecessor(cfg, records, receipt, out):
    adoption = cfg["compatible_predecessor"]
    manifest_path = under_root(adoption["manifest"])
    if sha(manifest_path) != adoption["manifest_sha256"]:
        raise ValueError("Compatible predecessor manifest differs")
    manifest = json.loads(manifest_path.read_text())
    old = under_root(manifest["predecessor_output"])
    old_cfg = json.loads((old / "config.json").read_text())
    old_receipt = json.loads((old / "input_receipt.json").read_text())
    if sha(old / "config.json") != manifest["config_sha256"] or sha(old / "input_receipt.json") != manifest["input_receipt_sha256"]:
        raise ValueError("Predecessor receipts changed")
    if old_receipt["config_sha256"] != manifest["source_config_sha256"]:
        raise ValueError("Predecessor source config identity differs")
    for key in ["notebook_commit", "checkpoint_dir", "checkpoint_sha256", "populations", "ambiguous_repcount_video_ids", "inference", "runtime", "execution_authorization"]:
        if old_cfg[key] != cfg[key]:
            raise ValueError(f"Incompatible predecessor scientific/resource field: {key}")
    for key in ["manifests", "weights", "count", "ambiguous_records"]:
        if old_receipt[key] != receipt[key]:
            raise ValueError(f"Predecessor frozen inputs differ: {key}")
    for key in ["repnet_official_core.py", "frozen_metrics.py", "env-spec.json", "setup_repnet_env.py"]:
        if old_cfg["source_sha256"][key] != cfg["source_sha256"][key]:
            raise ValueError(f"Predecessor algorithm/environment differs: {key}")
    if manifest["n"] != 189 or len(manifest["records"]) != 189:
        raise ValueError("Expected exactly 189 immutable predecessor records")
    (out / "records").mkdir()
    (out / "traces").mkdir()
    for i, entry in enumerate(manifest["records"]):
        path = old / "records" / entry["record"]
        trace = old / "traces" / entry["trace"]
        if entry["index"] != i or path.name != f"{i:04d}.json" or sha(path) != entry["record_sha256"] or sha(trace) != entry["trace_sha256"]:
            raise ValueError("Predecessor record/trace bytes changed")
        previous = json.loads(path.read_text())
        for key in ["dataset", "split", "video_id", "sha256"]:
            if previous[key] != records[i][key]:
                raise ValueError("Predecessor record population identity differs")
        if previous["gt"] != float(records[i]["count"]) or not math.isfinite(previous["pred_count_continuous"]):
            raise ValueError("Predecessor target/prediction differs")
        shutil.copyfile(path, out / "records" / path.name)
        shutil.copyfile(trace, out / "traces" / trace.name)
        if sha(out / "records" / path.name) != entry["record_sha256"] or sha(out / "traces" / trace.name) != entry["trace_sha256"]:
            raise ValueError("Adopted bytes differ")
    write_json(out / "adoption_receipt.json", dict(
        status="189_unchanged_records_adopted", adopted_n=189, remaining_n=168,
        predecessor_code_commit=manifest["predecessor_code_commit"],
        predecessor_output=str(old), predecessor_manifest_sha256=sha(manifest_path),
        predecessor_config_snapshot_sha256=manifest["config_sha256"],
        predecessor_source_config_sha256=manifest["source_config_sha256"],
        new_config_snapshot_sha256=sha(out / "config.json"),
        new_source_config_sha256=receipt["config_sha256"],
        predecessor_sources=old_cfg["source_sha256"], new_sources=cfg["source_sha256"],
        compatible_repair=cfg["decode_completeness_exception"], records=manifest["records"]))


def load_model(cfg, out):
    runtime = cfg["runtime"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime["gpu"])
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    import numpy as np
    import tensorflow as tf
    import keras
    if sys.version_info[:2] != (3, 10) or tf.__version__ != "2.16.1" or keras.__version__ != "3.3.3" or np.__version__ != "1.26.4":
        raise RuntimeError("Installed Python/TF/Keras/NumPy differ from the frozen environment.")
    tf.random.set_seed(runtime["seed"])
    np.random.seed(runtime["seed"])
    tf.config.threading.set_intra_op_parallelism_threads(runtime["cpu_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError(f"Expected one visible GPU, got {devices}")
    tf.config.set_logical_device_configuration(
        devices[0], [tf.config.LogicalDeviceConfiguration(
            memory_limit=runtime["gpu_memory_limit_mb"])])
    core = load_module("repnet_official_core", HERE / "repnet_official_core.py")
    start = time.monotonic()
    model = core.get_repnet_model(cfg["checkpoint_dir"])
    mappings = core.MAPPING_NEW_TO_OLD_LAYER_NAMES
    if len(mappings) != len(model.weights):
        raise ValueError(f"Mapping does not cover all model weights: {len(mappings)} vs {len(model.weights)}")
    reader = tf.train.load_checkpoint(str(Path(cfg["checkpoint_dir"]) / "ckpt-70"))
    restored = []
    # Same official positional mapping; stricter completeness and exact equality.
    # Exact shapes and complete equality supplement the literal Keras3 core.
    for ((new_name, old_name), weight) in zip(mappings, model.weights):
        key = "model/" + old_name
        value = reader.get_tensor(key)
        if tuple(weight.shape) != tuple(value.shape):
            raise ValueError(f"Shape mismatch at {weight.name}: {weight.shape} vs {value.shape}")
        weight.assign(value)
        if not np.array_equal(weight.numpy(), value):
            raise ValueError(f"Restore equality failed at {weight.name}")
        restored.append(dict(variable=weight.name, official_mapping_name=new_name,
                             checkpoint_key=key, shape=list(value.shape)))
    write_json(out / "restore.json", dict(all_weights_exact=True, count=len(restored),
               variables=restored, tensorflow=tf.__version__,
               keras=getattr(tf.keras, "__version__", "bundled"),
               load_seconds=time.monotonic() - start))
    return core, model, tf, np


def run(cfg, records, out, resume):
    gpu_before = gpu_check(cfg)
    core, model, tf, np = load_model(cfg, out)
    frozen_metrics = load_module("frozen_metrics", HERE / "frozen_metrics.py")
    predictions = []
    traces = out / "traces"
    traces.mkdir(exist_ok=resume)
    row_dir = out / "records"
    row_dir.mkdir(exist_ok=resume)
    for idx, row in enumerate(records):
            record_path = row_dir / f"{idx:04d}.json"
            trace_name = f"{idx:04d}.npz"
            if resume and record_path.exists():
                result = json.loads(record_path.read_text())
                for field in ["dataset", "split", "video_id", "sha256"]:
                    if result[field] != row[field]:
                        raise ValueError(f"Resume record identity mismatch: {record_path}")
                if result["gt"] != float(row["count"]) or not math.isfinite(result["pred_count_continuous"]):
                    raise ValueError(f"Resume prediction mismatch: {record_path}")
                if sha(traces / result["trace"]) != result["trace_sha256"]:
                    raise ValueError(f"Resume trace missing or corrupt: {record_path}")
                predictions.append(result)
                continue
            started = time.monotonic()
            frames, fps = core.read_video(row["file"])
            full_frames = len(frames)
            decode_exception = None
            if full_frames != row["expected_full_frames"]:
                exception = cfg["decode_completeness_exception"]
                if not (row["dataset"] == "UCFRep" and row["video_id"] == exception["video_id"]
                        and row["sha256"] == exception["media_sha256"]
                        and row["decode_metadata_sha256"] == exception["metadata_sha256"]
                        and row["expected_full_frames"] == exception["frozen_full_frames"] == 599
                        and full_frames == exception["official_full_frames"] == 598
                        and row.get("frame_range") == exception["roi"] == [0, 583]
                        and row["expected_input_frames"] == exception["roi_frames"] == 584):
                    raise ValueError(f"Incomplete/disagreeing decode: {row['file']} {full_frames} != {row['expected_full_frames']}")
                decode_exception = exception
            requested = row.get("frame_range")
            effective = None
            if requested is not None:
                a, b = [int(x) for x in requested]
                if a < 0 or a >= full_frames or b < a:
                    raise ValueError(f"Invalid frozen crop: {row['file']} {requested}")
                effective = [a, min(b, full_frames - 1)]
                frames = np.ascontiguousarray(frames[effective[0]:effective[1] + 1])
            n_frames = len(frames)
            if n_frames != row["expected_input_frames"]:
                raise ValueError(f"Model input length differs from frozen protocol: {row['file']}")
            if decode_exception and hashlib.sha256(frames.tobytes()).hexdigest() != decode_exception["roi_sha256"]:
                raise ValueError("Evidenced exact-row ROI bytes changed")
            if n_frames == 0 or not np.isfinite(frames).all():
                raise ValueError(f"Empty/nonfinite decoded video: {row['file']}")
            decode_seconds = time.monotonic() - started
            model_started = time.monotonic()
            period, score, within, per_frame, stride = core.get_counts(
                model, frames, **cfg["inference"])
            count = float(np.sum(per_frame))
            score = float(score)
            if not math.isfinite(count) or not math.isfinite(score) or count < 0:
                raise ValueError(f"Invalid prediction for {row['file']}")
            temporary_trace = traces / (trace_name + ".tmp")
            with temporary_trace.open("wb") as trace_file:
                np.savez_compressed(trace_file,
                                per_frame_counts=np.asarray(per_frame),
                                within_period=np.asarray(within))
            temporary_trace.replace(traces / trace_name)
            result = dict(population=row["population"], dataset=row["dataset"],
                          split=row["split"], video_id=row["video_id"],
                          file=row["file"], sha256=row["sha256"], gt=float(row["count"]),
                          pred_count_continuous=count,
                          pred_count_rounded=float(np.round(count)), score=score,
                          period=float(period) if math.isfinite(float(period)) else None,
                          chosen_stride=int(stride), decoded_frames=n_frames,
                          frames_full_video=full_frames,
                          frame_range_requested=requested, frame_range_effective=effective,
                          frame_range_clamped=(requested is not None and effective[1] != requested[1]),
                          decode_completeness_exception=decode_exception,
                          decode_metadata_sha256=row["decode_metadata_sha256"],
                          fps=float(fps), decode_seconds=decode_seconds,
                          inference_seconds=time.monotonic() - model_started,
                          ambiguous_input=row["ambiguous_input"], trace=trace_name,
                          trace_sha256=sha(traces / trace_name))
            write_json(record_path, result)
            predictions.append(result)
            write_json(out / "progress.json", dict(completed=len(predictions), total=len(records)))
            print(f"{idx + 1}/{len(records)} {row['population']} {row['video_id']} "
                  f"pred={count:.6f} elapsed={time.monotonic()-started:.2f}", flush=True)
            del frames, within, per_frame
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for result in predictions:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
    groups = {}
    for pop in cfg["populations"]:
        rows = [r for r in predictions if r["population"] == pop["name"]]
        pred = np.array([r["pred_count_continuous"] for r in rows])
        gt = np.array([r["gt"] for r in rows])
        pos = gt > 0
        groups[pop["name"]] = {
            "n": len(rows),
            "harmonized": frozen_metrics.count_metrics(pred, gt),
            "derived_native_continuous": {
                "note": "Author notebook returns continuous counts, no dataset metric. These are derived metrics.",
                "mae_positive_over_gt": float(np.mean(np.abs(pred[pos] - gt[pos]) / gt[pos])),
                "obo_positive_continuous": float(np.mean(np.abs(pred[pos] - gt[pos]) <= 1)),
            },
            "zero_count_records": [r for r in rows if r["gt"] == 0],
            "ambiguous_records": [r for r in rows if r["ambiguous_input"]],
        }
    write_json(out / "metrics.json", groups)
    write_json(out / "completion.json", dict(status="complete", n=len(predictions),
               gpu_before=gpu_before,
               peak_memory=tf.config.experimental.get_memory_info("GPU:0")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["preflight", "witness", "evaluate"], required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse exact-matching atomic per-video results")
    parser.add_argument("--adopt-predecessor", action="store_true", help="Explicit compatible adoption into a new output")
    args = parser.parse_args()
    config_path = under_root(args.config)
    cfg = json.loads(config_path.read_text())
    output = under_root(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    try:
        records, receipt = preflight(cfg)
        frozen_receipt = dict(receipt, config_sha256=sha(config_path))
        if args.resume:
            if json.loads((output / "config.json").read_text()) != cfg or json.loads((output / "input_receipt.json").read_text()) != frozen_receipt:
                raise ValueError("Resume contract/input/source receipt differs")
        else:
            write_json(output / "config.json", cfg)
            write_json(output / "input_receipt.json", frozen_receipt)
        if args.adopt_predecessor:
            if args.resume or args.mode != "evaluate":
                raise ValueError("Adoption requires a fresh evaluation output")
            adopt_predecessor(cfg, records, frozen_receipt, output)
        if args.mode == "preflight":
            print("PREFLIGHT_OK", len(records), flush=True)
        elif args.mode == "witness":
            gpu_check(cfg)
            core, model, tf, np = load_model(cfg, output)
            result = model(tf.zeros((1, 64, 112, 112, 3)), training=False)
            if not all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in result):
                raise ValueError("Nonfinite seeded model witness")
            write_json(output / "witness.json", dict(status="complete",
                       shapes=[x.shape.as_list() for x in result],
                       peak_memory=tf.config.experimental.get_memory_info("GPU:0")))
            print("WITNESS_OK", flush=True)
        else:
            run(cfg, records, output, args.resume or args.adopt_predecessor)
    except BaseException:
        write_json(output / "failure.json", dict(status="failed", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()

