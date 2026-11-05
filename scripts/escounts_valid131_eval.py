"""Full VALID131 zero-shot two-offset inference with unchanged official model."""
import argparse
import math
from pathlib import Path

from escounts_valid131_common import (
    CONTRACT, DECODER_SHA, batch_window, configure_runtime, data_path,
    density_target, load_official, merge_predictions, score_records, sha256,
    source_identity, two_offset_tokens, valid_rows, verify_sha, write_json,
    read_json, verify_fields, save_or_verify,
)


def evaluate_one(model, token_array, row, device):
    import torch
    first, second = two_offset_tokens(token_array, row)
    x1, x2 = batch_window(first, device), batch_window(second, device)
    # Same official thw for both passes, derived from first window.
    thw = [list(first.shape[-3:])]
    placeholder = torch.zeros((1, 1, 768), device=device, dtype=torch.float32)
    shot_num = 0
    assert shot_num == 0
    with torch.no_grad():
        p1 = model(x1, placeholder, thw, shot_num=shot_num)
        p2 = model(x2, placeholder, thw, shot_num=shot_num)
        prediction = merge_predictions(p1, p2)
        raw = torch.sum(prediction, dim=1).type(torch.FloatTensor).to(device) / 100
        rounded = raw.round()
    if raw.numel() != 1 or not torch.isfinite(prediction).all() or not torch.isfinite(raw).all():
        raise ValueError(f"Nonfinite/invalid prediction for {row['name']}")
    target, target_sum = density_target(row)
    if prediction.shape[1] != target.numel():
        raise ValueError(f"Prediction/target length mismatch: {row['name']}: {prediction.shape[1]} / {target.numel()}")
    return {"name": row["name"], "window_lengths": [first.shape[1], second.shape[1]],
        "prediction_lengths": [p1.shape[1], p2.shape[1]], "merged_length": prediction.shape[1],
        "raw_count": raw.item(), "rounded_count": rounded.item(), "native_target": target_sum.item(),
        "csv_count": float(row["count"]), "density_prediction": prediction.cpu().float().tolist()[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--token-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    configure_runtime(args.output)
    identity = source_identity(args.source)
    rows = valid_rows(args.source)
    token_root = data_path(args.token_dir)
    paths = [data_path(token_root / Path(row["name"]).with_suffix(".npz")) for row in rows]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    verify_sha(args.decoder, DECODER_SHA)
    import numpy as np
    import torch
    torch.manual_seed(0)
    lineage = {"contract": CONTRACT, "source_commit": identity["commit"],
        "source_file_sha256": identity["source_file_sha256"], "parameters": identity["parameters"],
        "adapter_file_sha256": identity["adapter_file_sha256"],
        "decoder_sha256": DECODER_SHA, "torch": torch.__version__, "device": args.device}
    records, pending = [], []
    for index, (row, path) in enumerate(zip(rows, paths)):
        expected = {**lineage, "name": row["name"], "token_sha256": sha256(path)}
        row_path = Path(str(args.output) + f".row-{index + 1:03d}.json")
        if row_path.exists():
            record = read_json(row_path)
            verify_fields(record, expected)
            if not all(math.isfinite(record[key]) for key in ("raw_count", "rounded_count", "native_target", "csv_count")):
                raise ValueError(f"Invalid saved row: {row_path}")
            _, expected_target = density_target(row)
            if record["csv_count"] != float(row["count"]) or record["native_target"] != expected_target.item():
                raise ValueError(f"Saved targets differ from fixed VALID CSV: {row_path}")
            records.append(record)
        else:
            records.append(None)
            pending.append((index, row, path, expected, row_path))
    model = None
    if pending:
        Model, cfg = load_official(args.source)
        model = Model(cfg=cfg, use_precomputed=True, token_pool_ratio=0.4,
            iterative_shots=True, encodings="mae", window_size=(4, 7, 7)).to(args.device)
        weights = torch.load(args.decoder, map_location="cpu", weights_only=True)["model_state_dict"]
        if any(k.startswith("module.") for k in weights):
            raise ValueError("Expected unwrapped official decoder keys; no implicit prefix rewrite")
        # The actual checkpoint has 106 unprefixed keys. Exact restoration is
        # still verified against the instantiated official model at execution.
        model.load_state_dict(weights, strict=True)
        model.eval()
    for index, row, path, expected, row_path in pending:
        with np.load(path, allow_pickle=False) as data:
            record = evaluate_one(model, data["arr_0"], row, args.device)
        record.update(expected)
        record["absolute_error_native"] = abs(record["rounded_count"] - record["native_target"])
        record["relative_error_native"] = float(np.float32(record["absolute_error_native"]) / (np.float32(record["native_target"]) + np.float32(0.1)))
        record["obo_native"] = record["absolute_error_native"] <= 1
        record["obz_native"] = abs(round(record["rounded_count"]) - round(record["native_target"])) == 0
        records[index] = record
        write_json(row_path, record)
    if len(records) != 131 or sum(r["csv_count"] > 0 for r in records) != 130:
        raise ValueError("Full 131 / positive 130 population not preserved")
    save_or_verify(args.output, {"contract": CONTRACT, **identity, "torch": torch.__version__,
        "device": args.device, "split": "valid", "window_mode": "two_offset", "shot_num": 0,
        "decoder_sha256": DECODER_SHA, "complete": True, "records": records, "metrics": score_records(records)})


if __name__ == "__main__":
    main()
