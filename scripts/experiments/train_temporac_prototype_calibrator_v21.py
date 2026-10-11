import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from train_skeleton_spectral_calibrator_v21 import (
    build_matrix,
    compute_metrics,
    load_base_predictions,
    load_split,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--train_predictions", required=True)
    parser.add_argument("--val_predictions", required=True)
    parser.add_argument("--test_predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--refit_train_val", action="store_true")
    return parser.parse_args()


def source_id(metadata):
    source_ids = metadata["video_name"].removesuffix(".mp4").split("+")
    return source_ids[metadata["person_index"]]


def nearest_predictions(reference_x, reference_y, query_x, scaler, neighbors):
    scaled_reference = scaler.fit_transform(reference_x)
    scaled_query = scaler.transform(query_x)
    distances = np.sqrt(
        np.mean(
            (scaled_query[:, None, :] - scaled_reference[None, :, :]) ** 2,
            axis=-1,
        )
    )
    indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    return np.median(reference_y[indices], axis=1), distances, indices


def inner_validation_predictions(features, targets, groups, metadata, scaler, neighbors):
    scaled = scaler.fit_transform(features)
    distances = np.sqrt(
        np.mean((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=-1)
    )
    videos = np.asarray([row["video_name"] for row in metadata])
    distances[videos[:, None] == videos[None, :]] = np.inf
    indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    predictions = np.median(targets[indices], axis=1)
    return predictions, compute_metrics(predictions, targets, groups)


def selection_key(metrics):
    within_constraint = metrics["avg_mae"] <= 0.223
    return (
        int(within_constraint),
        metrics["avg_obo"] if within_constraint else -metrics["avg_mae"],
        -metrics["avg_mae"],
        -metrics["mae"],
    )


def merge_counting_predictions(path, metadata, predictions, output_path, count_key):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    values = {}
    for row, prediction in zip(metadata, predictions):
        values[(row["video_name"], row["person_index"])] = float(prediction)
    for row in rows:
        row[count_key] = [
            values[(row["video_name"], person_index)]
            for person_index in range(int(row["num_persons"]))
        ]
    Path(output_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for split in ["train", "val", "test"]:
        samples = load_split(getattr(args, f"{split}_data"))
        base = load_base_predictions(getattr(args, f"{split}_predictions"))
        payloads[split] = build_matrix(samples, base)

    train_x, train_y, train_groups, train_metadata = payloads["train"]
    candidates = []
    selected = None
    for scaler_name, scaler in [
        ("standard", StandardScaler()),
        ("robust", RobustScaler()),
    ]:
        for neighbors in [1, 3, 5]:
            _, metrics = inner_validation_predictions(
                train_x,
                train_y,
                train_groups,
                train_metadata,
                scaler,
                neighbors,
            )
            candidate = {
                "scaler": scaler_name,
                "neighbors": neighbors,
                "inner_train_metrics": metrics,
            }
            candidates.append(candidate)
            if selected is None or selection_key(metrics) > selection_key(
                selected["inner_train_metrics"]
            ):
                selected = candidate

    scaler_type = StandardScaler if selected["scaler"] == "standard" else RobustScaler
    val_x, val_y, val_groups, val_metadata = payloads["val"]
    val_prediction, _, _ = nearest_predictions(
        train_x,
        train_y,
        val_x,
        scaler_type(),
        selected["neighbors"],
    )
    val_metrics = compute_metrics(val_prediction, val_y, val_groups)

    final_x = train_x
    final_y = train_y
    final_metadata = train_metadata
    if args.refit_train_val:
        final_x = np.concatenate([train_x, val_x], axis=0)
        final_y = np.concatenate([train_y, val_y], axis=0)
        final_metadata = train_metadata + val_metadata
    test_x, test_y, test_groups, test_metadata = payloads["test"]
    final_scaler = scaler_type()
    test_prediction, test_distances, test_indices = nearest_predictions(
        final_x,
        final_y,
        test_x,
        final_scaler,
        selected["neighbors"],
    )
    test_metrics = compute_metrics(test_prediction, test_y, test_groups)
    baseline_metrics = compute_metrics(test_x[:, 0], test_y, test_groups)

    nearest_rows = []
    for row, target, prediction, indices, distances in zip(
        test_metadata,
        test_y,
        test_prediction,
        test_indices,
        test_distances,
    ):
        sorted_indices = indices[np.argsort(distances[indices])]
        nearest_rows.append(
            {
                **row,
                "source_id": source_id(row),
                "gt_count": float(target),
                "prototype_count": float(prediction),
                "neighbors": [
                    {
                        "video_name": final_metadata[int(index)]["video_name"],
                        "person_index": final_metadata[int(index)]["person_index"],
                        "source_id": source_id(final_metadata[int(index)]),
                        "count": float(final_y[int(index)]),
                        "distance": float(distances[int(index)]),
                    }
                    for index in sorted_indices
                ],
            }
        )

    report = {
        "selection_protocol": "select scaler and k by train-only leave-one-composition-out validation; optionally refit prototypes on source train+val; evaluate test once",
        "refit_train_val": bool(args.refit_train_val),
        "candidates": candidates,
        "selected": selected,
        "source_id_overlap": {
            "train_val": sorted(
                set(map(source_id, train_metadata))
                & set(map(source_id, val_metadata))
            ),
            "train_test": sorted(
                set(map(source_id, train_metadata))
                & set(map(source_id, test_metadata))
            ),
            "val_test": sorted(
                set(map(source_id, val_metadata))
                & set(map(source_id, test_metadata))
            ),
        },
        "val_metrics_before_refit": val_metrics,
        "baseline_test_metrics": baseline_metrics,
        "prototype_test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "test_neighbors.json").write_text(
        json.dumps(nearest_rows, indent=2), encoding="utf-8"
    )
    merge_counting_predictions(
        args.val_predictions,
        val_metadata,
        val_prediction,
        output_dir / "val_predictions_counting.json",
        "pred_count_prototype_per_person",
    )
    merge_counting_predictions(
        args.test_predictions,
        test_metadata,
        test_prediction,
        output_dir / "test_predictions_counting.json",
        "pred_count_prototype_per_person",
    )
    joblib.dump(
        {
            "scaler": final_scaler,
            "features": final_scaler.transform(final_x),
            "targets": final_y,
            "metadata": final_metadata,
            "neighbors": selected["neighbors"],
        },
        output_dir / "prototype_calibrator.joblib",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
