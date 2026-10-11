import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--train_predictions", required=True)
    parser.add_argument("--val_predictions", required=True)
    parser.add_argument("--test_predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--refit_train_val",
        action="store_true",
        help="After validation-only selection, refit the selected estimator on source train+val.",
    )
    return parser.parse_args()


def load_split(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if len(payload) != 1:
            key = Path(path).stem
            return payload[key]
        return next(iter(payload.values()))
    return payload


def load_base_predictions(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup = {}
    for row in rows:
        values = row.get("pred_direct_count_per_person", row["pred_count_per_person"])
        for person_index, value in enumerate(values):
            lookup[(row["video_name"], person_index)] = float(value)
    return lookup


def spectral_summary(signals):
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, None]
    length = signals.shape[0]
    if length < 8 or signals.shape[1] == 0:
        return np.zeros(18, dtype=np.float64)
    time = np.linspace(-1.0, 1.0, length)
    design = np.stack([time, np.ones_like(time)], axis=1)
    coefficients = np.linalg.lstsq(design, signals, rcond=None)[0]
    signals = signals - design @ coefficients
    scale = signals.std(axis=0, keepdims=True)
    active = scale.reshape(-1) > 1e-5
    if not np.any(active):
        return np.zeros(18, dtype=np.float64)
    signals = signals[:, active] / scale[:, active]
    window = np.hanning(length)[:, None]
    power = np.abs(np.fft.rfft(signals * window, axis=0)) ** 2
    power[0] = 0.0
    normalized = power / np.maximum(power.sum(axis=0, keepdims=True), 1e-8)
    aggregate_mean = normalized.mean(axis=1)
    aggregate_median = np.median(normalized, axis=1)
    frequencies = np.arange(power.shape[0], dtype=np.float64)

    def describe(spectrum):
        if spectrum.size <= 1 or spectrum.sum() <= 1e-8:
            return np.zeros(9, dtype=np.float64)
        order = np.argsort(spectrum[1:])[::-1] + 1
        top = np.pad(order[:4], (0, max(0, 4 - len(order))), constant_values=0)
        total = max(float(spectrum.sum()), 1e-8)
        centroid = float((frequencies * spectrum).sum() / total)
        variance = float((((frequencies - centroid) ** 2) * spectrum).sum() / total)
        probabilities = spectrum / total
        entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
        return np.asarray(
            [
                float(top[0]),
                float(top[1]),
                float(top[2]),
                float(top[3]),
                centroid,
                np.sqrt(max(variance, 0.0)),
                float(spectrum[top[0]] / total),
                float(spectrum[top[1]] / total),
                entropy,
            ]
        )

    return np.concatenate([describe(aggregate_mean), describe(aggregate_median)])


def autocorrelation_summary(signals):
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[:, None]
    length = signals.shape[0]
    if length < 8:
        return np.zeros(12, dtype=np.float64)
    signals = signals - signals.mean(axis=0, keepdims=True)
    scale = signals.std(axis=0, keepdims=True)
    active = scale.reshape(-1) > 1e-5
    if not np.any(active):
        return np.zeros(12, dtype=np.float64)
    signals = signals[:, active] / scale[:, active]
    fft_size = 1 << int(np.ceil(np.log2(2 * length - 1)))
    spectrum = np.fft.rfft(signals, n=fft_size, axis=0)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size, axis=0)[:length]
    correlation = correlation / np.maximum(correlation[0:1], 1e-8)
    aggregate = np.median(correlation, axis=1)
    max_lag = max(3, min(length // 2, 96))
    values = aggregate[2 : max_lag + 1]
    if values.size == 0:
        return np.zeros(12, dtype=np.float64)
    order = np.argsort(values)[::-1]
    lags = order[:4] + 2
    lags = np.pad(lags, (0, max(0, 4 - len(lags))), constant_values=2)
    lag_values = aggregate[lags]
    counts = length / np.maximum(lags, 1)
    return np.concatenate(
        [
            lags.astype(np.float64),
            counts.astype(np.float64),
            lag_values.astype(np.float64),
        ]
    )


def extract_features(motion, frame_mask, base_count):
    motion = np.asarray(motion, dtype=np.float64)
    frame_mask = np.asarray(frame_mask, dtype=bool)
    valid_indices = np.flatnonzero(frame_mask)
    if valid_indices.size < 8:
        valid_indices = np.arange(motion.shape[0])
    sequence = motion[valid_indices]
    coordinates = sequence[..., :2]
    confidence = np.clip(sequence[..., 2], 0.0, 1.0)
    visible = confidence > 1e-4
    center = np.zeros((len(sequence), 1, 2), dtype=np.float64)
    for frame_index in range(len(sequence)):
        mask = visible[frame_index]
        if np.any(mask):
            center[frame_index, 0] = coordinates[frame_index, mask].mean(axis=0)
    centered = coordinates - center
    body_scale = np.sqrt((centered**2).sum(axis=-1))
    scale = np.median(body_scale[visible]) if np.any(visible) else 1.0
    scale = max(float(scale), 1e-4)
    centered = centered / scale
    centered[~visible] = 0.0
    velocity = np.diff(centered, axis=0, prepend=centered[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    radial = np.sqrt((centered**2).sum(axis=-1))
    speed = np.sqrt((velocity**2).sum(axis=-1))

    scalar_features = np.asarray(
        [
            float(base_count),
            float(len(valid_indices)),
            float(len(valid_indices) / max(motion.shape[0], 1)),
            float(confidence.mean()),
            float(visible.mean()),
            float(scale),
            float(np.mean(np.abs(centered))),
            float(np.std(centered)),
            float(np.mean(np.abs(velocity))),
            float(np.std(velocity)),
            float(np.quantile(np.abs(velocity), 0.75)),
            float(np.quantile(np.abs(velocity), 0.95)),
            float(np.mean(np.abs(acceleration))),
            float(np.std(acceleration)),
            float(np.mean(speed)),
            float(np.std(speed)),
            float(np.quantile(speed, 0.75)),
            float(np.quantile(speed, 0.95)),
        ],
        dtype=np.float64,
    )
    per_joint = np.concatenate(
        [
            np.mean(np.abs(velocity), axis=(0, 2)),
            np.std(velocity, axis=(0, 2)),
            np.mean(speed, axis=0),
            np.std(speed, axis=0),
        ]
    )
    signal_groups = [
        centered.reshape(len(sequence), -1),
        velocity.reshape(len(sequence), -1),
        radial,
        speed,
    ]
    spectral = np.concatenate([spectral_summary(group) for group in signal_groups])
    autocorrelation = np.concatenate(
        [autocorrelation_summary(group) for group in signal_groups]
    )
    return np.nan_to_num(
        np.concatenate([scalar_features, per_joint, spectral, autocorrelation]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def build_matrix(samples, base_lookup):
    features = []
    targets = []
    groups = []
    metadata = []
    for sample_index, sample in enumerate(samples):
        person_mask = np.asarray(sample["person_mask"], dtype=bool)
        for person_index in np.flatnonzero(person_mask):
            key = (sample["video_name"], int(person_index))
            base_count = base_lookup[key]
            features.append(
                extract_features(
                    sample["motion"][person_index],
                    sample["frame_mask"][person_index],
                    base_count,
                )
            )
            targets.append(float(sample["count_gt"][person_index]))
            groups.append(sample_index)
            metadata.append(
                {
                    "video_name": sample["video_name"],
                    "person_index": int(person_index),
                    "base_count": base_count,
                    "source_json": sample.get("source_json", ""),
                }
            )
    return (
        np.asarray(features),
        np.asarray(targets),
        np.asarray(groups),
        metadata,
    )


def compute_metrics(predictions, targets, groups):
    video_mae = []
    video_obo = []
    avg_mae = []
    avg_obo = []
    for group in np.unique(groups):
        mask = groups == group
        pred = predictions[mask]
        target = targets[mask]
        total_error = abs(float(pred.sum() - target.sum()))
        video_mae.append(total_error / max(float(target.sum()), 1e-8))
        video_obo.append(float(total_error <= 1.0))
        errors = np.abs(pred - target)
        avg_mae.append(float(np.mean(errors / np.maximum(target, 1e-8))))
        avg_obo.append(float(np.mean(errors <= 1.0)))
    return {
        "mae": float(np.mean(video_mae)),
        "obo": float(np.mean(video_obo)),
        "avg_mae": float(np.mean(avg_mae)),
        "avg_obo": float(np.mean(avg_obo)),
        "legacy_person_mae": float(np.mean(np.abs(predictions - targets))),
    }


def candidate_models(seed):
    for minimum_leaf in [1, 2, 4, 8]:
        for maximum_features in [0.5, 0.8, 1.0]:
            yield (
                f"extra_trees_leaf{minimum_leaf}_feat{maximum_features}",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=minimum_leaf,
                    max_features=maximum_features,
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
    for minimum_leaf in [1, 2, 4, 8]:
        yield (
            f"random_forest_leaf{minimum_leaf}",
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=minimum_leaf,
                max_features=0.8,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    for depth in [1, 2, 3]:
        yield (
            f"gradient_boosting_depth{depth}",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=depth,
                loss="huber",
                random_state=seed,
            ),
        )
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        yield (
            f"ridge_{alpha}",
            make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
        )


def selection_key(metrics):
    within_constraint = metrics["avg_mae"] <= 0.40
    return (
        int(within_constraint),
        metrics["avg_obo"] if within_constraint else -metrics["avg_mae"],
        -metrics["avg_mae"],
        -metrics["mae"],
    )


def apply_selected_calibration(model, features, selected):
    raw_prediction = model.predict(features)
    prediction = (
        selected["blend"] * raw_prediction
        + (1.0 - selected["blend"]) * features[:, 0]
    )
    prediction = selected["scale"] * prediction + selected["offset"]
    prediction = np.maximum(prediction, 0.0)
    if selected["rounding"] > 0:
        prediction = (
            np.round(prediction / selected["rounding"]) * selected["rounding"]
        )
    return prediction


def prediction_rows(metadata, targets, predictions):
    return [
        {
            **item,
            "gt_count": float(target),
            "calibrated_count": float(prediction),
        }
        for item, target, prediction in zip(metadata, targets, predictions)
    ]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_payloads = {}
    for split in ["train", "val", "test"]:
        samples = load_split(getattr(args, f"{split}_data"))
        lookup = load_base_predictions(getattr(args, f"{split}_predictions"))
        split_payloads[split] = build_matrix(samples, lookup)

    train_x, train_y, _, _ = split_payloads["train"]
    val_x, val_y, val_groups, val_metadata = split_payloads["val"]
    trials = []
    best = None
    for model_name, model in candidate_models(args.seed):
        model.fit(train_x, train_y)
        raw_prediction = model.predict(val_x)
        base_prediction = val_x[:, 0]
        for blend in np.linspace(0.0, 1.0, 11):
            blended = blend * raw_prediction + (1.0 - blend) * base_prediction
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                for offset in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                    calibrated = scale * blended + offset
                    for rounding in [0.0, 0.5, 1.0]:
                        prediction = np.maximum(calibrated, 0.0)
                        if rounding > 0:
                            prediction = np.round(prediction / rounding) * rounding
                        metrics = compute_metrics(prediction, val_y, val_groups)
                        trial = {
                            "model_name": model_name,
                            "blend": float(blend),
                            "scale": float(scale),
                            "offset": float(offset),
                            "rounding": float(rounding),
                            "val_metrics": metrics,
                        }
                        trials.append(trial)
                        if best is None or selection_key(metrics) > selection_key(
                            best["val_metrics"]
                        ):
                            best = {**trial, "model": model}

    val_prediction = apply_selected_calibration(best["model"], val_x, best)
    final_model = best["model"]
    if args.refit_train_val:
        final_model = clone(best["model"])
        final_model.fit(
            np.concatenate([train_x, val_x], axis=0),
            np.concatenate([train_y, val_y], axis=0),
        )

    test_x, test_y, test_groups, test_metadata = split_payloads["test"]
    test_prediction = apply_selected_calibration(final_model, test_x, best)
    test_metrics = compute_metrics(test_prediction, test_y, test_groups)
    baseline_metrics = compute_metrics(test_x[:, 0], test_y, test_groups)

    serializable_best = {key: value for key, value in best.items() if key != "model"}
    report = {
        "selection_protocol": "fit train only; select model/blend/rounding on validation only; evaluate test once",
        "refit_train_val_after_selection": bool(args.refit_train_val),
        "feature_dim": int(train_x.shape[1]),
        "train_persons": int(train_x.shape[0]),
        "val_persons": int(val_x.shape[0]),
        "test_persons": int(test_x.shape[0]),
        "selected": serializable_best,
        "baseline_test_metrics": baseline_metrics,
        "calibrated_test_metrics": test_metrics,
    }
    Path(output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    Path(output_dir / "validation_trials.json").write_text(
        json.dumps(trials, indent=2), encoding="utf-8"
    )
    Path(output_dir / "val_predictions.json").write_text(
        json.dumps(
            prediction_rows(val_metadata, val_y, val_prediction),
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(output_dir / "test_predictions.json").write_text(
        json.dumps(
            prediction_rows(test_metadata, test_y, test_prediction),
            indent=2,
        ),
        encoding="utf-8",
    )
    joblib.dump(final_model, output_dir / "calibrator.joblib")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
