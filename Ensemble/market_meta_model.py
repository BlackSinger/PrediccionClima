from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config_store
import lightgbm_model
import weather_common

STATE_FEATURE_COLUMNS = [
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_speed_mph",
    "pressure_in",
    "wind_dir_missing",
    "wind_dir_sin",
    "wind_dir_cos",
    *weather_common.CONDITION_COLUMNS,
    "hour",
    "minute",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

DERIVED_STATE_COLUMNS = [
    "max_temperature_so_far_today_f",
    "min_temperature_so_far_today_f",
    "day_range_so_far_f",
    "observations_so_far_today",
    "hours_since_midnight",
    "hours_remaining_in_day",
    "current_minus_day_min_f",
    "current_below_day_max_so_far_f",
]

LIVE_EXCLUDED_COLUMNS = {
    "forecast_origin_datetime",
    "split",
    "forecast_day",
    "final_daily_max_f",
    "final_daily_max_c",
}


def get_default_train_config() -> dict[str, object]:
    shared = config_store.load_section("shared")
    return {
        "seed": int(shared["seed"]),
        "num_boost_round": 300,
        "early_stopping_rounds": 25,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 160,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.0,
        "num_threads": 0,
    }


DEFAULT_TRAIN_CONFIG = get_default_train_config()


def fahrenheit_to_celsius_bucket(values: np.ndarray | pd.Series | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.rint((array - 32.0) * (5.0 / 9.0)).astype(np.int32)


def clip_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def binary_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=np.float64)
    probs = clip_probabilities(y_prob)
    brier = float(np.mean(np.square(probs - labels)))
    logloss = float(
        -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
    )
    return {
        "positive_rate": float(np.mean(labels)),
        "brier": brier,
        "logloss": logloss,
    }


def multiclass_probability_metrics(
    true_index: np.ndarray,
    probability_matrix: np.ndarray,
) -> dict[str, float]:
    if len(true_index) == 0:
        return {
            "top1_accuracy": float("nan"),
            "mean_true_bin_probability": float("nan"),
            "multiclass_logloss": float("nan"),
        }

    probs = np.asarray(probability_matrix, dtype=np.float64)
    safe_index = np.asarray(true_index, dtype=np.int32)
    chosen = clip_probabilities(probs[np.arange(len(safe_index)), safe_index])
    top1 = np.argmax(probs, axis=1)
    return {
        "top1_accuracy": float(np.mean(top1 == safe_index)),
        "mean_true_bin_probability": float(np.mean(chosen)),
        "multiclass_logloss": float(-np.mean(np.log(chosen))),
    }


def fit_isotonic_calibrator(
    raw_probability: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, object]:
    scores = np.asarray(raw_probability, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.float64)
    if scores.size == 0:
        return {
            "kind": "constant",
            "x_upper_bounds": [1.0],
            "y_values": [0.5],
        }

    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    blocks: list[dict[str, float]] = []
    for score, label in zip(scores_sorted, labels_sorted, strict=False):
        blocks.append(
            {
                "sum_y": float(label),
                "weight": 1.0,
                "x_min": float(score),
                "x_max": float(score),
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["sum_y"] / left["weight"]
            right_mean = right["sum_y"] / right["weight"]
            if left_mean <= right_mean:
                break
            merged = {
                "sum_y": left["sum_y"] + right["sum_y"],
                "weight": left["weight"] + right["weight"],
                "x_min": left["x_min"],
                "x_max": right["x_max"],
            }
            blocks[-2:] = [merged]

    return {
        "kind": "isotonic",
        "x_upper_bounds": [float(block["x_max"]) for block in blocks],
        "y_values": [float(block["sum_y"] / block["weight"]) for block in blocks],
    }


def apply_isotonic_calibrator(
    raw_probability: np.ndarray,
    calibrator: dict[str, object],
) -> np.ndarray:
    scores = np.asarray(raw_probability, dtype=np.float64)
    x_upper_bounds = np.asarray(calibrator["x_upper_bounds"], dtype=np.float64)
    y_values = np.asarray(calibrator["y_values"], dtype=np.float64)
    if x_upper_bounds.size == 0 or y_values.size == 0:
        return np.full(scores.shape, 0.5, dtype=np.float64)
    bucket_index = np.searchsorted(x_upper_bounds, scores, side="left")
    bucket_index = np.clip(bucket_index, 0, len(y_values) - 1)
    return np.clip(y_values[bucket_index], 0.0, 1.0)


def enforce_monotonic_cumulative(probability_matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability_matrix, dtype=np.float64), 0.0, 1.0)
    if clipped.ndim != 2 or clipped.shape[1] == 0:
        return clipped
    return np.minimum.accumulate(clipped, axis=1)


def threshold_matrix_from_targets(
    target_c: np.ndarray,
    thresholds_c: list[int],
) -> np.ndarray:
    targets = np.asarray(target_c, dtype=np.int32).reshape(-1, 1)
    thresholds = np.asarray(thresholds_c, dtype=np.int32).reshape(1, -1)
    return (targets >= thresholds).astype(np.float32)


def bucket_distribution_from_cumulative(
    cumulative_probabilities: np.ndarray,
    support_min_c: int,
    support_max_c: int,
) -> tuple[list[str], np.ndarray]:
    matrix = np.asarray(cumulative_probabilities, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Se esperaba una matriz 2D de probabilidades acumuladas.")
    if matrix.shape[1] == 0:
        labels = [f"{support_min_c}C_or_below", f"{support_max_c}C_or_above"]
        tail = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (matrix.shape[0], 1))
        return labels, tail

    bucket_columns = [1.0 - matrix[:, 0]]
    bucket_labels = [f"{support_min_c}C_or_below"]
    for idx, threshold_c in enumerate(range(support_min_c + 1, support_max_c)):
        bucket_columns.append(matrix[:, idx] - matrix[:, idx + 1])
        bucket_labels.append(f"{threshold_c}C")
    bucket_columns.append(matrix[:, -1])
    bucket_labels.append(f"{support_max_c}C_or_above")
    distribution = np.column_stack(bucket_columns)
    distribution = np.clip(distribution, 0.0, 1.0)
    normalizer = distribution.sum(axis=1, keepdims=True)
    normalizer = np.where(normalizer <= 0.0, 1.0, normalizer)
    return bucket_labels, distribution / normalizer


def true_bucket_index(
    target_c: np.ndarray,
    support_min_c: int,
    support_max_c: int,
) -> np.ndarray:
    values = np.asarray(target_c, dtype=np.int32)
    result = np.empty_like(values, dtype=np.int32)
    result[values <= support_min_c] = 0
    result[values >= support_max_c] = support_max_c - support_min_c
    middle_mask = (values > support_min_c) & (values < support_max_c)
    result[middle_mask] = values[middle_mask] - support_min_c
    return result


def prepare_state_frame(clean_df: pd.DataFrame) -> pd.DataFrame:
    state = clean_df.copy()
    state["observation_datetime"] = pd.to_datetime(state["observation_datetime"])
    state = state.sort_values("observation_datetime").reset_index(drop=True)
    state["forecast_day"] = state["observation_datetime"].dt.normalize()
    state["max_temperature_so_far_today_f"] = (
        state.groupby("forecast_day")["temperature_f"].cummax()
    )
    state["min_temperature_so_far_today_f"] = (
        state.groupby("forecast_day")["temperature_f"].cummin()
    )
    state["day_range_so_far_f"] = (
        state["max_temperature_so_far_today_f"] - state["min_temperature_so_far_today_f"]
    )
    state["observations_so_far_today"] = state.groupby("forecast_day").cumcount() + 1
    state["hours_since_midnight"] = state["hour"] + state["minute"] / 60.0
    state["hours_remaining_in_day"] = np.maximum(0.0, 24.0 - state["hours_since_midnight"])
    state["current_minus_day_min_f"] = (
        state["temperature_f"] - state["min_temperature_so_far_today_f"]
    )
    state["current_below_day_max_so_far_f"] = (
        state["max_temperature_so_far_today_f"] - state["temperature_f"]
    )
    state["final_daily_max_f"] = state.groupby("forecast_day")["temperature_f"].transform("max")
    keep_columns = [
        "observation_datetime",
        "forecast_day",
        "final_daily_max_f",
        *STATE_FEATURE_COLUMNS,
        *DERIVED_STATE_COLUMNS,
    ]
    return state[keep_columns].drop_duplicates(subset=["observation_datetime"], keep="last")


def build_forecast_snapshot_features(ensemble_predictions: pd.DataFrame) -> pd.DataFrame:
    working = ensemble_predictions.copy()
    working["forecast_origin_datetime"] = pd.to_datetime(working["forecast_origin_datetime"])
    working["observation_datetime"] = pd.to_datetime(working["observation_datetime"])
    working = working.sort_values(
        ["forecast_origin_datetime", "split", "horizon_step"]
    ).reset_index(drop=True)
    working["origin_day"] = working["forecast_origin_datetime"].dt.normalize()
    working["target_day"] = working["observation_datetime"].dt.normalize()
    working["interval_width_f"] = (
        working["upper_prediction_interval_f"] - working["lower_prediction_interval_f"]
    )
    group_cols = ["forecast_origin_datetime", "split"]

    summary = (
        working.groupby(group_cols)
        .agg(
            horizon_count=("horizon_step", "count"),
            ensemble_pred_mean_f=("predicted_temperature_f_ensemble", "mean"),
            ensemble_pred_max_f=("predicted_temperature_f_ensemble", "max"),
            ensemble_pred_min_f=("predicted_temperature_f_ensemble", "min"),
            ensemble_pred_std_f=("predicted_temperature_f_ensemble", "std"),
            upper_pred_max_f=("upper_prediction_interval_f", "max"),
            lower_pred_min_f=("lower_prediction_interval_f", "min"),
            interval_width_mean_f=("interval_width_f", "mean"),
            interval_width_max_f=("interval_width_f", "max"),
        )
        .fillna({"ensemble_pred_std_f": 0.0})
    )

    peak_idx = working.groupby(group_cols)["predicted_temperature_f_ensemble"].idxmax()
    peak_rows = (
        working.loc[
            peak_idx,
            group_cols
            + [
                "horizon_step",
                "predicted_temperature_f_ensemble",
                "upper_prediction_interval_f",
                "lower_prediction_interval_f",
            ],
        ]
        .rename(
            columns={
                "horizon_step": "peak_horizon_step",
                "predicted_temperature_f_ensemble": "peak_prediction_f",
                "upper_prediction_interval_f": "peak_upper_interval_f",
                "lower_prediction_interval_f": "peak_lower_interval_f",
            }
        )
        .set_index(group_cols)
    )
    summary = summary.join(peak_rows)

    same_day = working[working["origin_day"] == working["target_day"]].copy()
    if same_day.empty:
        summary["same_day_forecast_count"] = 0
    else:
        same_day_summary = same_day.groupby(group_cols).agg(
            same_day_forecast_count=("horizon_step", "count"),
            same_day_pred_mean_f=("predicted_temperature_f_ensemble", "mean"),
            same_day_pred_max_f=("predicted_temperature_f_ensemble", "max"),
            same_day_pred_min_f=("predicted_temperature_f_ensemble", "min"),
            same_day_upper_max_f=("upper_prediction_interval_f", "max"),
            same_day_lower_min_f=("lower_prediction_interval_f", "min"),
            same_day_interval_width_mean_f=("interval_width_f", "mean"),
        )
        same_day_peak_idx = same_day.groupby(group_cols)["predicted_temperature_f_ensemble"].idxmax()
        same_day_peak_rows = (
            same_day.loc[
                same_day_peak_idx,
                group_cols
                + [
                    "horizon_step",
                    "predicted_temperature_f_ensemble",
                    "upper_prediction_interval_f",
                    "lower_prediction_interval_f",
                ],
            ]
            .rename(
                columns={
                    "horizon_step": "same_day_peak_horizon_step",
                    "predicted_temperature_f_ensemble": "same_day_peak_prediction_f",
                    "upper_prediction_interval_f": "same_day_peak_upper_interval_f",
                    "lower_prediction_interval_f": "same_day_peak_lower_interval_f",
                }
            )
            .set_index(group_cols)
        )
        summary = summary.join(same_day_summary).join(same_day_peak_rows)

    summary["same_day_forecast_count"] = summary["same_day_forecast_count"].fillna(0).astype(np.int32)
    summary["next_day_forecast_count"] = (
        summary["horizon_count"] - summary["same_day_forecast_count"]
    ).astype(np.int32)

    for column_name, prefix in (
        ("predicted_temperature_f_ensemble", "ensemble_prediction"),
        ("lower_prediction_interval_f", "lower_interval"),
        ("upper_prediction_interval_f", "upper_interval"),
        ("interval_width_f", "interval_width"),
    ):
        pivot = (
            working.set_index(group_cols + ["horizon_step"])[column_name]
            .unstack("horizon_step")
            .sort_index(axis=1)
        )
        pivot.columns = [
            f"{prefix}_h{int(horizon_step):02d}_f" for horizon_step in pivot.columns
        ]
        summary = summary.join(pivot)

    return summary.reset_index()


def build_training_dataset(
    ensemble_predictions: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    forecast_features = build_forecast_snapshot_features(ensemble_predictions)
    state_frame = prepare_state_frame(clean_df)
    merged = forecast_features.merge(
        state_frame,
        left_on="forecast_origin_datetime",
        right_on="observation_datetime",
        how="left",
    )
    missing_state = merged["observation_datetime"].isna().sum()
    if missing_state:
        raise ValueError(
            f"Faltan {int(missing_state)} snapshots del estado meteorologico para entrenar "
            "el meta-modelo probabilistico."
        )

    merged = merged.drop(columns=["observation_datetime"])
    merged["max_gain_vs_current_f"] = (
        merged["ensemble_pred_max_f"] - merged["temperature_f"]
    )
    merged["same_day_max_gain_vs_current_f"] = (
        merged["same_day_pred_max_f"] - merged["temperature_f"]
    )
    merged["upper_max_gain_vs_current_f"] = (
        merged["upper_pred_max_f"] - merged["temperature_f"]
    )
    merged["same_day_upper_gain_vs_current_f"] = (
        merged["same_day_upper_max_f"] - merged["temperature_f"]
    )
    merged["final_daily_max_c"] = fahrenheit_to_celsius_bucket(
        merged["final_daily_max_f"].to_numpy()
    )
    return merged.sort_values("forecast_origin_datetime").reset_index(drop=True)


def build_live_snapshot(
    current_clean_df: pd.DataFrame,
    live_forecast: dict[str, object],
) -> pd.DataFrame:
    if current_clean_df.empty:
        raise ValueError("No hay observaciones limpias para construir el snapshot live.")
    forecasts = list(live_forecast.get("forecasts", []))
    if not forecasts:
        raise ValueError("No hay filas de forecast live para construir el meta-modelo.")

    latest_df = current_clean_df.copy()
    latest_df["observation_datetime"] = pd.to_datetime(latest_df["observation_datetime"])
    latest_df = latest_df.sort_values("observation_datetime").reset_index(drop=True)
    latest_timestamp = pd.Timestamp(latest_df.iloc[-1]["observation_datetime"])
    state_frame = prepare_state_frame(latest_df)
    live_state = state_frame[state_frame["observation_datetime"] == latest_timestamp]
    if live_state.empty:
        raise ValueError("No se pudo localizar el snapshot live dentro del estado limpio.")

    forecast_frame = pd.DataFrame(forecasts)
    forecast_frame["forecast_origin_datetime"] = latest_timestamp
    forecast_frame["split"] = "live"
    forecast_frame["horizon_step"] = pd.to_numeric(
        forecast_frame["horizon_observations_ahead"],
        errors="coerce",
    ).astype(np.int32)
    forecast_frame["observation_datetime"] = pd.to_datetime(
        forecast_frame["forecast_target_timestamp"]
    )
    forecast_frame = forecast_frame.rename(
        columns={"ensemble_prediction_f": "predicted_temperature_f_ensemble"}
    )
    snapshot = build_forecast_snapshot_features(forecast_frame)
    merged = snapshot.merge(
        live_state,
        left_on="forecast_origin_datetime",
        right_on="observation_datetime",
        how="left",
    ).drop(columns=["observation_datetime"])

    merged["max_gain_vs_current_f"] = (
        merged["ensemble_pred_max_f"] - merged["temperature_f"]
    )
    merged["same_day_max_gain_vs_current_f"] = (
        merged["same_day_pred_max_f"] - merged["temperature_f"]
    )
    merged["upper_max_gain_vs_current_f"] = (
        merged["upper_pred_max_f"] - merged["temperature_f"]
    )
    merged["same_day_upper_gain_vs_current_f"] = (
        merged["same_day_upper_max_f"] - merged["temperature_f"]
    )
    return merged.reset_index(drop=True)


def resolve_feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataset.columns
        if column not in LIVE_EXCLUDED_COLUMNS
    ]


def resolve_support_bounds(train_frame: pd.DataFrame) -> tuple[int, int]:
    min_c = int(train_frame["final_daily_max_c"].min())
    max_c = int(train_frame["final_daily_max_c"].max())
    if min_c >= max_c:
        raise ValueError("El meta-modelo requiere al menos dos bins de temperatura diarios.")
    return min_c, max_c


def fit_single_threshold_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold_c: int,
    train_config: dict[str, object],
) -> dict[str, object]:
    x_train = x_train.copy()
    x_val = x_val.copy()
    train_positive_rate = float(np.mean(y_train))
    result: dict[str, object] = {
        "threshold_c": int(threshold_c),
        "train_positive_rate": train_positive_rate,
    }
    if np.unique(y_train).size < 2:
        result.update(
            {
                "kind": "constant",
                "best_iteration": 0,
                "calibrator": {
                    "kind": "constant",
                    "x_upper_bounds": [1.0],
                    "y_values": [train_positive_rate],
                },
            }
        )
        return result

    lgb = lightgbm_model.require_lightgbm()
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": float(train_config["learning_rate"]),
        "num_leaves": int(train_config["num_leaves"]),
        "max_depth": int(train_config["max_depth"]),
        "min_data_in_leaf": int(train_config["min_data_in_leaf"]),
        "feature_fraction": float(train_config["feature_fraction"]),
        "bagging_fraction": float(train_config["bagging_fraction"]),
        "bagging_freq": int(train_config["bagging_freq"]),
        "lambda_l2": float(train_config["lambda_l2"]),
        "min_gain_to_split": float(train_config["min_gain_to_split"]),
        "verbosity": -1,
        "seed": int(train_config["seed"]) + int(threshold_c),
        "feature_fraction_seed": int(train_config["seed"]) + int(threshold_c),
        "bagging_seed": int(train_config["seed"]) + int(threshold_c),
        "data_random_seed": int(train_config["seed"]) + int(threshold_c),
        "num_threads": int(train_config["num_threads"]),
    }
    train_set = lgb.Dataset(
        x_train,
        label=np.asarray(y_train, dtype=np.float32),
        feature_name=list(x_train.columns),
        free_raw_data=False,
    )
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks: list[Any] = []
    if len(x_val) > 0:
        val_set = lgb.Dataset(
            x_val,
            label=np.asarray(y_val, dtype=np.float32),
            feature_name=list(x_train.columns),
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets.append(val_set)
        valid_names.append("val")
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=int(train_config["early_stopping_rounds"]),
                first_metric_only=True,
                verbose=False,
            )
        )

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(train_config["num_boost_round"]),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    best_iteration = int(booster.best_iteration or train_config["num_boost_round"])
    val_raw_probability = booster.predict(x_val, num_iteration=best_iteration)
    calibrator = fit_isotonic_calibrator(val_raw_probability, y_val)
    result.update(
        {
            "kind": "lightgbm",
            "booster": booster,
            "best_iteration": best_iteration,
            "calibrator": calibrator,
        }
    )
    return result


def predict_threshold_model(
    model_bundle: dict[str, object],
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    if len(feature_frame) == 0:
        return np.empty(0, dtype=np.float64)
    if model_bundle["kind"] == "constant":
        raw_probability = np.full(
            len(feature_frame),
            float(model_bundle["train_positive_rate"]),
            dtype=np.float64,
        )
    else:
        booster = model_bundle["booster"]
        raw_probability = np.asarray(
            booster.predict(
                feature_frame,
                num_iteration=int(model_bundle["best_iteration"]),
            ),
            dtype=np.float64,
        )
    return apply_isotonic_calibrator(raw_probability, model_bundle["calibrator"])


def compute_feature_importance(
    threshold_models: list[dict[str, object]],
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for model_bundle in threshold_models:
        if model_bundle["kind"] != "lightgbm":
            continue
        booster = model_bundle["booster"]
        gains = booster.feature_importance(importance_type="gain")
        splits = booster.feature_importance(importance_type="split")
        for feature_name, gain_value, split_value in zip(
            feature_columns,
            gains,
            splits,
            strict=False,
        ):
            rows.append(
                {
                    "threshold_c": int(model_bundle["threshold_c"]),
                    "feature": str(feature_name),
                    "gain_importance": float(gain_value),
                    "split_importance": float(split_value),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "gain_importance",
                "split_importance",
                "mean_gain_importance",
                "mean_split_importance",
            ]
        )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("feature", as_index=False)
        .agg(
            gain_importance=("gain_importance", "sum"),
            split_importance=("split_importance", "sum"),
            mean_gain_importance=("gain_importance", "mean"),
            mean_split_importance=("split_importance", "mean"),
        )
        .sort_values(
            ["mean_gain_importance", "gain_importance", "feature"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def fit_market_meta_model(
    ensemble_predictions: pd.DataFrame,
    clean_df: pd.DataFrame,
    live_forecast: dict[str, object],
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    train_config = {**DEFAULT_TRAIN_CONFIG, **(config or {})}
    dataset = build_training_dataset(ensemble_predictions, clean_df)
    feature_columns = resolve_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("No se pudieron resolver features para el meta-modelo.")

    split_frames = {
        split_name: dataset[dataset["split"] == split_name].reset_index(drop=True)
        for split_name in ("train", "val", "test")
    }
    train_frame = split_frames["train"]
    val_frame = split_frames["val"]
    test_frame = split_frames["test"]
    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError(
            "El meta-modelo probabilistico requiere splits train/val/test con datos."
        )

    support_min_c, support_max_c = resolve_support_bounds(train_frame)
    thresholds_c = list(range(support_min_c + 1, support_max_c + 1))

    x_by_split = {
        split_name: frame[feature_columns]
        for split_name, frame in split_frames.items()
    }
    y_threshold_by_split = {
        split_name: threshold_matrix_from_targets(frame["final_daily_max_c"], thresholds_c)
        for split_name, frame in split_frames.items()
    }

    threshold_models: list[dict[str, object]] = []
    threshold_metric_rows: list[dict[str, object]] = []

    for idx, threshold_c in enumerate(thresholds_c):
        model_bundle = fit_single_threshold_model(
            x_train=x_by_split["train"],
            y_train=y_threshold_by_split["train"][:, idx],
            x_val=x_by_split["val"],
            y_val=y_threshold_by_split["val"][:, idx],
            threshold_c=threshold_c,
            train_config=train_config,
        )
        threshold_models.append(model_bundle)

    probability_by_split: dict[str, np.ndarray] = {}
    for split_name, feature_frame in x_by_split.items():
        split_probabilities = np.column_stack(
            [
                predict_threshold_model(model_bundle, feature_frame)
                for model_bundle in threshold_models
            ]
        )
        split_probabilities = enforce_monotonic_cumulative(split_probabilities)
        probability_by_split[split_name] = split_probabilities

        y_matrix = y_threshold_by_split[split_name]
        for idx, threshold_c in enumerate(thresholds_c):
            metrics = binary_probability_metrics(y_matrix[:, idx], split_probabilities[:, idx])
            threshold_metric_rows.append(
                {
                    "split": split_name,
                    "threshold_c": int(threshold_c),
                    **metrics,
                }
            )

    threshold_metrics_df = pd.DataFrame(threshold_metric_rows)
    overall_threshold_metrics = (
        threshold_metrics_df.groupby("split", as_index=False)
        .agg(
            mean_positive_rate=("positive_rate", "mean"),
            mean_brier=("brier", "mean"),
            mean_logloss=("logloss", "mean"),
        )
        .set_index("split")
        .to_dict(orient="index")
    )

    _, bucket_distribution_val = bucket_distribution_from_cumulative(
        probability_by_split["val"],
        support_min_c=support_min_c,
        support_max_c=support_max_c,
    )
    bucket_labels, bucket_distribution_test = bucket_distribution_from_cumulative(
        probability_by_split["test"],
        support_min_c=support_min_c,
        support_max_c=support_max_c,
    )

    market_bin_metrics = {
        "val": multiclass_probability_metrics(
            true_bucket_index(val_frame["final_daily_max_c"], support_min_c, support_max_c),
            bucket_distribution_val,
        ),
        "test": multiclass_probability_metrics(
            true_bucket_index(test_frame["final_daily_max_c"], support_min_c, support_max_c),
            bucket_distribution_test,
        ),
    }

    live_frame = build_live_snapshot(clean_df, live_forecast)
    live_features = live_frame[feature_columns]
    live_cumulative = np.column_stack(
        [predict_threshold_model(model_bundle, live_features) for model_bundle in threshold_models]
    )
    live_cumulative = enforce_monotonic_cumulative(live_cumulative)
    live_bucket_labels, live_bucket_distribution = bucket_distribution_from_cumulative(
        live_cumulative,
        support_min_c=support_min_c,
        support_max_c=support_max_c,
    )

    live_probability_rows = [
        {
            "threshold_c": int(threshold_c),
            "probability_daily_max_at_or_above": float(live_cumulative[0, idx]),
        }
        for idx, threshold_c in enumerate(thresholds_c)
    ]
    live_market_rows = [
        {
            "market_bin": str(label),
            "probability": float(live_bucket_distribution[0, idx]),
        }
        for idx, label in enumerate(live_bucket_labels)
    ]
    top_market_rows = sorted(
        live_market_rows,
        key=lambda row: (-row["probability"], row["market_bin"]),
    )[:8]
    top_market_row = top_market_rows[0]

    live_distribution_payload = {
        "forecast_origin_datetime": str(live_frame.iloc[0]["forecast_origin_datetime"]),
        "rounding_rule_for_market_bins": "round((daily_max_f - 32) * 5 / 9)",
        "support": {
            "lower_tail_c": int(support_min_c),
            "upper_tail_c": int(support_max_c),
            "thresholds_c": [int(value) for value in thresholds_c],
        },
        "current_state_f": {
            "temperature_f": float(live_frame.iloc[0]["temperature_f"]),
            "max_temperature_so_far_today_f": float(
                live_frame.iloc[0]["max_temperature_so_far_today_f"]
            ),
            "min_temperature_so_far_today_f": float(
                live_frame.iloc[0]["min_temperature_so_far_today_f"]
            ),
        },
        "forecast_features_f": {
            "ensemble_peak_forecast_f": float(live_frame.iloc[0]["ensemble_pred_max_f"]),
            "same_day_peak_forecast_f": (
                None
                if pd.isna(live_frame.iloc[0]["same_day_pred_max_f"])
                else float(live_frame.iloc[0]["same_day_pred_max_f"])
            ),
            "upper_peak_forecast_f": float(live_frame.iloc[0]["upper_pred_max_f"]),
            "same_day_upper_peak_forecast_f": (
                None
                if pd.isna(live_frame.iloc[0]["same_day_upper_max_f"])
                else float(live_frame.iloc[0]["same_day_upper_max_f"])
            ),
            "same_day_forecast_count": int(live_frame.iloc[0]["same_day_forecast_count"]),
            "next_day_forecast_count": int(live_frame.iloc[0]["next_day_forecast_count"]),
        },
        "probabilities_at_or_above_c": live_probability_rows,
        "market_bin_probabilities": live_market_rows,
        "top_market_bins": top_market_rows,
        "most_likely_market_bin": top_market_row,
    }

    feature_importance_df = compute_feature_importance(threshold_models, feature_columns)
    top_features = feature_importance_df.head(12).to_dict(orient="records")

    summary = {
        "model_type": "threshold_lightgbm_with_isotonic_calibration",
        "target_definition": "final_daily_max_same_day_bucketed_to_integer_celsius",
        "rounding_rule_for_market_bins": "round((daily_max_f - 32) * 5 / 9)",
        "support": {
            "lower_tail_c": int(support_min_c),
            "upper_tail_c": int(support_max_c),
            "threshold_count": int(len(thresholds_c)),
            "thresholds_c": [int(value) for value in thresholds_c],
        },
        "dataset": {
            "row_count": int(len(dataset)),
            "feature_count": int(len(feature_columns)),
            "split_counts": {
                split_name: int(len(frame)) for split_name, frame in split_frames.items()
            },
            "split_periods": {
                split_name: {
                    "start": str(frame["forecast_origin_datetime"].min()),
                    "end": str(frame["forecast_origin_datetime"].max()),
                }
                for split_name, frame in split_frames.items()
            },
        },
        "metrics": {
            "cumulative_thresholds": overall_threshold_metrics,
            "market_bins": market_bin_metrics,
        },
        "live_summary": {
            "forecast_origin_datetime": str(live_frame.iloc[0]["forecast_origin_datetime"]),
            "most_likely_market_bin": top_market_row,
            "top_market_bins": top_market_rows,
        },
        "top_features": top_features,
    }

    return {
        "summary": summary,
        "live_forecast": live_distribution_payload,
        "threshold_metrics": threshold_metrics_df,
        "feature_importance": feature_importance_df,
    }
