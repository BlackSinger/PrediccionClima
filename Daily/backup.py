from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("/content")

DIRECTION_LABELS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]

CONDITION_COLUMNS = [
    "cond_fair",
    "cond_cloudy",
    "cond_partly_cloudy",
    "cond_mostly_cloudy",
    "cond_rain",
    "cond_drizzle",
    "cond_shower",
    "cond_thunder",
    "cond_fog",
    "cond_mist",
    "cond_haze",
    "cond_smoke",
    "cond_dust",
    "cond_windy",
    "cond_heavy",
    "cond_light",
    "cond_vicinity",
]

MODEL_FEATURES = [
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_speed_mph",
    "pressure_in",
    "wind_dir_missing",
    "wind_dir_sin",
    "wind_dir_cos",
    *CONDITION_COLUMNS,
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
    "time_idx",
]

TARGET_COLUMN = "temperature_f"
LAG_SOURCE_FEATURES = [
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_speed_mph",
    "pressure_in",
]
LAG_STEP_CANDIDATES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 72, 96]
ROLLING_WINDOW_CANDIDATES = [3, 6, 12, 24, 48, 96]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo LightGBM con observaciones intradiarias para "
            "predecir temperature_f y opcionalmente realiza tuning de "
            "hiperparametros con validacion walk-forward."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / "wunderground_ezeiza_daily_2014_2026.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "artifacts/lightgbm_daily_temperature",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=96,
        help="Cantidad de observaciones previas usadas para construir features.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Cantidad de observaciones hacia adelante a predecir.",
    )
    parser.add_argument("--num-boost-round", type=int, default=1500)
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Rounds de early stopping sobre validacion.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=40)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Cada cuantos rounds imprimir metricas de entrenamiento.",
    )
    parser.add_argument(
        "--interval-coverage",
        type=float,
        default=0.80,
        help=(
            "Cobertura objetivo del intervalo de prediccion basado en cuantiles "
            "de residuales del split de validacion."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=120,
        help="Gap maximo permitido entre observaciones consecutivas para una muestra valida.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Cantidad de hilos para LightGBM. Usa 1 por defecto para mayor estabilidad.",
    )
    parser.add_argument("--seed", type=int, default=24217956)

    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help=(
            "Activa una busqueda aleatoria de hiperparametros con validacion "
            "walk-forward sobre el split de entrenamiento."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Cantidad total de trials del random search (incluye baseline).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cantidad de folds walk-forward usados durante el tuning.",
    )
    parser.add_argument(
        "--cv-gap",
        type=int,
        default=8,
        help=(
            "Gap purgado entre entrenamiento y validacion dentro de cada fold, "
            "expresado en cantidad de muestras supervisadas."
        ),
    )
    parser.add_argument(
        "--cv-min-train-fraction",
        type=float,
        default=0.50,
        help="Fraccion minima del split train reservada para el primer fold de entrenamiento.",
    )
    parser.add_argument(
        "--tuning-num-boost-round",
        type=int,
        default=700,
        help="Maximo de boosting rounds por fold durante el tuning.",
    )
    parser.add_argument(
        "--tuning-patience",
        type=int,
        default=50,
        help="Patience de early stopping usada en cada fold de tuning.",
    )
    parser.add_argument(
        "--tuning-log-every",
        type=int,
        default=0,
        help="Frecuencia de logs durante el tuning. 0 desactiva logs por fold.",
    )
    parser.add_argument(
        "--search-metric",
        choices=["mae", "rmse"],
        default="mae",
        help="Metrica primaria usada para elegir el mejor trial del tuning.",
    )

    if argv is None and "ipykernel" in sys.modules:
        argv = []
    return parser.parse_args(argv)


def require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "No se encontro el paquete 'lightgbm'. Instala las dependencias o "
            "ejecuta 'pip install lightgbm'."
        ) from exc
    return lgb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def add_condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    cond = df["condition"].fillna("").str.lower()

    df["cond_fair"] = cond.str.contains(r"\bfair\b").astype(int)
    df["cond_cloudy"] = cond.str.contains(r"cloudy").astype(int)
    df["cond_partly_cloudy"] = cond.str.contains(r"partly cloudy").astype(int)
    df["cond_mostly_cloudy"] = cond.str.contains(r"mostly cloudy").astype(int)
    df["cond_rain"] = cond.str.contains(r"\brain\b").astype(int)
    df["cond_drizzle"] = cond.str.contains(r"drizzle").astype(int)
    df["cond_shower"] = cond.str.contains(r"shower|showers").astype(int)
    df["cond_thunder"] = cond.str.contains(r"thunder|t-storm").astype(int)
    df["cond_fog"] = cond.str.contains(r"\bfog\b").astype(int)
    df["cond_mist"] = cond.str.contains(r"\bmist\b").astype(int)
    df["cond_haze"] = cond.str.contains(r"\bhaze\b").astype(int)
    df["cond_smoke"] = cond.str.contains(r"\bsmoke\b").astype(int)
    df["cond_dust"] = cond.str.contains(r"dust").astype(int)
    df["cond_windy"] = cond.str.contains(r"windy").astype(int)
    df["cond_heavy"] = cond.str.contains(r"heavy").astype(int)
    df["cond_light"] = cond.str.contains(r"light").astype(int)
    df["cond_vicinity"] = cond.str.contains(r"vicinity").astype(int)
    return df


def apply_range_interpolation(
    df: pd.DataFrame,
    column: str,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    series = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(False, index=df.index)
    if lower is not None:
        mask |= series < lower
    if upper is not None:
        mask |= series > upper
    df[column] = series.mask(mask).interpolate(limit_direction="both")


def clean_eda_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["observation_datetime"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["observation_time"].str.strip(),
        format="%Y-%m-%d %I:%M %p",
        errors="coerce",
    )

    df = df.dropna(subset=["observation_datetime"]).sort_values(
        ["observation_datetime", "observation_index"],
        kind="mergesort",
    )
    df = df.reset_index(drop=True)

    numeric_cols = [
        "temperature_f",
        "dew_point_f",
        "humidity_pct",
        "wind_speed_mph",
        "wind_gust_mph",
        "pressure_in",
        "precip_in",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["time_idx"] = np.arange(len(df), dtype=np.float32)
    df["hour"] = df["observation_datetime"].dt.hour.astype(np.float32)
    df["minute"] = df["observation_datetime"].dt.minute.astype(np.float32)

    direction_to_degrees = {
        direction: idx * 22.5 for idx, direction in enumerate(DIRECTION_LABELS)
    }
    df["wind_dir_missing"] = df["wind_direction"].isna().astype(np.float32)
    degrees = df["wind_direction"].map(direction_to_degrees)
    radians = np.deg2rad(degrees)
    df["wind_dir_sin"] = np.sin(radians).fillna(0.0).astype(np.float32)
    df["wind_dir_cos"] = np.cos(radians).fillna(0.0).astype(np.float32)

    df = add_condition_flags(df)

    apply_range_interpolation(df, "temperature_f", lower=30.0, upper=110.0)
    apply_range_interpolation(df, "dew_point_f", lower=10.0, upper=83.0)
    apply_range_interpolation(df, "humidity_pct", lower=2.0, upper=100.0)
    apply_range_interpolation(df, "wind_speed_mph", lower=0.0, upper=60.0)
    apply_range_interpolation(df, "pressure_in", lower=26.5, upper=31.5)

    interpolate_cols = [
        "temperature_f",
        "dew_point_f",
        "humidity_pct",
        "wind_speed_mph",
        "pressure_in",
    ]
    df[interpolate_cols] = df[interpolate_cols].interpolate(limit_direction="both")

    hour_fraction = df["hour"] + df["minute"] / 60.0
    day_of_year = df["observation_datetime"].dt.dayofyear.astype(np.float32)
    month = df["observation_datetime"].dt.month.astype(np.float32)

    df["hour_sin"] = np.sin(2 * np.pi * hour_fraction / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour_fraction / 24.0).astype(np.float32)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0).astype(np.float32)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)

    drop_cols = [
        "wind_gust_mph",
        "precip_in",
        "date",
        "year",
        "month",
        "day",
        "observation_index",
        "observation_time",
        "condition",
        "wind_direction",
    ]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    first_cols = ["time_idx", "hour", "minute", "observation_datetime"]
    ordered_cols = first_cols + [col for col in df.columns if col not in first_cols]
    df = df[ordered_cols]
    return df


def compute_run_lengths(
    timestamps: pd.Series,
    max_gap_minutes: int,
) -> np.ndarray:
    run_lengths = np.ones(len(timestamps), dtype=np.int32)
    for idx in range(1, len(timestamps)):
        delta_minutes = (
            timestamps.iloc[idx] - timestamps.iloc[idx - 1]
        ).total_seconds() / 60.0
        if 0 < delta_minutes <= max_gap_minutes:
            run_lengths[idx] = run_lengths[idx - 1] + 1
    return run_lengths


def resolve_lag_steps(lookback: int) -> list[int]:
    steps = [step for step in LAG_STEP_CANDIDATES if step <= lookback]
    if lookback not in steps:
        steps.append(lookback)
    return sorted(set(steps))


def resolve_rolling_windows(lookback: int) -> list[int]:
    windows = [window for window in ROLLING_WINDOW_CANDIDATES if window <= lookback]
    if lookback not in windows:
        windows.append(lookback)
    return sorted(set(windows))


def build_engineered_feature_names(
    base_feature_cols: list[str],
    lag_source_features: list[str],
    lag_steps: list[int],
    rolling_windows: list[int],
) -> list[str]:
    names = [f"{col}_t0" for col in base_feature_cols]
    for feature_name in lag_source_features:
        for lag_step in lag_steps:
            names.append(f"{feature_name}_lag_{lag_step}")
    for feature_name in lag_source_features:
        for window_size in rolling_windows:
            prefix = f"{feature_name}_window_{window_size}"
            names.extend(
                [
                    f"{prefix}_mean",
                    f"{prefix}_std",
                    f"{prefix}_min",
                    f"{prefix}_max",
                    f"{prefix}_delta_from_mean",
                ]
            )
    return names


def build_feature_row(
    window: np.ndarray,
    feature_cols: list[str],
    feature_index: dict[str, int],
    lag_source_features: list[str],
    lag_steps: list[int],
    rolling_windows: list[int],
) -> list[float]:
    latest = window[-1]
    values = [float(value) for value in latest]

    for feature_name in lag_source_features:
        series = window[:, feature_index[feature_name]]
        for lag_step in lag_steps:
            values.append(float(series[-lag_step]))

    for feature_name in lag_source_features:
        series = window[:, feature_index[feature_name]]
        latest_value = float(series[-1])
        for window_size in rolling_windows:
            window_values = series[-window_size:]
            mean_value = float(np.mean(window_values))
            values.extend(
                [
                    mean_value,
                    float(np.std(window_values)),
                    float(np.min(window_values)),
                    float(np.max(window_values)),
                    latest_value - mean_value,
                ]
            )

    expected_size = len(feature_cols) + len(lag_source_features) * len(lag_steps)
    expected_size += len(lag_source_features) * len(rolling_windows) * 5
    if len(values) != expected_size:
        raise ValueError(
            f"Se esperaban {expected_size} features ingenierizadas pero se generaron {len(values)}."
        )
    return values


def build_supervised_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    max_gap_minutes: int,
) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
    n_rows = len(df)
    train_end = max(int(n_rows * train_ratio), lookback + horizon)
    val_end = int(n_rows * (train_ratio + val_ratio))
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n_rows - 1)

    feature_values = df[feature_cols].to_numpy(dtype=np.float32)
    target_values = df[target_col].to_numpy(dtype=np.float32)
    run_lengths = compute_run_lengths(df["observation_datetime"], max_gap_minutes)

    lag_steps = resolve_lag_steps(lookback)
    rolling_windows = resolve_rolling_windows(lookback)
    feature_names = build_engineered_feature_names(
        base_feature_cols=feature_cols,
        lag_source_features=LAG_SOURCE_FEATURES,
        lag_steps=lag_steps,
        rolling_windows=rolling_windows,
    )
    feature_index = {name: idx for idx, name in enumerate(feature_cols)}

    buckets: dict[str, dict[str, list[object]]] = {
        "train": {"x": [], "y": [], "date": []},
        "val": {"x": [], "y": [], "date": []},
        "test": {"x": [], "y": [], "date": []},
    }

    required_run = lookback + horizon
    for target_idx in range(required_run - 1, n_rows):
        if run_lengths[target_idx] < required_run:
            continue

        input_end = target_idx - horizon + 1
        input_start = input_end - lookback
        if input_start < 0:
            continue

        if target_idx < train_end:
            split = "train"
        elif target_idx < val_end:
            split = "val"
        else:
            split = "test"

        window = feature_values[input_start:input_end]
        row_values = build_feature_row(
            window=window,
            feature_cols=feature_cols,
            feature_index=feature_index,
            lag_source_features=LAG_SOURCE_FEATURES,
            lag_steps=lag_steps,
            rolling_windows=rolling_windows,
        )
        buckets[split]["x"].append(row_values)
        buckets[split]["y"].append(float(target_values[target_idx]))
        buckets[split]["date"].append(df.iloc[target_idx]["observation_datetime"])

    split_frames: dict[str, dict[str, object]] = {}
    for split_name, split_values in buckets.items():
        if not split_values["x"]:
            raise ValueError(f"No hay muestras disponibles para el split '{split_name}'.")

        x_frame = pd.DataFrame(split_values["x"], columns=feature_names, dtype=np.float32)
        split_frames[split_name] = {
            "x": x_frame,
            "y": np.asarray(split_values["y"], dtype=np.float32),
            "date": np.asarray(split_values["date"], dtype="datetime64[ns]"),
        }

    split_info = {
        "train_end_timestamp": df.iloc[train_end - 1]["observation_datetime"].strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "val_end_timestamp": df.iloc[val_end - 1]["observation_datetime"].strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "total_rows": n_rows,
        "max_gap_minutes": max_gap_minutes,
    }
    feature_engineering = {
        "lag_source_features": LAG_SOURCE_FEATURES,
        "lag_steps": lag_steps,
        "rolling_windows": rolling_windows,
    }
    return split_frames, split_info, feature_engineering


def build_history_frame(evals_result: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    if not evals_result:
        return pd.DataFrame(columns=["iteration"])

    first_dataset = next(iter(evals_result.values()))
    first_metric = next(iter(first_dataset.values()))
    rows: list[dict[str, float | int]] = []
    for iteration in range(len(first_metric)):
        row: dict[str, float | int] = {"iteration": iteration + 1}
        for dataset_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                row[f"{dataset_name}_{metric_name}"] = float(values[iteration])
        rows.append(row)
    return pd.DataFrame(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def build_lgb_params(
    args: argparse.Namespace,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "regression",
        "metric": ["l1", "l2"],
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "min_gain_to_split": args.min_gain_to_split,
        "max_bin": args.max_bin,
        "seed": args.seed,
        "verbosity": -1,
        "feature_pre_filter": False,
        "force_col_wise": True,
    }
    params["num_threads"] = max(int(args.num_threads), 1)
    if overrides:
        params.update(overrides)
    return params


def train_model(
    lgb: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    args: argparse.Namespace,
    params_override: dict[str, object] | None = None,
    num_boost_round: int | None = None,
    patience: int | None = None,
    log_every: int | None = None,
    record_history: bool = True,
    verbose_early_stopping: bool = True,
) -> tuple[Any, pd.DataFrame, dict[str, object]]:
    params = build_lgb_params(args=args, overrides=params_override)

    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    val_set = lgb.Dataset(val_x, label=val_y, reference=train_set, free_raw_data=False)

    evals_result: dict[str, dict[str, list[float]]] = {}
    callbacks: list[Any] = [
        lgb.early_stopping(
            stopping_rounds=patience or args.patience,
            first_metric_only=True,
            verbose=verbose_early_stopping,
        )
    ]
    if record_history:
        callbacks.append(lgb.record_evaluation(evals_result))
    effective_log_every = args.log_every if log_every is None else log_every
    if effective_log_every > 0:
        callbacks.append(lgb.log_evaluation(period=effective_log_every))

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round or args.num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    history = build_history_frame(evals_result) if record_history else pd.DataFrame(columns=["iteration"])
    return booster, history, params


def predict(
    booster: Any,
    features: pd.DataFrame,
    num_iteration: int,
) -> np.ndarray:
    return booster.predict(features, num_iteration=num_iteration)


def calibrate_prediction_interval(
    residuals: np.ndarray,
    coverage: float,
    reference_split: str,
) -> dict[str, float | str]:
    if not 0.0 < coverage < 1.0:
        raise ValueError("--interval-coverage debe estar entre 0 y 1.")

    alpha = 1.0 - coverage
    lower_quantile = float(np.quantile(residuals, alpha / 2.0))
    upper_quantile = float(np.quantile(residuals, 1.0 - alpha / 2.0))
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))

    return {
        "method": "validation_residual_quantiles",
        "reference_split": reference_split,
        "coverage_target": float(coverage),
        "alpha": float(alpha),
        "lower_residual_quantile_f": lower_quantile,
        "upper_residual_quantile_f": upper_quantile,
        "residual_median_f": residual_median,
        "residual_mad_f": residual_mad,
    }


def build_prediction_interval(
    y_pred: np.ndarray,
    prediction_band: dict[str, float | str],
) -> tuple[np.ndarray, np.ndarray]:
    lower_shift = float(prediction_band["lower_residual_quantile_f"])
    upper_shift = float(prediction_band["upper_residual_quantile_f"])
    return y_pred + lower_shift, y_pred + upper_shift


def build_predictions_frame(
    split_name: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_interval: np.ndarray,
    upper_interval: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_datetime": pd.to_datetime(dates),
            "split": split_name,
            "actual_temperature_f": y_true,
            "predicted_temperature_f": y_pred,
            "lower_prediction_interval_f": lower_interval,
            "upper_prediction_interval_f": upper_interval,
        }
    )


def select_history_columns(history: pd.DataFrame) -> tuple[str | None, str | None, str]:
    if "train_l1" in history.columns and "val_l1" in history.columns:
        return "train_l1", "val_l1", "l1"
    if "train_l2" in history.columns and "val_l2" in history.columns:
        return "train_l2", "val_l2", "l2"

    metric_columns = [column for column in history.columns if column != "iteration"]
    if len(metric_columns) >= 2:
        return metric_columns[0], metric_columns[1], "metric"
    return None, None, "metric"


def save_plots(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_band: dict[str, float | str],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    train_metric_col, val_metric_col, metric_label = select_history_columns(history)
    if train_metric_col and val_metric_col and not history.empty:
        axes[0].plot(history["iteration"], history[train_metric_col], label="Train")
        axes[0].plot(history["iteration"], history[val_metric_col], label="Validation")
        axes[0].set_title("Evolucion de la metrica de entrenamiento")
        axes[0].set_xlabel("Boosting round")
        axes[0].set_ylabel(metric_label)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Sin historial de metricas disponible", ha="center", va="center")
        axes[0].set_axis_off()

    test_predictions = predictions[predictions["split"] == "test"].sort_values("observation_datetime")
    axes[1].plot(
        test_predictions["observation_datetime"],
        test_predictions["actual_temperature_f"],
        label="Real",
        linewidth=1.1,
    )
    axes[1].plot(
        test_predictions["observation_datetime"],
        test_predictions["predicted_temperature_f"],
        label="Prediccion",
        linewidth=1.1,
    )
    coverage_target = float(prediction_band["coverage_target"])
    axes[1].fill_between(
        test_predictions["observation_datetime"],
        test_predictions["lower_prediction_interval_f"],
        test_predictions["upper_prediction_interval_f"],
        label=f"Intervalo {coverage_target:.0%} por cuantiles",
        alpha=0.2,
    )
    axes[1].set_title("Prediccion de temperature_f en test")
    axes[1].set_xlabel("Fecha y hora")
    axes[1].set_ylabel("Temperatura (F)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_diagnostics.png", dpi=160)
    plt.close(fig)


def save_feature_importance(booster: Any, output_dir: Path) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain_importance": booster.feature_importance(importance_type="gain"),
            "split_importance": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain_importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    top_importance = importance.head(25).sort_values("gain_importance", ascending=True)
    if not top_importance.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_importance["feature"], top_importance["gain_importance"])
        ax.set_title("Top features por ganancia")
        ax.set_xlabel("Gain importance")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_importance.png", dpi=160)
        plt.close(fig)

    return importance


def to_builtin(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def normalize_param_dict(params: dict[str, object]) -> dict[str, object]:
    return {key: to_builtin(value) for key, value in params.items()}


def loguniform_sample(rng: random.Random, low: float, high: float) -> float:
    return float(10 ** rng.uniform(math.log10(low), math.log10(high)))


def sample_hyperparameter_overrides(rng: random.Random) -> dict[str, object]:
    max_depth = rng.choice([-1, 6, 8, 10, 12, 16])
    num_leaves = rng.choice([15, 31, 63, 95, 127, 191, 255])
    if max_depth > 0:
        num_leaves = min(num_leaves, 2 ** max_depth)
        num_leaves = max(num_leaves, 4)

    bagging_fraction = round(rng.uniform(0.60, 1.00), 3)
    feature_fraction = round(rng.uniform(0.60, 1.00), 3)

    lambda_l1 = 0.0 if rng.random() < 0.35 else loguniform_sample(rng, 1e-4, 10.0)
    lambda_l2 = loguniform_sample(rng, 1e-3, 20.0)
    min_gain_to_split = 0.0 if rng.random() < 0.50 else loguniform_sample(rng, 1e-4, 0.3)

    params: dict[str, object] = {
        "learning_rate": loguniform_sample(rng, 0.01, 0.08),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "min_data_in_leaf": int(rng.choice([10, 20, 30, 40, 60, 80, 120, 160, 240])),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": int(rng.choice([0, 1, 2, 3, 5, 7])),
        "lambda_l1": float(lambda_l1),
        "lambda_l2": float(lambda_l2),
        "min_gain_to_split": float(min_gain_to_split),
        "max_bin": int(rng.choice([127, 255, 511])),
    }
    return normalize_param_dict(params)


def build_walk_forward_splits(
    n_samples: int,
    n_splits: int,
    gap: int,
    min_train_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("cv_folds debe ser al menos 2 para hacer tuning robusto.")
    if not 0.05 <= min_train_fraction < 0.95:
        raise ValueError("cv_min_train_fraction debe estar entre 0.05 y 0.95.")

    min_train_size = max(int(n_samples * min_train_fraction), 200)
    available = n_samples - min_train_size - gap
    if available < n_splits:
        raise ValueError(
            "No hay suficientes muestras para armar folds walk-forward. "
            "Reduce cv_folds, reduce cv_gap o baja cv_min_train_fraction."
        )

    val_size = max(available // n_splits, 1)
    train_end = min_train_size
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_idx in range(n_splits):
        val_start = train_end + gap
        val_end = val_start + val_size
        if fold_idx == n_splits - 1:
            val_end = n_samples
        if val_start >= n_samples or val_start >= val_end:
            break

        train_idx = np.arange(0, train_end, dtype=np.int32)
        val_idx = np.arange(val_start, val_end, dtype=np.int32)
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append((train_idx, val_idx))
        train_end = val_end

    if len(folds) < 2:
        raise ValueError(
            "No se pudieron construir suficientes folds de validacion temporal."
        )
    return folds


def evaluate_candidate_params(
    lgb: Any,
    x: pd.DataFrame,
    y: np.ndarray,
    args: argparse.Namespace,
    param_overrides: dict[str, object],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    fold_rows: list[dict[str, object]] = []
    maes: list[float] = []
    rmses: list[float] = []
    mapes: list[float] = []
    r2s: list[float] = []
    best_iterations: list[int] = []

    for fold_number, (train_idx, val_idx) in enumerate(folds, start=1):
        train_x = x.iloc[train_idx]
        val_x = x.iloc[val_idx]
        train_y = y[train_idx]
        val_y = y[val_idx]

        booster, _, _ = train_model(
            lgb=lgb,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            args=args,
            params_override=param_overrides,
            num_boost_round=min(args.tuning_num_boost_round, args.num_boost_round),
            patience=min(args.tuning_patience, args.patience),
            log_every=args.tuning_log_every,
            record_history=False,
            verbose_early_stopping=False,
        )
        best_iteration = int(booster.best_iteration or booster.current_iteration())
        y_pred = predict(booster=booster, features=val_x, num_iteration=best_iteration)
        metrics = regression_metrics(val_y, y_pred)

        maes.append(metrics["mae"])
        rmses.append(metrics["rmse"])
        mapes.append(metrics["mape"])
        r2s.append(metrics["r2"])
        best_iterations.append(best_iteration)

        fold_rows.append(
            {
                "fold": fold_number,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
                "r2": float(metrics["r2"]),
                "best_iteration": best_iteration,
            }
        )

    summary: dict[str, object] = {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_rmse_std": float(np.std(rmses)),
        "cv_mape_mean": float(np.mean(mapes)),
        "cv_r2_mean": float(np.mean(r2s)),
        "cv_best_iteration_mean": float(np.mean(best_iterations)),
        "cv_best_iteration_median": int(np.median(best_iterations)),
        "cv_best_iteration_max": int(np.max(best_iterations)),
    }
    return summary, fold_rows


def flatten_search_result_row(
    trial_index: int,
    elapsed_seconds: float,
    param_overrides: dict[str, object],
    summary: dict[str, object],
    is_baseline: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trial": int(trial_index),
        "is_baseline": bool(is_baseline),
        "elapsed_seconds": float(elapsed_seconds),
    }
    row.update(normalize_param_dict(param_overrides))
    row.update(normalize_param_dict(summary))
    return row


def save_hyperparameter_search_plot(results: pd.DataFrame, output_dir: Path, metric_name: str) -> None:
    if results.empty or metric_name not in results.columns:
        return

    ordered = results.sort_values("trial").reset_index(drop=True)
    best_so_far = ordered[metric_name].cummin()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ordered["trial"], ordered[metric_name], marker="o", linewidth=1.0, label="Trial")
    ax.plot(ordered["trial"], best_so_far, linewidth=1.5, label="Mejor hasta ahora")
    ax.set_title(f"Busqueda de hiperparametros por {metric_name}")
    ax.set_xlabel("Trial")
    ax.set_ylabel(metric_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hyperparameter_search.png", dpi=160)
    plt.close(fig)


def run_hyperparameter_search(
    lgb: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = random.Random(args.seed)
    folds = build_walk_forward_splits(
        n_samples=len(train_x),
        n_splits=args.cv_folds,
        gap=args.cv_gap,
        min_train_fraction=args.cv_min_train_fraction,
    )

    baseline = normalize_param_dict(
        {
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "lambda_l1": args.lambda_l1,
            "lambda_l2": args.lambda_l2,
            "min_gain_to_split": args.min_gain_to_split,
            "max_bin": args.max_bin,
        }
    )

    candidate_params: list[dict[str, object]] = [baseline]
    seen = {json.dumps(baseline, sort_keys=True)}
    target_trials = max(args.n_trials, 1)
    while len(candidate_params) < target_trials:
        candidate = sample_hyperparameter_overrides(rng)
        signature = json.dumps(candidate, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        candidate_params.append(candidate)

    summary_rows: list[dict[str, object]] = []
    fold_rows_all: list[dict[str, object]] = []
    metric_column = f"cv_{args.search_metric}_mean"
    best_score = math.inf
    best_params = baseline
    best_summary: dict[str, object] | None = None
    started_at = time.time()

    print("\n[Hyperparameter Search] Iniciando random search con validacion walk-forward")
    print(
        json.dumps(
            {
                "n_trials": len(candidate_params),
                "cv_folds": args.cv_folds,
                "cv_gap": args.cv_gap,
                "cv_min_train_fraction": args.cv_min_train_fraction,
                "metric": args.search_metric,
            },
            indent=2,
        )
    )

    for trial_index, param_overrides in enumerate(candidate_params, start=1):
        trial_started = time.time()
        summary, fold_rows = evaluate_candidate_params(
            lgb=lgb,
            x=train_x,
            y=train_y,
            args=args,
            param_overrides=param_overrides,
            folds=folds,
        )
        elapsed_seconds = time.time() - trial_started
        summary_row = flatten_search_result_row(
            trial_index=trial_index,
            elapsed_seconds=elapsed_seconds,
            param_overrides=param_overrides,
            summary=summary,
            is_baseline=(trial_index == 1),
        )
        summary_rows.append(summary_row)

        for fold_row in fold_rows:
            fold_row_with_trial = dict(fold_row)
            fold_row_with_trial["trial"] = int(trial_index)
            fold_row_with_trial.update(param_overrides)
            fold_rows_all.append(fold_row_with_trial)

        current_score = float(summary_row[metric_column])
        is_best = current_score < best_score
        if is_best:
            best_score = current_score
            best_params = param_overrides
            best_summary = summary_row

        print(
            f"[Hyperparameter Search] Trial {trial_index}/{len(candidate_params)} | "
            f"{metric_column}={current_score:.5f} | "
            f"best_so_far={best_score:.5f} | params={json.dumps(param_overrides, sort_keys=True)}"
        )

    results = pd.DataFrame(summary_rows).sort_values(metric_column, ascending=True)
    fold_results = pd.DataFrame(fold_rows_all)
    results.to_csv(output_dir / "hyperparameter_search_results.csv", index=False)
    fold_results.to_csv(output_dir / "hyperparameter_search_fold_results.csv", index=False)
    save_hyperparameter_search_plot(results, output_dir, metric_column)

    search_summary = {
        "enabled": True,
        "search_type": "random_search_walk_forward_cv",
        "search_metric": args.search_metric,
        "n_trials": int(len(candidate_params)),
        "cv_folds": int(args.cv_folds),
        "cv_gap": int(args.cv_gap),
        "cv_min_train_fraction": float(args.cv_min_train_fraction),
        "tuning_num_boost_round": int(min(args.tuning_num_boost_round, args.num_boost_round)),
        "tuning_patience": int(min(args.tuning_patience, args.patience)),
        "total_elapsed_seconds": float(time.time() - started_at),
        "best_params": normalize_param_dict(best_params),
        "best_result": normalize_param_dict(best_summary or {}),
    }
    with (output_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as fp:
        json.dump(search_summary, fp, indent=2)

    return best_params, results, fold_results, search_summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    lgb = require_lightgbm()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data
    if not data_path.is_absolute():
        data_path = SCRIPT_DIR / data_path

    raw_df = pd.read_csv(data_path)
    clean_df = clean_eda_data(raw_df)
    clean_df.to_csv(output_dir / "cleaned_weather_for_lightgbm.csv", index=False)

    split_data, split_info, feature_engineering = build_supervised_table(
        df=clean_df,
        feature_cols=MODEL_FEATURES,
        target_col=TARGET_COLUMN,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_gap_minutes=args.max_gap_minutes,
    )

    best_param_overrides: dict[str, object] | None = None
    hyperparameter_search_summary: dict[str, object] = {"enabled": False}
    if args.tune_hyperparameters:
        (
            best_param_overrides,
            search_results,
            search_fold_results,
            hyperparameter_search_summary,
        ) = run_hyperparameter_search(
            lgb=lgb,
            train_x=split_data["train"]["x"],
            train_y=split_data["train"]["y"],
            args=args,
            output_dir=output_dir,
        )
    else:
        search_results = pd.DataFrame()
        search_fold_results = pd.DataFrame()

    booster, history, final_model_params = train_model(
        lgb=lgb,
        train_x=split_data["train"]["x"],
        train_y=split_data["train"]["y"],
        val_x=split_data["val"]["x"],
        val_y=split_data["val"]["y"],
        args=args,
        params_override=best_param_overrides,
    )

    best_iteration = int(booster.best_iteration or booster.current_iteration())

    metrics: dict[str, dict[str, float]] = {}
    split_predictions: dict[str, dict[str, np.ndarray]] = {}
    for split_name, values in split_data.items():
        y_pred = predict(
            booster=booster,
            features=values["x"],
            num_iteration=best_iteration,
        )
        y_true = values["y"]
        metrics[split_name] = regression_metrics(y_true, y_pred)
        split_predictions[split_name] = {
            "date": values["date"],
            "y_true": y_true,
            "y_pred": y_pred,
        }

    validation_residuals = (
        split_predictions["val"]["y_true"] - split_predictions["val"]["y_pred"]
    )
    prediction_band = calibrate_prediction_interval(
        residuals=validation_residuals,
        coverage=args.interval_coverage,
        reference_split="val",
    )
    val_lower, val_upper = build_prediction_interval(
        split_predictions["val"]["y_pred"],
        prediction_band,
    )
    prediction_band["empirical_coverage_reference_split"] = float(
        np.mean(
            (split_predictions["val"]["y_true"] >= val_lower)
            & (split_predictions["val"]["y_true"] <= val_upper)
        )
    )
    prediction_band["mean_interval_width_f"] = float(np.mean(val_upper - val_lower))

    prediction_frames: list[pd.DataFrame] = []
    for split_name, values in split_predictions.items():
        lower_interval, upper_interval = build_prediction_interval(
            values["y_pred"],
            prediction_band,
        )
        inside_interval = (values["y_true"] >= lower_interval) & (
            values["y_true"] <= upper_interval
        )
        metrics[split_name]["coverage_with_prediction_interval"] = float(
            np.mean(inside_interval)
        )
        metrics[split_name]["mean_prediction_interval_width_f"] = float(
            np.mean(upper_interval - lower_interval)
        )
        prediction_frames.append(
            build_predictions_frame(
                split_name=split_name,
                dates=values["date"],
                y_true=values["y_true"],
                y_pred=values["y_pred"],
                lower_interval=lower_interval,
                upper_interval=upper_interval,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    history.to_csv(output_dir / "history.csv", index=False)
    save_plots(history, predictions, prediction_band, output_dir)
    importance = save_feature_importance(booster, output_dir)

    booster.save_model(
        str(output_dir / "lightgbm_daily_temperature.txt"),
        num_iteration=best_iteration,
    )

    metadata = {
        "model_type": "lightgbm",
        "feature_columns": split_data["train"]["x"].columns.tolist(),
        "target_column": TARGET_COLUMN,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "max_gap_minutes": args.max_gap_minutes,
        "best_iteration": int(best_iteration),
        "split_info": split_info,
        "feature_engineering": feature_engineering,
        "lightgbm_params": normalize_param_dict(final_model_params),
        "hyperparameter_search": hyperparameter_search_summary,
    }
    with (output_dir / "lightgbm_daily_temperature_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(metadata, fp, indent=2)

    summary = {
        "device": "cpu",
        "rows_after_cleaning": int(len(clean_df)),
        "feature_count": len(MODEL_FEATURES),
        "engineered_feature_count": int(split_data["train"]["x"].shape[1]),
        "sequence_counts": {
            split: int(values["x"].shape[0]) for split, values in split_data.items()
        },
        "split_info": split_info,
        "metrics": metrics,
        "prediction_band": prediction_band,
        "model_type": "lightgbm",
        "model_config": {
            "num_leaves": int(final_model_params["num_leaves"]),
            "max_depth": int(final_model_params["max_depth"]),
            "min_data_in_leaf": int(final_model_params["min_data_in_leaf"]),
            "feature_fraction": float(final_model_params["feature_fraction"]),
            "bagging_fraction": float(final_model_params["bagging_fraction"]),
            "bagging_freq": int(final_model_params["bagging_freq"]),
            "lambda_l1": float(final_model_params["lambda_l1"]),
            "lambda_l2": float(final_model_params["lambda_l2"]),
            "min_gain_to_split": float(final_model_params["min_gain_to_split"]),
            "max_bin": int(final_model_params["max_bin"]),
            "learning_rate": float(final_model_params["learning_rate"]),
            "best_iteration": int(best_iteration),
            "top_feature_by_gain": (
                str(importance.iloc[0]["feature"]) if not importance.empty else None
            ),
        },
        "feature_engineering": feature_engineering,
        "training_config": {
            "objective": "regression",
            "metrics": ["l1", "l2"],
            "learning_rate": float(final_model_params["learning_rate"]),
            "num_boost_round": int(args.num_boost_round),
            "patience": int(args.patience),
        },
        "hyperparameter_search": hyperparameter_search_summary,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\nResumen final")
    print(json.dumps(summary, indent=2))

    if args.tune_hyperparameters and not search_results.empty:
        print("\nTop 5 configuraciones encontradas")
        cols = [
            "trial",
            "cv_mae_mean",
            "cv_rmse_mean",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_data_in_leaf",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "lambda_l1",
            "lambda_l2",
            "min_gain_to_split",
            "max_bin",
        ]
        display_cols = [col for col in cols if col in search_results.columns]
        print(search_results[display_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()