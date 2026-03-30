from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path("/content")

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
SEARCH_PARAM_NAMES = [
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
]
SEARCH_PARAM_CANDIDATES: dict[str, list[float | int]] = {
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [31, 63, 127, 255, 511],
    "max_depth": [-1, 6, 8, 10, 12],
    "min_data_in_leaf": [10, 20, 40, 60, 80, 100, 120],
    "feature_fraction": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "bagging_fraction": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "bagging_freq": [0, 1, 3, 5],
    "lambda_l1": [0.0, 0.1, 0.5, 1.0],
    "lambda_l2": [0.0, 0.5, 1.0, 2.0, 5.0],
    "min_gain_to_split": [0.0, 0.01, 0.05, 0.1],
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo LightGBM con observaciones intradiarias para "
            "predecir temperature_f."
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
        default=6,
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
        "--search-trials",
        type=int,
        default=50,
        help=(
            "Cantidad de trials aleatorios para buscar mejores hiperparametros. "
            "Usa 0 para desactivar la busqueda y entrenar solo con los valores actuales."
        ),
    )
    parser.add_argument(
        "--search-metric",
        choices=["mae", "rmse", "r2"],
        default="mae",
        help="Metrica de validacion usada para elegir el mejor trial.",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=120,
        help="Gap maximo permitido entre observaciones consecutivas para una muestra valida.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="Cantidad de hilos para LightGBM. 0 deja el default del backend.",
    )
    parser.add_argument("--seed", type=int, default=24217956)
    return parser.parse_args([])


def require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "No se encontro el paquete 'lightgbm'. Instala las dependencias de "
            "PrediccionClima/Daily/requirements.txt o ejecuta 'pip install lightgbm'."
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


def build_lgbm_params(
    args: argparse.Namespace,
    overrides: dict[str, float | int] | None = None,
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
        "seed": args.seed,
        "verbosity": -1,
        "feature_pre_filter": False,
        "force_col_wise": True,
    }
    if overrides:
        params.update(overrides)
    if args.num_threads > 0:
        params["num_threads"] = args.num_threads
    return params


def build_trial_signature(params: dict[str, float | int]) -> tuple[float | int, ...]:
    return tuple(params[name] for name in SEARCH_PARAM_NAMES)


def build_search_trials(args: argparse.Namespace) -> list[dict[str, object]]:
    baseline_params = {name: getattr(args, name) for name in SEARCH_PARAM_NAMES}
    trials: list[dict[str, object]] = [
        {"trial_id": 0, "trial_source": "baseline", "params": baseline_params}
    ]
    if args.search_trials <= 0:
        return trials

    rng = random.Random(args.seed)
    seen_signatures = {build_trial_signature(baseline_params)}
    attempts = 0
    max_attempts = max(args.search_trials * 25, 100)

    while len(trials) < args.search_trials + 1 and attempts < max_attempts:
        attempts += 1
        sampled_params = {
            name: rng.choice(SEARCH_PARAM_CANDIDATES[name]) for name in SEARCH_PARAM_NAMES
        }
        signature = build_trial_signature(sampled_params)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        trials.append(
            {
                "trial_id": len(trials),
                "trial_source": "random_search",
                "params": sampled_params,
            }
        )
    return trials


def is_better_search_score(
    candidate_score: float,
    current_best_score: float | None,
    search_metric: str,
) -> bool:
    if current_best_score is None:
        return True
    if search_metric == "r2":
        return candidate_score > current_best_score
    return candidate_score < current_best_score


def compute_search_score(metrics: dict[str, float], search_metric: str) -> float:
    return float(metrics[search_metric])


def train_model(
    lgb: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    params: dict[str, object],
    num_boost_round: int,
    patience: int,
    log_every: int,
) -> tuple[Any, pd.DataFrame, dict[str, object]]:
    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    val_set = lgb.Dataset(val_x, label=val_y, reference=train_set, free_raw_data=False)

    evals_result: dict[str, dict[str, list[float]]] = {}
    callbacks: list[Any] = [
        lgb.early_stopping(
            stopping_rounds=patience,
            first_metric_only=True,
            verbose=True,
        ),
        lgb.record_evaluation(evals_result),
    ]
    if log_every > 0:
        callbacks.append(lgb.log_evaluation(period=log_every))

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    history = build_history_frame(evals_result)
    return booster, history, params


def predict(
    booster: Any,
    features: pd.DataFrame,
    num_iteration: int,
) -> np.ndarray:
    return booster.predict(features, num_iteration=num_iteration)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def run_hyperparameter_search(
    lgb: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, pd.DataFrame, dict[str, object], pd.DataFrame, dict[str, object]]:
    trials = build_search_trials(args)
    search_rows: list[dict[str, object]] = []

    best_bundle: tuple[Any, pd.DataFrame, dict[str, object]] | None = None
    best_trial_row: dict[str, object] | None = None
    best_search_score: float | None = None

    for trial_index, trial in enumerate(trials, start=1):
        trial_params = build_lgbm_params(
            args=args,
            overrides=trial["params"],
        )
        print(
            f"\nSearch trial {trial_index}/{len(trials)} | "
            f"id={trial['trial_id']} | source={trial['trial_source']}"
        )
        booster, history, fitted_params = train_model(
            lgb=lgb,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            params=trial_params,
            num_boost_round=args.num_boost_round,
            patience=args.patience,
            log_every=0,
        )
        best_iteration = booster.best_iteration or booster.current_iteration()
        val_pred = predict(booster, val_x, num_iteration=best_iteration)
        val_metrics = regression_metrics(val_y, val_pred)
        search_score = compute_search_score(val_metrics, args.search_metric)

        trial_row: dict[str, object] = {
            "trial_id": int(trial["trial_id"]),
            "trial_source": str(trial["trial_source"]),
            "best_iteration": int(best_iteration),
            "search_metric": args.search_metric,
            "search_score": float(search_score),
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mape": float(val_metrics["mape"]),
            "val_r2": float(val_metrics["r2"]),
        }
        for param_name in SEARCH_PARAM_NAMES:
            trial_row[param_name] = fitted_params[param_name]
        search_rows.append(trial_row)

        print(
            f"  val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if is_better_search_score(search_score, best_search_score, args.search_metric):
            best_search_score = search_score
            best_bundle = (booster, history, fitted_params)
            best_trial_row = trial_row

    if best_bundle is None or best_trial_row is None:
        raise RuntimeError("La busqueda de hiperparametros no produjo ningun modelo valido.")

    search_results = pd.DataFrame(search_rows)
    ascending = args.search_metric != "r2"
    search_results = search_results.sort_values(
        ["search_score", "val_rmse", "trial_id"],
        ascending=[ascending, True, True],
    ).reset_index(drop=True)
    search_results.insert(0, "rank", np.arange(1, len(search_results) + 1))

    search_summary = {
        "enabled": bool(args.search_trials > 0),
        "trial_count": int(len(search_results)),
        "search_metric": args.search_metric,
        "best_trial_id": int(best_trial_row["trial_id"]),
        "best_trial_source": str(best_trial_row["trial_source"]),
        "best_search_score": float(best_trial_row["search_score"]),
        "best_params": {
            param_name: best_trial_row[param_name] for param_name in SEARCH_PARAM_NAMES
        },
    }
    print(
        "\nMejor trial | "
        f"id={search_summary['best_trial_id']} | "
        f"source={search_summary['best_trial_source']} | "
        f"{args.search_metric}={search_summary['best_search_score']:.4f}"
    )
    booster, history, params = best_bundle
    return booster, history, params, search_results, search_summary


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

    booster, history, model_params, search_results, search_summary = run_hyperparameter_search(
        lgb=lgb,
        train_x=split_data["train"]["x"],
        train_y=split_data["train"]["y"],
        val_x=split_data["val"]["x"],
        val_y=split_data["val"]["y"],
        args=args,
    )
    search_results.to_csv(output_dir / "hyperparameter_search.csv", index=False)
    with (output_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as fp:
        json.dump(search_summary, fp, indent=2)

    best_iteration = booster.best_iteration or booster.current_iteration()

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
        "lightgbm_params": model_params,
        "hyperparameter_search": search_summary,
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
            "num_leaves": int(model_params["num_leaves"]),
            "max_depth": int(model_params["max_depth"]),
            "min_data_in_leaf": int(model_params["min_data_in_leaf"]),
            "feature_fraction": float(model_params["feature_fraction"]),
            "bagging_fraction": float(model_params["bagging_fraction"]),
            "bagging_freq": int(model_params["bagging_freq"]),
            "lambda_l1": float(model_params["lambda_l1"]),
            "lambda_l2": float(model_params["lambda_l2"]),
            "min_gain_to_split": float(model_params["min_gain_to_split"]),
            "best_iteration": int(best_iteration),
            "top_feature_by_gain": (
                str(importance.iloc[0]["feature"]) if not importance.empty else None
            ),
        },
        "feature_engineering": feature_engineering,
        "hyperparameter_search": search_summary,
        "training_config": {
            "objective": "regression",
            "metrics": ["l1", "l2"],
            "learning_rate": float(model_params["learning_rate"]),
            "num_boost_round": args.num_boost_round,
            "patience": args.patience,
        },
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\nResumen final")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
