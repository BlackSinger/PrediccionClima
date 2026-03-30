from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

import config_store
import weather_common

LAG_SOURCE_FEATURES = config_store.load_section(
    "lightgbm",
    "feature_engineering",
    "lag_source_features",
)
LAG_STEP_CANDIDATES = config_store.load_section(
    "lightgbm",
    "feature_engineering",
    "lag_step_candidates",
)
ROLLING_WINDOW_CANDIDATES = config_store.load_section(
    "lightgbm",
    "feature_engineering",
    "rolling_window_candidates",
)
SEARCH_PARAM_NAMES = config_store.load_section("lightgbm", "search", "param_names")
SEARCH_PARAM_CANDIDATES: dict[str, list[float | int]] = config_store.load_section(
    "lightgbm",
    "search",
    "param_candidates",
)


def get_default_train_config() -> dict[str, object]:
    shared = config_store.load_section("shared")
    train_defaults = config_store.load_section("lightgbm", "train_defaults")
    return {
        **train_defaults,
        "horizon": int(shared["forecast_horizon"]),
        "train_ratio": float(shared["train_ratio"]),
        "val_ratio": float(shared["val_ratio"]),
        "interval_coverage": float(shared["interval_coverage"]),
        "max_gap_minutes": int(shared["max_gap_minutes"]),
        "seed": int(shared["seed"]),
    }


DEFAULT_TRAIN_CONFIG = get_default_train_config()


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
    run_lengths = weather_common.compute_run_lengths(
        df["observation_datetime"],
        max_gap_minutes,
    )

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
    train_config: dict[str, object],
    overrides: dict[str, float | int] | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "regression",
        "metric": ["l1", "l2"],
        "boosting_type": "gbdt",
        "learning_rate": float(train_config["learning_rate"]),
        "num_leaves": int(train_config["num_leaves"]),
        "max_depth": int(train_config["max_depth"]),
        "min_data_in_leaf": int(train_config["min_data_in_leaf"]),
        "feature_fraction": float(train_config["feature_fraction"]),
        "bagging_fraction": float(train_config["bagging_fraction"]),
        "bagging_freq": int(train_config["bagging_freq"]),
        "lambda_l1": float(train_config["lambda_l1"]),
        "lambda_l2": float(train_config["lambda_l2"]),
        "min_gain_to_split": float(train_config["min_gain_to_split"]),
        "seed": int(train_config["seed"]),
        "verbosity": -1,
        "feature_pre_filter": False,
        "force_col_wise": True,
    }
    if overrides:
        params.update(overrides)
    if int(train_config["num_threads"]) > 0:
        params["num_threads"] = int(train_config["num_threads"])
    return params


def build_trial_signature(params: dict[str, float | int]) -> tuple[float | int, ...]:
    return tuple(params[name] for name in SEARCH_PARAM_NAMES)


def build_search_trials(train_config: dict[str, object]) -> list[dict[str, object]]:
    baseline_params = {name: train_config[name] for name in SEARCH_PARAM_NAMES}
    trials: list[dict[str, object]] = [
        {"trial_id": 0, "trial_source": "baseline", "params": baseline_params}
    ]
    search_trials = int(train_config["search_trials"])
    if search_trials <= 0:
        return trials

    rng = random.Random(int(train_config["seed"]))
    seen_signatures = {build_trial_signature(baseline_params)}
    attempts = 0
    max_attempts = max(search_trials * 25, 100)

    while len(trials) < search_trials + 1 and attempts < max_attempts:
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def train_model(
    lgb: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    params: dict[str, object],
    num_boost_round: int,
    patience: int,
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


def run_hyperparameter_search(
    lgb: Any,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    train_config: dict[str, object],
) -> tuple[Any, pd.DataFrame, dict[str, object], pd.DataFrame, dict[str, object]]:
    trials = build_search_trials(train_config)
    search_rows: list[dict[str, object]] = []

    best_bundle: tuple[Any, pd.DataFrame, dict[str, object]] | None = None
    best_trial_row: dict[str, object] | None = None
    best_search_score: float | None = None
    search_metric = str(train_config["search_metric"])

    for trial_index, trial in enumerate(trials, start=1):
        trial_params = build_lgbm_params(
            train_config=train_config,
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
            num_boost_round=int(train_config["num_boost_round"]),
            patience=int(train_config["patience"]),
        )
        best_iteration = booster.best_iteration or booster.current_iteration()
        val_pred = predict(booster, val_x, num_iteration=best_iteration)
        val_metrics = regression_metrics(val_y, val_pred)
        search_score = compute_search_score(val_metrics, search_metric)

        trial_row: dict[str, object] = {
            "trial_id": int(trial["trial_id"]),
            "trial_source": str(trial["trial_source"]),
            "best_iteration": int(best_iteration),
            "search_metric": search_metric,
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

        if is_better_search_score(search_score, best_search_score, search_metric):
            best_search_score = search_score
            best_bundle = (booster, history, fitted_params)
            best_trial_row = trial_row

    if best_bundle is None or best_trial_row is None:
        raise RuntimeError("La busqueda de hiperparametros no produjo ningun modelo valido.")

    search_results = pd.DataFrame(search_rows)
    ascending = search_metric != "r2"
    search_results = search_results.sort_values(
        ["search_score", "val_rmse", "trial_id"],
        ascending=[ascending, True, True],
    ).reset_index(drop=True)
    search_results.insert(0, "rank", np.arange(1, len(search_results) + 1))

    search_summary = {
        "enabled": bool(int(train_config["search_trials"]) > 0),
        "trial_count": int(len(search_results)),
        "search_metric": search_metric,
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
        f"{search_metric}={search_summary['best_search_score']:.4f}"
    )
    booster, history, params = best_bundle
    return booster, history, params, search_results, search_summary


def calibrate_prediction_interval(
    residuals: np.ndarray,
    coverage: float,
    reference_split: str,
) -> dict[str, float | str]:
    if not 0.0 < coverage < 1.0:
        raise ValueError("interval_coverage debe estar entre 0 y 1.")

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


def predict(
    booster: Any,
    features: pd.DataFrame,
    num_iteration: int,
) -> np.ndarray:
    return booster.predict(features, num_iteration=num_iteration)


def load_bundle(
    lightgbm_model_path: Path,
    lightgbm_metadata_path: Path,
) -> dict[str, object]:
    lgb = require_lightgbm()
    metadata = json.loads(lightgbm_metadata_path.read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(lightgbm_model_path))
    return {"booster": booster, "metadata": metadata}


def build_buckets(
    clean_df: pd.DataFrame,
    metadata: dict[str, object],
) -> dict[str, dict[str, object]]:
    lookback = int(metadata["lookback"])
    horizon = int(metadata["horizon"])
    max_gap_minutes = int(metadata["max_gap_minutes"])
    train_end_timestamp, val_end_timestamp = weather_common.get_split_cutoffs(
        metadata["split_info"]
    )
    lag_source_features = list(metadata["feature_engineering"]["lag_source_features"])
    lag_steps = list(metadata["feature_engineering"]["lag_steps"])
    rolling_windows = list(metadata["feature_engineering"]["rolling_windows"])
    feature_cols = weather_common.MODEL_FEATURES
    feature_names = list(metadata["feature_columns"])

    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    feature_values = clean_df[feature_cols].to_numpy(dtype=np.float32)
    target_values = clean_df[str(metadata["target_column"])].to_numpy(dtype=np.float32)
    run_lengths = weather_common.compute_run_lengths(
        clean_df["observation_datetime"],
        max_gap_minutes,
    )

    buckets: dict[str, dict[str, list[object]]] = {
        "train": {"x": [], "y": [], "date": []},
        "val": {"x": [], "y": [], "date": []},
        "test": {"x": [], "y": [], "date": []},
    }
    required_run = lookback + horizon
    for target_idx in range(required_run - 1, len(clean_df)):
        if run_lengths[target_idx] < required_run:
            continue

        input_end = target_idx - horizon + 1
        input_start = input_end - lookback
        if input_start < 0:
            continue

        target_timestamp = pd.Timestamp(clean_df.iloc[target_idx]["observation_datetime"])
        split_name = weather_common.assign_split(
            target_timestamp,
            train_end_timestamp,
            val_end_timestamp,
        )
        window = feature_values[input_start:input_end]
        row_values = build_feature_row(
            window=window,
            feature_cols=feature_cols,
            feature_index=feature_index,
            lag_source_features=lag_source_features,
            lag_steps=lag_steps,
            rolling_windows=rolling_windows,
        )
        buckets[split_name]["x"].append(row_values)
        buckets[split_name]["y"].append(float(target_values[target_idx]))
        buckets[split_name]["date"].append(target_timestamp.to_datetime64())

    arrays: dict[str, dict[str, object]] = {}
    for split_name, values in buckets.items():
        arrays[split_name] = {
            "x": pd.DataFrame(values["x"], columns=feature_names, dtype=np.float32),
            "y": np.asarray(values["y"], dtype=np.float32),
            "date": np.asarray(values["date"], dtype="datetime64[ns]"),
        }
    return arrays


def predict_split(
    lightgbm_bundle: dict[str, object],
    split_arrays: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    metadata = lightgbm_bundle["metadata"]
    best_iteration = int(metadata["best_iteration"])
    y_pred = predict(
        lightgbm_bundle["booster"],
        split_arrays["x"],
        num_iteration=best_iteration,
    )
    y_true = np.asarray(split_arrays["y"], dtype=np.float32)
    return y_true, y_pred


def build_live_prediction(
    clean_df: pd.DataFrame,
    lightgbm_bundle: dict[str, object],
) -> float:
    metadata = lightgbm_bundle["metadata"]
    lookback = int(metadata["lookback"])
    run_lengths = weather_common.compute_run_lengths(
        clean_df["observation_datetime"],
        int(metadata["max_gap_minutes"]),
    )
    if int(run_lengths[-1]) < lookback:
        raise ValueError(
            "No hay suficientes observaciones consecutivas para inferir con LightGBM."
        )

    lag_source_features = list(metadata["feature_engineering"]["lag_source_features"])
    lag_steps = list(metadata["feature_engineering"]["lag_steps"])
    rolling_windows = list(metadata["feature_engineering"]["rolling_windows"])
    feature_cols = weather_common.MODEL_FEATURES
    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    window = clean_df[feature_cols].to_numpy(dtype=np.float32)[-lookback:]
    row_values = build_feature_row(
        window=window,
        feature_cols=feature_cols,
        feature_index=feature_index,
        lag_source_features=lag_source_features,
        lag_steps=lag_steps,
        rolling_windows=rolling_windows,
    )
    feature_frame = pd.DataFrame(
        [row_values],
        columns=list(metadata["feature_columns"]),
        dtype=np.float32,
    )
    pred = predict(
        lightgbm_bundle["booster"],
        feature_frame,
        num_iteration=int(metadata["best_iteration"]),
    )
    return float(np.asarray(pred).squeeze())


def train_and_save(
    data_path: Path,
    output_dir: Path,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    train_config = dict(DEFAULT_TRAIN_CONFIG)
    if config:
        train_config.update(config)

    set_seed(int(train_config["seed"]))
    lgb = require_lightgbm()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(data_path)
    clean_df = weather_common.clean_eda_data(raw_df)
    clean_df.to_csv(output_dir / "cleaned_weather_for_lightgbm.csv", index=False)

    split_data, split_info, feature_engineering = build_supervised_table(
        df=clean_df,
        feature_cols=weather_common.MODEL_FEATURES,
        target_col=weather_common.TARGET_COLUMN,
        lookback=int(train_config["lookback"]),
        horizon=int(train_config["horizon"]),
        train_ratio=float(train_config["train_ratio"]),
        val_ratio=float(train_config["val_ratio"]),
        max_gap_minutes=int(train_config["max_gap_minutes"]),
    )

    (
        booster,
        history,
        model_params,
        search_results,
        search_summary,
    ) = run_hyperparameter_search(
        lgb=lgb,
        train_x=split_data["train"]["x"],
        train_y=split_data["train"]["y"],
        val_x=split_data["val"]["x"],
        val_y=split_data["val"]["y"],
        train_config=train_config,
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
        y_true = np.asarray(values["y"], dtype=np.float32)
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
        coverage=float(train_config["interval_coverage"]),
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
            pd.DataFrame(
                {
                    "observation_datetime": pd.to_datetime(values["date"]),
                    "split": split_name,
                    "actual_temperature_f": values["y_true"],
                    "predicted_temperature_f": values["y_pred"],
                    "lower_prediction_interval_f": lower_interval,
                    "upper_prediction_interval_f": upper_interval,
                }
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    history.to_csv(output_dir / "history.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain_importance": booster.feature_importance(importance_type="gain"),
            "split_importance": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain_importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    booster.save_model(
        str(output_dir / "lightgbm_daily_temperature.txt"),
        num_iteration=best_iteration,
    )

    metadata = {
        "model_type": "lightgbm",
        "feature_columns": split_data["train"]["x"].columns.tolist(),
        "target_column": weather_common.TARGET_COLUMN,
        "lookback": int(train_config["lookback"]),
        "horizon": int(train_config["horizon"]),
        "max_gap_minutes": int(train_config["max_gap_minutes"]),
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
        "feature_count": len(weather_common.MODEL_FEATURES),
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
            "num_boost_round": int(train_config["num_boost_round"]),
            "patience": int(train_config["patience"]),
            "search_trials": int(train_config["search_trials"]),
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return {
        "summary": summary,
        "artifact_path": output_dir / "lightgbm_daily_temperature.txt",
        "metadata_path": output_dir / "lightgbm_daily_temperature_metadata.json",
        "cleaned_data_path": output_dir / "cleaned_weather_for_lightgbm.csv",
    }
