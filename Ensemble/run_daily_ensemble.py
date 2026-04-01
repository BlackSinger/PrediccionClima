from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from argparse import BooleanOptionalAction
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config_store
import gru_model
import lightgbm_model
import weather_common
import weather_update

SCRIPT_DIR = Path(__file__).resolve().parent
APP_CONFIG = config_store.load_config()
SHARED_DEFAULTS = APP_CONFIG["shared"]
PATH_DEFAULTS = APP_CONFIG["paths"]
RUNTIME_DEFAULTS = APP_CONFIG["ensemble"]["runtime_defaults"]


def print_gpu_status() -> dict[str, object]:
    """Imprime y retorna el estado de GPU disponible para el ensemble."""
    gpu_available = torch.cuda.is_available()
    gpu_info: dict[str, object] = {
        "cuda_available": gpu_available,
        "device_count": torch.cuda.device_count() if gpu_available else 0,
    }
    if gpu_available:
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["cuda_version"] = torch.version.cuda or "unknown"
        mem = torch.cuda.get_device_properties(0).total_mem
        gpu_info["total_memory_gb"] = round(mem / (1024 ** 3), 2)
        print(f"\n🚀 GPU detectada: {gpu_info['device_name']}")
        print(f"   CUDA {gpu_info['cuda_version']} | {gpu_info['total_memory_gb']} GB")
        print(f"   Los modelos usaran GPU para acelerar los calculos.\n")
    else:
        print("\n⚠️  No se detecto GPU con CUDA. Ejecutando en CPU.\n")
    return gpu_info


def resolve_path(path_value: Path, base_dir: Path | None = None) -> Path:
    if path_value.is_absolute():
        return path_value
    if base_dir is None:
        return (Path.cwd() / path_value).resolve()
    return (base_dir / path_value).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Actualiza el CSV meteorologico si hace falta y calcula un ensemble "
            "entre los modelos GRU y LightGBM."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=config_store.resolve_config_path(PATH_DEFAULTS["data"]),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config_store.resolve_config_path(PATH_DEFAULTS["output_root"]),
    )
    parser.add_argument(
        "--gru-artifact",
        type=Path,
        default=config_store.resolve_config_path(PATH_DEFAULTS["gru_artifact"]),
    )
    parser.add_argument(
        "--gru-cleaned-data",
        type=Path,
        default=config_store.resolve_config_path(PATH_DEFAULTS["gru_cleaned_data"]),
    )
    parser.add_argument(
        "--lightgbm-model",
        type=Path,
        default=config_store.resolve_config_path(PATH_DEFAULTS["lightgbm_model"]),
    )
    parser.add_argument(
        "--lightgbm-metadata",
        type=Path,
        default=config_store.resolve_config_path(PATH_DEFAULTS["lightgbm_metadata"]),
    )
    parser.add_argument(
        "--lightgbm-cleaned-data",
        type=Path,
        default=config_store.resolve_config_path(PATH_DEFAULTS["lightgbm_cleaned_data"]),
    )
    parser.add_argument(
        "--weight-grid-step",
        type=float,
        default=float(RUNTIME_DEFAULTS["weight_grid_step"]),
        help="Paso de la grilla para buscar el peso optimo de la GRU en validacion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(RUNTIME_DEFAULTS["batch_size"]),
        help="Batch size para inferencia de la GRU sobre splits historicos.",
    )
    parser.add_argument(
        "--headless",
        action=BooleanOptionalAction,
        default=bool(RUNTIME_DEFAULTS["headless"]),
        help="Ejecuta Chromium en modo headless al actualizar datos.",
    )
    parser.add_argument(
        "--update-data",
        action=BooleanOptionalAction,
        default=bool(RUNTIME_DEFAULTS["update_data"]),
        help="Si falta informacion hasta hoy, descarga los dias faltantes.",
    )
    parser.add_argument(
        "--retrain-models",
        action=BooleanOptionalAction,
        default=bool(RUNTIME_DEFAULTS["retrain_models"]),
        help="Reentrena GRU y LightGBM despues de actualizar los datos.",
    )
    parser.add_argument(
        "--lightgbm-search-trials",
        type=int,
        default=int(RUNTIME_DEFAULTS["lightgbm_search_trials"]),
        help=(
            "Cantidad de trials de busqueda para LightGBM durante el reentrenamiento. "
            "En flujo operativo conviene dejarlo en 0 para usar los mejores params conocidos."
        ),
    )
    return parser.parse_args()


def build_output_layout(output_root: Path, run_started_at: datetime) -> dict[str, object]:
    run_id = run_started_at.strftime("%Y-%m-%d_%H%M%S")
    runs_dir = output_root / PATH_DEFAULTS["runs_directory"]
    latest_dir = output_root / PATH_DEFAULTS["latest_directory"]
    status_path = output_root / PATH_DEFAULTS["status_file"]
    run_dir = runs_dir / run_id
    return {
        "run_id": run_id,
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "latest_dir": latest_dir,
        "status_path": status_path,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def refresh_latest_directory(run_dir: Path, latest_dir: Path) -> None:
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)


def snapshot_model_artifacts(
    run_dir: Path,
    gru_artifact_path: Path,
    lightgbm_model_path: Path,
) -> None:
    snapshot_root = run_dir / "models"
    gru_snapshot_dir = snapshot_root / gru_artifact_path.parent.name
    lightgbm_snapshot_dir = snapshot_root / lightgbm_model_path.parent.name

    if gru_snapshot_dir.exists():
        shutil.rmtree(gru_snapshot_dir)
    if lightgbm_snapshot_dir.exists():
        shutil.rmtree(lightgbm_snapshot_dir)

    shutil.copytree(gru_artifact_path.parent, gru_snapshot_dir)
    shutil.copytree(lightgbm_model_path.parent, lightgbm_snapshot_dir)


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def serialize_value(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def build_last_observation_snapshot(
    raw_df: pd.DataFrame,
    last_observation_datetime: str,
) -> dict[str, object]:
    if raw_df.empty:
        return {"observation_datetime": last_observation_datetime}

    last_row = raw_df.iloc[-1]
    snapshot = {column: serialize_value(last_row[column]) for column in raw_df.columns}
    snapshot["observation_datetime"] = last_observation_datetime
    return snapshot


def compute_file_metadata(path: Path) -> dict[str, object]:
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)

    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": hasher.hexdigest(),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(
            timespec="seconds"
        ),
    }


def collect_model_metrics(
    gru_bundle: dict[str, object],
    lightgbm_bundle: dict[str, object],
    gru_artifact_path: Path,
    lightgbm_model_path: Path,
    retraining_summary: dict[str, object] | None,
    ensemble_metrics: dict[str, dict[str, float]] | None,
) -> dict[str, object]:
    gru_summary = (
        retraining_summary.get("gru") if retraining_summary else None
    ) or read_json_if_exists(gru_artifact_path.parent / "metrics.json")
    lightgbm_summary = (
        retraining_summary.get("lightgbm") if retraining_summary else None
    ) or read_json_if_exists(lightgbm_model_path.parent / "metrics.json")

    gru_metrics = None
    if isinstance(gru_summary, dict):
        gru_metrics = gru_summary.get("metrics")
    if gru_metrics is None:
        gru_metrics = gru_bundle["payload"].get("metrics")

    lightgbm_metrics = None
    if isinstance(lightgbm_summary, dict):
        lightgbm_metrics = lightgbm_summary.get("metrics")

    return {
        "gru": gru_metrics,
        "lightgbm": lightgbm_metrics,
        "ensemble": ensemble_metrics,
    }


def build_execution_manifest(
    data_path: Path,
    raw_df: pd.DataFrame,
    data_update_summary: dict[str, object],
    retraining_summary: dict[str, object] | None,
    gru_bundle: dict[str, object],
    lightgbm_bundle: dict[str, object],
    gru_artifact_path: Path,
    lightgbm_model_path: Path,
    compatibility_issues: list[str],
    ensemble_metrics: dict[str, dict[str, float]] | None,
    gpu_info: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu_info": gpu_info,
        "last_observation_used": build_last_observation_snapshot(
            raw_df=raw_df,
            last_observation_datetime=str(data_update_summary["last_observation_datetime"]),
        ),
        "config_snapshot": config_store.load_config(),
        "data_update": data_update_summary,
        "csv_metadata": compute_file_metadata(data_path),
        "metrics": collect_model_metrics(
            gru_bundle=gru_bundle,
            lightgbm_bundle=lightgbm_bundle,
            gru_artifact_path=gru_artifact_path,
            lightgbm_model_path=lightgbm_model_path,
            retraining_summary=retraining_summary,
            ensemble_metrics=ensemble_metrics,
        ),
        "compatibility_issues": compatibility_issues,
        "retraining": retraining_summary,
    }


def build_status_payload(
    run_id: str,
    run_dir: Path,
    latest_dir: Path,
    summary: dict[str, object],
    execution_manifest: dict[str, object],
    live_forecast: dict[str, object] | None,
) -> dict[str, object]:
    compatibility_issues = list(summary.get("compatibility_issues", []))
    status_value = "ready" if not compatibility_issues and live_forecast else "blocked"
    return {
        "status": status_value,
        "generated_at": execution_manifest["generated_at"],
        "run_id": run_id,
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "last_observation_used": execution_manifest["last_observation_used"],
        "data_update": execution_manifest["data_update"],
        "csv_metadata": execution_manifest["csv_metadata"],
        "compatibility_issues": compatibility_issues,
        "retraining": execution_manifest["retraining"],
        "metrics": execution_manifest["metrics"],
        "gru_model": summary.get("gru_model"),
        "lightgbm_model": summary.get("lightgbm_model"),
        "ensemble_weights": summary.get("ensemble_weights"),
        "prediction_band": summary.get("prediction_band"),
        "live_forecast": live_forecast,
    }


def validate_model_compatibility(
    gru_bundle: dict[str, object],
    lightgbm_bundle: dict[str, object],
) -> list[str]:
    payload = gru_bundle["payload"]
    metadata = lightgbm_bundle["metadata"]
    issues: list[str] = []
    gru_horizons = weather_common.normalize_forecast_horizons(
        payload.get("forecast_horizons"),
        fallback_horizon=int(payload["horizon"]),
    )
    lightgbm_horizons = weather_common.normalize_forecast_horizons(
        metadata.get("forecast_horizons"),
        fallback_horizon=int(metadata["horizon"]),
    )

    if payload["target_column"] != metadata["target_column"]:
        issues.append(
            "target_column distinto: "
            f"GRU={payload['target_column']} vs LightGBM={metadata['target_column']}"
        )
    if int(payload["horizon"]) != int(metadata["horizon"]):
        issues.append(
            "horizon distinto: "
            f"GRU={payload['horizon']} vs LightGBM={metadata['horizon']}"
        )
    if gru_horizons != lightgbm_horizons:
        issues.append(
            "forecast_horizons distintos: "
            f"GRU={gru_horizons} vs LightGBM={lightgbm_horizons}"
        )
    return issues


def retrain_models(
    data_path: Path,
    gru_artifact_path: Path,
    lightgbm_model_path: Path,
    lightgbm_search_trials: int,
) -> dict[str, object]:
    gru_output_dir = gru_artifact_path.parent
    lightgbm_output_dir = lightgbm_model_path.parent

    print("\nReentrenando GRU con los datos mas recientes...")
    gru_result = gru_model.train_and_save(
        data_path=data_path,
        output_dir=gru_output_dir,
    )

    print("\nReentrenando LightGBM con los datos mas recientes...")
    lightgbm_result = lightgbm_model.train_and_save(
        data_path=data_path,
        output_dir=lightgbm_output_dir,
        config={
            "search_trials": int(lightgbm_search_trials),
        },
    )

    return {
        "gru": gru_result["summary"],
        "lightgbm": lightgbm_result["summary"],
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def calibrate_prediction_interval(
    residuals: np.ndarray,
    coverage: float,
) -> dict[str, float]:
    alpha = 1.0 - coverage
    lower_quantile = float(np.quantile(residuals, alpha / 2.0))
    upper_quantile = float(np.quantile(residuals, 1.0 - alpha / 2.0))
    residual_median = float(np.median(residuals))
    residual_mad = float(np.median(np.abs(residuals - residual_median)))
    return {
        "coverage_target": float(coverage),
        "alpha": float(alpha),
        "lower_residual_quantile_f": lower_quantile,
        "upper_residual_quantile_f": upper_quantile,
        "residual_median_f": residual_median,
        "residual_mad_f": residual_mad,
    }


def build_prediction_interval(
    y_pred: np.ndarray,
    prediction_band: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        y_pred + float(prediction_band["lower_residual_quantile_f"]),
        y_pred + float(prediction_band["upper_residual_quantile_f"]),
    )


def flatten_model_predictions(
    split_name: str,
    split_arrays: dict[str, object],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    forecast_horizons: list[int],
    prediction_column: str,
    actual_column: str,
) -> pd.DataFrame:
    y_true_2d = np.asarray(y_true, dtype=np.float32)
    y_pred_2d = np.asarray(y_pred, dtype=np.float32)
    if y_true_2d.ndim == 1:
        y_true_2d = y_true_2d.reshape(-1, 1)
    if y_pred_2d.ndim == 1:
        y_pred_2d = y_pred_2d.reshape(-1, 1)

    anchor_dates = pd.to_datetime(split_arrays["anchor_date"])
    target_dates = pd.to_datetime(np.asarray(split_arrays["target_dates"]).reshape(-1))
    sample_count = y_true_2d.shape[0]
    horizon_count = len(forecast_horizons)
    return pd.DataFrame(
        {
            "forecast_origin_datetime": np.repeat(anchor_dates, horizon_count),
            "observation_datetime": target_dates,
            "split": np.repeat(split_name, sample_count * horizon_count),
            "horizon_step": np.tile(np.asarray(forecast_horizons, dtype=np.int32), sample_count),
            actual_column: y_true_2d.reshape(-1),
            prediction_column: y_pred_2d.reshape(-1),
        }
    )


def build_multihorizon_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
) -> dict[str, object]:
    overall: dict[str, dict[str, float]] = {}
    by_horizon: dict[str, dict[str, dict[str, float]]] = {}

    for split_name, split_frame in frame.groupby("split"):
        overall[str(split_name)] = regression_metrics(
            split_frame["actual_temperature_f"].to_numpy(),
            split_frame[prediction_column].to_numpy(),
        )

    for horizon_step, horizon_frame in frame.groupby("horizon_step"):
        horizon_metrics: dict[str, dict[str, float]] = {}
        for split_name, split_frame in horizon_frame.groupby("split"):
            horizon_metrics[str(split_name)] = regression_metrics(
                split_frame["actual_temperature_f"].to_numpy(),
                split_frame[prediction_column].to_numpy(),
            )
        by_horizon[str(int(horizon_step))] = horizon_metrics

    return {"overall": overall, "by_horizon": by_horizon}


def build_historical_predictions(
    gru_bundle: dict[str, object],
    lightgbm_bundle: dict[str, object],
    gru_clean_df: pd.DataFrame,
    lightgbm_clean_df: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    gru_sequences = gru_model.build_sequence_buckets(gru_clean_df, gru_bundle["payload"])
    lightgbm_rows = lightgbm_model.build_buckets(
        lightgbm_clean_df,
        lightgbm_bundle["metadata"],
    )
    forecast_horizons = weather_common.normalize_forecast_horizons(
        gru_bundle["payload"].get("forecast_horizons"),
        fallback_horizon=int(gru_bundle["payload"]["horizon"]),
    )

    merged_frames: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        gru_y_true, gru_y_pred = gru_model.predict_split(
            gru_bundle,
            gru_sequences[split_name],
            batch_size=batch_size,
        )
        lgb_y_true, lgb_y_pred = lightgbm_model.predict_split(
            lightgbm_bundle,
            lightgbm_rows[split_name],
        )

        gru_frame = flatten_model_predictions(
            split_name=split_name,
            split_arrays=gru_sequences[split_name],
            y_true=gru_y_true,
            y_pred=gru_y_pred,
            forecast_horizons=forecast_horizons,
            prediction_column="predicted_temperature_f_gru",
            actual_column="actual_temperature_f",
        )
        lgb_frame = flatten_model_predictions(
            split_name=split_name,
            split_arrays=lightgbm_rows[split_name],
            y_true=lgb_y_true,
            y_pred=lgb_y_pred,
            forecast_horizons=forecast_horizons,
            prediction_column="predicted_temperature_f_lgb",
            actual_column="actual_temperature_f_lgb",
        )
        merged = gru_frame.merge(
            lgb_frame,
            on=["forecast_origin_datetime", "observation_datetime", "split", "horizon_step"],
            how="inner",
        )
        if not merged.empty:
            actual_gap = np.max(
                np.abs(
                    merged["actual_temperature_f"].to_numpy()
                    - merged["actual_temperature_f_lgb"].to_numpy()
                )
            )
            if actual_gap > 1e-5:
                raise ValueError(
                    "Las predicciones historicas de GRU y LightGBM no comparten "
                    "el mismo target real en las fechas cruzadas."
                )
            merged = merged.drop(columns=["actual_temperature_f_lgb"])
        merged_frames.append(merged)

    historical = pd.concat(merged_frames, ignore_index=True)
    if historical.empty:
        raise ValueError("No hubo interseccion de predicciones historicas entre GRU y LightGBM.")
    return historical


def fit_ensemble_weights(
    historical_predictions: pd.DataFrame,
    weight_grid_step: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    if not 0 < weight_grid_step <= 1:
        raise ValueError("--weight-grid-step debe estar entre 0 y 1.")

    validation = historical_predictions[historical_predictions["split"] == "val"].copy()
    if validation.empty:
        raise ValueError("No hay predicciones de validacion para ajustar los pesos del ensemble.")

    grid = np.arange(0.0, 1.0 + weight_grid_step / 2.0, weight_grid_step)
    search_rows: list[dict[str, float]] = []
    weight_rows: list[dict[str, object]] = []

    for horizon_step, horizon_frame in validation.groupby("horizon_step"):
        best_weight = 0.0
        best_metrics: dict[str, float] | None = None
        for gru_weight in grid:
            ensemble_pred = (
                gru_weight * horizon_frame["predicted_temperature_f_gru"].to_numpy()
                + (1.0 - gru_weight) * horizon_frame["predicted_temperature_f_lgb"].to_numpy()
            )
            metrics = regression_metrics(
                horizon_frame["actual_temperature_f"].to_numpy(),
                ensemble_pred,
            )
            row = {
                "horizon_step": int(horizon_step),
                "gru_weight": float(gru_weight),
                "lightgbm_weight": float(1.0 - gru_weight),
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_mape": metrics["mape"],
                "val_r2": metrics["r2"],
            }
            search_rows.append(row)
            if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
                best_metrics = metrics
                best_weight = float(gru_weight)

        if best_metrics is None:
            raise RuntimeError(
                f"No se pudo seleccionar un peso para el ensemble en horizon_step={horizon_step}."
            )
        weight_rows.append(
            {
                "horizon_step": int(horizon_step),
                "gru_weight": best_weight,
                "lightgbm_weight": float(1.0 - best_weight),
                "validation_metrics": best_metrics,
            }
        )

    search_df = pd.DataFrame(search_rows).sort_values(
        ["horizon_step", "val_mae", "val_rmse", "gru_weight"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    search_df.insert(0, "rank", np.arange(1, len(search_df) + 1))
    summary = {
        "selection_metric": "mae",
        "by_horizon": weight_rows,
    }
    return summary, search_df


def apply_ensemble_to_history(
    historical_predictions: pd.DataFrame,
    ensemble_weights: dict[str, object],
    interval_coverage: float,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    predictions = historical_predictions.copy()
    weight_by_horizon = {
        int(row["horizon_step"]): row for row in ensemble_weights["by_horizon"]
    }
    predictions["predicted_temperature_f_ensemble"] = np.nan

    band_rows: list[dict[str, float]] = []
    for horizon_step, horizon_frame in predictions.groupby("horizon_step"):
        horizon_weights = weight_by_horizon[int(horizon_step)]
        ensemble_pred = (
            float(horizon_weights["gru_weight"]) * horizon_frame["predicted_temperature_f_gru"]
            + float(horizon_weights["lightgbm_weight"])
            * horizon_frame["predicted_temperature_f_lgb"]
        )
        predictions.loc[horizon_frame.index, "predicted_temperature_f_ensemble"] = ensemble_pred

        validation = predictions[
            (predictions["split"] == "val") & (predictions["horizon_step"] == int(horizon_step))
        ]
        band = calibrate_prediction_interval(
            validation["actual_temperature_f"].to_numpy()
            - validation["predicted_temperature_f_ensemble"].to_numpy(),
            coverage=interval_coverage,
        )
        band["horizon_step"] = int(horizon_step)
        band_rows.append(band)

        for split_name, split_frame in predictions[
            predictions["horizon_step"] == int(horizon_step)
        ].groupby("split"):
            y_pred = split_frame["predicted_temperature_f_ensemble"].to_numpy()
            lower, upper = build_prediction_interval(y_pred, band)
            predictions.loc[split_frame.index, "lower_prediction_interval_f"] = lower
            predictions.loc[split_frame.index, "upper_prediction_interval_f"] = upper

    metrics_payload = build_multihorizon_metrics(
        predictions,
        prediction_column="predicted_temperature_f_ensemble",
    )
    for split_name, split_frame in predictions.groupby("split"):
        y_true = split_frame["actual_temperature_f"].to_numpy()
        lower = split_frame["lower_prediction_interval_f"].to_numpy()
        upper = split_frame["upper_prediction_interval_f"].to_numpy()
        metrics_payload["overall"][str(split_name)]["coverage_with_prediction_interval"] = float(
            np.mean((y_true >= lower) & (y_true <= upper))
        )
        metrics_payload["overall"][str(split_name)]["mean_prediction_interval_width_f"] = float(
            np.mean(upper - lower)
        )

    for horizon_step, horizon_frame in predictions.groupby("horizon_step"):
        for split_name, split_frame in horizon_frame.groupby("split"):
            y_true = split_frame["actual_temperature_f"].to_numpy()
            lower = split_frame["lower_prediction_interval_f"].to_numpy()
            upper = split_frame["upper_prediction_interval_f"].to_numpy()
            metrics_payload["by_horizon"][str(int(horizon_step))][str(split_name)][
                "coverage_with_prediction_interval"
            ] = float(np.mean((y_true >= lower) & (y_true <= upper)))
            metrics_payload["by_horizon"][str(int(horizon_step))][str(split_name)][
                "mean_prediction_interval_width_f"
            ] = float(np.mean(upper - lower))

    prediction_band = {
        "overall": calibrate_prediction_interval(
            predictions[predictions["split"] == "val"]["actual_temperature_f"].to_numpy()
            - predictions[predictions["split"] == "val"][
                "predicted_temperature_f_ensemble"
            ].to_numpy(),
            coverage=interval_coverage,
        ),
        "by_horizon": band_rows,
    }
    prediction_band["overall"]["empirical_coverage_reference_split"] = metrics_payload["overall"][
        "val"
    ]["coverage_with_prediction_interval"]
    prediction_band["overall"]["mean_interval_width_f"] = metrics_payload["overall"]["val"][
        "mean_prediction_interval_width_f"
    ]
    for band_row in prediction_band["by_horizon"]:
        step_metrics = metrics_payload["by_horizon"][str(int(band_row["horizon_step"]))]["val"]
        band_row["empirical_coverage_reference_split"] = step_metrics[
            "coverage_with_prediction_interval"
        ]
        band_row["mean_interval_width_f"] = step_metrics["mean_prediction_interval_width_f"]

    return predictions, metrics_payload, prediction_band


def estimate_future_timestamps(
    clean_df: pd.DataFrame,
    horizons: list[int],
    max_gap_minutes: int,
) -> tuple[list[dict[str, object]], float]:
    timestamps = pd.to_datetime(clean_df["observation_datetime"])
    diffs = timestamps.diff().dt.total_seconds().div(60.0)
    valid_diffs = diffs[(diffs > 0) & (diffs <= max_gap_minutes)]
    if valid_diffs.empty:
        step_minutes = 60.0
    else:
        step_minutes = float(valid_diffs.tail(min(24, len(valid_diffs))).median())
    last_timestamp = pd.Timestamp(timestamps.iloc[-1])
    forecast_rows: list[dict[str, object]] = []
    for horizon_step in horizons:
        forecast_rows.append(
            {
                "horizon_step": int(horizon_step),
                "forecast_target_timestamp": (
                    last_timestamp + timedelta(minutes=step_minutes * int(horizon_step))
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return forecast_rows, step_minutes


def main() -> None:
    args = parse_args()

    gpu_info = print_gpu_status()

    output_root = resolve_path(args.output_dir)
    run_started_at = datetime.now().astimezone()
    output_layout = build_output_layout(output_root, run_started_at)
    run_dir = output_layout["run_dir"]
    latest_dir = output_layout["latest_dir"]
    status_path = output_layout["status_path"]
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(args.data)
    gru_artifact_path = resolve_path(args.gru_artifact)
    gru_cleaned_path = resolve_path(args.gru_cleaned_data)
    lightgbm_model_path = resolve_path(args.lightgbm_model)
    lightgbm_metadata_path = resolve_path(args.lightgbm_metadata)
    lightgbm_cleaned_path = resolve_path(args.lightgbm_cleaned_data)

    raw_df, data_update_summary = weather_update.maybe_update_weather_csv(
        data_path=data_path,
        headless=args.headless,
        update_data=args.update_data,
    )
    current_clean_df = weather_common.clean_eda_data(raw_df)
    current_clean_df.to_csv(run_dir / "cleaned_weather_current.csv", index=False)

    retraining_summary: dict[str, object] | None = None
    if args.retrain_models:
        retraining_summary = retrain_models(
            data_path=data_path,
            gru_artifact_path=gru_artifact_path,
            lightgbm_model_path=lightgbm_model_path,
            lightgbm_search_trials=args.lightgbm_search_trials,
        )

    gru_bundle = gru_model.load_bundle(gru_artifact_path)
    lightgbm_bundle = lightgbm_model.load_bundle(
        lightgbm_model_path,
        lightgbm_metadata_path,
    )
    snapshot_model_artifacts(
        run_dir=run_dir,
        gru_artifact_path=gru_artifact_path,
        lightgbm_model_path=lightgbm_model_path,
    )

    compatibility_issues = validate_model_compatibility(gru_bundle, lightgbm_bundle)
    if compatibility_issues:
        gru_horizons = weather_common.normalize_forecast_horizons(
            gru_bundle["payload"].get("forecast_horizons"),
            fallback_horizon=int(gru_bundle["payload"]["horizon"]),
        )
        lightgbm_horizons = weather_common.normalize_forecast_horizons(
            lightgbm_bundle["metadata"].get("forecast_horizons"),
            fallback_horizon=int(lightgbm_bundle["metadata"]["horizon"]),
        )
        execution_manifest = build_execution_manifest(
            data_path=data_path,
            raw_df=raw_df,
            data_update_summary=data_update_summary,
            retraining_summary=retraining_summary,
            gru_bundle=gru_bundle,
            lightgbm_bundle=lightgbm_bundle,
            gru_artifact_path=gru_artifact_path,
            lightgbm_model_path=lightgbm_model_path,
            compatibility_issues=compatibility_issues,
            ensemble_metrics=None,
            gpu_info=gpu_info,
        )

        summary = {
            "data_update": data_update_summary,
            "retraining": retraining_summary,
            "gru_model": {
                "lookback": int(gru_bundle["payload"]["lookback"]),
                "horizon": int(gru_bundle["payload"]["horizon"]),
                "forecast_horizons": gru_horizons,
                "max_gap_minutes": int(gru_bundle["payload"]["max_gap_minutes"]),
                "hidden_size": int(gru_bundle["hidden_size"]),
                "num_layers": int(gru_bundle["num_layers"]),
            },
            "lightgbm_model": {
                "lookback": int(lightgbm_bundle["metadata"]["lookback"]),
                "horizon": int(lightgbm_bundle["metadata"]["horizon"]),
                "forecast_horizons": lightgbm_horizons,
                "max_gap_minutes": int(lightgbm_bundle["metadata"]["max_gap_minutes"]),
                "best_iteration": int(lightgbm_bundle["metadata"]["best_iteration"]),
                "best_iterations": lightgbm_bundle["metadata"].get("best_iterations"),
            },
            "compatibility_issues": compatibility_issues,
            "ensemble_weights": None,
            "ensemble_metrics": None,
            "prediction_band": None,
            "live_forecast": None,
        }
        status_payload = build_status_payload(
            run_id=str(output_layout["run_id"]),
            run_dir=run_dir,
            latest_dir=latest_dir,
            summary=summary,
            execution_manifest=execution_manifest,
            live_forecast=None,
        )
        write_json(run_dir / "execution_manifest.json", execution_manifest)
        write_json(run_dir / "ensemble_summary.json", summary)
        write_json(run_dir / "status.json", status_payload)
        refresh_latest_directory(run_dir, latest_dir)
        write_json(status_path, status_payload)
        print("No se puede construir el ensemble con los artefactos actuales:")
        for issue in compatibility_issues:
            print(f"- {issue}")
        print("\nResumen ensemble")
        print(json.dumps(summary, indent=2))
        return

    gru_clean_df = pd.read_csv(gru_cleaned_path, parse_dates=["observation_datetime"])
    lightgbm_clean_df = pd.read_csv(
        lightgbm_cleaned_path,
        parse_dates=["observation_datetime"],
    )
    historical_predictions = build_historical_predictions(
        gru_bundle=gru_bundle,
        lightgbm_bundle=lightgbm_bundle,
        gru_clean_df=gru_clean_df,
        lightgbm_clean_df=lightgbm_clean_df,
        batch_size=args.batch_size,
    )
    historical_predictions.to_csv(run_dir / "historical_component_predictions.csv", index=False)

    ensemble_weights, weight_search = fit_ensemble_weights(
        historical_predictions=historical_predictions,
        weight_grid_step=args.weight_grid_step,
    )
    weight_search.to_csv(run_dir / "ensemble_weight_search.csv", index=False)

    ensemble_predictions, ensemble_metrics, prediction_band = apply_ensemble_to_history(
        historical_predictions,
        ensemble_weights,
        interval_coverage=float(SHARED_DEFAULTS["interval_coverage"]),
    )
    ensemble_predictions.to_csv(run_dir / "historical_ensemble_predictions.csv", index=False)

    forecast_horizons = weather_common.normalize_forecast_horizons(
        gru_bundle["payload"].get("forecast_horizons"),
        fallback_horizon=int(gru_bundle["payload"]["horizon"]),
    )
    forecast_rows, estimated_step_minutes = estimate_future_timestamps(
        current_clean_df,
        horizons=forecast_horizons,
        max_gap_minutes=min(
            int(gru_bundle["payload"]["max_gap_minutes"]),
            int(lightgbm_bundle["metadata"]["max_gap_minutes"]),
        ),
    )

    gru_prediction = np.asarray(
        gru_model.build_live_prediction(current_clean_df, gru_bundle),
        dtype=np.float32,
    ).reshape(-1)
    lightgbm_prediction = np.asarray(
        lightgbm_model.build_live_prediction(
            current_clean_df,
            lightgbm_bundle,
        ),
        dtype=np.float32,
    ).reshape(-1)
    weight_by_horizon = {
        int(row["horizon_step"]): row for row in ensemble_weights["by_horizon"]
    }
    band_by_horizon = {
        int(row["horizon_step"]): row for row in prediction_band["by_horizon"]
    }
    live_forecast_rows: list[dict[str, object]] = []
    for idx, forecast_row in enumerate(forecast_rows):
        horizon_step = int(forecast_row["horizon_step"])
        weights = weight_by_horizon[horizon_step]
        ensemble_prediction = (
            float(weights["gru_weight"]) * float(gru_prediction[idx])
            + float(weights["lightgbm_weight"]) * float(lightgbm_prediction[idx])
        )
        lower_interval, upper_interval = build_prediction_interval(
            np.asarray([ensemble_prediction], dtype=np.float32),
            band_by_horizon[horizon_step],
        )
        live_forecast_rows.append(
            {
                "horizon_observations_ahead": horizon_step,
                "forecast_target_timestamp": forecast_row["forecast_target_timestamp"],
                "gru_weight": float(weights["gru_weight"]),
                "lightgbm_weight": float(weights["lightgbm_weight"]),
                "gru_prediction_f": float(gru_prediction[idx]),
                "lightgbm_prediction_f": float(lightgbm_prediction[idx]),
                "ensemble_prediction_f": float(ensemble_prediction),
                "lower_prediction_interval_f": float(lower_interval[0]),
                "upper_prediction_interval_f": float(upper_interval[0]),
            }
        )

    live_forecast: dict[str, object] | None = {
        "last_observation_datetime": str(current_clean_df.iloc[-1]["observation_datetime"]),
        "estimated_step_minutes": estimated_step_minutes,
        "forecast_horizons": forecast_horizons,
        "forecasts": live_forecast_rows,
    }
    write_json(run_dir / "latest_live_forecast.json", live_forecast)

    summary = {
        "data_update": data_update_summary,
        "retraining": retraining_summary,
        "gru_model": {
            "lookback": int(gru_bundle["payload"]["lookback"]),
            "horizon": int(gru_bundle["payload"]["horizon"]),
            "forecast_horizons": forecast_horizons,
            "max_gap_minutes": int(gru_bundle["payload"]["max_gap_minutes"]),
            "hidden_size": int(gru_bundle["hidden_size"]),
            "num_layers": int(gru_bundle["num_layers"]),
        },
        "lightgbm_model": {
            "lookback": int(lightgbm_bundle["metadata"]["lookback"]),
            "horizon": int(lightgbm_bundle["metadata"]["horizon"]),
            "forecast_horizons": forecast_horizons,
            "max_gap_minutes": int(lightgbm_bundle["metadata"]["max_gap_minutes"]),
            "best_iteration": int(lightgbm_bundle["metadata"]["best_iteration"]),
            "best_iterations": lightgbm_bundle["metadata"].get("best_iterations"),
        },
        "compatibility_issues": compatibility_issues,
        "ensemble_weights": ensemble_weights,
        "ensemble_metrics": ensemble_metrics,
        "prediction_band": prediction_band,
        "live_forecast": live_forecast,
    }

    execution_manifest = build_execution_manifest(
        data_path=data_path,
        raw_df=raw_df,
        data_update_summary=data_update_summary,
        retraining_summary=retraining_summary,
        gru_bundle=gru_bundle,
        lightgbm_bundle=lightgbm_bundle,
        gru_artifact_path=gru_artifact_path,
        lightgbm_model_path=lightgbm_model_path,
        compatibility_issues=compatibility_issues,
        ensemble_metrics=ensemble_metrics,
        gpu_info=gpu_info,
    )
    status_payload = build_status_payload(
        run_id=str(output_layout["run_id"]),
        run_dir=run_dir,
        latest_dir=latest_dir,
        summary=summary,
        execution_manifest=execution_manifest,
        live_forecast=live_forecast,
    )
    write_json(run_dir / "execution_manifest.json", execution_manifest)
    write_json(run_dir / "ensemble_summary.json", summary)
    write_json(run_dir / "status.json", status_payload)
    refresh_latest_directory(run_dir, latest_dir)
    write_json(status_path, status_payload)

    print("\nResumen ensemble")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
