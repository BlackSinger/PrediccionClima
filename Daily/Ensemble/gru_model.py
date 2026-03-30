from __future__ import annotations

from copy import deepcopy
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config_store
import weather_common


def get_default_train_config() -> dict[str, object]:
    shared = config_store.load_section("shared")
    train_defaults = config_store.load_section("gru", "train_defaults")
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


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class GRUDailyTemperatureRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        output = self.head(hidden[-1])
        return output.squeeze(-1)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            predictions.append(model(x_batch).cpu().numpy())
            targets.append(y_batch.numpy())

    return np.concatenate(predictions), np.concatenate(targets)


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    max_gap_minutes: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, object]]:
    n_rows = len(df)
    train_end = max(int(n_rows * train_ratio), lookback + horizon)
    val_end = int(n_rows * (train_ratio + val_ratio))
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n_rows - 1)

    features = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)

    feature_mean = features[:train_end].mean(axis=0)
    feature_std = features[:train_end].std(axis=0)
    feature_std[feature_std == 0] = 1.0

    target_mean = float(target[:train_end].mean())
    target_std = float(target[:train_end].std())
    if target_std == 0:
        target_std = 1.0

    features_scaled = (features - feature_mean) / feature_std
    target_scaled = (target - target_mean) / target_std
    run_lengths = weather_common.compute_run_lengths(
        df["observation_datetime"],
        max_gap_minutes,
    )

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

        buckets[split]["x"].append(features_scaled[input_start:input_end])
        buckets[split]["y"].append(target_scaled[target_idx])
        buckets[split]["date"].append(df.iloc[target_idx]["observation_datetime"])

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_values in buckets.items():
        if not split_values["x"]:
            raise ValueError(f"No hay secuencias disponibles para el split '{split_name}'.")
        arrays[split_name] = {
            "x": np.asarray(split_values["x"], dtype=np.float32),
            "y": np.asarray(split_values["y"], dtype=np.float32),
            "date": np.asarray(split_values["date"], dtype="datetime64[ns]"),
        }

    scalers = {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": np.asarray([target_mean], dtype=np.float32),
        "target_std": np.asarray([target_std], dtype=np.float32),
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
    return arrays, scalers, split_info


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = criterion(model(x_batch), y_batch)
            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / total_items


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[nn.Module, pd.DataFrame]:
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

        train_loss = total_loss / total_items
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping en epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history_rows)


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


def infer_gru_architecture(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    hidden_size = int(state_dict["gru.weight_ih_l0"].shape[0] // 3)
    num_layers = len(
        [key for key in state_dict if re.fullmatch(r"gru\.weight_ih_l\d+", key)]
    )
    return {"hidden_size": hidden_size, "num_layers": num_layers}


def load_bundle(gru_artifact_path: Path) -> dict[str, object]:
    payload = torch.load(gru_artifact_path, map_location="cpu", weights_only=False)
    architecture = infer_gru_architecture(payload["model_state_dict"])
    model = GRUDailyTemperatureRegressor(
        input_size=len(payload["feature_columns"]),
        hidden_size=architecture["hidden_size"],
        num_layers=architecture["num_layers"],
        dropout=0.0,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return {
        "payload": payload,
        "model": model,
        "hidden_size": architecture["hidden_size"],
        "num_layers": architecture["num_layers"],
    }


def build_sequence_buckets(
    clean_df: pd.DataFrame,
    payload: dict[str, object],
) -> dict[str, dict[str, np.ndarray]]:
    feature_cols = list(payload["feature_columns"])
    target_col = str(payload["target_column"])
    lookback = int(payload["lookback"])
    horizon = int(payload["horizon"])
    max_gap_minutes = int(payload["max_gap_minutes"])
    train_end_timestamp, val_end_timestamp = weather_common.get_split_cutoffs(
        payload["split_info"]
    )

    features = clean_df[feature_cols].to_numpy(dtype=np.float32)
    targets = clean_df[target_col].to_numpy(dtype=np.float32)
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    target_mean = float(payload["target_mean"])
    target_std = float(payload["target_std"])

    features_scaled = (features - feature_mean) / feature_std
    targets_scaled = (targets - target_mean) / target_std
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
        buckets[split_name]["x"].append(features_scaled[input_start:input_end])
        buckets[split_name]["y"].append(targets_scaled[target_idx])
        buckets[split_name]["date"].append(target_timestamp.to_datetime64())

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name, values in buckets.items():
        arrays[split_name] = {
            "x": np.asarray(values["x"], dtype=np.float32),
            "y": np.asarray(values["y"], dtype=np.float32),
            "date": np.asarray(values["date"], dtype="datetime64[ns]"),
        }
    return arrays


def predict_split(
    gru_bundle: dict[str, object],
    split_arrays: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        split_arrays["x"],
        split_arrays["y"],
        batch_size=batch_size,
        shuffle=False,
    )
    y_pred_scaled, y_true_scaled = predict(gru_bundle["model"], loader, torch.device("cpu"))
    payload = gru_bundle["payload"]
    y_pred = denormalize(
        y_pred_scaled,
        float(payload["target_mean"]),
        float(payload["target_std"]),
    )
    y_true = denormalize(
        y_true_scaled,
        float(payload["target_mean"]),
        float(payload["target_std"]),
    )
    return y_true, y_pred


def build_live_prediction(
    clean_df: pd.DataFrame,
    gru_bundle: dict[str, object],
) -> float:
    payload = gru_bundle["payload"]
    lookback = int(payload["lookback"])
    run_lengths = weather_common.compute_run_lengths(
        clean_df["observation_datetime"],
        int(payload["max_gap_minutes"]),
    )
    if int(run_lengths[-1]) < lookback:
        raise ValueError(
            "No hay suficientes observaciones consecutivas para inferir con la GRU."
        )

    feature_cols = list(payload["feature_columns"])
    window = clean_df[feature_cols].to_numpy(dtype=np.float32)[-lookback:]
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    scaled_window = (window - feature_mean) / feature_std
    x_tensor = torch.from_numpy(scaled_window).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = gru_bundle["model"](x_tensor).cpu().numpy().squeeze()
    return float(
        denormalize(
            np.asarray([pred_scaled], dtype=np.float32),
            float(payload["target_mean"]),
            float(payload["target_std"]),
        )[0]
    )


def train_and_save(
    data_path: Path,
    output_dir: Path,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    train_config = dict(DEFAULT_TRAIN_CONFIG)
    if config:
        train_config.update(config)

    set_seed(int(train_config["seed"]))
    torch.set_float32_matmul_precision("high")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(data_path)
    clean_df = weather_common.clean_eda_data(raw_df)
    clean_df.to_csv(output_dir / "cleaned_weather_for_gru.csv", index=False)

    sequence_data, scalers, split_info = build_sequences(
        df=clean_df,
        feature_cols=weather_common.MODEL_FEATURES,
        target_col=weather_common.TARGET_COLUMN,
        lookback=int(train_config["lookback"]),
        horizon=int(train_config["horizon"]),
        train_ratio=float(train_config["train_ratio"]),
        val_ratio=float(train_config["val_ratio"]),
        max_gap_minutes=int(train_config["max_gap_minutes"]),
    )

    train_loader = make_loader(
        sequence_data["train"]["x"],
        sequence_data["train"]["y"],
        int(train_config["batch_size"]),
        shuffle=True,
    )
    val_loader = make_loader(
        sequence_data["val"]["x"],
        sequence_data["val"]["y"],
        int(train_config["batch_size"]),
        shuffle=False,
    )
    test_loader = make_loader(
        sequence_data["test"]["x"],
        sequence_data["test"]["y"],
        int(train_config["batch_size"]),
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUDailyTemperatureRegressor(
        input_size=len(weather_common.MODEL_FEATURES),
        hidden_size=int(train_config["hidden_size"]),
        num_layers=int(train_config["num_layers"]),
        dropout=float(train_config["dropout"]),
    ).to(device)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(train_config["epochs"]),
        learning_rate=float(train_config["learning_rate"]),
        patience=int(train_config["patience"]),
    )

    target_mean = float(scalers["target_mean"][0])
    target_std = float(scalers["target_std"][0])
    metrics: dict[str, dict[str, float]] = {}
    prediction_frames: list[pd.DataFrame] = []
    prediction_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    split_predictions: dict[str, dict[str, np.ndarray]] = {}

    for split_name, loader in prediction_loaders.items():
        y_pred_scaled, y_true_scaled = predict(model, loader, device)
        y_pred = denormalize(y_pred_scaled, target_mean, target_std)
        y_true = denormalize(y_true_scaled, target_mean, target_std)
        metrics[split_name] = regression_metrics(y_true, y_pred)
        split_predictions[split_name] = {
            "date": sequence_data[split_name]["date"],
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

    artifact_payload = {
        "model_state_dict": model.state_dict(),
        "feature_columns": weather_common.MODEL_FEATURES,
        "target_column": weather_common.TARGET_COLUMN,
        "lookback": int(train_config["lookback"]),
        "horizon": int(train_config["horizon"]),
        "max_gap_minutes": int(train_config["max_gap_minutes"]),
        "feature_mean": scalers["feature_mean"],
        "feature_std": scalers["feature_std"],
        "target_mean": target_mean,
        "target_std": target_std,
        "split_info": split_info,
        "metrics": metrics,
        "prediction_band": prediction_band,
    }
    torch.save(artifact_payload, output_dir / "gru_daily_temperature.pt")

    summary = {
        "device": str(device),
        "rows_after_cleaning": int(len(clean_df)),
        "feature_count": len(weather_common.MODEL_FEATURES),
        "sequence_counts": {
            split: int(values["x"].shape[0]) for split, values in sequence_data.items()
        },
        "split_info": split_info,
        "metrics": metrics,
        "prediction_band": prediction_band,
        "training_config": {
            "lookback": int(train_config["lookback"]),
            "horizon": int(train_config["horizon"]),
            "hidden_size": int(train_config["hidden_size"]),
            "num_layers": int(train_config["num_layers"]),
            "dropout": float(train_config["dropout"]),
            "batch_size": int(train_config["batch_size"]),
            "epochs": int(train_config["epochs"]),
            "patience": int(train_config["patience"]),
            "learning_rate": float(train_config["learning_rate"]),
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return {
        "summary": summary,
        "artifact_path": output_dir / "gru_daily_temperature.pt",
        "cleaned_data_path": output_dir / "cleaned_weather_for_gru.csv",
    }
