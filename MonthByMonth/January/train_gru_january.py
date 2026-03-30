from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_MONTH = 1
TARGET_MONTH_OUTPUT_NAME = "january"
TARGET_MONTH_DISPLAY_NAME = "enero"

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
    "day_of_year_sin",
    "day_of_year_cos",
    "time_idx",
]

TARGET_COLUMN = "temperature_f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena una GRU con observaciones intradiarias de enero para predecir "
            "temperature_f."
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
        default=SCRIPT_DIR / "artifacts/gru_january_temperature",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=96,
        help="Cantidad de observaciones previas usadas como entrada.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Cantidad de observaciones hacia adelante a predecir.",
    )
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Proporcion de bloques enero-anio usados para entrenamiento.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proporcion de bloques enero-anio usados para validacion.",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=120,
        help="Gap maximo permitido entre observaciones consecutivas para una secuencia valida.",
    )
    parser.add_argument("--seed", type=int, default=24217956)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    df = df[df["observation_datetime"].dt.month == TARGET_MONTH].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No se encontraron observaciones para el mes objetivo ({TARGET_MONTH_DISPLAY_NAME})."
        )

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

    df["period_id"] = df["observation_datetime"].dt.to_period("M").astype(str)
    df["time_idx"] = np.arange(len(df), dtype=np.float32)
    df["hour"] = df["observation_datetime"].dt.hour.astype(np.float32)
    df["minute"] = df["observation_datetime"].dt.minute.astype(np.float32)

    direction_to_degrees = {
        direction: idx * 22.5 for idx, direction in enumerate(DIRECTION_LABELS)
    }
    degrees = df["wind_direction"].map(direction_to_degrees)
    radians = np.deg2rad(degrees)
    df["wind_dir_missing"] = degrees.isna().astype(np.float32)
    df["wind_dir_sin"] = np.sin(radians).fillna(0.0).astype(np.float32)
    df["wind_dir_cos"] = np.cos(radians).fillna(0.0).astype(np.float32)

    df = add_condition_flags(df)

    apply_range_interpolation(df, "temperature_f", lower=30.0, upper=110.0)
    apply_range_interpolation(df, "dew_point_f", lower=10.0, upper=83.0)
    apply_range_interpolation(df, "humidity_pct", lower=2.0, upper=100.0)
    # En el notebook aparece < 60, pero eso borraria casi toda la serie. Usamos > 60.
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

    df["hour_sin"] = np.sin(2 * np.pi * hour_fraction / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour_fraction / 24.0).astype(np.float32)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0).astype(np.float32)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0).astype(np.float32)
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

    first_cols = ["period_id", "time_idx", "hour", "minute", "observation_datetime"]
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


def split_period_ids(
    period_ids: pd.Series,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[str]]:
    unique_periods = pd.Index(period_ids.drop_duplicates()).astype(str).tolist()
    period_count = len(unique_periods)
    if period_count < 3:
        raise ValueError(
            "Se necesitan al menos 3 bloques enero-anio para separar train, val y test."
        )

    train_end = max(int(period_count * train_ratio), 1)
    train_end = min(train_end, period_count - 2)

    val_end = int(period_count * (train_ratio + val_ratio))
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, period_count - 1)

    splits = {
        "train": unique_periods[:train_end],
        "val": unique_periods[train_end:val_end],
        "test": unique_periods[val_end:],
    }
    if not splits["val"] or not splits["test"]:
        raise ValueError(
            "Los ratios actuales dejan algun split sin bloques enero-anio disponibles."
        )
    return splits


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
    period_splits = split_period_ids(df["period_id"], train_ratio, val_ratio)
    period_to_split = {
        period_id: split_name
        for split_name, period_list in period_splits.items()
        for period_id in period_list
    }

    features = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)
    periods = df["period_id"].to_numpy()

    train_mask = df["period_id"].isin(period_splits["train"]).to_numpy()
    feature_mean = features[train_mask].mean(axis=0)
    feature_std = features[train_mask].std(axis=0)
    feature_std[feature_std == 0] = 1.0

    target_mean = float(target[train_mask].mean())
    target_std = float(target[train_mask].std())
    if target_std == 0:
        target_std = 1.0

    features_scaled = (features - feature_mean) / feature_std
    target_scaled = (target - target_mean) / target_std

    run_lengths = compute_run_lengths(df["observation_datetime"], max_gap_minutes)

    buckets: dict[str, dict[str, list[object]]] = {
        "train": {"x": [], "y": [], "date": []},
        "val": {"x": [], "y": [], "date": []},
        "test": {"x": [], "y": [], "date": []},
    }

    required_run = lookback + horizon
    for target_idx in range(required_run - 1, len(df)):
        if run_lengths[target_idx] < required_run:
            continue

        input_end = target_idx - horizon + 1
        input_start = input_end - lookback
        if input_start < 0:
            continue

        # Cada enero-anio queda entero en un split para evitar mezclar la misma ventana estacional.
        if periods[input_start] != periods[target_idx]:
            continue

        split = period_to_split[periods[target_idx]]
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
        "month_filter": TARGET_MONTH_DISPLAY_NAME,
        "available_periods": pd.Index(df["period_id"].drop_duplicates()).astype(str).tolist(),
        "train_periods": period_splits["train"],
        "val_periods": period_splits["val"],
        "test_periods": period_splits["test"],
        "total_rows": int(len(df)),
        "max_gap_minutes": max_gap_minutes,
    }
    return arrays, scalers, split_info


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class GRUJanuaryTemperatureRegressor(nn.Module):
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
    criterion = nn.MSELoss()
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


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def build_predictions_frame(
    split_name: str,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_mae: np.ndarray,
    upper_mae: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_datetime": pd.to_datetime(dates),
            "split": split_name,
            "actual_temperature_f": y_true,
            "predicted_temperature_f": y_pred,
            "lower_mae": lower_mae,
            "upper_mae": upper_mae,
        }
    )


def save_plots(history: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Evolucion de la perdida")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE escalado")
    axes[0].legend()

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
    axes[1].fill_between(
        test_predictions["observation_datetime"],
        test_predictions["lower_mae"],
        test_predictions["upper_mae"],
        label="Banda +/- MAE",
        alpha=0.2,
    )
    axes[1].set_title("Prediccion de temperature_f en test para enero")
    axes[1].set_xlabel("Fecha y hora")
    axes[1].set_ylabel("Temperatura (F)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_diagnostics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data
    if not data_path.is_absolute():
        data_path = SCRIPT_DIR / data_path

    raw_df = pd.read_csv(data_path)
    clean_df = clean_eda_data(raw_df)
    clean_df.to_csv(output_dir / "cleaned_weather_for_gru_january.csv", index=False)

    sequence_data, scalers, split_info = build_sequences(
        df=clean_df,
        feature_cols=MODEL_FEATURES,
        target_col=TARGET_COLUMN,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_gap_minutes=args.max_gap_minutes,
    )

    train_loader = make_loader(
        sequence_data["train"]["x"],
        sequence_data["train"]["y"],
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        sequence_data["val"]["x"],
        sequence_data["val"]["y"],
        args.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        sequence_data["test"]["x"],
        sequence_data["test"]["y"],
        args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUJanuaryTemperatureRegressor(
        input_size=len(MODEL_FEATURES),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
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

    band_half_width = metrics["test"]["mae"]
    prediction_band = {
        "method": "plus_minus_mae",
        "reference_split": "test",
        "mae_f": band_half_width,
        "half_width_f": band_half_width,
    }

    for split_name, values in split_predictions.items():
        lower_mae = values["y_pred"] - band_half_width
        upper_mae = values["y_pred"] + band_half_width
        abs_error = np.abs(values["y_true"] - values["y_pred"])
        metrics[split_name]["coverage_with_mae_band"] = float(
            np.mean(abs_error <= band_half_width)
        )
        prediction_frames.append(
            build_predictions_frame(
                split_name,
                values["date"],
                values["y_true"],
                values["y_pred"],
                lower_mae,
                upper_mae,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    history.to_csv(output_dir / "history.csv", index=False)
    save_plots(history, predictions, output_dir)

    artifact_payload = {
        "model_state_dict": model.state_dict(),
        "feature_columns": MODEL_FEATURES,
        "target_column": TARGET_COLUMN,
        "target_month": TARGET_MONTH,
        "target_month_label": TARGET_MONTH_OUTPUT_NAME,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "max_gap_minutes": args.max_gap_minutes,
        "feature_mean": scalers["feature_mean"],
        "feature_std": scalers["feature_std"],
        "target_mean": target_mean,
        "target_std": target_std,
        "split_info": split_info,
        "metrics": metrics,
        "prediction_band": prediction_band,
    }
    torch.save(artifact_payload, output_dir / "gru_january_temperature.pt")

    summary = {
        "device": str(device),
        "month_filter": TARGET_MONTH_DISPLAY_NAME,
        "rows_after_cleaning": int(len(clean_df)),
        "feature_count": len(MODEL_FEATURES),
        "available_periods": split_info["available_periods"],
        "sequence_counts": {split: int(values["x"].shape[0]) for split, values in sequence_data.items()},
        "split_info": split_info,
        "metrics": metrics,
        "prediction_band": prediction_band,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\nResumen final")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
