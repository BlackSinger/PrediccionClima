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

# Enumeramos columnas numericas
EDA_NUMERIC_COLS = [
    "temp_max_f",
    "temp_avg_f",
    "temp_min_f",
    "dew_max_f",
    "dew_avg_f",
    "dew_min_f",
    "humidity_max",
    "humidity_avg",
    "humidity_min",
    "wind_max_mph",
    "wind_avg_mph",
    "wind_min_mph",
    "pressure_max_in",
    "pressure_avg_in",
    "pressure_min_in",
]

# Variables de entrada para el modelo
MODEL_FEATURES = [
    "temp_max_f",
    "temp_avg_f",
    "temp_min_f",
    "dew_max_f",
    "dew_avg_f",
    "dew_min_f",
    "humidity_max",
    "humidity_avg",
    "humidity_min",
    "wind_max_mph",
    "wind_avg_mph",
    "wind_min_mph",
    "pressure_max_in",
    "pressure_avg_in",
    "pressure_min_in",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "time_idx",
]

# Ordenamos las columnas para que el modelo aprenda las relaciones entre ellas
ORDERED_GROUPS = [
    ("temp_min_f", "temp_avg_f", "temp_max_f"),
    ("dew_min_f", "dew_avg_f", "dew_max_f"),
    ("humidity_min", "humidity_avg", "humidity_max"),
    ("wind_min_mph", "wind_avg_mph", "wind_max_mph"),
    ("pressure_min_in", "pressure_avg_in", "pressure_max_in"),
]

# Variable objetivo
TARGET_COLUMN = "temp_max_f"

# Funcion para parsear argumentos en la terminal y sea mas facil de configurar sin jurungar el codigo
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una GRU para predecir la temperatura maxima diaria."
    )
    parser.add_argument("--data", type=Path, default=Path("wunderground_ezeiza_2001_2026.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gru_temp_max"))
    parser.add_argument("--lookback", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=24217956)
    return parser.parse_args()

# Funcion para fijar la semilla y sea reproducible el resultado
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Funcion para limpiar datos (eliminar columnas innecesarias, convertir tipos de datos, etc) en el ipynb se analizo esto
def clean_eda_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "precip_total_in" in df.columns:
        df = df.drop(columns=["precip_total_in"])

    df["date"] = pd.to_datetime(df["date"])
    df[EDA_NUMERIC_COLS] = df[EDA_NUMERIC_COLS].astype(float)
    df = df.sort_values("date").reset_index(drop=True)

    mask = df["temp_max_f"] > 110
    df["temp_max_f"] = df["temp_max_f"].mask(mask).interpolate(limit_direction="both")

    mask = df["temp_min_f"] < 20
    df["temp_min_f"] = df["temp_min_f"].mask(mask).interpolate(limit_direction="both")

    mask = df["dew_max_f"] > 83
    df["dew_max_f"] = df["dew_max_f"].mask(mask).interpolate(limit_direction="both")

    mask = df["dew_min_f"] < 10
    df["dew_min_f"] = df["dew_min_f"].mask(mask).interpolate(limit_direction="both")

    mask = df["wind_max_mph"] > 75
    df["wind_max_mph"] = df["wind_max_mph"].mask(mask).interpolate(limit_direction="both")

    mask = df["wind_avg_mph"] > 30
    df["wind_avg_mph"] = df["wind_avg_mph"].mask(mask).interpolate(limit_direction="both")

    mask = (df["pressure_max_in"] < 29.2) | (df["pressure_max_in"] > 30.8)
    df["pressure_max_in"] = df["pressure_max_in"].mask(mask).interpolate(limit_direction="both")

    mask = (df["pressure_avg_in"] < 29.2) | (df["pressure_avg_in"] > 30.8)
    df["pressure_avg_in"] = df["pressure_avg_in"].mask(mask).interpolate(limit_direction="both")

    mask = (df["pressure_min_in"] < 29.0) | (df["pressure_min_in"] > 30.8)
    df["pressure_min_in"] = df["pressure_min_in"].mask(mask).interpolate(limit_direction="both")

    # Ordenamos las columnas para que el modelo aprenda las relaciones entre ellas, y tenga sentido, min, avg, max.
    for min_col, avg_col, max_col in ORDERED_GROUPS:
        ordered = np.sort(df[[min_col, avg_col, max_col]].to_numpy(), axis=1)
        df[min_col] = ordered[:, 0]
        df[avg_col] = ordered[:, 1]
        df[max_col] = ordered[:, 2]

    # Agregamos variables ciclicas para que el modelo entienda que el tiempo es ciclico, por ejemplo los meses del ano entienda que despues de diciembre viene enero(mapea la recta en un circulo)
    day_of_year = df["date"].dt.dayofyear.to_numpy()
    month = df["date"].dt.month.to_numpy()
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0) #mapeamos la recta en un circulo
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0) #mapeamos la recta en un circulo
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25) #mapeamos la recta en un circulo
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25) #mapeamos la recta en un circulo
    df["time_idx"] = np.linspace(0.0, 1.0, len(df), dtype=np.float32)
    return df

# Funcion para calcular la longitud de las secuencias, por si hay huecos, detecte y el modelo no salte un dia creyendo que es el siguiente
def compute_run_lengths(dates: pd.Series) -> np.ndarray:
    run_lengths = np.ones(len(dates), dtype=np.int32)
    for idx in range(1, len(dates)):
        if (dates.iloc[idx] - dates.iloc[idx - 1]).days == 1:
            run_lengths[idx] = run_lengths[idx - 1] + 1
    return run_lengths

# Funcion para construir secuencias, la finalidad de esta funcion es:
# Convertir el dataframe en un ejemplo listo para consumir por la GRU
# La funcion devuelve 3 cosas:
# 1. Un array de numpy con las secuencias de entrada
# 2. Medias y desviaciones estandar para escalar/desescalar
# 3. Informacion de donde termino train y val
def build_sequences(
    df: pd.DataFrame, # Dataframe
    feature_cols: list[str], # Columnas de entrada
    target_col: str, # Columna objetivo
    lookback: int, # Cantidad de dias a mirar hacia atras
    horizon: int, # Cantidad de dias a predecir
    train_ratio: float, # Porcentaje de datos para entrenamiento
    val_ratio: float, # Porcentaje de datos para validacion
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, object]]:
    n_rows = len(df)
    # Calculamos donde termina train y val
    train_end = max(int(n_rows * train_ratio), lookback + horizon)
    val_end = int(n_rows * (train_ratio + val_ratio))

    # Convertimos a numpy
    features = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)

    # Calculamos medias y desviaciones estandar para normalizar
    feature_mean = features[:train_end].mean(axis=0)
    feature_std = features[:train_end].std(axis=0)
    feature_std[feature_std == 0] = 1.0

    target_mean = target[:train_end].mean()
    target_std = target[:train_end].std()
    target_std = 1.0 if target_std == 0 else float(target_std)

    # Normalizamos los datos
    features_scaled = (features - feature_mean) / feature_std
    target_scaled = (target - target_mean) / target_std

    # Calculamos la longitud de las secuencias, para asegurarnos que realmente son 30 dias y no hay huecos
    run_lengths = compute_run_lengths(df["date"])
    # Esto crea contenedores vacios para guardar las secuencias
    buckets: dict[str, dict[str, list[object]]] = {
        "train": {"x": [], "y": [], "date": []},
        "val": {"x": [], "y": [], "date": []},
        "test": {"x": [], "y": [], "date": []},
    }

    for target_idx in range(lookback + horizon - 1, n_rows):
        # Si la longitud de la secuencia es menor a lookback + horizon, saltamos
        if run_lengths[target_idx] < lookback + horizon:
            continue

        # Calculamos el final y el inicio de la secuencia
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
        buckets[split]["date"].append(df.iloc[target_idx]["date"])

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
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": np.asarray([target_mean], dtype=np.float32),
        "target_std": np.asarray([target_std], dtype=np.float32),
    }

    split_info = {
        "train_end_date": df.iloc[train_end - 1]["date"].strftime("%Y-%m-%d"),
        "val_end_date": df.iloc[val_end - 1]["date"].strftime("%Y-%m-%d"),
        "total_rows": n_rows,
    }
    return arrays, scalers, split_info

# Funcion para crear loaders
def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Modelo GRU
class GRUTemperatureRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
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

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        output = self.head(hidden[-1])
        return output.squeeze(-1)

# Funcion para evaluar la perdida
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
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

# Funcion para entrenar el modelo
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

# Funcion para predecir
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

# Funcion para desnormalizar
def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean

# Funcion para calcular metricas de regresion
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred))) # Calcula el error absoluto medio
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred)))) # Calcula el error cuadratico medio, penaliza mas los errores grandes
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) # Calcula el error porcentual absoluto medio
    ss_res = float(np.sum(np.square(y_true - y_pred))) # Calcula la suma de los cuadrados de los residuos
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true)))) # Calcula la suma de los cuadrados totales
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0) # Calcula el coeficiente de determinacion (Explica la variabilidad de los datos) R=0 (igual que la media), R=1 Perfecto, R < 0, peor que colocar la media
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

# Funcion para construir dataframe de predicciones
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
            "date": pd.to_datetime(dates),
            "split": split_name,
            "actual_temp_max_f": y_true,
            "predicted_temp_max_f": y_pred,
            "lower_mae": lower_mae,
            "upper_mae": upper_mae,
        }
    )

# Funcion para guardar graficos
def save_plots(history: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Evolucion de la perdida")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE escalado")
    axes[0].legend()

    test_predictions = predictions[predictions["split"] == "test"].sort_values("date")
    axes[1].plot(
        test_predictions["date"],
        test_predictions["actual_temp_max_f"],
        label="Real",
        linewidth=1.3,
    )
    axes[1].plot(
        test_predictions["date"],
        test_predictions["predicted_temp_max_f"],
        label="Prediccion",
        linewidth=1.3,
    )
    axes[1].fill_between(
        test_predictions["date"],
        test_predictions["lower_mae"],
        test_predictions["upper_mae"],
        label="Banda +/- MAE",
        alpha=0.2,
    )
    axes[1].set_title("Prediccion de temp_max_f en test")
    axes[1].set_xlabel("Fecha")
    axes[1].set_ylabel("Temperatura maxima (F)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_diagnostics.png", dpi=160)
    plt.close(fig)

# Funcion principal
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.data)
    clean_df = clean_eda_data(raw_df)
    clean_df.to_csv(output_dir / "cleaned_weather_for_gru.csv", index=False)

    sequence_data, scalers, split_info = build_sequences(
        df=clean_df,
        feature_cols=MODEL_FEATURES,
        target_col=TARGET_COLUMN,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_loader = make_loader(
        sequence_data["train"]["x"], sequence_data["train"]["y"], args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        sequence_data["val"]["x"], sequence_data["val"]["y"], args.batch_size, shuffle=False
    )
    test_loader = make_loader(
        sequence_data["test"]["x"], sequence_data["test"]["y"], args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUTemperatureRegressor(
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
        "lookback": args.lookback,
        "horizon": args.horizon,
        "feature_mean": scalers["feature_mean"],
        "feature_std": scalers["feature_std"],
        "target_mean": target_mean,
        "target_std": target_std,
        "split_info": split_info,
        "metrics": metrics,
        "prediction_band": prediction_band,
    }
    torch.save(artifact_payload, output_dir / "gru_temp_max.pt")

    summary = {
        "device": str(device),
        "rows_after_cleaning": int(len(clean_df)),
        "feature_count": len(MODEL_FEATURES),
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
