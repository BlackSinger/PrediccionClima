from __future__ import annotations

import argparse
import json
from zoneinfo import ZoneInfo
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT_PATH = Path(__file__).with_name("temperature_log.json")
DEFAULT_OUTPUT_IMAGE = Path(__file__).with_name("temperature_hourly_max.png")
DEFAULT_OUTPUT_CSV = Path(__file__).with_name("temperature_hourly_max.csv")
BUENOS_AIRES_TIMEZONE = ZoneInfo("America/Argentina/Buenos_Aires")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agrupa las lecturas de temperatura por hora, calcula la maxima "
            "de cada franja y genera un grafico."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Archivo JSON de entrada. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE,
        help=f"Ruta del grafico PNG de salida. Default: {DEFAULT_OUTPUT_IMAGE}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Ruta del CSV agregado de salida. Default: {DEFAULT_OUTPUT_CSV}",
    )
    return parser.parse_args()


def load_readings(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("El JSON de entrada debe contener una lista de lecturas.")
    if not data:
        raise ValueError("El JSON de entrada no contiene lecturas.")

    dataframe = pd.DataFrame(data)
    required_columns = {"timestamp_buenos_aires", "temperature_f"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas en el JSON: {missing}")

    dataframe = dataframe.loc[:, ["timestamp_buenos_aires", "temperature_f"]].copy()
    dataframe["timestamp_buenos_aires"] = pd.to_datetime(
        dataframe["timestamp_buenos_aires"],
        errors="raise",
        utc=True,
    ).dt.tz_convert(BUENOS_AIRES_TIMEZONE)
    dataframe["temperature_f"] = pd.to_numeric(
        dataframe["temperature_f"],
        errors="raise",
    )
    dataframe = dataframe.sort_values("timestamp_buenos_aires").reset_index(drop=True)
    return dataframe


def build_hourly_max(dataframe: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        dataframe.assign(
            interval_start=dataframe["timestamp_buenos_aires"].dt.floor("1h")
        )
        .groupby("interval_start", as_index=False)["temperature_f"]
        .max()
        .rename(columns={"temperature_f": "max_temperature_f"})
        .sort_values("interval_start")
        .reset_index(drop=True)
    )
    return aggregated


def save_aggregated_csv(aggregated: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export_frame = pd.DataFrame(
        {
            "hora_buenos_aires": aggregated["interval_start"].dt.strftime(
                "%Y-%m-%d %H:%M"
            ),
            "temperatura_max_f": aggregated["max_temperature_f"],
        }
    )
    export_frame.to_csv(output_csv, index=False, encoding="utf-8")


def get_time_label_format(aggregated: pd.DataFrame) -> str:
    first_timestamp = aggregated["interval_start"].iloc[0]
    last_timestamp = aggregated["interval_start"].iloc[-1]
    if first_timestamp.date() == last_timestamp.date():
        return "%H:%M"
    return "%d %H:%M"


def save_plot(aggregated: pd.DataFrame, output_image: Path) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_timestamps = aggregated["interval_start"].dt.tz_localize(None)

    ax.plot(
        plot_timestamps,
        aggregated["max_temperature_f"],
        color="#1b6ca8",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
    )
    ax.fill_between(
        plot_timestamps,
        aggregated["max_temperature_f"],
        color="#1b6ca8",
        alpha=0.12,
    )

    first_timestamp = aggregated["interval_start"].iloc[0].tz_localize(None)
    last_timestamp = aggregated["interval_start"].iloc[-1].tz_localize(None)
    title = "Temperatura maxima por hora"
    subtitle = (
        f"Periodo: {first_timestamp:%Y-%m-%d %H:%M} a "
        f"{last_timestamp:%Y-%m-%d %H:%M} (hora de Buenos Aires)"
    )

    ax.set_title(f"{title}\n{subtitle}", pad=16)
    ax.set_xlabel("Hora del dia")
    ax.set_ylabel("Temperatura (F)")
    ax.set_xlim(first_timestamp, last_timestamp)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.DateFormatter(get_time_label_format(aggregated))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataframe = load_readings(args.input)
    aggregated = build_hourly_max(dataframe)
    save_aggregated_csv(aggregated, args.output_csv)
    save_plot(aggregated, args.output_image)

    print(f"Lecturas procesadas: {len(dataframe)}")
    print(f"Franjas horarias: {len(aggregated)}")
    print(f"CSV agregado: {args.output_csv}")
    print(f"Grafico generado: {args.output_image}")


if __name__ == "__main__":
    main()
