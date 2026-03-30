from __future__ import annotations

import numpy as np
import pandas as pd

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


def default_forecast_horizons(max_horizon: int) -> list[int]:
    if int(max_horizon) < 1:
        raise ValueError("max_horizon debe ser >= 1.")
    return list(range(1, int(max_horizon) + 1))


def normalize_forecast_horizons(
    forecast_horizons: object | None,
    fallback_horizon: int,
) -> list[int]:
    if forecast_horizons is None:
        return [int(fallback_horizon)]

    if not isinstance(forecast_horizons, list) or not forecast_horizons:
        raise ValueError("forecast_horizons debe ser una lista no vacia.")

    normalized = sorted({int(step) for step in forecast_horizons})
    if normalized[0] < 1:
        raise ValueError("forecast_horizons solo admite pasos >= 1.")
    return normalized


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
    return df[ordered_cols]


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


def get_split_cutoffs(split_info: dict[str, object]) -> tuple[pd.Timestamp, pd.Timestamp]:
    train_end = pd.Timestamp(split_info["train_end_timestamp"])
    val_end = pd.Timestamp(split_info["val_end_timestamp"])
    return train_end, val_end


def assign_split(
    target_timestamp: pd.Timestamp,
    train_end_timestamp: pd.Timestamp,
    val_end_timestamp: pd.Timestamp,
) -> str:
    if target_timestamp <= train_end_timestamp:
        return "train"
    if target_timestamp <= val_end_timestamp:
        return "val"
    return "test"
