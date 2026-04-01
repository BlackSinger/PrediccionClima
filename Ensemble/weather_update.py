from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

import pandas as pd

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:
    PlaywrightTimeoutError = None
    sync_playwright = None

BASE_URL = "https://www.wunderground.com/history/daily/ar/ezeiza/SAEZ/date"
NAVIGATION_TIMEOUT_MS = 60000
TABLE_TIMEOUT_MS = 60000

EXPECTED_HEADER = [
    "Time",
    "Temperature",
    "Dew Point",
    "Humidity",
    "Wind",
    "Wind Speed",
    "Wind Gust",
    "Pressure",
    "Precip.",
    "Condition",
]

COLUMNS = [
    "date",
    "year",
    "month",
    "day",
    "observation_index",
    "observation_time",
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_direction",
    "wind_speed_mph",
    "wind_gust_mph",
    "pressure_in",
    "precip_in",
    "condition",
    "source_url",
]

NUMERIC_COLUMNS = [
    "year",
    "month",
    "day",
    "observation_index",
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_speed_mph",
    "wind_gust_mph",
    "pressure_in",
    "precip_in",
]


def ensure_playwright() -> None:
    if sync_playwright is None or PlaywrightTimeoutError is None:
        raise SystemExit(
            "No se encontro el paquete 'playwright'. Instala las dependencias "
            "necesarias para poder actualizar los datos desde Weather Underground."
        )


def build_urls(start_date: date, end_date: date) -> list[str]:
    current_date = start_date
    urls: list[str] = []
    while current_date <= end_date:
        urls.append(
            f"{BASE_URL}/{current_date.year}-{current_date.month}-{current_date.day}"
        )
        current_date += timedelta(days=1)
    return urls


def clean_text(value: str) -> str:
    return value.replace("\xa0", " ").strip()


def parse_numeric(value: str) -> str | None:
    text = clean_text(value)
    if not text or text in {"--", "N/A"}:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return match.group(0) if match else None


def extract_day_rows(page: Any, item: str) -> list[dict[str, object]]:
    if PlaywrightTimeoutError is None:
        ensure_playwright()

    year, month, day = (int(part) for part in item.rsplit("/", 1)[-1].split("-"))
    target_date = date(year, month, day)

    try:
        page.goto(item, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        table = page.locator(".observation-table")
        table.wait_for(state="visible", timeout=TABLE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        print(f"Timeout al cargar o encontrar la tabla en {item}")
        return []

    rows = table.locator("tr")
    raw_rows = [
        [clean_text(value) for value in rows.nth(idx).locator("th, td").all_inner_texts()]
        for idx in range(rows.count())
    ]
    non_empty_rows = [row for row in raw_rows if row]

    if not non_empty_rows:
        print(f"No se encontraron filas en la tabla de {item}")
        return []

    header = non_empty_rows[0]
    if header != EXPECTED_HEADER:
        print(f"Estructura inesperada en {item}: encabezado {header}")
        return []

    extracted_rows: list[dict[str, object]] = []
    for observation_index, values in enumerate(non_empty_rows[1:], start=1):
        if len(values) != len(EXPECTED_HEADER):
            print(f"Fila inesperada en {item}: {values}")
            continue

        extracted_rows.append(
            {
                "date": target_date,
                "year": target_date.year,
                "month": target_date.month,
                "day": target_date.day,
                "observation_index": observation_index,
                "observation_time": values[0],
                "temperature_f": parse_numeric(values[1]),
                "dew_point_f": parse_numeric(values[2]),
                "humidity_pct": parse_numeric(values[3]),
                "wind_direction": values[4],
                "wind_speed_mph": parse_numeric(values[5]),
                "wind_gust_mph": parse_numeric(values[6]),
                "pressure_in": parse_numeric(values[7]),
                "precip_in": parse_numeric(values[8]),
                "condition": values[9],
                "source_url": item,
            }
        )

    if not extracted_rows:
        print(f"No se encontraron observaciones validas en {item}")

    return extracted_rows


def combine_observation_datetime(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["date"], errors="coerce")
    times = df["observation_time"].fillna("").astype(str).str.strip()
    return pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + times,
        format="%Y-%m-%d %I:%M %p",
        errors="coerce",
    )


def sort_raw_observations(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working["observation_datetime"] = combine_observation_datetime(working)
    working = working.dropna(subset=["date", "observation_datetime"])
    working = working.sort_values(
        ["observation_datetime", "observation_index"],
        kind="mergesort",
    )
    working = working.drop_duplicates(
        subset=["observation_datetime"],
        keep="last",
    )
    return working.reset_index(drop=True)


def get_latest_observation_info(raw_df: pd.DataFrame) -> dict[str, object]:
    sorted_df = sort_raw_observations(raw_df)
    if sorted_df.empty:
        raise ValueError("El CSV de datos meteorologicos esta vacio.")

    last_row = sorted_df.iloc[-1]
    return {
        "row": last_row,
        "observation_datetime": pd.Timestamp(last_row["observation_datetime"]),
        "observation_date": pd.Timestamp(last_row["date"]).date(),
        "sorted_df": sorted_df.drop(columns=["observation_datetime"]),
    }


def download_missing_rows(
    start_date: date,
    end_date: date,
    headless: bool,
) -> pd.DataFrame:
    if start_date > end_date:
        return pd.DataFrame(columns=COLUMNS)

    ensure_playwright()
    urls = build_urls(start_date, end_date)
    all_rows: list[dict[str, object]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            for item in urls:
                print(f"Extrayendo {item}")
                all_rows.extend(extract_day_rows(page, item))
        finally:
            browser.close()

    new_rows = pd.DataFrame(all_rows, columns=COLUMNS)
    if new_rows.empty:
        return new_rows

    for column in NUMERIC_COLUMNS:
        new_rows[column] = pd.to_numeric(new_rows[column], errors="coerce")
    new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce")
    return new_rows


def maybe_update_weather_csv(
    data_path,
    headless: bool,
    update_data: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = pd.read_csv(data_path)
    latest_info = get_latest_observation_info(raw_df)

    last_dt = latest_info["observation_datetime"]
    last_date = latest_info["observation_date"]
    today = date.today()
    print(
        "Ultima observacion actual | "
        f"fecha={last_dt.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"temperatura_f={latest_info['row']['temperature_f']}"
    )

    updated = False
    downloaded_rows = 0
    if update_data:
        fetch_start = last_date
        print(
            f"Datos desactualizados. Descargando desde {fetch_start.isoformat()} "
            f"hasta {today.isoformat()}."
        )
        new_rows = download_missing_rows(fetch_start, today, headless=headless)
        downloaded_rows = int(len(new_rows))
        if not new_rows.empty:
            combined = pd.concat([raw_df, new_rows], ignore_index=True)
            combined = sort_raw_observations(combined)
            combined.to_csv(data_path, index=False)
            raw_df = combined
            latest_info = get_latest_observation_info(raw_df)
            updated = True
        else:
            print("No se descargaron filas nuevas.")
    else:
        print("No hace falta actualizar el CSV meteorologico.")

    summary = {
        "updated": updated,
        "downloaded_rows": downloaded_rows,
        "last_observation_datetime": latest_info["observation_datetime"].strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "last_observation_date": latest_info["observation_date"].isoformat(),
    }
    return latest_info["sorted_df"], summary
