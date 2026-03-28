from argparse import ArgumentParser, BooleanOptionalAction
from datetime import date, datetime, timedelta
from pathlib import Path
import re

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.wunderground.com/history/daily/ar/ezeiza/SAEZ/date"
DEFAULT_START_DATE = date(2014, 1, 1)
TODAY = date.today()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / f"wunderground_ezeiza_daily_{DEFAULT_START_DATE.year}_{TODAY.year}.csv"
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


def parse_args():
    parser = ArgumentParser(
        description="Extrae observaciones diarias de Weather Underground para Ezeiza."
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
        help="Fecha inicial en formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=TODAY.isoformat(),
        help="Fecha final en formato YYYY-MM-DD.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Ruta del CSV de salida.",
    )
    parser.add_argument(
        "--headless",
        action=BooleanOptionalAction,
        default=True,
        help="Ejecuta Chromium en modo headless.",
    )
    return parser.parse_args()


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_urls(start_date, end_date):
    current_date = start_date
    urls = []

    while current_date <= end_date:
        urls.append(
            f"{BASE_URL}/{current_date.year}-{current_date.month}-{current_date.day}"
        )
        current_date += timedelta(days=1)

    return urls


def clean_text(value):
    return value.replace("\xa0", " ").strip()


def parse_numeric(value):
    text = clean_text(value)
    if not text or text in {"--", "N/A"}:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return match.group(0) if match else None


def extract_day_rows(page, item):
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
        [clean_text(value) for value in rows.nth(i).locator("th, td").all_inner_texts()]
        for i in range(rows.count())
    ]
    non_empty_rows = [row for row in raw_rows if row]

    if not non_empty_rows:
        print(f"No se encontraron filas en la tabla de {item}")
        return []

    header = non_empty_rows[0]
    if header != EXPECTED_HEADER:
        print(f"Estructura inesperada en {item}: encabezado {header}")
        return []

    extracted_rows = []
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


def main():
    args = parse_args()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    if start_date > end_date:
        raise ValueError("La fecha inicial no puede ser mayor que la fecha final.")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    urls = build_urls(start_date, end_date)
    all_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()

        for item in urls:
            print(f"Extrayendo {item}")
            all_rows.extend(extract_day_rows(page, item))

        browser.close()

    df_final = pd.DataFrame(all_rows, columns=COLUMNS)

    if not df_final.empty:
        for column in NUMERIC_COLUMNS:
            df_final[column] = pd.to_numeric(df_final[column], errors="coerce")

        df_final["date"] = pd.to_datetime(df_final["date"])
        df_final = df_final.sort_values(["date", "observation_index"]).reset_index(
            drop=True
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"Filas extraidas: {len(df_final)}")
    print(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    main()
