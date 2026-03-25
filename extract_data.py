from datetime import datetime

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.wunderground.com/history/monthly/ar/ezeiza/SAEZ/date"
OUTPUT_CSV = "wunderground_ezeiza_2001_2026.csv"
NAVIGATION_TIMEOUT_MS = 60000
TABLE_TIMEOUT_MS = 60000

TODAY = datetime.today()
URLS = [
    f"{BASE_URL}/{year}-{month:02d}-01"
    for year in range(2001, TODAY.year + 1)
    for month in range(1, 13)
    if (year, month) <= (TODAY.year, TODAY.month)
]

COLUMNS = [
    "day",
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
    "precip_total_in",
]

NUMERIC_COLUMNS = [
    "day",
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
    "precip_total_in",
]


def extract_month_rows(page, item):
    month_start = datetime.strptime(item.rsplit("/", 1)[-1], "%Y-%m-%d")

    try:
        page.goto(item, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        table = page.locator(".observation-table")
        table.wait_for(state="visible", timeout=TABLE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        print(f"Timeout al cargar o encontrar la tabla en {item}")
        return []

    rows = table.locator("tr")
    raw_rows = [
        rows.nth(i).locator("th, td").all_inner_texts()
        for i in range(rows.count())
    ]

    day_values = []
    idx = 0
    while idx < len(raw_rows):
        values = raw_rows[idx]
        if len(values) == 1 and values[0].strip().isdigit():
            break
        idx += 1

    while idx < len(raw_rows):
        values = raw_rows[idx]
        if len(values) != 1 or not values[0].strip().isdigit():
            break
        day_values.append(int(values[0].strip()))
        idx += 1

    if not day_values:
        print(f"No se encontraron dias en la tabla de {item}")
        return []

    month_rows = [{"day": day} for day in day_values]

    metric_groups = [
        ("temp", ("temp_max_f", "temp_avg_f", "temp_min_f")),
        ("dew", ("dew_max_f", "dew_avg_f", "dew_min_f")),
        ("humidity", ("humidity_max", "humidity_avg", "humidity_min")),
        ("wind", ("wind_max_mph", "wind_avg_mph", "wind_min_mph")),
        ("pressure", ("pressure_max_in", "pressure_avg_in", "pressure_min_in")),
    ]

    for _, columns in metric_groups:
        if idx >= len(raw_rows) or raw_rows[idx] != ["Max", "Avg", "Min"]:
            print(f"Estructura inesperada en {item}: falta encabezado Max/Avg/Min")
            return []

        idx += 1

        for row_data in month_rows:
            if idx >= len(raw_rows):
                print(f"Estructura inesperada en {item}: bloque incompleto")
                return []

            values = raw_rows[idx]
            if len(values) != 3:
                print(f"Estructura inesperada en {item}: fila de bloque invalida {values}")
                return []

            for column, value in zip(columns, values):
                row_data[column] = value

            idx += 1

    if idx >= len(raw_rows) or raw_rows[idx] != ["Total"]:
        print(f"Estructura inesperada en {item}: falta encabezado Total")
        return []

    idx += 1

    for row_data in month_rows:
        if idx >= len(raw_rows):
            print(f"Estructura inesperada en {item}: precipitacion incompleta")
            return []

        values = raw_rows[idx]
        if len(values) != 1:
            print(f"Estructura inesperada en {item}: precipitacion invalida {values}")
            return []

        row_data["precip_total_in"] = values[0]
        idx += 1

    row_dates = build_row_dates(month_start, [int(row_data["day"]) for row_data in month_rows])

    filtered_rows = []
    for row_data, row_date in zip(month_rows, row_dates):
        if row_date.year != month_start.year or row_date.month != month_start.month:
            continue

        row_data["date"] = row_date
        row_data["year"] = month_start.year
        row_data["month"] = month_start.month
        row_data["source_url"] = item
        filtered_rows.append(row_data)

    return filtered_rows


def shift_month(year, month, offset):
    month_index = year * 12 + (month - 1) + offset
    shifted_year, shifted_month_index = divmod(month_index, 12)
    return shifted_year, shifted_month_index + 1


def build_row_dates(month_start, day_values):
    if not day_values:
        return []

    if day_values[0] == 1:
        current_year, current_month = month_start.year, month_start.month
    else:
        current_year, current_month = shift_month(month_start.year, month_start.month, -1)

    row_dates = []
    previous_day = None

    for day in day_values:
        if previous_day is not None and day < previous_day:
            current_year, current_month = shift_month(current_year, current_month, 1)

        row_dates.append(datetime(current_year, current_month, day).date())
        previous_day = day

    return row_dates


if __name__ == "__main__":
    all_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for item in URLS:
            print(f"Extrayendo {item}")
            all_rows.extend(extract_month_rows(page, item))

        browser.close()

    df_final = pd.DataFrame(all_rows)

    if not df_final.empty:
        for column in NUMERIC_COLUMNS:
            df_final[column] = pd.to_numeric(df_final[column], errors="coerce")

        df_final["date"] = pd.to_datetime(df_final["date"])
        df_final = df_final.sort_values("date").reset_index(drop=True)

    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"Filas extraidas: {len(df_final)}")
    print(f"Archivo generado: {OUTPUT_CSV}")
