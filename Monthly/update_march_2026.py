from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

from extract_data import BASE_URL, COLUMNS, NUMERIC_COLUMNS, OUTPUT_CSV, extract_month_rows

TARGET_YEAR = 2026
TARGET_MONTH = 3
TARGET_URL = f"{BASE_URL}/{TARGET_YEAR}-{TARGET_MONTH:02d}-01"
CSV_PATH = Path(__file__).with_name(OUTPUT_CSV)
FINAL_COLUMNS = COLUMNS + ["date", "year", "month", "source_url"]


def download_target_month():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            month_rows = extract_month_rows(page, TARGET_URL)
        finally:
            browser.close()

    if not month_rows:
        raise RuntimeError(f"No se pudieron extraer filas para {TARGET_YEAR}-{TARGET_MONTH:02d}")

    month_df = pd.DataFrame(month_rows)

    for column in NUMERIC_COLUMNS:
        month_df[column] = pd.to_numeric(month_df[column], errors="coerce")

    month_df["date"] = pd.to_datetime(month_df["date"])
    month_df = month_df.sort_values("date").reset_index(drop=True)

    return month_df


def load_existing_data(csv_path):
    if not csv_path.exists():
        return pd.DataFrame(columns=FINAL_COLUMNS)

    existing_df = pd.read_csv(csv_path)

    if "date" in existing_df.columns:
        existing_df["date"] = pd.to_datetime(existing_df["date"])

    return existing_df


def update_csv(month_df, csv_path):
    existing_df = load_existing_data(csv_path)

    if existing_df.empty:
        updated_df = month_df.copy()
    else:
        keep_mask = ~(
            (existing_df["date"].dt.year == TARGET_YEAR)
            & (existing_df["date"].dt.month == TARGET_MONTH)
        )
        updated_df = pd.concat(
            [existing_df.loc[keep_mask].copy(), month_df],
            ignore_index=True,
        )

    for column in NUMERIC_COLUMNS:
        updated_df[column] = pd.to_numeric(updated_df[column], errors="coerce")

    updated_df["date"] = pd.to_datetime(updated_df["date"])
    updated_df = (
        updated_df.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    for column in FINAL_COLUMNS:
        if column not in updated_df.columns:
            updated_df[column] = pd.NA

    updated_df = updated_df[FINAL_COLUMNS]
    updated_df.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

    return updated_df


if __name__ == "__main__":
    print(f"Descargando {TARGET_YEAR}-{TARGET_MONTH:02d} desde {TARGET_URL}")
    month_df = download_target_month()
    updated_df = update_csv(month_df, CSV_PATH)

    print(f"Filas descargadas del mes: {len(month_df)}")
    print(
        "Rango actualizado del mes: "
        f"{month_df['date'].min().date()} -> {month_df['date'].max().date()}"
    )
    print(f"Filas totales del CSV: {len(updated_df)}")
    print(f"Archivo actualizado: {CSV_PATH}")
