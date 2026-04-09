from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:
    PlaywrightError = None
    PlaywrightTimeoutError = None
    sync_playwright = None

DEFAULT_URL = "https://www.wunderground.com/weather/us/SAEZ"
FALLBACK_URLS = [
    "https://www.wunderground.com/weather/SAEZ",
    "https://www.wunderground.com/weather/ar/ezeiza/SAEZ",
]
DEFAULT_INTERVAL_SECONDS = 10.0
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("temperature_log.json")
BUENOS_AIRES_TIMEZONE = ZoneInfo("America/Argentina/Buenos_Aires")
NAVIGATION_TIMEOUT_MS = 30000
SELECTOR_TIMEOUT_MS = 30000
EXPECTED_LOCATION_KEYWORDS = [
    "ezeiza",
    "minister pistarini",
]
TEMPERATURE_SELECTORS = [
    ".current-temp .wu-value.wu-value-to",
    ".current-temp .wu-value",
    ".current-temp .wu-unit-temperature",
    ".current-temp",
]


@dataclass
class TemperatureReading:
    timestamp_buenos_aires: str
    temperature_f: int | float
    source_url: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_buenos_aires": self.timestamp_buenos_aires,
            "temperature_f": self.temperature_f,
            "source_url": self.source_url,
        }


def ensure_playwright() -> None:
    if sync_playwright is None or PlaywrightTimeoutError is None or PlaywrightError is None:
        raise SystemExit(
            "No se encontro Playwright. Instala las dependencias antes de ejecutar el bot."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hace scraping de la temperatura actual en Weather Underground y guarda "
            "las lecturas en un archivo JSON."
        )
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Segundos entre scraping y scraping. Default: 10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Ruta del archivo JSON de salida. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ejecuta una sola lectura y termina.",
    )
    parser.add_argument(
        "--max-readings",
        type=int,
        default=None,
        help="Cantidad maxima de lecturas antes de terminar.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Abre el navegador visible en lugar de headless.",
    )
    return parser.parse_args()


def load_existing_readings(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"El archivo JSON existente no es valido: {output_path}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"El archivo JSON existente debe contener una lista de lecturas: {output_path}"
        )

    return data


def write_readings(output_path: Path, readings: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(readings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(output_path)


def extract_temperature_value(text: str) -> int | float:
    normalized_text = text.replace("\xa0", " ").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", normalized_text)
    if match is None:
        raise ValueError(f"No se pudo parsear la temperatura desde: {text!r}")

    number_text = match.group(0)
    if "." in number_text:
        return float(number_text)
    return int(number_text)


def get_page_snapshot(page: Any) -> tuple[str, str]:
    title = page.title().lower()
    body_text = page.locator("body").inner_text().lower()
    return title, body_text


def page_matches_expected_station(title: str, body_text: str) -> bool:
    haystack = f"{title}\n{body_text}"
    return all(keyword in haystack for keyword in EXPECTED_LOCATION_KEYWORDS)


def is_not_found_page(title: str, body_text: str) -> bool:
    return (
        "error" in title
        or "page not found" in title
        or "error 404" in body_text
        or "page not found" in body_text
    )


def try_extract_temperature(page: Any) -> int | float | None:
    for selector in TEMPERATURE_SELECTORS:
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=SELECTOR_TIMEOUT_MS)
            text = locator.inner_text()
        except PlaywrightTimeoutError:
            continue

        try:
            return extract_temperature_value(text)
        except ValueError:
            continue

    return None


def scrape_temperature(page: Any, candidate_urls: list[str]) -> tuple[int | float, str]:
    last_error: Exception | None = None

    for url in candidate_urls:
        try:
            page.goto(url, wait_until="commit", timeout=NAVIGATION_TIMEOUT_MS)
            title, body_text = get_page_snapshot(page)
            if is_not_found_page(title, body_text):
                continue
            if not page_matches_expected_station(title, body_text):
                continue

            temperature = try_extract_temperature(page)
            if temperature is not None:
                return temperature, url
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            "No fue posible obtener la temperatura desde Weather Underground."
        ) from last_error

    raise RuntimeError(
        "No fue posible encontrar una URL valida ni extraer la temperatura actual."
    )


def build_reading(temperature_f: int | float, source_url: str) -> TemperatureReading:
    timestamp = datetime.now(BUENOS_AIRES_TIMEZONE).isoformat(timespec="seconds")
    return TemperatureReading(
        timestamp_buenos_aires=timestamp,
        temperature_f=temperature_f,
        source_url=source_url,
    )


def should_stop(readings_taken: int, args: argparse.Namespace) -> bool:
    if args.once:
        return True
    if args.max_readings is not None and readings_taken >= args.max_readings:
        return True
    return False


def main() -> None:
    ensure_playwright()
    args = parse_args()

    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds debe ser mayor a 0.")
    if args.max_readings is not None and args.max_readings <= 0:
        raise SystemExit("--max-readings debe ser mayor a 0.")

    candidate_urls = [DEFAULT_URL, *FALLBACK_URLS]
    readings = load_existing_readings(args.output)
    readings_taken = 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        page = browser.new_page()

        try:
            while True:
                try:
                    temperature_f, resolved_url = scrape_temperature(page, candidate_urls)
                except RuntimeError as exc:
                    print(f"[ERROR] {exc}")
                    time.sleep(args.interval_seconds)
                    continue

                reading = build_reading(temperature_f, resolved_url)
                readings.append(reading.as_dict())
                write_readings(args.output, readings)
                readings_taken += 1

                print(
                    "[OK] "
                    f"{reading.timestamp_buenos_aires} | "
                    f"{reading.temperature_f} F | "
                    f"{reading.source_url}"
                )

                candidate_urls = [resolved_url, *[url for url in candidate_urls if url != resolved_url]]

                if should_stop(readings_taken, args):
                    break

                time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("Bot detenido por el usuario.")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
