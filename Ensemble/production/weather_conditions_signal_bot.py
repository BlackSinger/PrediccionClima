from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:
    PlaywrightError = None
    PlaywrightTimeoutError = None
    sync_playwright = None

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "weather_signal_data"
BUFFER_STATE_PATH = DATA_ROOT / "open_minute_buffer.json"
LATEST_READING_PATH = DATA_ROOT / "latest_reading.json"
PENDING_BATCHES_DIR = DATA_ROOT / "pending_batches"
UPLOADED_BATCHES_DIR = DATA_ROOT / "uploaded_batches"

BUENOS_AIRES_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
DEFAULT_URL = "https://www.wunderground.com/weather/ar/ezeiza/SAEZ"
FALLBACK_URLS = [
    "https://www.wunderground.com/weather/SAEZ",
    "https://www.wunderground.com/weather/us/SAEZ",
]
DEFAULT_INTERVAL_SECONDS = 30
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 60
DEFAULT_CLOUD_FUNCTION_URL = (
    "https://us-central1-polymarketsignals-88875.cloudfunctions.net/weatherSignals"
)
NAVIGATION_TIMEOUT_MS = 60000
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
REQUIRED_CONDITIONS = (
    "Pressure",
    "Visibility",
    "Clouds",
    "Dew Point",
    "Humidity",
    "Rainfall",
    "Snow Depth",
)


def ensure_playwright() -> None:
    if sync_playwright is None or PlaywrightTimeoutError is None or PlaywrightError is None:
        raise SystemExit(
            "No se encontro Playwright. Instala las dependencias antes de ejecutar el bot."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrapea condiciones meteorologicas de Weather Underground cada 10 segundos, "
            "arma batches JSON por minuto y los envia por POST a una Cloud Function."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Segundos entre scrapes consecutivos.",
    )
    parser.add_argument(
        "--cloud-function-url",
        default=os.environ.get("WEATHER_SIGNAL_CLOUD_FUNCTION_URL", DEFAULT_CLOUD_FUNCTION_URL),
        help=(
            "URL HTTP de la Cloud Function que recibira los batches JSON. "
            "Tambien se puede definir con WEATHER_SIGNAL_CLOUD_FUNCTION_URL."
        ),
    )
    parser.add_argument(
        "--upload-timeout-seconds",
        type=int,
        default=DEFAULT_UPLOAD_TIMEOUT_SECONDS,
        help="Timeout del POST HTTP en segundos.",
    )
    parser.add_argument(
        "--disable-upload",
        action="store_true",
        help="Genera batches pendientes localmente pero no hace POST HTTP.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Abre el navegador visible en lugar de headless.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Hace una sola lectura y termina.",
    )
    parser.add_argument(
        "--max-readings",
        type=int,
        default=None,
        help="Cantidad maxima de lecturas antes de terminar.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Detiene el bot ante un error de scraping o de upload.",
    )
    args = parser.parse_args()

    if args.interval_seconds <= 0:
        parser.error("--interval-seconds debe ser mayor a 0.")
    if args.upload_timeout_seconds <= 0:
        parser.error("--upload-timeout-seconds debe ser mayor a 0.")
    if args.max_readings is not None and args.max_readings <= 0:
        parser.error("--max-readings debe ser mayor a 0.")
    return args


def now_buenos_aires() -> datetime:
    return datetime.now(BUENOS_AIRES_TZ)


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "none"
    return value.isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[{format_timestamp(now_buenos_aires())}] No se pudo leer {path}: {exc}. "
            "Se usa estado vacio.",
            flush=True,
        )
        return default


def clean_text(value: str) -> str:
    return value.replace("\xa0", " ").strip()


def normalize_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return float(value)


def extract_numeric_value(text: str) -> int | float | None:
    normalized = clean_text(text)
    match = re.search(r"-?\d+(?:\.\d+)?", normalized.replace(",", ""))
    if match is None:
        return None
    return normalize_number(float(match.group(0)))


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


def try_extract_temperature(page: Any) -> dict[str, Any] | None:
    for selector in TEMPERATURE_SELECTORS:
        locator = page.locator(selector).first
        try:
            locator.wait_for(state="visible", timeout=SELECTOR_TIMEOUT_MS)
            text = clean_text(locator.inner_text())
        except PlaywrightTimeoutError:
            continue

        numeric_value = extract_numeric_value(text)
        if numeric_value is None:
            continue

        return {
            "value_f": numeric_value,
            "raw": text,
        }
    return None


def extract_updated_text(body_text: str) -> str | None:
    match = re.search(r"updated\s+.+?ago", body_text, flags=re.IGNORECASE)
    if match is None:
        return None
    return clean_text(match.group(0))


def extract_additional_conditions(page: Any) -> dict[str, str]:
    module = page.locator(".data-module.additional-conditions").first
    module.wait_for(state="visible", timeout=SELECTOR_TIMEOUT_MS)
    rows = module.locator(".row")
    result: dict[str, str] = {}
    for idx in range(rows.count()):
        row = rows.nth(idx)
        label = clean_text(row.locator(".small-4.columns").first.inner_text())
        value = clean_text(row.locator(".small-8.columns").first.inner_text())
        if label:
            result[label] = value

    missing = [label for label in REQUIRED_CONDITIONS if label not in result]
    if missing:
        raise RuntimeError(
            "Faltan condiciones adicionales requeridas: " + ", ".join(missing)
        )
    return result


def scrape_conditions(page: Any, candidate_urls: list[str]) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None

    for url in candidate_urls:
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(3000)
            title, body_text = get_page_snapshot(page)
            if is_not_found_page(title, body_text):
                continue
            if not page_matches_expected_station(title, body_text):
                continue

            temperature = try_extract_temperature(page)
            if temperature is None:
                raise RuntimeError("No se pudo extraer la temperatura actual.")

            additional = extract_additional_conditions(page)
            reading = build_reading_payload(
                source_url=url,
                page_title=title,
                updated_text=extract_updated_text(body_text),
                temperature=temperature,
                additional=additional,
            )
            return reading, url
        except (PlaywrightTimeoutError, PlaywrightError, RuntimeError) as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            "No fue posible extraer las condiciones actuales desde Weather Underground."
        ) from last_error
    raise RuntimeError("No fue posible encontrar una URL valida para la estacion SAEZ.")


def build_reading_payload(
    source_url: str,
    page_title: str,
    updated_text: str | None,
    temperature: dict[str, Any],
    additional: dict[str, str],
) -> dict[str, Any]:
    collected_at = now_buenos_aires()

    pressure_raw = additional["Pressure"]
    visibility_raw = additional["Visibility"]
    clouds_raw = additional["Clouds"]
    dew_point_raw = additional["Dew Point"]
    humidity_raw = additional["Humidity"]
    rainfall_raw = additional["Rainfall"]
    snow_depth_raw = additional["Snow Depth"]

    return {
        "collected_at_buenos_aires": collected_at.isoformat(timespec="seconds"),
        "source_url": source_url,
        "station_code": "SAEZ",
        "page_title": page_title,
        "updated_text": updated_text,
        "temperature_f": temperature["value_f"],
        "temperature_raw": temperature["raw"],
        "pressure_in": extract_numeric_value(pressure_raw),
        "pressure_raw": pressure_raw,
        "visibility_miles": extract_numeric_value(visibility_raw),
        "visibility_raw": visibility_raw,
        "clouds": clouds_raw,
        "dew_point_f": extract_numeric_value(dew_point_raw),
        "dew_point_raw": dew_point_raw,
        "humidity_pct": extract_numeric_value(humidity_raw),
        "humidity_raw": humidity_raw,
        "rainfall_in": extract_numeric_value(rainfall_raw),
        "rainfall_raw": rainfall_raw,
        "snow_depth_in": extract_numeric_value(snow_depth_raw),
        "snow_depth_raw": snow_depth_raw,
    }


def minute_bucket(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def minute_key(value: datetime) -> str:
    return minute_bucket(value).isoformat(timespec="seconds")


def batch_id_from_minute(value: datetime) -> str:
    return value.strftime("weather_batch_%Y-%m-%d_%H%M_ART")


def parse_minute_key(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_open_buffers() -> dict[str, list[dict[str, Any]]]:
    payload = load_json(BUFFER_STATE_PATH, {})
    if not isinstance(payload, dict):
        return {}

    result: dict[str, list[dict[str, Any]]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, list):
            result[key] = [item for item in value if isinstance(item, dict)]
    return result


def save_open_buffers(buffers: dict[str, list[dict[str, Any]]]) -> None:
    write_json(BUFFER_STATE_PATH, buffers)


def batch_file_path(directory: Path, bucket_time: datetime) -> Path:
    return directory / f"{batch_id_from_minute(bucket_time)}.json"


def build_batch_payload(
    bucket_time: datetime,
    readings: list[dict[str, Any]],
) -> dict[str, Any]:
    if not readings:
        raise ValueError("No se puede construir un batch vacio.")

    source_urls = sorted(
        {
            str(reading.get("source_url"))
            for reading in readings
            if isinstance(reading.get("source_url"), str) and reading.get("source_url")
        }
    )
    return {
        "batchId": batch_id_from_minute(bucket_time),
        "generatedAt": format_timestamp(now_buenos_aires()),
        "windowStartBuenosAires": bucket_time.isoformat(timespec="seconds"),
        "windowEndBuenosAires": (
            bucket_time + timedelta(minutes=1)
        ).isoformat(timespec="seconds"),
        "stationCode": "SAEZ",
        "sourceUrls": source_urls,
        "readingCount": len(readings),
        "readings": readings,
    }


def flush_completed_minutes(
    buffers: dict[str, list[dict[str, Any]]],
    now_value: datetime,
) -> bool:
    current_minute = minute_bucket(now_value)
    completed_keys = sorted(
        key
        for key in buffers
        if parse_minute_key(key) < current_minute
    )
    changed = False

    for key in completed_keys:
        bucket_time = parse_minute_key(key)
        pending_path = batch_file_path(PENDING_BATCHES_DIR, bucket_time)
        uploaded_path = batch_file_path(UPLOADED_BATCHES_DIR, bucket_time)
        if not pending_path.exists() and not uploaded_path.exists():
            payload = build_batch_payload(bucket_time=bucket_time, readings=buffers[key])
            write_json(pending_path, payload)
            print(
                f"[{format_timestamp(now_buenos_aires())}] Batch pendiente generado: {pending_path}",
                flush=True,
            )
        del buffers[key]
        changed = True

    if changed:
        save_open_buffers(buffers)
    return changed


def post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return int(getattr(response, "status", response.getcode())), response_body
    except urllib_error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} al llamar {url}: {response_body[:500] or exc.reason}"
        ) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Error de red al llamar {url}: {exc.reason}") from exc


def upload_pending_batches(
    cloud_function_url: str,
    timeout_seconds: int,
    disable_upload: bool,
) -> None:
    if disable_upload:
        return

    PENDING_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADED_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    pending_files = sorted(PENDING_BATCHES_DIR.glob("*.json"))

    for pending_file in pending_files:
        payload = load_json(pending_file, None)
        if not isinstance(payload, dict):
            raise RuntimeError(f"El batch pendiente es invalido: {pending_file}")

        http_status, response_body = post_json(
            url=cloud_function_url,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        uploaded_path = UPLOADED_BATCHES_DIR / pending_file.name
        pending_file.replace(uploaded_path)
        print(
            f"[{format_timestamp(now_buenos_aires())}] Batch enviado a {cloud_function_url} "
            f"con status HTTP {http_status}: {uploaded_path}",
            flush=True,
        )
        if response_body:
            print(
                f"[{format_timestamp(now_buenos_aires())}] Respuesta upload: "
                f"{response_body[:300]}",
                flush=True,
            )


def record_reading(
    reading: dict[str, Any],
    buffers: dict[str, list[dict[str, Any]]],
) -> None:
    collected_at = datetime.fromisoformat(str(reading["collected_at_buenos_aires"]))
    key = minute_key(collected_at)
    buffers.setdefault(key, []).append(reading)
    save_open_buffers(buffers)
    write_json(LATEST_READING_PATH, reading)


def should_stop(readings_taken: int, args: argparse.Namespace) -> bool:
    if args.once:
        return True
    if args.max_readings is not None and readings_taken >= args.max_readings:
        return True
    return False


def print_startup_summary(args: argparse.Namespace) -> None:
    print(f"Bot iniciado: {format_timestamp(now_buenos_aires())}", flush=True)
    print(f"Timezone operativa: {BUENOS_AIRES_TZ.key}", flush=True)
    print(f"URL principal: {DEFAULT_URL}", flush=True)
    print(f"Data root: {DATA_ROOT}", flush=True)
    print(f"Cloud Function target: {args.cloud_function_url}", flush=True)
    print(f"Uploads habilitados: {not args.disable_upload}", flush=True)
    print(f"Intervalo de scraping: {args.interval_seconds}s", flush=True)


def main() -> int:
    ensure_playwright()
    args = parse_args()
    print_startup_summary(args)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PENDING_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADED_BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    candidate_urls = [DEFAULT_URL, *FALLBACK_URLS]
    readings_taken = 0
    buffers = load_open_buffers()

    try:
        upload_pending_batches(
            cloud_function_url=args.cloud_function_url,
            timeout_seconds=args.upload_timeout_seconds,
            disable_upload=args.disable_upload,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", flush=True)
        if args.stop_on_error:
            return 2

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1440, "height": 2200})

        try:
            while True:
                try:
                    reading, resolved_url = scrape_conditions(page, candidate_urls)
                    candidate_urls = [resolved_url, *[url for url in candidate_urls if url != resolved_url]]
                    record_reading(reading, buffers)
                    readings_taken += 1
                    print(
                        "[OK] "
                        f"{reading['collected_at_buenos_aires']} | "
                        f"{reading['temperature_f']} F | "
                        f"Pressure={reading['pressure_in']} in | "
                        f"Humidity={reading['humidity_pct']} % | "
                        f"Clouds={reading['clouds']}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    print(f"[ERROR] {exc}", flush=True)
                    if args.stop_on_error:
                        return 2
                    time.sleep(args.interval_seconds)
                    continue

                flush_completed_minutes(buffers=buffers, now_value=now_buenos_aires())
                try:
                    upload_pending_batches(
                        cloud_function_url=args.cloud_function_url,
                        timeout_seconds=args.upload_timeout_seconds,
                        disable_upload=args.disable_upload,
                    )
                except RuntimeError as exc:
                    print(f"[ERROR] {exc}", flush=True)
                    if args.stop_on_error:
                        return 2

                if should_stop(readings_taken, args):
                    break

                time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("Bot detenido por el usuario.", flush=True)
        finally:
            flush_completed_minutes(buffers=buffers, now_value=now_buenos_aires())
            browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
