from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = SCRIPT_DIR.parent
RUN_DAILY_ENSEMBLE_PATH = ENSEMBLE_DIR / "run_daily_ensemble.py"
LATEST_ARTIFACTS_DIR = ENSEMBLE_DIR / "artifacts" / "latest"
LATEST_LIVE_FORECAST_PATH = LATEST_ARTIFACTS_DIR / "latest_live_forecast.json"
LATEST_MARKET_FORECAST_PATH = LATEST_ARTIFACTS_DIR / "latest_market_forecast.json"
LATEST_STATUS_PATH = LATEST_ARTIFACTS_DIR / "status.json"

ARGENTINA_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
DEFAULT_STATE_FILE = SCRIPT_DIR / "scheduler_state.json"
DEFAULT_COMBINED_PAYLOAD_FILE = LATEST_ARTIFACTS_DIR / "combined_forecast_payload.json"
DEFAULT_GRACE_SECONDS = 300
DEFAULT_SLEEP_CAP_SECONDS = 3600
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 30
DEFAULT_CLOUD_FUNCTION_URL = (
    "https://receivesignals-ccsbik7zha-uc.a.run.app"
)
SLOT_TIMES = (
    dt_time(hour=11, minute=0),
    dt_time(hour=11, minute=45),
    dt_time(hour=12, minute=30),
    dt_time(hour=13, minute=15),
    dt_time(hour=14, minute=0),
    dt_time(hour=14, minute=45),
    dt_time(hour=15, minute=30),
    dt_time(hour=16, minute=15),
    dt_time(hour=17, minute=0),
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Mantiene un scheduler liviano en memoria y ejecuta run_daily_ensemble.py "
            "todos los dias a las 11:00, 11:45, 12:30, 13:15, 14:00, 14:45, 15:30, "
            "16:15 y 17:00 hora de Argentina."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help="Archivo JSON para recordar el ultimo slot ejecutado.",
    )
    parser.add_argument(
        "--combined-payload-file",
        type=Path,
        default=DEFAULT_COMBINED_PAYLOAD_FILE,
        help="Archivo JSON combinado que se genera antes del POST HTTP.",
    )
    parser.add_argument(
        "--grace-seconds",
        type=int,
        default=DEFAULT_GRACE_SECONDS,
        help=(
            "Ventana extra para admitir pequenas demoras del reloj o del proceso "
            "respecto al slot programado."
        ),
    )
    parser.add_argument(
        "--sleep-cap-seconds",
        type=int,
        default=DEFAULT_SLEEP_CAP_SECONDS,
        help=(
            "Cuantos segundos maximo duerme por iteracion interna mientras espera "
            "el proximo slot."
        ),
    )
    parser.add_argument(
        "--upload-timeout-seconds",
        type=int,
        default=DEFAULT_UPLOAD_TIMEOUT_SECONDS,
        help="Timeout del POST HTTP en segundos.",
    )
    parser.add_argument(
        "--cloud-function-url",
        default=os.environ.get("FORECAST_CLOUD_FUNCTION_URL", DEFAULT_CLOUD_FUNCTION_URL),
        help=(
            "URL HTTP de la Cloud Function que recibira el JSON combinado. "
            "Tambien se puede definir con FORECAST_CLOUD_FUNCTION_URL."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Si falla run_daily_ensemble.py o el POST HTTP a la Cloud Function, "
            "detiene el scheduler."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Espera el proximo slot elegible, ejecuta una sola corrida y termina. "
            "Sirve para testing."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra configuracion y cronograma sin ejecutar nada.",
    )
    args, ensemble_args = parser.parse_known_args()

    if args.grace_seconds < 0:
        parser.error("--grace-seconds no puede ser negativo.")
    if args.sleep_cap_seconds <= 0:
        parser.error("--sleep-cap-seconds debe ser mayor que 0.")
    if args.upload_timeout_seconds <= 0:
        parser.error("--upload-timeout-seconds debe ser mayor que 0.")
    if not RUN_DAILY_ENSEMBLE_PATH.exists():
        parser.error(f"No se encontro {RUN_DAILY_ENSEMBLE_PATH}.")

    return args, ensemble_args


def now_argentina() -> datetime:
    return datetime.now(ARGENTINA_TZ)


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "none"
    return value.isoformat(timespec="seconds")


def build_slots_for_date(day: date) -> list[datetime]:
    return [
        datetime(
            year=day.year,
            month=day.month,
            day=day.day,
            hour=slot_time.hour,
            minute=slot_time.minute,
            tzinfo=ARGENTINA_TZ,
        )
        for slot_time in SLOT_TIMES
    ]


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[{format_timestamp(now_argentina())}] No se pudo leer state file {path}: {exc}. "
            "Se continua con estado vacio.",
            flush=True,
        )
        return {}

    if not isinstance(payload, dict):
        print(
            f"[{format_timestamp(now_argentina())}] State file invalido en {path}. "
            "Se continua con estado vacio.",
            flush=True,
        )
        return {}

    return payload


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def parse_state_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ARGENTINA_TZ)
    return parsed


def slot_already_started(last_slot_started: datetime | None, slot_time: datetime) -> bool:
    return last_slot_started is not None and last_slot_started >= slot_time


def slot_is_still_eligible(
    now_value: datetime,
    slot_time: datetime,
    next_slot_time: datetime | None,
    grace_seconds: int,
) -> bool:
    if slot_time > now_value:
        return False

    if next_slot_time is None:
        window_end = slot_time + timedelta(seconds=grace_seconds)
    else:
        window_end = next_slot_time + timedelta(seconds=grace_seconds)
    return now_value <= window_end


def find_due_slot(
    now_value: datetime,
    last_slot_started: datetime | None,
    grace_seconds: int,
) -> datetime | None:
    slots = build_slots_for_date(now_value.date())
    candidates: list[datetime] = []
    for idx, slot_time in enumerate(slots):
        if slot_already_started(last_slot_started, slot_time):
            continue
        next_slot_time = None if idx == len(slots) - 1 else slots[idx + 1]
        if slot_is_still_eligible(now_value, slot_time, next_slot_time, grace_seconds):
            candidates.append(slot_time)

    if not candidates:
        return None
    return candidates[-1]


def find_next_slot(now_value: datetime) -> datetime:
    for slot_time in build_slots_for_date(now_value.date()):
        if slot_time > now_value:
            return slot_time
    return build_slots_for_date(now_value.date() + timedelta(days=1))[0]


def wait_until(target_time: datetime, sleep_cap_seconds: int) -> None:
    while True:
        remaining_seconds = (target_time - now_argentina()).total_seconds()
        if remaining_seconds <= 0:
            return
        time.sleep(min(remaining_seconds, float(sleep_cap_seconds)))


def build_command(extra_args: list[str]) -> list[str]:
    return [sys.executable, str(RUN_DAILY_ENSEMBLE_PATH), *extra_args]


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No existe el archivo requerido: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"El archivo JSON es invalido: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path}.")
    return payload


def fahrenheit_to_celsius(value_f: object) -> float | None:
    if value_f is None:
        return None
    try:
        numeric = float(value_f)
    except (TypeError, ValueError):
        return None
    return round((numeric - 32.0) * (5.0 / 9.0), 2)


def build_combined_payload(
    slot_time: datetime,
    live_forecast: dict[str, Any],
    market_forecast: dict[str, Any],
    status_payload: dict[str, Any],
    cloud_function_url: str,
) -> dict[str, Any]:
    current_state_f = market_forecast.get("current_state_f", {})
    forecast_features_f = market_forecast.get("forecast_features_f", {})
    last_observation_used = status_payload.get("last_observation_used", {})
    source_url = None
    if isinstance(last_observation_used, dict):
        source_url = last_observation_used.get("source_url")

    current_temperature_f = current_state_f.get("temperature_f")
    max_so_far_f = current_state_f.get("max_temperature_so_far_today_f")
    ensemble_peak_forecast_f = forecast_features_f.get("ensemble_peak_forecast_f")
    forecast_origin_datetime = (
        market_forecast.get("forecast_origin_datetime")
        or live_forecast.get("last_observation_datetime")
    )

    return {
        "runId": status_payload.get("run_id"),
        "generatedAt": status_payload.get("generated_at") or format_timestamp(now_argentina()),
        "schedulerSlotAt": slot_time.isoformat(timespec="seconds"),
        "forecastOriginDatetime": forecast_origin_datetime,
        "currentTemperatureF": current_temperature_f,
        "currentTemperatureC": fahrenheit_to_celsius(current_temperature_f),
        "maxSoFarF": max_so_far_f,
        "maxSoFarC": fahrenheit_to_celsius(max_so_far_f),
        "ensemblePeakForecastF": ensemble_peak_forecast_f,
        "ensemblePeakForecastC": fahrenheit_to_celsius(ensemble_peak_forecast_f),
        "topMarketBins": market_forecast.get("top_market_bins", []),
        "probabilitiesAtOrAboveC": market_forecast.get("probabilities_at_or_above_c", []),
        "status": status_payload.get("status"),
        "sourceUrls": [source_url] if source_url else [],
        "cloudFunctionTarget": cloud_function_url,
        "latestLiveForecast": live_forecast,
        "latestMarketForecast": market_forecast,
    }


def write_combined_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


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


def upload_latest_artifacts(
    slot_time: datetime,
    cloud_function_url: str,
    combined_payload_file: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    live_forecast = load_json_file(LATEST_LIVE_FORECAST_PATH)
    market_forecast = load_json_file(LATEST_MARKET_FORECAST_PATH)
    status_payload = load_json_file(LATEST_STATUS_PATH)

    payload = build_combined_payload(
        slot_time=slot_time,
        live_forecast=live_forecast,
        market_forecast=market_forecast,
        status_payload=status_payload,
        cloud_function_url=cloud_function_url,
    )
    write_combined_payload(combined_payload_file, payload)

    http_status, response_body = post_json(
        url=cloud_function_url,
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    return {
        "payload_path": str(combined_payload_file),
        "http_status": int(http_status),
        "response_excerpt": response_body[:500],
        "run_id": payload.get("runId"),
    }


def run_slot(
    slot_time: datetime,
    state_file: Path,
    extra_args: list[str],
    cloud_function_url: str,
    combined_payload_file: Path,
    upload_timeout_seconds: int,
) -> dict[str, Any]:
    command = build_command(extra_args)
    command_display = subprocess.list2cmdline(command)
    started_at = now_argentina()

    state = load_state(state_file)
    state.update(
        {
            "last_slot_started": slot_time.isoformat(timespec="seconds"),
            "last_started_at": started_at.isoformat(timespec="seconds"),
            "last_command": command_display,
            "last_exit_code": None,
            "last_finished_at": None,
            "last_scheduler_exit_code": None,
        }
    )
    save_state(state_file, state)

    delay_seconds = (started_at - slot_time).total_seconds()
    if delay_seconds > 1:
        print(
            f"[{format_timestamp(started_at)}] Ejecutando slot {format_timestamp(slot_time)} "
            f"con demora de {delay_seconds:.1f}s.",
            flush=True,
        )
    else:
        print(
            f"[{format_timestamp(started_at)}] Ejecutando slot {format_timestamp(slot_time)}.",
            flush=True,
        )
    print(f"Comando: {command_display}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        command,
        cwd=str(ENSEMBLE_DIR),
        env=env,
        check=False,
    )
    finished_at = now_argentina()

    upload_summary: dict[str, Any] | None = None
    upload_error_message: str | None = None
    scheduler_exit_code = int(completed.returncode)

    if int(completed.returncode) == 0:
        try:
            upload_summary = upload_latest_artifacts(
                slot_time=slot_time,
                cloud_function_url=cloud_function_url,
                combined_payload_file=combined_payload_file,
                timeout_seconds=upload_timeout_seconds,
            )
            print(
                f"[{format_timestamp(now_argentina())}] Payload JSON combinado generado en "
                f"{upload_summary['payload_path']}.",
                flush=True,
            )
            print(
                f"[{format_timestamp(now_argentina())}] POST enviado a Cloud Function "
                f"{cloud_function_url} con status HTTP {upload_summary['http_status']}.",
                flush=True,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            upload_error_message = str(exc)
            scheduler_exit_code = 2
            print(
                f"[{format_timestamp(now_argentina())}] Error al generar o enviar el payload "
                f"de pronostico: {upload_error_message}",
                flush=True,
            )
    else:
        print(
            f"[{format_timestamp(now_argentina())}] Se omite el envio HTTP porque "
            f"run_daily_ensemble.py termino con exit_code={completed.returncode}.",
            flush=True,
        )

    state.update(
        {
            "last_exit_code": int(completed.returncode),
            "last_finished_at": finished_at.isoformat(timespec="seconds"),
            "last_scheduler_exit_code": int(scheduler_exit_code),
            "last_upload_url": cloud_function_url,
            "last_payload_path": str(combined_payload_file),
            "last_upload_ok": bool(upload_summary is not None),
            "last_upload_http_status": (
                None if upload_summary is None else int(upload_summary["http_status"])
            ),
            "last_upload_response_excerpt": (
                None if upload_summary is None else upload_summary["response_excerpt"]
            ),
            "last_upload_error": upload_error_message,
        }
    )
    save_state(state_file, state)

    print(
        f"[{format_timestamp(finished_at)}] Slot {format_timestamp(slot_time)} finalizado "
        f"con exit_code={completed.returncode} en {finished_at - started_at}.",
        flush=True,
    )
    return {
        "run_exit_code": int(completed.returncode),
        "scheduler_exit_code": int(scheduler_exit_code),
        "upload_ok": bool(upload_summary is not None),
    }


def print_schedule(reference_time: datetime, days: int = 2) -> None:
    for day_offset in range(days):
        current_day = reference_time.date() + timedelta(days=day_offset)
        print(f"Agenda para {current_day.isoformat()}:", flush=True)
        for slot_time in build_slots_for_date(current_day):
            print(f"- {format_timestamp(slot_time)}", flush=True)


def print_startup_summary(
    state_file: Path,
    combined_payload_file: Path,
    grace_seconds: int,
    sleep_cap_seconds: int,
    upload_timeout_seconds: int,
    ensemble_args: list[str],
    cloud_function_url: str,
) -> None:
    now_value = now_argentina()
    print(f"Scheduler iniciado: {format_timestamp(now_value)}", flush=True)
    print(f"Timezone operativa: {ARGENTINA_TZ.key}", flush=True)
    print(f"Run target: {RUN_DAILY_ENSEMBLE_PATH}", flush=True)
    print(f"State file: {state_file}", flush=True)
    print(f"Combined payload file: {combined_payload_file}", flush=True)
    print(f"Grace seconds: {grace_seconds}", flush=True)
    print(f"Sleep cap seconds: {sleep_cap_seconds}", flush=True)
    print(f"Upload timeout seconds: {upload_timeout_seconds}", flush=True)
    print(f"Cloud Function target: {cloud_function_url}", flush=True)
    if ensemble_args:
        print(
            "Argumentos reenviados a run_daily_ensemble.py: "
            f"{subprocess.list2cmdline(ensemble_args)}",
            flush=True,
        )
    else:
        print("Argumentos reenviados a run_daily_ensemble.py: ninguno", flush=True)


def main() -> int:
    args, ensemble_args = parse_args()
    print_startup_summary(
        state_file=args.state_file,
        combined_payload_file=args.combined_payload_file,
        grace_seconds=args.grace_seconds,
        sleep_cap_seconds=args.sleep_cap_seconds,
        upload_timeout_seconds=args.upload_timeout_seconds,
        ensemble_args=ensemble_args,
        cloud_function_url=args.cloud_function_url,
    )

    state = load_state(args.state_file)
    last_slot_started = parse_state_datetime(state.get("last_slot_started"))
    current_time = now_argentina()
    due_slot = find_due_slot(
        now_value=current_time,
        last_slot_started=last_slot_started,
        grace_seconds=args.grace_seconds,
    )
    next_slot = due_slot or find_next_slot(current_time)
    print(f"Ultimo slot registrado: {format_timestamp(last_slot_started)}", flush=True)
    print(f"Proximo slot elegible: {format_timestamp(next_slot)}", flush=True)

    if args.dry_run:
        print_schedule(current_time, days=2)
        return 0

    announced_next_slot: datetime | None = None
    try:
        while True:
            state = load_state(args.state_file)
            last_slot_started = parse_state_datetime(state.get("last_slot_started"))
            current_time = now_argentina()
            due_slot = find_due_slot(
                now_value=current_time,
                last_slot_started=last_slot_started,
                grace_seconds=args.grace_seconds,
            )

            if due_slot is not None:
                result = run_slot(
                    slot_time=due_slot,
                    state_file=args.state_file,
                    extra_args=ensemble_args,
                    cloud_function_url=args.cloud_function_url,
                    combined_payload_file=args.combined_payload_file,
                    upload_timeout_seconds=args.upload_timeout_seconds,
                )
                announced_next_slot = None
                if args.stop_on_error and result["scheduler_exit_code"] != 0:
                    return int(result["scheduler_exit_code"])
                if args.once:
                    return int(result["scheduler_exit_code"])
                continue

            next_slot = find_next_slot(current_time)
            if announced_next_slot != next_slot:
                print(
                    f"[{format_timestamp(current_time)}] Esperando proximo slot: "
                    f"{format_timestamp(next_slot)}",
                    flush=True,
                )
                announced_next_slot = next_slot
            wait_until(next_slot, sleep_cap_seconds=args.sleep_cap_seconds)
    except KeyboardInterrupt:
        print(
            f"\n[{format_timestamp(now_argentina())}] Scheduler interrumpido por el usuario.",
            flush=True,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
