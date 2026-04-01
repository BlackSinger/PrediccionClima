from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from argparse import BooleanOptionalAction
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DAILY_ENSEMBLE_PATH = SCRIPT_DIR / "run_daily_ensemble.py"


@dataclass
class AttemptResult:
    attempt_number: int
    scheduled_for: datetime
    started_at: datetime
    finished_at: datetime
    exit_code: int

    @property
    def duration(self) -> timedelta:
        return self.finished_at - self.started_at


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta run_daily_ensemble.py en intervalos regulares. "
            "Por defecto corre una vez al inicio y luego cada hora "
            "hasta completar 5 ejecuciones."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Cantidad total de ejecuciones de run_daily_ensemble.py.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=60.0,
        help="Minutos entre el inicio planificado de una corrida y la siguiente.",
    )
    parser.add_argument(
        "--delay-first-run-minutes",
        type=float,
        default=0.0,
        help="Demora inicial antes de la primera corrida. Por defecto es 0.",
    )
    parser.add_argument(
        "--stop-on-error",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Si se activa, corta la secuencia ante la primera corrida con error. "
            "Por defecto sigue intentando las restantes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra el cronograma calculado sin ejecutar run_daily_ensemble.py.",
    )
    args, ensemble_args = parser.parse_known_args()

    if args.runs <= 0:
        parser.error("--runs debe ser mayor que 0.")
    if args.interval_minutes < 0:
        parser.error("--interval-minutes no puede ser negativo.")
    if args.delay_first_run_minutes < 0:
        parser.error("--delay-first-run-minutes no puede ser negativo.")
    if not RUN_DAILY_ENSEMBLE_PATH.exists():
        parser.error(f"No se encontro {RUN_DAILY_ENSEMBLE_PATH}.")

    return args, ensemble_args


def now_local() -> datetime:
    return datetime.now().astimezone()


def format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def build_schedule(
    runs: int,
    interval_minutes: float,
    delay_first_run_minutes: float,
) -> list[datetime]:
    first_run_at = now_local() + timedelta(minutes=delay_first_run_minutes)
    interval = timedelta(minutes=interval_minutes)
    return [first_run_at + (interval * idx) for idx in range(runs)]


def wait_until(target_time: datetime) -> None:
    while True:
        remaining_seconds = (target_time - now_local()).total_seconds()
        if remaining_seconds <= 0:
            return
        time.sleep(min(remaining_seconds, 60.0))


def build_command(extra_args: list[str]) -> list[str]:
    return [sys.executable, str(RUN_DAILY_ENSEMBLE_PATH), *extra_args]


def run_attempt(
    attempt_number: int,
    scheduled_for: datetime,
    command: list[str],
) -> AttemptResult:
    actual_now = now_local()
    delay_seconds = (actual_now - scheduled_for).total_seconds()
    if delay_seconds > 5:
        print(
            (
                f"[{format_timestamp(actual_now)}] La corrida {attempt_number} "
                f"empieza con {delay_seconds:.1f}s de demora respecto al horario planificado."
            ),
            flush=True,
        )

    command_display = subprocess.list2cmdline(command)
    print(
        (
            f"[{format_timestamp(actual_now)}] Iniciando corrida {attempt_number} "
            f"programada para {format_timestamp(scheduled_for)}"
        ),
        flush=True,
    )
    print(f"Comando: {command_display}", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    started_at = now_local()
    completed = subprocess.run(
        command,
        cwd=str(SCRIPT_DIR),
        env=env,
        check=False,
    )
    finished_at = now_local()

    print(
        (
            f"[{format_timestamp(finished_at)}] Corrida {attempt_number} finalizada "
            f"con exit_code={completed.returncode} en {finished_at - started_at}."
        ),
        flush=True,
    )

    return AttemptResult(
        attempt_number=attempt_number,
        scheduled_for=scheduled_for,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=int(completed.returncode),
    )


def print_schedule(schedule: list[datetime]) -> None:
    print("Cronograma planificado:", flush=True)
    for idx, scheduled_for in enumerate(schedule, start=1):
        print(f"- Corrida {idx}: {format_timestamp(scheduled_for)}", flush=True)


def print_summary(results: list[AttemptResult]) -> None:
    print("\nResumen de ejecuciones:", flush=True)
    for result in results:
        print(
            (
                f"- Corrida {result.attempt_number}: exit_code={result.exit_code} | "
                f"programada={format_timestamp(result.scheduled_for)} | "
                f"inicio={format_timestamp(result.started_at)} | "
                f"duracion={result.duration}"
            ),
            flush=True,
        )


def main() -> int:
    args, ensemble_args = parse_args()
    schedule = build_schedule(
        runs=args.runs,
        interval_minutes=args.interval_minutes,
        delay_first_run_minutes=args.delay_first_run_minutes,
    )
    command = build_command(ensemble_args)

    print_schedule(schedule)
    if ensemble_args:
        print(
            "Argumentos reenviados a run_daily_ensemble.py: "
            f"{subprocess.list2cmdline(ensemble_args)}",
            flush=True,
        )

    if args.dry_run:
        print("Dry run activado: no se ejecutara ninguna corrida.", flush=True)
        return 0

    results: list[AttemptResult] = []
    try:
        for attempt_number, scheduled_for in enumerate(schedule, start=1):
            wait_until(scheduled_for)
            result = run_attempt(
                attempt_number=attempt_number,
                scheduled_for=scheduled_for,
                command=command,
            )
            results.append(result)
            if result.exit_code != 0 and args.stop_on_error:
                print(
                    "Se detiene la secuencia porque --stop-on-error esta activado.",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        print("\nEjecucion interrumpida por el usuario.", flush=True)
        print_summary(results)
        return 130

    print_summary(results)
    if any(result.exit_code != 0 for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
