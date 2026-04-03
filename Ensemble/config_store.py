from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"


class ConfigValidationError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(
        value, bool
    )


def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"'{path}' debe ser un objeto JSON.")
    return value


def _expect_key(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigValidationError(f"Falta la clave requerida '{path}.{key}'.")
    return mapping[key]


def _expect_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"'{path}' debe ser un string no vacio.")
    return value


def _expect_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ConfigValidationError(f"'{path}' debe ser booleano.")
    return value


def _expect_int(value: Any, path: str, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise ConfigValidationError(f"'{path}' debe ser un entero.")
    if minimum is not None and value < minimum:
        raise ConfigValidationError(f"'{path}' debe ser >= {minimum}.")
    return int(value)


def _expect_number(
    value: Any,
    path: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not _is_number(value):
        raise ConfigValidationError(f"'{path}' debe ser numerico.")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ConfigValidationError(f"'{path}' debe ser >= {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ConfigValidationError(f"'{path}' debe ser <= {maximum}.")
    return numeric


def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f"'{path}' debe ser una lista no vacia.")
    return value


def _validate_string_list(value: Any, path: str) -> list[str]:
    values = _expect_list(value, path)
    result: list[str] = []
    for idx, item in enumerate(values):
        result.append(_expect_string(item, f"{path}[{idx}]"))
    return result


def _validate_int_list(value: Any, path: str) -> list[int]:
    values = _expect_list(value, path)
    result: list[int] = []
    for idx, item in enumerate(values):
        result.append(_expect_int(item, f"{path}[{idx}]", minimum=0))
    return result


def _validate_number_list(value: Any, path: str) -> list[float]:
    values = _expect_list(value, path)
    result: list[float] = []
    for idx, item in enumerate(values):
        result.append(_expect_number(item, f"{path}[{idx}]"))
    return result


def _validate_config(config: Any) -> dict[str, Any]:
    root = _expect_mapping(config, "config")

    shared = _expect_mapping(_expect_key(root, "shared", "config"), "config.shared")
    _expect_int(_expect_key(shared, "forecast_horizon", "config.shared"), "config.shared.forecast_horizon", minimum=1)
    train_ratio = _expect_number(_expect_key(shared, "train_ratio", "config.shared"), "config.shared.train_ratio", minimum=0.0, maximum=1.0)
    val_ratio = _expect_number(_expect_key(shared, "val_ratio", "config.shared"), "config.shared.val_ratio", minimum=0.0, maximum=1.0)
    if train_ratio + val_ratio >= 1.0:
        raise ConfigValidationError(
            "'config.shared.train_ratio' + 'config.shared.val_ratio' debe ser menor a 1."
        )
    _expect_number(_expect_key(shared, "interval_coverage", "config.shared"), "config.shared.interval_coverage", minimum=0.0, maximum=1.0)
    _expect_int(_expect_key(shared, "max_gap_minutes", "config.shared"), "config.shared.max_gap_minutes", minimum=1)
    _expect_int(_expect_key(shared, "seed", "config.shared"), "config.shared.seed", minimum=0)

    paths = _expect_mapping(_expect_key(root, "paths", "config"), "config.paths")
    for key in (
        "data",
        "output_root",
        "runs_directory",
        "latest_directory",
        "status_file",
        "gru_artifact",
        "gru_cleaned_data",
        "lightgbm_model",
        "lightgbm_metadata",
        "lightgbm_cleaned_data",
    ):
        _expect_string(_expect_key(paths, key, "config.paths"), f"config.paths.{key}")

    ensemble = _expect_mapping(_expect_key(root, "ensemble", "config"), "config.ensemble")
    runtime = _expect_mapping(
        _expect_key(ensemble, "runtime_defaults", "config.ensemble"),
        "config.ensemble.runtime_defaults",
    )
    _expect_number(_expect_key(runtime, "weight_grid_step", "config.ensemble.runtime_defaults"), "config.ensemble.runtime_defaults.weight_grid_step", minimum=0.0, maximum=1.0)
    _expect_int(_expect_key(runtime, "batch_size", "config.ensemble.runtime_defaults"), "config.ensemble.runtime_defaults.batch_size", minimum=1)
    _expect_bool(_expect_key(runtime, "headless", "config.ensemble.runtime_defaults"), "config.ensemble.runtime_defaults.headless")
    _expect_bool(_expect_key(runtime, "update_data", "config.ensemble.runtime_defaults"), "config.ensemble.runtime_defaults.update_data")
    _expect_bool(_expect_key(runtime, "retrain_models", "config.ensemble.runtime_defaults"), "config.ensemble.runtime_defaults.retrain_models")
    _expect_int(_expect_key(runtime, "lightgbm_search_trials", "config.ensemble.runtime_defaults"), "config.ensemble.runtime_defaults.lightgbm_search_trials", minimum=0)
    _expect_bool(_expect_key(runtime, "save_detailed_artifacts", "config.ensemble.runtime_defaults"), "config.ensemble.runtime_defaults.save_detailed_artifacts")

    gru = _expect_mapping(_expect_key(root, "gru", "config"), "config.gru")
    gru_defaults = _expect_mapping(
        _expect_key(gru, "train_defaults", "config.gru"),
        "config.gru.train_defaults",
    )
    for key in ("lookback", "hidden_size", "num_layers", "batch_size", "epochs", "patience"):
        _expect_int(_expect_key(gru_defaults, key, "config.gru.train_defaults"), f"config.gru.train_defaults.{key}", minimum=1)
    _expect_number(_expect_key(gru_defaults, "dropout", "config.gru.train_defaults"), "config.gru.train_defaults.dropout", minimum=0.0, maximum=1.0)
    _expect_number(_expect_key(gru_defaults, "learning_rate", "config.gru.train_defaults"), "config.gru.train_defaults.learning_rate", minimum=0.0)

    lightgbm = _expect_mapping(_expect_key(root, "lightgbm", "config"), "config.lightgbm")
    feature_engineering = _expect_mapping(
        _expect_key(lightgbm, "feature_engineering", "config.lightgbm"),
        "config.lightgbm.feature_engineering",
    )
    _validate_string_list(
        _expect_key(feature_engineering, "lag_source_features", "config.lightgbm.feature_engineering"),
        "config.lightgbm.feature_engineering.lag_source_features",
    )
    _validate_int_list(
        _expect_key(feature_engineering, "lag_step_candidates", "config.lightgbm.feature_engineering"),
        "config.lightgbm.feature_engineering.lag_step_candidates",
    )
    _validate_int_list(
        _expect_key(feature_engineering, "rolling_window_candidates", "config.lightgbm.feature_engineering"),
        "config.lightgbm.feature_engineering.rolling_window_candidates",
    )

    search = _expect_mapping(
        _expect_key(lightgbm, "search", "config.lightgbm"),
        "config.lightgbm.search",
    )
    param_names = _validate_string_list(
        _expect_key(search, "param_names", "config.lightgbm.search"),
        "config.lightgbm.search.param_names",
    )
    param_candidates = _expect_mapping(
        _expect_key(search, "param_candidates", "config.lightgbm.search"),
        "config.lightgbm.search.param_candidates",
    )
    for param_name in param_names:
        values = _expect_key(
            param_candidates,
            param_name,
            "config.lightgbm.search.param_candidates",
        )
        _validate_number_list(values, f"config.lightgbm.search.param_candidates.{param_name}")

    lightgbm_defaults = _expect_mapping(
        _expect_key(lightgbm, "train_defaults", "config.lightgbm"),
        "config.lightgbm.train_defaults",
    )
    for key in (
        "lookback",
        "num_boost_round",
        "patience",
        "num_leaves",
        "min_data_in_leaf",
        "bagging_freq",
        "search_trials",
        "num_threads",
    ):
        _expect_int(
            _expect_key(lightgbm_defaults, key, "config.lightgbm.train_defaults"),
            f"config.lightgbm.train_defaults.{key}",
            minimum=0,
        )
    for key in (
        "learning_rate",
        "feature_fraction",
        "bagging_fraction",
        "lambda_l1",
        "lambda_l2",
        "min_gain_to_split",
    ):
        _expect_number(
            _expect_key(lightgbm_defaults, key, "config.lightgbm.train_defaults"),
            f"config.lightgbm.train_defaults.{key}",
        )
    _expect_int(
        _expect_key(lightgbm_defaults, "max_depth", "config.lightgbm.train_defaults"),
        "config.lightgbm.train_defaults.max_depth",
    )
    search_metric = _expect_string(
        _expect_key(lightgbm_defaults, "search_metric", "config.lightgbm.train_defaults"),
        "config.lightgbm.train_defaults.search_metric",
    )
    if search_metric not in {"mae", "rmse", "r2"}:
        raise ConfigValidationError(
            "'config.lightgbm.train_defaults.search_metric' debe ser 'mae', 'rmse' o 'r2'."
        )

    return root


@lru_cache(maxsize=1)
def _load_raw_config() -> dict[str, Any]:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fp:
            raw_config = json.load(fp)
    except FileNotFoundError as exc:
        raise SystemExit(f"No se encontro el archivo de configuracion: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Config invalido en {CONFIG_PATH}: JSON malformado ({exc.msg}, linea {exc.lineno})."
        ) from exc

    try:
        return _validate_config(raw_config)
    except ConfigValidationError as exc:
        raise SystemExit(f"Config invalido en {CONFIG_PATH}: {exc}") from exc


def load_config() -> dict[str, Any]:
    return deepcopy(_load_raw_config())


def load_section(*keys: str) -> Any:
    value: Any = _load_raw_config()
    for key in keys:
        value = value[key]
    return deepcopy(value)


def resolve_config_path(relative_path: str | Path) -> Path:
    return (SCRIPT_DIR / Path(relative_path)).resolve()
