from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    data_dir: str = "data"
    output_dir: str = "output"
    weather_file: str = "weather_airport.csv"
    dry_run: bool = False
    force_cache: bool = False
    max_poi_rows: int | None = None
    forecast_start: str | None = None
    horizon_days: int = 4
    history_days: int = 7
    validation_days: int = 1
    zone_ids: list[str] | None = None
    forecast_model: str = "timefm"
    timefm_repo: str = "google/timesfm-2.5-200m-pytorch"
    timefm_context_hours: int = 168
    timefm_step_horizon: int = 24
    timefm_exog_cols: list[str] | None = None
    timefm_diurnal_blend_alpha: float = 1.0
    timefm_roll_actuals: bool = True
    temperature: float = 0.2

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RunConfig":
        settings = raw or {}
        zone_ids = settings.get("zone_ids", settings.get("zones"))
        horizon_days = optional_int(settings.get("horizon_days"))
        history_days = optional_int(settings.get("history_days"))
        validation_days = optional_int(settings.get("validation_days"))
        timefm_context_hours = optional_int(settings.get("timefm_context_hours"))
        timefm_step_horizon = optional_int(settings.get("timefm_step_horizon"))
        timefm_diurnal_blend_alpha = optional_float(settings.get("timefm_diurnal_blend_alpha"))
        temperature = optional_float(settings.get("temperature"))
        return cls(
            data_dir=optional_str(settings.get("data_dir")) or "data",
            output_dir=optional_str(settings.get("output_dir")) or "output",
            weather_file=optional_str(settings.get("weather_file")) or "weather_airport.csv",
            dry_run=optional_bool(settings.get("dry_run"), False),
            force_cache=optional_bool(settings.get("force_cache"), False),
            max_poi_rows=optional_int(settings.get("max_poi_rows")),
            forecast_start=optional_str(settings.get("forecast_start")),
            horizon_days=horizon_days if horizon_days is not None else 4,
            history_days=history_days if history_days is not None else 7,
            validation_days=validation_days if validation_days is not None else 1,
            zone_ids=normalize_zone_id_list(zone_ids),
            forecast_model=optional_str(settings.get("forecast_model")) or "timefm",
            timefm_repo=optional_str(settings.get("timefm_repo")) or "google/timesfm-2.5-200m-pytorch",
            timefm_context_hours=timefm_context_hours if timefm_context_hours is not None else 168,
            timefm_step_horizon=timefm_step_horizon if timefm_step_horizon is not None else 24,
            timefm_exog_cols=normalize_zone_id_list(settings.get("timefm_exog_cols")),
            timefm_diurnal_blend_alpha=(
                timefm_diurnal_blend_alpha if timefm_diurnal_blend_alpha is not None else 1.0
            ),
            timefm_roll_actuals=optional_bool(settings.get("timefm_roll_actuals"), True),
            temperature=temperature if temperature is not None else 0.2,
        )


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None
    base_url: str
    model: str
    http_referer: str | None = None
    title: str | None = None
    timeout_seconds: float = 90.0

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        model: str | None = None,
        required: bool = True,
    ) -> "AgentConfig":
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Config file not found: {path}. Copy config.example.yaml to config.yaml."
                )
            return cls.default(model=model)

        raw = read_config_mapping(path)
        settings = raw.get("agent")
        if not isinstance(settings, dict):
            raise ValueError('Config key "agent" must contain a mapping')

        return cls(
            api_key=optional_str(settings.get("api_key")),
            base_url=optional_str(settings.get("base_url")) or "https://openrouter.ai/api/v1",
            model=model or optional_str(settings.get("model")) or "meta-llama/llama-3.1-8b-instruct",
            http_referer=optional_str(settings.get("http_referer")),
            title=optional_str(settings.get("title")) or "MAPF UrbanEV",
            timeout_seconds=float(settings.get("timeout_seconds", 90)),
        )

    @classmethod
    def default(cls, *, model: str | None = None) -> "AgentConfig":
        return cls(
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
            model=model or "meta-llama/llama-3.1-8b-instruct",
            title="MAPF UrbanEV",
        )


@dataclass(frozen=True)
class AppConfig:
    agent: AgentConfig
    run: RunConfig

    @classmethod
    def from_file(cls, path: Path, *, required: bool = False) -> "AppConfig":
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Config file not found: {path}. Copy config.example.yaml to config.yaml."
                )
            return cls(agent=AgentConfig.default(), run=RunConfig())

        raw = read_config_mapping(path)
        agent_settings = raw.get("agent")
        if not isinstance(agent_settings, dict):
            raise ValueError('Config key "agent" must contain a mapping')
        run_settings = raw.get("run", {})
        if run_settings is None:
            run_settings = {}
        if not isinstance(run_settings, dict):
            raise ValueError('Config key "run" must contain a mapping')
        return cls(
            agent=AgentConfig.from_file(path, required=required),
            run=RunConfig.from_mapping(run_settings),
        )


def read_config_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    value = read_yaml_mapping(text)
    if not isinstance(value, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return value


def read_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return parse_simple_yaml(text)

    value = yaml.safe_load(text) or {}
    if not isinstance(value, dict):
        raise ValueError("YAML config must contain a mapping")
    return value


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        line = strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            key, value = split_yaml_key_value(stripped)
            current_list_key = None
            if value is None:
                root[key] = {}
                current = root[key]
            else:
                root[key] = parse_yaml_scalar(value)
                current = root
            continue

        if indent == 2 and stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"List item without a key: {raw_line}")
            current[current_list_key].append(parse_yaml_scalar(stripped[2:].strip()))
            continue

        if indent == 2:
            key, value = split_yaml_key_value(stripped)
            if value is None:
                current[key] = []
                current_list_key = key
            else:
                current[key] = parse_yaml_scalar(value)
                current_list_key = None
            continue

        if indent == 4 and stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"List item without a key: {raw_line}")
            current[current_list_key].append(parse_yaml_scalar(stripped[2:].strip()))
            continue

        raise ValueError(f"Unsupported YAML line: {raw_line}")

    return root


def split_yaml_key_value(line: str) -> tuple[str, str | None]:
    if ":" not in line:
        raise ValueError(f"Expected key/value line: {line}")
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    return key, value if value else None


def parse_yaml_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_yaml_scalar(part.strip()) for part in inner.split(",")]

    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def strip_yaml_comment(line: str) -> str:
    quote: str | None = None
    for idx, char in enumerate(line):
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        elif char == "#" and quote is None:
            return line[:idx]
    return line


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def optional_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean config value: {value!r}")


def normalize_zone_id_list(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    raw_values = value if isinstance(value, list) else [value]
    zone_ids: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            zone_id = part.strip()
            if zone_id and zone_id not in seen:
                zone_ids.append(zone_id)
                seen.add(zone_id)
    return zone_ids or None
