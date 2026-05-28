from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from agents import AgentChatClient, run_all_zone_chains
from config import AgentConfig, normalize_forecast_model_name
from data_loader import build_zone_profiles, load_pipeline_data
from forecasting import ForecastResult, forecast_zone
from reporting import safe_filename, write_outputs
from zone_selection import select_zone_categories


def run_pipeline(
    *,
    data_dir: Path,
    output_dir: Path,
    config_path: Path = Path("config.yaml"),
    model: str | None = None,
    weather_file: str = "weather_airport.csv",
    dry_run: bool = False,
    force_cache: bool = False,
    max_poi_rows: int | None = None,
    forecast_start: str | None = None,
    horizon_days: int = 4,
    history_days: int = 7,
    validation_days: int = 1,
    zone_ids: str | Iterable[str] | None = None,
    forecast_model: str = "timesfm",
    timesfm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timesfm_context_hours: int = 168,
    timesfm_step_horizon: int = 24,
    timesfm_exog_cols: list[str] | None = None,
    timesfm_diurnal_blend_alpha: float = 1.0,
    timesfm_roll_actuals: bool = True,
    seasonal_diurnal_blend_alpha: float = 0.0,
    chronos_repo: str = "amazon/chronos-2",
    chronos_context_hours: int = 512,
    chronos_step_horizon: int = 24,
    chronos_diurnal_blend_alpha: float = 0.0,
    chronos_device: str = "auto",
    chronos_roll_actuals: bool = True,
    lstm_context_hours: int = 24,
    lstm_step_horizon: int = 24,
    lstm_exog_cols: list[str] | None = None,
    lstm_hidden_size: int = 64,
    lstm_num_layers: int = 1,
    lstm_epochs: int = 50,
    lstm_learning_rate: float = 0.001,
    lstm_batch_size: int = 32,
    lstm_diurnal_blend_alpha: float = 0.0,
    lstm_device: str = "auto",
    lstm_roll_actuals: bool = True,
    lstm_seed: int = 42,
    temperature: float = 0.2,
) -> dict[str, Path]:
    forecast_model = normalize_forecast_model_name(forecast_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = forecast_output_dir(output_dir, forecast_model)
    profiles = build_zone_profiles(
        data_dir,
        output_dir / "cache",
        force_cache=force_cache,
        max_poi_rows=max_poi_rows,
    )
    requested_zone_ids = normalize_zone_ids(zone_ids)
    selected_zones = (
        select_requested_zones(profiles, requested_zone_ids)
        if requested_zone_ids
        else select_zone_categories(profiles)
    )
    selected_zone_ids = selected_zones["zone_id"].astype(str).tolist()
    pipeline_data = load_pipeline_data(
        data_dir,
        profiles,
        selected_zone_ids,
        weather_file=weather_file,
    )
    contexts, forecast_results = build_contexts(
        pipeline_data=pipeline_data,
        selected_zones=selected_zones,
        forecast_start=forecast_start,
        horizon_days=horizon_days,
        history_days=history_days,
        validation_days=validation_days,
        forecast_model=forecast_model,
        timesfm_repo=timesfm_repo,
        timesfm_context_hours=timesfm_context_hours,
        timesfm_step_horizon=timesfm_step_horizon,
        timesfm_exog_cols=timesfm_exog_cols,
        timesfm_diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
        timesfm_roll_actuals=timesfm_roll_actuals,
        seasonal_diurnal_blend_alpha=seasonal_diurnal_blend_alpha,
        chronos_repo=chronos_repo,
        chronos_context_hours=chronos_context_hours,
        chronos_step_horizon=chronos_step_horizon,
        chronos_diurnal_blend_alpha=chronos_diurnal_blend_alpha,
        chronos_device=chronos_device,
        chronos_roll_actuals=chronos_roll_actuals,
        lstm_context_hours=lstm_context_hours,
        lstm_step_horizon=lstm_step_horizon,
        lstm_exog_cols=lstm_exog_cols,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_epochs=lstm_epochs,
        lstm_learning_rate=lstm_learning_rate,
        lstm_batch_size=lstm_batch_size,
        lstm_diurnal_blend_alpha=lstm_diurnal_blend_alpha,
        lstm_device=lstm_device,
        lstm_roll_actuals=lstm_roll_actuals,
        lstm_seed=lstm_seed,
    )

    if dry_run:
        client = None
    else:
        config = AgentConfig.from_file(config_path, model=model, required=True)
        if not config.api_key:
            raise RuntimeError("agent.api_key is required in config.yaml, or pass --dry-run")
        client = AgentChatClient(config)
    reports = asyncio.run(
        run_all_zone_chains(contexts, client=client, temperature=temperature)
    )
    return write_outputs(
        output_dir=run_output_dir,
        selected_zones=selected_zones,
        contexts=contexts,
        reports=reports,
        forecast_results=forecast_results,
    )


def forecast_output_dir(output_dir: Path, forecast_model: str) -> Path:
    normalized = normalize_forecast_model_name(forecast_model)
    return output_dir / safe_filename(normalized or "forecast")


def build_contexts(
    *,
    pipeline_data,
    selected_zones: pd.DataFrame,
    forecast_start: str | None,
    horizon_days: int,
    history_days: int,
    validation_days: int,
    forecast_model: str,
    timesfm_repo: str,
    timesfm_context_hours: int,
    timesfm_step_horizon: int,
    timesfm_exog_cols: list[str] | None,
    timesfm_diurnal_blend_alpha: float,
    timesfm_roll_actuals: bool,
    seasonal_diurnal_blend_alpha: float,
    chronos_repo: str,
    chronos_context_hours: int,
    chronos_step_horizon: int,
    chronos_diurnal_blend_alpha: float,
    chronos_device: str,
    chronos_roll_actuals: bool,
    lstm_context_hours: int,
    lstm_step_horizon: int,
    lstm_exog_cols: list[str] | None,
    lstm_hidden_size: int,
    lstm_num_layers: int,
    lstm_epochs: int,
    lstm_learning_rate: float,
    lstm_batch_size: int,
    lstm_diurnal_blend_alpha: float,
    lstm_device: str,
    lstm_roll_actuals: bool,
    lstm_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, ForecastResult]]:
    start = pd.Timestamp(forecast_start) if forecast_start else None
    contexts = []
    forecast_results = {}
    profiles_by_zone = pipeline_data.profiles.set_index("zone_id", drop=False)
    for row in selected_zones.to_dict(orient="records"):
        zone_id = str(row["zone_id"])
        profile = profiles_by_zone.loc[zone_id].to_dict()
        result = forecast_zone(
            zone_id=zone_id,
            category=row["category"],
            load=pipeline_data.load,
            service_price=pipeline_data.service_price,
            energy_price=pipeline_data.energy_price,
            occupancy=pipeline_data.occupancy,
            weather=pipeline_data.weather,
            profile=profile,
            forecast_start=start,
            horizon_days=horizon_days,
            history_days=history_days,
            validation_days=validation_days,
            forecast_model=forecast_model,
            timesfm_repo=timesfm_repo,
            timesfm_context_hours=timesfm_context_hours,
            timesfm_step_horizon=timesfm_step_horizon,
            timesfm_exog_cols=timesfm_exog_cols,
            timesfm_diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
            timesfm_roll_actuals=timesfm_roll_actuals,
            seasonal_diurnal_blend_alpha=seasonal_diurnal_blend_alpha,
            chronos_repo=chronos_repo,
            chronos_context_hours=chronos_context_hours,
            chronos_step_horizon=chronos_step_horizon,
            chronos_diurnal_blend_alpha=chronos_diurnal_blend_alpha,
            chronos_device=chronos_device,
            chronos_roll_actuals=chronos_roll_actuals,
            lstm_context_hours=lstm_context_hours,
            lstm_step_horizon=lstm_step_horizon,
            lstm_exog_cols=lstm_exog_cols,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_epochs=lstm_epochs,
            lstm_learning_rate=lstm_learning_rate,
            lstm_batch_size=lstm_batch_size,
            lstm_diurnal_blend_alpha=lstm_diurnal_blend_alpha,
            lstm_device=lstm_device,
            lstm_roll_actuals=lstm_roll_actuals,
            lstm_seed=lstm_seed,
        )
        forecast_results[zone_id] = result
        context = {
            **result.summary,
            "selection_reason": row["selection_reason"],
            "instructions": {
                "forecast_task": "Predict next 1-4 days of EV charging load.",
                "behavior_task": "Explain demand using POI, weather, and temporal markers.",
                "pricing_task": "Suggest service-price shift from stress and elasticity proxy.",
            },
        }
        contexts.append(context)
    return contexts, forecast_results


def normalize_zone_ids(zone_ids: str | Iterable[str] | None) -> list[str]:
    if zone_ids is None:
        return []

    raw_values = [zone_ids] if isinstance(zone_ids, str) else list(zone_ids)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in str(raw).replace(";", ",").split(","):
            zone_id = part.strip()
            if zone_id and zone_id not in seen:
                normalized.append(zone_id)
                seen.add(zone_id)
    return normalized


def select_requested_zones(profiles: pd.DataFrame, zone_ids: Iterable[str]) -> pd.DataFrame:
    requested = normalize_zone_ids(zone_ids)
    if not requested:
        raise ValueError("At least one zone id is required.")

    frame = profiles.copy()
    frame["zone_id"] = frame["zone_id"].astype(str)
    available = set(frame["zone_id"])
    missing = [zone_id for zone_id in requested if zone_id not in available]
    if missing:
        examples = ", ".join(sorted(available)[:10])
        raise ValueError(
            f"Unknown zone id(s): {', '.join(missing)}. "
            f"Available zone id examples: {examples}"
        )

    selected = frame.set_index("zone_id", drop=False).loc[requested].reset_index(drop=True)
    selected.insert(0, "category", "User-selected")
    selected.insert(2, "selection_score", None)
    selected.insert(3, "selection_reason", "User-specified zone for direct validation.")

    preferred_columns = [
        "category",
        "zone_id",
        "selection_score",
        "selection_reason",
        "longitude",
        "latitude",
        "station_count",
        "charge_count",
        "capacity_kw_proxy",
        "mean_load_kwh",
        "peak_load_kwh",
        "peak_capacity_ratio",
        "load_cv",
        "burstiness_p99_mean",
        "morning_ratio",
        "noon_ratio",
        "evening_ratio",
        "night_ratio",
        "weekend_ratio",
        "poi_food",
        "poi_business",
        "poi_lifestyle",
        "poi_total",
        "mean_service_price",
    ]
    return selected[[col for col in preferred_columns if col in selected.columns]]
