from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from agents import AgentChatClient, run_all_zone_chains
from config import AgentConfig, normalize_forecast_model_name
from data_loader import build_zone_3h_load_quantiles, build_zone_profiles, load_pipeline_data
from forecasting import ForecastResult, forecast_zone
from reporting import safe_filename, write_outputs
from zone_selection import select_zone_categories


STRESS_LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Extreme High": 3}


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
    ar_diurnal_blend_alpha: float = 0.0,
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
    zone_load_quantiles = build_zone_3h_load_quantiles(
        data_dir,
        output_dir / "cache",
        force_cache=force_cache,
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
        zone_load_quantiles=zone_load_quantiles,
        timesfm_repo=timesfm_repo,
        timesfm_context_hours=timesfm_context_hours,
        timesfm_step_horizon=timesfm_step_horizon,
        timesfm_exog_cols=timesfm_exog_cols,
        timesfm_diurnal_blend_alpha=timesfm_diurnal_blend_alpha,
        timesfm_roll_actuals=timesfm_roll_actuals,
        ar_diurnal_blend_alpha=ar_diurnal_blend_alpha,
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
    zone_load_quantiles: pd.DataFrame,
    timesfm_repo: str,
    timesfm_context_hours: int,
    timesfm_step_horizon: int,
    timesfm_exog_cols: list[str] | None,
    timesfm_diurnal_blend_alpha: float,
    timesfm_roll_actuals: bool,
    ar_diurnal_blend_alpha: float,
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
    stress_thresholds_by_zone = zone_load_quantiles.set_index("zone_id", drop=False)
    for row in selected_zones.to_dict(orient="records"):
        zone_id = str(row["zone_id"])
        profile = profiles_by_zone.loc[zone_id].to_dict()
        zone_stress_thresholds = load_stress_thresholds(stress_thresholds_by_zone, zone_id)
        raw_result = forecast_zone(
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
            ar_diurnal_blend_alpha=ar_diurnal_blend_alpha,
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
        pricing_windows_3h = build_pricing_windows_3h(raw_result.hourly, stress_thresholds=zone_stress_thresholds)
        summary = apply_load_quantile_stress(raw_result.summary, zone_stress_thresholds, pricing_windows_3h)
        result = ForecastResult(hourly=raw_result.hourly, summary=summary)
        forecast_results[zone_id] = result
        hourly_forecast = build_agent_hourly_data(result.hourly)
        hourly_averages = build_hourly_averages(result.hourly)
        horizon_text = format_horizon_days(summary.get("forecast_horizon_days", horizon_days))
        context = {
            **summary,
            "selection_reason": row["selection_reason"],
            "hourly_averages": hourly_averages,
            "hourly_forecast": hourly_forecast,
            "pricing_windows_3h": pricing_windows_3h,
            "instructions": {
                "forecast_task": f"Predict next {horizon_text} of EV charging load.",
                "behavior_task": "Explain demand using POI, weather, and temporal markers.",
                "pricing_task": "Suggest service-price shifts for each 3-hour pricing window.",
            },
        }
        contexts.append(context)
    return contexts, forecast_results


def load_stress_thresholds(quantiles_by_zone: pd.DataFrame, zone_id: str) -> dict[str, Any]:
    if zone_id not in quantiles_by_zone.index:
        return {
            "available": False,
            "source_file": "volume.csv",
            "window_hours": 3,
            "historical_windows": 0,
            "q50": 0.0,
            "q80": 0.0,
            "q95": 0.0,
        }
    row = quantiles_by_zone.loc[zone_id]
    return {
        "available": True,
        "source_file": str(row.get("stress_source_file", "volume.csv")),
        "window_hours": int(row.get("stress_window_hours", 3) or 3),
        "historical_windows": int(row.get("historical_3h_windows", 0) or 0),
        "q50": safe_float(row.get("load_3h_q50_kwh")),
        "q80": safe_float(row.get("load_3h_q80_kwh")),
        "q95": safe_float(row.get("load_3h_q95_kwh")),
    }


def classify_load_stress(load_value: Any, thresholds: dict[str, Any]) -> str:
    if not thresholds.get("available", True):
        return "Low"
    try:
        load = float(load_value)
    except (TypeError, ValueError):
        load = 0.0
    if load > float(thresholds.get("q95", 0.0)):
        return "Extreme High"
    if load > float(thresholds.get("q80", 0.0)):
        return "High"
    if load > float(thresholds.get("q50", 0.0)):
        return "Medium"
    return "Low"


def apply_load_quantile_stress(
    summary: dict[str, Any],
    thresholds: dict[str, Any],
    pricing_windows_3h: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(summary)
    stress_loads = [safe_float(window.get("sum_predicted_kwh")) for window in pricing_windows_3h]
    stress_load = max(stress_loads) if stress_loads else 0.0
    window_levels = [window.get("load_stress_level") for window in pricing_windows_3h]
    updated["grid_stress_level"] = max_stress_level(window_levels) if window_levels else classify_load_stress(stress_load, thresholds)
    updated["grid_stress_basis"] = "forecast_3h_sum_predicted_kwh_vs_zone_volume_csv_3h_load_quantiles"
    updated["grid_stress_load_kwh"] = stress_load
    updated["grid_stress_source_file"] = thresholds.get("source_file", "volume.csv")
    updated["grid_stress_window_hours"] = thresholds.get("window_hours", 3)
    updated["grid_stress_historical_windows"] = thresholds.get("historical_windows", 0)
    updated["grid_stress_q50_kwh"] = thresholds.get("q50", 0.0)
    updated["grid_stress_q80_kwh"] = thresholds.get("q80", 0.0)
    updated["grid_stress_q95_kwh"] = thresholds.get("q95", 0.0)
    updated.update(stress_evaluation_metrics(pricing_windows_3h))
    return updated


def stress_evaluation_metrics(pricing_windows_3h: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated_windows = []
    for window in pricing_windows_3h:
        predicted_rank = stress_rank(window.get("load_stress_level") or window.get("grid_stress_level"))
        actual_rank = stress_rank(window.get("actual_load_stress_level") or window.get("actual_grid_stress_level"))
        if predicted_rank is None or actual_rank is None:
            continue
        evaluated_windows.append((window, predicted_rank, actual_rank))

    if not evaluated_windows:
        return {
            "actual_grid_stress_level": None,
            "actual_grid_stress_load_kwh": None,
            "stress_accuracy": None,
            "miss_stress_rate": None,
            "stress_eval_windows": 0,
            "stress_miss_count": None,
        }

    correct_count = sum(1 for _, predicted_rank, actual_rank in evaluated_windows if predicted_rank == actual_rank)
    miss_count = sum(1 for _, predicted_rank, actual_rank in evaluated_windows if actual_rank > predicted_rank)
    actual_levels = [
        window.get("actual_load_stress_level") or window.get("actual_grid_stress_level")
        for window, _, _ in evaluated_windows
    ]
    actual_loads = [
        safe_float(window.get("actual_stress_load_3h_kwh") or window.get("sum_actual_kwh"))
        for window, _, _ in evaluated_windows
    ]
    total = len(evaluated_windows)
    return {
        "actual_grid_stress_level": max_stress_level(actual_levels),
        "actual_grid_stress_load_kwh": max(actual_loads) if actual_loads else None,
        "stress_accuracy": round(correct_count / total, 4),
        "miss_stress_rate": round(miss_count / total, 4),
        "stress_eval_windows": total,
        "stress_miss_count": miss_count,
    }


def build_agent_hourly_data(hourly: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "time",
        "predicted_kwh",
        "q10_kwh",
        "q50_kwh",
        "q90_kwh",
        "actual_kwh",
        "error_kwh",
        "abs_pct_error",
        "s_price",
        "e_price",
        "occupancy",
        "T",
        "U",
        "nRAIN",
        "hour",
        "is_weekend",
    ]
    frame = hourly[[col for col in columns if col in hourly.columns]].copy()
    if "time" in frame:
        frame["time"] = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return [{key: clean_context_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def build_hourly_averages(hourly: pd.DataFrame) -> dict[str, Any]:
    columns = {
        "predicted_kwh": "mean_predicted_kwh",
        "actual_kwh": "mean_actual_kwh",
        "s_price": "mean_service_price",
        "e_price": "mean_energy_price",
        "occupancy": "mean_occupancy",
        "T": "mean_temp_c",
        "U": "mean_humidity",
        "nRAIN": "mean_rain",
        "abs_pct_error": "mean_abs_pct_error",
    }
    averages: dict[str, Any] = {}
    for source_col, output_col in columns.items():
        if source_col in hourly:
            value = pd.to_numeric(hourly[source_col], errors="coerce").mean(skipna=True)
            averages[output_col] = clean_context_value(value)
    if "predicted_kwh" in hourly:
        predicted = pd.to_numeric(hourly["predicted_kwh"], errors="coerce")
        averages["peak_predicted_kwh"] = clean_context_value(predicted.max(skipna=True))
        averages["total_predicted_kwh"] = clean_context_value(predicted.sum(skipna=True))
    return averages


def build_pricing_windows_3h(
    hourly: pd.DataFrame,
    *,
    stress_thresholds: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    frame = hourly.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time").reset_index(drop=True)
    windows: list[dict[str, Any]] = []
    for idx in range(0, len(frame), 3):
        chunk = frame.iloc[idx : idx + 3]
        if chunk.empty:
            continue
        window: dict[str, Any] = {
            "window_start": chunk["time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "window_end": chunk["time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "hours": int(len(chunk)),
        }
        aggregations = {
            "predicted_kwh": ("mean_predicted_kwh", "sum_predicted_kwh", "peak_predicted_kwh"),
            "actual_kwh": ("mean_actual_kwh", "sum_actual_kwh", "peak_actual_kwh"),
        }
        for source_col, (mean_col, sum_col, max_col) in aggregations.items():
            if source_col in chunk:
                values = pd.to_numeric(chunk[source_col], errors="coerce")
                window[mean_col] = clean_context_value(values.mean(skipna=True))
                window[sum_col] = clean_context_value(values.sum(skipna=True) if values.notna().any() else None)
                window[max_col] = clean_context_value(values.max(skipna=True))
        mean_columns = {
            "s_price": "mean_service_price",
            "e_price": "mean_energy_price",
            "occupancy": "mean_occupancy",
            "T": "mean_temp_c",
            "U": "mean_humidity",
            "abs_pct_error": "mean_abs_pct_error",
        }
        for source_col, output_col in mean_columns.items():
            if source_col in chunk:
                window[output_col] = clean_context_value(pd.to_numeric(chunk[source_col], errors="coerce").mean(skipna=True))
        if "nRAIN" in chunk:
            window["total_rain"] = clean_context_value(pd.to_numeric(chunk["nRAIN"], errors="coerce").sum(skipna=True))
        if stress_thresholds is not None:
            stress_load = window.get("sum_predicted_kwh")
            stress_level = classify_load_stress(stress_load, stress_thresholds)
            actual_stress_load = window.get("sum_actual_kwh")
            actual_stress_level = (
                classify_load_stress(actual_stress_load, stress_thresholds)
                if actual_stress_load is not None
                else None
            )
            window["load_stress_level"] = stress_level
            window["grid_stress_level"] = stress_level
            window["stress_load_3h_kwh"] = clean_context_value(stress_load)
            window["actual_load_stress_level"] = actual_stress_level
            window["actual_grid_stress_level"] = actual_stress_level
            window["actual_stress_load_3h_kwh"] = clean_context_value(actual_stress_load)
            predicted_rank = stress_rank(stress_level)
            actual_rank = stress_rank(actual_stress_level)
            window["stress_correct"] = predicted_rank == actual_rank if actual_rank is not None else None
            window["stress_missed"] = actual_rank > predicted_rank if actual_rank is not None and predicted_rank is not None else None
            window["stress_source_file"] = stress_thresholds.get("source_file", "volume.csv")
            window["stress_window_hours"] = stress_thresholds.get("window_hours", 3)
            window["load_3h_q50_kwh"] = stress_thresholds.get("q50", 0.0)
            window["load_3h_q80_kwh"] = stress_thresholds.get("q80", 0.0)
            window["load_3h_q95_kwh"] = stress_thresholds.get("q95", 0.0)
        windows.append(window)
    return windows


def clean_context_value(value: Any, ndigits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, ndigits)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, ndigits)


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return round(number, 4)


def max_stress_level(levels: list[Any]) -> str:
    normalized = [str(level) for level in levels if stress_rank(level) is not None]
    if not normalized:
        return "Low"
    return max(normalized, key=lambda level: STRESS_LEVEL_ORDER[level])


def stress_rank(level: Any) -> int | None:
    return STRESS_LEVEL_ORDER.get(str(level))


def format_horizon_days(value: Any) -> str:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return "configured days"
    return "1 day" if days == 1 else f"{days} days"


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
