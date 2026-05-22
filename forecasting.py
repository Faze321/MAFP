from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    hourly: pd.DataFrame
    summary: dict[str, Any]


def forecast_zone(
    *,
    zone_id: str,
    category: str,
    load: pd.DataFrame,
    service_price: pd.DataFrame,
    occupancy: pd.DataFrame,
    weather: pd.DataFrame,
    profile: dict[str, Any],
    forecast_start: pd.Timestamp | None,
    horizon_days: int,
    history_days: int,
) -> ForecastResult:
    zone_id = str(zone_id)
    horizon_hours = horizon_days * 24
    if forecast_start is None:
        forecast_start = load["time"].max() - pd.Timedelta(hours=horizon_hours - 1)
    forecast_start = pd.Timestamp(forecast_start)
    forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours - 1)
    history_start = forecast_start - pd.Timedelta(days=history_days)
    history_end = forecast_start - pd.Timedelta(hours=1)

    series = load[["time", zone_id]].rename(columns={zone_id: "actual_kwh"}).copy()
    history = series[(series["time"] >= history_start) & (series["time"] <= history_end)]
    actual = series[(series["time"] >= forecast_start) & (series["time"] <= forecast_end)]
    if len(history) < 24:
        raise ValueError(f"Not enough history for zone {zone_id}: found {len(history)} hours")

    hourly = seasonal_naive_forecast(history, forecast_start, horizon_hours)
    hourly = hourly.merge(actual, on="time", how="left")
    hourly = add_error_columns(hourly)
    metrics = compute_forecast_metrics(hourly)

    price_summary = summarize_price(service_price, zone_id, history_start, forecast_end)
    weather_summary = summarize_weather(weather, forecast_start, forecast_end)
    occupancy_summary = summarize_occupancy(occupancy, zone_id, history_start, forecast_end)
    capacity = float(profile.get("capacity_kw_proxy", 0.0) or 0.0)
    forecast_total = float(hourly["predicted_kwh"].sum())
    forecast_peak = float(hourly["predicted_kwh"].max())
    previous_window = history.tail(horizon_hours)
    previous_total = float(previous_window["actual_kwh"].sum()) if not previous_window.empty else 0.0
    predicted_change_pct = pct_change(forecast_total, previous_total)
    actual_total = float(hourly["actual_kwh"].sum()) if hourly["actual_kwh"].notna().any() else None
    actual_peak = float(hourly["actual_kwh"].max()) if hourly["actual_kwh"].notna().any() else None

    peak_capacity_ratio = forecast_peak / capacity if capacity > 0 else 0.0
    stress_level = stress_from_ratio(peak_capacity_ratio)

    summary = {
        "zone_id": zone_id,
        "category": category,
        "history_start": history_start.isoformat(),
        "history_end": history_end.isoformat(),
        "forecast_start": forecast_start.isoformat(),
        "forecast_end": forecast_end.isoformat(),
        "forecast_total_kwh": round(forecast_total, 2),
        "forecast_peak_kwh": round(forecast_peak, 2),
        "predicted_change_pct": round(predicted_change_pct, 2),
        "actual_total_kwh": round(actual_total, 2) if actual_total is not None else None,
        "actual_peak_kwh": round(actual_peak, 2) if actual_peak is not None else None,
        "mae_kwh": metrics.get("MAE"),
        "rmse_kwh": metrics.get("RMSE"),
        "mape_pct": metrics.get("MAPE_pct"),
        "rae": metrics.get("RAE"),
        "wape_pct": metrics.get("WAPE_pct"),
        "capacity_kw_proxy": round(capacity, 2),
        "peak_capacity_ratio": round(peak_capacity_ratio, 3),
        "grid_stress_level": stress_level,
        "price": price_summary,
        "weather": weather_summary,
        "occupancy": occupancy_summary,
        "profile": compact_profile(profile),
        "daily_history_kwh": daily_totals(history, "actual_kwh"),
        "daily_forecast_kwh": daily_totals(hourly, "predicted_kwh"),
        "hourly_shape": hourly_shape(history),
        "metrics": metrics,
    }
    return ForecastResult(hourly=hourly, summary=summary)


def add_error_columns(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    if "actual_kwh" not in frame:
        frame["actual_kwh"] = np.nan
    frame["error_kwh"] = frame["actual_kwh"] - frame["predicted_kwh"]
    frame["abs_error_kwh"] = frame["error_kwh"].abs()
    frame["abs_pct_error"] = np.where(
        frame["actual_kwh"].abs() > 1e-9,
        frame["abs_error_kwh"] / frame["actual_kwh"].abs() * 100.0,
        np.nan,
    )
    return frame


def compute_forecast_metrics(hourly: pd.DataFrame) -> dict[str, float | None]:
    valid = hourly[["actual_kwh", "predicted_kwh"]].dropna()
    if valid.empty:
        return {
            "MAE": None,
            "RMSE": None,
            "MAPE_pct": None,
            "RAE": None,
            "WAPE_pct": None,
            "n": 0,
        }

    actual = valid["actual_kwh"].astype(float).to_numpy()
    predicted = valid["predicted_kwh"].astype(float).to_numpy()
    error = actual - predicted
    abs_error = np.abs(error)
    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean(error * error)))
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
    mape = float(np.nanmean(abs_error / denom) * 100.0)
    rae_denom = float(np.sum(np.abs(actual - actual.mean())))
    rae = float(np.sum(abs_error) / rae_denom) if rae_denom > 1e-9 else None
    actual_sum = float(np.sum(np.abs(actual)))
    wape = float(np.sum(abs_error) / actual_sum * 100.0) if actual_sum > 1e-9 else None

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE_pct": round(mape, 4),
        "RAE": round(rae, 4) if rae is not None else None,
        "WAPE_pct": round(wape, 4) if wape is not None else None,
        "n": int(len(valid)),
    }


def seasonal_naive_forecast(history: pd.DataFrame, forecast_start: pd.Timestamp, horizon_hours: int) -> pd.DataFrame:
    hist = history.set_index("time")["actual_kwh"].astype(float).sort_index()
    target_index = pd.date_range(forecast_start, periods=horizon_hours, freq="h")
    hourly_mean = hist.groupby(hist.index.hour).mean()
    seasonal_values = hist.reindex(target_index - pd.Timedelta(days=7)).to_numpy()
    fallback = np.array([hourly_mean.loc[t.hour] for t in target_index], dtype=float)
    predicted = np.where(np.isnan(seasonal_values), fallback, 0.72 * seasonal_values + 0.28 * fallback)

    last_24 = hist.tail(24).sum()
    mean_daily = hist.resample("D").sum().mean()
    trend_scale = 1.0 if mean_daily == 0 or np.isnan(mean_daily) else np.clip(last_24 / mean_daily, 0.75, 1.25)
    predicted = np.maximum(predicted * trend_scale, 0.0)
    return pd.DataFrame({"time": target_index, "predicted_kwh": predicted})


def summarize_price(price: pd.DataFrame, zone_id: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    frame = price[(price["time"] >= start) & (price["time"] <= end)]
    values = frame[zone_id].replace(0, np.nan).astype(float)
    return {
        "mean_service_price": finite_round(values.mean(skipna=True), 4),
        "min_service_price": finite_round(values.min(skipna=True), 4),
        "max_service_price": finite_round(values.max(skipna=True), 4),
    }


def summarize_occupancy(occupancy: pd.DataFrame, zone_id: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    frame = occupancy[(occupancy["time"] >= start) & (occupancy["time"] <= end)]
    values = frame[zone_id].astype(float)
    return {
        "mean": round(float(values.mean() or 0), 2),
        "peak": round(float(values.max() or 0), 2),
    }


def summarize_weather(weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float | int]:
    frame = weather[(weather["time"] >= start) & (weather["time"] <= end)]
    if frame.empty:
        return {"mean_temp_c": 0.0, "mean_humidity": 0.0, "rain_hours": 0, "total_rain": 0.0}
    return {
        "mean_temp_c": round(float(frame["T"].mean()), 2),
        "mean_humidity": round(float(frame["U"].mean()), 2),
        "rain_hours": int((frame["nRAIN"] > 0).sum()),
        "total_rain": round(float(frame["nRAIN"].sum()), 2),
    }


def daily_totals(frame: pd.DataFrame, value_col: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    values = frame.set_index("time")[value_col].astype(float).resample("D").sum()
    return [{"date": idx.date().isoformat(), "kwh": round(float(value), 2)} for idx, value in values.items()]


def hourly_shape(history: pd.DataFrame) -> dict[str, float]:
    hist = history.copy()
    hist["hour"] = hist["time"].dt.hour
    means = hist.groupby("hour")["actual_kwh"].mean()
    return {
        "morning_7_10": round(float(means.loc[7:10].mean()), 2),
        "noon_11_14": round(float(means.loc[11:14].mean()), 2),
        "evening_17_22": round(float(means.loc[17:22].mean()), 2),
        "night_20_6": round(float(pd.concat([means.loc[20:23], means.loc[0:6]]).mean()), 2),
    }


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "station_count",
        "charge_count",
        "capacity_kw_proxy",
        "mean_load_kwh",
        "peak_load_kwh",
        "peak_capacity_ratio",
        "load_cv",
        "burstiness_p99_mean",
        "morning_ratio",
        "evening_ratio",
        "night_ratio",
        "weekend_ratio",
        "poi_food",
        "poi_business",
        "poi_lifestyle",
        "poi_total",
        "mean_service_price",
    ]
    return {key: round_float(profile.get(key)) for key in keys}


def round_float(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def finite_round(value: Any, ndigits: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(number) or np.isinf(number):
        return 0.0
    return round(number, ndigits)


def stress_from_ratio(ratio: float) -> str:
    if ratio >= 0.9:
        return "Critical"
    if ratio >= 0.7:
        return "Moderate"
    return "Low"


def pct_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (current / baseline - 1.0) * 100.0
