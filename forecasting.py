from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_TIMEFM_MODEL_CACHE: dict[tuple[str, int, int], Any] = {}
DEFAULT_TIMEFM_EXOG_COLS = ["T", "U", "nRAIN", "e_price", "is_weekend", "temp_price_idx"]


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
    energy_price: pd.DataFrame,
    occupancy: pd.DataFrame,
    weather: pd.DataFrame,
    profile: dict[str, Any],
    forecast_start: pd.Timestamp | None,
    horizon_days: int,
    history_days: int,
    validation_days: int = 1,
    forecast_model: str = "timefm",
    timefm_repo: str = "google/timesfm-2.5-200m-pytorch",
    timefm_context_hours: int = 168,
    timefm_step_horizon: int = 24,
    timefm_exog_cols: list[str] | None = None,
    timefm_diurnal_blend_alpha: float = 1.0,
    timefm_roll_actuals: bool = True,
) -> ForecastResult:
    zone_id = str(zone_id)
    horizon_hours = horizon_days * 24
    if forecast_start is None:
        forecast_start = load["time"].max() - pd.Timedelta(hours=horizon_hours - 1)
    forecast_start = pd.Timestamp(forecast_start)
    forecast_end = forecast_start + pd.Timedelta(hours=horizon_hours - 1)
    validation_hours = max(0, int(validation_days) * 24)
    validation_start = forecast_start - pd.Timedelta(hours=validation_hours) if validation_hours else forecast_start
    validation_end = forecast_start - pd.Timedelta(hours=1)
    history_start = validation_start - pd.Timedelta(days=history_days)
    history_end = validation_start - pd.Timedelta(hours=1)

    zone_frame = build_zone_model_frame(load, service_price, energy_price, occupancy, weather, zone_id)
    history = zone_frame[(zone_frame["time"] >= history_start) & (zone_frame["time"] <= history_end)]
    validation = zone_frame[(zone_frame["time"] >= validation_start) & (zone_frame["time"] <= validation_end)]
    actual = zone_frame[(zone_frame["time"] >= forecast_start) & (zone_frame["time"] <= forecast_end)][["time", "actual_kwh"]]
    if len(history) < 24:
        raise ValueError(f"Not enough history for zone {zone_id}: found {len(history)} hours")

    hourly = forecast_load(
        history,
        validation,
        zone_frame,
        forecast_start,
        horizon_hours,
        model_name=forecast_model,
        timefm_repo=timefm_repo,
        timefm_context_hours=timefm_context_hours,
        timefm_step_horizon=timefm_step_horizon,
        timefm_exog_cols=timefm_exog_cols or DEFAULT_TIMEFM_EXOG_COLS,
        timefm_diurnal_blend_alpha=timefm_diurnal_blend_alpha,
        timefm_roll_actuals=timefm_roll_actuals,
    )
    forecast_attrs = dict(hourly.attrs)
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
        "validation_start": validation_start.isoformat() if validation_hours else None,
        "validation_end": validation_end.isoformat() if validation_hours else None,
        "forecast_start": forecast_start.isoformat(),
        "forecast_end": forecast_end.isoformat(),
        "forecast_model": forecast_model,
        "timefm_covariates": timefm_exog_cols or DEFAULT_TIMEFM_EXOG_COLS,
        "timefm_diurnal_blend_alpha": round(float(timefm_diurnal_blend_alpha), 4),
        "timefm_roll_actuals": bool(timefm_roll_actuals),
        "calibration": forecast_attrs.get("calibration"),
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


def forecast_load(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    model_name: str,
    timefm_repo: str,
    timefm_context_hours: int,
    timefm_step_horizon: int,
    timefm_exog_cols: list[str],
    timefm_diurnal_blend_alpha: float,
    timefm_roll_actuals: bool,
) -> pd.DataFrame:
    normalized = model_name.strip().lower().replace("-", "_")
    if normalized in {"seasonal", "seasonal_naive", "naive"}:
        return seasonal_naive_forecast(history, forecast_start, horizon_hours)
    if normalized == "timefm":
        return timefm_forecast(
            history,
            validation,
            full_frame,
            forecast_start,
            horizon_hours,
            repo=timefm_repo,
            context_hours=timefm_context_hours,
            step_horizon=timefm_step_horizon,
            exog_cols=timefm_exog_cols,
            diurnal_blend_alpha=timefm_diurnal_blend_alpha,
            roll_actuals=timefm_roll_actuals,
        )
    raise ValueError(f"Unsupported forecast_model: {model_name}")


def timefm_forecast(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    forecast_start: pd.Timestamp,
    horizon_hours: int,
    *,
    repo: str,
    context_hours: int,
    step_horizon: int,
    exog_cols: list[str],
    diurnal_blend_alpha: float,
    roll_actuals: bool,
) -> pd.DataFrame:
    context_load = history["actual_kwh"].astype(float).to_numpy(dtype=np.float64)
    if len(context_load) == 0:
        raise ValueError("TimeFM requires at least one historical value")

    step = max(1, int(step_horizon))
    compile_horizon = max(step, 32)
    model = load_timefm_model(repo, context_hours, compile_horizon)
    rolling_load = context_load.copy()
    rolling_exog = build_exog_matrix(history, exog_cols)
    diurnal_by_hour = build_diurnal_profile(history)

    calibration: dict[str, Any] = {
        "enabled": False,
        "bias_mean": 0.0,
        "bias_max_abs": 0.0,
        "metrics": None,
    }
    bias_vec = np.zeros(step, dtype=np.float64)
    if not validation.empty:
        val_horizon = min(step, len(validation))
        val_exog = align_exog(build_exog_matrix(validation, exog_cols), step)
        val_raw, _, _ = run_timefm_prediction(
            model,
            rolling_load,
            rolling_exog,
            val_exog,
            step,
            context_hours,
            exog_cols,
        )
        val_times = pd.to_datetime(validation["time"].iloc[:val_horizon])
        val_blended = blend_with_diurnal(
            val_raw[:val_horizon],
            diurnal_for_times(val_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        val_actual = validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)[:val_horizon]
        bias_vec = align_vector(val_blended - val_actual, step)
        val_cmp = pd.DataFrame({"actual_kwh": val_actual, "predicted_kwh": val_blended})
        calibration = {
            "enabled": True,
            "bias_mean": round(float(np.nanmean(bias_vec)), 4),
            "bias_max_abs": round(float(np.nanmax(np.abs(bias_vec))), 4),
            "metrics": compute_forecast_metrics(val_cmp),
        }
        rolling_load = np.concatenate([rolling_load, validation["actual_kwh"].astype(float).to_numpy(dtype=np.float64)])
        rolling_exog = np.vstack([rolling_exog, build_exog_matrix(validation, exog_cols)])

    rows: list[pd.DataFrame] = []
    remaining = horizon_hours
    offset = 0
    while remaining > 0:
        chunk_horizon = min(step, remaining)
        chunk_start = forecast_start + pd.Timedelta(hours=offset)
        chunk_times = pd.date_range(chunk_start, periods=chunk_horizon, freq="h")
        chunk_frame = (
            full_frame.set_index("time")
            .reindex(chunk_times)
            .rename_axis("time")
            .reset_index()
        )
        chunk_exog = align_exog(build_exog_matrix(chunk_frame, exog_cols), chunk_horizon)
        raw, q10, q90 = run_timefm_prediction(
            model,
            rolling_load,
            rolling_exog,
            chunk_exog,
            chunk_horizon,
            context_hours,
            exog_cols,
        )
        bias = align_vector(bias_vec, chunk_horizon)
        bias_corrected = np.clip(raw[:chunk_horizon] - bias, 0, None)
        point = blend_with_diurnal(
            bias_corrected,
            diurnal_for_times(chunk_times, diurnal_by_hour),
            diurnal_blend_alpha,
        )
        q10 = np.clip(q10[:chunk_horizon] - bias, 0, None)
        q90 = np.clip(q90[:chunk_horizon] - bias, 0, None)

        rows.append(
            pd.DataFrame(
                {
                    "time": chunk_times,
                    "raw_predicted_kwh": raw[:chunk_horizon],
                    "bias_corrected_kwh": bias_corrected,
                    "predicted_kwh": point,
                    "q10_kwh": q10,
                    "q50_kwh": point,
                    "q90_kwh": q90,
                }
            )
        )

        actual_chunk = chunk_frame.get("actual_kwh")
        if roll_actuals and actual_chunk is not None and actual_chunk.notna().all():
            roll_values = actual_chunk.astype(float).to_numpy(dtype=np.float64)
        else:
            roll_values = point
        rolling_load = np.concatenate([rolling_load, roll_values])
        rolling_exog = np.vstack([rolling_exog, chunk_exog])
        remaining -= chunk_horizon
        offset += chunk_horizon

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["time", "predicted_kwh"])
    result.attrs["calibration"] = calibration
    return result


def load_timefm_model(repo: str, context_hours: int, max_horizon: int):
    key = (repo, int(context_hours), int(max_horizon))
    if key in _TIMEFM_MODEL_CACHE:
        return _TIMEFM_MODEL_CACHE[key]

    try:
        from timesfm import ForecastConfig, TimesFM_2p5_200M_torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "timesfm is required for forecast_model: timefm. "
            "Install the official google-research/timesfm package and make sure "
            "this Python environment can import it."
        ) from exc

    model = TimesFM_2p5_200M_torch.from_pretrained(repo)
    model.compile(
        ForecastConfig(
            max_context=int(context_hours),
            max_horizon=int(max_horizon),
            normalize_inputs=True,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )
    _TIMEFM_MODEL_CACHE[key] = model
    return model


def run_timefm_prediction(
    model,
    context_load: np.ndarray,
    exog_context: np.ndarray | None,
    exog_horizon: np.ndarray | None,
    horizon: int,
    context_hours: int,
    exog_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = context_load[-context_hours:].astype(np.float64)
    if exog_context is not None and exog_horizon is not None and hasattr(model, "forecast_with_covariates"):
        exog_ctx = exog_context[-len(ctx) :].astype(np.float64)
        dyn_cov = {}
        for idx, col in enumerate(exog_cols):
            if idx < exog_ctx.shape[1] and idx < exog_horizon.shape[1]:
                combined = np.concatenate([exog_ctx[:, idx], exog_horizon[:, idx]])
                dyn_cov[col] = [combined.tolist()]
        try:
            result = model.forecast_with_covariates(
                inputs=[ctx.tolist()],
                dynamic_numerical_covariates=dyn_cov if dyn_cov else None,
                xreg_mode="xreg + timesfm",
                normalize_xreg_target_per_input=True,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TimeFM covariate forecasting requires the xreg dependencies. "
                "Install torch, jax, jaxlib, scikit-learn, and the official "
                "google-research/timesfm package."
            ) from exc
    else:
        result = model.forecast(horizon=int(horizon), inputs=[ctx])
    return parse_timefm_result(result, horizon)


def parse_timefm_result(result: Any, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(result, tuple):
        point_out, quantile_out = result
    else:
        arr = np.asarray(result)
        if arr.ndim == 3 and arr.shape[-1] > 5:
            point_out = arr[:, :, 5]
            quantile_out = arr
        else:
            point_out = arr
            quantile_out = None

    point = first_series(point_out, horizon)
    if quantile_out is None:
        q10 = np.full(horizon, np.nan, dtype=np.float64)
        q90 = np.full(horizon, np.nan, dtype=np.float64)
    else:
        quantiles = np.asarray(quantile_out, dtype=np.float64)
        if quantiles.ndim == 3:
            q10 = quantiles[0, :horizon, 0]
            q90_idx = min(8, quantiles.shape[-1] - 1)
            q90 = quantiles[0, :horizon, q90_idx]
        else:
            q10 = np.full(horizon, np.nan, dtype=np.float64)
            q90 = np.full(horizon, np.nan, dtype=np.float64)
    return np.clip(point, 0, None), np.clip(q10, 0, None), np.clip(q90, 0, None)


def first_series(values: Any, horizon: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:horizon]
    return arr[0, :horizon]


def build_zone_model_frame(
    load: pd.DataFrame,
    service_price: pd.DataFrame,
    energy_price: pd.DataFrame,
    occupancy: pd.DataFrame,
    weather: pd.DataFrame,
    zone_id: str,
) -> pd.DataFrame:
    frame = load[["time", zone_id]].rename(columns={zone_id: "actual_kwh"}).copy()
    frame = frame.merge(
        energy_price[["time", zone_id]].rename(columns={zone_id: "e_price"}),
        on="time",
        how="left",
    )
    frame = frame.merge(
        service_price[["time", zone_id]].rename(columns={zone_id: "s_price"}),
        on="time",
        how="left",
    )
    frame = frame.merge(
        occupancy[["time", zone_id]].rename(columns={zone_id: "occupancy"}),
        on="time",
        how="left",
    )
    frame = frame.merge(weather, on="time", how="left")
    frame = frame.sort_values("time").reset_index(drop=True)
    frame["hour"] = frame["time"].dt.hour
    frame["is_weekend"] = frame["time"].dt.dayofweek.isin([5, 6]).astype(float)
    if "T" in frame and "e_price" in frame:
        frame["temp_price_idx"] = frame["T"].astype(float) * frame["e_price"].astype(float)
    else:
        frame["temp_price_idx"] = 0.0
    return frame


def build_exog_matrix(frame: pd.DataFrame, exog_cols: list[str]) -> np.ndarray:
    if not exog_cols:
        return np.empty((len(frame), 0), dtype=np.float64)
    exog = frame.reindex(columns=exog_cols).copy()
    exog = exog.apply(pd.to_numeric, errors="coerce")
    exog = exog.ffill().bfill()
    means = exog.mean(numeric_only=True)
    exog = exog.fillna(means).fillna(0.0)
    return exog.to_numpy(dtype=np.float64)


def align_exog(matrix: np.ndarray, target: int) -> np.ndarray:
    if len(matrix) == target:
        return matrix
    if len(matrix) > target:
        return matrix[:target]
    if len(matrix) == 0:
        return np.zeros((target, 0), dtype=np.float64)
    pad = np.tile(matrix[-1], (target - len(matrix), 1))
    return np.vstack([matrix, pad])


def align_vector(values: np.ndarray, target: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == target:
        return arr
    if len(arr) > target:
        return arr[:target]
    if len(arr) == 0:
        return np.zeros(target, dtype=np.float64)
    return np.concatenate([arr, np.repeat(arr[-1], target - len(arr))])


def build_diurnal_profile(history: pd.DataFrame) -> pd.Series:
    base = history.copy()
    base["hour"] = base["time"].dt.hour
    fallback = float(base["actual_kwh"].mean()) if not base.empty else 0.0
    return base.groupby("hour")["actual_kwh"].mean().reindex(range(24), fill_value=fallback)


def diurnal_for_times(times: pd.Series | pd.DatetimeIndex, diurnal_by_hour: pd.Series) -> np.ndarray:
    index = pd.DatetimeIndex(times)
    return np.asarray([diurnal_by_hour.loc[int(ts.hour)] for ts in index], dtype=np.float64)


def blend_with_diurnal(fc: np.ndarray, diurnal: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0 or len(fc) == 0:
        return np.clip(fc, 0, None)
    fc_mean = float(np.nanmean(fc))
    diurnal_mean = float(np.nanmean(diurnal))
    scaled = diurnal * (fc_mean / diurnal_mean) if abs(diurnal_mean) > 1e-9 else diurnal
    return np.clip((1.0 - alpha) * fc + alpha * scaled, 0, None)


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
