from __future__ import annotations

import json
from typing import Any


SYSTEM_MESSAGE = (
    "You are a concise EV charging demand analyst. Use only the provided UrbanEV context. Return valid JSON only, with no markdown."
)


def horizon_label(context: dict[str, Any]) -> str:
    days = context.get("forecast_horizon_days")
    try:
        value = int(days)
    except (TypeError, ValueError):
        return "configured days"
    return "1 day" if value == 1 else f"{value} days"


def grid_prompt(context: dict[str, Any]) -> str:
    horizon = horizon_label(context)
    return (
        f"""Phase A / Grid Analyst.

        Task: predict the next {horizon} of charging load for this zone. Use the baseline forecast as the numerical anchor unless the context strongly justifies a small adjustment.

        Return JSON with keys: forecast_total_kwh, forecast_peak_kwh, predicted_change_pct, grid_stress_level, forecast_summary.
        grid_stress_level must be exactly one of: Low, Medium, High, Extreme High.

        Context:{json.dumps(context, ensure_ascii=False)}"""
    )


def behavior_prompt(context: dict[str, Any], grid_report: dict[str, Any]) -> str:
    horizon = horizon_label(context)
    return (
        f"""Phase B / Behavioural Agent.

        Task: explain why demand looks this way over the next {horizon} using POI mix, weather, temporal markers, hourly forecast data, 3-hour pricing windows, and the load shape. Keep it specific and short.

        Return JSON with keys: agent_reasoning, demand_drivers, confidence.

        Context:{json.dumps(context, ensure_ascii=False)}
        Grid report:{json.dumps(grid_report, ensure_ascii=False)}"""
    )


def economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
) -> str:
    horizon = horizon_label(context)
    return (
        f"""Phase C / Market Economist.
        
        Task: prescribe service-fee shifts for the next {horizon} using each 3-hour window's load_stress_level, category, hourly forecast data, and the provided 3-hour pricing windows. Residential users are more price-sensitive; CBD and hub users are less price-sensitive. Avoid extreme changes.

        Return JSON with keys: suggested_price_shift_pct, action_label, price_rationale, price_change_windows_3h.
        price_change_windows_3h must contain one item for each context.pricing_windows_3h item, with keys: window_start, window_end, suggested_price_shift_pct, action_label, price_rationale.

        Context:\n{json.dumps(context, ensure_ascii=False)}
        Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}
        Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}"""
    )
