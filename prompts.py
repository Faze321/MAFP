from __future__ import annotations

import json
from typing import Any


SYSTEM_MESSAGE = (
    "You are a concise EV charging demand analyst. Use only the provided UrbanEV context. Return valid JSON only, with no markdown."
)


def grid_prompt(context: dict[str, Any]) -> str:
    return (
        f"""Phase A / Grid Analyst.

        Task: predict the next 1-4 days of charging load for this zone. Use the baseline forecast as the numerical anchor unless the context strongly justifies a small adjustment.

        Return JSON with keys: forecast_total_kwh, forecast_peak_kwh, predicted_change_pct, grid_stress_level, forecast_summary.

        Context:{json.dumps(context, ensure_ascii=False)}"""
    )


def behavior_prompt(context: dict[str, Any], grid_report: dict[str, Any]) -> str:
    return (
        f"""Phase A / Behavioural Agent.

        Task: explain why demand looks this way using POI mix, weather, temporal

        markers, and the load shape. Keep it specific and short.

        Return JSON with keys: agent_reasoning, demand_drivers, confidence.

        Context:{json.dumps(context, ensure_ascii=False)}
        Grid report:{json.dumps(grid_report, ensure_ascii=False)}"""
    )


def economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
) -> str:
    return (
        f"""Phase B / Market Economist.
        
        Task: prescribe a service-fee shift using load stress, category, and a simple price-elasticity rule. Residential users are more price-sensitive; CBD and hub users are less price-sensitive. Avoid extreme changes.

        Return JSON with keys: suggested_price_shift_pct, action_label, price_rationale.

        Context:\n{json.dumps(context, ensure_ascii=False)}
        Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}
        Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}"""
    )
