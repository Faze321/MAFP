from __future__ import annotations

import json
from typing import Any


SYSTEM_MESSAGE = (
    "You are a concise EV charging demand analyst. Use only the provided UrbanEV "
    "context. Return valid JSON only, with no markdown."
)


def grid_prompt(context: dict[str, Any]) -> str:
    return (
        "Phase A / Grid Analyst.\n"
        "Task: predict the next 1-4 days of charging load for this zone. "
        "Use the baseline forecast as the numerical anchor unless the context strongly "
        "justifies a small adjustment.\n\n"
        "Return JSON with keys: forecast_total_kwh, forecast_peak_kwh, "
        "predicted_change_pct, grid_stress_level, forecast_summary.\n\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False)}"
    )


def behavior_prompt(context: dict[str, Any], grid_report: dict[str, Any]) -> str:
    return (
        "Phase A / Behavioural Agent.\n"
        "Task: explain why demand looks this way using POI mix, weather, temporal "
        "markers, and the load shape. Keep it specific and short.\n\n"
        "Return JSON with keys: agent_reasoning, demand_drivers, confidence.\n\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False)}\n"
        f"Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}"
    )


def economist_prompt(
    context: dict[str, Any],
    grid_report: dict[str, Any],
    behavior_report: dict[str, Any],
) -> str:
    return (
        "Phase B / Market Economist.\n"
        "Task: prescribe a service-fee shift using load stress, category, and a simple "
        "price-elasticity rule. Residential users are more price-sensitive; CBD and hub "
        "users are less price-sensitive. Avoid extreme changes.\n\n"
        "Return JSON with keys: suggested_price_shift_pct, action_label, price_rationale.\n\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False)}\n"
        f"Grid report:\n{json.dumps(grid_report, ensure_ascii=False)}\n"
        f"Behaviour report:\n{json.dumps(behavior_report, ensure_ascii=False)}"
    )
