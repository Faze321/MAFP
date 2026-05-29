from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from config import AgentConfig
from prompts import SYSTEM_MESSAGE, behavior_prompt, economist_prompt, grid_prompt


GRID_STRESS_LEVELS = ("Low", "Medium", "High", "Extreme High")
GRID_STRESS_LEVEL_BY_KEY = {level.lower(): level for level in GRID_STRESS_LEVELS}
GRID_STRESS_LEVEL_BY_KEY.update(
    {
        "moderate": "Medium",
        "critical": "Extreme High",
        "critica": "Extreme High",
        "extrame high": "Extreme High",
        "extreme_high": "Extreme High",
    }
)


class ChatClient(Protocol):
    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        ...


@dataclass
class AgentChatClient:
    config: AgentConfig

    def __post_init__(self) -> None:
        if not self.config.api_key:
            raise ValueError("agent.api_key is required when dry-run is disabled")
        from openai import AsyncOpenAI

        headers = {}
        # if self.config.http_referer:
        #     headers["HTTP-Referer"] = self.config.http_referer
        # if self.config.title:
        #     headers["X-Title"] = self.config.title
        #     headers["X-OpenRouter-Title"] = self.config.title
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            default_headers=headers or None,
            timeout=self.config.timeout_seconds,
        )

    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content or "{}"
        return extract_json_object(content)


@dataclass
class DryRunChatClient:
    async def complete_json(self, prompt: str, *, temperature: float) -> dict[str, Any]:
        raise RuntimeError("DryRunChatClient should not receive raw prompts")


async def run_zone_chain(
    context: dict[str, Any],
    *,
    client: ChatClient | None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    if client is None:
        return heuristic_zone_chain(context)

    grid = merge_grid_fallback(
        await client.complete_json(grid_prompt(context), temperature=temperature),
        context,
    )
    behavior = merge_behavior_fallback(
        await client.complete_json(behavior_prompt(context, grid), temperature=temperature),
        context,
    )
    economist = merge_economist_fallback(
        await client.complete_json(economist_prompt(context, grid, behavior), temperature=temperature),
        context,
    )
    return combine_reports(context, grid, behavior, economist, source="model")


async def run_all_zone_chains(
    contexts: list[dict[str, Any]],
    *,
    client: ChatClient | None,
    temperature: float = 0.2,
) -> list[dict[str, Any]]:
    tasks = [run_zone_chain(context, client=client, temperature=temperature) for context in contexts]
    return await asyncio.gather(*tasks)


def heuristic_zone_chain(context: dict[str, Any]) -> dict[str, Any]:
    grid = heuristic_grid(context)
    behavior = heuristic_behavior(context)
    economist = heuristic_economist(context, grid)
    return combine_reports(context, grid, behavior, economist, source="dry-run")


def heuristic_grid(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "forecast_total_kwh": context["forecast_total_kwh"],
        "forecast_peak_kwh": context["forecast_peak_kwh"],
        "predicted_change_pct": context["predicted_change_pct"],
        "grid_stress_level": normalize_grid_stress_level(context.get("grid_stress_level"), "Low"),
        "forecast_summary": (
            f"{context['category']} zone forecast is anchored to the prior-week hourly "
            f"shape with a {context['predicted_change_pct']:+.1f}% change versus the "
            "recent comparable window."
        ),
    }


def heuristic_behavior(context: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    weather = context["weather"]
    shape = context["hourly_shape"]
    windows = context.get("pricing_windows_3h") or []
    horizon_days = context.get("forecast_horizon_days", "configured")
    forecast_start = str(context.get("forecast_start") or "the forecast start")
    forecast_end = str(context.get("forecast_end") or "the forecast end")
    change = as_float(context.get("predicted_change_pct"), 0)
    drivers = []
    high_windows = [
        window
        for window in windows
        if normalize_grid_stress_level(window.get("load_stress_level") or window.get("grid_stress_level"), "Low")
        in {"High", "Extreme High"}
    ]
    peak_window = max(windows, key=lambda window: as_float(window.get("sum_predicted_kwh"), 0), default=None)
    if high_windows:
        drivers.append(f"{len(high_windows)} high-stress 3-hour windows")
    if peak_window:
        drivers.append(
            "peak window "
            f"{peak_window.get('window_start')} to {peak_window.get('window_end')} "
            f"at {as_float(peak_window.get('sum_predicted_kwh'), 0):.1f} kWh"
        )
    if abs(change) >= 5:
        drivers.append(f"{change:+.1f}% load change versus comparable history")
    else:
        drivers.append("stable total load versus comparable history")
    if weather["rain_hours"] > 0:
        drivers.append(f"{weather['rain_hours']} rainy hours")
    if shape["night_20_6"] >= shape["morning_7_10"]:
        drivers.append("night plateau")
    if shape["evening_17_22"] >= shape["morning_7_10"]:
        drivers.append("evening lift")
    profile = context.get("profile") or {}
    poi_total = profile.get("poi_total", 0)
    return {
        "agent_reasoning": (
            f"For the {horizon_days}-day window from {forecast_start} to {forecast_end}, "
            f"{category} demand is best explained by {', '.join(drivers)} and the "
            f"local POI mix of {poi_total} assigned POIs."
        ),
        "demand_drivers": drivers,
        "confidence": "medium",
    }


def heuristic_economist(context: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    stress = grid["grid_stress_level"]
    change = float(grid["predicted_change_pct"])
    shift, label = base_price_shift(stress, category, change)
    window_shifts = build_heuristic_price_windows(context, grid)
    return {
        "suggested_price_shift_pct": shift,
        "action_label": label,
        "price_rationale": (
            f"{stress} stress with {change:+.1f}% expected load change; "
            f"category elasticity proxy is {category}."
        ),
        "price_change_windows_3h": window_shifts,
    }


def combine_reports(
    context: dict[str, Any],
    grid: dict[str, Any],
    behavior: dict[str, Any],
    economist: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "category": context["category"],
        "zone_id": context["zone_id"],
        "predicted_load_kwh": as_float(grid.get("forecast_total_kwh"), context["forecast_total_kwh"]),
        "predicted_peak_kwh": as_float(grid.get("forecast_peak_kwh"), context["forecast_peak_kwh"]),
        "predicted_change_pct": as_float(grid.get("predicted_change_pct"), context["predicted_change_pct"]),
        "actual_load_kwh": context.get("actual_total_kwh"),
        "mae_kwh": context.get("mae_kwh"),
        "rmse_kwh": context.get("rmse_kwh"),
        "mape_pct": context.get("mape_pct"),
        "rae": context.get("rae"),
        "wape_pct": context.get("wape_pct"),
        "grid_stress_level": normalize_grid_stress_level(
            grid.get("grid_stress_level"),
            context["grid_stress_level"],
        ),
        "agent_reasoning": str(behavior.get("agent_reasoning") or ""),
        "suggested_price_shift_pct": as_float(economist.get("suggested_price_shift_pct"), 0),
        "action_label": str(economist.get("action_label") or ""),
        "price_rationale": str(economist.get("price_rationale") or ""),
        "price_change_windows_3h": normalize_price_windows(
            economist.get("price_change_windows_3h"),
            context.get("pricing_windows_3h", []),
        ),
        "source": source,
    }


def merge_grid_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_grid(context)
    merged = {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}
    merged["grid_stress_level"] = normalize_grid_stress_level(
        merged.get("grid_stress_level"),
        fallback["grid_stress_level"],
    )
    return merged


def merge_behavior_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_behavior(context)
    return {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}


def merge_economist_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_economist(context, heuristic_grid(context))
    return {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def as_float(value: Any, fallback: float) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return round(float(fallback), 2)


def base_price_shift(stress: str, category: str, change: float) -> tuple[int, str]:
    if stress == "Extreme High" and category == "Residential":
        return 15, "Peak deflection fee"
    if stress == "Extreme High":
        return 10, "Congestion fee"
    if stress == "High":
        return 8, "High-load premium"
    if stress == "Medium" and change > 10:
        return 5, "Medium peak premium"
    if stress == "Low" and change < -10:
        return -5, "Utilization incentive"
    return 0, "Hold price"


def build_heuristic_price_windows(
    context: dict[str, Any],
    grid: dict[str, Any],
) -> list[dict[str, Any]]:
    windows = context.get("pricing_windows_3h") or []
    averages = context.get("hourly_averages") or {}
    mean_load = as_float(averages.get("mean_predicted_kwh"), 0)
    mean_energy_price = as_float(averages.get("mean_energy_price"), 0)
    stress = normalize_grid_stress_level(grid.get("grid_stress_level"), context.get("grid_stress_level"))
    category = str(context.get("category") or "")
    change = as_float(grid.get("predicted_change_pct"), context.get("predicted_change_pct", 0))
    results = []
    for window in windows:
        window_stress = normalize_grid_stress_level(
            window.get("load_stress_level") or window.get("grid_stress_level"),
            stress,
        )
        window_base_shift, _ = base_price_shift(window_stress, category, change)
        window_load = as_float(window.get("mean_predicted_kwh"), mean_load)
        window_energy_price = as_float(window.get("mean_energy_price"), mean_energy_price)
        adjustment = 0
        if mean_load > 0:
            load_ratio = window_load / mean_load
            if load_ratio >= 1.15:
                adjustment += 3
            elif load_ratio <= 0.85:
                adjustment -= 3
        if mean_energy_price > 0:
            energy_ratio = window_energy_price / mean_energy_price
            if energy_ratio >= 1.08:
                adjustment += 2
            elif energy_ratio <= 0.92:
                adjustment -= 1
        shift = int(max(-15, min(15, round(window_base_shift + adjustment))))
        if shift > window_base_shift:
            label = "Raise price"
        elif shift < window_base_shift:
            label = "Reduce price"
        else:
            label = "Hold window price"
        results.append(
            {
                "window_start": window.get("window_start"),
                "window_end": window.get("window_end"),
                "sum_predicted_kwh": window.get("sum_predicted_kwh"),
                "mean_predicted_kwh": window.get("mean_predicted_kwh"),
                "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                "stress_load_3h_kwh": window.get("stress_load_3h_kwh"),
                "load_3h_q50_kwh": window.get("load_3h_q50_kwh"),
                "load_3h_q80_kwh": window.get("load_3h_q80_kwh"),
                "load_3h_q95_kwh": window.get("load_3h_q95_kwh"),
                "suggested_price_shift_pct": shift,
                "action_label": label,
                "price_rationale": (
                    f"{window_stress} 3-hour stress; mean load {window_load:.2f} kWh "
                    f"vs horizon mean {mean_load:.2f} kWh."
                ),
            }
        )
    return results


def normalize_price_windows(value: Any, fallback_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        value = []
    normalized = []
    for idx, fallback in enumerate(fallback_windows):
        item = value[idx] if idx < len(value) and isinstance(value[idx], dict) else {}
        normalized.append(
            {
                "window_start": item.get("window_start") or fallback.get("window_start"),
                "window_end": item.get("window_end") or fallback.get("window_end"),
                "sum_predicted_kwh": fallback.get("sum_predicted_kwh"),
                "mean_predicted_kwh": fallback.get("mean_predicted_kwh"),
                "load_stress_level": fallback.get("load_stress_level") or fallback.get("grid_stress_level"),
                "stress_load_3h_kwh": fallback.get("stress_load_3h_kwh"),
                "load_3h_q50_kwh": fallback.get("load_3h_q50_kwh"),
                "load_3h_q80_kwh": fallback.get("load_3h_q80_kwh"),
                "load_3h_q95_kwh": fallback.get("load_3h_q95_kwh"),
                "suggested_price_shift_pct": as_float(item.get("suggested_price_shift_pct"), 0),
                "action_label": str(item.get("action_label") or ""),
                "price_rationale": str(item.get("price_rationale") or ""),
            }
        )
    return normalized


def normalize_grid_stress_level(value: Any, fallback: Any = "Low") -> str:
    key = str(value or "").strip().lower()
    if key in GRID_STRESS_LEVEL_BY_KEY:
        return GRID_STRESS_LEVEL_BY_KEY[key]
    fallback_key = str(fallback or "").strip().lower()
    return GRID_STRESS_LEVEL_BY_KEY.get(fallback_key, "Low")
