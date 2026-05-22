from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from config import AgentConfig
from prompts import SYSTEM_MESSAGE, behavior_prompt, economist_prompt, grid_prompt


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
        "grid_stress_level": context["grid_stress_level"],
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
    drivers = []
    if weather["rain_hours"] > 0:
        drivers.append(f"{weather['rain_hours']} rainy hours")
    if shape["night_20_6"] >= shape["morning_7_10"]:
        drivers.append("night plateau")
    if shape["evening_17_22"] >= shape["morning_7_10"]:
        drivers.append("evening lift")
    if not drivers:
        drivers.append("stable hourly profile")
    return {
        "agent_reasoning": (
            f"{category} demand is best explained by {', '.join(drivers)} and the "
            f"local POI mix of {context['profile']['poi_total']} assigned POIs."
        ),
        "demand_drivers": drivers,
        "confidence": "medium",
    }


def heuristic_economist(context: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    category = context["category"]
    stress = grid["grid_stress_level"]
    change = float(grid["predicted_change_pct"])
    if stress == "Critical" and category == "Residential":
        shift = 15
        label = "Peak deflection fee"
    elif stress == "Critical":
        shift = 10
        label = "Congestion fee"
    elif stress == "Moderate" and change > 10:
        shift = 5
        label = "Moderate peak premium"
    elif stress == "Low" and change < -10:
        shift = -5
        label = "Utilization incentive"
    else:
        shift = 0
        label = "Hold price"
    return {
        "suggested_price_shift_pct": shift,
        "action_label": label,
        "price_rationale": (
            f"{stress} stress with {change:+.1f}% expected load change; "
            f"category elasticity proxy is {category}."
        ),
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
        "wape_pct": context.get("wape_pct"),
        "grid_stress_level": str(grid.get("grid_stress_level") or context["grid_stress_level"]),
        "agent_reasoning": str(behavior.get("agent_reasoning") or ""),
        "suggested_price_shift_pct": as_float(economist.get("suggested_price_shift_pct"), 0),
        "action_label": str(economist.get("action_label") or ""),
        "price_rationale": str(economist.get("price_rationale") or ""),
        "source": source,
    }


def merge_grid_fallback(report: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_grid(context)
    return {**fallback, **{key: value for key, value in report.items() if value not in (None, "")}}


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
