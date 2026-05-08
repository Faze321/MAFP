from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from agents import OpenRouterChatClient, run_all_zone_chains
from config import OpenRouterConfig
from data_loader import build_zone_profiles, load_pipeline_data
from forecasting import forecast_zone
from reporting import write_outputs
from zone_selection import select_zone_categories


def run_pipeline(
    *,
    data_dir: Path,
    output_dir: Path,
    config_path: Path = Path("config.json"),
    model: str | None = None,
    dry_run: bool = False,
    force_cache: bool = False,
    max_poi_rows: int | None = None,
    forecast_start: str | None = None,
    horizon_days: int = 4,
    history_days: int = 7,
    temperature: float = 0.2,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = build_zone_profiles(
        data_dir,
        output_dir / "cache",
        force_cache=force_cache,
        max_poi_rows=max_poi_rows,
    )
    selected_zones = select_zone_categories(profiles)
    selected_zone_ids = selected_zones["zone_id"].astype(str).tolist()
    pipeline_data = load_pipeline_data(data_dir, profiles, selected_zone_ids)
    contexts = build_contexts(
        pipeline_data=pipeline_data,
        selected_zones=selected_zones,
        forecast_start=forecast_start,
        horizon_days=horizon_days,
        history_days=history_days,
    )

    if dry_run:
        client = None
    else:
        config = OpenRouterConfig.from_json(config_path, model=model, required=True)
        if not config.api_key:
            raise RuntimeError("openrouter.api_key is required in config.json, or pass --dry-run")
        client = OpenRouterChatClient(config)
    reports = asyncio.run(
        run_all_zone_chains(contexts, client=client, temperature=temperature)
    )
    return write_outputs(
        output_dir=output_dir,
        selected_zones=selected_zones,
        contexts=contexts,
        reports=reports,
    )


def build_contexts(
    *,
    pipeline_data,
    selected_zones: pd.DataFrame,
    forecast_start: str | None,
    horizon_days: int,
    history_days: int,
) -> list[dict[str, Any]]:
    start = pd.Timestamp(forecast_start) if forecast_start else None
    contexts = []
    profiles_by_zone = pipeline_data.profiles.set_index("zone_id", drop=False)
    for row in selected_zones.to_dict(orient="records"):
        zone_id = str(row["zone_id"])
        profile = profiles_by_zone.loc[zone_id].to_dict()
        result = forecast_zone(
            zone_id=zone_id,
            category=row["category"],
            load=pipeline_data.load,
            service_price=pipeline_data.service_price,
            occupancy=pipeline_data.occupancy,
            weather=pipeline_data.weather,
            profile=profile,
            forecast_start=start,
            horizon_days=horizon_days,
            history_days=history_days,
        )
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
    return contexts
