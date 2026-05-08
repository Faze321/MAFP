from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORY_ORDER = [
    "CBD / Office",
    "Residential",
    "Transport Hub",
    "Commercial / Mall",
    "Industrial",
]


def select_zone_categories(profiles: pd.DataFrame) -> pd.DataFrame:
    frame = profiles.copy()
    scores = pd.DataFrame({"zone_id": frame["zone_id"].astype(str)})
    scores["CBD / Office"] = (
        1.25 * z(frame["poi_business_density"])
        + 0.85 * z(frame["morning_ratio"])
        + 0.55 * z(frame["noon_ratio"])
        + 0.25 * z(frame["mean_load_kwh"])
        - 0.25 * z(frame["night_ratio"])
    )
    scores["Residential"] = (
        1.05 * z(frame["poi_business_density"])
        + 1.10 * z(frame["night_ratio"])
        + 0.55 * z(frame["evening_ratio"])
        - 0.20 * z(frame["morning_ratio"])
    )
    scores["Transport Hub"] = (
        1.20 * z(frame["charge_count"])
        + 0.95 * z(frame["burstiness_p99_mean"])
        + 0.50 * z(frame["load_cv"])
        + 0.35 * z(frame["peak_load_kwh"])
    )
    scores["Commercial / Mall"] = (
        0.90 * z(frame["poi_food_density"])
        + 0.90 * z(frame["poi_lifestyle_density"])
        + 0.80 * z(frame["weekend_ratio"])
        + 0.65 * z(frame["evening_ratio"])
    )
    scores["Industrial"] = (
        0.90 * z(frame["charge_count"])
        + 0.75 * z(frame["mean_load_kwh"])
        - 1.15 * z(frame["load_cv"])
        + 0.55 * z(1.0 - (frame["weekend_ratio"] - 1.0).abs())
    )

    selected_rows = []
    used_zone_ids: set[str] = set()
    by_zone = frame.set_index("zone_id", drop=False)
    for category in CATEGORY_ORDER:
        ranked = scores[["zone_id", category]].sort_values(category, ascending=False)
        choice = next(row for row in ranked.itertuples(index=False) if row.zone_id not in used_zone_ids)
        used_zone_ids.add(str(choice.zone_id))
        profile = by_zone.loc[str(choice.zone_id)].to_dict()
        selected_rows.append(
            {
                **profile,
                "category": category,
                "selection_score": float(getattr(choice, category.replace(" ", "_").replace("/", "_"), choice[1])),
                "selection_reason": selection_reason(category, profile),
            }
        )

    selected = pd.DataFrame(selected_rows)
    return selected[
        [
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
    ]


def z(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr, dtype=float)
    return (arr - np.nanmean(arr)) / std


def selection_reason(category: str, profile: dict) -> str:
    if category == "CBD / Office":
        return (
            "High business/residential POI density with stronger morning and noon demand signals."
        )
    if category == "Residential":
        return "High night/evening charging plateau, used as a residential demand proxy."
    if category == "Transport Hub":
        return "High charging capacity and bursty load profile, used as a hub proxy."
    if category == "Commercial / Mall":
        return "High food/lifestyle POI density with evening or weekend demand lift."
    return "Stable high base load and capacity, used as an industrial proxy."
