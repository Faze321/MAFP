import unittest
from pathlib import Path

import pandas as pd

from agents import extract_json_object
from config import AgentConfig, AppConfig
from forecasting import seasonal_naive_forecast
from orchestrator import normalize_zone_ids, select_requested_zones
from zone_selection import select_zone_categories


class AgentParsingTests(unittest.TestCase):
    def test_extracts_json_from_markdown_fence(self):
        payload = extract_json_object('```json\n{"a": 1, "b": "x"}\n```')
        self.assertEqual(payload, {"a": 1, "b": "x"})


class ConfigTests(unittest.TestCase):
    def test_reads_agent_yaml_config(self):
        config = AgentConfig.from_file(Path("config.example.yaml"))
        self.assertEqual(config.api_key, "sk-...")
        self.assertEqual(config.model, "meta-llama/llama-3.1-8b-instruct")
        self.assertEqual(config.timeout_seconds, 90)

    def test_reads_run_yaml_config(self):
        config = AppConfig.from_file(Path("config.example.yaml"))
        self.assertTrue(config.run.dry_run)
        self.assertEqual(config.run.horizon_days, 1)
        self.assertEqual(config.run.history_days, 7)
        self.assertEqual(config.run.zone_ids, ["102"])


class ForecastingTests(unittest.TestCase):
    def test_seasonal_forecast_keeps_hourly_horizon(self):
        history = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=24 * 7, freq="h"),
                "actual_kwh": [float(hour % 24) for hour in range(24 * 7)],
            }
        )
        result = seasonal_naive_forecast(history, pd.Timestamp("2023-01-08"), 48)
        self.assertEqual(len(result), 48)
        self.assertIn("predicted_kwh", result)
        self.assertGreater(result["predicted_kwh"].sum(), 0)


class SelectionTests(unittest.TestCase):
    def test_normalizes_requested_zone_ids(self):
        zone_ids = normalize_zone_ids(["102,104", " 108 ", "104"])
        self.assertEqual(zone_ids, ["102", "104", "108"])

    def test_selects_requested_zones_in_user_order(self):
        profiles = pd.DataFrame(
            {
                "zone_id": ["101", "102", "104"],
                "longitude": [0.0, 1.0, 2.0],
                "latitude": [0.0, 1.0, 2.0],
                "station_count": [1, 2, 3],
                "charge_count": [10, 20, 30],
                "capacity_kw_proxy": [110.0, 220.0, 330.0],
                "mean_load_kwh": [5.0, 6.0, 7.0],
                "peak_load_kwh": [8.0, 9.0, 10.0],
                "peak_capacity_ratio": [0.1, 0.2, 0.3],
                "load_cv": [0.1, 0.2, 0.3],
                "burstiness_p99_mean": [1.1, 1.2, 1.3],
                "morning_ratio": [1.0, 1.0, 1.0],
                "noon_ratio": [1.0, 1.0, 1.0],
                "evening_ratio": [1.0, 1.0, 1.0],
                "night_ratio": [1.0, 1.0, 1.0],
                "weekend_ratio": [1.0, 1.0, 1.0],
                "poi_food": [0, 0, 0],
                "poi_business": [0, 0, 0],
                "poi_lifestyle": [0, 0, 0],
                "poi_total": [0, 0, 0],
                "mean_service_price": [0.7, 0.8, 0.9],
            }
        )
        selected = select_requested_zones(profiles, ["104", "102"])
        self.assertEqual(selected["zone_id"].tolist(), ["104", "102"])
        self.assertEqual(selected["category"].tolist(), ["User-selected", "User-selected"])

    def test_selects_five_unique_categories(self):
        profiles = pd.DataFrame(
            {
                "zone_id": ["1", "2", "3", "4", "5", "6"],
                "poi_business_density": [10, 100, 5, 3, 2, 1],
                "poi_food_density": [5, 2, 8, 100, 1, 1],
                "poi_lifestyle_density": [5, 3, 8, 80, 1, 1],
                "morning_ratio": [1.5, 1.0, 1.0, 1.0, 0.8, 0.7],
                "noon_ratio": [1.4, 1.0, 1.0, 1.0, 0.9, 0.8],
                "evening_ratio": [0.8, 1.5, 1.0, 1.4, 1.0, 0.8],
                "night_ratio": [0.7, 1.7, 1.0, 1.0, 1.0, 0.8],
                "weekend_ratio": [1.0, 1.0, 1.1, 1.8, 1.0, 0.9],
                "mean_load_kwh": [100, 120, 200, 150, 180, 90],
                "peak_load_kwh": [180, 200, 500, 250, 220, 100],
                "charge_count": [20, 30, 400, 40, 300, 10],
                "burstiness_p99_mean": [1.5, 1.6, 5.0, 2.0, 1.2, 1.0],
                "load_cv": [0.5, 0.6, 2.0, 0.8, 0.1, 0.4],
                "longitude": [0] * 6,
                "latitude": [0] * 6,
                "station_count": [1] * 6,
                "capacity_kw_proxy": [220, 330, 4400, 440, 3300, 110],
                "peak_capacity_ratio": [0.8, 0.6, 0.1, 0.5, 0.07, 0.9],
                "poi_food": [0] * 6,
                "poi_business": [0] * 6,
                "poi_lifestyle": [0] * 6,
                "poi_total": [0] * 6,
                "mean_service_price": [0.76] * 6,
            }
        )
        selected = select_zone_categories(profiles)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected["zone_id"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()
