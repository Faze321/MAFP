import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from agents import extract_json_object
from config import AgentConfig, AppConfig, RunConfig
from forecasting import (
    build_zone_model_frame,
    chronos_forecast,
    compute_forecast_metrics,
    lstm_forecast,
    patch_timesfm_hub_kwargs,
    rebuild_quantile_interval,
    seasonal_naive_forecast,
)
from orchestrator import forecast_output_dir, normalize_zone_ids, select_requested_zones
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
        self.assertEqual(config.run.weather_file, "weather_central.csv")
        self.assertEqual(config.run.forecast_start, "2022-09-09 00:00:00")
        self.assertEqual(config.run.horizon_days, 6)
        self.assertEqual(config.run.history_days, 7)
        self.assertEqual(config.run.validation_days, 1)
        self.assertEqual(config.run.zone_ids, ["102"])
        self.assertEqual(config.run.forecast_model, "timesfm")
        self.assertEqual(config.run.timesfm_repo, "google/timesfm-2.5-200m-pytorch")
        self.assertEqual(config.run.timesfm_exog_cols[:4], ["T", "U", "nRAIN", "e_price"])
        self.assertEqual(config.run.seasonal_diurnal_blend_alpha, 0.0)
        self.assertEqual(config.run.chronos_repo, "amazon/chronos-2")
        self.assertEqual(config.run.chronos_context_hours, 512)
        self.assertEqual(config.run.chronos_diurnal_blend_alpha, 0.0)
        self.assertEqual(config.run.lstm_context_hours, 24)
        self.assertEqual(config.run.lstm_epochs, 50)
        self.assertEqual(config.run.lstm_diurnal_blend_alpha, 0.0)

    def test_reads_timesfm_config_keys(self):
        config = RunConfig.from_mapping(
            {
                "forecast_model": "timesfm",
                "timesfm_repo": "google/timesfm-2.5-200m-pytorch",
                "timesfm_context_hours": 48,
                "timesfm_exog_cols": ["T", "U"],
            }
        )
        self.assertEqual(config.forecast_model, "timesfm")
        self.assertEqual(config.timesfm_context_hours, 48)
        self.assertEqual(config.timesfm_exog_cols, ["T", "U"])


class ForecastingTests(unittest.TestCase):
    def test_timesfm_loader_strips_huggingface_hub_kwargs(self):
        class FakeTimesFM:
            seen_kwargs = None

            @classmethod
            def _from_pretrained(cls, **kwargs):
                cls.seen_kwargs = kwargs
                return cls()

        patch_timesfm_hub_kwargs(FakeTimesFM)
        FakeTimesFM._from_pretrained(
            model_id="fake/repo",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies={"http": "http://proxy"},
            resume_download=None,
            local_files_only=False,
            token=None,
            config={"model": "fake"},
        )

        self.assertNotIn("proxies", FakeTimesFM.seen_kwargs)
        self.assertNotIn("resume_download", FakeTimesFM.seen_kwargs)
        self.assertEqual(FakeTimesFM.seen_kwargs["config"], {"model": "fake"})

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

    def test_forecast_metrics_include_error_standards(self):
        hourly = pd.DataFrame(
            {
                "actual_kwh": [10.0, 20.0, 30.0],
                "predicted_kwh": [12.0, 18.0, 33.0],
            }
        )
        metrics = compute_forecast_metrics(hourly)
        self.assertEqual(metrics["n"], 3)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE_pct", metrics)
        self.assertIn("RAE", metrics)

    def test_build_zone_model_frame_adds_notebook_covariates(self):
        times = pd.date_range("2023-01-01", periods=2, freq="h")
        load = pd.DataFrame({"time": times, "102": [10.0, 11.0]})
        service_price = pd.DataFrame({"time": times, "102": [0.5, 0.6]})
        energy_price = pd.DataFrame({"time": times, "102": [1.0, 1.2]})
        occupancy = pd.DataFrame({"time": times, "102": [0.1, 0.2]})
        weather = pd.DataFrame({"time": times, "T": [20.0, 21.0], "U": [60.0, 61.0], "nRAIN": [0.0, 1.0]})
        frame = build_zone_model_frame(load, service_price, energy_price, occupancy, weather, "102")
        self.assertIn("e_price", frame)
        self.assertIn("is_weekend", frame)
        self.assertIn("temp_price_idx", frame)
        self.assertAlmostEqual(frame["temp_price_idx"].iloc[1], 25.2)

    def test_rebuild_quantile_interval_centers_final_prediction(self):
        point = np.array([100.0, 2.0, 50.0])
        raw_q10 = np.array([40.0, 0.0, np.nan])
        raw_q90 = np.array([80.0, 10.0, np.nan])
        q10, q90 = rebuild_quantile_interval(point, raw_q10, raw_q90)
        self.assertEqual(q10[0], 80.0)
        self.assertEqual(q90[0], 120.0)
        self.assertEqual(q10[1], 0.0)
        self.assertEqual(q90[1], 7.0)
        self.assertTrue(np.isnan(q10[2]))
        self.assertTrue(np.isnan(q90[2]))
        self.assertTrue(np.all(q10[:2] <= point[:2]))
        self.assertTrue(np.all(q90[:2] >= point[:2]))

    def test_chronos_forecast_accepts_chronos2_list_output(self):
        test_case = self

        class FakeChronos2:
            def predict(
                self,
                inputs,
                prediction_length=None,
                batch_size=256,
                context_length=None,
                cross_learning=False,
                limit_prediction_length=False,
                **kwargs,
            ):
                return None

            def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
                test_case.assertEqual(kwargs, {"limit_prediction_length": False})
                center = torch.arange(1, prediction_length + 1, dtype=torch.float32)
                quantiles = torch.stack([center - 0.5, center, center + 0.5], dim=-1).unsqueeze(0)
                return [quantiles], [center.unsqueeze(0)]

        history = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-01", periods=24, freq="h"),
                "actual_kwh": [float(hour + 1) for hour in range(24)],
            }
        )
        full_frame = pd.DataFrame(
            {
                "time": pd.date_range("2023-01-02", periods=2, freq="h"),
                "actual_kwh": [1.0, 2.0],
            }
        )
        with patch("forecasting.load_chronos_model", return_value=FakeChronos2()):
            result = chronos_forecast(
                history,
                pd.DataFrame(),
                full_frame,
                pd.Timestamp("2023-01-02"),
                2,
                repo="fake",
                context_hours=24,
                step_horizon=2,
                diurnal_blend_alpha=0.0,
                device="cpu",
                roll_actuals=False,
            )
        self.assertEqual(result["predicted_kwh"].tolist(), [1.0, 2.0])
        self.assertEqual(result["q10_kwh"].tolist(), [0.5, 1.5])
        self.assertEqual(result["q90_kwh"].tolist(), [1.5, 2.5])

    def test_lstm_forecast_produces_hourly_predictions(self):
        times = pd.date_range("2023-01-01", periods=60, freq="h")
        values = 20.0 + 5.0 * np.sin(np.arange(60) / 24.0 * 2.0 * np.pi)
        frame = pd.DataFrame({"time": times, "actual_kwh": values})
        history = frame.iloc[:48].reset_index(drop=True)
        validation = frame.iloc[48:54].reset_index(drop=True)
        full_frame = frame.iloc[54:60].reset_index(drop=True)

        result = lstm_forecast(
            history,
            validation,
            full_frame,
            pd.Timestamp("2023-01-03 06:00:00"),
            6,
            context_hours=12,
            step_horizon=3,
            exog_cols=[],
            hidden_size=8,
            num_layers=1,
            epochs=2,
            learning_rate=0.01,
            batch_size=8,
            diurnal_blend_alpha=0.0,
            device="cpu",
            roll_actuals=False,
            seed=7,
        )

        self.assertEqual(len(result), 6)
        self.assertTrue(result["predicted_kwh"].notna().all())
        self.assertTrue((result["predicted_kwh"] >= 0).all())
        self.assertTrue((result["q10_kwh"] <= result["predicted_kwh"]).all())
        self.assertTrue((result["q90_kwh"] >= result["predicted_kwh"]).all())


class SelectionTests(unittest.TestCase):
    def test_forecast_output_dir_uses_model_subfolder(self):
        self.assertEqual(forecast_output_dir(Path("output"), "chronos"), Path("output") / "chronos")
        self.assertEqual(forecast_output_dir(Path("output"), "lstm"), Path("output") / "lstm")
        self.assertEqual(forecast_output_dir(Path("output"), "timesfm"), Path("output") / "timesfm")
        self.assertEqual(
            forecast_output_dir(Path("output"), "seasonal-naive"),
            Path("output") / "seasonal_naive",
        )

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
