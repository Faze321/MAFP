from __future__ import annotations

import argparse
from pathlib import Path

from config import AppConfig, normalize_forecast_model_list, normalize_string_list
from orchestrator import run_experiment_matrix, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Run Multi-Agent Prescriptive Forecasting on UrbanEV data.",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config file for run and model settings.")
    parser.add_argument("--data-dir", default=None, help="Directory containing UrbanEV CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated reports.")
    parser.add_argument("--weather-file", default=None, help="Weather CSV file under data-dir.")
    parser.add_argument(
        "--forecast-model",
        default=None,
        help="Forecast model: timesfm, chronos, lstm, or AR.",
    )
    parser.add_argument("--model", default=None, help="Model id, for example openai/gpt-4o-mini.")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip LLM calls and use deterministic heuristics.",
    )
    parser.add_argument(
        "--force-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rebuild cached zone profiles and POI assignment.",
    )
    parser.add_argument("--max-poi-rows", type=int, default=None, help="Limit POI rows for quick experiments.")
    parser.add_argument("--forecast-start", default=None, help="ISO timestamp for the forecast window start.")
    parser.add_argument(
        "--forecast-starts",
        nargs="+",
        default=None,
        help="ISO timestamp(s) for an experiment matrix. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument("--horizon-days", type=int, default=None, help="Forecast horizon.")
    parser.add_argument("--history-days", type=int, default=None, help="History window used for the zone snippets.")
    parser.add_argument(
        "--zones",
        nargs="+",
        default=None,
        help=(
            "UrbanEV zone id(s) to validate directly. If omitted, the pipeline keeps the automatic five-zone category selection."
        ),
    )
    parser.add_argument(
        "--forecast-models",
        nargs="+",
        default=None,
        help="Forecast model(s) for an experiment matrix: timesfm, chronos, lstm, or AR.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="LLM sampling temperature.")
    return parser


def resolve_forecast_starts(args, run_config) -> list[str]:
    cli_starts = normalize_string_list(args.forecast_starts)
    if cli_starts:
        return cli_starts
    if args.forecast_start:
        return [args.forecast_start]
    if run_config.forecast_starts:
        return run_config.forecast_starts
    return [run_config.forecast_start] if run_config.forecast_start else []


def resolve_forecast_models(args, run_config) -> list[str]:
    cli_models = normalize_forecast_model_list(args.forecast_models)
    if cli_models:
        return cli_models
    if args.forecast_model:
        return normalize_forecast_model_list([args.forecast_model]) or [args.forecast_model]
    if run_config.forecast_models:
        return run_config.forecast_models
    return [run_config.forecast_model]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    app_config = AppConfig.from_file(config_path, required=False)
    run_config = app_config.run
    forecast_starts = resolve_forecast_starts(args, run_config)
    forecast_models = resolve_forecast_models(args, run_config)
    run_matrix = len(forecast_starts) > 1 or len(forecast_models) > 1

    common_kwargs = {
        "data_dir": Path(args.data_dir or run_config.data_dir),
        "output_dir": Path(args.output_dir or run_config.output_dir),
        "config_path": config_path,
        "model": args.model,
        "weather_file": args.weather_file or run_config.weather_file,
        "dry_run": args.dry_run if args.dry_run is not None else run_config.dry_run,
        "force_cache": args.force_cache if args.force_cache is not None else run_config.force_cache,
        "max_poi_rows": args.max_poi_rows if args.max_poi_rows is not None else run_config.max_poi_rows,
        "horizon_days": args.horizon_days if args.horizon_days is not None else run_config.horizon_days,
        "history_days": args.history_days if args.history_days is not None else run_config.history_days,
        "validation_days": run_config.validation_days,
        "zone_ids": args.zones if args.zones is not None else run_config.zone_ids,
        "timesfm_repo": run_config.timesfm_repo,
        "timesfm_context_hours": run_config.timesfm_context_hours,
        "timesfm_step_horizon": run_config.timesfm_step_horizon,
        "timesfm_exog_cols": run_config.timesfm_exog_cols,
        "timesfm_diurnal_blend_alpha": run_config.timesfm_diurnal_blend_alpha,
        "timesfm_roll_actuals": run_config.timesfm_roll_actuals,
        "ar_diurnal_blend_alpha": run_config.ar_diurnal_blend_alpha,
        "chronos_repo": run_config.chronos_repo,
        "chronos_context_hours": run_config.chronos_context_hours,
        "chronos_step_horizon": run_config.chronos_step_horizon,
        "chronos_diurnal_blend_alpha": run_config.chronos_diurnal_blend_alpha,
        "chronos_device": run_config.chronos_device,
        "chronos_roll_actuals": run_config.chronos_roll_actuals,
        "lstm_context_hours": run_config.lstm_context_hours,
        "lstm_step_horizon": run_config.lstm_step_horizon,
        "lstm_exog_cols": run_config.lstm_exog_cols,
        "lstm_hidden_size": run_config.lstm_hidden_size,
        "lstm_num_layers": run_config.lstm_num_layers,
        "lstm_epochs": run_config.lstm_epochs,
        "lstm_learning_rate": run_config.lstm_learning_rate,
        "lstm_batch_size": run_config.lstm_batch_size,
        "lstm_diurnal_blend_alpha": run_config.lstm_diurnal_blend_alpha,
        "lstm_device": run_config.lstm_device,
        "lstm_roll_actuals": run_config.lstm_roll_actuals,
        "lstm_seed": run_config.lstm_seed,
        "temperature": args.temperature if args.temperature is not None else run_config.temperature,
    }

    if run_matrix:
        if not forecast_starts:
            raise ValueError("Experiment matrix requires at least one forecast start.")
        outputs = run_experiment_matrix(
            forecast_starts=forecast_starts,
            forecast_models=forecast_models,
            **common_kwargs,
        )
    else:
        outputs = run_pipeline(
            forecast_start=forecast_starts[0] if forecast_starts else None,
            forecast_model=forecast_models[0],
            **common_kwargs,
        )
    print("Generated outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
