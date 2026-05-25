from __future__ import annotations

import argparse
from pathlib import Path

from config import AppConfig
from orchestrator import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Run Multi-Agent Prescriptive Forecasting on UrbanEV data.",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config file for run and model settings.")
    parser.add_argument("--data-dir", default=None, help="Directory containing UrbanEV CSV files.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated reports.")
    parser.add_argument("--weather-file", default=None, help="Weather CSV file under data-dir.")
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
    parser.add_argument("--temperature", type=float, default=None, help="LLM sampling temperature.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    app_config = AppConfig.from_file(config_path, required=False)
    run_config = app_config.run

    outputs = run_pipeline(
        data_dir=Path(args.data_dir or run_config.data_dir),
        output_dir=Path(args.output_dir or run_config.output_dir),
        config_path=config_path,
        model=args.model,
        weather_file=args.weather_file or run_config.weather_file,
        dry_run=args.dry_run if args.dry_run is not None else run_config.dry_run,
        force_cache=args.force_cache if args.force_cache is not None else run_config.force_cache,
        max_poi_rows=args.max_poi_rows if args.max_poi_rows is not None else run_config.max_poi_rows,
        forecast_start=args.forecast_start or run_config.forecast_start,
        horizon_days=args.horizon_days if args.horizon_days is not None else run_config.horizon_days,
        history_days=args.history_days if args.history_days is not None else run_config.history_days,
        validation_days=run_config.validation_days,
        zone_ids=args.zones if args.zones is not None else run_config.zone_ids,
        forecast_model=run_config.forecast_model,
        timefm_repo=run_config.timefm_repo,
        timefm_context_hours=run_config.timefm_context_hours,
        timefm_step_horizon=run_config.timefm_step_horizon,
        timefm_exog_cols=run_config.timefm_exog_cols,
        timefm_diurnal_blend_alpha=run_config.timefm_diurnal_blend_alpha,
        timefm_roll_actuals=run_config.timefm_roll_actuals,
        temperature=args.temperature if args.temperature is not None else run_config.temperature,
    )
    print("Generated outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
