from __future__ import annotations

import argparse
from pathlib import Path

from orchestrator import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Run Multi-Agent Prescriptive Forecasting on UrbanEV data.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing UrbanEV CSV files.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated reports.")
    parser.add_argument("--model", default=None, help="OpenRouter model id, for example openai/gpt-4o-mini.")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls and use deterministic heuristics.")
    parser.add_argument("--force-cache", action="store_true", help="Rebuild cached zone profiles and POI assignment.")
    parser.add_argument("--max-poi-rows", type=int, default=None, help="Limit POI rows for quick experiments.")
    parser.add_argument("--forecast-start", default=None, help="ISO timestamp for the forecast window start.")
    parser.add_argument("--horizon-days", type=int, default=4, help="Forecast horizon.")
    parser.add_argument("--history-days", type=int, default=7, help="History window used for the zone snippets.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)


    outputs = run_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model=args.model,
        dry_run=args.dry_run,
        force_cache=args.force_cache,
        max_poi_rows=args.max_poi_rows,
        forecast_start=args.forecast_start,
        horizon_days=args.horizon_days,
        history_days=args.history_days,
        temperature=args.temperature,
    )
    print("Generated outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
