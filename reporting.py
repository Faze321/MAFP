from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


TRACE_COLUMNS = [
    "category",
    "zone_id",
    "predicted_load_kwh",
    "predicted_change_pct",
    "actual_load_kwh",
    "mae_kwh",
    "rmse_kwh",
    "mape_pct",
    "rae",
    "wape_pct",
    "grid_stress_level",
    "actual_grid_stress_level",
    "stress_accuracy",
    "miss_stress_rate",
    "stress_eval_windows",
    "stress_miss_count",
    "agent_reasoning",
    "suggested_price_shift_pct",
    "action_label",
    "price_rationale",
    "source",
]


def write_outputs(
    *,
    output_dir: Path,
    selected_zones: pd.DataFrame,
    contexts: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    forecast_results: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_zones.csv"
    contexts_path = output_dir / "context_snippets.json"
    trace_csv = output_dir / "rationale_trace.csv"
    trace_md = output_dir / "rationale_trace.md"
    trace_json = output_dir / "rationale_trace.json"
    metrics_csv = output_dir / "forecast_metrics.csv"
    metrics_md = output_dir / "forecast_metrics.md"
    price_schedule_csv = output_dir / "price_schedule_3h.csv"
    price_schedule_md = output_dir / "price_schedule_3h.md"
    details_dir = output_dir / "forecast_details"

    selected_zones.to_csv(selected_path, index=False)
    contexts_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")

    trace = pd.DataFrame(reports)
    trace = trace[[col for col in TRACE_COLUMNS if col in trace.columns]]
    trace.to_csv(trace_csv, index=False)
    trace_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_md.write_text(markdown_table(trace), encoding="utf-8")
    write_price_schedule_outputs(price_schedule_csv, price_schedule_md, reports)
    metrics = write_forecast_outputs(details_dir, metrics_csv, metrics_md, forecast_results)

    outputs = {
        "selected_zones": selected_path,
        "context_snippets": contexts_path,
        "rationale_trace_csv": trace_csv,
        "rationale_trace_md": trace_md,
        "rationale_trace_json": trace_json,
        "forecast_metrics_csv": metrics_csv,
        "forecast_metrics_md": metrics_md,
        "price_schedule_3h_csv": price_schedule_csv,
        "price_schedule_3h_md": price_schedule_md,
        "forecast_details_dir": details_dir,
    }
    outputs.update(metrics)
    return outputs


def write_forecast_outputs(
    details_dir: Path,
    metrics_csv: Path,
    metrics_md: Path,
    forecast_results: dict[str, Any],
) -> dict[str, Path]:
    details_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    plot_paths = {}

    for zone_id, result in forecast_results.items():
        safe_zone = safe_filename(zone_id)
        hourly = result.hourly.copy()
        hourly.insert(0, "zone_id", zone_id)
        hourly.insert(1, "category", result.summary.get("category"))
        hourly_path = details_dir / f"zone_{safe_zone}_forecast_vs_actual.csv"
        plot_path = details_dir / f"zone_{safe_zone}_forecast_plot.png"
        old_svg_path = details_dir / f"zone_{safe_zone}_forecast_plot.svg"
        hourly.to_csv(hourly_path, index=False)
        if old_svg_path.exists():
            old_svg_path.unlink()
        write_zone_plot(plot_path, result.summary, hourly)
        plot_paths[f"zone_{safe_zone}_forecast_csv"] = hourly_path
        plot_paths[f"zone_{safe_zone}_forecast_plot"] = plot_path

        metrics = result.summary.get("metrics", {}) or {}
        metric_rows.append(
            {
                "zone_id": zone_id,
                "category": result.summary.get("category"),
                "forecast_model": result.summary.get("forecast_model"),
                "calibration_enabled": (result.summary.get("calibration") or {}).get("enabled"),
                "bias_mean": (result.summary.get("calibration") or {}).get("bias_mean"),
                "bias_max_abs": (result.summary.get("calibration") or {}).get("bias_max_abs"),
                "forecast_start": result.summary.get("forecast_start"),
                "forecast_end": result.summary.get("forecast_end"),
                "n": metrics.get("n"),
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "MAPE_pct": metrics.get("MAPE_pct"),
                "RAE": metrics.get("RAE"),
                "WAPE_pct": metrics.get("WAPE_pct"),
                "forecast_total_kwh": result.summary.get("forecast_total_kwh"),
                "actual_total_kwh": result.summary.get("actual_total_kwh"),
                "forecast_peak_kwh": result.summary.get("forecast_peak_kwh"),
                "actual_peak_kwh": result.summary.get("actual_peak_kwh"),
                "grid_stress_level": result.summary.get("grid_stress_level"),
                "actual_grid_stress_level": result.summary.get("actual_grid_stress_level"),
                "stress_accuracy": result.summary.get("stress_accuracy"),
                "miss_stress_rate": result.summary.get("miss_stress_rate"),
                "stress_eval_windows": result.summary.get("stress_eval_windows"),
                "stress_miss_count": result.summary.get("stress_miss_count"),
                "grid_stress_basis": result.summary.get("grid_stress_basis"),
                "grid_stress_load_kwh": result.summary.get("grid_stress_load_kwh"),
                "actual_grid_stress_load_kwh": result.summary.get("actual_grid_stress_load_kwh"),
                "grid_stress_source_file": result.summary.get("grid_stress_source_file"),
                "grid_stress_window_hours": result.summary.get("grid_stress_window_hours"),
                "grid_stress_historical_windows": result.summary.get("grid_stress_historical_windows"),
                "grid_stress_q50_kwh": result.summary.get("grid_stress_q50_kwh"),
                "grid_stress_q80_kwh": result.summary.get("grid_stress_q80_kwh"),
                "grid_stress_q95_kwh": result.summary.get("grid_stress_q95_kwh"),
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(metrics_csv, index=False)
    metrics_md.write_text(markdown_table(metrics_frame), encoding="utf-8")
    return plot_paths


def write_price_schedule_outputs(price_schedule_csv: Path, price_schedule_md: Path, reports: list[dict[str, Any]]) -> None:
    rows = []
    for report in reports:
        for window in report.get("price_change_windows_3h") or []:
            if not isinstance(window, dict):
                continue
            rows.append(
                {
                    "zone_id": report.get("zone_id"),
                    "category": report.get("category"),
                    "window_start": window.get("window_start"),
                    "window_end": window.get("window_end"),
                    "sum_predicted_kwh": window.get("sum_predicted_kwh"),
                    "mean_predicted_kwh": window.get("mean_predicted_kwh"),
                    "sum_actual_kwh": window.get("sum_actual_kwh"),
                    "load_stress_level": window.get("load_stress_level") or window.get("grid_stress_level"),
                    "stress_load_3h_kwh": window.get("stress_load_3h_kwh"),
                    "actual_load_stress_level": window.get("actual_load_stress_level") or window.get("actual_grid_stress_level"),
                    "actual_stress_load_3h_kwh": window.get("actual_stress_load_3h_kwh"),
                    "stress_correct": window.get("stress_correct"),
                    "stress_missed": window.get("stress_missed"),
                    "load_3h_q50_kwh": window.get("load_3h_q50_kwh"),
                    "load_3h_q80_kwh": window.get("load_3h_q80_kwh"),
                    "load_3h_q95_kwh": window.get("load_3h_q95_kwh"),
                    "suggested_price_shift_pct": window.get("suggested_price_shift_pct"),
                    "action_label": window.get("action_label"),
                    "price_rationale": window.get("price_rationale"),
                    "source": report.get("source"),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(price_schedule_csv, index=False)
    price_schedule_md.write_text(markdown_table(frame), encoding="utf-8")


def write_zone_plot(path: Path, summary: dict[str, Any], hourly: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.dates as mdates
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for forecast plots. Install it with: pip install matplotlib") from exc

    frame = hourly.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame["hour_index"] = range(len(frame))

    fig = plt.figure(figsize=(16, 9), dpi=140)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[3.2, 1.0], hspace=0.46, wspace=0.22)
    ax_main = fig.add_subplot(gs[0, :])
    ax_resid = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])

    ax_main.plot(
        frame["time"],
        frame["actual_kwh"],
        color="#185FA5",
        lw=2.0,
        marker="o",
        ms=3.2,
        label="Actual",
    )
    if {"q10_kwh", "q90_kwh"}.issubset(frame.columns) and frame[["q10_kwh", "q90_kwh"]].notna().any().any():
        ax_main.fill_between(
            frame["time"],
            frame["q10_kwh"],
            frame["q90_kwh"],
            color="#D85A30",
            alpha=0.14,
            label="P10-P90",
        )
    ax_main.plot(
        frame["time"],
        frame["predicted_kwh"],
        color="#D85A30",
        lw=2.0,
        ls="--",
        marker="s",
        ms=3.0,
        label="Predicted",
    )
    ax_main.set_title(f"Zone {summary.get('zone_id')} Forecast vs Actual", fontsize=15, fontweight="bold")
    ax_main.set_xlabel("Time")
    ax_main.set_ylabel("Load (kWh)")
    ax_main.grid(axis="y", alpha=0.25)
    ax_main.legend(loc="upper left", frameon=False)

    errors = frame["error_kwh"].astype(float)
    colors = ["#0F6E56" if value >= 0 else "#D85A30" for value in errors]
    ax_resid.bar(frame["time"], errors, color=colors, width=0.03, alpha=0.82)
    ax_resid.axhline(0, color="#6B7280", lw=0.9)
    ax_resid.set_title("Residuals (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax_resid.set_xlabel("Time")
    ax_resid.set_ylabel("Error (kWh)")
    ax_resid.grid(axis="y", alpha=0.22)

    metrics = summary.get("metrics", {}) or {}
    metric_lines = [
        ("MAE", format_metric(metrics.get("MAE"))),
        ("RMSE", format_metric(metrics.get("RMSE"))),
        ("MAPE", f"{format_metric(metrics.get('MAPE_pct'))}%"),
        ("RAE", format_metric(metrics.get("RAE"))),
        ("WAPE", f"{format_metric(metrics.get('WAPE_pct'))}%"),
    ]
    ax_metrics.axis("off")
    ax_metrics.set_title("Evaluation", fontsize=11, fontweight="bold", loc="left")
    for idx, (label, value) in enumerate(metric_lines):
        y = 0.88 - idx * 0.16
        ax_metrics.text(0.02, y, label, fontsize=11, color="#374151", transform=ax_metrics.transAxes)
        ax_metrics.text(0.98, y, value, fontsize=11, fontweight="bold", ha="right", transform=ax_metrics.transAxes)

    for axis in (ax_main, ax_resid):
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.DateFormatter("%m-%d %H:%M")
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(formatter)
        axis.tick_params(axis="x", labelbottom=True)
        axis.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for col in display.columns:
        display[col] = display[col].map(format_cell)
    widths = {
        col: max(len(str(col)), *(len(str(value)) for value in display[col]))
        for col in display.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in display.columns) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\n", " ")
    return text[:220] + "..." if len(text) > 223 else text
