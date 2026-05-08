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
    "wape_pct",
    "grid_stress_level",
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
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_zones.csv"
    contexts_path = output_dir / "context_snippets.json"
    trace_csv = output_dir / "rationale_trace.csv"
    trace_md = output_dir / "rationale_trace.md"
    trace_json = output_dir / "rationale_trace.json"

    selected_zones.to_csv(selected_path, index=False)
    contexts_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")

    trace = pd.DataFrame(reports)
    trace = trace[[col for col in TRACE_COLUMNS if col in trace.columns]]
    trace.to_csv(trace_csv, index=False)
    trace_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_md.write_text(markdown_table(trace), encoding="utf-8")

    return {
        "selected_zones": selected_path,
        "context_snippets": contexts_path,
        "rationale_trace_csv": trace_csv,
        "rationale_trace_md": trace_md,
        "rationale_trace_json": trace_json,
    }


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
