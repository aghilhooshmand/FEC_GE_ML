#!/usr/bin/env python3
"""
Create one HTML report with grouped Plotly bar charts for complexity metrics.

Reads:  data/datasets_complexity_summary.csv
Writes: a single HTML file containing multiple charts:
  - One chart per metric group (e.g. c*, f*, l*, n*, t*).
  - Each chart has one bar trace per metric (legend).
  - X-axis is dataset_file.

Usage:
  python3 plot_complexity_bars.py
  python3 plot_complexity_bars.py --input data/datasets_complexity_summary.csv --output plots/complexity_bars/complexity_grouped_report.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


EXCLUDED_COLUMNS = {
    "dataset_file",
    "label_column",
    "error",
}


def _candidate_metric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in EXCLUDED_COLUMNS:
            continue
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().any():
            cols.append(c)
    return sorted(cols)


def _metric_group(metric: str) -> str:
    m = metric.lower()
    if m.startswith("c"):
        return "c"
    if m.startswith("f"):
        return "f"
    if m.startswith("l"):
        return "l"
    if m.startswith("n"):
        return "n"
    if m.startswith("t"):
        return "t"
    return "other"


def _group_metrics(metrics: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for metric in metrics:
        grp = _metric_group(metric)
        groups.setdefault(grp, []).append(metric)
    return groups


def _build_group_chart(df: pd.DataFrame, group_name: str, metrics: list[str]) -> go.Figure | None:
    # Sort datasets by the first metric in the group (high to low).
    sort_metric = metrics[0]
    temp = df[["dataset_file"] + metrics].copy()
    for metric in metrics:
        temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
    temp = temp.sort_values(sort_metric, ascending=False, na_position="last")
    if temp.empty:
        return None

    fig = go.Figure()
    x_vals = temp["dataset_file"].tolist()
    for metric in metrics:
        y_vals = temp[metric].tolist()
        fig.add_trace(go.Bar(name=metric, x=x_vals, y=y_vals))

    fig.update_layout(
        title=f"Group '{group_name}' metrics by dataset (sorted by {sort_metric}, high to low)",
        template="plotly_white",
        xaxis_title="dataset_file",
        yaxis_title="metric value",
        xaxis_tickangle=-60,
        barmode="group",
        margin=dict(l=60, r=30, t=70, b=180),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one HTML report with grouped metric bar charts from datasets_complexity_summary.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/datasets_complexity_summary.csv"),
        help="Input complexity summary CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/complexity_bars/complexity_grouped_report.html"),
        help="Output HTML report path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "dataset_file" not in df.columns:
        raise SystemExit("Input CSV must contain 'dataset_file' column.")

    metrics = _candidate_metric_columns(df)
    if not metrics:
        raise SystemExit("No numeric metric columns found to plot.")

    groups = _group_metrics(metrics)
    group_order = ["c", "f", "l", "n", "t", "other"]

    sections: list[str] = []
    for group_name in group_order:
        group_metrics = groups.get(group_name, [])
        if not group_metrics:
            continue
        fig = _build_group_chart(df, group_name, group_metrics)
        if fig is None:
            continue
        sections.append(f"<h2>Group: {group_name}</h2>")
        sections.append(
            "<p>Bars are grouped by metric (legend). X-axis is dataset. "
            f"Datasets are sorted descending by <b>{group_metrics[0]}</b>.</p>"
        )
        sections.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))

    if not sections:
        raise SystemExit("No charts were created.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    html = (
        "<html><head><meta charset='utf-8' />"
        "<title>Datasets Complexity Grouped Charts</title>"
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>"
        "</head><body>"
        "<h1>Datasets Complexity Grouped Charts</h1>"
        + "".join(sections)
        + "</body></html>"
    )
    args.output.write_text(html, encoding="utf-8")
    print(f"Created grouped report: {args.output}")


if __name__ == "__main__":
    main()
