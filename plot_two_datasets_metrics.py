#!/usr/bin/env python3
"""
Plot metric comparison for two datasets from datasets_complexity_summary.csv.

Default pair: sonar.csv vs hepatitis.csv
X-axis: metrics
Y-axis: metric value
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


EXCLUDED_COLUMNS = {"dataset_file", "label_column", "error"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two datasets across complexity metrics.")
    parser.add_argument("--input", type=Path, default=Path("data/datasets_complexity_summary.csv"))
    parser.add_argument("--dataset-a", type=str, default="sonar.csv")
    parser.add_argument("--dataset-b", type=str, default="hepatitis.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/complexity_bars/sonar_vs_hepatitis_metrics.html"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "dataset_file" not in df.columns:
        raise SystemExit("Input CSV must contain dataset_file column.")

    pair = df[df["dataset_file"].isin([args.dataset_a, args.dataset_b])].copy()
    if pair["dataset_file"].nunique() != 2:
        present = sorted(pair["dataset_file"].dropna().unique().tolist())
        raise SystemExit(
            f"Need both datasets in input. Found: {present}. "
            f"Requested: {args.dataset_a}, {args.dataset_b}"
        )

    metric_cols: list[str] = []
    for c in df.columns:
        if c in EXCLUDED_COLUMNS:
            continue
        num = pd.to_numeric(pair[c], errors="coerce")
        if num.notna().any():
            metric_cols.append(c)

    # Keep only metrics where at least one dataset has a numeric value.
    metric_cols = [c for c in metric_cols if pd.to_numeric(pair[c], errors="coerce").notna().any()]
    metric_cols = sorted(metric_cols)

    a_row = pair[pair["dataset_file"] == args.dataset_a].iloc[0]
    b_row = pair[pair["dataset_file"] == args.dataset_b].iloc[0]

    y_a = [pd.to_numeric(pd.Series([a_row[c]]), errors="coerce").iloc[0] for c in metric_cols]
    y_b = [pd.to_numeric(pd.Series([b_row[c]]), errors="coerce").iloc[0] for c in metric_cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=args.dataset_a, x=metric_cols, y=y_a))
    fig.add_trace(go.Bar(name=args.dataset_b, x=metric_cols, y=y_b))
    fig.update_layout(
        title=f"Complexity metrics: {args.dataset_a} vs {args.dataset_b}",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        template="plotly_white",
        xaxis_tickangle=-60,
        margin=dict(l=60, r=30, t=70, b=220),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output), include_plotlyjs="cdn", full_html=True)
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
