#!/usr/bin/env python3
"""Summarize stock forecast matrix runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "backtests" / "stock_forecast_method_matrix_vol10"


METRICS = {
    "sharpe": "Sharpe",
    "cagr": "CAGR",
    "vol": "Volatility",
    "mdd": "Max drawdown",
    "avg_cost_annual": "Annual cost",
    "avg_turnover_annual": "Annual turnover",
    "avg_gross_exposure": "Avg gross exposure",
    "avg_idm": "Avg IDM",
}


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def split_strategy(strategy: str) -> tuple[str, str]:
    if "__" not in strategy:
        return strategy, ""
    forecast, selection = strategy.rsplit("__", 1)
    return forecast, selection


def load_matrix(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = pd.read_csv(input_dir / "all_stats.csv")
    diagnostics = pd.read_csv(input_dir / "all_diagnostics.csv")
    counts = pd.read_csv(input_dir / "all_selection_counts.csv")
    for frame in (stats, diagnostics, counts):
        if "strategy" in frame.columns:
            parsed = frame["strategy"].map(split_strategy)
            frame["forecast_set"] = parsed.map(lambda item: item[0])
            frame["selection_mode"] = parsed.map(lambda item: item[1])
    return stats, diagnostics, counts


def combined_results(stats: pd.DataFrame, diagnostics: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    non_benchmark = stats[~stats["strategy"].str.startswith("Benchmark")].copy()
    out = non_benchmark.merge(
        diagnostics.drop(columns=["forecast_set", "selection_mode"], errors="ignore"),
        on=["strategy", "universe"],
        how="left",
    )
    out = out.merge(
        counts.drop(columns=["forecast_set", "selection_mode"], errors="ignore"),
        on=["strategy", "universe"],
        how="left",
    )
    parsed = out["strategy"].map(split_strategy)
    out["forecast_set"] = parsed.map(lambda item: item[0])
    out["selection_mode"] = parsed.map(lambda item: item[1])
    return out


def benchmark_table(stats: pd.DataFrame) -> pd.DataFrame:
    return stats[stats["strategy"].str.startswith("Benchmark")].copy()


def metric_matrix(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = results.copy()
    frame["column"] = frame["universe"] + " | " + frame["selection_mode"]
    matrix = frame.pivot_table(index="forecast_set", columns="column", values=metric, aggfunc="first")
    return matrix.sort_index()


def plot_heatmap(matrix: pd.DataFrame, metric: str, out_path: Path) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(12, matrix.shape[1] * 2.2), max(6, matrix.shape[0] * 0.36)))
    cmap = "RdYlGn" if metric not in {"mdd", "avg_cost_annual", "avg_turnover_annual", "vol"} else "RdYlGn_r"
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(METRICS.get(metric, metric))
    fig.colorbar(image, ax=ax, shrink=0.85)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = values[row, col]
            if not np.isfinite(value):
                continue
            label = f"{value:.2f}" if metric == "sharpe" else f"{value:.0%}"
            ax.text(col, row, label, ha="center", va="center", fontsize=7, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(results: pd.DataFrame, benchmarks: pd.DataFrame, output_dir: Path) -> None:
    top = results.sort_values(["universe", "selection_mode", "sharpe"], ascending=[True, True, False])
    best = top.groupby(["universe", "selection_mode"], as_index=False).head(5)

    average_rank = (
        results.groupby(["forecast_set", "selection_mode"])
        .agg(
            mean_sharpe=("sharpe", "mean"),
            median_sharpe=("sharpe", "median"),
            mean_cagr=("cagr", "mean"),
            mean_mdd=("mdd", "mean"),
            mean_cost=("avg_cost_annual", "mean"),
            markets=("universe", "nunique"),
        )
        .reset_index()
        .sort_values("mean_sharpe", ascending=False)
    )

    lines = [
        "# Stock Forecast Method Matrix",
        "",
        "All rows use the same Rob-style stock risk system. Only the forecast set and top-10% selection rule vary.",
        "",
        "## Best By Market And Selection",
        "",
        "| Universe | Selection | Forecast | CAGR | Vol | Sharpe | MDD | Ann Cost | Avg Names |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['universe']} | {row['selection_mode']} | {row['forecast_set']} | "
            f"{pct(row['cagr'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | "
            f"{pct(row['avg_cost_annual'])} | {row['avg_selected_names']:.0f} |"
        )

    lines += [
        "",
        "## Average Across Markets",
        "",
        "| Selection | Forecast | Mean Sharpe | Median Sharpe | Mean CAGR | Mean MDD | Mean Ann Cost | Markets |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in average_rank.iterrows():
        lines.append(
            f"| {row['selection_mode']} | {row['forecast_set']} | {num(row['mean_sharpe'])} | "
            f"{num(row['median_sharpe'])} | {pct(row['mean_cagr'])} | {pct(row['mean_mdd'])} | "
            f"{pct(row['mean_cost'])} | {int(row['markets'])} |"
        )

    lines += [
        "",
        "## Benchmarks",
        "",
        "| Universe | Benchmark | CAGR | Vol | Sharpe | MDD |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in benchmarks.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {pct(row['cagr'])} | {pct(row['vol'])} | "
            f"{num(row['sharpe'])} | {pct(row['mdd'])} |"
        )

    lines += [
        "",
        "## Files",
        "",
    ]
    for metric in METRICS:
        lines.append(f"- `{metric}_matrix.csv` and `{metric}_heatmap.png`.")

    (output_dir / "matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    stats, diagnostics, counts = load_matrix(args.input_dir)
    results = combined_results(stats, diagnostics, counts)
    benchmarks = benchmark_table(stats)
    out_dir = args.input_dir / "matrix_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "combined_results.csv", index=False)
    benchmarks.to_csv(out_dir / "benchmarks.csv", index=False)
    for metric in METRICS:
        matrix = metric_matrix(results, metric)
        matrix.to_csv(out_dir / f"{metric}_matrix.csv")
        plot_heatmap(matrix, metric, out_dir / f"{metric}_heatmap.png")
    write_report(results, benchmarks, out_dir)
    print(out_dir / "matrix_report.md")


if __name__ == "__main__":
    main()
