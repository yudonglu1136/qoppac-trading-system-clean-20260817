#!/usr/bin/env python3
"""Plot one cross-universe stock forecast comparison chart."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rob_style_stock_backtest as rob_stock  # noqa: E402


DEFAULT_OUT = ROOT / "backtests" / "stock_forecast_sector_idm_vol10_cross_universe"
DEFAULT_STRATEGY = "intermediate_mom__sector_top10_long"
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-08-07"
DEFAULT_CAPITAL = 500_000.0


def load_strategy(out_dir: Path, key: str, strategy: str, capital: float) -> pd.Series:
    path = out_dir / key / f"{strategy}_daily.csv"
    daily = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return (daily["equity"] / capital).rename(f"{rob_stock.SUPPORTED_UNIVERSES[key]} strategy")


def load_benchmark(key: str, start: str, end: str, capital: float) -> pd.Series:
    daily = rob_stock.benchmark_daily(key, start, end, capital)
    label = daily["daily_return"].name.replace("Benchmark: ", "")
    return (daily["equity"] / capital).rename(f"{label} benchmark")


def plot_cross_universe(
    out_dir: Path,
    universes: list[str],
    strategy: str,
    start: str,
    end: str,
    capital: float,
) -> Path:
    streams = {}
    for key in universes:
        streams[f"{key}_strategy"] = load_strategy(out_dir, key, strategy, capital)
        streams[f"{key}_benchmark"] = load_benchmark(key, start, end, capital)

    common_start = max(series.dropna().index.min() for series in streams.values())
    common_end = min(series.dropna().index.max() for series in streams.values())
    aligned = pd.concat({name: series.loc[common_start:common_end] for name, series in streams.items()}, axis=1)
    aligned = aligned.ffill().dropna(how="any")
    aligned = aligned / aligned.iloc[0]

    colors = {"sp500": "#2563eb", "eem": "#059669", "efa": "#d97706"}
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.10)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)

    for key in universes:
        strategy_col = f"{key}_strategy"
        benchmark_col = f"{key}_benchmark"
        color = colors.get(key)
        ax0.plot(aligned.index, aligned[strategy_col], color=color, linewidth=1.8, label=streams[strategy_col].name)
        ax0.plot(
            aligned.index,
            aligned[benchmark_col],
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.75,
            label=streams[benchmark_col].name,
        )
        ax1.plot(aligned.index, aligned[strategy_col] / aligned[strategy_col].cummax() - 1.0, color=color, linewidth=1.2)
        ax1.plot(
            aligned.index,
            aligned[benchmark_col] / aligned[benchmark_col].cummax() - 1.0,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )

    ax0.set_title("Top 10% Sector-Neutral Stock Forecast Across SPY, EM, and Developed ex-US")
    ax0.set_yscale("log")
    ax0.set_ylabel("Growth of $1, rebased to common start")
    ax1.set_ylabel("Drawdown")
    ax1.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax0.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    path = out_dir / "cross_universe_intermediate_mom_vs_benchmarks.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    aligned.to_csv(out_dir / "cross_universe_aligned_equity.csv")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--universes", nargs="+", default=["sp500", "eem", "efa"], choices=sorted(rob_stock.SUPPORTED_UNIVERSES))
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = plot_cross_universe(args.out_dir, args.universes, args.strategy, args.start, args.end, args.capital)
    print(path)


if __name__ == "__main__":
    main()
