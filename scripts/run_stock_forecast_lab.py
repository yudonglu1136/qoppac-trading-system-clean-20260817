#!/usr/bin/env python3
"""Test equity-specific forecasts inside the Rob-style stock portfolio engine."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_benchmark_aware_stock_momentum as baw  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402


OUT = ROOT / "backtests" / "stock_forecast_lab"
BUSINESS_DAYS = 252.0
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-08-07"
DEFAULT_CAPITAL = 500_000.0
DEFAULT_VOL_TARGET = 0.25
DEFAULT_IDM = 2.75
DEFAULT_WEIGHT_MODE = "benchmark"
FORECAST_CAP = 20.0


FORECAST_SETS = {
    "mom_12_1": ["mom_12_1"],
    "mom_6_1": ["mom_6_1"],
    "intermediate_mom": ["intermediate_mom"],
    "high_52w": ["high_52w"],
    "residual_mom": ["residual_mom"],
    "low_vol": ["low_vol"],
    "short_reversal": ["short_reversal"],
    "sector_rel_mom_12_1": ["sector_rel_mom_12_1"],
    "sector_rel_mom_6_1": ["sector_rel_mom_6_1"],
    "sector_rel_intermediate_mom": ["sector_rel_intermediate_mom"],
    "classic_mom_combo": ["mom_12_1", "mom_6_1", "intermediate_mom", "high_52w", "residual_mom"],
    "defensive_mom_combo": ["mom_12_1", "intermediate_mom", "high_52w", "residual_mom", "low_vol"],
    "mom_reversal_combo": ["mom_12_1", "high_52w", "residual_mom", "short_reversal"],
    "stock_alpha_combo": {
        "residual_mom": 0.30,
        "mom_12_1": 0.25,
        "sector_rel_mom_12_1": 0.15,
        "high_52w": 0.10,
        "low_vol": 0.10,
        "short_reversal": 0.10,
    },
    "stock_alpha_mom_only": {
        "residual_mom": 0.35,
        "mom_12_1": 0.30,
        "sector_rel_mom_12_1": 0.20,
        "high_52w": 0.15,
    },
    "stock_alpha_defensive": {
        "mom_12_1": 0.25,
        "residual_mom": 0.25,
        "sector_rel_mom_12_1": 0.20,
        "low_vol": 0.20,
        "short_reversal": 0.10,
    },
}


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def active_mask(annual: pd.DataFrame, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    weights = rob_stock.daily_base_weights(annual, columns, index, "equal")
    return weights > 0.0


def rank_forecast(signal: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    masked = signal.where(active).replace([np.inf, -np.inf], np.nan)
    ranks = masked.rank(axis=1, pct=True)
    counts = masked.notna().sum(axis=1)
    forecast = (ranks - 0.5) * 40.0
    forecast = forecast.where(counts >= 50)
    return forecast.clip(-FORECAST_CAP, FORECAST_CAP)


def trailing_return(price: pd.DataFrame, start_lag: int, end_lag: int) -> pd.DataFrame:
    return price.shift(end_lag) / price.shift(start_lag) - 1.0


def residual_momentum(price: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    returns = price.pct_change(fill_method=None).mask(lambda frame: frame.abs() > rob_stock.MAX_ABS_DAILY_RETURN)
    shifted_stock = returns.shift(21)
    shifted_benchmark = benchmark_returns.reindex(returns.index).shift(21)
    beta = shifted_stock.rolling(252, min_periods=126).cov(shifted_benchmark)
    beta = beta.div(shifted_benchmark.rolling(252, min_periods=126).var(), axis=0)
    stock_mom = shifted_stock.rolling(252, min_periods=126).sum()
    bench_mom = shifted_benchmark.rolling(252, min_periods=126).sum()
    return stock_mom.sub(beta.mul(bench_mom, axis=0), axis=0)


def sector_relative_signal(signal: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in annual.columns:
        return signal * np.nan
    sector_median = rob_stock.dynamic_group_matrix(signal, annual, "median")
    return signal - sector_median


def raw_signal_library(
    price: pd.DataFrame,
    benchmark_returns: pd.Series,
    annual: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    rolling_high = price.shift(1).rolling(252, min_periods=126).max()
    vol_126 = price.pct_change(fill_method=None).rolling(126, min_periods=63).std()
    recent_21 = price.shift(1) / price.shift(22) - 1.0
    signals = {
        "mom_12_1": trailing_return(price, 252, 21),
        "mom_6_1": trailing_return(price, 126, 21),
        "intermediate_mom": trailing_return(price, 252, 126),
        "high_52w": price.shift(1) / rolling_high - 1.0,
        "residual_mom": residual_momentum(price, benchmark_returns),
        "low_vol": -vol_126,
        "short_reversal": -recent_21,
    }
    if annual is not None:
        signals["sector_rel_mom_12_1"] = sector_relative_signal(signals["mom_12_1"], annual)
        signals["sector_rel_mom_6_1"] = sector_relative_signal(signals["mom_6_1"], annual)
        signals["sector_rel_intermediate_mom"] = sector_relative_signal(signals["intermediate_mom"], annual)
    return signals


def combine_forecast_set(
    library: dict[str, pd.DataFrame],
    rule_names: list[str] | dict[str, float],
    active: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(rule_names, dict):
        weights = pd.Series(rule_names, dtype=float)
        weights = weights / weights.sum()
    else:
        weights = pd.Series(1.0 / len(rule_names), index=rule_names, dtype=float)

    missing = [name for name in weights.index if name not in library]
    if missing:
        raise KeyError(f"Forecast rules missing from signal library: {missing}")

    forecasts = [rank_forecast(library[name], active) for name in weights.index]
    stacked = pd.concat(forecasts, axis=1, keys=weights.index)
    weighted = stacked.fillna(0.0).mul(weights, axis=1, level=0)
    available_weight = stacked.notna().astype(float).mul(weights, axis=1, level=0)
    combined = weighted.T.groupby(level=1).sum().T
    combined = combined.div(available_weight.T.groupby(level=1).sum().T.replace(0.0, np.nan))
    fdm = min(math.sqrt(len(weights)), 2.0)
    combined = (combined * fdm).clip(-FORECAST_CAP, FORECAST_CAP)
    rule_table = pd.DataFrame(
        {
            "rule": weights.index,
            "forecast_weight": weights.to_numpy(),
            "forecast_div_multiplier": fdm,
            "scaling": "cross-sectional rank mapped to [-20, 20]",
        }
    )
    return combined, rule_table


def performance_stats(daily: pd.DataFrame, capital: float) -> dict[str, float | str]:
    return rob_stock.performance_stats_from_equity(daily["daily_return"], daily["equity"] / capital)


def diagnostics_row(name: str, daily: pd.DataFrame, capital: float) -> dict[str, float | str]:
    active = rob_stock.trim_active_daily(daily)
    return {
        "strategy": name,
        "avg_turnover_annual": active["turnover"].mean() * BUSINESS_DAYS,
        "avg_cost_annual": active["costs"].mean() * BUSINESS_DAYS / capital,
        "avg_gross_exposure": active["gross_exposure"].mean(),
        "avg_net_exposure": active["net_exposure"].mean(),
        "avg_long_exposure": active["long_exposure"].mean(),
        "avg_short_exposure": active["short_exposure"].mean(),
        "avg_active_names": active["active_names"].mean(),
    }


def yearly_returns(daily_by_name: dict[str, pd.DataFrame], capital: float) -> pd.DataFrame:
    return rob_stock.yearly_returns_from_equity(daily_by_name, capital)


def plot_universe(key: str, daily_by_name: dict[str, pd.DataFrame], annual: pd.DataFrame, out_dir: Path, capital: float) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.1, 1.6], hspace=0.16)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[2, 0])

    for name, daily in daily_by_name.items():
        equity = daily["equity"].dropna() / capital
        ax0.plot(equity.index, equity, label=name, linewidth=1.4)
        ax1.plot(equity.index, equity / equity.cummax() - 1.0, linewidth=0.9)

    annual = annual.dropna(how="all")
    x = np.arange(len(annual.index))
    width = min(0.12, 0.8 / max(len(annual.columns), 1))
    for i, name in enumerate(annual.columns):
        ax2.bar(x + (i - (len(annual.columns) - 1) / 2) * width, annual[name], width=width, label=name)
    ax0.set_title(f"{rob_stock.SUPPORTED_UNIVERSES[key]} Equity Forecast Lab")
    ax0.set_yscale("log")
    ax0.set_ylabel("Growth of $1")
    ax1.set_ylabel("Drawdown")
    ax2.set_ylabel("Year return")
    ax2.axhline(0.0, color="#555555", linewidth=0.8)
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([str(year) for year in annual.index[::2]], rotation=45, ha="right")
    ax0.legend(loc="upper left", ncol=3, fontsize=8)
    ax2.legend(loc="upper left", ncol=3, fontsize=7)
    ax1.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{key}_forecast_lab.png", dpi=180)
    plt.close(fig)


def run_universe(
    key: str,
    forecast_sets: list[str],
    start: str,
    end: str,
    *,
    weight_mode: str,
    capital: float,
    vol_target: float,
    idm: float,
) -> dict[str, pd.DataFrame]:
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)

    annual = rob_stock.load_annual(key, start, end)
    price = rob_stock.load_price(key, annual, start, end)
    benchmark = baw.load_benchmark(key, start, end)
    library = raw_signal_library(price, benchmark, annual)
    active = active_mask(annual, price.columns, price.index)
    price_vol = rob_stock.rob.mixed_vol(price.diff())

    daily_by_name: dict[str, pd.DataFrame] = {}
    stats_rows = []
    diagnostics_rows = []
    rule_tables = []
    for forecast_set in forecast_sets:
        rule_names = FORECAST_SETS[forecast_set]
        forecast, rule_table = combine_forecast_set(library, rule_names, active)
        positions, _target, _instrument_weights, risk = rob_stock.target_positions(
            price,
            price_vol,
            forecast,
            annual,
            weight_mode,
            capital=capital,
            vol_target=vol_target,
            idm=idm,
        )
        daily, _by_instrument = rob_stock.pnl_from_stock_positions(
            positions,
            price,
            capital,
            rob_stock.DEFAULT_COST_PER_DOLLAR,
        )
        label = forecast_set
        daily = daily.join(risk).loc[start:end]
        daily_by_name[label] = rob_stock.trim_active_daily(daily)
        stats_rows.append({"strategy": label, **performance_stats(daily_by_name[label], capital)})
        diagnostics_rows.append(diagnostics_row(label, daily_by_name[label], capital))
        rule_table.insert(0, "forecast_set", forecast_set)
        rule_tables.append(rule_table)
        daily.to_csv(out_dir / f"{label}_daily.csv")

    benchmark_daily = rob_stock.benchmark_daily(key, start, end, capital)
    common_start = max(frame.index.min() for frame in daily_by_name.values())
    common_end = min(min(frame.index.max() for frame in daily_by_name.values()), benchmark_daily.index.max())
    daily_by_name = {name: frame.loc[common_start:common_end] for name, frame in daily_by_name.items()}
    benchmark_daily = rob_stock.rebase_daily(benchmark_daily.loc[common_start:common_end], capital, compound=True)
    daily_by_name[benchmark.name] = benchmark_daily
    stats_rows.append({"strategy": benchmark.name, **performance_stats(benchmark_daily, capital)})

    stats = pd.DataFrame(stats_rows)
    diagnostics = pd.DataFrame(diagnostics_rows)
    annual_returns = yearly_returns(daily_by_name, capital)
    rules = pd.concat(rule_tables, ignore_index=True)
    stats.to_csv(out_dir / "stats.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    annual_returns.to_csv(out_dir / "yearly_returns.csv")
    rules.to_csv(out_dir / "forecast_rules.csv", index=False)
    plot_universe(key, daily_by_name, annual_returns, out_dir, capital)
    write_universe_summary(key, stats, diagnostics, rules, out_dir, start, end, weight_mode, vol_target, idm)
    return {
        "stats": stats.assign(universe=rob_stock.SUPPORTED_UNIVERSES[key]),
        "diagnostics": diagnostics.assign(universe=rob_stock.SUPPORTED_UNIVERSES[key]),
    }


def write_universe_summary(
    key: str,
    stats: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rules: pd.DataFrame,
    out_dir: Path,
    start: str,
    end: str,
    weight_mode: str,
    vol_target: float,
    idm: float,
) -> None:
    lines = [
        f"# {rob_stock.SUPPORTED_UNIVERSES[key]} Forecast Lab",
        "",
        f"- Sample requested: {start} to {end}.",
        f"- Portfolio engine: Rob-style stock sizing, `{weight_mode}` instrument weights, vol target {vol_target:.0%}, IDM {idm:.2f}.",
        "- Forecast scaling: cross-sectional ranks among point-in-time active constituents mapped to [-20, 20]; combo FDM = min(sqrt(number of rules), 2).",
        "- Candidate evidence base: 12-1 momentum, 6-1 momentum, intermediate momentum, 52-week high, residual momentum, low volatility, and one-month reversal.",
        "",
        "## Performance",
        "",
        "| Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )
    lines += [
        "",
        "## Diagnostics",
        "",
        "| Strategy | Ann Turnover | Ann Cost | Avg Gross | Avg Net | Avg Long | Avg Short | Avg Names |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {row['strategy']} | {pct(row['avg_turnover_annual'])} | {pct(row['avg_cost_annual'])} | {pct(row['avg_gross_exposure'])} | {pct(row['avg_net_exposure'])} | {pct(row['avg_long_exposure'])} | {pct(row['avg_short_exposure'])} | {row['avg_active_names']:.0f} |"
        )
    lines += [
        "",
        "## Forecast Sets",
        "",
    ]
    for forecast_set, frame in rules.groupby("forecast_set"):
        names = ", ".join(frame["rule"])
        lines.append(f"- `{forecast_set}`: {names}.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_summary(all_stats: pd.DataFrame, all_diag: pd.DataFrame) -> None:
    lines = [
        "# Stock Forecast Lab",
        "",
        "Equity-specific price forecasts tested inside the same Rob-style stock execution and risk framework.",
        "",
        "## Performance",
        "",
        "| Universe | Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_stats.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )
    lines += [
        "",
        "## Diagnostics",
        "",
        "| Universe | Strategy | Ann Turnover | Ann Cost | Avg Gross | Avg Net | Avg Long | Avg Short | Avg Names |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_diag.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {pct(row['avg_turnover_annual'])} | {pct(row['avg_cost_annual'])} | {pct(row['avg_gross_exposure'])} | {pct(row['avg_net_exposure'])} | {pct(row['avg_long_exposure'])} | {pct(row['avg_short_exposure'])} | {row['avg_active_names']:.0f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", nargs="+", default=["sp500", "eem", "efa"], choices=sorted(rob_stock.SUPPORTED_UNIVERSES))
    parser.add_argument("--forecast-sets", nargs="+", default=list(FORECAST_SETS), choices=sorted(FORECAST_SETS))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--weight-mode", default=DEFAULT_WEIGHT_MODE, choices=sorted(rob_stock.WEIGHT_MODES))
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--vol-target", type=float, default=DEFAULT_VOL_TARGET)
    parser.add_argument("--idm", type=float, default=DEFAULT_IDM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = []
    all_diag = []
    for key in args.universes:
        result = run_universe(
            key,
            args.forecast_sets,
            args.start,
            args.end,
            weight_mode=args.weight_mode,
            capital=args.capital,
            vol_target=args.vol_target,
            idm=args.idm,
        )
        all_stats.append(result["stats"])
        all_diag.append(result["diagnostics"])
    stats = pd.concat(all_stats, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True)
    stats.to_csv(OUT / "all_stats.csv", index=False)
    diag.to_csv(OUT / "all_diagnostics.csv", index=False)
    write_combined_summary(stats, diag)
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
