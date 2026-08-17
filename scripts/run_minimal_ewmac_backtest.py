#!/usr/bin/env python3
"""Run a small EWMAC futures backtest from pysystemtrade's shipped CSV data.

This is intentionally lightweight: it reads adjusted futures prices directly
from the repo's CSV files and implements the core Rob Carver mechanics:
forecast -> volatility target sizing -> approximate futures P&L.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PST = ROOT / "github" / "pysystemtrade"
FUTURES_DATA = PST / "data" / "futures"
ADJUSTED = FUTURES_DATA / "adjusted_prices_csv"
CONFIG = FUTURES_DATA / "csvconfig"
OUT = ROOT / "backtests" / "minimal_ewmac"


@dataclass(frozen=True)
class InstrumentMeta:
    point_size: float
    currency: str
    asset_class: str
    per_block: float
    percentage: float
    per_trade: float
    spread_cost: float


def load_meta() -> dict[str, InstrumentMeta]:
    instruments: dict[str, dict[str, str]] = {}
    with (CONFIG / "instrumentconfig.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            instruments[row["Instrument"]] = row

    spread_costs: dict[str, float] = {}
    with (CONFIG / "spreadcosts.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            spread_costs[row["Instrument"]] = float(row["SpreadCost"] or 0.0)

    meta: dict[str, InstrumentMeta] = {}
    for instrument, row in instruments.items():
        meta[instrument] = InstrumentMeta(
            point_size=float(row["Pointsize"] or 0.0),
            currency=row["Currency"],
            asset_class=row["AssetClass"],
            per_block=float(row["PerBlock"] or 0.0),
            percentage=float(row["Percentage"] or 0.0),
            per_trade=float(row["PerTrade"] or 0.0),
            spread_cost=spread_costs.get(instrument, 0.0),
        )
    return meta


def load_daily_adjusted_price(instrument: str) -> pd.Series:
    path = ADJUSTED / f"{instrument}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, parse_dates=["DATETIME"])
    data = data.set_index("DATETIME")["price"].sort_index()
    daily = data.resample("1B").last().ffill()
    daily.name = instrument
    return daily


def ewmac_forecast(price: pd.Series, fast: int = 32, slow: int = 128) -> tuple[pd.Series, pd.Series]:
    returns = price.diff()
    fast_vol = returns.ewm(span=35, min_periods=10).std()
    slow_vol = returns.ewm(span=35 * 20, min_periods=100).std()
    vol = 0.65 * fast_vol + 0.35 * slow_vol
    vol = vol.ffill().replace(0.0, np.nan)

    raw = (price.ewm(span=fast, min_periods=fast).mean() - price.ewm(span=slow, min_periods=slow).mean()) / vol
    forecast = raw.clip(lower=-20.0, upper=20.0)
    forecast.name = "forecast"
    vol.name = "daily_price_vol"
    return forecast, vol


def run_instrument(
    instrument: str,
    meta: InstrumentMeta,
    capital: float,
    annual_vol_target: float,
    weight: float,
    diversification_multiplier: float,
    fast: int,
    slow: int,
    average_absolute_forecast: float,
) -> pd.DataFrame:
    price = load_daily_adjusted_price(instrument)
    forecast, price_vol = ewmac_forecast(price, fast=fast, slow=slow)
    daily_cash_vol_target = capital * annual_vol_target / math.sqrt(256.0)
    instrument_cash_vol = price_vol.abs() * meta.point_size
    subsystem_position = daily_cash_vol_target / instrument_cash_vol
    target_position = (
        subsystem_position
        * forecast
        / average_absolute_forecast
        * weight
        * diversification_multiplier
    )
    rounded_position = target_position.round()
    held_position = rounded_position.shift(1).fillna(0.0)

    price_change = price.diff().fillna(0.0)
    gross_pnl = held_position * price_change * meta.point_size

    trades = rounded_position.diff().abs().fillna(0.0)
    approximate_cost_per_contract = meta.per_block + meta.per_trade + meta.spread_cost * meta.point_size
    costs = trades * approximate_cost_per_contract
    net_pnl = gross_pnl - costs

    out = pd.DataFrame(
        {
            "price": price,
            "forecast": forecast,
            "price_vol": price_vol,
            "target_position": target_position,
            "rounded_position": rounded_position,
            "held_position": held_position,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "net_pnl": net_pnl,
        }
    )
    out["instrument"] = instrument
    return out


def stats_from_portfolio(portfolio: pd.DataFrame, capital: float) -> dict[str, float]:
    daily_returns = portfolio["daily_return"].dropna()
    if daily_returns.empty:
        return {}
    ann_return = daily_returns.mean() * 256.0
    ann_vol = daily_returns.std() * math.sqrt(256.0)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    return {
        "start": str(daily_returns.index.min().date()),
        "end": str(daily_returns.index.max().date()),
        "years": len(daily_returns) / 256.0,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": portfolio["drawdown"].min(),
        "total_return": portfolio["equity"].iloc[-1] / capital - 1.0,
    }


def format_stat(key: str, value: float | str) -> str:
    if key in {"start", "end"}:
        return str(value)
    if key == "years":
        return f"{value:.1f}"
    if key == "sharpe":
        return f"{value:.2f}"
    return f"{value:.2%}"


def write_summary(stats: dict[str, float], instruments: list[str], capital: float) -> None:
    lines = [
        "# Minimal EWMAC Backtest",
        "",
        "## Data Source",
        "",
        f"- Adjusted prices: `{ADJUSTED}`",
        f"- Instrument metadata: `{CONFIG / 'instrumentconfig.csv'}`",
        f"- Spread costs: `{CONFIG / 'spreadcosts.csv'}`",
        "- These are the sample CSV files shipped with `pysystemtrade`; the repo docs warn these data are stale and suitable for learning mechanics, not live research.",
        "",
        "## Setup",
        "",
        f"- Instruments: {', '.join(instruments)}",
        f"- Capital: ${capital:,.0f}",
        "- Annual volatility target: 25%",
        "- Rule: EWMAC 32/128 on back-adjusted futures prices",
        "- Forecast cap/floor: +/-20",
        "- Average absolute forecast denominator: 10",
        "- Position rounding: nearest whole contract",
        "- Cost model: approximate spread + per-block/per-trade cost from shipped CSV config",
        "",
        "## Results",
        "",
    ]
    for key, value in stats.items():
        lines.append(f"- {key}: {format_stat(key, value)}")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instruments = ["SP500_micro", "NASDAQ_micro", "GOLD_micro", "CRUDE_W"]
    capital = 500_000.0
    annual_vol_target = 0.25
    diversification_multiplier = math.sqrt(len(instruments))
    average_absolute_forecast = 10.0
    fast = 32
    slow = 128

    meta = load_meta()
    frames = []
    for instrument in instruments:
        frames.append(
            run_instrument(
                instrument=instrument,
                meta=meta[instrument],
                capital=capital,
                annual_vol_target=annual_vol_target,
                weight=1.0 / len(instruments),
                diversification_multiplier=diversification_multiplier,
                fast=fast,
                slow=slow,
                average_absolute_forecast=average_absolute_forecast,
            )
        )

    common_start = max(frame["forecast"].first_valid_index() for frame in frames)
    frames = [frame.loc[common_start:] for frame in frames]
    panel = pd.concat(frames)
    panel.to_csv(OUT / "instrument_panel.csv")

    pnl_by_instrument = panel.reset_index().pivot_table(
        index="DATETIME", columns="instrument", values="net_pnl", aggfunc="sum"
    ).sort_index()
    pnl_by_instrument = pnl_by_instrument.fillna(0.0)
    portfolio = pd.DataFrame(index=pnl_by_instrument.index)
    portfolio["gross_pnl"] = panel.reset_index().pivot_table(
        index="DATETIME", columns="instrument", values="gross_pnl", aggfunc="sum"
    ).sort_index().fillna(0.0).sum(axis=1)
    portfolio["costs"] = panel.reset_index().pivot_table(
        index="DATETIME", columns="instrument", values="costs", aggfunc="sum"
    ).sort_index().fillna(0.0).sum(axis=1)
    portfolio["net_pnl"] = pnl_by_instrument.sum(axis=1)
    portfolio["daily_return"] = portfolio["net_pnl"] / capital
    portfolio["equity"] = capital + portfolio["net_pnl"].cumsum()
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1.0
    portfolio.to_csv(OUT / "portfolio_daily.csv")

    stats = stats_from_portfolio(portfolio, capital)
    write_summary(stats, instruments, capital)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    portfolio["equity"].plot(ax=axes[0], title="Minimal EWMAC Portfolio Equity")
    axes[0].set_ylabel("USD")
    portfolio["drawdown"].plot(ax=axes[1], title="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown.png", dpi=160)

    print(f"Wrote results to {OUT}")
    for key, value in stats.items():
        print(f"{key}: {format_stat(key, value)}")


if __name__ == "__main__":
    main()
