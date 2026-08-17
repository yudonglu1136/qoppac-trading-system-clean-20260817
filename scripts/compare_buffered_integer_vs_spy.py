#!/usr/bin/env python3
"""Compare the 40-capacity buffered-integer strategy with SPY."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKTEST = ROOT / "backtests" / "rob_style_40_capacity" / "portfolio_daily.csv"
OUT = ROOT / "backtests" / "rob_style_40_capacity_spy_comparison"
YAHOO_HISTORY_PAGE = "https://finance.yahoo.com/quote/SPY/history/"
TRADING_DAYS = 252.0


def yahoo_chart_url(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    start_dt = dt.datetime.combine(start.date(), dt.time(0, 0))
    # Yahoo period2 is exclusive, so move one day past the requested end.
    end_dt = dt.datetime.combine((end + pd.Timedelta(days=1)).date(), dt.time(0, 0))
    params = urlencode(
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "history|div|splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"


def load_buffered_integer_equity() -> pd.Series:
    portfolio = pd.read_csv(BACKTEST, header=[0, 1], index_col=0, parse_dates=True)
    equity = portfolio[("buffered_integer", "equity")].dropna()
    equity.name = "strategy_equity"
    return equity


def fetch_spy_adjusted_close(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, str]:
    url = yahoo_chart_url("SPY", start, end)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        data = json.loads(response.read())

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]
    dates = pd.to_datetime(timestamps, unit="s").normalize()
    series = pd.Series(adj_close, index=dates, name="SPY_adjusted_close", dtype=float).dropna()
    return series[~series.index.duplicated(keep="last")], url


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def max_drawdown_duration(drawdown_series: pd.Series) -> int:
    max_duration = 0
    current = 0
    for value in drawdown_series:
        if value < 0:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_duration


def metrics(nav: pd.Series) -> dict[str, float | str]:
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    elapsed_years = (nav.index[-1] - nav.index[0]).days / 365.25
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
    annual_return = returns.mean() * TRADING_DAYS
    annual_vol = returns.std() * math.sqrt(TRADING_DAYS)
    sharpe = annual_return / annual_vol if annual_vol else np.nan
    dd = drawdown(nav)
    mdd = dd.min()
    return {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "years": elapsed_years,
        "total_return": total_return,
        "cagr": cagr,
        "annual_return_arithmetic": annual_return,
        "annual_vol": annual_vol,
        "sharpe_0rf": sharpe,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd else np.nan,
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "win_rate": (returns > 0).mean(),
        "max_drawdown_duration_trading_days": max_drawdown_duration(dd),
        "ending_value_of_500k": 500_000.0 * nav.iloc[-1] / nav.iloc[0],
    }


def format_metric(key: str, value: float | str) -> str:
    if key in {"start", "end"}:
        return str(value)
    if key == "years":
        return f"{value:.1f}"
    if key in {"sharpe_0rf", "calmar"}:
        return f"{value:.2f}"
    if key == "max_drawdown_duration_trading_days":
        return f"{value:.0f}"
    if key == "ending_value_of_500k":
        return f"${value:,.0f}"
    return f"{value:.2%}"


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    strategy_equity = load_buffered_integer_equity()
    spy_adj, source_url = fetch_spy_adjusted_close(strategy_equity.index.min(), strategy_equity.index.max())

    # Compare on SPY trading days. Strategy equity is forward-filled onto those
    # dates, so returns accrued on non-SPY business days are captured at the
    # next comparable SPY date.
    start = max(strategy_equity.index.min(), spy_adj.index.min())
    end = min(strategy_equity.index.max(), spy_adj.index.max())
    comparison_index = spy_adj.loc[start:end].index
    strategy_on_spy_dates = strategy_equity.reindex(comparison_index, method="ffill")
    spy_on_dates = spy_adj.reindex(comparison_index).dropna()
    strategy_on_spy_dates = strategy_on_spy_dates.reindex(spy_on_dates.index, method="ffill")
    first_strategy_value = strategy_on_spy_dates.dropna().iloc[0]
    first_spy_value = spy_on_dates.dropna().iloc[0]

    nav = pd.DataFrame(
        {
            "Strategy_Buffered_Integer": strategy_on_spy_dates / first_strategy_value,
            "SPY_Adjusted_Close": spy_on_dates / first_spy_value,
        }
    ).dropna()
    returns = nav.pct_change().dropna()
    daily = pd.concat(
        [
            nav,
            returns.add_suffix("_return"),
            drawdown(nav).add_suffix("_drawdown"),
        ],
        axis=1,
    )
    daily.index.name = "date"

    metric_rows = []
    for name in nav.columns:
        row = {"series": name, **metrics(nav[name])}
        metric_rows.append(row)
    metric_df = pd.DataFrame(metric_rows)

    aligned_returns = returns.dropna()
    corr = aligned_returns["Strategy_Buffered_Integer"].corr(aligned_returns["SPY_Adjusted_Close"])
    beta = (
        aligned_returns["Strategy_Buffered_Integer"].cov(aligned_returns["SPY_Adjusted_Close"])
        / aligned_returns["SPY_Adjusted_Close"].var()
    )
    tracking = aligned_returns["Strategy_Buffered_Integer"] - aligned_returns["SPY_Adjusted_Close"]
    extra = pd.DataFrame(
        [
            {"series": "Strategy_vs_SPY", "metric": "daily_correlation", "value": corr},
            {"series": "Strategy_vs_SPY", "metric": "beta_to_spy", "value": beta},
            {
                "series": "Strategy_vs_SPY",
                "metric": "tracking_error",
                "value": tracking.std() * math.sqrt(TRADING_DAYS),
            },
            {
                "series": "Strategy_vs_SPY",
                "metric": "annualised_excess_return_arithmetic",
                "value": tracking.mean() * TRADING_DAYS,
            },
        ]
    )
    return daily, metric_df, source_url, extra


def write_outputs(daily: pd.DataFrame, metrics_df: pd.DataFrame, source_url: str, extra: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "daily_comparison.csv")
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    extra.to_csv(OUT / "relative_metrics.csv", index=False)

    nav_cols = ["Strategy_Buffered_Integer", "SPY_Adjusted_Close"]
    dd_cols = [f"{col}_drawdown" for col in nav_cols]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    daily[nav_cols].plot(ax=axes[0])
    axes[0].set_title("Buffered Integer Strategy vs SPY")
    axes[0].set_ylabel("Growth of $1")
    daily[dd_cols].plot(ax=axes[1])
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(OUT / "strategy_vs_spy.png", dpi=160)

    lines = [
        "# Buffered Integer Strategy vs SPY",
        "",
        "## Data",
        "",
        f"- Strategy: `{BACKTEST}`",
        f"- SPY source: Yahoo Finance adjusted close, `{YAHOO_HISTORY_PAGE}`",
        f"- SPY chart endpoint used: `{source_url}`",
        "- Metrics use zero risk-free rate and daily returns on SPY trading dates.",
        "",
        "## Metrics",
        "",
    ]
    display_keys = [
        "start",
        "end",
        "years",
        "total_return",
        "cagr",
        "annual_vol",
        "sharpe_0rf",
        "max_drawdown",
        "calmar",
        "best_day",
        "worst_day",
        "win_rate",
        "max_drawdown_duration_trading_days",
        "ending_value_of_500k",
    ]
    header = "| Series | " + " | ".join(display_keys) + " |"
    divider = "|---" * (len(display_keys) + 1) + "|"
    lines.extend([header, divider])
    for row in metrics_df.to_dict("records"):
        values = [row["series"]] + [format_metric(key, row[key]) for key in display_keys]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Relative Metrics", ""])
    for row in extra.to_dict("records"):
        value = row["value"]
        if row["metric"] in {"daily_correlation", "beta_to_spy"}:
            formatted = f"{value:.2f}"
        else:
            formatted = f"{value:.2%}"
        lines.append(f"- {row['metric']}: {formatted}")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    daily, metrics_df, source_url, extra = build_comparison()
    write_outputs(daily, metrics_df, source_url, extra)
    print(f"Wrote comparison to {OUT}")
    print(metrics_df.to_string(index=False))
    print(extra.to_string(index=False))


if __name__ == "__main__":
    main()
