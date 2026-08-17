#!/usr/bin/env python3
"""Compare the 17 selected and 40 no-equity strategies with DBMF and KMLM."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {
    "17 selected": ROOT / "backtests" / "rob_style_us_rates_selected_no_vol" / "portfolio_daily.csv",
    "40 no-equity": ROOT / "backtests" / "rob_style_no_equity_40" / "portfolio_daily.csv",
}
ETF_SYMBOLS = ["DBMF", "KMLM"]
OUT = ROOT / "backtests" / "strategies_vs_dbmf_kmlm"
TRADING_DAYS = 252.0

COLORS = {
    "17 selected": "#1f4e79",
    "40 no-equity": "#6f8f2f",
    "DBMF": "#8f5a12",
    "KMLM": "#8a3f73",
}


def yahoo_chart_url(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    start_dt = dt.datetime.combine(start.date(), dt.time(0, 0))
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


def fetch_adjusted_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, str]:
    url = yahoo_chart_url(symbol, start, end)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        data = json.loads(response.read())

    result = data["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s").normalize()
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]
    series = pd.Series(adj_close, index=dates, name=f"{symbol}_adjusted_close", dtype=float).dropna()
    return series[~series.index.duplicated(keep="last")], url


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def load_strategy_equity(path: Path) -> pd.Series:
    portfolio = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return portfolio[("buffered_integer", "equity")].dropna()


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    equities = {name: load_strategy_equity(path) for name, path in STRATEGIES.items()}
    strategy_start = min(series.index.min() for series in equities.values())
    strategy_end = min(series.index.max() for series in equities.values())

    etf_closes: dict[str, pd.Series] = {}
    source_urls: dict[str, str] = {}
    for symbol in ETF_SYMBOLS:
        etf_closes[symbol], source_urls[symbol] = fetch_adjusted_close(symbol, strategy_start, strategy_end)

    start = max(
        max(series.index.min() for series in equities.values()),
        max(close.index.min() for close in etf_closes.values()),
    )
    end = min(strategy_end, *(close.index.max() for close in etf_closes.values()))

    comparison_index = etf_closes[ETF_SYMBOLS[0]].loc[start:end].index
    for symbol in ETF_SYMBOLS[1:]:
        comparison_index = comparison_index.intersection(etf_closes[symbol].loc[start:end].index)

    daily = pd.DataFrame(index=comparison_index)
    for name, equity in equities.items():
        on_dates = equity.reindex(comparison_index, method="ffill")
        daily[f"{name}_nav"] = on_dates / on_dates.dropna().iloc[0]
        daily[f"{name}_return"] = daily[f"{name}_nav"].pct_change()
        daily[f"{name}_drawdown"] = drawdown(daily[f"{name}_nav"])

    for symbol in ETF_SYMBOLS:
        etf_on_dates = etf_closes[symbol].reindex(comparison_index)
        daily[f"{symbol}_adjusted_close"] = etf_on_dates
        daily[f"{symbol}_nav"] = daily[f"{symbol}_adjusted_close"] / daily[f"{symbol}_adjusted_close"].iloc[0]
        daily[f"{symbol}_return"] = daily[f"{symbol}_nav"].pct_change()
        daily[f"{symbol}_drawdown"] = drawdown(daily[f"{symbol}_nav"])

    daily = daily.dropna(subset=[f"{symbol}_nav" for symbol in ETF_SYMBOLS])
    daily.index.name = "date"

    series_order = [*STRATEGIES.keys(), *ETF_SYMBOLS]
    metrics = pd.DataFrame(
        [{"series": name, **performance_metrics(daily[f"{name}_nav"])} for name in series_order]
    )
    annual = annual_return_metrics(daily, series_order)
    return daily, metrics, annual, source_urls


def performance_metrics(nav: pd.Series) -> dict[str, float | str]:
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    elapsed_years = (nav.index[-1] - nav.index[0]).days / 365.25
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
    annual_return = returns.mean() * TRADING_DAYS
    annual_vol = returns.std() * math.sqrt(TRADING_DAYS)
    max_dd = drawdown(nav).min()
    return {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "years": elapsed_years,
        "total_return": total_return,
        "cagr": cagr,
        "annual_return_arithmetic": annual_return,
        "annual_vol": annual_vol,
        "sharpe_0rf": annual_return / annual_vol if annual_vol else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd else np.nan,
    }


def annual_return_metrics(daily: pd.DataFrame, series_order: list[str]) -> pd.DataFrame:
    rows = []
    for year, frame in daily.groupby(daily.index.year):
        row = {
            "year": int(year),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "days": len(frame),
        }
        for name in series_order:
            nav = frame[f"{name}_nav"].dropna()
            row[f"{name}_return"] = nav.iloc[-1] / nav.iloc[0] - 1.0 if len(nav) >= 2 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sample_note"] = ""
        out.loc[out.index.min(), "sample_note"] = "partial first year"
        out.loc[out.index.max(), "sample_note"] = "partial final year"
    return out


def pct_formatter(x: float, _pos: int) -> str:
    return f"{x:.0%}"


def nav_formatter(x: float, _pos: int) -> str:
    return f"{x:.1f}x"


def plot_outputs(daily: pd.DataFrame, metrics: pd.DataFrame, annual: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c5cbd3",
            "axes.labelcolor": "#2f343b",
            "axes.titleweight": "bold",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    neutral = "#626b76"
    series_order = [*STRATEGIES.keys(), *ETF_SYMBOLS]

    fig = plt.figure(figsize=(18, 17), constrained_layout=False)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.25, 0.9, 1.05, 0.95], hspace=0.48, wspace=0.18)
    ax_nav = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_nav)
    ax_annual = fig.add_subplot(gs[2, :])
    ax_pct_metrics = fig.add_subplot(gs[3, 0])
    ax_sharpe = fig.add_subplot(gs[3, 1])

    for name in series_order:
        ax_nav.plot(daily.index, daily[f"{name}_nav"], color=COLORS[name], lw=2.0, label=name)
    ax_nav.set_title("Strategies vs DBMF and KMLM: Growth of $1")
    ax_nav.set_ylabel("Growth of $1")
    ax_nav.yaxis.set_major_formatter(nav_formatter)
    ax_nav.grid(True, color="#e7ebef", lw=0.8)
    ax_nav.legend(loc="upper left", frameon=False, ncol=4)
    ax_nav.text(
        0.99,
        0.04,
        "Comparison starts at the first common DBMF/KMLM adjusted-close date.",
        transform=ax_nav.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=neutral,
    )

    for name in series_order:
        ax_dd.plot(daily.index, daily[f"{name}_drawdown"], color=COLORS[name], lw=1.6, label=name)
    ax_dd.axhline(0, color="#9aa3ad", lw=0.8)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_dd.grid(True, color="#e7ebef", lw=0.8)
    ax_dd.legend(loc="lower left", frameon=False, ncol=4)

    x_annual = np.arange(len(annual))
    annual_width = 0.19
    annual_offsets = {
        "17 selected": -1.5 * annual_width,
        "40 no-equity": -0.5 * annual_width,
        "DBMF": 0.5 * annual_width,
        "KMLM": 1.5 * annual_width,
    }
    for name in series_order:
        ax_annual.bar(
            x_annual + annual_offsets[name],
            annual[f"{name}_return"],
            width=annual_width,
            color=COLORS[name],
            label=name,
        )
    ax_annual.axhline(0, color="#6b7280", lw=1.0)
    ax_annual.set_xticks(x_annual)
    ax_annual.set_xticklabels(annual["year"].astype(str), rotation=0)
    ax_annual.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_annual.set_title("Calendar-Year Returns")
    ax_annual.set_ylabel("Year return")
    ax_annual.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_annual.legend(loc="upper left", frameon=False, ncol=4)
    ax_annual.text(
        0.99,
        0.04,
        "First and final years are partial sample years.",
        transform=ax_annual.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=neutral,
    )

    metric_frame = metrics.set_index("series").loc[series_order]
    percent_metrics = ["cagr", "annual_vol", "max_drawdown"]
    percent_labels = ["CAGR", "Vol", "MDD"]
    x_metrics = np.arange(len(percent_metrics))
    metric_width = 0.18
    for idx, name in enumerate(series_order):
        values = [metric_frame.loc[name, metric] for metric in percent_metrics]
        ax_pct_metrics.bar(
            x_metrics + (idx - 1.5) * metric_width,
            values,
            width=metric_width,
            color=COLORS[name],
            label=name,
        )
    ax_pct_metrics.axhline(0, color="#6b7280", lw=1.0)
    ax_pct_metrics.set_xticks(x_metrics)
    ax_pct_metrics.set_xticklabels(percent_labels)
    ax_pct_metrics.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.0%}"))
    ax_pct_metrics.set_title("Return, Volatility, And Max Drawdown")
    ax_pct_metrics.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_pct_metrics.legend(loc="lower left", frameon=False, ncol=2)

    sharpe_values = metric_frame["sharpe_0rf"]
    ax_sharpe.bar(sharpe_values.index, sharpe_values.values, color=[COLORS[name] for name in sharpe_values.index])
    ax_sharpe.axhline(0, color="#6b7280", lw=1.0)
    ax_sharpe.set_title("Sharpe Ratio")
    ax_sharpe.set_ylabel("Sharpe, zero risk-free")
    ax_sharpe.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_sharpe.tick_params(axis="x", labelrotation=0)

    for ax in [ax_nav, ax_dd]:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    start = daily.index.min().date()
    end = daily.index.max().date()
    fig.suptitle(
        "17 Selected and 40 No-Equity Strategies vs DBMF and KMLM",
        fontsize=17,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.02,
        0.965,
        f"Daily comparison on common DBMF/KMLM trading dates, {start} to {end}. Strategy curves use buffered-integer NAV; ETFs use adjusted close.",
        fontsize=10,
        color=neutral,
        ha="left",
    )
    fig.subplots_adjust(top=0.93, left=0.08, right=0.98, bottom=0.07)
    fig.savefig(OUT / "strategies_vs_dbmf_kmlm_big.png", dpi=180)
    fig.savefig(OUT / "strategies_vs_dbmf_kmlm_big.pdf")
    plt.close(fig)


def format_metric(key: str, value: float | str) -> str:
    if key in {"start", "end"}:
        return str(value)
    if key == "years":
        return f"{value:.1f}"
    if key in {"sharpe_0rf", "calmar"}:
        return f"{value:.2f}"
    return f"{value:.2%}"


def write_summary(
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    annual: pd.DataFrame,
    source_urls: dict[str, str],
) -> None:
    display_keys = [
        "start",
        "end",
        "years",
        "total_return",
        "cagr",
        "annual_return_arithmetic",
        "annual_vol",
        "sharpe_0rf",
        "max_drawdown",
        "calmar",
    ]
    wins = {}
    for strategy in STRATEGIES:
        for symbol in ETF_SYMBOLS:
            wins[(strategy, symbol)] = (annual[f"{strategy}_return"] > annual[f"{symbol}_return"]).sum()
    wins_40_vs_17 = (annual["40 no-equity_return"] > annual["17 selected_return"]).sum()

    lines = [
        "# Strategies vs DBMF and KMLM",
        "",
        "## Data",
        "",
        f"- 17 selected strategy: `{STRATEGIES['17 selected']}`",
        f"- 40 no-equity strategy: `{STRATEGIES['40 no-equity']}`",
        "- DBMF source: Yahoo Finance adjusted close, `https://finance.yahoo.com/quote/DBMF/history/`",
        "- KMLM source: Yahoo Finance adjusted close, `https://finance.yahoo.com/quote/KMLM/history/`",
        f"- DBMF chart endpoint used: `{source_urls['DBMF']}`",
        f"- KMLM chart endpoint used: `{source_urls['KMLM']}`",
        f"- Comparison starts on `{daily.index.min().date()}`, the first common DBMF/KMLM adjusted-close date after the strategies begin.",
        f"- Comparison ends on `{daily.index.max().date()}`, the common end date.",
        "",
        "## Metrics",
        "",
        "| Series | " + " | ".join(display_keys) + " |",
        "|---" * (len(display_keys) + 1) + "|",
    ]
    for row in metrics.to_dict("records"):
        values = [row["series"]] + [format_metric(key, row[key]) for key in display_keys]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Calendar-Year Read", ""])
    for strategy in STRATEGIES:
        for symbol in ETF_SYMBOLS:
            lines.append(
                f"- {strategy} beat {symbol} in {wins[(strategy, symbol)]:.0f} / {len(annual):.0f} calendar-year rows."
            )
    lines.extend(
        [
            f"- 40 no-equity beat 17 selected in {wins_40_vs_17:.0f} / {len(annual):.0f} calendar-year rows.",
            "- First and final calendar-year rows are partial sample years.",
            "",
            "## Outputs",
            "",
            "- Big chart: `strategies_vs_dbmf_kmlm_big.png`",
            "- Daily data: `daily_comparison.csv`",
            "- Metrics: `metrics.csv`",
            "- Calendar-year returns: `annual_return_metrics.csv`",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily, metrics, annual, source_urls = build_comparison()
    daily.to_csv(OUT / "daily_comparison.csv")
    metrics.to_csv(OUT / "metrics.csv", index=False)
    annual.to_csv(OUT / "annual_return_metrics.csv", index=False)
    plot_outputs(daily, metrics, annual)
    write_summary(daily, metrics, annual, source_urls)

    print(f"Wrote comparison to {OUT}")
    print(metrics.to_string(index=False))
    print()
    print(annual.to_string(index=False))


if __name__ == "__main__":
    main()
