#!/usr/bin/env python3
"""Run S&P 500 ranked long-only stock trend portfolios.

Requested variant of the ranked stock test:

- Stock-only forecasts; no futures carry or roll signal.
- Rank current S&P 500 stocks by combined EWMAC + breakout forecast.
- Hold only the highest ranked names.
- Test top-20 and top-40 equal-weight long-only portfolios.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_sp500_stock_trend_backtest as stock_base
import run_sp500_ranked_long_short_backtest as ranked_ls


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "sp500_ranked_long_only"

COST_PER_DOLLAR_TRADED = 0.0005
MIN_HISTORY_DAYS = 260
MIN_ACTIVE_NAMES = 100
BUSINESS_DAYS = 252.0


def build_long_only_weights(forecast: pd.DataFrame, total_names: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rebalance_dates = stock_base.last_rebalance_dates(forecast.index)
    target_on_rebalance = pd.DataFrame(0.0, index=rebalance_dates, columns=forecast.columns)
    rows: list[dict[str, float | int | str]] = []

    for date in rebalance_dates:
        scores = forecast.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        if len(scores) < max(total_names, MIN_ACTIVE_NAMES):
            continue
        long_names = scores.nlargest(total_names).index
        weight = 1.0 / total_names
        target_on_rebalance.loc[date, long_names] = weight
        for rank, ticker in enumerate(long_names, start=1):
            rows.append(
                {
                    "date": date,
                    "portfolio_size": total_names,
                    "rank": rank,
                    "ticker": ticker,
                    "forecast": float(scores[ticker]),
                    "weight": weight,
                }
            )

    weights = target_on_rebalance.reindex(forecast.index).ffill().fillna(0.0)
    return weights, pd.DataFrame(rows)


def run_portfolio(price: pd.DataFrame, forecast: pd.DataFrame, total_names: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights, selections = build_long_only_weights(forecast, total_names)
    returns = price.pct_change().fillna(0.0)
    held = weights.shift(1).fillna(0.0)

    gross_return = (held * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * COST_PER_DOLLAR_TRADED
    net_return = gross_return - costs
    equity = (1.0 + net_return.fillna(0.0)).cumprod()

    daily = pd.DataFrame(
        {
            "gross_return": gross_return,
            "costs": costs,
            "net_return": net_return,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
            "turnover": turnover,
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
            "long_count": (weights > 0.0).sum(axis=1),
        },
        index=price.index,
    )
    return daily, weights, selections


def trim_to_first_position(returns: pd.Series, weights: pd.DataFrame) -> pd.Series:
    gross = weights.abs().sum(axis=1)
    active = gross[gross > 0.0]
    if active.empty:
        return returns.dropna()
    return returns.loc[active.index[0] :].dropna()


def performance_stats(returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    ann_return = returns.mean() * BUSINESS_DAYS
    vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": years,
        "total_return": equity.iloc[-1] - 1.0,
        "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
        "ann_return": ann_return,
        "vol": vol,
        "sharpe": ann_return / vol if vol else np.nan,
        "mdd": (equity / equity.cummax() - 1.0).min(),
    }


def make_stats_table(streams: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in streams.items():
        row = {"strategy": name}
        row.update(performance_stats(returns))
        rows.append(row)
    return pd.DataFrame(rows)


def yearly_returns(streams: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(streams).dropna(how="all").groupby(pd.Grouper(freq="YE")).apply(
        lambda frame: (1.0 + frame).prod() - 1.0
    )


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 126) -> pd.Series:
    aligned = pd.concat([a, b], axis=1).dropna()
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def crisis_table(streams: dict[str, pd.Series]) -> pd.DataFrame:
    windows = {
        "dotcom_2000_2002": ("2000-03-24", "2002-10-09"),
        "gfc_2007_2009": ("2007-10-09", "2009-03-09"),
        "covid_2020": ("2020-02-19", "2020-03-23"),
        "inflation_2022": ("2022-01-03", "2022-10-12"),
    }
    rows = []
    spy = streams.get("SPY")
    for label, (start, end) in windows.items():
        for name, series in streams.items():
            period = series.loc[start:end].dropna()
            if period.empty:
                continue
            equity = (1.0 + period).cumprod()
            row = {
                "window": label,
                "strategy": name,
                "return": equity.iloc[-1] - 1.0,
                "vol": period.std() * math.sqrt(BUSINESS_DAYS),
                "mdd": (equity / equity.cummax() - 1.0).min(),
            }
            if name != "SPY" and spy is not None:
                row["corr_to_spy"] = series.loc[start:end].corr(spy.loc[start:end])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_results(streams: dict[str, pd.Series], annual: pd.DataFrame, corr20: pd.Series, corr40: pd.Series) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"Top 20 long-only": "#1f77b4", "Top 40 long-only": "#ff7f0e", "SPY": "#4c4c4c"}

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(4, 1, height_ratios=[3.0, 1.1, 1.1, 1.45], hspace=0.18)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(grid[3, 0])

    aligned = pd.DataFrame(streams).dropna(how="all")
    for name in streams:
        series = aligned[name].dropna()
        equity = (1.0 + series).cumprod()
        ax0.plot(equity.index, equity, label=name, color=colors.get(name), linewidth=1.8)
        drawdown = equity / equity.cummax() - 1.0
        ax1.plot(drawdown.index, drawdown, label=name, color=colors.get(name), linewidth=1.1)

    ax2.plot(corr20.index, corr20, color=colors["Top 20 long-only"], label="Top 20 to SPY", linewidth=1.2)
    ax2.plot(corr40.index, corr40, color=colors["Top 40 long-only"], label="Top 40 to SPY", linewidth=1.2)
    ax2.axhline(0.0, color="#777777", linewidth=0.8)
    ax2.legend(loc="upper left", ncol=2)

    annual = annual[["Top 20 long-only", "Top 40 long-only", "SPY"]].dropna(how="all")
    annual.index = annual.index.year
    x = np.arange(len(annual.index))
    width = 0.25
    for offset, name in zip([-width, 0, width], annual.columns):
        ax3.bar(x + offset, annual[name], width=width, label=name, color=colors.get(name), alpha=0.9)
    ax3.axhline(0.0, color="#555555", linewidth=0.8)
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([str(year) for year in annual.index[::2]], rotation=45, ha="right")

    ax0.set_title("S&P 500 Ranked Long-Only Trend vs SPY")
    ax0.set_ylabel("Growth of $1")
    ax0.set_yscale("log")
    ax0.legend(loc="upper left", ncol=3)
    ax1.set_ylabel("Drawdown")
    ax2.set_ylabel("126d corr")
    ax3.set_ylabel("Year return")
    ax1.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax3.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(OUT / "ranked_long_only_top20_top40_vs_spy.png", dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def write_summary(stats: pd.DataFrame, crisis: pd.DataFrame, usable_count: int, latest_price_date: pd.Timestamp) -> None:
    lines = [
        "# S&P 500 Ranked Long-Only Trend Backtest",
        "",
        f"- Data: current S&P 500 constituents from Wikipedia; adjusted close prices from yfinance through {latest_price_date.date()}.",
        f"- Usable stocks: {usable_count}.",
        "- Signal: EWMAC 16/64, 32/128, 64/256 plus breakout 64, 128, 256. No carry and no roll.",
        "- Portfolio: weekly rebalance; top 20 or top 40 forecast names; equal-weight; 100% long-only.",
        f"- Trading cost assumption: {COST_PER_DOLLAR_TRADED:.2%} of notional traded.",
        "- Caveat: today's S&P 500 members are used historically, so this has survivorship and membership look-ahead bias.",
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
        "## Crisis Windows",
        "",
        "| Window | Strategy | Return | Vol | MDD | Corr To SPY |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in crisis.iterrows():
        corr = "" if "corr_to_spy" not in row or pd.isna(row["corr_to_spy"]) else f"{row['corr_to_spy']:.2f}"
        lines.append(
            f"| {row['window']} | {row['strategy']} | {pct(row['return'])} | {pct(row['vol'])} | {pct(row['mdd'])} | {corr} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh Wikipedia/yfinance cache.")
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    config = stock_base.RunConfig(
        refresh=args.refresh,
        chunk_size=args.chunk_size,
        target_vol=stock_base.TARGET_VOL,
        cost_per_dollar=COST_PER_DOLLAR_TRADED,
        gross_cap=stock_base.GROSS_LEVERAGE_CAP,
        per_name_cap=stock_base.PER_NAME_ABS_CAP,
    )

    constituents = stock_base.load_sp500_constituents(args.refresh)
    tickers = constituents["yahoo_symbol"].dropna().drop_duplicates().tolist()
    price = stock_base.download_adjusted_close(tickers, config)
    spy_price = stock_base.download_spy(config)
    usable = [column for column in price.columns if price[column].notna().sum() >= MIN_HISTORY_DAYS]
    price = price[usable].sort_index().ffill(limit=5)
    latest_price_date = price.dropna(how="all").index.max()
    forecast, _pct_vol, rule_table = stock_base.build_forecasts(price)

    daily20, weights20, selections20 = run_portfolio(price, forecast, 20)
    daily40, weights40, selections40 = run_portfolio(price, forecast, 40)
    ret20 = trim_to_first_position(daily20["net_return"].rename("Top 20 long-only"), weights20)
    ret40 = trim_to_first_position(daily40["net_return"].rename("Top 40 long-only"), weights40)
    spy_ret = spy_price.pct_change().rename("SPY").dropna()
    common_start = max(ret20.index.min(), ret40.index.min(), spy_ret.index.min())
    common_end = min(ret20.index.max(), ret40.index.max(), spy_ret.index.max())
    streams = {
        "Top 20 long-only": ret20.loc[common_start:common_end],
        "Top 40 long-only": ret40.loc[common_start:common_end],
        "SPY": spy_ret.loc[common_start:common_end],
    }

    stats = make_stats_table(streams)
    annual = yearly_returns(streams)
    crisis = crisis_table(streams)
    corr20 = rolling_corr(streams["Top 20 long-only"], streams["SPY"])
    corr40 = rolling_corr(streams["Top 40 long-only"], streams["SPY"])

    daily20.to_csv(OUT / "portfolio_daily_top20.csv")
    daily40.to_csv(OUT / "portfolio_daily_top40.csv")
    weights20.iloc[::5].to_csv(OUT / "weekly_weights_top20.csv")
    weights40.iloc[::5].to_csv(OUT / "weekly_weights_top40.csv")
    pd.concat([selections20, selections40], ignore_index=True).to_csv(OUT / "rebalance_selections.csv", index=False)
    constituents.to_csv(OUT / "constituents_used.csv", index=False)
    rule_table.to_csv(OUT / "rule_scalars.csv", index=False)
    stats.to_csv(OUT / "stats.csv", index=False)
    annual.to_csv(OUT / "yearly_returns.csv")
    crisis.to_csv(OUT / "crisis_windows.csv", index=False)
    pd.DataFrame({"Top 20 to SPY": corr20, "Top 40 to SPY": corr40}).to_csv(OUT / "rolling_corr_to_spy.csv")

    plot_results(streams, annual, corr20, corr40)
    write_summary(stats, crisis, len(usable), latest_price_date)

    print(stats.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
