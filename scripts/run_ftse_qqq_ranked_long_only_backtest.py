#!/usr/bin/env python3
"""Run ranked long-only trend tests on FTSE 100 and Nasdaq-100 stocks.

This is the same stock-only method used in the S&P 500 long-only test:

- Current index constituents from Wikipedia.
- Adjusted close prices from yfinance.
- Combined EWMAC + breakout forecast; no carry and no roll.
- Weekly rebalance into the highest forecast names.
- Top-20 and Top-40 equal-weight long-only portfolios.

Using today's constituents historically creates survivorship and membership
look-ahead bias. FTSE 100 benchmark returns use ^FTSE, which is a price index
and therefore not a total-return benchmark.
"""

from __future__ import annotations

import argparse
import io
import math
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import run_sp500_stock_trend_backtest as stock_base


warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUT = ROOT / "backtests" / "ftse_qqq_ranked_long_only"

START = "2000-01-01"
BUSINESS_DAYS = 252.0
COST_PER_DOLLAR_TRADED = 0.0005
MIN_HISTORY_DAYS = 260
TOP_COUNTS = [20, 40]

FTSE_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


@dataclass(frozen=True)
class UniverseSpec:
    key: str
    title: str
    source_url: str
    benchmark_ticker: str
    benchmark_label: str
    min_active_floor: int


UNIVERSES = [
    UniverseSpec(
        key="ftse100",
        title="FTSE 100",
        source_url=FTSE_URL,
        benchmark_ticker="^FTSE",
        benchmark_label="FTSE 100 (^FTSE)",
        min_active_floor=50,
    ),
    UniverseSpec(
        key="nasdaq100",
        title="Nasdaq-100 / QQQ",
        source_url=NASDAQ100_URL,
        benchmark_ticker="QQQ",
        benchmark_label="QQQ",
        min_active_floor=50,
    ),
]


def wiki_tables(url: str) -> list[pd.DataFrame]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 qoppac research script"})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def clean_yahoo_symbol(symbol: str, suffix: str = "") -> str:
    cleaned = str(symbol).strip().replace(".", "-")
    return f"{cleaned}{suffix}"


def load_constituents(spec: UniverseSpec, refresh: bool) -> pd.DataFrame:
    data_dir = DATA_ROOT / f"{spec.key}_yfinance"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "constituents_current.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    tables = wiki_tables(spec.source_url)
    if spec.key == "ftse100":
        table = next(table for table in tables if {"Company", "Ticker"}.issubset(set(table.columns)))
        constituents = table[["Company", "Ticker"]].copy()
        constituents["yahoo_symbol"] = constituents["Ticker"].map(lambda symbol: clean_yahoo_symbol(symbol, ".L"))
    elif spec.key == "nasdaq100":
        table = next(table for table in tables if {"Company", "Ticker"}.issubset(set(table.columns)))
        constituents = table[["Company", "Ticker"]].copy()
        constituents["yahoo_symbol"] = constituents["Ticker"].map(clean_yahoo_symbol)
    else:
        raise ValueError(spec.key)

    constituents["source_url"] = spec.source_url
    constituents["fetched_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    constituents.to_csv(cache, index=False)
    return constituents


def close_from_download(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        if len(tickers) == 1 and "Close" in data.columns:
            return data[["Close"]].rename(columns={"Close": tickers[0]})
        return pd.DataFrame()

    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        if (ticker, "Close") in data.columns:
            closes[ticker] = data[(ticker, "Close")]
        elif ("Close", ticker) in data.columns:
            closes[ticker] = data[("Close", ticker)]
    return pd.DataFrame(closes)


def download_prices(tickers: list[str], data_dir: Path, refresh: bool, chunk_size: int) -> pd.DataFrame:
    cache = data_dir / "adj_close.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date").sort_index()

    chunks: list[pd.DataFrame] = []
    failures: list[str] = []
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        print(f"{data_dir.name}: downloading {start + 1}-{start + len(chunk)} / {len(tickers)}")
        try:
            data = yf.download(
                chunk,
                start=START,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=30,
            )
            close = close_from_download(data, chunk)
            if close.empty:
                failures.append(",".join(chunk))
            else:
                chunks.append(close)
        except Exception as exc:  # pragma: no cover - network edge
            failures.append(f"{','.join(chunk)} :: {exc}")
        time.sleep(0.5)

    if not chunks:
        raise RuntimeError(f"No prices downloaded for {data_dir.name}")
    prices = pd.concat(chunks, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(how="all")
    prices.index.name = "Date"
    prices.to_csv(cache)
    if failures:
        (data_dir / "failed_download_chunks.txt").write_text("\n".join(failures), encoding="utf-8")
    return prices


def download_benchmark(spec: UniverseSpec, data_dir: Path, refresh: bool) -> pd.Series:
    cache = data_dir / "benchmark_adj_close.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date")[spec.benchmark_ticker].sort_index()

    data = yf.download(spec.benchmark_ticker, start=START, auto_adjust=True, progress=False, timeout=30)
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", spec.benchmark_ticker)]
    else:
        close = data["Close"]
    benchmark = close.rename(spec.benchmark_ticker).dropna()
    benchmark.index.name = "Date"
    benchmark.to_csv(cache)
    return benchmark


def build_long_only_weights(
    forecast: pd.DataFrame,
    total_names: int,
    min_active_floor: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_active = max(total_names * 2, min_active_floor)
    rebalance_dates = stock_base.last_rebalance_dates(forecast.index)
    target_on_rebalance = pd.DataFrame(0.0, index=rebalance_dates, columns=forecast.columns)
    rows: list[dict[str, float | int | str]] = []

    for date in rebalance_dates:
        scores = forecast.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
        if len(scores) < min_active:
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
                    "active_names": len(scores),
                    "min_active": min_active,
                }
            )

    weights = target_on_rebalance.reindex(forecast.index).ffill().fillna(0.0)
    return weights, pd.DataFrame(rows)


def run_portfolio(
    price: pd.DataFrame,
    forecast: pd.DataFrame,
    total_names: int,
    min_active_floor: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights, selections = build_long_only_weights(forecast, total_names, min_active_floor)
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
    active = weights.abs().sum(axis=1)
    active = active[active > 0.0]
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
    benchmark_name = next(name for name in streams if name.startswith("Benchmark"))
    benchmark = streams[benchmark_name]
    for window, (start, end) in windows.items():
        for name, series in streams.items():
            period = series.loc[start:end].dropna()
            if period.empty:
                continue
            equity = (1.0 + period).cumprod()
            row = {
                "window": window,
                "strategy": name,
                "return": equity.iloc[-1] - 1.0,
                "vol": period.std() * math.sqrt(BUSINESS_DAYS),
                "mdd": (equity / equity.cummax() - 1.0).min(),
            }
            if name != benchmark_name:
                row["corr_to_benchmark"] = series.loc[start:end].corr(benchmark.loc[start:end])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_results(
    spec: UniverseSpec,
    streams: dict[str, pd.Series],
    annual: pd.DataFrame,
    corr20: pd.Series,
    corr40: pd.Series,
    out_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    top20 = "Top 20 long-only"
    top40 = "Top 40 long-only"
    benchmark = f"Benchmark: {spec.benchmark_label}"
    colors = {top20: "#1f77b4", top40: "#ff7f0e", benchmark: "#4c4c4c"}

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

    ax2.plot(corr20.index, corr20, color=colors[top20], label=f"{top20} corr", linewidth=1.2)
    ax2.plot(corr40.index, corr40, color=colors[top40], label=f"{top40} corr", linewidth=1.2)
    ax2.axhline(0.0, color="#777777", linewidth=0.8)
    ax2.legend(loc="upper left", ncol=2)

    annual = annual[[top20, top40, benchmark]].dropna(how="all")
    annual.index = annual.index.year
    x = np.arange(len(annual.index))
    width = 0.25
    for offset, name in zip([-width, 0, width], annual.columns):
        ax3.bar(x + offset, annual[name], width=width, label=name, color=colors.get(name), alpha=0.9)
    ax3.axhline(0.0, color="#555555", linewidth=0.8)
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([str(year) for year in annual.index[::2]], rotation=45, ha="right")

    ax0.set_title(f"{spec.title} Ranked Long-Only Trend vs Benchmark")
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
    fig.savefig(out_dir / f"{spec.key}_ranked_long_only_vs_benchmark.png", dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def write_universe_summary(
    spec: UniverseSpec,
    stats: pd.DataFrame,
    crisis: pd.DataFrame,
    usable_count: int,
    latest_price_date: pd.Timestamp,
    out_dir: Path,
) -> None:
    lines = [
        f"# {spec.title} Ranked Long-Only Trend Backtest",
        "",
        f"- Constituents source: {spec.source_url}",
        f"- yfinance latest constituent price date: {latest_price_date.date()}",
        f"- Usable stocks: {usable_count}.",
        "- Signal: EWMAC 16/64, 32/128, 64/256 plus breakout 64, 128, 256. No carry and no roll.",
        "- Portfolio: weekly rebalance; top 20 or top 40 forecast names; equal-weight; 100% long-only.",
        f"- Trading cost assumption: {COST_PER_DOLLAR_TRADED:.2%} of notional traded.",
        "- Caveat: today's constituents are used historically, so this has survivorship and membership look-ahead bias.",
    ]
    if spec.key == "ftse100":
        lines.append("- Benchmark caveat: ^FTSE is a price index, not a total-return benchmark.")
    lines += [
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
        "| Window | Strategy | Return | Vol | MDD | Corr To Benchmark |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in crisis.iterrows():
        corr = (
            ""
            if "corr_to_benchmark" not in row or pd.isna(row["corr_to_benchmark"])
            else f"{row['corr_to_benchmark']:.2f}"
        )
        lines.append(
            f"| {row['window']} | {row['strategy']} | {pct(row['return'])} | {pct(row['vol'])} | {pct(row['mdd'])} | {corr} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_universe(spec: UniverseSpec, refresh: bool, chunk_size: int) -> pd.DataFrame:
    data_dir = DATA_ROOT / f"{spec.key}_yfinance"
    out_dir = OUT / spec.key
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    constituents = load_constituents(spec, refresh)
    tickers = constituents["yahoo_symbol"].dropna().drop_duplicates().tolist()
    prices = download_prices(tickers, data_dir, refresh, chunk_size)
    benchmark_price = download_benchmark(spec, data_dir, refresh)

    usable = [column for column in prices.columns if prices[column].notna().sum() >= MIN_HISTORY_DAYS]
    prices = prices[usable].sort_index().ffill(limit=5)
    latest_price_date = prices.dropna(how="all").index.max()
    forecast, _pct_vol, rule_table = stock_base.build_forecasts(prices)

    streams: dict[str, pd.Series] = {}
    selections_all: list[pd.DataFrame] = []
    benchmark_name = f"Benchmark: {spec.benchmark_label}"
    for count in TOP_COUNTS:
        daily, weights, selections = run_portfolio(prices, forecast, count, spec.min_active_floor)
        label = f"Top {count} long-only"
        streams[label] = trim_to_first_position(daily["net_return"].rename(label), weights)
        daily.to_csv(out_dir / f"portfolio_daily_top{count}.csv")
        weights.iloc[::5].to_csv(out_dir / f"weekly_weights_top{count}.csv")
        selections_all.append(selections)

    benchmark_returns = benchmark_price.pct_change().rename(benchmark_name).dropna()
    common_start = max([series.index.min() for series in streams.values()] + [benchmark_returns.index.min()])
    common_end = min([series.index.max() for series in streams.values()] + [benchmark_returns.index.max()])
    streams = {name: series.loc[common_start:common_end] for name, series in streams.items()}
    streams[benchmark_name] = benchmark_returns.loc[common_start:common_end]

    stats = make_stats_table(streams)
    annual = yearly_returns(streams)
    crisis = crisis_table(streams)
    corr20 = rolling_corr(streams["Top 20 long-only"], streams[benchmark_name])
    corr40 = rolling_corr(streams["Top 40 long-only"], streams[benchmark_name])

    constituents.to_csv(out_dir / "constituents_used.csv", index=False)
    pd.concat(selections_all, ignore_index=True).to_csv(out_dir / "rebalance_selections.csv", index=False)
    rule_table.to_csv(out_dir / "rule_scalars.csv", index=False)
    stats.to_csv(out_dir / "stats.csv", index=False)
    annual.to_csv(out_dir / "yearly_returns.csv")
    crisis.to_csv(out_dir / "crisis_windows.csv", index=False)
    pd.DataFrame({"Top 20 to benchmark": corr20, "Top 40 to benchmark": corr40}).to_csv(
        out_dir / "rolling_corr_to_benchmark.csv"
    )
    plot_results(spec, streams, annual, corr20, corr40, out_dir)
    write_universe_summary(spec, stats, crisis, len(usable), latest_price_date, out_dir)

    stats.insert(0, "universe", spec.title)
    print(f"\n{spec.title}")
    print(stats.to_string(index=False))
    return stats


def write_combined_summary(all_stats: pd.DataFrame) -> None:
    lines = [
        "# FTSE 100 And Nasdaq-100 Ranked Long-Only Backtests",
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
        "- FTSE benchmark is ^FTSE price index; constituent portfolios use yfinance adjusted prices.",
        "- All universes use current constituents historically, so results have survivorship and membership look-ahead bias.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh constituent and yfinance caches.")
    parser.add_argument("--chunk-size", type=int, default=40)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = pd.concat([run_universe(spec, args.refresh, args.chunk_size) for spec in UNIVERSES], ignore_index=True)
    all_stats.to_csv(OUT / "stats_all.csv", index=False)
    write_combined_summary(all_stats)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
