#!/usr/bin/env python3
"""Run a stock-universe trend-following test on current S&P 500 constituents.

This is a research analogue of the Rob Carver / CTA-style futures tests in this
workspace. Stocks do not have futures carry, contract rolls, point values, or
clean short/borrow information in yfinance, so this script uses the transferable
parts only:

- EWMAC and breakout trend forecasts.
- Per-stock volatility normalisation.
- Dynamic active-universe normalisation as histories appear.
- Weekly buffered-ish rebalancing through target-weight drift.

It intentionally stores the constituent list used for the run because using
today's S&P 500 members back to 2000 has survivorship/look-ahead bias.
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


warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sp500_yfinance"
OUT = ROOT / "backtests" / "sp500_stock_trend"

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
START = "2000-01-01"
BUSINESS_DAYS = 252.0
TARGET_VOL = 0.25
FORECAST_CAP = 20.0
AVERAGE_ABS_FORECAST = 10.0
COST_PER_DOLLAR_TRADED = 0.0005
GROSS_LEVERAGE_CAP = 2.0
PER_NAME_ABS_CAP = 0.015
MIN_HISTORY_DAYS = 260
MIN_ACTIVE_NAMES = 100
REBALANCE_FREQ = "W-FRI"

SELF_17 = ROOT / "backtests" / "rob_style_us_rates_selected_no_vol" / "portfolio_daily.csv"
SELF_40 = ROOT / "backtests" / "rob_style_no_equity_40" / "portfolio_daily.csv"


@dataclass(frozen=True)
class RunConfig:
    refresh: bool
    chunk_size: int
    target_vol: float
    cost_per_dollar: float
    gross_cap: float
    per_name_cap: float


def clean_yahoo_ticker(symbol: str) -> str:
    return symbol.replace(".", "-").strip()


def load_sp500_constituents(refresh: bool) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "sp500_constituents_current.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    request = urllib.request.Request(
        WIKI_SP500_URL,
        headers={"User-Agent": "Mozilla/5.0 qoppac-trading-system research script"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    constituents = tables[0].copy()
    constituents["yahoo_symbol"] = constituents["Symbol"].map(clean_yahoo_ticker)
    constituents["source_url"] = WIKI_SP500_URL
    constituents["fetched_at_utc"] = pd.Timestamp.utcnow().isoformat()
    constituents.to_csv(cache, index=False)
    return constituents


def close_from_yfinance_download(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
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
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes)


def download_adjusted_close(tickers: list[str], config: RunConfig) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "adj_close.csv"
    if cache.exists() and not config.refresh:
        prices = pd.read_csv(cache, parse_dates=["Date"]).set_index("Date").sort_index()
        return prices

    all_closes: list[pd.DataFrame] = []
    failed_chunks: list[str] = []
    for start in range(0, len(tickers), config.chunk_size):
        chunk = tickers[start : start + config.chunk_size]
        print(f"Downloading {start + 1}-{start + len(chunk)} / {len(tickers)}")
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
            close = close_from_yfinance_download(data, chunk)
            if close.empty:
                failed_chunks.append(",".join(chunk))
            else:
                all_closes.append(close)
        except Exception as exc:  # pragma: no cover - network edge
            failed_chunks.append(f"{','.join(chunk)} :: {exc}")
        time.sleep(0.5)

    if not all_closes:
        raise RuntimeError("No yfinance prices were downloaded.")

    prices = pd.concat(all_closes, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(how="all")
    prices.index.name = "Date"
    prices.to_csv(cache)

    if failed_chunks:
        (DATA_DIR / "failed_download_chunks.txt").write_text("\n".join(failed_chunks), encoding="utf-8")

    return prices


def download_spy(config: RunConfig) -> pd.Series:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "spy_adj_close.csv"
    if cache.exists() and not config.refresh:
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date")["SPY"].sort_index()

    data = yf.download("SPY", start=START, auto_adjust=True, progress=False, timeout=30)
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", "SPY")]
    else:
        close = data["Close"]
    spy = close.rename("SPY").dropna()
    spy.index.name = "Date"
    spy.to_csv(cache)
    return spy


def mixed_vol(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    fast = returns.ewm(adjust=True, span=35, min_periods=20).std()
    slow = fast.ewm(span=int(20 * BUSINESS_DAYS), min_periods=20).mean()
    vol = 0.65 * fast + 0.35 * slow
    return vol.ffill().clip(lower=1.0e-8)


def ewmac_forecast(price: pd.DataFrame, pct_vol: pd.DataFrame, fast: int) -> pd.DataFrame:
    slow = fast * 4
    fast_ewma = price.ewm(span=fast, min_periods=max(2, fast // 2)).mean()
    slow_ewma = price.ewm(span=slow, min_periods=max(2, slow // 2)).mean()
    return ((fast_ewma - slow_ewma) / price.abs().replace(0.0, np.nan)) / pct_vol


def breakout_forecast(price: pd.DataFrame, lookback: int) -> pd.DataFrame:
    smooth = max(int(lookback / 4.0), 1)
    roll_max = price.rolling(lookback, min_periods=max(20, lookback // 2)).max()
    roll_min = price.rolling(lookback, min_periods=max(20, lookback // 2)).min()
    roll_range = (roll_max - roll_min).replace(0.0, np.nan)
    raw = 40.0 * ((price - (roll_max + roll_min) / 2.0) / roll_range)
    return raw.ewm(span=smooth, min_periods=max(2, smooth // 2)).mean()


def robust_rule_scalar(raw: pd.DataFrame) -> float:
    sample = raw.replace([np.inf, -np.inf], np.nan).stack().abs()
    sample = sample[(sample > 0.0) & sample.notna()]
    if sample.empty:
        return 1.0
    median_abs = float(sample.quantile(0.50))
    if not np.isfinite(median_abs) or median_abs <= 0.0:
        return 1.0
    return AVERAGE_ABS_FORECAST / median_abs


def build_forecasts(price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = price.pct_change()
    pct_vol = mixed_vol(returns)
    valid_history = price.notna().rolling(MIN_HISTORY_DAYS, min_periods=MIN_HISTORY_DAYS).sum() >= MIN_HISTORY_DAYS

    rule_forecasts: list[pd.DataFrame] = []
    rule_rows: list[dict[str, float | str]] = []

    for fast in [16, 32, 64]:
        name = f"ewmac{fast}_{fast * 4}"
        raw = ewmac_forecast(price, pct_vol, fast)
        scalar = robust_rule_scalar(raw)
        forecast = (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        rule_forecasts.append(forecast)
        rule_rows.append({"rule": name, "scalar": scalar})

    for lookback in [64, 128, 256]:
        name = f"breakout{lookback}"
        raw = breakout_forecast(price, lookback)
        scalar = robust_rule_scalar(raw)
        forecast = (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        rule_forecasts.append(forecast)
        rule_rows.append({"rule": name, "scalar": scalar})

    stacked = pd.concat(rule_forecasts, axis=1, keys=range(len(rule_forecasts)))
    combined = stacked.T.groupby(level=1).mean().T
    combined = combined.where(valid_history).clip(-FORECAST_CAP, FORECAST_CAP)
    rule_table = pd.DataFrame(rule_rows)
    return combined, pct_vol.where(valid_history), rule_table


def last_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    period = index.to_series(index=index).dt.to_period(REBALANCE_FREQ)
    return index.to_series(index=index).groupby(period).tail(1).index


def build_weights(
    forecast: pd.DataFrame,
    pct_vol: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ann_vol = pct_vol * math.sqrt(BUSINESS_DAYS)
    score = forecast / AVERAGE_ABS_FORECAST
    valid = score.notna() & ann_vol.notna() & (ann_vol > 0.0)
    active_count = valid.sum(axis=1)

    raw = (score / ann_vol).where(valid)
    diag_ex_ante_vol = ((raw.fillna(0.0) * ann_vol.fillna(0.0)) ** 2).sum(axis=1).pow(0.5)
    scaled = raw.mul(config.target_vol / diag_ex_ante_vol.replace(0.0, np.nan), axis=0)
    scaled = scaled.where(active_count >= MIN_ACTIVE_NAMES)
    scaled = scaled.clip(lower=-config.per_name_cap, upper=config.per_name_cap)

    gross = scaled.abs().sum(axis=1)
    gross_multiplier = (config.gross_cap / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    target = scaled.mul(gross_multiplier, axis=0).fillna(0.0)

    rebalance_dates = last_rebalance_dates(target.index)
    weekly_target = target.loc[rebalance_dates]
    weights = weekly_target.reindex(target.index).ffill().fillna(0.0)

    diagnostics = pd.DataFrame(
        {
            "active_names": active_count,
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
            "long_exposure": weights.clip(lower=0.0).sum(axis=1),
            "short_exposure": weights.clip(upper=0.0).sum(axis=1),
        },
        index=target.index,
    )
    return weights, diagnostics


def run_stock_strategy(price: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price = price.sort_index().ffill(limit=5)
    returns = price.pct_change().fillna(0.0)
    forecast, pct_vol, rule_table = build_forecasts(price)
    weights, diagnostics = build_weights(forecast, pct_vol, config)

    held = weights.shift(1).fillna(0.0)
    gross_return = (held * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * config.cost_per_dollar
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
        },
        index=price.index,
    )
    daily = daily.join(diagnostics)
    return daily, weights, forecast, rule_table


def load_futures_return_stream(path: Path, name: str) -> pd.Series:
    data = pd.read_csv(path, header=[0, 1], skiprows=[2], index_col=0, parse_dates=True)
    ret = pd.to_numeric(data[("buffered_integer", "daily_return")], errors="coerce")
    return ret.rename(name).dropna()


def performance_stats(returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "start": "",
            "end": "",
            "years": np.nan,
            "total_return": np.nan,
            "cagr": np.nan,
            "ann_return": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
        }

    equity = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    total_return = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    ann_return = returns.mean() * BUSINESS_DAYS
    vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    sharpe = ann_return / vol if vol and np.isfinite(vol) else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "ann_return": ann_return,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": drawdown.min(),
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
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1]).rename("rolling_corr")


def crisis_table(streams: dict[str, pd.Series]) -> pd.DataFrame:
    windows = {
        "dotcom_2000_2002": ("2000-03-24", "2002-10-09"),
        "gfc_2007_2009": ("2007-10-09", "2009-03-09"),
        "covid_2020": ("2020-02-19", "2020-03-23"),
        "inflation_2022": ("2022-01-03", "2022-10-12"),
    }
    rows = []
    stock = streams.get("SP500 stocks trend")
    spy = streams.get("SPY")
    for label, (start, end) in windows.items():
        for name, series in streams.items():
            period = series.loc[start:end].dropna()
            if period.empty:
                continue
            row = {
                "window": label,
                "strategy": name,
                "return": (1.0 + period).prod() - 1.0,
                "vol": period.std() * math.sqrt(BUSINESS_DAYS),
                "mdd": ((1.0 + period).cumprod() / (1.0 + period).cumprod().cummax() - 1.0).min(),
            }
            if name == "SP500 stocks trend" and stock is not None and spy is not None:
                row["corr_to_spy"] = stock.loc[start:end].corr(spy.loc[start:end])
            rows.append(row)
    return pd.DataFrame(rows)


def save_summary(
    full_stats: pd.DataFrame,
    common_stats: pd.DataFrame,
    crisis: pd.DataFrame,
    rule_table: pd.DataFrame,
    tickers_used: list[str],
    latest_price_date: pd.Timestamp,
) -> None:
    def pct(value: float) -> str:
        return "" if pd.isna(value) else f"{value:.2%}"

    def num(value: float) -> str:
        return "" if pd.isna(value) else f"{value:.2f}"

    lines = [
        "# S&P 500 Stock Trend Backtest",
        "",
        f"- Universe source: current S&P 500 table from {WIKI_SP500_URL}",
        f"- Yahoo/yfinance latest price date: {latest_price_date.date()}",
        f"- Tickers downloaded/usable: {len(tickers_used)}",
        f"- Rules: EWMAC 16/64, 32/128, 64/256; breakout 64, 128, 256; no carry.",
        f"- Sizing: {TARGET_VOL:.0%} annual target vol, weekly rebalance, gross cap {GROSS_LEVERAGE_CAP:.1f}x, per-name cap {PER_NAME_ABS_CAP:.1%}, trading cost {COST_PER_DOLLAR_TRADED:.2%} of notional traded.",
        "- Important caveat: this uses today's S&P 500 members back through history, so it has survivorship and membership look-ahead bias.",
        "",
        "## Full stock/yfinance period",
        "",
        "| Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in full_stats.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )

    lines += [
        "",
        "## Common period with local futures systems",
        "",
        "| Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in common_stats.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )

    lines += [
        "",
        "## Crisis windows",
        "",
        "| Window | Strategy | Return | Vol | MDD | Corr To SPY |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in crisis.iterrows():
        corr = "" if "corr_to_spy" not in row or pd.isna(row["corr_to_spy"]) else f"{row['corr_to_spy']:.2f}"
        lines.append(
            f"| {row['window']} | {row['strategy']} | {pct(row['return'])} | {pct(row['vol'])} | {pct(row['mdd'])} | {corr} |"
        )

    lines += ["", "## Forecast scalars", "", rule_table.to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(streams: dict[str, pd.Series], yearly: pd.DataFrame, rolling: pd.Series) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "SP500 stocks trend": "#1f77b4",
        "Self 17 futures": "#2ca02c",
        "Self 40 futures": "#ff7f0e",
        "SPY": "#4c4c4c",
    }

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(4, 1, height_ratios=[3.0, 1.15, 1.15, 1.5], hspace=0.18)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(grid[3, 0])

    aligned = pd.DataFrame(streams).dropna(how="all")
    for name in streams:
        series = aligned[name].dropna()
        if series.empty:
            continue
        equity = (1.0 + series).cumprod()
        ax0.plot(equity.index, equity, label=name, color=colors.get(name), linewidth=1.8)
        dd = equity / equity.cummax() - 1.0
        ax1.plot(dd.index, dd, label=name, color=colors.get(name), linewidth=1.1)

    ax2.plot(rolling.index, rolling, color="#1f77b4", linewidth=1.3)
    ax2.axhline(0.0, color="#777777", linewidth=0.8)
    ax2.set_ylabel("126d corr")
    ax2.set_title("SP500 stock trend rolling correlation to SPY")

    annual = yearly.copy()
    annual.index = annual.index.year
    annual = annual[["SP500 stocks trend", "SPY", "Self 17 futures", "Self 40 futures"]].dropna(how="all")
    x = np.arange(len(annual.index))
    width = 0.20
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, name in zip(offsets, annual.columns):
        ax3.bar(x + offset, annual[name], width=width, label=name, color=colors.get(name), alpha=0.90)
    ax3.axhline(0.0, color="#555555", linewidth=0.8)
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([str(year) for year in annual.index[::2]], rotation=45, ha="right")
    ax3.set_ylabel("Year return")
    ax3.set_title("Calendar-year returns")

    ax0.set_title("S&P 500 Stock Trend vs SPY and Existing Futures Systems")
    ax0.set_ylabel("Growth of $1")
    ax0.set_yscale("log")
    ax0.legend(loc="upper left", ncol=4)
    ax1.set_ylabel("Drawdown")
    ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax3.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(OUT / "sp500_stock_trend_vs_spy_self17_self40.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh Wikipedia and yfinance caches.")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--target-vol", type=float, default=TARGET_VOL)
    parser.add_argument("--cost-bps", type=float, default=COST_PER_DOLLAR_TRADED * 10000.0)
    parser.add_argument("--gross-cap", type=float, default=GROSS_LEVERAGE_CAP)
    parser.add_argument("--per-name-cap", type=float, default=PER_NAME_ABS_CAP)
    args = parser.parse_args()

    config = RunConfig(
        refresh=args.refresh,
        chunk_size=args.chunk_size,
        target_vol=args.target_vol,
        cost_per_dollar=args.cost_bps / 10000.0,
        gross_cap=args.gross_cap,
        per_name_cap=args.per_name_cap,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    constituents = load_sp500_constituents(config.refresh)
    tickers = constituents["yahoo_symbol"].dropna().drop_duplicates().tolist()
    price = download_adjusted_close(tickers, config)
    spy_price = download_spy(config)

    usable_tickers = [column for column in price.columns if price[column].notna().sum() >= MIN_HISTORY_DAYS]
    price = price[usable_tickers]
    latest_price_date = price.dropna(how="all").index.max()

    stock_daily, weights, forecast, rule_table = run_stock_strategy(price, config)
    spy_returns = spy_price.pct_change().rename("SPY").dropna()
    stock_returns = stock_daily["net_return"].rename("SP500 stocks trend")
    self17 = load_futures_return_stream(SELF_17, "Self 17 futures")
    self40 = load_futures_return_stream(SELF_40, "Self 40 futures")

    full_streams = {
        "SP500 stocks trend": stock_returns,
        "SPY": spy_returns,
    }
    full_stats = make_stats_table(full_streams)

    common_start = max(stock_returns.dropna().index.min(), spy_returns.dropna().index.min(), self17.index.min(), self40.index.min())
    common_end = min(stock_returns.dropna().index.max(), spy_returns.dropna().index.max(), self17.index.max(), self40.index.max())
    common_streams = {
        "SP500 stocks trend": stock_returns.loc[common_start:common_end],
        "SPY": spy_returns.loc[common_start:common_end],
        "Self 17 futures": self17.loc[common_start:common_end],
        "Self 40 futures": self40.loc[common_start:common_end],
    }
    common_stats = make_stats_table(common_streams)
    annual = yearly_returns(common_streams)
    corr = rolling_corr(common_streams["SP500 stocks trend"], common_streams["SPY"])
    crisis = crisis_table(common_streams)

    stock_daily.to_csv(OUT / "portfolio_daily.csv")
    weights.iloc[::5].to_csv(OUT / "weekly_weights_snapshot.csv")
    forecast.iloc[::5].to_csv(OUT / "weekly_forecast_snapshot.csv")
    rule_table.to_csv(OUT / "rule_scalars.csv", index=False)
    constituents.to_csv(OUT / "constituents_used.csv", index=False)
    full_stats.to_csv(OUT / "stats_full_stock_spy.csv", index=False)
    common_stats.to_csv(OUT / "stats_common_with_futures.csv", index=False)
    annual.to_csv(OUT / "yearly_returns_common.csv")
    crisis.to_csv(OUT / "crisis_windows.csv", index=False)
    corr.to_csv(OUT / "rolling_corr_stocktrend_spy.csv")

    save_summary(full_stats, common_stats, crisis, rule_table, usable_tickers, latest_price_date)
    plot_comparison(common_streams, annual, corr)

    print("\nFull stock/SPY period")
    print(full_stats.to_string(index=False))
    print("\nCommon period")
    print(common_stats.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
