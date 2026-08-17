#!/usr/bin/env python3
"""Backtest a custom stock basket with constant Rob forecast versus equal weight and SPY."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rob_style_backtest as rob  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402


TICKERS = ["BKNG", "CRUS", "NOW", "YOU", "EW", "STRL", "VMI", "INTA"]
BENCHMARK = "SPY"
DOWNLOAD_START = "2024-01-01"
BACKTEST_START = "2026-01-01"
END = "2026-08-07"
CAPITAL = 500_000.0
VOL_TARGET = 0.10
IDM = 2.75
CONSTANT_FORECAST = 10.0
OUT = ROOT / "backtests" / "custom_constant_forecast_basket_2026_ytd"


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


def download_prices(refresh: bool) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "adj_close.csv"
    tickers = TICKERS + [BENCHMARK]
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date").sort_index()

    data = yf.download(
        tickers,
        start=DOWNLOAD_START,
        end="2026-08-10",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
        timeout=30,
    )
    close = close_from_download(data, tickers).sort_index()
    close = close.loc[:END]
    close.index.name = "Date"
    close.to_csv(cache)
    return close


def clean_prices(price: pd.DataFrame) -> pd.DataFrame:
    price = price.replace([np.inf, -np.inf], np.nan).sort_index()
    returns = price.pct_change(fill_method=None)
    price = price.mask((price <= 0.0) | (returns.abs() > rob_stock.MAX_ABS_DAILY_RETURN))
    return price.ffill(limit=5)


def basket_annual(index: pd.DatetimeIndex) -> pd.DataFrame:
    snapshot_date = max(pd.Timestamp(DOWNLOAD_START), index.min())
    return pd.DataFrame(
        {
            "snapshot_date": [snapshot_date] * len(TICKERS),
            "symbol": TICKERS,
            "sector": ["Custom Basket"] * len(TICKERS),
            "weight": [1.0 / len(TICKERS)] * len(TICKERS),
        }
    )


def performance_stats(returns: pd.Series, equity: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    equity = equity.reindex(returns.index).dropna()
    if returns.empty or equity.empty:
        return {"start": "", "end": "", "total_return": math.nan, "cagr": math.nan, "vol": math.nan, "sharpe": math.nan, "max_drawdown": math.nan}
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1.0 / rob_stock.BUSINESS_DAYS)
    equity_ratio = equity / equity.iloc[0]
    total_return = float(equity_ratio.iloc[-1] - 1.0)
    cagr = float(equity_ratio.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(returns.std() * math.sqrt(rob_stock.BUSINESS_DAYS))
    drawdown = equity_ratio / equity_ratio.cummax() - 1.0
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": float(returns.mean() * rob_stock.BUSINESS_DAYS / vol) if vol else math.nan,
        "max_drawdown": float(drawdown.min()),
    }


def equal_weight_daily(price: pd.DataFrame) -> pd.DataFrame:
    returns = price.pct_change(fill_method=None).fillna(0.0)
    target = pd.DataFrame(1.0 / len(price.columns), index=price.index, columns=price.columns)
    target = target.where(price.notna(), 0.0)
    target = target.div(target.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    gross_values: list[float] = []
    turnover_values: list[float] = []
    previous_target = pd.Series(0.0, index=price.columns)
    for date in price.index:
        day_return = returns.loc[date]
        gross_return = float((previous_target * day_return).sum())
        denominator = 1.0 + gross_return
        if denominator <= 0.0:
            drifted = pd.Series(0.0, index=price.columns)
        else:
            drifted = previous_target * (1.0 + day_return) / denominator
        current_target = target.loc[date]
        turnover = float((current_target - drifted).abs().sum())
        gross_values.append(gross_return)
        turnover_values.append(turnover)
        previous_target = current_target

    gross_return = pd.Series(gross_values, index=price.index)
    turnover = pd.Series(turnover_values, index=price.index)
    net_return = gross_return - turnover * rob_stock.DEFAULT_COST_PER_DOLLAR
    equity = CAPITAL * (1.0 + net_return).cumprod()
    return pd.DataFrame(
        {
            "daily_return": net_return,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
            "turnover": turnover,
            "gross_exposure": target.abs().sum(axis=1),
            "net_exposure": target.sum(axis=1),
            "active_names": target.gt(0.0).sum(axis=1),
            "costs": turnover * rob_stock.DEFAULT_COST_PER_DOLLAR * CAPITAL,
        },
        index=price.index,
    )


def spy_daily(price: pd.Series) -> pd.DataFrame:
    returns = price.pct_change(fill_method=None).fillna(0.0)
    equity = CAPITAL * (1.0 + returns).cumprod()
    return pd.DataFrame(
        {
            "daily_return": returns,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
        },
        index=price.index,
    )


def run_backtest(refresh: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    raw = download_prices(refresh)
    missing = [ticker for ticker in TICKERS + [BENCHMARK] if ticker not in raw.columns or raw[ticker].dropna().empty]
    if missing:
        raise RuntimeError(f"Missing price data: {missing}")

    prices = clean_prices(raw[TICKERS + [BENCHMARK]]).loc[:END]
    basket_price = prices[TICKERS]
    annual = basket_annual(basket_price.index)
    price_vol = rob.mixed_vol(basket_price.diff())

    forecast = pd.DataFrame(np.nan, index=basket_price.index, columns=basket_price.columns)
    forecast.loc[BACKTEST_START:, :] = CONSTANT_FORECAST

    positions, target, instrument_weights, risk = rob_stock.target_positions(
        basket_price,
        price_vol,
        forecast,
        annual,
        "equal",
        capital=CAPITAL,
        vol_target=VOL_TARGET,
        idm=IDM,
    )
    rob_daily, by_instrument = rob_stock.pnl_from_stock_positions(
        positions,
        basket_price,
        CAPITAL,
        rob_stock.DEFAULT_COST_PER_DOLLAR,
    )
    rob_daily = rob_daily.join(risk)

    equal_daily = equal_weight_daily(basket_price)
    spy = spy_daily(prices[BENCHMARK])

    first = pd.Timestamp(BACKTEST_START)
    daily = {
        "Rob constant +10 forecast": rob_stock.trim_active_daily(rob_daily.loc[first:]),
        "Equal weight basket": rob_stock.trim_active_daily(equal_daily.loc[first:]),
        "SPY": spy.loc[first:].dropna(subset=["daily_return"]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUT / "prices_used.csv")
    positions.loc[first:].to_csv(OUT / "rob_positions.csv")
    target.loc[first:].to_csv(OUT / "rob_target_positions.csv")
    instrument_weights.loc[first:].to_csv(OUT / "rob_instrument_weights.csv")
    risk.loc[first:].to_csv(OUT / "rob_risk_overlay.csv")
    by_instrument.loc[first:].to_csv(OUT / "rob_by_instrument.csv")
    for name, frame in daily.items():
        frame.to_csv(OUT / f"{slug(name)}_daily.csv")
    return daily, prices, positions.loc[first:]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus")


def make_summary(daily: dict[str, pd.DataFrame], positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in daily.items():
        stats = performance_stats(frame["daily_return"], frame["equity"])
        row = {"strategy": name, **stats}
        if "turnover" in frame.columns:
            row["ann_turnover"] = float(frame["turnover"].mean() * rob_stock.BUSINESS_DAYS)
        if "costs" in frame.columns:
            row["ann_cost_pct_capital"] = float(frame["costs"].mean() * rob_stock.BUSINESS_DAYS / CAPITAL)
        if "gross_exposure" in frame.columns:
            row["avg_gross_exposure"] = float(frame["gross_exposure"].mean())
            row["avg_net_exposure"] = float(frame["net_exposure"].mean())
            row["avg_active_names"] = float(frame["active_names"].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary_stats.csv", index=False)
    positions.describe().T.to_csv(OUT / "rob_position_summary.csv")
    return summary


def plot_results(daily: dict[str, pd.DataFrame], summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [2.0, 0.9, 0.9]})
    colors = {
        "Rob constant +10 forecast": "#0F766E",
        "Equal weight basket": "#2563EB",
        "SPY": "#334155",
    }

    for name, frame in daily.items():
        equity = frame["equity"] / frame["equity"].iloc[0]
        axes[0].plot(equity.index, equity, lw=2.2, color=colors[name], label=name)
        axes[1].plot(frame.index, frame["drawdown"], lw=1.6, color=colors[name], label=name)

    axes[0].set_title("BKNG, CRUS, NOW, YOU, EW, STRL, VMI, INTA: Constant Forecast Rob Sizing vs Equal Weight vs SPY")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].grid(True, color="#E2E8F0")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, color="#E2E8F0")

    rob_frame = daily["Rob constant +10 forecast"]
    axes[2].plot(rob_frame.index, rob_frame["gross_exposure"], color="#7C3AED", lw=1.8, label="Rob gross exposure")
    axes[2].plot(rob_frame.index, rob_frame["net_exposure"], color="#D97706", lw=1.4, label="Rob net exposure")
    axes[2].set_ylabel("Exposure / capital")
    axes[2].grid(True, color="#E2E8F0")
    axes[2].legend(loc="upper left", frameon=False)

    table_text = []
    for row in summary.itertuples(index=False):
        table_text.append(
            f"{row.strategy}: Total {row.total_return:.1%}, CAGR {row.cagr:.1%}, "
            f"Vol {row.vol:.1%}, Sharpe {row.sharpe:.2f}, MDD {row.max_drawdown:.1%}"
        )
    fig.text(0.01, 0.01, "\n".join(table_text), fontsize=9, color="#334155")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = OUT / "constant_forecast_basket_vs_spy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(summary: pd.DataFrame, plot_path: Path) -> Path:
    def pct(value: float) -> str:
        return "" if pd.isna(value) else f"{value:.2%}"

    def num(value: float) -> str:
        return "" if pd.isna(value) else f"{value:.2f}"

    lines = [
        "# Custom Constant Forecast Basket 2026 YTD",
        "",
        f"- Tickers: {', '.join(TICKERS)}.",
        f"- Benchmark: {BENCHMARK}.",
        f"- Backtest period: {BACKTEST_START} to {END}.",
        f"- Price warm-up starts {DOWNLOAD_START}; adjusted close from yfinance is cached in `adj_close.csv`.",
        f"- Rob forecast: constant `{CONSTANT_FORECAST:.1f}` for every stock after {BACKTEST_START}.",
        f"- Rob sizing: capital ${CAPITAL:,.0f}, annual volatility target {VOL_TARGET:.0%}, IDM {IDM:.2f}, equal instrument risk weights, buffered integer shares.",
        f"- Costs: {rob_stock.DEFAULT_COST_PER_DOLLAR:.2%} of traded notional for Rob and equal-weight basket; SPY benchmark has no trading cost.",
        "",
        "## Performance",
        "",
        "| Strategy | Start | End | Total Return | CAGR | Vol | Sharpe | MaxDD | Ann Turnover | Ann Cost | Avg Gross | Avg Net | Avg Names |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['start']} | {row['end']} | {pct(row['total_return'])} | {pct(row['cagr'])} | "
            f"{pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['max_drawdown'])} | "
            f"{pct(row.get('ann_turnover', math.nan))} | {pct(row.get('ann_cost_pct_capital', math.nan))} | "
            f"{pct(row.get('avg_gross_exposure', math.nan))} | {pct(row.get('avg_net_exposure', math.nan))} | "
            f"{num(row.get('avg_active_names', math.nan))} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- Plot: `{plot_path.name}`.",
        "- `summary_stats.csv`.",
        "- `rob_positions.csv`.",
        "- `rob_risk_overlay.csv`.",
        "- `prices_used.csv`.",
    ]
    report = OUT / "summary.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    refresh = "--refresh" in sys.argv
    daily, _prices, positions = run_backtest(refresh)
    summary = make_summary(daily, positions)
    plot_path = plot_results(daily, summary)
    report = write_report(summary, plot_path)
    print(report)
    print(plot_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
