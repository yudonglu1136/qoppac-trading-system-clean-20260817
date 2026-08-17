#!/usr/bin/env python3
"""Compare KMLM-style rules on the 40 no-equity universe with Rob 40 no-equity.

The script separates three questions:

1. How well does a public KMLM-style replication fit actual KMLM ETF returns?
2. What happens if the same KMLM-style rule is applied to the user's 40
   no-equity futures universe?
3. Which sleeve pairs better with SPY in a 30/70 annual-rebalance portfolio?
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import run_kmlm_like_backtest as public_kmlm
import run_rob_style_backtest as bt
import run_rob_style_no_equity_40_backtest as noeq40


OUT = ROOT / "backtests" / "kmlm_40_universe_comparison"
ETF_DATA = ROOT / "data" / "etf_yfinance"

CAPITAL = 500_000.0
BUSINESS_DAYS = bt.BUSINESS_DAYS
START_DATE = "1970-02-03"
SPY_START = "1993-01-29"
TARGET_VOL = 0.15
SIGNAL_LOOKBACK_DAYS = 252
VOL_WINDOW_DAYS = int(3 * BUSINESS_DAYS)
MIN_VOL_DAYS = int(BUSINESS_DAYS)
SCALER_MIN = 0.50
SCALER_MAX = 5.00
DEFAULT_EARLY_SCALER = 2.50

CRISIS_WINDOWS = {
    "Dot-com bear": ("2000-03-24", "2002-10-09"),
    "GFC": ("2007-10-09", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 inflation bear": ("2022-01-03", "2022-10-12"),
}


@dataclass(frozen=True)
class SimulatedStrategy:
    name: str
    universe_name: str
    returns: pd.Series
    nav: pd.Series
    positions: pd.DataFrame
    sector_weights: pd.DataFrame
    signal: pd.DataFrame


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}%}"


def num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def flatten_universe(universe: dict[str, list[str]]) -> list[str]:
    return [instrument for members in universe.values() for instrument in members]


def group_for_universe(universe: dict[str, list[str]]) -> dict[str, str]:
    return {
        instrument: group
        for group, members in universe.items()
        for instrument in members
    }


def load_adjusted_prices(selected: list[str]) -> pd.DataFrame:
    prices = [bt.load_daily_adjusted_price(instrument) for instrument in selected]
    return pd.concat(prices, axis=1).sort_index().loc["1968-01-01":]


def load_traded_price(instrument: str) -> pd.Series:
    data = pd.read_csv(bt.MULTIPLE / f"{instrument}.csv", parse_dates=["DATETIME"])
    price = data.set_index("DATETIME")["PRICE"].sort_index().resample("1B").last().ffill()
    return price.rename(instrument)


def load_traded_prices(selected: list[str], index: pd.Index) -> pd.DataFrame:
    prices = [load_traded_price(instrument) for instrument in selected]
    return pd.concat(prices, axis=1).sort_index().reindex(index).ffill()


def moving_average_signal(price: pd.DataFrame) -> pd.DataFrame:
    average = price.rolling(SIGNAL_LOOKBACK_DAYS, min_periods=SIGNAL_LOOKBACK_DAYS).mean()
    signal = price.ge(average).astype(float).where(average.notna())
    return signal.replace(0.0, -1.0)


def one_x_notional_returns(price: pd.DataFrame, traded_price: pd.DataFrame) -> pd.DataFrame:
    denominator = traded_price.shift(1).abs().replace(0.0, np.nan)
    return price.diff() / denominator


def sector_weights_from_historical_vol(
    returns_1x: pd.DataFrame,
    signal: pd.DataFrame,
    universe: dict[str, list[str]],
) -> pd.DataFrame:
    basket_returns = {}
    for group, members in universe.items():
        active = signal[members].notna()
        basket_returns[group] = returns_1x[members].where(active).mean(axis=1)

    basket = pd.DataFrame(basket_returns, index=returns_1x.index)
    rolling_vol = basket.rolling(VOL_WINDOW_DAYS, min_periods=MIN_VOL_DAYS).std() * math.sqrt(BUSINESS_DAYS)
    expanding_vol = basket.expanding(min_periods=60).std() * math.sqrt(BUSINESS_DAYS)
    vol = rolling_vol.combine_first(expanding_vol).shift(1)

    inv = 1.0 / vol.replace(0.0, np.nan)
    weights = inv.div(inv.sum(axis=1), axis=0)

    active_group_count = pd.DataFrame(
        {group: signal[members].notna().sum(axis=1) for group, members in universe.items()},
        index=signal.index,
    )
    active_groups = active_group_count.gt(0)
    equal_fallback = active_groups.astype(float).div(active_groups.sum(axis=1).replace(0.0, np.nan), axis=0)
    weights = weights.where(active_groups)
    weights = weights.where(weights.sum(axis=1).gt(0.0)).combine_first(equal_fallback)
    return weights.ffill().fillna(1.0 / len(universe))


def rebalance_dates(index: pd.Index, start_date: str) -> set[pd.Timestamp]:
    frame = pd.Series(index=index, data=index)
    dates = frame.groupby(frame.index.to_period("M")).first().loc[start_date:]
    return set(pd.to_datetime(dates.values))


def scaler_for_date(scale: float | pd.Series, previous_date: pd.Timestamp) -> float:
    if isinstance(scale, pd.Series):
        value = float(scale.reindex([previous_date], method="ffill").iloc[0])
        return value if np.isfinite(value) else DEFAULT_EARLY_SCALER
    return float(scale)


def build_positions(
    price: pd.DataFrame,
    traded_price: pd.DataFrame,
    fx: pd.DataFrame,
    meta: dict[str, bt.InstrumentMeta],
    signal: pd.DataFrame,
    sector_weights: pd.DataFrame,
    universe: dict[str, list[str]],
    scale: float | pd.Series,
    start_date: str,
) -> pd.DataFrame:
    selected = flatten_universe(universe)
    idx = price.index
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in selected})
    positions = pd.DataFrame(0.0, index=idx, columns=selected)
    current = pd.Series(0.0, index=selected)
    rebalances = rebalance_dates(idx, start_date)

    for offset, date in enumerate(idx):
        if date in rebalances and offset > 0:
            previous_date = idx[offset - 1]
            desired = pd.Series(0.0, index=selected)
            active_by_group: dict[str, list[str]] = {}
            for group, members in universe.items():
                active = [
                    instrument
                    for instrument in members
                    if pd.notna(signal.at[previous_date, instrument])
                    and pd.notna(traded_price.at[date, instrument])
                    and pd.notna(fx.at[date, instrument])
                    and traded_price.at[date, instrument] > 0.0
                ]
                if active:
                    active_by_group[group] = active

            if active_by_group:
                group_weights = sector_weights.loc[previous_date, list(active_by_group)].astype(float)
                if not np.isfinite(group_weights.sum()) or group_weights.sum() <= 0.0:
                    group_weights = pd.Series(1.0, index=list(active_by_group))
                group_weights = group_weights / group_weights.sum()
                scaler = scaler_for_date(scale, previous_date)

                for group, active in active_by_group.items():
                    for instrument in active:
                        contract_value = (
                            traded_price.at[date, instrument]
                            * point_sizes[instrument]
                            * fx.at[date, instrument]
                        )
                        if not np.isfinite(contract_value) or contract_value <= 0.0:
                            continue
                        notional = CAPITAL * scaler * group_weights[group] / len(active)
                        desired[instrument] = notional * signal.at[previous_date, instrument] / contract_value
                current = desired
        positions.loc[date] = current

    return positions


def simulate_kmlm_rule(
    name: str,
    universe_name: str,
    universe: dict[str, list[str]],
    start_date: str = START_DATE,
) -> SimulatedStrategy:
    selected = flatten_universe(universe)
    meta = bt.load_meta()
    missing = [instrument for instrument in selected if not bt.has_required_files(instrument, meta)]
    if missing:
        raise RuntimeError(f"{name}: missing local futures files: {missing}")

    price = load_adjusted_prices(selected)
    traded_price = load_traded_prices(selected, price.index)
    fx = bt.load_fx_matrix(selected, meta, price.index)
    signal = moving_average_signal(price)
    returns_1x = one_x_notional_returns(price, traded_price)
    sector_weights = sector_weights_from_historical_vol(returns_1x, signal, universe)

    raw_positions = build_positions(
        price, traded_price, fx, meta, signal, sector_weights, universe, scale=1.0, start_date=start_date
    )
    zero_costs = pd.DataFrame(0.0, index=price.index, columns=selected)
    raw_daily, _ = bt.pnl_from_positions(raw_positions, price, fx, meta, selected, zero_costs)
    raw_returns = raw_daily["gross_pnl"] / CAPITAL
    raw_vol = raw_returns.rolling(VOL_WINDOW_DAYS, min_periods=MIN_VOL_DAYS).std() * math.sqrt(BUSINESS_DAYS)
    raw_vol = raw_vol.combine_first(raw_returns.expanding(min_periods=126).std() * math.sqrt(BUSINESS_DAYS))
    rolling_scaler = (TARGET_VOL / raw_vol.shift(1)).clip(SCALER_MIN, SCALER_MAX)
    rolling_scaler = rolling_scaler.ffill().fillna(DEFAULT_EARLY_SCALER)

    positions = build_positions(
        price,
        traded_price,
        fx,
        meta,
        signal,
        sector_weights,
        universe,
        scale=rolling_scaler,
        start_date=start_date,
    )
    costs = bt.cost_matrix(selected, meta, price, fx)
    daily, _ = bt.pnl_from_positions(positions, price, fx, meta, selected, costs)
    returns = (daily["net_pnl"] / CAPITAL).loc[start_date:].rename(name)
    nav = (1.0 + returns.fillna(0.0)).cumprod().rename(name)
    return SimulatedStrategy(
        name=name,
        universe_name=universe_name,
        returns=returns,
        nav=nav,
        positions=positions.loc[start_date:],
        sector_weights=sector_weights.loc[start_date:],
        signal=signal.loc[start_date:],
    )


def load_rob40_returns() -> pd.Series:
    path = ROOT / "backtests" / "rob_style_no_equity_40_long" / "portfolio_daily.csv"
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return frame[("buffered_integer", "daily_return")].rename("Rob 40 no-equity").sort_index()


def download_adjusted_close(ticker: str, start: str = "1990-01-01") -> pd.Series:
    data = yf.download(ticker, start=start, auto_adjust=True, progress=False, timeout=30)
    if data.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close[ticker]
    close = close.dropna().rename(ticker)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def load_or_download_etfs() -> pd.DataFrame:
    ETF_DATA.mkdir(parents=True, exist_ok=True)
    kmlm = download_adjusted_close("KMLM", "2020-01-01")
    spy = download_adjusted_close("SPY", "1993-01-01")
    close = pd.concat([kmlm, spy], axis=1).sort_index()
    close.to_csv(ETF_DATA / "kmlm_spy_adj_close.csv", index_label="date")
    return close


def returns_from_close(close: pd.Series, name: str) -> pd.Series:
    return close.pct_change().dropna().rename(name)


def nav_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def metrics_from_returns(name: str, returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    nav = nav_from_returns(returns)
    years = (returns.index.max() - returns.index.min()).days / 365.25
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    drawdown = nav / nav.cummax() - 1.0
    downside = returns[returns < 0.0].std() * math.sqrt(BUSINESS_DAYS)
    return {
        "series": name,
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "years": years,
        "total_return": nav.iloc[-1] - 1.0,
        "cagr": nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan,
        "ann_return_arithmetic": ann_return,
        "ann_vol": ann_vol,
        "sharpe_0rf": ann_return / ann_vol if ann_vol else np.nan,
        "sortino_0rf": ann_return / downside if downside else np.nan,
        "max_drawdown": float(drawdown.min()),
        "calmar": (nav.iloc[-1] ** (1.0 / years) - 1.0) / abs(float(drawdown.min()))
        if years > 0 and drawdown.min() < 0.0
        else np.nan,
    }


def align_returns(streams: dict[str, pd.Series], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    aligned = pd.concat(streams, axis=1).sort_index()
    if start is not None:
        aligned = aligned.loc[start:]
    if end is not None:
        aligned = aligned.loc[:end]
    return aligned.dropna(how="any")


def fit_to_actual(actual: pd.Series, simulated: pd.Series, name: str) -> tuple[dict[str, float | str], pd.Series]:
    frame = align_returns({"actual": actual, "simulated": simulated})
    x = frame["simulated"].to_numpy()
    y = frame["actual"].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    intercept, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = pd.Series(intercept + beta * simulated, index=simulated.index, name=f"{name} beta+alpha fitted")
    overlap_fitted = pd.Series(intercept + beta * frame["simulated"], index=frame.index)
    residual = frame["actual"] - overlap_fitted
    corr = frame["actual"].corr(frame["simulated"])
    row = {
        "replication": name,
        "start": str(frame.index.min().date()),
        "end": str(frame.index.max().date()),
        "days": len(frame),
        "actual_total_return": nav_from_returns(frame["actual"]).iloc[-1] - 1.0,
        "rep_total_return": nav_from_returns(frame["simulated"]).iloc[-1] - 1.0,
        "fitted_total_return": nav_from_returns(overlap_fitted).iloc[-1] - 1.0,
        "actual_vol": frame["actual"].std() * math.sqrt(BUSINESS_DAYS),
        "rep_vol": frame["simulated"].std() * math.sqrt(BUSINESS_DAYS),
        "fitted_vol": overlap_fitted.std() * math.sqrt(BUSINESS_DAYS),
        "corr_daily": corr,
        "r2_daily": corr**2 if pd.notna(corr) else np.nan,
        "ols_beta": beta,
        "ols_intercept_ann": intercept * BUSINESS_DAYS,
        "tracking_error_raw": (frame["actual"] - frame["simulated"]).std() * math.sqrt(BUSINESS_DAYS),
        "tracking_error_fitted": residual.std() * math.sqrt(BUSINESS_DAYS),
        "active_max_dd_fitted": float(nav_from_returns(residual).div(nav_from_returns(residual).cummax()).sub(1.0).min()),
    }
    return row, fitted


def annual_rebalance_portfolio(returns: pd.DataFrame, target_weights: dict[str, float]) -> pd.Series:
    returns = returns.dropna().copy()
    sleeve_values = {asset: target_weights[asset] for asset in target_weights}
    previous_total = sum(sleeve_values.values())
    current_year: int | None = None
    records = []
    for date, row in returns.iterrows():
        if current_year is None or date.year != current_year:
            total_before = sum(sleeve_values.values())
            sleeve_values = {asset: total_before * weight for asset, weight in target_weights.items()}
            current_year = date.year
        for asset in sleeve_values:
            sleeve_values[asset] *= 1.0 + float(row[asset])
        total_after = sum(sleeve_values.values())
        records.append((date, total_after / previous_total - 1.0))
        previous_total = total_after
    return pd.Series(dict(records), name="portfolio_return").sort_index()


def build_spy_mixes(strategy_returns: dict[str, pd.Series], spy_returns: pd.Series, start: str, end: str) -> dict[str, pd.Series]:
    mixes = {}
    for name, returns in strategy_returns.items():
        aligned = align_returns({name: returns, "SPY": spy_returns}, start=start, end=end)
        mix = annual_rebalance_portfolio(aligned, {name: 0.30, "SPY": 0.70})
        mixes[f"30 {name} / 70 SPY annual"] = mix
    return mixes


def crisis_metrics(streams: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for window, (start, end) in CRISIS_WINDOWS.items():
        for name, returns in streams.items():
            sub = returns.loc[start:end].dropna()
            if len(sub) < 2:
                continue
            rows.append({"window": window, **metrics_from_returns(name, sub)})
    return pd.DataFrame(rows)


def annual_returns(streams: dict[str, pd.Series]) -> pd.DataFrame:
    aligned = pd.concat(streams, axis=1).sort_index()
    rows = []
    for year, frame in aligned.groupby(aligned.index.year):
        row = {"year": int(year), "start": str(frame.index.min().date()), "end": str(frame.index.max().date())}
        for name in aligned.columns:
            series = frame[name].dropna()
            row[name] = nav_from_returns(series).iloc[-1] - 1.0 if len(series) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(
    long_streams: dict[str, pd.Series],
    fit_streams: dict[str, pd.Series],
    mix_streams: dict[str, pd.Series],
    spy_returns: pd.Series,
) -> None:
    colors = {
        "Rob 40 no-equity": "#335c81",
        "KMLM rule on 40 no-equity": "#8f5d2c",
        "KMLM public-22 rule": "#6d3f8f",
        "KMLM actual": "#111827",
        "SPY": "#222222",
        "30 Rob 40 no-equity / 70 SPY annual": "#4477aa",
        "30 KMLM rule on 40 no-equity / 70 SPY annual": "#cc6677",
        "30 KMLM public-22 rule / 70 SPY annual": "#117733",
        "30 KMLM actual / 70 SPY annual": "#ddcc77",
    }
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), gridspec_kw={"height_ratios": [1.4, 1.0, 1.1, 1.25, 1.15]})

    long_aligned = align_returns(long_streams, start=SPY_START)
    navs = long_aligned.apply(nav_from_returns)
    for name in navs.columns:
        axes[0].plot(navs.index, navs[name], label=name, color=colors.get(name))
    axes[0].set_yscale("log")
    axes[0].set_title("Long-Term Futures Sleeves, Common SPY-Era Overlap")
    axes[0].set_ylabel("Growth of $1, log")
    axes[0].legend(loc="upper left", ncols=3)
    axes[0].grid(alpha=0.25)

    drawdowns = navs / navs.cummax() - 1.0
    for name in drawdowns.columns:
        axes[1].plot(drawdowns.index, drawdowns[name], color=colors.get(name), label=name)
    axes[1].set_title("Sleeve Drawdowns")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].grid(alpha=0.25)

    fit_aligned = align_returns(fit_streams)
    fit_navs = fit_aligned.apply(nav_from_returns)
    for name in fit_navs.columns:
        axes[2].plot(fit_navs.index, fit_navs[name], color=colors.get(name), label=name)
    axes[2].set_title("Actual KMLM ETF Fit Window")
    axes[2].set_ylabel("Growth of $1")
    axes[2].legend(loc="upper left", ncols=3)
    axes[2].grid(alpha=0.25)

    mix_with_spy = {"SPY": spy_returns, **mix_streams}
    mix_aligned = align_returns(mix_with_spy, start=SPY_START, end=str(navs.index.max().date()))
    mix_navs = mix_aligned.apply(nav_from_returns)
    for name in mix_navs.columns:
        axes[3].plot(mix_navs.index, mix_navs[name], color=colors.get(name), label=name)
    axes[3].set_yscale("log")
    axes[3].set_title("30% Futures Sleeve / 70% SPY, Annual Rebalance")
    axes[3].set_ylabel("Growth of $1, log")
    axes[3].legend(loc="upper left", ncols=2)
    axes[3].grid(alpha=0.25)

    annual = annual_returns({"SPY": spy_returns, **mix_streams})
    annual = annual[(annual["year"] >= 2000) & (annual["year"] <= int(navs.index.max().year))]
    columns = ["SPY"] + list(mix_streams)
    x = np.arange(len(annual))
    width = min(0.75 / len(columns), 0.18)
    offsets = np.linspace(-width * (len(columns) - 1) / 2, width * (len(columns) - 1) / 2, len(columns))
    for offset, name in zip(offsets, columns):
        axes[4].bar(x + offset, annual[name] * 100.0, width=width, color=colors.get(name), label=name)
    axes[4].axhline(0.0, color="#666666", linewidth=0.8)
    axes[4].set_title("Calendar-Year Returns: SPY vs 30/70 Mixes")
    axes[4].set_ylabel("Return (%)")
    tick_locs = [i for i, year in enumerate(annual["year"]) if int(year) % 2 == 0]
    axes[4].set_xticks(tick_locs)
    axes[4].set_xticklabels([str(int(annual.iloc[i]["year"])) for i in tick_locs])
    axes[4].legend(loc="upper left", ncols=2)
    axes[4].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "kmlm_40_vs_no_equity_spy_mix.png", dpi=180)
    plt.close(fig)


def markdown_table_metrics(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for col in ["total_return", "cagr", "ann_return_arithmetic", "ann_vol", "max_drawdown"]:
        if col in show:
            show[col] = show[col].map(pct)
    for col in ["sharpe_0rf", "sortino_0rf", "calmar", "corr_to_spy"]:
        if col in show:
            show[col] = show[col].map(num)
    if "years" in show:
        show["years"] = show["years"].map(lambda value: f"{value:.1f}")
    return show.to_markdown(index=False)


def write_summary(
    long_metrics: pd.DataFrame,
    spy_mix_metrics: pd.DataFrame,
    fit_metrics: pd.DataFrame,
    actual_recent_metrics: pd.DataFrame,
    crisis: pd.DataFrame,
) -> None:
    fit_show = fit_metrics.copy()
    for col in [
        "actual_total_return",
        "rep_total_return",
        "fitted_total_return",
        "actual_vol",
        "rep_vol",
        "fitted_vol",
        "ols_intercept_ann",
        "tracking_error_raw",
        "tracking_error_fitted",
        "active_max_dd_fitted",
    ]:
        fit_show[col] = fit_show[col].map(pct)
    for col in ["corr_daily", "r2_daily", "ols_beta"]:
        fit_show[col] = fit_show[col].map(num)

    lines = [
        "# KMLM Rule On 40 No-Equity Universe vs Rob 40 No-Equity",
        "",
        "## What Was Tested",
        "",
        "- `KMLM public-22 rule`: public KMLM/MLM-style 22-market universe, 12-month moving-average long/short signal, monthly rebalance, inverse-vol sector weights, 15% trailing volatility scaling.",
        "- `KMLM rule on 40 no-equity`: the same KMLM rule stack applied to the current 40 no-equity futures universe, including Bond, FX, Ags, Metals, OilGas, and Vol buckets.",
        "- `Rob 40 no-equity`: the existing best Rob-style 40 no-equity system using the full Rob forecast/risk framework and buffered integer positions.",
        "- Actual KMLM ETF and SPY were refreshed from yfinance. Simulated futures data still ends at the local CSV endpoint, 2024-03-28/29.",
        "",
        "## Fit To Actual KMLM ETF",
        "",
        fit_show.to_markdown(index=False),
        "",
        "Interpretation: the closer proxy for actual KMLM is the public 22-market KMLM-style rule, not the 40-universe version. The 40-universe version answers a different question: what if KMLM's simple monthly 12-month trend engine traded your broader no-equity futures universe?",
        "",
        "## Long-Term Futures Sleeve Metrics",
        "",
        markdown_table_metrics(long_metrics),
        "",
        "## 30/70 With SPY, Annual Rebalance",
        "",
        markdown_table_metrics(spy_mix_metrics),
        "",
        "## Actual KMLM ETF Recent Metrics",
        "",
        markdown_table_metrics(actual_recent_metrics),
        "",
        "## Crisis Windows",
        "",
        markdown_table_metrics(crisis),
        "",
        "## Conclusion",
        "",
        "- For replicating actual KMLM, the public 22-market version fits materially better than the 40-universe KMLM-rule variant.",
        "- For long-term standalone performance on the local data, Rob 40 no-equity remains stronger than the KMLM-style rule variants. The reason is not just universe; it is the richer Rob forecast stack plus instrument-level volatility sizing and buffered execution.",
        "- For pairing with SPY at a fixed 30% capital weight, Rob 40 no-equity gives the strongest hedge/return improvement in this local history, but it also contributes a higher-volatility sleeve than KMLM's 15V design.",
        "- KMLM-style is cleaner and closer to a commercial CTA ETF design; Rob 40 no-equity is more aggressive and less ETF-like. If the objective is pure portfolio convexity with SPY, the local evidence favors Rob 40. If the objective is tracking actual KMLM behavior, use the public 22-market proxy.",
        "",
        "## Files",
        "",
        "- `simulated_return_streams.csv`",
        "- `fit_to_actual_kmlm.csv`",
        "- `long_term_metrics.csv`",
        "- `spy_mix_metrics.csv`",
        "- `actual_kmlm_recent_metrics.csv`",
        "- `crisis_metrics.csv`",
        "- `kmlm_40_vs_no_equity_spy_mix.png`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    etf_close = load_or_download_etfs()
    actual_kmlm = returns_from_close(etf_close["KMLM"], "KMLM actual")
    spy_returns = returns_from_close(etf_close["SPY"], "SPY")

    public22 = simulate_kmlm_rule("KMLM public-22 rule", "Public KMLM 22", public_kmlm.KMLM_MARKETS)
    forty_rule = simulate_kmlm_rule("KMLM rule on 40 no-equity", "No-equity 40", noeq40.NO_EQUITY_UNIVERSE)
    rob40 = load_rob40_returns()

    futures_end = min(public22.returns.index.max(), forty_rule.returns.index.max(), rob40.index.max())
    long_streams = {
        "Rob 40 no-equity": rob40.loc[:futures_end],
        "KMLM rule on 40 no-equity": forty_rule.returns.loc[:futures_end],
        "KMLM public-22 rule": public22.returns.loc[:futures_end],
    }

    fit_rows = []
    fitted_streams = {"KMLM actual": actual_kmlm}
    for strategy in [public22, forty_rule]:
        overlap_end = min(strategy.returns.index.max(), actual_kmlm.index.max())
        row, fitted = fit_to_actual(actual_kmlm.loc[:overlap_end], strategy.returns.loc[:overlap_end], strategy.name)
        fit_rows.append(row)
        fitted_streams[strategy.name] = strategy.returns.loc[row["start"] : row["end"]]
        fitted_streams[f"{strategy.name} fitted"] = fitted.loc[row["start"] : row["end"]]
    fit_metrics = pd.DataFrame(fit_rows)

    long_metrics = pd.DataFrame([metrics_from_returns(name, returns) for name, returns in long_streams.items()])
    spy_for_long = spy_returns.loc[:futures_end]
    for row in long_metrics.index:
        name = long_metrics.at[row, "series"]
        aligned = align_returns({name: long_streams[name], "SPY": spy_for_long}, start=SPY_START)
        long_metrics.at[row, "corr_to_spy"] = aligned[name].corr(aligned["SPY"])

    mix_streams = build_spy_mixes(long_streams, spy_returns, SPY_START, str(futures_end.date()))
    actual_mix = build_spy_mixes({"KMLM actual": actual_kmlm}, spy_returns, str(actual_kmlm.index.min().date()), str(actual_kmlm.index.max().date()))
    spy_mix_metrics = pd.DataFrame(
        [metrics_from_returns("SPY", spy_returns.loc[SPY_START:futures_end])] +
        [metrics_from_returns(name, returns) for name, returns in mix_streams.items()]
    )
    for row in spy_mix_metrics.index:
        name = spy_mix_metrics.at[row, "series"]
        series = spy_returns.loc[SPY_START:futures_end] if name == "SPY" else mix_streams[name]
        aligned = align_returns({name: series, "SPY": spy_returns}, start=SPY_START, end=str(futures_end.date()))
        spy_mix_metrics.at[row, "corr_to_spy"] = aligned[name].corr(aligned["SPY"])

    actual_recent_metrics = pd.DataFrame(
        [metrics_from_returns("KMLM actual", actual_kmlm), metrics_from_returns("SPY same dates", spy_returns.loc[actual_kmlm.index.min() : actual_kmlm.index.max()])]
        + [metrics_from_returns(name, returns) for name, returns in actual_mix.items()]
    )
    for row in actual_recent_metrics.index:
        name = actual_recent_metrics.at[row, "series"]
        series = actual_kmlm if name == "KMLM actual" else (
            spy_returns.loc[actual_kmlm.index.min() : actual_kmlm.index.max()] if name == "SPY same dates" else actual_mix[name]
        )
        aligned = align_returns({name: series, "SPY": spy_returns})
        actual_recent_metrics.at[row, "corr_to_spy"] = aligned[name].corr(aligned["SPY"])

    combined_streams = {**long_streams, "SPY": spy_returns.loc[:futures_end], **mix_streams}
    crisis = crisis_metrics(combined_streams)

    simulated_streams = pd.concat(
        {
            **long_streams,
            "KMLM actual": actual_kmlm,
            "SPY": spy_returns,
            **mix_streams,
            **actual_mix,
        },
        axis=1,
    ).sort_index()
    simulated_streams.to_csv(OUT / "simulated_return_streams.csv", index_label="date")
    fit_metrics.to_csv(OUT / "fit_to_actual_kmlm.csv", index=False)
    long_metrics.to_csv(OUT / "long_term_metrics.csv", index=False)
    spy_mix_metrics.to_csv(OUT / "spy_mix_metrics.csv", index=False)
    actual_recent_metrics.to_csv(OUT / "actual_kmlm_recent_metrics.csv", index=False)
    crisis.to_csv(OUT / "crisis_metrics.csv", index=False)
    annual_returns({**long_streams, "SPY": spy_returns.loc[:futures_end], **mix_streams}).to_csv(
        OUT / "annual_returns.csv", index=False
    )
    forty_rule.sector_weights.to_csv(OUT / "kmlm_rule_40_sector_weights.csv", index_label="date")
    forty_rule.positions.to_csv(OUT / "kmlm_rule_40_positions.csv", index_label="date")

    plot_outputs(long_streams, fitted_streams, mix_streams, spy_returns.loc[:futures_end])
    write_summary(long_metrics, spy_mix_metrics, fit_metrics, actual_recent_metrics, crisis)

    print(f"Wrote {OUT}")
    print(long_metrics.to_string(index=False))
    print(spy_mix_metrics.to_string(index=False))
    print(fit_metrics.to_string(index=False))
    print(actual_recent_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
