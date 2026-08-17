#!/usr/bin/env python3
"""Run a KMLM / MLM-index style managed futures backtest from local CSV data.

This is an educational public-methodology replication, not the official KFA
MLM Index. It uses the public 22-market KMLM universe, monthly rebalancing, a
12-month moving-average long/short signal, sector-level inverse-vol weights,
and a 15% target-volatility version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

import run_rob_style_backtest as bt


OUT = bt.ROOT / "backtests" / "kmlm_like_2000"
START_DATE = "2000-01-03"
SIGNAL_LOOKBACK_DAYS = 252
VOL_WINDOW_DAYS = int(3 * bt.BUSINESS_DAYS)
MIN_VOL_DAYS = int(bt.BUSINESS_DAYS)
TARGET_VOL = 0.15
CAPITAL = 500_000.0
SCALER_MIN = 0.50
SCALER_MAX = 5.00
DEFAULT_EARLY_SCALER = 2.50


KMLM_MARKETS: dict[str, list[str]] = {
    "Commodity": [
        "CORN",
        "CRUDE_W",
        "COPPER",
        "GOLD",
        "HEATOIL",
        "LIVECOW",
        "GAS_US",
        "SOYBEAN",
        "SUGAR11",
        "WHEAT",
        "GASOILINE",
    ],
    "Currency": ["GBP", "CAD", "AUD", "EUR", "JPY", "CHF"],
    "Fixed Income": ["CAD10", "BUND", "JGB-SGX-mini", "GILT", "US10"],
}

MARKET_LABELS = {
    "CORN": "Corn",
    "CRUDE_W": "Crude oil",
    "COPPER": "Copper",
    "GOLD": "Gold",
    "HEATOIL": "Heating oil",
    "LIVECOW": "Live cattle",
    "GAS_US": "Natural gas",
    "SOYBEAN": "Soybeans",
    "SUGAR11": "Sugar",
    "WHEAT": "Wheat",
    "GASOILINE": "Gasoline",
    "GBP": "British pound",
    "CAD": "Canadian dollar",
    "AUD": "Australian dollar",
    "EUR": "Euro",
    "JPY": "Japanese yen",
    "CHF": "Swiss franc",
    "CAD10": "Canadian 10Y bond",
    "BUND": "Euro bund",
    "JGB-SGX-mini": "Japanese government bond",
    "GILT": "Long gilt",
    "US10": "US 10Y Treasury",
}

CRISIS_WINDOWS = {
    "Dot-com bear": ("2000-03-24", "2002-10-09"),
    "2008 calendar": ("2008-01-02", "2008-12-31"),
    "GFC peak-to-trough": ("2007-10-09", "2009-03-09"),
    "2022": ("2022-01-03", "2022-12-30"),
}


@dataclass(frozen=True)
class BacktestResult:
    name: str
    positions: pd.DataFrame
    daily: pd.DataFrame
    by_instrument: pd.DataFrame
    scale: float | pd.Series


def instruments() -> list[str]:
    return [instrument for members in KMLM_MARKETS.values() for instrument in members]


def group_for_instrument() -> dict[str, str]:
    return {
        instrument: group
        for group, members in KMLM_MARKETS.items()
        for instrument in members
    }


def load_full_adjusted_prices(selected: list[str]) -> pd.DataFrame:
    prices = [bt.load_daily_adjusted_price(instrument) for instrument in selected]
    price = pd.concat(prices, axis=1).sort_index()
    return price.loc["1997-01-01":]


def load_traded_price(instrument: str) -> pd.Series:
    data = pd.read_csv(bt.MULTIPLE / f"{instrument}.csv", parse_dates=["DATETIME"])
    price = data.set_index("DATETIME")["PRICE"].sort_index().resample("1B").last().ffill()
    return price.rename(instrument)


def load_traded_prices(selected: list[str], index: pd.Index) -> pd.DataFrame:
    prices = [load_traded_price(instrument) for instrument in selected]
    out = pd.concat(prices, axis=1).sort_index()
    return out.reindex(index).ffill()


def moving_average_signal(price: pd.DataFrame) -> pd.DataFrame:
    average = price.rolling(SIGNAL_LOOKBACK_DAYS, min_periods=SIGNAL_LOOKBACK_DAYS).mean()
    signal = price.ge(average).astype(float).where(average.notna())
    return signal.replace(0.0, -1.0)


def one_x_notional_returns(price: pd.DataFrame, traded_price: pd.DataFrame) -> pd.DataFrame:
    denominator = traded_price.shift(1).abs().replace(0.0, np.nan)
    return price.diff() / denominator


def sector_weights_from_historical_vol(returns_1x: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    groups = group_for_instrument()
    basket_returns = {}
    for group, members in KMLM_MARKETS.items():
        active = signal[members].notna()
        basket_returns[group] = returns_1x[members].where(active).mean(axis=1)

    basket = pd.DataFrame(basket_returns, index=returns_1x.index)
    rolling_vol = basket.rolling(VOL_WINDOW_DAYS, min_periods=MIN_VOL_DAYS).std() * math.sqrt(bt.BUSINESS_DAYS)
    expanding_vol = basket.expanding(min_periods=60).std() * math.sqrt(bt.BUSINESS_DAYS)
    vol = rolling_vol.combine_first(expanding_vol).shift(1)

    inv = 1.0 / vol.replace(0.0, np.nan)
    weights = inv.div(inv.sum(axis=1), axis=0)

    active_group_count = pd.DataFrame(
        {
            group: signal[members].notna().sum(axis=1)
            for group, members in KMLM_MARKETS.items()
        },
        index=signal.index,
    )
    active_groups = active_group_count.gt(0)
    equal_fallback = active_groups.astype(float).div(active_groups.sum(axis=1).replace(0.0, np.nan), axis=0)
    weights = weights.where(active_groups)
    weights = weights.where(weights.sum(axis=1).gt(0.0)).combine_first(equal_fallback)
    return weights.ffill().fillna(1.0 / len(KMLM_MARKETS))


def rebalance_dates(index: pd.Index) -> list[pd.Timestamp]:
    frame = pd.Series(index=index, data=index)
    return list(frame.groupby(frame.index.to_period("M")).first().loc[START_DATE:].values)


def build_positions(
    price: pd.DataFrame,
    traded_price: pd.DataFrame,
    fx: pd.DataFrame,
    meta: dict[str, bt.InstrumentMeta],
    signal: pd.DataFrame,
    sector_weights: pd.DataFrame,
    scale: float | pd.Series = 1.0,
) -> pd.DataFrame:
    selected = instruments()
    idx = price.index
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in selected})
    positions = pd.DataFrame(0.0, index=idx, columns=selected)
    current = pd.Series(0.0, index=selected)
    rebalances = set(rebalance_dates(idx))
    selected_groups = group_for_instrument()

    for offset, date in enumerate(idx):
        if date in rebalances and offset > 0:
            previous_date = idx[offset - 1]
            desired = pd.Series(0.0, index=selected)
            active_by_group: dict[str, list[str]] = {}
            for group, members in KMLM_MARKETS.items():
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

                if isinstance(scale, pd.Series):
                    scaler = float(scale.reindex([previous_date], method="ffill").iloc[0])
                    if not np.isfinite(scaler):
                        scaler = DEFAULT_EARLY_SCALER
                else:
                    scaler = float(scale)

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
                        desired[instrument] = (
                            notional * signal.at[previous_date, instrument] / contract_value
                        )
                current = desired
        positions.loc[date] = current

    return positions


def daily_with_equity(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["daily_return"] = out["net_pnl"] / CAPITAL
    out["equity"] = CAPITAL + out["net_pnl"].cumsum()
    out["nav"] = out["equity"] / CAPITAL
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1.0
    out["compound_nav"] = (1.0 + out["daily_return"]).cumprod()
    out["compound_drawdown"] = out["compound_nav"] / out["compound_nav"].cummax() - 1.0
    return out


def performance_metrics(nav: pd.Series) -> dict[str, float | str]:
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    elapsed_years = (nav.index[-1] - nav.index[0]).days / 365.25
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0 if total_return > -1 else np.nan
    annual_return = returns.mean() * 252.0
    annual_vol = returns.std() * math.sqrt(252.0)
    max_drawdown = (nav / nav.cummax() - 1.0).min()
    return {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "years": elapsed_years,
        "total_return": total_return,
        "cagr": cagr,
        "annual_return_arithmetic": annual_return,
        "annual_vol": annual_vol,
        "sharpe_0rf": annual_return / annual_vol if annual_vol else np.nan,
        "max_drawdown": max_drawdown,
        "calmar": cagr / abs(max_drawdown) if max_drawdown else np.nan,
    }


def annual_return_table(daily_nav: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in daily_nav.groupby(daily_nav.index.year):
        row = {"year": int(year), "start": str(frame.index.min().date()), "end": str(frame.index.max().date())}
        for column in daily_nav.columns:
            nav = frame[column].dropna()
            row[column] = nav.iloc[-1] / nav.iloc[0] - 1.0 if len(nav) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_backtests() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, BacktestResult], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = instruments()
    meta = bt.load_meta()
    missing = [instrument for instrument in selected if not bt.has_required_files(instrument, meta)]
    if missing:
        raise RuntimeError(f"Missing required local futures files: {missing}")

    price = load_full_adjusted_prices(selected)
    traded_price = load_traded_prices(selected, price.index)
    fx = bt.load_fx_matrix(selected, meta, price.index)
    signal = moving_average_signal(price)
    returns_1x = one_x_notional_returns(price, traded_price)
    sector_weights = sector_weights_from_historical_vol(returns_1x, signal)

    raw_positions = build_positions(price, traded_price, fx, meta, signal, sector_weights, scale=1.0)
    zero_costs = pd.DataFrame(0.0, index=price.index, columns=selected)
    raw_daily, _raw_by_instrument = bt.pnl_from_positions(raw_positions, price, fx, meta, selected, zero_costs)
    raw_returns = raw_daily["gross_pnl"] / CAPITAL
    raw_vol = raw_returns.rolling(VOL_WINDOW_DAYS, min_periods=MIN_VOL_DAYS).std() * math.sqrt(bt.BUSINESS_DAYS)
    raw_vol = raw_vol.combine_first(raw_returns.expanding(min_periods=126).std() * math.sqrt(bt.BUSINESS_DAYS))
    rolling_scaler = (TARGET_VOL / raw_vol.shift(1)).clip(SCALER_MIN, SCALER_MAX)
    rolling_scaler = rolling_scaler.bfill().ffill().fillna(DEFAULT_EARLY_SCALER)

    sample_raw_vol = raw_returns.loc[START_DATE:].std() * math.sqrt(bt.BUSINESS_DAYS)
    static_scaler = TARGET_VOL / sample_raw_vol

    costs = bt.cost_matrix(selected, meta, price, fx)
    results = {}
    for name, positions, scale in [
        ("KMLM-like rolling 15V", build_positions(price, traded_price, fx, meta, signal, sector_weights, rolling_scaler), rolling_scaler),
        ("KMLM-like static 15V", raw_positions * static_scaler, static_scaler),
    ]:
        daily, by_instrument = bt.pnl_from_positions(positions, price, fx, meta, selected, costs)
        daily = daily_with_equity(daily.loc[START_DATE:])
        results[name] = BacktestResult(
            name=name,
            positions=positions.loc[START_DATE:],
            daily=daily,
            by_instrument=by_instrument.loc[:, (slice(None), selected)].loc[START_DATE:],
            scale=scale,
        )

    return price.loc[START_DATE:], traded_price.loc[START_DATE:], results, signal.loc[START_DATE:], sector_weights.loc[START_DATE:], returns_1x.loc[START_DATE:]


def load_existing_strategy_nav(path: str) -> pd.Series:
    portfolio = pd.read_csv(bt.ROOT / path, header=[0, 1], index_col=0, parse_dates=True)
    return portfolio[("buffered_integer", "equity")].dropna() / CAPITAL


def load_existing_strategy_return(path: str) -> pd.Series:
    portfolio = pd.read_csv(bt.ROOT / path, header=[0, 1], index_col=0, parse_dates=True)
    return portfolio[("buffered_integer", "daily_return")].dropna()


def comparison_nav(results: dict[str, BacktestResult]) -> pd.DataFrame:
    navs = {
        name: result.daily["nav"]
        for name, result in results.items()
    }
    existing = {
        "17 selected": load_existing_strategy_nav("backtests/rob_style_us_rates_selected_no_vol/portfolio_daily.csv"),
        "40 no-equity": load_existing_strategy_nav("backtests/rob_style_no_equity_40/portfolio_daily.csv"),
    }
    navs = {**existing, **navs}
    start = max(nav.index.min() for nav in navs.values())
    end = min(nav.index.max() for nav in navs.values())
    common_index = pd.date_range(start, end, freq="B")
    out = pd.DataFrame({name: nav.reindex(common_index, method="ffill") for name, nav in navs.items()})
    out = out.dropna()
    return out.div(out.iloc[0])


def return_streams(results: dict[str, BacktestResult]) -> pd.DataFrame:
    returns = {
        "17 selected": load_existing_strategy_return("backtests/rob_style_us_rates_selected_no_vol/portfolio_daily.csv"),
        "40 no-equity": load_existing_strategy_return("backtests/rob_style_no_equity_40/portfolio_daily.csv"),
    }
    for name, result in results.items():
        returns[name] = result.daily["daily_return"]

    start = max(series.index.min() for series in returns.values())
    end = min(series.index.max() for series in returns.values())
    common_index = pd.date_range(start, end, freq="B")
    return pd.DataFrame({name: series.reindex(common_index).fillna(0.0) for name, series in returns.items()})


def return_stream_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    elapsed_years = (returns.index[-1] - returns.index[0]).days / 365.25
    for name in returns.columns:
        series = returns[name]
        compound_nav = (1.0 + series).cumprod()
        ann_return = series.mean() * 252.0
        ann_vol = series.std() * math.sqrt(252.0)
        rows.append(
            {
                "series": name,
                "start": str(returns.index[0].date()),
                "end": str(returns.index[-1].date()),
                "years": elapsed_years,
                "pnl_pct_500k": series.sum(),
                "compound_total_return": compound_nav.iloc[-1] - 1.0,
                "compound_cagr": compound_nav.iloc[-1] ** (1.0 / elapsed_years) - 1.0,
                "annual_return_arithmetic": ann_return,
                "annual_vol": ann_vol,
                "sharpe_0rf": ann_return / ann_vol if ann_vol else np.nan,
                "compound_max_drawdown": (compound_nav / compound_nav.cummax() - 1.0).min(),
            }
        )
    return pd.DataFrame(rows)


def return_stream_annual(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in returns.groupby(returns.index.year):
        row = {"year": int(year), "start": str(frame.index.min().date()), "end": str(frame.index.max().date())}
        for name in returns.columns:
            row[f"{name}_pnl_pct_500k"] = frame[name].sum()
            row[f"{name}_compound_return"] = (1.0 + frame[name]).prod() - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def return_stream_crisis(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, (start, end) in CRISIS_WINDOWS.items():
        frame = returns.loc[start:end]
        for name in returns.columns:
            series = frame[name].dropna()
            if len(series) < 2:
                continue
            compound_nav = (1.0 + series).cumprod()
            rows.append(
                {
                    "window": window,
                    "series": name,
                    "start": str(series.index[0].date()),
                    "end": str(series.index[-1].date()),
                    "pnl_pct_500k": series.sum(),
                    "compound_return": compound_nav.iloc[-1] - 1.0,
                    "vol": series.std() * math.sqrt(252.0),
                    "compound_max_drawdown": (compound_nav / compound_nav.cummax() - 1.0).min(),
                }
            )
    return pd.DataFrame(rows)


def crisis_table(results: dict[str, BacktestResult], comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, (start, end) in CRISIS_WINDOWS.items():
        for name in comparison.columns:
            nav = comparison[name].loc[start:end].dropna()
            if len(nav) < 2:
                continue
            rows.append(
                {
                    "window": window,
                    "series": name,
                    "start": str(nav.index[0].date()),
                    "end": str(nav.index[-1].date()),
                    "return": nav.iloc[-1] / nav.iloc[0] - 1.0,
                    "max_drawdown": (nav / nav.cummax() - 1.0).min(),
                    "vol": nav.pct_change().std() * math.sqrt(252.0),
                }
            )
    return pd.DataFrame(rows)


def asset_and_instrument_breakdown(
    results: dict[str, BacktestResult],
    signal: pd.DataFrame,
    traded_price: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = group_for_instrument()
    asset_rows = []
    instrument_rows = []
    for result in results.values():
        for window, (start, end) in CRISIS_WINDOWS.items():
            for instrument in instruments():
                pnl = result.by_instrument[("net_pnl", instrument)].loc[start:end].sum()
                position = result.by_instrument[("position", instrument)].loc[start:end]
                prices = traded_price[instrument].loc[start:end].dropna()
                price_change = prices.iloc[-1] / prices.iloc[0] - 1.0 if len(prices) > 1 and prices.iloc[0] else np.nan
                instrument_rows.append(
                    {
                        "strategy": result.name,
                        "window": window,
                        "asset_class": groups[instrument],
                        "instrument": instrument,
                        "label": MARKET_LABELS[instrument],
                        "futures_price_change": price_change,
                        "avg_position_contracts": position.mean(),
                        "pct_days_long": (position > 0).mean(),
                        "pct_days_short": (position < 0).mean(),
                        "net_pnl_usd": pnl,
                        "pnl_pct_500k": pnl / CAPITAL,
                    }
                )

            inst = pd.DataFrame([row for row in instrument_rows if row["strategy"] == result.name and row["window"] == window])
            for group, frame in inst.groupby("asset_class"):
                asset_rows.append(
                    {
                        "strategy": result.name,
                        "window": window,
                        "asset_class": group,
                        "instrument_count": len(frame),
                        "median_futures_price_change": frame["futures_price_change"].median(),
                        "mean_futures_price_change": frame["futures_price_change"].mean(),
                        "net_pnl_usd": frame["net_pnl_usd"].sum(),
                        "pnl_pct_500k": frame["pnl_pct_500k"].sum(),
                        "avg_long_markets": frame["pct_days_long"].sum(),
                        "avg_short_markets": frame["pct_days_short"].sum(),
                    }
                )

    return pd.DataFrame(asset_rows), pd.DataFrame(instrument_rows)


def pct_fmt(value: float) -> str:
    return f"{value:.1%}"


def money_fmt(value: float) -> str:
    return f"${value:,.0f}"


def write_outputs(
    price: pd.DataFrame,
    traded_price: pd.DataFrame,
    results: dict[str, BacktestResult],
    signal: pd.DataFrame,
    sector_weights: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = comparison_nav(results)
    metrics = pd.DataFrame(
        [{"series": name, **performance_metrics(comparison[name])} for name in comparison.columns]
    )
    annual = annual_return_table(comparison)
    crisis = crisis_table(results, comparison)
    streams = return_streams(results)
    stream_metrics = return_stream_metrics(streams)
    stream_annual = return_stream_annual(streams)
    stream_crisis = return_stream_crisis(streams)
    asset_breakdown, instrument_breakdown = asset_and_instrument_breakdown(results, signal, traded_price)

    comparison.to_csv(OUT / "comparison_nav.csv", index_label="date")
    metrics.to_csv(OUT / "metrics.csv", index=False)
    annual.to_csv(OUT / "annual_returns.csv", index=False)
    crisis.to_csv(OUT / "crisis_metrics.csv", index=False)
    streams.to_csv(OUT / "return_streams.csv", index_label="date")
    stream_metrics.to_csv(OUT / "return_stream_metrics.csv", index=False)
    stream_annual.to_csv(OUT / "return_stream_annual.csv", index=False)
    stream_crisis.to_csv(OUT / "return_stream_crisis.csv", index=False)
    asset_breakdown.to_csv(OUT / "asset_breakdown.csv", index=False)
    instrument_breakdown.to_csv(OUT / "instrument_breakdown.csv", index=False)
    signal.to_csv(OUT / "signals.csv", index_label="date")
    sector_weights.to_csv(OUT / "sector_weights.csv", index_label="date")
    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "")
        result.daily.to_csv(OUT / f"{safe_name}_daily.csv", index_label="date")
        result.positions.to_csv(OUT / f"{safe_name}_positions.csv", index_label="date")

    plot_comparison(comparison, annual, metrics)
    write_summary(metrics, crisis, stream_metrics, stream_crisis, asset_breakdown, instrument_breakdown, results)


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def plot_comparison(comparison: pd.DataFrame, annual: pd.DataFrame, metrics: pd.DataFrame) -> None:
    colors = {
        "17 selected": "#1f4e79",
        "40 no-equity": "#6f8f2f",
        "KMLM-like rolling 15V": "#8a3f73",
        "KMLM-like static 15V": "#d17c2f",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c5cbd3",
            "axes.labelcolor": "#2f343b",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(18, 16), constrained_layout=False)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.2, 0.85, 1.0], hspace=0.34)
    ax_nav = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_nav)
    ax_ann = fig.add_subplot(gs[2])

    for name in comparison.columns:
        ax_nav.plot(comparison.index, comparison[name], lw=2.0, label=name, color=colors.get(name))
        ax_dd.plot(comparison.index, drawdown(comparison[name]), lw=1.6, color=colors.get(name))

    ax_nav.set_title("KMLM-like Replication vs Existing Futures Systems: Growth of $1", weight="bold")
    ax_nav.set_ylabel("Growth of $1")
    ax_nav.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.1f}x"))
    ax_nav.grid(True, color="#e7ebef", lw=0.8)
    ax_nav.legend(loc="upper left", frameon=False, ncol=4)

    ax_dd.axhline(0, color="#9aa3ad", lw=0.8)
    ax_dd.set_title("Drawdown", weight="bold")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0%}"))
    ax_dd.grid(True, color="#e7ebef", lw=0.8)

    annual_plot = annual.copy()
    years = annual_plot["year"].astype(int).to_numpy()
    series = list(comparison.columns)
    width = 0.18
    offsets = np.linspace(-width * 1.5, width * 1.5, len(series))
    for offset, name in zip(offsets, series):
        ax_ann.bar(years + offset, annual_plot[name], width=width, color=colors.get(name), label=name)
    ax_ann.axhline(0, color="#9aa3ad", lw=0.8)
    ax_ann.set_title("Calendar-Year Returns", weight="bold")
    ax_ann.set_ylabel("Return")
    ax_ann.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:.0%}"))
    ax_ann.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_ann.legend(loc="upper left", frameon=False, ncol=4)
    ax_ann.set_xlim(years.min() - 0.8, years.max() + 0.8)

    ax_dd.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Local CSV data through 2024-03-28; KMLM-like excludes ETF fee/collateral yield and is not official index data.",
        y=0.985,
        fontsize=10,
        color="#626b76",
    )
    fig.savefig(OUT / "kmlm_like_2000_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    metrics: pd.DataFrame,
    crisis: pd.DataFrame,
    stream_metrics: pd.DataFrame,
    stream_crisis: pd.DataFrame,
    asset_breakdown: pd.DataFrame,
    instrument_breakdown: pd.DataFrame,
    results: dict[str, BacktestResult],
) -> None:
    lines = [
        "# KMLM-Like 2000 Backtest",
        "",
        "## Method",
        "",
        "- Universe: public KMLM-style 22 markets: 11 commodities, 6 currencies, 5 fixed-income futures.",
        "- Signal: 12-month / 252-business-day moving average; above = long, below = short.",
        "- Rebalance: first business day of each month, using prior business day's signal.",
        "- Allocation: three sectors weighted by inverse historical basket volatility; constituents equal dollar weighted inside each active sector.",
        "- P&L: local adjusted futures-price differences, FX translated to USD, with local transaction-cost estimates.",
        "- Rolling 15V scales exposure using trailing realised volatility; static 15V uses one full-sample scalar and is diagnostic/look-ahead.",
        "- This is not the official KFA MLM Index or KMLM ETF NAV: no ETF fee, no collateral yield, and no exact licensed index roll/committee details.",
        "",
        "## Return-Stream Metrics",
        "",
        "These use daily P&L divided by the original USD 500k capital. This is the cleanest apples-to-apples risk stream for judging crisis convexity.",
        "",
        "| Series | P&L / 500k | Compound CAGR | Ann ret | Vol | Sharpe | Compound MDD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stream_metrics.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["series"],
                    pct_fmt(row["pnl_pct_500k"]),
                    pct_fmt(row["compound_cagr"]),
                    pct_fmt(row["annual_return_arithmetic"]),
                    pct_fmt(row["annual_vol"]),
                    f"{row['sharpe_0rf']:.2f}",
                    pct_fmt(row["compound_max_drawdown"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Additive Equity Metrics",
            "",
            "These use the fixed-capital equity curve `500k + cumulative P&L`, matching the earlier local Rob reports.",
            "",
            "| Series | CAGR | Ann ret | Vol | Sharpe | MDD | Total |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["series"],
                    pct_fmt(row["cagr"]),
                    pct_fmt(row["annual_return_arithmetic"]),
                    pct_fmt(row["annual_vol"]),
                    f"{row['sharpe_0rf']:.2f}",
                    pct_fmt(row["max_drawdown"]),
                    pct_fmt(row["total_return"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Crisis Return Streams", ""])
    for window in CRISIS_WINDOWS:
        lines.extend(
            [
                f"### {window}",
                "",
                "| Series | P&L / 500k | Compound return | Vol | Compound MDD |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        rows = stream_crisis[stream_crisis["window"].eq(window)].sort_values("pnl_pct_500k", ascending=False)
        for row in rows.to_dict("records"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["series"],
                        pct_fmt(row["pnl_pct_500k"]),
                        pct_fmt(row["compound_return"]),
                        pct_fmt(row["vol"]),
                        pct_fmt(row["compound_max_drawdown"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(["## Additive Equity Crisis Returns", ""])
    for window in CRISIS_WINDOWS:
        lines.extend(
            [
                f"### {window}",
                "",
                "| Series | Return | Vol | MDD |",
                "|---|---:|---:|---:|",
            ]
        )
        rows = crisis[crisis["window"].eq(window)].sort_values("return", ascending=False)
        for row in rows.to_dict("records"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["series"],
                        pct_fmt(row["return"]),
                        pct_fmt(row["vol"]),
                        pct_fmt(row["max_drawdown"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(["## KMLM-Like Asset P&L In Crises", ""])
    for strategy in results:
        lines.extend([f"### {strategy}", ""])
        rows = asset_breakdown[
            asset_breakdown["strategy"].eq(strategy)
            & asset_breakdown["window"].isin(["Dot-com bear", "2008 calendar", "2022"])
        ].sort_values(["window", "pnl_pct_500k"], ascending=[True, False])
        for window in ["Dot-com bear", "2008 calendar", "2022"]:
            lines.extend(
                [
                    f"#### {window}",
                    "",
                    "| Asset | Futures median move | P&L | P&L / 500k | Avg long mkts | Avg short mkts |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in rows[rows["window"].eq(window)].to_dict("records"):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row["asset_class"],
                            pct_fmt(row["median_futures_price_change"]),
                            money_fmt(row["net_pnl_usd"]),
                            pct_fmt(row["pnl_pct_500k"]),
                            f"{row['avg_long_markets']:.1f}",
                            f"{row['avg_short_markets']:.1f}",
                        ]
                    )
                    + " |"
                )
            lines.append("")

    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    price, traded_price, results, signal, sector_weights, _returns_1x = run_backtests()
    write_outputs(price, traded_price, results, signal, sector_weights)
    metrics = pd.read_csv(OUT / "metrics.csv")
    crisis = pd.read_csv(OUT / "crisis_metrics.csv")
    print(f"Wrote KMLM-like outputs to {OUT}")
    print("\nMetrics")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "years": lambda x: f"{x:.1f}",
                "total_return": lambda x: f"{x:.1%}",
                "cagr": lambda x: f"{x:.1%}",
                "annual_return_arithmetic": lambda x: f"{x:.1%}",
                "annual_vol": lambda x: f"{x:.1%}",
                "sharpe_0rf": lambda x: f"{x:.2f}",
                "max_drawdown": lambda x: f"{x:.1%}",
                "calmar": lambda x: f"{x:.2f}",
            },
        )
    )
    print("\nCrisis windows")
    print(
        crisis.to_string(
            index=False,
            formatters={
                "return": lambda x: f"{x:.1%}",
                "max_drawdown": lambda x: f"{x:.1%}",
                "vol": lambda x: f"{x:.1%}",
            },
        )
    )


if __name__ == "__main__":
    main()
