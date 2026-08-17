#!/usr/bin/env python3
"""Run a 40-market multi-rule futures backtest from shipped pysystemtrade CSV data."""

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
MULTIPLE = FUTURES_DATA / "multiple_prices_csv"
FX = FUTURES_DATA / "fx_prices_csv"
CONFIG = FUTURES_DATA / "csvconfig"
OUT = ROOT / "backtests" / "forty_market_multirule"

CAPITAL = 500_000.0
ANNUAL_VOL_TARGET = 0.25
AVERAGE_ABS_FORECAST = 10.0
FORECAST_CAP = 20.0
MAX_FDM = 2.5
PORTFOLIO_IDM = 2.5
BUSINESS_DAYS = 256.0


BUCKETS = {
    "Equity": [
        "SP500_micro",  # ES/MES proxy
        "NASDAQ_micro",
        "DOW",
        "SP400",
        "EURO600",
        "DAX",
        "CAC",
        "IBEX",
        "AEX",
        "FTSE100",
        "NIKKEI",
        "HANG",
    ],
    "Commodities": [
        "GOLD_micro",  # GC/MGC proxy
        "SILVER",
        "COPPER",
        "CRUDE_W",  # CL proxy
        "BRENT_W",
        "GAS_US",
        "HEATOIL",
        "GASOIL",
        "CORN",
        "SOYBEAN",
        "WHEAT",
        "COFFEE",
        "COCOA",
        "COTTON",
        "LEANHOG",
        "LIVECOW",
    ],
    "FX": [
        "AUD",
        "CAD",
        "CHF",
        "EUR",
        "GBP",
        "JPY",
        "MXP",
        "NZD",
        "SEK",
        "NOK",
        "ZAR",
        "DX",
    ],
}

RULE_WEIGHTS = {
    "ewmac16_64": 0.10,
    "ewmac32_128": 0.10,
    "ewmac64_256": 0.10,
    "breakout40": 0.10,
    "breakout80": 0.10,
    "breakout160": 0.10,
    "carry30": 0.1333333333,
    "carry90": 0.1333333333,
    "carry180": 0.1333333334,
}


@dataclass(frozen=True)
class InstrumentMeta:
    point_size: float
    currency: str
    asset_class: str
    per_block: float
    percentage: float
    per_trade: float
    spread_cost: float


def all_instruments() -> list[str]:
    return [instrument for instruments in BUCKETS.values() for instrument in instruments]


def instrument_bucket(instrument: str) -> str:
    for bucket, instruments in BUCKETS.items():
        if instrument in instruments:
            return bucket
    raise KeyError(instrument)


def instrument_weight(instrument: str) -> float:
    bucket = instrument_bucket(instrument)
    return (1.0 / len(BUCKETS)) / len(BUCKETS[bucket])


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
    data = pd.read_csv(path, parse_dates=["DATETIME"])
    series = data.set_index("DATETIME")["price"].sort_index()
    daily = series.resample("1B").last().ffill()
    daily.name = instrument
    return daily


def load_daily_multiple(instrument: str) -> pd.DataFrame:
    path = MULTIPLE / f"{instrument}.csv"
    data = pd.read_csv(path, parse_dates=["DATETIME"])
    data = data.set_index("DATETIME").sort_index()
    return data.resample("1B").last().ffill()


def load_fx_rate(currency: str, index: pd.Index) -> pd.Series:
    if currency == "USD":
        return pd.Series(1.0, index=index, name="fx")
    path = FX / f"{currency}USD.csv"
    if not path.exists():
        raise FileNotFoundError(f"No FX file for {currency}: {path}")
    data = pd.read_csv(path, parse_dates=["DATETIME"])
    fx = data.set_index("DATETIME")["PRICE"].sort_index().resample("1B").last().ffill()
    return fx.reindex(index, method="ffill").bfill().rename("fx")


def mixed_daily_price_vol(price: pd.Series) -> pd.Series:
    returns = price.diff()
    fast_vol = returns.ewm(span=35, min_periods=10).std()
    slow_vol = returns.ewm(span=35 * 20, min_periods=100).std()
    vol = 0.65 * fast_vol + 0.35 * slow_vol
    return vol.ffill().replace(0.0, np.nan).rename("daily_price_vol")


def ewmac(price: pd.Series, vol: pd.Series, fast: int, slow: int) -> pd.Series:
    raw = (price.ewm(span=fast, min_periods=fast).mean() - price.ewm(span=slow, min_periods=slow).mean()) / vol
    return raw


def breakout(price: pd.Series, lookback: int) -> pd.Series:
    smooth = max(int(lookback / 4.0), 1)
    roll_max = price.rolling(lookback, min_periods=max(int(math.ceil(lookback / 2.0)), 2)).max()
    roll_min = price.rolling(lookback, min_periods=max(int(math.ceil(lookback / 2.0)), 2)).min()
    roll_range = (roll_max - roll_min).replace(0.0, np.nan)
    output = 40.0 * ((price - (roll_max + roll_min) / 2.0) / roll_range)
    return output.ewm(span=smooth, min_periods=max(int(math.ceil(smooth / 2.0)), 1)).mean()


def contract_year_fraction(contract_series: pd.Series) -> pd.Series:
    contract = pd.to_numeric(contract_series, errors="coerce")
    years = np.floor(contract / 10000.0)
    months = (contract % 10000.0) / 100.0
    return years + months / 12.0


def raw_carry(price: pd.Series, price_vol: pd.Series, instrument: str) -> pd.Series:
    multiple = load_daily_multiple(instrument)
    raw_roll = multiple["PRICE"] - multiple["CARRY"]
    raw_roll = raw_roll.replace(0.0, np.nan)
    differential = contract_year_fraction(multiple["CARRY_CONTRACT"]) - contract_year_fraction(
        multiple["PRICE_CONTRACT"]
    )
    floor = 1.0 / 365.0
    differential = differential.where(differential.abs() >= floor, np.sign(differential).replace(0, 1) * floor)
    annualised_roll = raw_roll / differential
    aligned_vol = price_vol.reindex(annualised_roll.index, method="ffill")
    ann_stdev = aligned_vol * math.sqrt(BUSINESS_DAYS)
    carry = annualised_roll / ann_stdev
    return carry.reindex(price.index, method="ffill")


def scale_and_cap(raw: pd.Series) -> pd.Series:
    valid = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 100:
        return raw * np.nan
    mean_abs = valid.abs().mean()
    if not np.isfinite(mean_abs) or mean_abs <= 0:
        return raw * np.nan
    scalar = np.clip(AVERAGE_ABS_FORECAST / mean_abs, 0.01, 100.0)
    return (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)


def forecasts_for_instrument(instrument: str, price: pd.Series, price_vol: pd.Series) -> pd.DataFrame:
    carry = raw_carry(price, price_vol, instrument)
    raw_forecasts = {
        "ewmac16_64": ewmac(price, price_vol, 16, 64),
        "ewmac32_128": ewmac(price, price_vol, 32, 128),
        "ewmac64_256": ewmac(price, price_vol, 64, 256),
        "breakout40": breakout(price, 40),
        "breakout80": breakout(price, 80),
        "breakout160": breakout(price, 160),
        "carry30": carry.ewm(span=30, min_periods=15).mean(),
        "carry90": carry.ewm(span=90, min_periods=45).mean(),
        "carry180": carry.ewm(span=180, min_periods=90).mean(),
    }
    forecasts = pd.DataFrame({name: scale_and_cap(series) for name, series in raw_forecasts.items()})
    return forecasts


def forecast_diversification_multiplier(forecasts: pd.DataFrame) -> float:
    valid_columns = [col for col in forecasts.columns if forecasts[col].dropna().shape[0] >= 250]
    if len(valid_columns) < 2:
        return 1.0
    weights = pd.Series({col: RULE_WEIGHTS[col] for col in valid_columns}, dtype=float)
    weights = weights / weights.sum()
    corr = forecasts[valid_columns].dropna().corr().fillna(0.0)
    corr_values = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 1.0)
    variance = float(weights.values.T @ corr_values @ weights.values)
    if variance <= 0 or not np.isfinite(variance):
        return 1.0
    return float(np.clip(1.0 / math.sqrt(variance), 1.0, MAX_FDM))


def combine_forecasts(forecasts: pd.DataFrame, fdm: float) -> pd.Series:
    weights = pd.Series(RULE_WEIGHTS, dtype=float)
    weighted = forecasts.mul(weights, axis=1)
    available_weight = forecasts.notna().mul(weights, axis=1).sum(axis=1)
    combined = weighted.sum(axis=1) / available_weight.replace(0.0, np.nan)
    return (combined * fdm).clip(-FORECAST_CAP, FORECAST_CAP).rename("combined_forecast")


def cost_per_contract_usd(meta: InstrumentMeta, fx: pd.Series, price: pd.Series) -> pd.Series:
    local_fixed = meta.per_block + meta.per_trade + meta.spread_cost * meta.point_size
    local_percentage = (meta.percentage / 100.0) * price.abs() * meta.point_size
    return (local_fixed + local_percentage).reindex(fx.index, method="ffill").fillna(local_fixed) * fx


def run_instrument(instrument: str, meta: InstrumentMeta) -> tuple[pd.DataFrame, dict[str, float]]:
    price = load_daily_adjusted_price(instrument)
    price_vol = mixed_daily_price_vol(price)
    fx = load_fx_rate(meta.currency, price.index)
    forecasts = forecasts_for_instrument(instrument, price, price_vol)
    fdm = forecast_diversification_multiplier(forecasts)
    combined_forecast = combine_forecasts(forecasts, fdm)

    daily_cash_vol_target = CAPITAL * ANNUAL_VOL_TARGET / math.sqrt(BUSINESS_DAYS)
    instrument_cash_vol = price_vol.abs() * meta.point_size * fx
    subsystem_position = daily_cash_vol_target / instrument_cash_vol
    target_position = (
        subsystem_position
        * combined_forecast
        / AVERAGE_ABS_FORECAST
        * instrument_weight(instrument)
        * PORTFOLIO_IDM
    )
    rounded_position = target_position.round()
    held_position = rounded_position.shift(1).fillna(0.0)
    price_change = price.diff().fillna(0.0)
    gross_pnl = held_position * price_change * meta.point_size * fx
    trades = rounded_position.diff().abs().fillna(0.0)
    costs = trades * cost_per_contract_usd(meta, fx, price)
    net_pnl = gross_pnl - costs

    out = pd.DataFrame(
        {
            "price": price,
            "fx": fx,
            "price_vol": price_vol,
            "combined_forecast": combined_forecast,
            "target_position": target_position,
            "rounded_position": rounded_position,
            "held_position": held_position,
            "gross_pnl": gross_pnl,
            "costs": costs,
            "trades": trades,
            "net_pnl": net_pnl,
        }
    )
    for col in forecasts.columns:
        out[f"forecast_{col}"] = forecasts[col]
    out["instrument"] = instrument
    out["bucket"] = instrument_bucket(instrument)
    out["instrument_weight"] = instrument_weight(instrument)
    out["fdm"] = fdm

    diagnostic = {
        "fdm": fdm,
        "weight": instrument_weight(instrument),
        "first_forecast": combined_forecast.first_valid_index(),
        "last_price": price.last_valid_index(),
        "valid_forecast_days": int(combined_forecast.dropna().shape[0]),
    }
    return out, diagnostic


def stats_from_portfolio(portfolio: pd.DataFrame) -> dict[str, float | str]:
    daily_returns = portfolio["daily_return"].dropna()
    ann_return = daily_returns.mean() * BUSINESS_DAYS
    ann_vol = daily_returns.std() * math.sqrt(BUSINESS_DAYS)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    return {
        "start": str(portfolio.index.min().date()),
        "end": str(portfolio.index.max().date()),
        "years": len(portfolio) / BUSINESS_DAYS,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": portfolio["drawdown"].min(),
        "total_return": portfolio["equity"].iloc[-1] / CAPITAL - 1.0,
        "total_costs": portfolio["costs"].sum(),
    }


def format_stat(key: str, value: float | str) -> str:
    if key in {"start", "end"}:
        return str(value)
    if key == "years":
        return f"{value:.1f}"
    if key == "sharpe":
        return f"{value:.2f}"
    if key == "total_costs":
        return f"${value:,.0f}"
    return f"{value:.2%}"


def write_summary(
    stats: dict[str, float | str],
    diagnostics: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    instrument_summary: pd.DataFrame,
) -> None:
    lines = [
        "# 40-Market Multi-Rule Futures Backtest",
        "",
        "## Data Source",
        "",
        f"- Adjusted prices: `{ADJUSTED}`",
        f"- Multiple/carry prices: `{MULTIPLE}`",
        f"- FX: `{FX}`",
        f"- Instrument metadata and costs: `{CONFIG}`",
        "- These are shipped `pysystemtrade` CSV files, mostly ending around 2024-03-28. They are for learning and framework tests, not current live research.",
        "",
        "## Universe",
        "",
    ]
    for bucket, instruments in BUCKETS.items():
        lines.append(f"- {bucket} ({len(instruments)}): {', '.join(instruments)}")
    lines.extend(
        [
            "",
            "## Rule Weights",
            "",
        ]
    )
    for rule, weight in RULE_WEIGHTS.items():
        lines.append(f"- {rule}: {weight:.2%}")
    lines.extend(
        [
            "",
            "## Portfolio Settings",
            "",
            f"- Capital: ${CAPITAL:,.0f}",
            f"- Annual volatility target: {ANNUAL_VOL_TARGET:.0%}",
            f"- Forecast cap/floor: +/-{FORECAST_CAP:.0f}",
            f"- Average absolute forecast denominator: {AVERAGE_ABS_FORECAST:.0f}",
            f"- Max FDM: {MAX_FDM:.1f}",
            f"- Portfolio IDM: {PORTFOLIO_IDM:.1f}",
            "- Buckets: one third each to Equity, Commodities, FX; equal weight inside each bucket.",
            "- Position rounding: nearest whole contract.",
            "",
            "## Results",
            "",
        ]
    )
    for key, value in stats.items():
        lines.append(f"- {key}: {format_stat(key, value)}")
    lines.extend(
        [
            "",
            "## Bucket Results",
            "",
        ]
    )
    for row in bucket_summary.itertuples(index=False):
        lines.append(
            f"- {row.bucket}: net ${row.net_pnl:,.0f}; gross ${row.gross_pnl:,.0f}; costs ${row.costs:,.0f}"
        )
    lines.extend(
        [
            "",
            "## Highest Cost Instruments",
            "",
        ]
    )
    for row in instrument_summary.sort_values("costs", ascending=False).head(5).itertuples(index=False):
        lines.append(
            f"- {row.instrument} ({row.bucket}): costs ${row.costs:,.0f}; net ${row.net_pnl:,.0f}; trades {row.total_traded_contracts:,.0f}"
        )
    lines.extend(
        [
            "",
            "## FDM Summary",
            "",
            f"- average FDM: {diagnostics['fdm'].mean():.2f}",
            f"- min FDM: {diagnostics['fdm'].min():.2f}",
            f"- max FDM: {diagnostics['fdm'].max():.2f}",
            "",
            "## ES/GC/CL Mapping",
            "",
            "- ES: `SP500_micro`",
            "- GC: `GOLD_micro`",
            "- CL: `CRUDE_W`",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    required = all_instruments()
    missing = [
        instrument
        for instrument in required
        if instrument not in meta
        or not (ADJUSTED / f"{instrument}.csv").exists()
        or not (MULTIPLE / f"{instrument}.csv").exists()
    ]
    if missing:
        raise RuntimeError(f"Missing required data for: {missing}")

    frames = []
    diagnostics = []
    for instrument in required:
        frame, diagnostic = run_instrument(instrument, meta[instrument])
        frames.append(frame)
        diagnostics.append({"instrument": instrument, "bucket": instrument_bucket(instrument), **diagnostic})

    common_start = max(frame["combined_forecast"].first_valid_index() for frame in frames)
    common_end = min(frame["price"].last_valid_index() for frame in frames)
    frames = [frame.loc[common_start:common_end] for frame in frames]
    panel = pd.concat(frames).sort_index()
    panel.index.name = "DATETIME"
    panel.to_csv(OUT / "instrument_panel.csv")

    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df["first_forecast"] = diagnostics_df["first_forecast"].astype(str)
    diagnostics_df["last_price"] = diagnostics_df["last_price"].astype(str)
    diagnostics_df.to_csv(OUT / "fdm_and_universe.csv", index=False)

    flat = panel.reset_index().sort_values(["instrument", "DATETIME"])
    instrument_summary = (
        flat.groupby(["bucket", "instrument"])
        .agg(
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            costs=("costs", "sum"),
            total_traded_contracts=("trades", "sum"),
            avg_abs_position=("held_position", lambda x: x.abs().mean()),
            last_position=("rounded_position", "last"),
        )
        .reset_index()
        .sort_values(["bucket", "instrument"])
    )
    instrument_summary.to_csv(OUT / "instrument_summary.csv", index=False)
    bucket_summary = (
        instrument_summary.groupby("bucket")[["net_pnl", "gross_pnl", "costs"]]
        .sum()
        .reset_index()
        .sort_values("bucket")
    )
    bucket_summary.to_csv(OUT / "bucket_summary.csv", index=False)

    portfolio = pd.DataFrame(index=sorted(flat["DATETIME"].unique()))
    for field in ["gross_pnl", "costs", "net_pnl"]:
        portfolio[field] = flat.pivot_table(index="DATETIME", columns="instrument", values=field, aggfunc="sum").fillna(0.0).sum(axis=1)
    portfolio["daily_return"] = portfolio["net_pnl"] / CAPITAL
    portfolio["equity"] = CAPITAL + portfolio["net_pnl"].cumsum()
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1.0
    portfolio.to_csv(OUT / "portfolio_daily.csv")

    bucket_pnl = flat.pivot_table(index="DATETIME", columns="bucket", values="net_pnl", aggfunc="sum").fillna(0.0)
    bucket_pnl.to_csv(OUT / "bucket_pnl_daily.csv")

    stats = stats_from_portfolio(portfolio)
    write_summary(stats, diagnostics_df, bucket_summary, instrument_summary)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    portfolio["equity"].plot(ax=axes[0], title="40-Market Multi-Rule Portfolio Equity")
    axes[0].set_ylabel("USD")
    portfolio["drawdown"].plot(ax=axes[1], title="Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    bucket_pnl.cumsum().plot(ax=axes[2], title="Cumulative P&L by Bucket")
    axes[2].set_ylabel("USD")
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown_bucket.png", dpi=160)

    print(f"Wrote results to {OUT}")
    for key, value in stats.items():
        print(f"{key}: {format_stat(key, value)}")
    print(f"average_fdm: {diagnostics_df['fdm'].mean():.2f}")


if __name__ == "__main__":
    main()
