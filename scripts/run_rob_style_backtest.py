#!/usr/bin/env python3
"""Run a Rob Carver style multi-rule futures backtest from local CSV data.

This is still a compact educational implementation, but it uses the live-style
configuration in ``systems/provided/rob_system/config.yaml`` where local CSV
coverage permits:

- Rob instrument weights, forecast scalars, forecast weights, and FDM values.
- 40 rule variations: breakout, carry, relcarry, momentum, normmom,
  assettrend, relmomentum, cross-sectional mean reversion, skew, and accel.
- Volatility attenuation for the rules listed in the Rob config.
- Dynamic active-instrument weight normalisation as histories appear.
- A simple risk overlay and a portfolio-level integer/tracking-error buffer.
"""

from __future__ import annotations

import csv
import bisect
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PST = ROOT / "github" / "pysystemtrade"
FUTURES_DATA = PST / "data" / "futures"
ADJUSTED = FUTURES_DATA / "adjusted_prices_csv"
MULTIPLE = FUTURES_DATA / "multiple_prices_csv"
FX = FUTURES_DATA / "fx_prices_csv"
CONFIG_DIR = FUTURES_DATA / "csvconfig"
ROB_CONFIG = PST / "systems" / "provided" / "rob_system" / "config.yaml"
OUT = Path(os.environ.get("ROB_BACKTEST_OUT", ROOT / "backtests" / "rob_style_multirule"))

BUSINESS_DAYS = 256.0
AVERAGE_ABS_FORECAST = 10.0
FORECAST_CAP = 20.0
START_DATE = os.environ.get("ROB_BACKTEST_START_DATE", "2000-01-19")
BUFFER_SIZE = 0.10
TRACKING_ERROR_BUFFER = 0.0125


@dataclass(frozen=True)
class InstrumentMeta:
    point_size: float
    currency: str
    asset_class: str
    per_block: float
    percentage: float
    per_trade: float
    spread_cost: float


def load_rob_config() -> dict:
    return yaml.safe_load(ROB_CONFIG.read_text(encoding="utf-8"))


def load_meta() -> dict[str, InstrumentMeta]:
    instruments: dict[str, dict[str, str]] = {}
    with (CONFIG_DIR / "instrumentconfig.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            instruments[row["Instrument"]] = row

    spread_costs: dict[str, float] = {}
    with (CONFIG_DIR / "spreadcosts.csv").open(newline="") as handle:
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


def has_required_files(instrument: str, meta: dict[str, InstrumentMeta]) -> bool:
    if instrument not in meta:
        return False
    if not (ADJUSTED / f"{instrument}.csv").exists():
        return False
    if not (MULTIPLE / f"{instrument}.csv").exists():
        return False
    currency = meta[instrument].currency
    return currency == "USD" or (FX / f"{currency}USD.csv").exists()


def load_daily_adjusted_price(instrument: str) -> pd.Series:
    data = pd.read_csv(ADJUSTED / f"{instrument}.csv", parse_dates=["DATETIME"])
    price = data.set_index("DATETIME")["price"].sort_index()
    return price.resample("1B").last().ffill().rename(instrument)


def load_price_matrix(instruments: list[str]) -> pd.DataFrame:
    prices = [load_daily_adjusted_price(instrument) for instrument in instruments]
    price_df = pd.concat(prices, axis=1).sort_index()
    return price_df.loc[START_DATE:]


def align_prices_to_as_of(
    price: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    max_stale_business_days: int = 1,
) -> pd.DataFrame:
    """Align asynchronous market closes without allowing unlimited stale fills."""
    if price.empty:
        return price
    as_of = pd.Timestamp(as_of_date).normalize()
    start = pd.Timestamp(price.index.min()).normalize()
    calendar = pd.date_range(start, as_of, freq="B")
    aligned = price.reindex(price.index.union(calendar)).sort_index()
    aligned = aligned.ffill(limit=max_stale_business_days)
    return aligned.reindex(calendar)


def load_fx_rate(currency: str, index: pd.Index) -> pd.Series:
    if currency == "USD":
        return pd.Series(1.0, index=index)
    data = pd.read_csv(FX / f"{currency}USD.csv", parse_dates=["DATETIME"])
    fx = data.set_index("DATETIME")["PRICE"].sort_index().resample("1B").last().ffill()
    return fx.reindex(index, method="ffill")


def load_fx_matrix(instruments: list[str], meta: dict[str, InstrumentMeta], index: pd.Index) -> pd.DataFrame:
    by_currency: dict[str, pd.Series] = {}
    for instrument in instruments:
        currency = meta[instrument].currency
        if currency not in by_currency:
            by_currency[currency] = load_fx_rate(currency, index)
    return pd.DataFrame({instrument: by_currency[meta[instrument].currency] for instrument in instruments}, index=index)


def mixed_vol(daily_returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    fast = daily_returns.ewm(adjust=True, span=35, min_periods=10).std()
    slow = fast.ewm(span=int(20 * BUSINESS_DAYS), min_periods=10).mean()
    vol = 0.65 * fast + 0.35 * slow
    return vol.ffill().clip(lower=1.0e-10)


def ewmac(price: pd.DataFrame, vol: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    fast_ewma = price.ewm(span=fast, min_periods=1).mean()
    slow_ewma = price.ewm(span=slow, min_periods=1).mean()
    return (fast_ewma - slow_ewma) / vol.ffill()


def breakout(price: pd.DataFrame, lookback: int) -> pd.DataFrame:
    smooth = max(int(lookback / 4.0), 1)
    min_periods = int(min(len(price), math.ceil(lookback / 2.0)))
    roll_max = price.rolling(lookback, min_periods=min_periods).max()
    roll_min = price.rolling(lookback, min_periods=min_periods).min()
    roll_range = (roll_max - roll_min).replace(0.0, np.nan)
    raw = 40.0 * ((price - (roll_max + roll_min) / 2.0) / roll_range)
    return raw.ewm(span=smooth, min_periods=int(math.ceil(smooth / 2.0))).mean()


def contract_year_fraction(contract_series: pd.Series) -> pd.Series:
    contract = pd.to_numeric(contract_series, errors="coerce")
    years = np.floor(contract / 10000.0)
    months = (contract % 10000.0) / 100.0
    return years + months / 12.0


def raw_carry_for_instrument(instrument: str, price_vol: pd.Series, index: pd.Index) -> pd.Series:
    data = pd.read_csv(MULTIPLE / f"{instrument}.csv", parse_dates=["DATETIME"])
    multiple = data.set_index("DATETIME").sort_index().resample("1B").last().ffill()
    raw_roll = (multiple["PRICE"] - multiple["CARRY"]).replace(0.0, np.nan)
    differential = contract_year_fraction(multiple["CARRY_CONTRACT"]) - contract_year_fraction(
        multiple["PRICE_CONTRACT"]
    )
    floor = 1.0 / 365.0
    safe_sign = np.sign(differential).replace(0, 1)
    differential = differential.where(differential.abs() >= floor, safe_sign * floor)
    annualised_roll = raw_roll / differential
    ann_stdev = price_vol.reindex(annualised_roll.index, method="ffill") * math.sqrt(BUSINESS_DAYS)
    carry = annualised_roll / ann_stdev
    return carry.reindex(index, method="ffill")


def load_raw_carry_matrix(instruments: list[str], price_vol: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {instrument: raw_carry_for_instrument(instrument, price_vol[instrument], index) for instrument in instruments},
        index=index,
    )


def ewm_com(series_or_df: pd.DataFrame | pd.Series, com_days: int, min_periods: int | None = None):
    if min_periods is None:
        min_periods = max(2, int(com_days / 2))
    return series_or_df.ewm(com=com_days, min_periods=min_periods).mean()


def relative_momentum(
    normalised_price: pd.DataFrame, asset_class_price: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    ewma_span = max(int(horizon / 4.0), 2)
    outperformance = normalised_price.ffill() - asset_class_price.ffill()
    outperformance = outperformance.mask(outperformance == 0.0)
    average_outperformance = (outperformance - outperformance.shift(horizon)) / horizon
    return average_outperformance.ewm(span=ewma_span).mean()


def cross_sectional_mean_reversion(
    normalised_price: pd.DataFrame, asset_class_price: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    ewma_span = max(int(horizon / 4.0), 2)
    outperformance = normalised_price.ffill() - asset_class_price.ffill()
    relative_return = outperformance.diff()
    outperformance_over_horizon = relative_return.rolling(horizon).mean()
    return -outperformance_over_horizon.ewm(span=ewma_span).mean()


def factor_rule(demeaned_factor: pd.DataFrame, smooth: int) -> pd.DataFrame:
    vol = mixed_vol(demeaned_factor)
    normalised = demeaned_factor / vol
    return normalised.ewm(span=smooth).mean()


def causal_quantile_of_points(series: pd.Series) -> pd.Series:
    """Percentile of each value using only observations available before it."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    coordinates = sorted({float(value) for value in values if np.isfinite(value)})
    counts = [0] * (len(coordinates) + 1)

    def prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += counts[index]
            index -= index & -index
        return total

    def add(index: int) -> None:
        while index < len(counts):
            counts[index] += 1
            index += index & -index

    result = np.full(len(values), np.nan, dtype=float)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        coordinate = bisect.bisect_left(coordinates, float(value)) + 1
        result[index] = prefix_sum(coordinate - 1) / float(index + 1)
        add(coordinate)
    return pd.Series(result, index=series.index, name=series.name)


def vol_attenuation(price: pd.DataFrame, price_vol: pd.DataFrame) -> pd.DataFrame:
    daily_pct_vol = price_vol / price.abs().replace(0.0, np.nan)
    ten_year_vol = daily_pct_vol.rolling(2500, min_periods=10).mean()
    normalised_vol = daily_pct_vol / ten_year_vol
    vol_quantile = normalised_vol.apply(causal_quantile_of_points)
    attenuation = 2.0 - 1.5 * vol_quantile
    return attenuation.clip(lower=0.5, upper=2.0).ewm(span=10).mean().fillna(1.0)


def make_asset_class_matrix(
    source: pd.DataFrame,
    instruments: list[str],
    meta: dict[str, InstrumentMeta],
    reducer: str,
) -> pd.DataFrame:
    by_asset: dict[str, pd.Series] = {}
    for asset_class in sorted({meta[instrument].asset_class for instrument in instruments}):
        columns = [instrument for instrument in instruments if meta[instrument].asset_class == asset_class]
        if reducer == "median":
            by_asset[asset_class] = source[columns].median(axis=1)
        elif reducer == "mean":
            by_asset[asset_class] = source[columns].ffill().mean(axis=1)
        else:
            raise ValueError(reducer)
    return pd.DataFrame({instrument: by_asset[meta[instrument].asset_class] for instrument in instruments}, index=source.index)


def build_rule_forecasts(
    config: dict,
    instruments: list[str],
    meta: dict[str, InstrumentMeta],
    price: pd.DataFrame,
    price_vol: pd.DataFrame,
    raw_carry: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    returns = price.diff()
    norm_returns = returns / price_vol.shift(1)
    normalised_price = norm_returns.cumsum()
    normalised_vol = mixed_vol(normalised_price.diff())
    asset_class_returns = make_asset_class_matrix(norm_returns, instruments, meta, "median")
    asset_class_price = asset_class_returns.cumsum()
    asset_class_vol = mixed_vol(asset_class_price.diff())

    smoothed_carry_90 = ewm_com(raw_carry, 90)
    median_carry_by_asset = make_asset_class_matrix(smoothed_carry_90, instruments, meta, "median")

    pct_returns = price.pct_change()
    neg_skew_180 = -pct_returns.rolling(180, min_periods=90).skew()
    neg_skew_365 = -pct_returns.rolling(365, min_periods=180).skew()
    global_skew_avg_180 = neg_skew_180.ffill().mean(axis=1).ewm(span=int(15 * BUSINESS_DAYS), min_periods=50).mean()
    global_skew_avg_365 = neg_skew_365.ffill().mean(axis=1).ewm(span=int(15 * BUSINESS_DAYS), min_periods=50).mean()
    asset_skew_avg_180 = make_asset_class_matrix(neg_skew_180, instruments, meta, "mean")
    asset_skew_avg_365 = make_asset_class_matrix(neg_skew_365, instruments, meta, "mean")

    attenuation = vol_attenuation(price, price_vol)
    use_attenuation = set(config.get("use_attenuation", []))
    scalars = config["forecast_scalars"]
    forecasts: dict[str, pd.DataFrame] = {}

    def add_forecast(name: str, raw: pd.DataFrame) -> None:
        if name in use_attenuation:
            raw = raw * attenuation
        scalar = float(scalars.get(name, 1.0))
        forecasts[name] = (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)

    for lookback in [10, 20, 40, 80, 160, 320]:
        add_forecast(f"breakout{lookback}", breakout(price, lookback))

    for smooth_days in [10, 30, 60, 125]:
        add_forecast(f"carry{smooth_days}", ewm_com(raw_carry, smooth_days))

    add_forecast("relcarry", smoothed_carry_90 - median_carry_by_asset)

    for fast in [2, 4, 8, 16, 32, 64]:
        add_forecast(f"assettrend{fast}", ewmac(asset_class_price, asset_class_vol, fast, fast * 4))
        add_forecast(f"normmom{fast}", ewmac(normalised_price, normalised_vol, fast, fast * 4))

    for fast in [4, 8, 16, 32, 64]:
        add_forecast(f"momentum{fast}", ewmac(price, price_vol, fast, fast * 4))

    for horizon in [10, 20, 40, 80]:
        add_forecast(f"relmomentum{horizon}", relative_momentum(normalised_price, asset_class_price, horizon))

    add_forecast("mrinasset1000", cross_sectional_mean_reversion(normalised_price, asset_class_price, 1000))

    add_forecast("skewabs180", factor_rule(neg_skew_180.sub(global_skew_avg_180, axis=0), 45))
    add_forecast("skewabs365", factor_rule(neg_skew_365.sub(global_skew_avg_365, axis=0), 90))
    add_forecast("skewrv180", factor_rule(neg_skew_180 - asset_skew_avg_180, 45))
    add_forecast("skewrv365", factor_rule(neg_skew_365 - asset_skew_avg_365, 90))

    for fast in [16, 32, 64]:
        ewmac_signal = ewmac(price, price_vol, fast, fast * 4)
        add_forecast(f"accel{fast}", ewmac_signal - ewmac_signal.shift(fast))

    return forecasts


def combine_forecasts(
    forecasts: dict[str, pd.DataFrame],
    config: dict,
    instruments: list[str],
    index: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.DataFrame(index=index, columns=instruments, dtype=float)
    rule_weight_used = pd.DataFrame(0.0, index=index, columns=instruments)
    forecast_weights = config["forecast_weights"]
    fdm_config = config["forecast_div_multiplier"]

    for instrument in instruments:
        weights = {
            rule: float(weight)
            for rule, weight in forecast_weights.get(instrument, {}).items()
            if rule in forecasts and float(weight) != 0.0
        }
        if not weights:
            continue
        weighted_sum = pd.Series(0.0, index=index)
        available_weight = pd.Series(0.0, index=index)
        for rule, weight in weights.items():
            forecast = forecasts[rule][instrument]
            valid = forecast.notna()
            weighted_sum = weighted_sum.add(forecast.fillna(0.0) * weight, fill_value=0.0)
            available_weight = available_weight.add(valid.astype(float) * weight, fill_value=0.0)
        raw_combined = weighted_sum / available_weight.replace(0.0, np.nan)
        fdm = float(fdm_config.get(instrument, 1.0))
        combined[instrument] = (raw_combined * fdm).clip(-FORECAST_CAP, FORECAST_CAP)
        rule_weight_used[instrument] = available_weight

    return combined, rule_weight_used


def cost_per_contract_usd(meta: InstrumentMeta, fx: pd.Series, price: pd.Series) -> pd.Series:
    local_fixed = meta.per_block + meta.per_trade + meta.spread_cost * meta.point_size
    local_percentage = (meta.percentage / 100.0) * price.abs() * meta.point_size
    return (local_fixed + local_percentage).reindex(fx.index, method="ffill").fillna(local_fixed) * fx


def cost_matrix(
    instruments: list[str],
    meta: dict[str, InstrumentMeta],
    price: pd.DataFrame,
    fx: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {instrument: cost_per_contract_usd(meta[instrument], fx[instrument], price[instrument]) for instrument in instruments},
        index=price.index,
    )


def base_instrument_weights(config: dict, instruments: list[str]) -> pd.Series:
    raw = pd.Series({instrument: float(config["instrument_weights"][instrument]) for instrument in instruments})
    return raw / raw.sum()


def active_weight_matrix(base_weights: pd.Series, valid: pd.DataFrame) -> pd.DataFrame:
    weighted = valid.astype(float).mul(base_weights, axis=1)
    row_sum = weighted.sum(axis=1).replace(0.0, np.nan)
    return weighted.div(row_sum, axis=0)


def initial_target_positions(
    config: dict,
    instruments: list[str],
    meta: dict[str, InstrumentMeta],
    price: pd.DataFrame,
    price_vol: pd.DataFrame,
    fx: pd.DataFrame,
    combined_forecast: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    capital = float(config["notional_trading_capital"])
    vol_target = float(config["percentage_vol_target"]) / 100.0
    idm = float(config.get("instrument_div_multiplier", 1.0))
    daily_cash_vol_target = capital * vol_target / math.sqrt(BUSINESS_DAYS)

    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    unit_daily_cash_vol = price_vol.abs().mul(point_sizes, axis=1) * fx
    valid = (
        price.notna()
        & fx.notna()
        & combined_forecast.notna()
        & unit_daily_cash_vol.notna()
        & (unit_daily_cash_vol > 0.0)
    )
    daily_weights = active_weight_matrix(base_instrument_weights(config, instruments), valid)
    subsystem_position = daily_cash_vol_target / unit_daily_cash_vol
    target = subsystem_position * (combined_forecast / AVERAGE_ABS_FORECAST) * daily_weights * idm
    return target.replace([np.inf, -np.inf], np.nan), daily_weights, unit_daily_cash_vol


def risk_multiplier_for_targets(
    config: dict,
    target: pd.DataFrame,
    unit_daily_cash_vol: pd.DataFrame,
    price: pd.DataFrame,
    fx: pd.DataFrame,
    meta: dict[str, InstrumentMeta],
    instruments: list[str],
) -> pd.Series:
    capital = float(config["notional_trading_capital"])
    vol_target = float(config["percentage_vol_target"]) / 100.0
    risk_overlay = config.get("risk_overlay", {})
    max_normal = float(risk_overlay.get("max_risk_fraction_normal_risk", 1.75))
    max_sum_abs = float(risk_overlay.get("max_risk_limit_sum_abs_risk", 4.0))
    max_leverage = float(risk_overlay.get("max_risk_leverage", 20.0))

    unit_ann_risk = unit_daily_cash_vol * math.sqrt(BUSINESS_DAYS) / capital
    normal_risk = ((target.fillna(0.0) * unit_ann_risk.fillna(0.0)) ** 2).sum(axis=1).pow(0.5)
    sum_abs_risk = (target.fillna(0.0).abs() * unit_ann_risk.fillna(0.0)).sum(axis=1)

    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    notional = (target.fillna(0.0).abs() * price.abs().mul(point_sizes, axis=1) * fx).sum(axis=1)
    leverage = notional / capital

    multiplier = pd.Series(1.0, index=target.index)
    multiplier = np.minimum(multiplier, (vol_target * max_normal / normal_risk.replace(0.0, np.nan)).fillna(1.0))
    multiplier = np.minimum(multiplier, (vol_target * max_sum_abs / sum_abs_risk.replace(0.0, np.nan)).fillna(1.0))
    multiplier = np.minimum(multiplier, (max_leverage / leverage.replace(0.0, np.nan)).fillna(1.0))
    return multiplier.clip(lower=0.0, upper=1.0).rename("risk_multiplier")


def apply_position_buffer(target: pd.DataFrame, buffer: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(0.0, index=target.index, columns=target.columns)
    prior = pd.Series(0.0, index=target.columns)
    for date in target.index:
        desired = target.loc[date].fillna(0.0)
        edge = buffer.loc[date].fillna(0.0)
        new_position = prior.copy()
        too_high = desired > prior + edge
        too_low = desired < prior - edge
        new_position[too_high] = desired[too_high] - edge[too_high]
        new_position[too_low] = desired[too_low] + edge[too_low]
        output.loc[date] = new_position
        prior = new_position
    return output


def optimise_integer_positions(
    target: pd.DataFrame,
    unit_daily_cash_vol: pd.DataFrame,
    config: dict,
    max_steps_per_day: int = 100,
) -> pd.DataFrame:
    capital = float(config["notional_trading_capital"])
    unit_ann_risk = (unit_daily_cash_vol * math.sqrt(BUSINESS_DAYS) / capital).fillna(0.0)
    output = pd.DataFrame(0.0, index=target.index, columns=target.columns)
    prior = pd.Series(0.0, index=target.columns)

    for date in target.index:
        desired = target.loc[date].fillna(0.0)
        risk = unit_ann_risk.loc[date].fillna(0.0)
        current = prior.copy()

        def tracking_error(position: pd.Series) -> float:
            return float((((position - desired) * risk) ** 2).sum() ** 0.5)

        current_te = tracking_error(current)
        if current_te > TRACKING_ERROR_BUFFER:
            ideal = desired.round()
            for _ in range(max_steps_per_day):
                diff = ideal - current
                candidates = diff[diff != 0.0]
                if candidates.empty:
                    break
                step = np.sign(candidates)
                current_error = ((current[candidates.index] - desired[candidates.index]) * risk[candidates.index]) ** 2
                next_error = ((current[candidates.index] + step - desired[candidates.index]) * risk[candidates.index]) ** 2
                reduction = current_error - next_error
                reduction = reduction[reduction > 0.0]
                if reduction.empty:
                    break
                best = reduction.idxmax()
                current[best] += float(np.sign(diff[best]))
                current_te = tracking_error(current)
                if current_te <= TRACKING_ERROR_BUFFER:
                    break

        output.loc[date] = current
        prior = current
    return output


def pnl_from_positions(
    positions: pd.DataFrame,
    price: pd.DataFrame,
    fx: pd.DataFrame,
    meta: dict[str, InstrumentMeta],
    instruments: list[str],
    costs_per_contract: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    held = positions.shift(1).fillna(0.0)
    price_change = price.diff().fillna(0.0)
    gross = held * price_change.mul(point_sizes, axis=1) * fx
    trades = positions.diff().abs().fillna(0.0)
    costs = trades * costs_per_contract
    net = gross - costs
    daily = pd.DataFrame(
        {
            "gross_pnl": gross.sum(axis=1),
            "costs": costs.sum(axis=1),
            "net_pnl": net.sum(axis=1),
        },
        index=price.index,
    )
    return daily, pd.concat({"gross_pnl": gross, "costs": costs, "net_pnl": net, "position": positions, "trades": trades}, axis=1)


def portfolio_stats(daily: pd.DataFrame, capital: float) -> dict[str, float | str]:
    daily = daily.copy()
    daily["daily_return"] = daily["net_pnl"] / capital
    daily["equity"] = capital + daily["net_pnl"].cumsum()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    returns = daily["daily_return"].dropna()
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    sharpe = ann_return / ann_vol if ann_vol else np.nan
    return {
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "years": len(daily) / BUSINESS_DAYS,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": daily["drawdown"].min(),
        "total_return": daily["equity"].iloc[-1] / capital - 1.0,
        "total_costs": daily["costs"].sum(),
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


def instrument_summary(
    net_by_instrument: pd.DataFrame,
    instruments: list[str],
    meta: dict[str, InstrumentMeta],
    base_weights: pd.Series,
    combined_forecast: pd.DataFrame,
    rule_weight_used: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    rows = []
    for instrument in instruments:
        position = net_by_instrument["position"][instrument]
        trades = net_by_instrument["trades"][instrument]
        rows.append(
            {
                "instrument": instrument,
                "asset_class": meta[instrument].asset_class,
                "base_weight": base_weights[instrument],
                "fdm": float(config["forecast_div_multiplier"].get(instrument, 1.0)),
                "first_forecast": combined_forecast[instrument].first_valid_index(),
                "last_price": combined_forecast.index.max(),
                "nonzero_config_rules": int(
                    sum(float(weight) != 0.0 for weight in config["forecast_weights"].get(instrument, {}).values())
                ),
                "avg_available_rule_weight": float(rule_weight_used[instrument].mean()),
                "net_pnl": net_by_instrument["net_pnl"][instrument].sum(),
                "gross_pnl": net_by_instrument["gross_pnl"][instrument].sum(),
                "costs": net_by_instrument["costs"][instrument].sum(),
                "total_traded_contracts": trades.sum(),
                "avg_abs_position": position.abs().mean(),
                "last_position": position.iloc[-1],
            }
        )
    summary = pd.DataFrame(rows)
    summary["first_forecast"] = summary["first_forecast"].astype(str)
    return summary.sort_values(["asset_class", "instrument"])


def write_summary(
    config: dict,
    instruments: list[str],
    missing: list[str],
    continuous_stats: dict[str, float | str],
    integer_stats: dict[str, float | str],
    asset_summary: pd.DataFrame,
    instr_summary: pd.DataFrame,
    risk_multiplier: pd.Series,
) -> None:
    lines = [
        "# Rob-Style Multi-Rule Futures Backtest",
        "",
        "## What Changed Versus The 40-Market Teaching Run",
        "",
        "- Uses Rob's `rob_system/config.yaml` instrument weights where local CSV data exists.",
        "- Uses 40 configured rule variations rather than only 9 EWMAC/breakout/carry rules.",
        "- Uses Rob's fixed forecast scalars, forecast weights, and forecast diversification multipliers.",
        "- Adds volatility attenuation for Rob's configured trend-like rules.",
        "- Adds dynamic active-instrument weight normalisation when histories are missing.",
        "- Adds an approximate risk overlay and a portfolio tracking-error integer optimiser.",
        "- Keeps local CSV data only; this is not connected to Rob's production database.",
        "",
        "## Data Source",
        "",
        f"- Adjusted prices: `{ADJUSTED}`",
        f"- Multiple/carry prices: `{MULTIPLE}`",
        f"- FX: `{FX}`",
        f"- Rob config: `{ROB_CONFIG}`",
        f"- Config instruments: {len(config['instrument_weights'])}",
        f"- Local usable Rob-config instruments: {len(instruments)}",
        f"- Missing local instruments: {', '.join(missing) if missing else 'none'}",
        "",
        "## Portfolio Settings",
        "",
        f"- Capital: ${float(config['notional_trading_capital']):,.0f}",
        f"- Annual volatility target: {float(config['percentage_vol_target']):.1f}%",
        f"- Instrument diversification multiplier: {float(config.get('instrument_div_multiplier', 1.0)):.2f}",
        f"- Forecast cap/floor: +/-{FORECAST_CAP:.0f}",
        f"- Tracking error buffer: {TRACKING_ERROR_BUFFER:.2%}",
        f"- Average risk multiplier: {risk_multiplier.mean():.2f}",
        f"- Minimum risk multiplier: {risk_multiplier.min():.2f}",
        "",
        "## Continuous Position Results",
        "",
    ]
    for key, value in continuous_stats.items():
        lines.append(f"- {key}: {format_stat(key, value)}")
    lines.extend(["", "## Buffered Integer Results", ""])
    for key, value in integer_stats.items():
        lines.append(f"- {key}: {format_stat(key, value)}")
    lines.extend(["", "## Asset Class P&L, Buffered Integer", ""])
    for row in asset_summary.itertuples(index=False):
        lines.append(
            f"- {row.asset_class}: net ${row.net_pnl:,.0f}; gross ${row.gross_pnl:,.0f}; costs ${row.costs:,.0f}"
        )
    lines.extend(["", "## Highest Cost Instruments", ""])
    for row in instr_summary.sort_values("costs", ascending=False).head(10).itertuples(index=False):
        lines.append(
            f"- {row.instrument} ({row.asset_class}): costs ${row.costs:,.0f}; net ${row.net_pnl:,.0f}; trades {row.total_traded_contracts:,.0f}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The volatility attenuation quantile is approximated with full-sample ranks for speed, so this file remains a research scaffold.",
            "- The integer optimiser uses a diagonal tracking-error approximation, not the full pysystemtrade covariance optimiser.",
            "- The local CSV dataset stops around 2024-03-28 and is not a live Polygon/Barchart/IB data pipeline.",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(
    continuous_daily: pd.DataFrame,
    integer_daily: pd.DataFrame,
    asset_daily: pd.DataFrame,
    capital: float,
) -> None:
    continuous_equity = capital + continuous_daily["net_pnl"].cumsum()
    integer_equity = capital + integer_daily["net_pnl"].cumsum()
    integer_drawdown = integer_equity / integer_equity.cummax() - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    continuous_equity.plot(ax=axes[0], label="Continuous")
    integer_equity.plot(ax=axes[0], label="Buffered integer")
    axes[0].set_title("Rob-Style Portfolio Equity")
    axes[0].set_ylabel("USD")
    axes[0].legend()

    integer_drawdown.plot(ax=axes[1], title="Buffered Integer Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

    asset_daily.cumsum().plot(ax=axes[2], title="Buffered Integer Cumulative P&L by Asset Class")
    axes[2].set_ylabel("USD")
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown_assetclass.png", dpi=160)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_rob_config()
    meta = load_meta()
    all_config_instruments = list(config["instrument_weights"].keys())
    instruments = [instrument for instrument in all_config_instruments if has_required_files(instrument, meta)]
    missing = [instrument for instrument in all_config_instruments if instrument not in instruments]

    price = load_price_matrix(instruments)
    price_vol = mixed_vol(price.diff())
    fx = load_fx_matrix(instruments, meta, price.index)
    raw_carry = load_raw_carry_matrix(instruments, price_vol, price.index)
    forecasts = build_rule_forecasts(config, instruments, meta, price, price_vol, raw_carry)
    combined_forecast, rule_weight_used = combine_forecasts(forecasts, config, instruments, price.index)
    target, daily_weights, unit_daily_cash_vol = initial_target_positions(
        config, instruments, meta, price, price_vol, fx, combined_forecast
    )
    risk_multiplier = risk_multiplier_for_targets(config, target, unit_daily_cash_vol, price, fx, meta, instruments)
    target = target.mul(risk_multiplier, axis=0)

    costs = cost_matrix(instruments, meta, price, fx)
    continuous_daily, continuous_by_instrument = pnl_from_positions(
        target.fillna(0.0), price, fx, meta, instruments, costs
    )

    idm = float(config.get("instrument_div_multiplier", 1.0))
    subsystem_position = (
        float(config["notional_trading_capital"])
        * (float(config["percentage_vol_target"]) / 100.0)
        / math.sqrt(BUSINESS_DAYS)
        / unit_daily_cash_vol
    )
    buffer = (subsystem_position.abs() * daily_weights * idm * BUFFER_SIZE).mul(risk_multiplier, axis=0)
    buffered_target = apply_position_buffer(target.fillna(0.0), buffer.fillna(0.0))
    integer_positions = optimise_integer_positions(buffered_target, unit_daily_cash_vol, config)
    integer_daily, integer_by_instrument = pnl_from_positions(integer_positions, price, fx, meta, instruments, costs)

    capital = float(config["notional_trading_capital"])
    continuous_stats = portfolio_stats(continuous_daily, capital)
    integer_stats = portfolio_stats(integer_daily, capital)

    base_weights = base_instrument_weights(config, instruments)
    instr_summary = instrument_summary(
        integer_by_instrument,
        instruments,
        meta,
        base_weights,
        combined_forecast,
        rule_weight_used,
        config,
    )
    asset_summary = (
        instr_summary.groupby("asset_class")[["net_pnl", "gross_pnl", "costs"]]
        .sum()
        .reset_index()
        .sort_values("asset_class")
    )
    asset_daily = pd.DataFrame(index=price.index)
    for asset_class in asset_summary["asset_class"]:
        columns = [instrument for instrument in instruments if meta[instrument].asset_class == asset_class]
        asset_daily[asset_class] = integer_by_instrument["net_pnl"][columns].sum(axis=1)

    continuous_out = continuous_daily.copy()
    continuous_out["daily_return"] = continuous_out["net_pnl"] / capital
    continuous_out["equity"] = capital + continuous_out["net_pnl"].cumsum()
    continuous_out["drawdown"] = continuous_out["equity"] / continuous_out["equity"].cummax() - 1.0
    integer_out = integer_daily.copy()
    integer_out["daily_return"] = integer_out["net_pnl"] / capital
    integer_out["equity"] = capital + integer_out["net_pnl"].cumsum()
    integer_out["drawdown"] = integer_out["equity"] / integer_out["equity"].cummax() - 1.0

    portfolio = pd.concat({"continuous": continuous_out, "buffered_integer": integer_out}, axis=1)
    portfolio.to_csv(OUT / "portfolio_daily.csv")
    instr_summary.to_csv(OUT / "instrument_summary.csv", index=False)
    asset_summary.to_csv(OUT / "asset_class_summary.csv", index=False)
    daily_weights.to_csv(OUT / "daily_instrument_weights.csv")
    risk_multiplier.to_csv(OUT / "risk_multiplier.csv")
    asset_daily.to_csv(OUT / "asset_class_daily_pnl.csv")

    rule_rows = []
    for rule_name in sorted(forecasts):
        family = re.sub(r"\d+$", "", rule_name)
        rule_rows.append(
            {
                "rule": rule_name,
                "family": family,
                "forecast_scalar": float(config["forecast_scalars"].get(rule_name, 1.0)),
                "attenuated": rule_name in set(config.get("use_attenuation", [])),
                "non_na_forecast_days": int(forecasts[rule_name].notna().sum().sum()),
            }
        )
    pd.DataFrame(rule_rows).to_csv(OUT / "rule_coverage.csv", index=False)

    universe = instr_summary[
        [
            "instrument",
            "asset_class",
            "base_weight",
            "fdm",
            "first_forecast",
            "nonzero_config_rules",
            "avg_available_rule_weight",
        ]
    ].copy()
    universe.to_csv(OUT / "universe.csv", index=False)

    write_summary(
        config,
        instruments,
        missing,
        continuous_stats,
        integer_stats,
        asset_summary,
        instr_summary,
        risk_multiplier,
    )
    plot_results(continuous_daily, integer_daily, asset_daily, capital)

    print(f"Wrote results to {OUT}")
    print(f"usable instruments: {len(instruments)} / {len(all_config_instruments)}")
    print("continuous:")
    for key, value in continuous_stats.items():
        print(f"  {key}: {format_stat(key, value)}")
    print("buffered_integer:")
    for key, value in integer_stats.items():
        print(f"  {key}: {format_stat(key, value)}")
    print(f"average_risk_multiplier: {risk_multiplier.mean():.2f}")


if __name__ == "__main__":
    main()
