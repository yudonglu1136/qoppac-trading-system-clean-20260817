#!/usr/bin/env python3
"""Rob-style stock trend backtests with no carry or roll signals.

This is the equity version of the compact Rob-style futures script.  It keeps
the portfolio construction machinery: Rob forecast scalars and weights, FDM,
volatility sizing, instrument weights, IDM, risk overlay, buffers, integer
positions, and costs.  The only deliberately removed rules are carry/roll rules
which do not exist for ordinary stock price data.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_benchmark_aware_stock_momentum as baw  # noqa: E402
import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402
import run_rob_style_backtest as rob  # noqa: E402


OUT = ROOT / "backtests" / "rob_style_stock"
BUSINESS_DAYS = 252.0
AVERAGE_ABS_FORECAST = 10.0
FORECAST_CAP = 20.0
BUFFER_SIZE = 0.10
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-08-07"

DEFAULT_CAPITAL = 500_000.0
DEFAULT_VOL_TARGET = 0.25
DEFAULT_COST_PER_DOLLAR = 0.0005
MAX_ABS_DAILY_RETURN = 0.50
MIN_TRADEABLE_PRICE = 0.25
MAX_ROLLING_MEDIAN_RATIO = 20.0

SUPPORTED_UNIVERSES = {
    "sp500": "SPY / S&P 500",
    "eem": "Emerging Markets / EEM",
    "efa": "Developed Markets ex-US / EFA",
}

WEIGHT_MODES = {
    "equal": "Equal active stock risk weights",
    "benchmark": "Point-in-time ETF/index weights as instrument risk weights",
    "sector_equal": "Equal sector risk weights, equal stocks within each sector",
}


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def performance_stats_from_equity(daily_return: pd.Series, equity_ratio: pd.Series) -> dict[str, float | str]:
    returns = daily_return.dropna()
    equity_ratio = equity_ratio.reindex(returns.index).dropna()
    if returns.empty or equity_ratio.empty:
        return {}
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    final_equity = equity_ratio.iloc[-1]
    cagr = final_equity ** (1.0 / years) - 1.0 if years > 0 and final_equity > 0.0 else np.nan
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": years,
        "total_return": final_equity - 1.0,
        "cagr": cagr,
        "ann_return": ann_return,
        "vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else np.nan,
        "mdd": (equity_ratio / equity_ratio.cummax() - 1.0).min(),
    }


def load_sp500_annual_with_legacy(start: str, end: str) -> pd.DataFrame:
    """Use annual S&P membership before cached SPY holding snapshots begin."""
    legacy_path = pit.DATA_ROOT / "sp500" / "annual_constituents.csv"
    legacy = pd.read_csv(legacy_path)
    legacy["snapshot_date"] = pd.to_datetime(legacy["snapshot_date"], errors="coerce")
    legacy = legacy[
        (legacy["snapshot_date"].dt.year >= pd.Timestamp(start).year)
        & (legacy["snapshot_date"].le(pd.Timestamp(end)))
    ].copy()
    if "weight" not in legacy.columns:
        legacy["weight"] = np.nan
    if "sector" not in legacy.columns:
        legacy["sector"] = "Equity"

    holdings = baw.load_spy_holdings(start, end)
    holdings["snapshot_date"] = pd.to_datetime(holdings["snapshot_date"], errors="coerce")
    if holdings.empty:
        return legacy

    first_holding_snapshot = holdings["snapshot_date"].min()
    legacy = legacy[legacy["snapshot_date"].lt(first_holding_snapshot)]
    return pd.concat([legacy, holdings], ignore_index=True, sort=False)


def load_etf_holding_snapshots(key: str, end: str) -> pd.DataFrame:
    """Load every archived ETF holding snapshot available by date.

    The older annual loader selected one snapshot per calendar year.  For
    cross-sectional research we need the active universe to be driven by the
    actual point-in-time snapshots available in the local cache.
    """
    path = pit.DATA_ROOT / key / "holding_snapshots.csv"
    if not path.exists():
        return baw.load_annual(key, DEFAULT_START, end).copy()
    snapshots = pd.read_csv(path)
    snapshots["holding_asof"] = pd.to_datetime(snapshots.get("holding_asof"), errors="coerce")
    snapshots["archive_date"] = pd.to_datetime(snapshots.get("archive_date"), errors="coerce")
    snapshots["snapshot_date"] = snapshots["archive_date"].fillna(snapshots["holding_asof"])
    snapshots = snapshots.dropna(subset=["snapshot_date", "symbol"])
    snapshots = snapshots[snapshots["snapshot_date"].le(pd.Timestamp(end))].copy()
    snapshots["year"] = snapshots["snapshot_date"].dt.year
    snapshots["revision_timestamp"] = snapshots.get("archive_timestamp", "")
    snapshots["revision_id"] = snapshots.get("archive_timestamp", "")
    snapshots["source_asof"] = snapshots["holding_asof"]
    if "weight" in snapshots.columns:
        snapshots["weight"] = pd.to_numeric(snapshots["weight"], errors="coerce")
    return snapshots.sort_values(["snapshot_date", "symbol"])


def load_annual(key: str, start: str, end: str) -> pd.DataFrame:
    if key == "sp500":
        annual = load_sp500_annual_with_legacy(start, end)
    elif key in {"eem", "efa"}:
        annual = load_etf_holding_snapshots(key, end)
    else:
        annual = baw.load_annual(key, start, end).copy()
    annual["symbol"] = annual["symbol"].astype(str)
    annual["snapshot_date"] = pd.to_datetime(annual["snapshot_date"], errors="coerce")
    annual = annual.dropna(subset=["snapshot_date", "symbol"])
    if "asset_class" in annual.columns:
        annual = annual[annual["asset_class"].astype(str).str.contains("Equity", case=False, na=True)]
    if "sector" not in annual.columns:
        annual["sector"] = "Equity"
    annual["sector"] = annual["sector"].fillna("Unknown").astype(str)
    if "weight" in annual.columns:
        annual["weight"] = pd.to_numeric(annual["weight"], errors="coerce")
    return annual.sort_values(["snapshot_date", "symbol"])


def load_price(key: str, annual: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    data_dir = pit.DATA_ROOT / key
    symbols = pd.Index(sorted(annual["symbol"].dropna().astype(str).unique()))
    price = pd.read_csv(data_dir / "adj_close.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    price = price.loc[:, ~price.columns.duplicated()]
    price = price.reindex(columns=symbols)
    if key in {"eem", "efa"}:
        price = pit.convert_em_prices_to_usd(price, annual, data_dir, False).reindex(columns=symbols)
    price = price.loc[:end]
    rolling_median = price.shift(1).rolling(252, min_periods=60).median()
    price_ratio = price / rolling_median.replace(0.0, np.nan)
    raw_returns = price.pct_change(fill_method=None)
    bad_data = (
        (price <= 0.0)
        | (price < MIN_TRADEABLE_PRICE)
        | (price_ratio > MAX_ROLLING_MEDIAN_RATIO)
        | (price_ratio < 1.0 / MAX_ROLLING_MEDIAN_RATIO)
        | (raw_returns.abs() > MAX_ABS_DAILY_RETURN)
    )
    price = price.mask(bad_data).ffill(limit=5).mask(bad_data)
    for _ in range(3):
        clean_returns = price.pct_change(fill_method=None)
        clean_bad = clean_returns.abs() > MAX_ABS_DAILY_RETURN
        if not bool(clean_bad.any().any()):
            break
        price = price.mask(clean_bad)
    first_needed = pd.Timestamp(start) - pd.offsets.BDay(3000)
    return price.loc[first_needed:end]


def rob_price_rule_weights(config: dict) -> pd.Series:
    forecast_weights = config["forecast_weights"]
    instruments = list(forecast_weights)
    rules = sorted({rule for weights in forecast_weights.values() for rule in weights})
    averaged = {}
    for rule in rules:
        lower = rule.lower()
        if "carry" in lower or "roll" in lower:
            continue
        value = sum(float(forecast_weights[instrument].get(rule, 0.0)) for instrument in instruments) / len(instruments)
        if value > 0.0:
            averaged[rule] = value
    weights = pd.Series(averaged, dtype=float)
    return weights / weights.sum()


def median_rob_fdm(config: dict) -> float:
    values = pd.Series(config["forecast_div_multiplier"], dtype=float)
    return float(values.median())


def snapshot_base_weights(annual: pd.DataFrame, columns: pd.Index, mode: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    dates: list[pd.Timestamp] = []
    for snapshot_date, frame in annual.groupby("snapshot_date"):
        frame = frame.drop_duplicates("symbol").set_index("symbol")
        frame = frame[frame.index.isin(columns)]
        if frame.empty:
            continue

        if mode == "benchmark" and "weight" in frame.columns and frame["weight"].notna().any():
            raw = pd.to_numeric(frame["weight"], errors="coerce").dropna()
            raw = raw[raw > 0.0]
            if raw.empty:
                raw = pd.Series(1.0, index=frame.index)
        elif mode == "sector_equal":
            sector = frame["sector"].fillna("Unknown").astype(str)
            sectors = sorted(sector.unique())
            pieces = []
            for sector_name in sectors:
                members = sector[sector.eq(sector_name)].index
                pieces.append(pd.Series(1.0 / len(sectors) / len(members), index=members))
            raw = pd.concat(pieces)
        else:
            raw = pd.Series(1.0, index=frame.index)

        weights = raw / raw.sum()
        row = pd.Series(0.0, index=columns, dtype=float)
        row.loc[weights.index] = weights
        rows.append(row)
        dates.append(pd.Timestamp(snapshot_date))

    if not rows:
        return pd.DataFrame(columns=columns, dtype=float)
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(dates)).sort_index()
    return out[~out.index.duplicated(keep="last")]


def daily_base_weights(annual: pd.DataFrame, columns: pd.Index, index: pd.DatetimeIndex, mode: str) -> pd.DataFrame:
    snapshots = snapshot_base_weights(annual, columns, mode)
    if snapshots.empty:
        return pd.DataFrame(0.0, index=index, columns=columns)
    return snapshots.reindex(index, method="ffill").fillna(0.0)


def dynamic_group_matrix(source: pd.DataFrame, annual: pd.DataFrame, reducer: str) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=source.index, columns=source.columns, dtype=float)
    snapshots = sorted(pd.to_datetime(annual["snapshot_date"].dropna().unique()))
    if not snapshots:
        return out

    for i, snapshot_date in enumerate(snapshots):
        next_snapshot = snapshots[i + 1] if i + 1 < len(snapshots) else pd.Timestamp.max
        mask = (source.index >= snapshot_date) & (source.index < next_snapshot)
        if not mask.any():
            continue
        period_index = source.index[mask]
        frame = annual[annual["snapshot_date"].eq(snapshot_date)].drop_duplicates("symbol").set_index("symbol")
        frame = frame[frame.index.isin(source.columns)]
        if frame.empty:
            continue
        sectors = frame["sector"].fillna("Unknown").astype(str)
        for _sector_name, members_index in sectors.groupby(sectors).groups.items():
            members = [symbol for symbol in members_index if symbol in source.columns]
            if not members:
                continue
            if reducer == "median":
                group_series = source.loc[period_index, members].median(axis=1)
            elif reducer == "mean":
                group_series = source.loc[period_index, members].mean(axis=1)
            else:  # pragma: no cover
                raise ValueError(reducer)
            out.loc[period_index, members] = np.repeat(
                group_series.to_numpy()[:, None],
                len(members),
                axis=1,
            )
    return out


def add_forecast_to_combiner(
    name: str,
    raw: pd.DataFrame,
    *,
    rule_weights: pd.Series,
    scalars: dict,
    use_attenuation: set[str],
    attenuation: pd.DataFrame,
    weighted_sum: pd.DataFrame,
    available_weight: pd.DataFrame,
    rows: list[dict[str, float | str]],
) -> None:
    weight = float(rule_weights.get(name, 0.0))
    if weight == 0.0:
        return
    if name in use_attenuation:
        raw = raw * attenuation
    scalar = float(scalars.get(name, 1.0))
    forecast = (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
    valid = forecast.notna()
    weighted_sum += forecast.fillna(0.0) * weight
    available_weight += valid.astype(float) * weight
    rows.append({"rule": name, "forecast_weight": weight, "scalar": scalar})


def build_rob_style_stock_forecast(
    price: pd.DataFrame,
    annual: pd.DataFrame,
    config: dict,
    fdm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rule_weights = rob_price_rule_weights(config)
    returns = price.diff()
    price_vol = rob.mixed_vol(returns)
    norm_returns = returns / price_vol.shift(1)
    normalised_price = norm_returns.cumsum()
    normalised_vol = rob.mixed_vol(normalised_price.diff())

    asset_class_returns = dynamic_group_matrix(norm_returns, annual, "median")
    asset_class_price = asset_class_returns.cumsum()
    asset_class_vol = rob.mixed_vol(asset_class_price.diff())

    pct_returns = price.pct_change(fill_method=None).mask(lambda frame: frame.abs() > MAX_ABS_DAILY_RETURN)
    neg_skew_180 = -pct_returns.rolling(180, min_periods=90).skew()
    neg_skew_365 = -pct_returns.rolling(365, min_periods=180).skew()
    global_skew_avg_180 = neg_skew_180.ffill().mean(axis=1).ewm(span=int(15 * BUSINESS_DAYS), min_periods=50).mean()
    global_skew_avg_365 = neg_skew_365.ffill().mean(axis=1).ewm(span=int(15 * BUSINESS_DAYS), min_periods=50).mean()
    asset_skew_avg_180 = dynamic_group_matrix(neg_skew_180, annual, "mean")
    asset_skew_avg_365 = dynamic_group_matrix(neg_skew_365, annual, "mean")

    attenuation = rob.vol_attenuation(price, price_vol)
    use_attenuation = set(config.get("use_attenuation", []))
    scalars = config["forecast_scalars"]
    weighted_sum = pd.DataFrame(0.0, index=price.index, columns=price.columns)
    available_weight = pd.DataFrame(0.0, index=price.index, columns=price.columns)
    rows: list[dict[str, float | str]] = []

    def add(name: str, raw: pd.DataFrame) -> None:
        add_forecast_to_combiner(
            name,
            raw,
            rule_weights=rule_weights,
            scalars=scalars,
            use_attenuation=use_attenuation,
            attenuation=attenuation,
            weighted_sum=weighted_sum,
            available_weight=available_weight,
            rows=rows,
        )

    for lookback in [10, 20, 40, 80, 160, 320]:
        add(f"breakout{lookback}", rob.breakout(price, lookback))

    for fast in [2, 4, 8, 16, 32, 64]:
        add(f"assettrend{fast}", rob.ewmac(asset_class_price, asset_class_vol, fast, fast * 4))
        add(f"normmom{fast}", rob.ewmac(normalised_price, normalised_vol, fast, fast * 4))

    for fast in [4, 8, 16, 32, 64]:
        add(f"momentum{fast}", rob.ewmac(price, price_vol, fast, fast * 4))

    for horizon in [10, 20, 40, 80]:
        add(f"relmomentum{horizon}", rob.relative_momentum(normalised_price, asset_class_price, horizon))

    add("mrinasset1000", rob.cross_sectional_mean_reversion(normalised_price, asset_class_price, 1000))
    add("skewabs180", rob.factor_rule(neg_skew_180.sub(global_skew_avg_180, axis=0), 45))
    add("skewabs365", rob.factor_rule(neg_skew_365.sub(global_skew_avg_365, axis=0), 90))
    add("skewrv180", rob.factor_rule(neg_skew_180 - asset_skew_avg_180, 45))
    add("skewrv365", rob.factor_rule(neg_skew_365 - asset_skew_avg_365, 90))

    for fast in [16, 32, 64]:
        ewmac_signal = rob.ewmac(price, price_vol, fast, fast * 4)
        add(f"accel{fast}", ewmac_signal - ewmac_signal.shift(fast))

    valid_history = price.notna().rolling(pit.MIN_HISTORY_DAYS, min_periods=pit.MIN_HISTORY_DAYS).sum() >= pit.MIN_HISTORY_DAYS
    combined = (weighted_sum / available_weight.replace(0.0, np.nan) * fdm).clip(-FORECAST_CAP, FORECAST_CAP)
    combined = combined.where(valid_history)
    return combined, price_vol, pd.DataFrame(rows)


def risk_multiplier_for_targets(
    target: pd.DataFrame,
    unit_daily_cash_vol: pd.DataFrame,
    price: pd.DataFrame,
    capital: float,
    vol_target: float,
) -> pd.DataFrame:
    unit_ann_risk = unit_daily_cash_vol * math.sqrt(BUSINESS_DAYS) / capital
    normal_risk = ((target.fillna(0.0) * unit_ann_risk.fillna(0.0)) ** 2).sum(axis=1).pow(0.5)
    sum_abs_risk = (target.fillna(0.0).abs() * unit_ann_risk.fillna(0.0)).sum(axis=1)
    leverage = (target.fillna(0.0).abs() * price.abs().ffill()).sum(axis=1) / capital

    max_normal = 1.75
    max_sum_abs = 4.0
    max_leverage = 20.0
    multiplier = pd.Series(1.0, index=target.index)
    multiplier = np.minimum(multiplier, (vol_target * max_normal / normal_risk.replace(0.0, np.nan)).fillna(1.0))
    multiplier = np.minimum(multiplier, (vol_target * max_sum_abs / sum_abs_risk.replace(0.0, np.nan)).fillna(1.0))
    multiplier = np.minimum(multiplier, (max_leverage / leverage.replace(0.0, np.nan)).fillna(1.0))
    out = pd.DataFrame(
        {
            "risk_multiplier": multiplier.clip(lower=0.0, upper=1.0),
            "ex_ante_normal_risk": normal_risk,
            "ex_ante_sum_abs_risk": sum_abs_risk,
            "pre_overlay_leverage": leverage,
        },
        index=target.index,
    )
    return out


def apply_position_buffer(target: pd.DataFrame, buffer: pd.DataFrame) -> pd.DataFrame:
    target_values = target.fillna(0.0).to_numpy(dtype=float)
    buffer_values = buffer.fillna(0.0).to_numpy(dtype=float)
    output = np.zeros_like(target_values)
    prior = np.zeros(target_values.shape[1], dtype=float)
    for row_index in range(target_values.shape[0]):
        desired = target_values[row_index]
        edge = buffer_values[row_index]
        new_position = prior.copy()
        too_high = desired > prior + edge
        too_low = desired < prior - edge
        new_position[too_high] = desired[too_high] - edge[too_high]
        new_position[too_low] = desired[too_low] + edge[too_low]
        output[row_index] = new_position
        prior = new_position
    return pd.DataFrame(output, index=target.index, columns=target.columns)


def target_positions(
    price: pd.DataFrame,
    price_vol: pd.DataFrame,
    forecast: pd.DataFrame,
    annual: pd.DataFrame,
    mode: str,
    *,
    capital: float,
    vol_target: float,
    idm: float | pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_cash_vol_target = capital * vol_target / math.sqrt(BUSINESS_DAYS)
    unit_daily_cash_vol = price_vol.abs().replace(0.0, np.nan)
    base = daily_base_weights(annual, price.columns, price.index, mode)
    valid = forecast.notna() & unit_daily_cash_vol.notna() & (unit_daily_cash_vol > 0.0) & price.notna()
    active_base = base.where(valid).fillna(0.0)
    active_sum = active_base.sum(axis=1).replace(0.0, np.nan)
    instrument_weights = active_base.div(active_sum, axis=0).fillna(0.0)

    subsystem_position = daily_cash_vol_target / unit_daily_cash_vol
    if isinstance(idm, pd.Series):
        idm_series = idm.reindex(price.index).ffill().fillna(1.0).astype(float)
    else:
        idm_series = pd.Series(float(idm), index=price.index)

    target = subsystem_position * (forecast / AVERAGE_ABS_FORECAST) * instrument_weights
    target = target.mul(idm_series, axis=0)
    target = target.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    risk = risk_multiplier_for_targets(target, unit_daily_cash_vol, price, capital, vol_target)
    risk["idm"] = idm_series
    target = target.mul(risk["risk_multiplier"], axis=0)
    buffer = subsystem_position.abs() * instrument_weights * BUFFER_SIZE
    buffer = buffer.mul(idm_series, axis=0).mul(risk["risk_multiplier"], axis=0)
    buffered = apply_position_buffer(target, buffer.fillna(0.0))
    integer_positions = buffered.round()
    return integer_positions, target, instrument_weights, risk


def pnl_from_stock_positions(
    positions: pd.DataFrame,
    price: pd.DataFrame,
    capital: float,
    cost_per_dollar: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    held = positions.shift(1).fillna(0.0)
    returns = price.pct_change(fill_method=None)
    price_change = price.diff().mask(returns.abs() > MAX_ABS_DAILY_RETURN).fillna(0.0)
    gross = held * price_change
    trades = positions.diff().abs().fillna(positions.abs())
    traded_notional = trades * price.abs().ffill().bfill()
    costs = traded_notional * cost_per_dollar
    net = gross - costs

    current_notional = positions * price.abs().ffill()
    gross_exposure = current_notional.abs().sum(axis=1) / capital
    net_exposure = current_notional.sum(axis=1) / capital
    long_exposure = current_notional.clip(lower=0.0).sum(axis=1) / capital
    short_exposure = current_notional.clip(upper=0.0).sum(axis=1) / capital
    daily_return = net.sum(axis=1) / capital
    equity = capital + net.sum(axis=1).cumsum()

    daily = pd.DataFrame(
        {
            "gross_pnl": gross.sum(axis=1),
            "costs": costs.sum(axis=1),
            "net_pnl": net.sum(axis=1),
            "daily_return": daily_return,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "active_names": (positions != 0.0).sum(axis=1),
            "turnover": traded_notional.sum(axis=1) / capital,
        },
        index=price.index,
    )
    by_instrument = pd.concat({"gross_pnl": gross, "costs": costs, "net_pnl": net, "position": positions}, axis=1)
    return daily, by_instrument


def benchmark_daily(key: str, start: str, end: str, capital: float) -> pd.DataFrame:
    returns = baw.load_benchmark(key, start, end)
    equity = capital * (1.0 + returns.fillna(0.0)).cumprod()
    return pd.DataFrame(
        {
            "daily_return": returns,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
        },
        index=returns.index,
    )


def trim_active_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if "gross_exposure" not in daily:
        return daily.dropna(subset=["daily_return"])
    active = daily["gross_exposure"][daily["gross_exposure"] > 0.0]
    if active.empty:
        return daily.dropna(subset=["daily_return"])
    return daily.loc[active.index[0] :].dropna(subset=["daily_return"])


def yearly_returns_from_equity(daily_by_name: dict[str, pd.DataFrame], capital: float) -> pd.DataFrame:
    rows = []
    for name, daily in daily_by_name.items():
        series = daily["equity"].dropna() / capital
        for year, frame in series.groupby(series.index.year):
            if frame.empty:
                continue
            rows.append({"year": int(year), "strategy": name, "return": frame.iloc[-1] / frame.iloc[0] - 1.0})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="year", columns="strategy", values="return")


def diagnostics_table(daily_by_strategy: dict[str, pd.DataFrame], capital: float) -> pd.DataFrame:
    rows = []
    for strategy, daily in daily_by_strategy.items():
        active = trim_active_daily(daily)
        rows.append(
            {
                "strategy": strategy,
                "avg_turnover_annual": active["turnover"].mean() * BUSINESS_DAYS,
                "avg_cost_annual": active["costs"].mean() * BUSINESS_DAYS / capital,
                "avg_gross_exposure": active["gross_exposure"].mean(),
                "avg_net_exposure": active["net_exposure"].mean(),
                "avg_long_exposure": active["long_exposure"].mean(),
                "avg_short_exposure": active["short_exposure"].mean(),
                "avg_active_names": active["active_names"].mean(),
                "max_gross_exposure": active["gross_exposure"].max(),
                "min_net_exposure": active["net_exposure"].min(),
                "max_net_exposure": active["net_exposure"].max(),
            }
        )
    return pd.DataFrame(rows)


def rebase_daily(daily: pd.DataFrame, capital: float, *, compound: bool) -> pd.DataFrame:
    out = daily.copy()
    returns = out["daily_return"].fillna(0.0)
    if compound:
        out["equity"] = capital * (1.0 + returns).cumprod()
    else:
        out["equity"] = capital * (1.0 + returns.cumsum())
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def plot_universe(
    key: str,
    daily_by_name: dict[str, pd.DataFrame],
    annual_returns: pd.DataFrame,
    out_dir: Path,
    capital: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.1, 1.6], hspace=0.16)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[2, 0])

    for name, daily in daily_by_name.items():
        equity_ratio = daily["equity"].dropna() / capital
        ax0.plot(equity_ratio.index, equity_ratio, label=name, linewidth=1.6)
        ax1.plot(equity_ratio.index, equity_ratio / equity_ratio.cummax() - 1.0, linewidth=1.0)

    annual_plot = annual_returns.dropna(how="all")
    x = np.arange(len(annual_plot.index))
    width = min(0.18, 0.8 / max(len(annual_plot.columns), 1))
    for i, name in enumerate(annual_plot.columns):
        ax2.bar(x + (i - (len(annual_plot.columns) - 1) / 2) * width, annual_plot[name], width=width, label=name)

    ax0.set_title(f"{SUPPORTED_UNIVERSES[key]} Rob-Style Stock System")
    ax0.set_yscale("log")
    ax0.set_ylabel("Growth of $1")
    ax1.set_ylabel("Drawdown")
    ax2.set_ylabel("Year return")
    ax2.axhline(0.0, color="#555555", linewidth=0.8)
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([str(year) for year in annual_plot.index[::2]], rotation=45, ha="right")
    ax0.legend(loc="upper left", ncol=2)
    ax2.legend(loc="upper left", ncol=2, fontsize=8)
    ax1.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{key}_rob_style_stock.png", dpi=180)
    plt.close(fig)


def write_universe_summary(
    key: str,
    stats: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rule_table: pd.DataFrame,
    out_dir: Path,
    *,
    capital: float,
    vol_target: float,
    idm: float,
    fdm: float,
    start: str,
    end: str,
) -> None:
    top_rules = rule_table.sort_values("forecast_weight", ascending=False).head(6)
    top_rule_text = ", ".join(
        f"{row.rule} {row.forecast_weight:.1%}" for row in top_rules.itertuples(index=False)
    )
    lines = [
        f"# {SUPPORTED_UNIVERSES[key]} Rob-Style Stock System",
        "",
        f"- Sample requested: {start} to {end}; actual start depends on first point-in-time holdings snapshot and forecast history.",
        "- Signals: Rob price-based rules only: breakout, momentum/EWMAC, normmom, assettrend, relative momentum, mean reversion, skew, and acceleration.",
        "- Excluded signals: carry and roll.",
        f"- Sizing: ${capital:,.0f} notional capital, {vol_target:.0%} annual volatility target, forecast / 10, instrument weights, IDM {idm:.2f}.",
        f"- FDM: {fdm:.2f}, using Rob config median as the stock default.",
        f"- Buffer: {BUFFER_SIZE:.0%} of forecast-10 position before integer share rounding.",
        f"- Cost: {DEFAULT_COST_PER_DOLLAR:.2%} of notional traded; borrow and financing costs are not included.",
        f"- Data hygiene: non-positive prices, prices below ${MIN_TRADEABLE_PRICE:.2f}, one-day moves above {MAX_ABS_DAILY_RETURN:.0%}, and 20x deviations from trailing one-year median are treated as missing.",
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
        "## Portfolio Diagnostics",
        "",
        "| Strategy | Ann Turnover | Ann Cost | Avg Gross | Avg Net | Avg Long | Avg Short | Avg Names | Max Gross | Net Range |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {row['strategy']} | {pct(row['avg_turnover_annual'])} | {pct(row['avg_cost_annual'])} | {pct(row['avg_gross_exposure'])} | {pct(row['avg_net_exposure'])} | {pct(row['avg_long_exposure'])} | {pct(row['avg_short_exposure'])} | {row['avg_active_names']:.0f} | {pct(row['max_gross_exposure'])} | {pct(row['min_net_exposure'])} to {pct(row['max_net_exposure'])} |"
        )

    lines += [
        "",
        "## Rule Weights",
        "",
        f"- Active non-carry rules: {len(rule_table)}.",
        f"- Largest rule weights: {top_rule_text}.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_universe(
    key: str,
    start: str,
    end: str,
    modes: list[str],
    *,
    capital: float,
    vol_target: float,
    idm: float,
    fdm: float,
) -> dict[str, pd.DataFrame]:
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    config = rob.load_rob_config()
    annual = load_annual(key, start, end)
    price = load_price(key, annual, start, end)
    forecast, price_vol, rule_table = build_rob_style_stock_forecast(price, annual, config, fdm)
    benchmark = benchmark_daily(key, start, end, capital)

    daily_by_name: dict[str, pd.DataFrame] = {}
    risk_frames = []
    for mode in modes:
        label = f"rob_{mode}"
        positions, _target, _instrument_weights, risk = target_positions(
            price,
            price_vol,
            forecast,
            annual,
            mode,
            capital=capital,
            vol_target=vol_target,
            idm=idm,
        )
        daily, _by_instrument = pnl_from_stock_positions(positions, price, capital, DEFAULT_COST_PER_DOLLAR)
        daily = daily.join(risk)
        daily = daily.loc[start:end]
        daily_by_name[label] = trim_active_daily(daily)
        daily.to_csv(out_dir / f"{label}_daily.csv")
        positions.iloc[::5].to_csv(out_dir / f"{label}_weekly_positions.csv")
        risk.assign(strategy=label).to_csv(out_dir / f"{label}_risk_overlay.csv")
        risk_frames.append(risk.assign(strategy=label))

    benchmark_label = baw.load_benchmark(key, start, end).name
    daily_by_name[benchmark_label] = benchmark.loc[start:end]

    common_start = max(frame.index.min() for frame in daily_by_name.values() if not frame.empty)
    common_end = min(frame.index.max() for frame in daily_by_name.values() if not frame.empty)
    daily_by_name = {name: frame.loc[common_start:common_end] for name, frame in daily_by_name.items()}
    daily_by_name = {
        name: rebase_daily(frame, capital, compound=not name.startswith("rob_"))
        for name, frame in daily_by_name.items()
    }

    stats_rows = []
    for name, daily in daily_by_name.items():
        stats = performance_stats_from_equity(daily["daily_return"], daily["equity"] / capital)
        stats_rows.append({"strategy": name, **stats})
    stats = pd.DataFrame(stats_rows)
    diagnostics = diagnostics_table({name: frame for name, frame in daily_by_name.items() if name.startswith("rob_")}, capital)
    annual_returns = yearly_returns_from_equity(daily_by_name, capital)

    rule_table.to_csv(out_dir / "rule_weights_and_scalars.csv", index=False)
    annual.to_csv(out_dir / "point_in_time_constituents_used.csv", index=False)
    stats.to_csv(out_dir / "stats.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    annual_returns.to_csv(out_dir / "yearly_returns.csv")
    if risk_frames:
        pd.concat(risk_frames).to_csv(out_dir / "risk_overlay.csv")
    plot_universe(key, daily_by_name, annual_returns, out_dir, capital)
    write_universe_summary(
        key,
        stats,
        diagnostics,
        rule_table,
        out_dir,
        capital=capital,
        vol_target=vol_target,
        idm=idm,
        fdm=fdm,
        start=start,
        end=end,
    )

    return {
        "stats": stats.assign(universe=SUPPORTED_UNIVERSES[key]),
        "diagnostics": diagnostics.assign(universe=SUPPORTED_UNIVERSES[key]),
        "annual_returns": annual_returns.assign(universe=SUPPORTED_UNIVERSES[key]),
    }


def write_combined_summary(all_stats: pd.DataFrame, all_diag: pd.DataFrame, *, fdm: float, idm: float) -> None:
    lines = [
        "# Rob-Style Stock System",
        "",
        "This run translates the Rob Carver futures system to stock universes while excluding carry/roll signals.",
        f"FDM uses the Rob config median ({fdm:.2f}); IDM is {idm:.2f}.",
        "",
        "## Performance",
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
        "## Diagnostics",
        "",
        "| Universe | Strategy | Ann Turnover | Ann Cost | Avg Gross | Avg Net | Avg Long | Avg Short | Avg Names | Max Gross |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_diag.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {pct(row['avg_turnover_annual'])} | {pct(row['avg_cost_annual'])} | {pct(row['avg_gross_exposure'])} | {pct(row['avg_net_exposure'])} | {pct(row['avg_long_exposure'])} | {pct(row['avg_short_exposure'])} | {row['avg_active_names']:.0f} | {pct(row['max_gross_exposure'])} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", nargs="+", default=list(SUPPORTED_UNIVERSES), choices=sorted(SUPPORTED_UNIVERSES))
    parser.add_argument("--weight-modes", nargs="+", default=["equal", "benchmark", "sector_equal"], choices=sorted(WEIGHT_MODES))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--vol-target", type=float, default=DEFAULT_VOL_TARGET)
    parser.add_argument("--idm", type=float, default=2.75)
    parser.add_argument("--fdm", default="median_rob", help="Use 'median_rob' or a numeric FDM value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = rob.load_rob_config()
    fdm = median_rob_fdm(config) if args.fdm == "median_rob" else float(args.fdm)

    all_stats = []
    all_diag = []
    for key in args.universes:
        result = run_universe(
            key,
            args.start,
            args.end,
            args.weight_modes,
            capital=args.capital,
            vol_target=args.vol_target,
            idm=args.idm,
            fdm=fdm,
        )
        all_stats.append(result["stats"])
        all_diag.append(result["diagnostics"])

    stats = pd.concat(all_stats, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True)
    stats.to_csv(OUT / "all_stats.csv", index=False)
    diag.to_csv(OUT / "all_diagnostics.csv", index=False)
    write_combined_summary(stats, diag, fdm=fdm, idm=args.idm)
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
