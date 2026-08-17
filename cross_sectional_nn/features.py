"""Feature engineering for pooled cross-sectional stock models."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rob_style_backtest as rob  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402

from .config import FeatureConfig
from .universe import UniverseData

BUSINESS_DAYS = 252.0


@dataclass
class FeatureFrames:
    frames: dict[str, pd.DataFrame]
    metadata: pd.DataFrame


def log_return(price: pd.DataFrame | pd.Series, horizon: int) -> pd.DataFrame | pd.Series:
    return np.log(price / price.shift(horizon))


def annualized_vol(log_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return log_returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(BUSINESS_DAYS)


def finite_log(frame: pd.DataFrame) -> pd.DataFrame:
    logged = np.log(frame.where(frame > 0.0))
    return logged.replace([np.inf, -np.inf], np.nan)


def rank_scaled(frame: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    masked = frame.where(active)
    ranks = masked.rank(axis=1)
    counts = masked.notna().sum(axis=1)
    pct = ranks.sub(1.0).div((counts - 1.0).replace(0.0, np.nan), axis=0)
    pct = pct.where(counts > 1, 0.5)
    return (2.0 * pct - 1.0).where(active)


def sector_rank_scaled(frame: pd.DataFrame, active: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for date in frame.index:
        values = frame.loc[date].where(active.loc[date]).dropna()
        if values.empty:
            continue
        sectors = sector.loc[date, values.index].fillna("Unknown").astype(str)
        for _sector_name, members in sectors.groupby(sectors).groups.items():
            member_values = values.reindex(list(members)).dropna()
            if member_values.empty:
                continue
            if len(member_values) == 1:
                out.loc[date, member_values.index] = 0.0
            else:
                pct = (member_values.rank() - 1.0) / (len(member_values) - 1.0)
                out.loc[date, member_values.index] = 2.0 * pct - 1.0
    return out


def breakout(price: pd.DataFrame, window: int) -> pd.DataFrame:
    high = price.shift(1).rolling(window, min_periods=max(5, window // 2)).max()
    low = price.shift(1).rolling(window, min_periods=max(5, window // 2)).min()
    return (2.0 * (price - low) / (high - low).replace(0.0, np.nan) - 1.0).clip(-3.0, 3.0)


def drawdown(price: pd.DataFrame, window: int) -> pd.DataFrame:
    high = price.shift(1).rolling(window, min_periods=max(5, window // 2)).max()
    return price / high.replace(0.0, np.nan) - 1.0


def sector_group_mean(frame: pd.DataFrame, active: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    change = sector.ne(sector.shift()).any(axis=1)
    starts = list(sector.index[change])
    if not starts:
        return out
    starts.append(pd.Timestamp.max)
    for start, end in zip(starts[:-1], starts[1:]):
        period_index = frame.index[(frame.index >= start) & (frame.index < end)]
        if len(period_index) == 0:
            continue
        sector_row = sector.loc[start].fillna("Unknown").astype(str)
        sector_values = sector_row.to_numpy()
        for sector_name in pd.unique(sector_values):
            members = sector_row.index[sector_values == sector_name]
            if len(members) == 0:
                continue
            values = frame.loc[period_index, members].where(active.loc[period_index, members])
            group_mean = values.mean(axis=1)
            out.loc[period_index, members] = np.repeat(group_mean.to_numpy()[:, None], len(members), axis=1)
    return out


def build_feature_frames(data: UniverseData, config: FeatureConfig) -> FeatureFrames:
    price = data.price
    active = data.active
    log_price = np.log(price)
    log_returns = log_price.diff()
    sigma_60 = annualized_vol(log_returns, 60)
    daily_sigma_60 = sigma_60 / np.sqrt(BUSINESS_DAYS)
    benchmark_log = np.log(data.benchmark_price).reindex(price.index).ffill()
    benchmark_returns = {h: benchmark_log - benchmark_log.shift(h) for h in set(config.momentum_horizons + config.relative_horizons)}

    frames: dict[str, pd.DataFrame] = {}

    for horizon in config.momentum_horizons:
        raw = log_return(price, horizon)
        denom = sigma_60 * np.sqrt(horizon / BUSINESS_DAYS)
        frames[f"ret_{horizon}"] = (raw / denom.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)

    for horizon in config.relative_horizons:
        stock_ret = log_return(price, horizon)
        rel = stock_ret.sub(benchmark_returns[horizon], axis=0)
        denom = sigma_60 * np.sqrt(horizon / BUSINESS_DAYS)
        frames[f"rel_ret_{horizon}"] = (rel / denom.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)

    for horizon in config.sector_relative_horizons:
        stock_ret = log_return(price, horizon)
        sector_ret = sector_group_mean(stock_ret, active, data.sector)
        denom = sigma_60 * np.sqrt(horizon / BUSINESS_DAYS)
        frames[f"sector_rel_{horizon}"] = ((stock_ret - sector_ret) / denom.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)

    price_vol = rob_stock.rob.mixed_vol(price.diff())
    for fast, slow in config.ewmac_pairs:
        frames[f"ewmac_{fast}_{slow}"] = rob.ewmac(price, price_vol, fast, slow).clip(-10.0, 10.0).where(active)

    vol_frames: dict[int, pd.DataFrame] = {}
    for window in config.vol_windows:
        vol = annualized_vol(log_returns, window).where(active)
        vol_frames[window] = vol
        frames[f"vol_{window}"] = vol.clip(0.0, 5.0)

    frames["vol_ratio_10_60"] = (vol_frames[10] / vol_frames[60].replace(0.0, np.nan)).clip(0.0, 10.0).where(active)
    frames["vol_ratio_20_120"] = (vol_frames[20] / vol_frames[120].replace(0.0, np.nan)).clip(0.0, 10.0).where(active)
    vol_20 = vol_frames[20]
    vol_mean = vol_20.rolling(252, min_periods=60).mean()
    vol_std = vol_20.rolling(252, min_periods=60).std()
    frames["vol_zscore"] = ((vol_20 - vol_mean) / vol_std.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)

    for window in config.breakout_windows:
        frames[f"breakout_{window}"] = breakout(price, window).where(active)

    for window in config.drawdown_windows:
        frames[f"drawdown_{window}"] = drawdown(price, window).clip(-1.0, 0.5).where(active)

    if config.include_volume_features and not data.volume.empty:
        volume = data.volume.where(active)
        ohlcv_close = data.ohlcv_close.reindex_like(price).where(active)
        traded_value = (ohlcv_close * volume).where(lambda x: x > 0.0)
        log_dollar_volume = finite_log(traded_value)
        log_volume = finite_log(volume)
        frames["log_dollar_volume"] = log_dollar_volume.clip(lower=0.0, upper=30.0).where(active)
        frames["volume_ratio_5_60"] = (
            volume.rolling(5, min_periods=3).mean() / volume.rolling(60, min_periods=20).mean().replace(0.0, np.nan)
        ).clip(0.0, 10.0).where(active)
        frames["volume_ratio_20_120"] = (
            volume.rolling(20, min_periods=10).mean() / volume.rolling(120, min_periods=40).mean().replace(0.0, np.nan)
        ).clip(0.0, 10.0).where(active)
        volume_mean = log_volume.rolling(252, min_periods=60).mean()
        volume_std = log_volume.rolling(252, min_periods=60).std()
        frames["volume_zscore"] = ((log_volume - volume_mean) / volume_std.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)

    if config.include_ohlc_features and not data.open_price.empty:
        open_price = data.open_price.reindex_like(price)
        high_price = data.high_price.reindex_like(price)
        low_price = data.low_price.reindex_like(price)
        ohlcv_close = data.ohlcv_close.reindex_like(price)
        overnight = np.log(open_price / ohlcv_close.shift(1))
        intraday = np.log(ohlcv_close / open_price)
        gap = open_price / ohlcv_close.shift(1) - 1.0
        high_low_range = (high_price - low_price) / ohlcv_close.replace(0.0, np.nan)
        close_location = (ohlcv_close - low_price) / (high_price - low_price).replace(0.0, np.nan)
        frames["overnight_return"] = (overnight / daily_sigma_60.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)
        frames["intraday_return"] = (intraday / daily_sigma_60.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)
        frames["high_low_range"] = (high_low_range / daily_sigma_60.replace(0.0, np.nan)).clip(0.0, 10.0).where(active)
        frames["close_location"] = close_location.clip(0.0, 1.0).where(active)
        frames["gap"] = (gap / daily_sigma_60.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)

    base_names = list(frames)
    if config.add_cross_sectional_ranks:
        for name in base_names:
            frames[f"{name}_xrank"] = rank_scaled(frames[name], active)

    if config.add_sector_ranks:
        for name in base_names:
            frames[f"{name}_srank"] = sector_rank_scaled(frames[name], active, data.sector)

    metadata_rows = []
    for name, formula, lookback, normalisation, ranked in feature_dictionary_rows(config):
        if name not in frames:
            continue
        metadata_rows.append(
            {
                "feature": name,
                "formula": formula,
                "lookback": lookback,
                "normalization": normalisation,
                "cross_sectional_rank": ranked,
                "leakage_risk": "uses same-day or trailing data only; no future prices",
            }
        )
    return FeatureFrames(frames=frames, metadata=pd.DataFrame(metadata_rows))


def feature_dictionary_rows(config: FeatureConfig) -> list[tuple[str, str, str, str, bool]]:
    rows: list[tuple[str, str, str, str, bool]] = []
    for horizon in config.momentum_horizons:
        rows.append((f"ret_{horizon}", f"log(P_t/P_t-{horizon})", f"{horizon} trading days", "stock sigma_60 * sqrt(h/252)", False))
    for horizon in config.relative_horizons:
        rows.append((f"rel_ret_{horizon}", f"stock {horizon}D log return - market {horizon}D log return", f"{horizon} trading days", "stock sigma_60 * sqrt(h/252)", False))
    for horizon in config.sector_relative_horizons:
        rows.append((f"sector_rel_{horizon}", f"stock {horizon}D log return - same-sector mean {horizon}D log return", f"{horizon} trading days", "stock sigma_60 * sqrt(h/252)", False))
    for fast, slow in config.ewmac_pairs:
        rows.append((f"ewmac_{fast}_{slow}", f"Rob EWMAC fast={fast}, slow={slow}", f"{slow} EWM span", "Rob price-vol normalization", False))
    for window in config.vol_windows:
        rows.append((f"vol_{window}", f"rolling std(log returns) * sqrt(252)", f"{window} trading days", "annualized volatility", False))
    rows.append(("vol_ratio_10_60", "vol_10 / vol_60", "60 trading days", "ratio", False))
    rows.append(("vol_ratio_20_120", "vol_20 / vol_120", "120 trading days", "ratio", False))
    rows.append(("vol_zscore", "(vol_20 - trailing 252D mean) / trailing 252D std", "252 trading days", "time-series z-score fit only on history", False))
    for window in config.breakout_windows:
        rows.append((f"breakout_{window}", "2*(P_t - rolling_low)/(rolling_high - rolling_low)-1", f"{window} trading days", "bounded oscillator", False))
    for window in config.drawdown_windows:
        rows.append((f"drawdown_{window}", "P_t / trailing rolling_max - 1", f"{window} trading days", "raw drawdown", False))
    if config.include_volume_features:
        rows.append(("log_dollar_volume", "log(adjusted OHLC close * reported daily volume)", "same day", "raw log value; global scaler fit on train only", False))
        rows.append(("volume_ratio_5_60", "avg_volume_5 / avg_volume_60", "60 trading days", "ratio", False))
        rows.append(("volume_ratio_20_120", "avg_volume_20 / avg_volume_120", "120 trading days", "ratio", False))
        rows.append(("volume_zscore", "(log volume - trailing 252D mean) / trailing 252D std", "252 trading days", "time-series z-score fit only on history", False))
    if config.include_ohlc_features:
        rows.append(("overnight_return", "log(Open_t / OHLC_Close_t-1)", "1 trading day", "stock sigma_60 / sqrt(252)", False))
        rows.append(("intraday_return", "log(OHLC_Close_t / Open_t)", "same day", "stock sigma_60 / sqrt(252)", False))
        rows.append(("high_low_range", "(High_t - Low_t) / OHLC_Close_t", "same day", "stock sigma_60 / sqrt(252)", False))
        rows.append(("close_location", "(OHLC_Close_t - Low_t) / (High_t - Low_t)", "same day", "bounded 0..1", False))
        rows.append(("gap", "Open_t / OHLC_Close_t-1 - 1", "1 trading day", "stock sigma_60 / sqrt(252)", False))
    if config.add_cross_sectional_ranks:
        rows.extend([(f"{name}_xrank", f"daily market percentile rank of {name}", "same day", "2*pct_rank-1", True) for name, *_ in rows.copy()])
    return rows
