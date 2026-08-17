"""Target construction for cross-sectional alpha models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .features import BUSINESS_DAYS, annualized_vol, sector_group_mean
from .universe import UniverseData


def forward_log_return(price: pd.DataFrame | pd.Series, horizon: int) -> pd.DataFrame | pd.Series:
    return np.log(price.shift(-horizon) / price)


def build_target_frame(data: UniverseData, config: FeatureConfig, target_type: str = "market") -> pd.DataFrame:
    horizon = config.target_horizon
    stock_forward = forward_log_return(data.price, horizon)
    log_returns = np.log(data.price).diff()
    sigma = annualized_vol(log_returns, config.target_vol_window)
    denom = sigma * np.sqrt(horizon / BUSINESS_DAYS)

    if target_type == "market":
        benchmark_forward = forward_log_return(np.log(data.benchmark_price).pipe(np.exp), horizon)
        excess = stock_forward.sub(benchmark_forward, axis=0)
    elif target_type == "sector":
        sector_forward = sector_group_mean(stock_forward, data.active, data.sector)
        excess = stock_forward - sector_forward
    else:
        raise ValueError(f"unknown target_type: {target_type}")

    target = excess / denom.replace(0.0, np.nan)
    return target.clip(-config.target_clip, config.target_clip).where(data.active)


def label_end_dates(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    values = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    if len(index) > horizon:
        values.iloc[:-horizon] = index[horizon:]
    return values

