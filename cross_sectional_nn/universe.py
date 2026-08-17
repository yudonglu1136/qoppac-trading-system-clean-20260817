"""Point-in-time stock universe loading for cross-sectional forecasts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_benchmark_aware_stock_momentum as baw  # noqa: E402
import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402

from .config import MARKET_LABELS


@dataclass
class UniverseData:
    key: str
    market: str
    annual: pd.DataFrame
    price: pd.DataFrame
    benchmark_price: pd.Series
    active: pd.DataFrame
    sector: pd.DataFrame
    open_price: pd.DataFrame
    high_price: pd.DataFrame
    low_price: pd.DataFrame
    ohlcv_close: pd.DataFrame
    volume: pd.DataFrame


def load_benchmark_price(key: str, start: str, end: str) -> pd.Series:
    spec = next(spec for spec in pit.UNIVERSES if spec.key == key)
    path = pit.DATA_ROOT / key / "benchmark_adj_close.csv"
    price = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()[spec.benchmark_ticker]
    return price.loc[:end].ffill().rename(spec.benchmark_label)


def load_ohlcv_field(key: str, field: str, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    path = pit.DATA_ROOT / key / "ohlcv" / f"{field}.csv"
    if not path.exists():
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.reindex(index=index, columns=columns)


def sector_frame_from_annual(annual: pd.DataFrame, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    snapshots = sorted(pd.to_datetime(annual["snapshot_date"].dropna().unique()))
    sectors = pd.DataFrame("Unknown", index=index, columns=columns, dtype="object")
    if not snapshots:
        return sectors

    for i, snapshot_date in enumerate(snapshots):
        next_snapshot = snapshots[i + 1] if i + 1 < len(snapshots) else pd.Timestamp.max
        mask = (index >= snapshot_date) & (index < next_snapshot)
        if not mask.any():
            continue
        frame = annual[annual["snapshot_date"].eq(snapshot_date)].drop_duplicates("symbol").set_index("symbol")
        frame = frame[frame.index.isin(columns)]
        if frame.empty:
            continue
        values = frame.get("sector", pd.Series("Unknown", index=frame.index)).fillna("Unknown").astype(str)
        sectors.loc[mask, values.index] = values
    return sectors


def active_mask(annual: pd.DataFrame, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    weights = rob_stock.daily_base_weights(annual, columns, index, "equal")
    return weights > 0.0


def load_universe(key: str, start: str, end: str) -> UniverseData:
    annual = rob_stock.load_annual(key, start, end)
    price = rob_stock.load_price(key, annual, start, end)
    benchmark_price = load_benchmark_price(key, start, end).reindex(price.index).ffill()
    active = active_mask(annual, price.columns, price.index)
    sector = sector_frame_from_annual(annual, price.columns, price.index)
    open_price = load_ohlcv_field(key, "open", price.columns, price.index)
    high_price = load_ohlcv_field(key, "high", price.columns, price.index)
    low_price = load_ohlcv_field(key, "low", price.columns, price.index)
    ohlcv_close = load_ohlcv_field(key, "close", price.columns, price.index)
    volume = load_ohlcv_field(key, "volume", price.columns, price.index)
    return UniverseData(
        key=key,
        market=MARKET_LABELS[key],
        annual=annual,
        price=price,
        benchmark_price=benchmark_price,
        active=active,
        sector=sector,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        ohlcv_close=ohlcv_close,
        volume=volume,
    )


def annual_membership_counts(universes: list[UniverseData]) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for data in universes:
        annual = data.annual.copy()
        annual["year"] = pd.to_datetime(annual["snapshot_date"]).dt.year
        for year, frame in annual.groupby("year"):
            rows.append({"market": data.market, "year": int(year), "stock_count": int(frame["symbol"].nunique())})
    return pd.DataFrame(rows).sort_values(["market", "year"])
