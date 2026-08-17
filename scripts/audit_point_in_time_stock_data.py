#!/usr/bin/env python3
"""Audit point-in-time stock universe sector and OHLCV coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402


OUT = ROOT / "backtests" / "point_in_time_data_audit"
UNIVERSES = ("sp500", "eem", "efa")
FIELDS = ("open", "high", "low", "close", "volume")


def load_panel(key: str, field: str) -> pd.DataFrame:
    path = pit.DATA_ROOT / key / "ohlcv" / f"{field}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")


def unique_annual(annual: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["symbol", "name", "sector"] if col in annual.columns]
    return annual[cols].drop_duplicates("symbol").set_index("symbol")


def symbols_with_any_data(panel: pd.DataFrame) -> set[str]:
    if panel.empty:
        return set()
    return {str(column) for column in panel.columns if panel[column].notna().any()}


def symbols_with_min_obs(panel: pd.DataFrame, minimum: int) -> set[str]:
    if panel.empty:
        return set()
    return {str(column) for column in panel.columns if panel[column].notna().sum() >= minimum}


def field_coverage_rows(key: str, annual: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> list[dict]:
    symbols = sorted(annual["symbol"].dropna().astype(str).unique())
    rows = []
    for field, panel in panels.items():
        any_data = symbols_with_any_data(panel)
        min_252 = symbols_with_min_obs(panel, 252)
        rows.append(
            {
                "universe": key,
                "field": field,
                "requested_symbols": len(symbols),
                "symbols_with_any_data": sum(symbol in any_data for symbol in symbols),
                "symbols_with_252_obs": sum(symbol in min_252 for symbol in symbols),
                "coverage_any": sum(symbol in any_data for symbol in symbols) / len(symbols),
                "coverage_252": sum(symbol in min_252 for symbol in symbols) / len(symbols),
                "first_date": panel.index.min().date().isoformat() if not panel.empty else "",
                "last_date": panel.index.max().date().isoformat() if not panel.empty else "",
            }
        )
    available_all = set(symbols)
    min_252_all = set(symbols)
    for panel in panels.values():
        available_all &= symbols_with_any_data(panel)
        min_252_all &= symbols_with_min_obs(panel, 252)
    rows.append(
        {
            "universe": key,
            "field": "all_ohlcv",
            "requested_symbols": len(symbols),
            "symbols_with_any_data": len(available_all),
            "symbols_with_252_obs": len(min_252_all),
            "coverage_any": len(available_all) / len(symbols),
            "coverage_252": len(min_252_all) / len(symbols),
            "first_date": panels["close"].index.min().date().isoformat() if not panels["close"].empty else "",
            "last_date": panels["close"].index.max().date().isoformat() if not panels["close"].empty else "",
        }
    )
    return rows


def yearly_coverage_rows(key: str, annual: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> list[dict]:
    close = panels["close"]
    annual = annual.copy()
    annual["snapshot_date"] = pd.to_datetime(annual["snapshot_date"], errors="coerce")
    annual["year"] = annual["snapshot_date"].dt.year
    rows = []
    for year, frame in annual.dropna(subset=["year"]).groupby("year"):
        year = int(year)
        members = sorted(frame["symbol"].dropna().astype(str).unique())
        start = pd.Timestamp(f"{year}-01-01")
        end = min(pd.Timestamp(f"{year}-12-31"), close.index.max()) if not close.empty else pd.Timestamp(f"{year}-12-31")
        year_close = close.loc[(close.index >= start) & (close.index <= end), close.columns.intersection(members)]
        any_close = symbols_with_any_data(year_close)
        min_126 = symbols_with_min_obs(year_close, 126)
        sector = frame.get("sector", pd.Series(index=frame.index, dtype=object)).replace("", pd.NA)
        sector_coverage = sector.notna().mean() if len(frame) else 0.0
        rows.append(
            {
                "universe": key,
                "year": year,
                "members": len(members),
                "symbols_with_close_in_year": sum(symbol in any_close for symbol in members),
                "symbols_with_126_close_obs_in_year": sum(symbol in min_126 for symbol in members),
                "close_coverage_in_year": sum(symbol in any_close for symbol in members) / len(members),
                "sector_coverage": sector_coverage,
            }
        )
    return rows


def empty_symbol_rows(key: str, annual: pd.DataFrame, close: pd.DataFrame) -> list[dict]:
    symbols = sorted(annual["symbol"].dropna().astype(str).unique())
    available = symbols_with_any_data(close)
    details = unique_annual(annual)
    rows = []
    for symbol in symbols:
        if symbol in available:
            continue
        row = details.reindex([symbol]).iloc[0]
        rows.append(
            {
                "universe": key,
                "symbol": symbol,
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    field_rows = []
    yearly_rows = []
    empty_rows = []
    for key in UNIVERSES:
        annual = pd.read_csv(pit.DATA_ROOT / key / "annual_constituents.csv")
        panels = {field: load_panel(key, field) for field in FIELDS}
        field_rows.extend(field_coverage_rows(key, annual, panels))
        yearly_rows.extend(yearly_coverage_rows(key, annual, panels))
        empty_rows.extend(empty_symbol_rows(key, annual, panels["close"]))

    field_coverage = pd.DataFrame(field_rows)
    yearly_coverage = pd.DataFrame(yearly_rows)
    empty_symbols = pd.DataFrame(empty_rows)
    field_coverage.to_csv(OUT / "field_coverage.csv", index=False)
    yearly_coverage.to_csv(OUT / "yearly_coverage.csv", index=False)
    empty_symbols.to_csv(OUT / "empty_ohlcv_symbols.csv", index=False)
    print(field_coverage[field_coverage["field"].eq("all_ohlcv")].to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
