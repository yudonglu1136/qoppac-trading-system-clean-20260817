from __future__ import annotations

import pandas as pd
import pytest

import run_rob_style_stock_backtest as rob_stock


def require_pit_universe_data(key: str) -> None:
    path = rob_stock.pit.DATA_ROOT / key / "annual_constituents.csv"
    if not path.exists():
        pytest.skip(f"local point-in-time universe data not present: {path}")


def test_sp500_loader_uses_pre_2020_legacy_membership() -> None:
    require_pit_universe_data("sp500")
    annual = rob_stock.load_annual("sp500", "2016-01-01", "2026-08-07")
    assert pd.to_datetime(annual["snapshot_date"]).min() <= pd.Timestamp("2016-01-01")
    assert annual.loc[pd.to_datetime(annual["snapshot_date"]).dt.year.eq(2016), "symbol"].nunique() > 400
