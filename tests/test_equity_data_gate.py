from __future__ import annotations

import inspect

import pytest

from cross_sectional_nn.config import FeatureConfig
import run_rob_style_stock_backtest as rob_stock


def require_pit_universe_data(key: str) -> None:
    path = rob_stock.pit.DATA_ROOT / key / "annual_constituents.csv"
    if not path.exists():
        pytest.skip(f"local point-in-time universe data not present: {path}")


def test_eem_loader_uses_snapshot_level_holdings() -> None:
    require_pit_universe_data("eem")
    annual = rob_stock.load_annual("eem", "2016-01-01", "2026-08-07")
    assert annual["snapshot_date"].nunique() > 11


def test_stock_price_loader_keeps_membership_symbols() -> None:
    source = inspect.getsource(rob_stock.load_price)
    assert "reindex(columns=symbols)" in source
    assert "notna().sum() >= pit.MIN_HISTORY_DAYS" not in source


def test_ohlc_features_disabled_by_default() -> None:
    assert FeatureConfig().include_ohlc_features is False
