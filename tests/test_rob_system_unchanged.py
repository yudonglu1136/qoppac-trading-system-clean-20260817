from __future__ import annotations

import inspect

import run_rob_style_stock_backtest as rob_stock


def test_neural_package_does_not_enter_position_sizing() -> None:
    source = inspect.getsource(rob_stock.target_positions)
    assert "cross_sectional_nn" not in source
    assert "forecast / AVERAGE_ABS_FORECAST" in source


def test_execution_uses_next_day_held_positions() -> None:
    source = inspect.getsource(rob_stock.pnl_from_stock_positions)
    assert "held = positions.shift(1)" in source

