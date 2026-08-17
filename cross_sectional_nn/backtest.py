"""Backtest generated forecasts through the existing Rob stock system."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_benchmark_aware_stock_momentum as baw  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402
import run_stock_forecast_lab as lab  # noqa: E402
import run_stock_forecast_tail_lab as tail_lab  # noqa: E402

from .forecast_conversion import forecast_matrix, raw_alpha_to_forecast


def model_forecast_matrix(
    predictions: pd.DataFrame,
    market_key: str,
    model: str,
    *,
    forecast_mode: str = "dense_rank",
    selection_frac: float = 0.10,
    selected_forecast: float = 10.0,
    score_smoothing_span: int = 1,
    selection_interval: str = "daily",
    exit_frac: float | None = None,
) -> pd.DataFrame:
    frame = predictions[predictions["model"].eq(model)]
    return forecast_matrix(
        frame,
        market_key,
        forecast_mode=forecast_mode,
        selection_frac=selection_frac,
        selected_forecast=selected_forecast,
        score_smoothing_span=score_smoothing_span,
        selection_interval=selection_interval,
        exit_frac=exit_frac,
    )


def original_rob_forecast_matrix(key: str, start: str, end: str, forecast_set: str = "intermediate_mom") -> pd.DataFrame:
    annual = rob_stock.load_annual(key, start, end)
    price = rob_stock.load_price(key, annual, start, end)
    benchmark = baw.load_benchmark(key, start, end)
    library = lab.raw_signal_library(price, benchmark)
    active = lab.active_mask(annual, price.columns, price.index)
    forecast, _rule_table = lab.combine_forecast_set(library, lab.FORECAST_SETS[forecast_set], active)
    return forecast


def combined_forecast(rob_forecast: pd.DataFrame, neural_forecast: pd.DataFrame, neural_weight: float = 0.20) -> pd.DataFrame:
    aligned_neural = neural_forecast.reindex(index=rob_forecast.index, columns=rob_forecast.columns)
    return ((1.0 - neural_weight) * rob_forecast + neural_weight * aligned_neural).clip(-20.0, 20.0)


def run_forecast_backtest(
    key: str,
    forecast: pd.DataFrame,
    label: str,
    start: str,
    end: str,
    out_dir: Path,
    *,
    capital: float = 500_000.0,
    vol_target: float = 0.10,
    weight_mode: str = "sector_equal",
    idm_method: str = "rob_estimated",
    idm: float = 2.75,
    cost_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    annual = rob_stock.load_annual(key, start, end)
    price = rob_stock.load_price(key, annual, start, end)
    price_vol = rob_stock.rob.mixed_vol(price.diff())
    forecast = forecast.reindex(index=price.index, columns=price.columns)
    if idm_method == "rob_estimated":
        pre_idm_weights = tail_lab.instrument_weights_for_forecast(price, price_vol, forecast, annual, weight_mode)
        idm_value, raw_idm = tail_lab.estimate_rob_idm(price, price_vol, pre_idm_weights)
        pd.concat([raw_idm, idm_value], axis=1).to_csv(out_dir / f"{key}_{label}_idm.csv")
    else:
        idm_value = idm

    positions, _target, _instrument_weights, risk = rob_stock.target_positions(
        price,
        price_vol,
        forecast,
        annual,
        weight_mode,
        capital=capital,
        vol_target=vol_target,
        idm=idm_value,
    )
    daily, _by_instrument = rob_stock.pnl_from_stock_positions(
        positions,
        price,
        capital,
        rob_stock.DEFAULT_COST_PER_DOLLAR * cost_multiplier,
    )
    daily = daily.join(risk).loc[start:end]
    daily = rob_stock.trim_active_daily(daily)
    daily.to_csv(out_dir / f"{key}_{label}_daily.csv")
    stats = rob_stock.performance_stats_from_equity(daily["daily_return"], daily["equity"] / capital)
    stats.update(
        {
            "avg_turnover_annual": daily["turnover"].mean() * 252.0,
            "avg_cost_annual": daily["costs"].mean() * 252.0 / capital,
            "avg_gross_exposure": daily["gross_exposure"].mean(),
            "avg_net_exposure": daily["net_exposure"].mean(),
            "avg_idm": daily["idm"].mean() if "idm" in daily else float("nan"),
        }
    )
    stats = {"market": key, "strategy": label, "cost_multiplier": cost_multiplier, **stats}
    return daily, stats


def forecast_long_from_matrix(matrix: pd.DataFrame, key: str, market: str, model: str) -> pd.DataFrame:
    frame = matrix.stack().rename("forecast").reset_index()
    frame.columns = ["Date", "Stock", "forecast"]
    frame["MarketKey"] = key
    frame["Market"] = market
    frame["model"] = model
    return frame


def save_forecast_matrix(
    predictions: pd.DataFrame,
    out_dir: Path,
    model: str,
    *,
    forecast_mode: str = "dense_rank",
    selection_frac: float = 0.10,
    selected_forecast: float = 10.0,
    score_smoothing_span: int = 1,
    selection_interval: str = "daily",
    exit_frac: float | None = None,
) -> None:
    converted = raw_alpha_to_forecast(
        predictions[predictions["model"].eq(model)],
        forecast_mode=forecast_mode,
        selection_frac=selection_frac,
        selected_forecast=selected_forecast,
        score_smoothing_span=score_smoothing_span,
        selection_interval=selection_interval,
        exit_frac=exit_frac,
    )
    for key, frame in converted.groupby("MarketKey"):
        if frame.duplicated(["Date", "Stock"]).any():
            frame = frame.groupby(["Date", "Stock"], as_index=False, dropna=False)["forecast"].mean()
        matrix = frame.pivot(index="Date", columns="Stock", values="forecast").sort_index()
        matrix.to_csv(out_dir / f"{key}_{model}_forecast.csv")
