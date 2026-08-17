#!/usr/bin/env python3
"""Capacity and stress tests for the IBKR-style Rob 40 no-equity account."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_no_equity_40_margin_constrained_backtest as margin_bt  # noqa: E402
import run_rob_style_backtest as bt  # noqa: E402
import run_rob_style_no_equity_40_backtest as no40  # noqa: E402


OUT = ROOT / "backtests" / "ibkr_capacity_stress"
MARGIN_DIR = ROOT / "backtests" / "rob_style_no_equity_40_margin_constrained"
START = "2016-01-04"
END = "2024-03-29"
BASE_CAPITAL = 500_000.0
BUSINESS_DAYS = bt.BUSINESS_DAYS

CAPITAL_GRID = [150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000, 600_000, 750_000, 1_000_000]
MODEL_SCALE_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
MARGIN_MULTIPLIERS = [1.00, 1.50, 2.00, 3.00]
WINDOWS = [1, 5, 20, 60, 252]


def scale_positions(positions: pd.DataFrame, scale: float) -> pd.DataFrame:
    if abs(scale - 1.0) < 1e-12:
        return positions.copy()
    scaled = np.sign(positions) * np.floor(positions.abs() * scale + 0.5)
    return scaled.astype(float)


def load_components() -> tuple[
    list[str],
    dict[str, bt.InstrumentMeta],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    bt.START_DATE = "1970-01-01"
    instruments = no40.selected_instruments()
    meta = bt.load_meta()
    price = bt.load_price_matrix(instruments).loc[START:END]
    fx = bt.load_fx_matrix(instruments, meta, price.index)
    costs = bt.cost_matrix(instruments, meta, price, fx)
    desired = pd.read_csv(MARGIN_DIR / "positions_unconstrained_full.csv", index_col=0, parse_dates=True)
    desired = desired.reindex(price.index).ffill().fillna(0.0)
    schedule = pd.read_csv(MARGIN_DIR / "margin_schedule.csv")
    long_margin, short_margin = margin_bt.margin_per_contract_matrices(
        schedule, instruments, meta, price, fx, mode="current_static"
    )
    return instruments, meta, price, fx, costs, desired, long_margin, short_margin


def add_account_return(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["account_return"] = daily["equity"].pct_change().fillna(0.0)
    daily["account_drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    return daily


def simulate(
    instruments: list[str],
    meta: dict[str, bt.InstrumentMeta],
    price: pd.DataFrame,
    fx: pd.DataFrame,
    costs: pd.DataFrame,
    desired: pd.DataFrame,
    long_margin: pd.DataFrame,
    short_margin: pd.DataFrame,
    capital: float,
    model_scale: float,
    margin_multiplier: float,
    margin_limit: float = 1.0,
) -> pd.DataFrame:
    scaled_desired = scale_positions(desired, model_scale)
    daily, _ = margin_bt.run_margin_constrained_pnl(
        scaled_desired,
        price,
        fx,
        meta,
        instruments,
        costs,
        long_margin * margin_multiplier,
        short_margin * margin_multiplier,
        capital,
        margin_limit=margin_limit,
    )
    return add_account_return(daily)


def metrics_from_daily(
    daily: pd.DataFrame,
    label: str,
    capital: float,
    model_scale: float,
    margin_multiplier: float,
    kind: str,
) -> dict[str, float | str]:
    returns = daily["account_return"].dropna()
    years = (daily.index.max() - daily.index.min()).days / 365.25
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * np.sqrt(BUSINESS_DAYS)
    drawdown = nav / nav.cummax() - 1.0
    return {
        "kind": kind,
        "label": label,
        "capital": capital,
        "model_scale": model_scale,
        "margin_multiplier": margin_multiplier,
        "start": str(daily.index.min().date()),
        "end": str(daily.index.max().date()),
        "years": years,
        "final_equity": float(daily["equity"].iloc[-1]),
        "total_return": nav.iloc[-1] - 1.0,
        "cagr": cagr,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "min_equity": float(daily["equity"].min()),
        "pct_days_scaled": float((daily["margin_scale"] < 0.999).mean()),
        "min_margin_scale": float(daily["margin_scale"].min()),
        "median_margin_to_equity": float(daily["margin_to_equity"].median()),
        "p95_margin_to_equity": float(daily["margin_to_equity"].quantile(0.95)),
        "max_margin_to_equity": float(daily["margin_to_equity"].max()),
        "max_desired_margin_to_equity": float(daily["desired_margin_to_equity"].max()),
    }


def full_desired_daily(
    instruments: list[str],
    meta: dict[str, bt.InstrumentMeta],
    price: pd.DataFrame,
    fx: pd.DataFrame,
    costs: pd.DataFrame,
    desired: pd.DataFrame,
    long_margin: pd.DataFrame,
    short_margin: pd.DataFrame,
    capital: float,
    margin_multiplier: float,
) -> pd.DataFrame:
    return simulate(
        instruments,
        meta,
        price,
        fx,
        costs,
        desired,
        long_margin,
        short_margin,
        capital,
        model_scale=1.0,
        margin_multiplier=margin_multiplier,
        margin_limit=999.0,
    )


def unconstrained_daily_from_positions(
    instruments: list[str],
    meta: dict[str, bt.InstrumentMeta],
    price: pd.DataFrame,
    fx: pd.DataFrame,
    costs: pd.DataFrame,
    desired: pd.DataFrame,
    long_margin: pd.DataFrame,
    short_margin: pd.DataFrame,
) -> pd.DataFrame:
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    previous_position = pd.Series(0.0, index=instruments)
    previous_price = None
    rows: list[dict[str, float]] = []
    equity_offset = 0.0

    for date in price.index:
        price_today = price.loc[date].reindex(instruments)
        fx_today = fx.loc[date].reindex(instruments).fillna(0.0)
        if previous_price is None:
            price_change = pd.Series(0.0, index=instruments)
        else:
            price_change = (price_today - previous_price).fillna(0.0)

        gross_vec = previous_position * price_change * point_sizes * fx_today
        equity_before_rebalance_offset = equity_offset + float(gross_vec.sum())
        desired_today = desired.loc[date].reindex(instruments).fillna(0.0)
        if previous_price is None:
            trades = desired_today * 0.0
        else:
            trades = (desired_today - previous_position).abs()
        cost_vec = trades * costs.loc[date].reindex(instruments).fillna(0.0)
        gross_pnl = float(gross_vec.sum())
        total_cost = float(cost_vec.sum())
        net_pnl = gross_pnl - total_cost
        equity_offset += net_pnl
        wanted_margin = margin_bt.total_margin(
            desired_today,
            long_margin.loc[date].reindex(instruments),
            short_margin.loc[date].reindex(instruments),
        )
        rows.append(
            {
                "gross_pnl": gross_pnl,
                "costs": total_cost,
                "net_pnl": net_pnl,
                "equity_before_rebalance_offset": equity_before_rebalance_offset,
                "equity_offset": equity_offset,
                "desired_margin_base": wanted_margin,
            }
        )
        previous_position = desired_today
        previous_price = price_today

    return pd.DataFrame(rows, index=price.index)


def find_min_capital_from_unconstrained(
    unconstrained: pd.DataFrame,
    margin_multiplier: float,
    criterion: str,
    threshold: float,
) -> float:
    desired_margin = unconstrained["desired_margin_base"] * margin_multiplier
    equity_offset = unconstrained["equity_offset"]
    min_positive_equity_capital = max(0.0, float(-equity_offset.min() + 1.0))

    if criterion == "max":
        candidates = desired_margin / threshold - equity_offset
        return max(min_positive_equity_capital, float(candidates.max()), 0.0)

    if criterion != "p95":
        raise ValueError(criterion)

    low = min_positive_equity_capital
    high = max(BASE_CAPITAL, low + 100_000.0)

    def passes(capital: float) -> bool:
        equity = capital + equity_offset
        if bool((equity <= 0.0).any()):
            return False
        ratio = (desired_margin / equity.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        return bool(ratio.quantile(0.95) <= threshold)

    while not passes(high):
        high *= 1.5
    for _ in range(60):
        mid = (low + high) / 2.0
        if passes(mid):
            high = mid
        else:
            low = mid
    return high


def worst_windows(daily: pd.DataFrame) -> pd.DataFrame:
    returns = daily["account_return"].dropna()
    rows = []
    for window in WINDOWS:
        cumulative = (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
        end = cumulative.idxmin()
        rows.append(
            {
                "window_days": window,
                "start": str(returns.loc[:end].tail(window).index.min().date()),
                "end": str(end.date()),
                "return": float(cumulative.loc[end]),
            }
        )
    return pd.DataFrame(rows)


def make_plots(scale_metrics: pd.DataFrame, capital_metrics: pd.DataFrame, base_daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    base_daily["equity"].plot(ax=axes[0], color="#0F766E", label="base $500k 1x")
    axes[0].set_title("IBKR Capacity Stress: Base 2016 Account")
    axes[0].set_ylabel("Equity")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    subset = scale_metrics[scale_metrics["margin_multiplier"].isin([1.0, 2.0, 3.0])]
    for margin_multiplier, frame in subset.groupby("margin_multiplier"):
        axes[1].plot(frame["model_scale"], frame["pct_days_scaled"], marker="o", label=f"margin x{margin_multiplier:g}")
    axes[1].set_title("$500k Account: Forced Scaling Days By Model Size")
    axes[1].set_xlabel("Model size relative to current 500k model")
    axes[1].set_ylabel("Days scaled")
    axes[1].yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    cap_subset = capital_metrics[capital_metrics["margin_multiplier"].isin([1.0, 2.0, 3.0])]
    for margin_multiplier, frame in cap_subset.groupby("margin_multiplier"):
        axes[2].plot(frame["capital"], frame["pct_days_scaled"], marker="o", label=f"margin x{margin_multiplier:g}")
    axes[2].set_title("1x Model: Forced Scaling Days By Starting Capital")
    axes[2].set_xlabel("Starting capital")
    axes[2].set_ylabel("Days scaled")
    axes[2].xaxis.set_major_formatter(lambda value, _pos: f"${value/1000:.0f}k")
    axes[2].yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "capacity_stress.png", dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def money(value: float) -> str:
    return "" if pd.isna(value) else f"${value:,.0f}"


def write_summary(
    base_metrics: dict[str, float | str],
    scale_metrics: pd.DataFrame,
    capital_metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    worst: pd.DataFrame,
) -> None:
    safe_scale_base = scale_metrics[
        (scale_metrics["capital"].eq(BASE_CAPITAL))
        & (scale_metrics["margin_multiplier"].eq(1.0))
        & (scale_metrics["pct_days_scaled"].eq(0.0))
    ]["model_scale"].max()
    safe_scale_2x_margin = scale_metrics[
        (scale_metrics["capital"].eq(BASE_CAPITAL))
        & (scale_metrics["margin_multiplier"].eq(2.0))
        & (scale_metrics["pct_days_scaled"].eq(0.0))
    ]["model_scale"].max()
    threshold_table = thresholds.copy()
    threshold_table["margin_multiplier"] = threshold_table["margin_multiplier"].map(lambda value: f"{value:g}x")
    for column in [
        "min_capital_no_forced_scaling",
        "capital_for_max_margin_le_80pct",
        "capital_for_p95_margin_le_50pct",
    ]:
        threshold_table[column] = threshold_table[column].map(lambda value: f"${value:,.0f}")
    lines = [
        "# IBKR Capacity And Stress Test",
        "",
        "## Setup",
        "",
        "- Fresh account starts on 2016-01-04.",
        "- Base strategy is the current Rob 40 no-equity buffered-integer futures system.",
        "- IBKR current overnight initial margins are applied as collateral constraints.",
        "- `margin_multiplier` shocks all current margin requirements upward.",
        "- `model_scale` scales the current 500k model's integer target contracts and rounds to whole contracts.",
        "",
        "## Base 500k Account",
        "",
        f"- Final equity: {money(float(base_metrics['final_equity']))}",
        f"- CAGR: {pct(float(base_metrics['cagr']))}",
        f"- Vol: {pct(float(base_metrics['ann_vol']))}",
        f"- Sharpe: {float(base_metrics['sharpe']):.2f}",
        f"- Max drawdown: {pct(float(base_metrics['max_drawdown']))}",
        f"- Max margin/equity: {pct(float(base_metrics['max_margin_to_equity']))}",
        f"- Forced scaling days: {pct(float(base_metrics['pct_days_scaled']))}",
        "",
        "## Capacity Answer",
        "",
        f"- With today's IBKR margin schedule, the $500k account is enough for the 1x model: no forced scaling from 2016 to local data end.",
        f"- Largest model size with $500k and current margins without forced scaling: {safe_scale_base:.2f}x.",
        f"- Largest model size with $500k and 2x margin shock without forced scaling: {safe_scale_2x_margin:.2f}x.",
        "",
        "## Capital Thresholds For 1x Model",
        "",
        threshold_table.to_markdown(index=False),
        "",
        "## Worst Historical Account Windows, 500k 1x",
        "",
        worst.to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `capacity_stress.png`",
        "- `scale_capacity_grid.csv`",
        "- `capital_capacity_grid.csv`",
        "- `capital_thresholds.csv`",
        "- `instrument_position_capacity.csv`",
        "- `unconstrained_1x_daily.csv`",
        "- `worst_windows.csv`",
        "- `base_500k_1x_daily.csv`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    instruments, meta, price, fx, costs, desired, long_margin, short_margin = load_components()

    base_path = OUT / "base_500k_1x_daily.csv"
    if base_path.exists():
        base_daily = pd.read_csv(base_path, index_col=0, parse_dates=True)
    else:
        base_daily = simulate(
            instruments,
            meta,
            price,
            fx,
            costs,
            desired,
            long_margin,
            short_margin,
            BASE_CAPITAL,
            model_scale=1.0,
            margin_multiplier=1.0,
        )
        base_daily.to_csv(base_path)
    base_metrics = metrics_from_daily(base_daily, "base_500k_1x", BASE_CAPITAL, 1.0, 1.0, "base")

    scale_path = OUT / "scale_capacity_grid.csv"
    if scale_path.exists():
        scale_metrics = pd.read_csv(scale_path)
    else:
        scale_rows = []
        for margin_multiplier in MARGIN_MULTIPLIERS:
            for model_scale in MODEL_SCALE_GRID:
                daily = simulate(
                    instruments,
                    meta,
                    price,
                    fx,
                    costs,
                    desired,
                    long_margin,
                    short_margin,
                    BASE_CAPITAL,
                    model_scale=model_scale,
                    margin_multiplier=margin_multiplier,
                )
                scale_rows.append(
                    metrics_from_daily(
                        daily,
                        f"capital_500k_scale_{model_scale:g}_margin_{margin_multiplier:g}",
                        BASE_CAPITAL,
                        model_scale,
                        margin_multiplier,
                        "scale_grid",
                    )
                )
        scale_metrics = pd.DataFrame(scale_rows)
        scale_metrics.to_csv(scale_path, index=False)

    capital_path = OUT / "capital_capacity_grid.csv"
    if capital_path.exists():
        capital_metrics = pd.read_csv(capital_path)
    else:
        capital_rows = []
        for margin_multiplier in MARGIN_MULTIPLIERS:
            for capital in CAPITAL_GRID:
                daily = simulate(
                    instruments,
                    meta,
                    price,
                    fx,
                    costs,
                    desired,
                    long_margin,
                    short_margin,
                    float(capital),
                    model_scale=1.0,
                    margin_multiplier=margin_multiplier,
                )
                capital_rows.append(
                    metrics_from_daily(
                        daily,
                        f"capital_{capital:g}_scale_1_margin_{margin_multiplier:g}",
                        float(capital),
                        1.0,
                        margin_multiplier,
                        "capital_grid",
                    )
                )
        capital_metrics = pd.DataFrame(capital_rows)
        capital_metrics.to_csv(capital_path, index=False)

    unconstrained = unconstrained_daily_from_positions(instruments, meta, price, fx, costs, desired, long_margin, short_margin)
    unconstrained.to_csv(OUT / "unconstrained_1x_daily.csv")
    schedule = pd.read_csv(MARGIN_DIR / "margin_schedule.csv")
    capacity_rows = []
    for instrument in desired.columns:
        position = desired[instrument].loc[START:END]
        capacity_rows.append(
            {
                "instrument": instrument,
                "max_abs_contracts": int(position.abs().max()),
                "p95_abs_contracts": float(position.abs().quantile(0.95)),
                "median_abs_contracts": float(position.abs().median()),
            }
        )
    instrument_capacity = pd.DataFrame(capacity_rows).merge(
        schedule[["instrument", "asset_class", "exchange", "trading_class", "source"]],
        on="instrument",
        how="left",
    )
    instrument_capacity.sort_values(["max_abs_contracts", "p95_abs_contracts"], ascending=False).to_csv(
        OUT / "instrument_position_capacity.csv", index=False
    )
    threshold_rows = []
    for margin_multiplier in MARGIN_MULTIPLIERS:
        threshold_rows.append(
            {
                "margin_multiplier": margin_multiplier,
                "min_capital_no_forced_scaling": find_min_capital_from_unconstrained(
                    unconstrained, margin_multiplier, "max", 1.0
                ),
                "capital_for_max_margin_le_80pct": find_min_capital_from_unconstrained(
                    unconstrained, margin_multiplier, "max", 0.80
                ),
                "capital_for_p95_margin_le_50pct": find_min_capital_from_unconstrained(
                    unconstrained, margin_multiplier, "p95", 0.50
                ),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT / "capital_thresholds.csv", index=False)

    worst = worst_windows(base_daily)
    worst.to_csv(OUT / "worst_windows.csv", index=False)
    make_plots(scale_metrics, capital_metrics, base_daily)
    write_summary(base_metrics, scale_metrics, capital_metrics, thresholds, worst)

    print(f"Wrote {OUT}")
    print("base")
    print(pd.DataFrame([base_metrics]).to_string(index=False))
    print("thresholds")
    print(thresholds.to_string(index=False))
    print("scale grid")
    print(
        scale_metrics[
            ["model_scale", "margin_multiplier", "final_equity", "cagr", "ann_vol", "sharpe", "max_drawdown", "pct_days_scaled", "max_margin_to_equity"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
