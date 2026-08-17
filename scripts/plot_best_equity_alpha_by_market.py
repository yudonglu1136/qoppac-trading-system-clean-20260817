#!/usr/bin/env python3
"""Plot the best strict-gate equity alpha diagnostic portfolio by market."""

from __future__ import annotations

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

import run_equity_alpha_signal_lab as lab  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402


OUT = ROOT / "research" / "equity_alpha" / "stage_02_04_signal_lab"
GATE_PATH = OUT / "horizon_gate.csv"
MATRIX_PATH = OUT / "signal_market_horizon_matrix.csv"
PLOT_PATH = OUT / "best_by_market_vs_benchmark.png"
SUMMARY_PATH = OUT / "best_by_market_vs_benchmark_summary.csv"


def performance_stats(returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "start": "",
            "end": "",
            "cagr": math.nan,
            "vol": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
        }
    equity = (1.0 + returns).cumprod()
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1.0 / lab.BUSINESS_DAYS)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(returns.std() * math.sqrt(lab.BUSINESS_DAYS))
    drawdown = equity / equity.cummax() - 1.0
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "cagr": cagr,
        "vol": vol,
        "sharpe": float(returns.mean() * lab.BUSINESS_DAYS / vol) if vol else math.nan,
        "max_drawdown": float(drawdown.min()),
    }


def load_best_rows() -> pd.DataFrame:
    gates = pd.read_csv(GATE_PATH)
    matrix = pd.read_csv(MATRIX_PATH)
    candidates = gates[gates["gate_status"].eq("PASS_CANDIDATE")].merge(
        matrix,
        on=["universe", "universe_key", "signal", "horizon"],
        how="left",
    )
    if candidates.empty:
        raise RuntimeError("No strict PASS_CANDIDATE rows found.")
    sort_cols = ["universe_key", "q5_q1_net_spread_ann", "mean_rank_ic"]
    best = candidates.sort_values(sort_cols, ascending=[True, False, False]).groupby("universe_key").head(1)
    return best.sort_values("universe_key").reset_index(drop=True)


def build_q5_q1_returns(key: str, signal_name: str, horizon: int) -> tuple[pd.Series, pd.Series, str, pd.Series]:
    data = lab.load_universe(key)
    signals, _dictionary = lab.build_signal_library(data)
    signal = signals[signal_name]
    ranks = lab.rank_frame(signal, data.active)

    top = ranks.gt(0.8)
    bottom = ranks.le(0.2)
    long_weights = top.div(top.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    short_weights = bottom.div(bottom.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    formation_weights = (long_weights - short_weights).where(ranks.notna()).fillna(0.0)
    held_weights = formation_weights.shift(1).rolling(horizon, min_periods=1).mean().fillna(0.0)

    stock_returns = data.price.pct_change(fill_method=None).mask(lambda frame: frame.abs() > rob_stock.MAX_ABS_DAILY_RETURN)
    gross = held_weights.mul(stock_returns.fillna(0.0)).sum(axis=1)
    turnover = held_weights.diff().abs().sum(axis=1).fillna(held_weights.abs().sum(axis=1))
    net = gross - turnover * lab.COST_PER_DOLLAR

    benchmark = data.benchmark_price.pct_change(fill_method=None).rename("benchmark")
    gross_exposure = held_weights.abs().sum(axis=1)
    active_dates = gross_exposure.gt(0.0)
    if active_dates.any():
        start = active_dates[active_dates].index[0]
        net = net.loc[start:]
        benchmark = benchmark.loc[start:]
        gross_exposure = gross_exposure.loc[start:]
    common = net.loc[lab.START : lab.DEV_END].index.intersection(benchmark.loc[lab.START : lab.DEV_END].dropna().index)
    return (
        net.reindex(common).rename(signal_name),
        benchmark.reindex(common),
        data.label,
        gross_exposure.reindex(common).rename("gross_exposure"),
    )


def plot(best: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(len(best), 1, figsize=(15, 12), sharex=False)
    if len(best) == 1:
        axes = [axes]

    summary_rows: list[dict[str, object]] = []
    for ax, row in zip(axes, best.itertuples(index=False)):
        strategy_returns, benchmark_returns, label, gross_exposure = build_q5_q1_returns(
            row.universe_key,
            row.signal,
            int(row.horizon),
        )
        strategy_equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()
        benchmark_equity = (1.0 + benchmark_returns.fillna(0.0)).cumprod()

        strategy_stats = performance_stats(strategy_returns)
        benchmark_stats = performance_stats(benchmark_returns)
        summary_rows.append(
            {
                "universe_key": row.universe_key,
                "universe": label,
                "best_signal": row.signal,
                "research_horizon": int(row.horizon),
                "rank_ic": float(row.mean_rank_ic),
                "diagnostic_net_spread_ann": float(row.q5_q1_net_spread_ann),
                "average_gross_exposure": float(gross_exposure.mean()),
                **{f"strategy_{key}": value for key, value in strategy_stats.items()},
                **{f"benchmark_{key}": value for key, value in benchmark_stats.items()},
            }
        )

        ax.plot(strategy_equity.index, strategy_equity, color="#0F766E", lw=2.2, label=f"{row.signal} Q5-Q1 net")
        ax.plot(benchmark_equity.index, benchmark_equity, color="#334155", lw=1.8, label="Benchmark")
        ax.axhline(1.0, color="#CBD5E1", lw=0.8)
        ax.grid(True, color="#E2E8F0", lw=0.8)
        ax.set_title(
            f"{label}: best strict-gate signal = {row.signal}, research horizon {int(row.horizon)}D",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Growth of $1")
        ax.legend(loc="upper left", frameon=False)
        text = (
            f"Strategy CAGR {strategy_stats['cagr']:.1%}, Sharpe {strategy_stats['sharpe']:.2f}, "
            f"MDD {strategy_stats['max_drawdown']:.1%}, gross {gross_exposure.mean():.1f}x\n"
            f"Benchmark CAGR {benchmark_stats['cagr']:.1%}, Sharpe {benchmark_stats['sharpe']:.2f}, "
            f"MDD {benchmark_stats['max_drawdown']:.1%}"
        )
        ax.text(
            0.99,
            0.04,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.9},
        )

    fig.suptitle(
        "Best Strict-Gate Cross-Sectional Alpha by Market vs Benchmark, Development Period Only",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.01,
        0.01,
        "Diagnostic portfolio: daily Q5-Q1 equal-weight long-short, one-day execution lag, rolling sleeves held for the selected research horizon, base transaction cost. Not a final Rob-integrated strategy.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main() -> None:
    best = load_best_rows()
    summary = plot(best)
    print(PLOT_PATH)
    print(SUMMARY_PATH)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
