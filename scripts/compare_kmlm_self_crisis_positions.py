#!/usr/bin/env python3
"""Compare self-built futures systems with KMLM-like positions in crisis phases."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

import crisis_future_bets_breakdown as cb
import run_kmlm_like_backtest as kmlm
import run_rob_style_backtest as bt


OUT = bt.ROOT / "backtests" / "kmlm_vs_self_crisis_position_analysis"
CAPITAL = 500_000.0

PHASES = {
    "Dotcom 2000": ("2000-03-24", "2000-12-29"),
    "Dotcom 2001": ("2001-01-02", "2001-12-31"),
    "Dotcom 2002": ("2002-01-02", "2002-10-09"),
    "2008 H1": ("2008-01-02", "2008-06-30"),
    "2008 H2": ("2008-07-01", "2008-12-31"),
    "2022 H1": ("2022-01-03", "2022-06-30"),
    "2022 H2": ("2022-07-01", "2022-12-30"),
}

TOTAL_WINDOWS = {
    "Dotcom Total": ("2000-03-24", "2002-10-09"),
    "2008 Calendar": ("2008-01-02", "2008-12-31"),
    "2022 Calendar": ("2022-01-03", "2022-12-30"),
}

SELF_STRATEGIES = {
    "Self 17": {
        "module": cb.selected17,
        "config_fn": cb.selected17.config_with_custom_weights,
    },
    "Self 40": {
        "module": cb.no40,
        "config_fn": cb.no40.config_with_no_equity_weights,
    },
}

SECTOR_MAP = {
    "Bond": "Fixed Income",
    "FX": "Currency",
    "Ags": "Commodity",
    "OilGas": "Commodity",
    "Metals": "Commodity",
    "Vol": "Vol",
}

SECTORS = ["Fixed Income", "Currency", "Commodity", "Vol"]
STRATEGY_ORDER = ["Self 17", "Self 40", "KMLM-like"]
COLORS = {
    "Self 17": "#1f4e79",
    "Self 40": "#6f8f2f",
    "KMLM-like": "#8a3f73",
}


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def bet_direction(value: float) -> str:
    if value > 0.01:
        return "net long"
    if value < -0.01:
        return "net short"
    return "near flat"


def traded_price_change(instrument: str, start: str, end: str) -> float:
    return cb.traded_price_change(instrument, start, end)


def self_strategy_results() -> dict[str, dict]:
    out = {}
    for name, spec in SELF_STRATEGIES.items():
        result = cb.run_strategy(spec["module"], spec["config_fn"])
        instruments = result["instruments"]
        meta = result["meta"]
        sector_by_instrument = {
            instrument: SECTOR_MAP.get(meta[instrument].asset_class, meta[instrument].asset_class)
            for instrument in instruments
        }
        unit_ann_risk_pct = (
            result["unit_daily_cash_vol"]
            * math.sqrt(bt.BUSINESS_DAYS)
            / float(result["config"]["notional_trading_capital"])
        )
        out[name] = {
            **result,
            "sector_by_instrument": sector_by_instrument,
            "signed_risk": result["positions"] * unit_ann_risk_pct,
            "gross_risk": result["positions"].abs() * unit_ann_risk_pct,
        }
    return out


def kmlm_result() -> dict:
    price, traded_price, results, signal, sector_weights, _returns_1x = kmlm.run_backtests()
    result = results["KMLM-like rolling 15V"]
    meta = bt.load_meta()
    instruments = kmlm.instruments()
    fx = bt.load_fx_matrix(instruments, meta, price.index)
    price_vol = bt.mixed_vol(price.diff())
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    unit_daily_cash_vol = price_vol.abs().mul(point_sizes, axis=1) * fx
    unit_ann_risk_pct = unit_daily_cash_vol * math.sqrt(bt.BUSINESS_DAYS) / CAPITAL
    sector_by_instrument = kmlm.group_for_instrument()
    return {
        "instruments": instruments,
        "price": price,
        "traded_price": traded_price,
        "positions": result.positions,
        "daily": result.daily,
        "by_instrument": result.by_instrument,
        "sector_by_instrument": sector_by_instrument,
        "signed_risk": result.positions * unit_ann_risk_pct.reindex(result.positions.index),
        "gross_risk": result.positions.abs() * unit_ann_risk_pct.reindex(result.positions.index),
    }


def price_change_for_strategy(strategy: str, data: dict, instrument: str, start: str, end: str) -> float:
    if strategy == "KMLM-like":
        prices = data["traded_price"][instrument].loc[start:end].dropna()
        if len(prices) < 2 or prices.iloc[0] == 0:
            return np.nan
        return prices.iloc[-1] / prices.iloc[0] - 1.0
    return traded_price_change(instrument, start, end)


def build_phase_rows(all_results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    instrument_rows = []
    total_rows = []
    windows = {**TOTAL_WINDOWS, **PHASES}

    for strategy, data in all_results.items():
        instruments = data["instruments"]
        sector_by_instrument = data["sector_by_instrument"]
        by_instrument = data["by_instrument"]
        positions = data["positions"]
        signed_risk = data["signed_risk"]
        gross_risk = data["gross_risk"]

        for window, (start, end) in windows.items():
            daily = data["daily"].loc[start:end]
            total_rows.append(
                {
                    "window": window,
                    "strategy": strategy,
                    "net_pnl_usd": daily["net_pnl"].sum(),
                    "pnl_pct_500k": daily["net_pnl"].sum() / CAPITAL,
                    "gross_pnl_usd": daily["gross_pnl"].sum(),
                    "costs_usd": daily["costs"].sum(),
                    "avg_portfolio_gross_risk_pct_500k": gross_risk.loc[start:end].sum(axis=1).mean(),
                    "avg_portfolio_net_risk_pct_500k": signed_risk.loc[start:end].sum(axis=1).mean(),
                }
            )

            for instrument in instruments:
                sector = sector_by_instrument[instrument]
                price_move = price_change_for_strategy(strategy, data, instrument, start, end)
                pnl = by_instrument[("net_pnl", instrument)].loc[start:end].sum()
                pos = positions[instrument].loc[start:end].fillna(0.0)
                risk = signed_risk[instrument].loc[start:end].fillna(0.0)
                gross = gross_risk[instrument].loc[start:end].fillna(0.0)
                instrument_rows.append(
                    {
                        "window": window,
                        "strategy": strategy,
                        "sector": sector,
                        "instrument": instrument,
                        "futures_price_change": price_move,
                        "avg_position_contracts": pos.mean(),
                        "pct_days_long": (pos > 0.0).mean(),
                        "pct_days_short": (pos < 0.0).mean(),
                        "avg_signed_risk_pct_500k": risk.mean(),
                        "avg_gross_risk_pct_500k": gross.mean(),
                        "net_pnl_usd": pnl,
                        "pnl_pct_500k": pnl / CAPITAL,
                    }
                )

            inst = pd.DataFrame(
                [
                    row
                    for row in instrument_rows
                    if row["window"] == window and row["strategy"] == strategy
                ]
            )
            for sector, group in inst.groupby("sector"):
                names = group["instrument"].tolist()
                class_signed = signed_risk[names].loc[start:end].sum(axis=1)
                class_gross = gross_risk[names].loc[start:end].sum(axis=1)
                class_positions = positions[names].loc[start:end].fillna(0.0)
                rows.append(
                    {
                        "window": window,
                        "strategy": strategy,
                        "sector": sector,
                        "instrument_count": len(names),
                        "median_futures_price_change": group["futures_price_change"].median(),
                        "mean_futures_price_change": group["futures_price_change"].mean(),
                        "avg_net_risk_pct_500k": class_signed.mean(),
                        "avg_gross_risk_pct_500k": class_gross.mean(),
                        "bet_direction": bet_direction(class_signed.mean()),
                        "avg_long_markets": (class_positions > 0.0).sum(axis=1).mean(),
                        "avg_short_markets": (class_positions < 0.0).sum(axis=1).mean(),
                        "net_pnl_usd": group["net_pnl_usd"].sum(),
                        "pnl_pct_500k": group["pnl_pct_500k"].sum(),
                    }
                )

    phase = pd.DataFrame(rows)
    instruments = pd.DataFrame(instrument_rows)
    totals = pd.DataFrame(total_rows)
    return phase, instruments, totals


def build_monthly_risk(all_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for strategy, data in all_results.items():
        signed = data["signed_risk"]
        gross = data["gross_risk"]
        sector_by_instrument = data["sector_by_instrument"]
        for sector in SECTORS:
            names = [instrument for instrument, bucket in sector_by_instrument.items() if bucket == sector]
            if not names:
                continue
            net_series = signed[names].sum(axis=1)
            gross_series = gross[names].sum(axis=1)
            monthly = pd.DataFrame({"net_risk": net_series, "gross_risk": gross_series}).resample("ME").mean()
            for date, row in monthly.dropna(how="all").iterrows():
                rows.append(
                    {
                        "date": date,
                        "strategy": strategy,
                        "sector": sector,
                        "avg_net_risk_pct_500k": row["net_risk"],
                        "avg_gross_risk_pct_500k": row["gross_risk"],
                    }
                )
    return pd.DataFrame(rows)


def phase_order(window: str) -> int:
    order = list(TOTAL_WINDOWS) + list(PHASES)
    return order.index(window) if window in order else 999


def plot_phase_comparison(phase: pd.DataFrame, totals: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phases = list(PHASES)
    plot_sectors = ["Fixed Income", "Currency", "Commodity"]

    fig = plt.figure(figsize=(18, 16), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 1.1], hspace=0.34, wspace=0.17)

    for col, sector in enumerate(plot_sectors):
        ax = fig.add_subplot(gs[0, col])
        subset = phase[phase["window"].isin(phases) & phase["sector"].eq(sector)]
        x = np.arange(len(phases))
        width = 0.22
        for offset, strategy in zip([-width, 0.0, width], STRATEGY_ORDER):
            values = [
                subset[subset["window"].eq(window) & subset["strategy"].eq(strategy)]["avg_net_risk_pct_500k"].sum()
                for window in phases
            ]
            ax.bar(x + offset, values, width=width, label=strategy, color=COLORS[strategy])
        ax.axhline(0, color="#8c96a0", lw=0.8)
        ax.set_title(f"{sector}: Avg Net Risk")
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        ax.grid(True, axis="y", color="#e7ebef", lw=0.8)
        if col == 0:
            ax.set_ylabel("Signed annual risk / 500k")
            ax.legend(frameon=False, loc="upper left")

    for col, sector in enumerate(plot_sectors):
        ax = fig.add_subplot(gs[1, col])
        subset = phase[phase["window"].isin(phases) & phase["sector"].eq(sector)]
        x = np.arange(len(phases))
        width = 0.22
        for offset, strategy in zip([-width, 0.0, width], STRATEGY_ORDER):
            values = [
                subset[subset["window"].eq(window) & subset["strategy"].eq(strategy)]["pnl_pct_500k"].sum()
                for window in phases
            ]
            ax.bar(x + offset, values, width=width, label=strategy, color=COLORS[strategy])
        ax.axhline(0, color="#8c96a0", lw=0.8)
        ax.set_title(f"{sector}: P&L Contribution")
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        ax.grid(True, axis="y", color="#e7ebef", lw=0.8)
        if col == 0:
            ax.set_ylabel("P&L / 500k")

    ax_total = fig.add_subplot(gs[2, :])
    x = np.arange(len(phases))
    width = 0.22
    for offset, strategy in zip([-width, 0.0, width], STRATEGY_ORDER):
        values = [
            totals[totals["window"].eq(window) & totals["strategy"].eq(strategy)]["pnl_pct_500k"].sum()
            for window in phases
        ]
        ax_total.bar(x + offset, values, width=width, label=strategy, color=COLORS[strategy])
    ax_total.axhline(0, color="#8c96a0", lw=0.8)
    ax_total.set_title("Total Phase P&L")
    ax_total.set_xticks(x)
    ax_total.set_xticklabels(phases, rotation=0)
    ax_total.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax_total.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_total.set_ylabel("P&L / 500k")
    ax_total.legend(frameon=False, loc="upper left", ncol=3)
    fig.suptitle("Self-Built Futures Systems vs KMLM-like: Synchronous Crisis Position and P&L Comparison", y=0.995)
    fig.savefig(OUT / "phase_position_pnl_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(phase: pd.DataFrame, instruments: pd.DataFrame, totals: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase.to_csv(OUT / "phase_sector_comparison.csv", index=False)
    instruments.to_csv(OUT / "phase_instrument_comparison.csv", index=False)
    totals.to_csv(OUT / "phase_total_comparison.csv", index=False)

    lines = [
        "# KMLM-like vs Self-Built Systems: Crisis Position Analysis",
        "",
        "Definitions:",
        "",
        "- `Avg net risk` is average signed annualised risk divided by the original USD 500k capital. Positive means net long; negative means net short.",
        "- `Avg gross risk` is average absolute annualised risk divided by USD 500k.",
        "- Futures price change uses the local `multiple_prices_csv` PRICE series, so it is a rolled futures price change, not a single contract total return.",
        "- KMLM-like is the local 22-market, monthly 12-month moving-average, rolling 15V replication, not official KMLM ETF history.",
        "",
        "## Total Phase P&L",
        "",
        "| Window | Self 17 | Self 40 | KMLM-like |",
        "|---|---:|---:|---:|",
    ]
    for window in [*TOTAL_WINDOWS, *PHASES]:
        row = totals[totals["window"].eq(window)].set_index("strategy")
        lines.append(
            "| "
            + " | ".join(
                [
                    window,
                    pct(row.loc["Self 17", "pnl_pct_500k"]),
                    pct(row.loc["Self 40", "pnl_pct_500k"]),
                    pct(row.loc["KMLM-like", "pnl_pct_500k"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Sector Phase Comparison", ""])
    for window in PHASES:
        lines.extend(
            [
                f"### {window}",
                "",
                "| Sector | Strategy | Futures move | Avg net risk | Avg gross risk | Bet | P&L / 500k |",
                "|---|---|---:|---:|---:|---|---:|",
            ]
        )
        rows = phase[phase["window"].eq(window) & phase["sector"].isin(["Fixed Income", "Currency", "Commodity"])].copy()
        rows["sector_order"] = rows["sector"].map({"Fixed Income": 0, "Currency": 1, "Commodity": 2})
        rows["strategy_order"] = rows["strategy"].map({strategy: idx for idx, strategy in enumerate(STRATEGY_ORDER)})
        rows = rows.sort_values(["sector_order", "strategy_order"])
        for row in rows.to_dict("records"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["sector"],
                        row["strategy"],
                        pct(row["median_futures_price_change"]),
                        pct(row["avg_net_risk_pct_500k"]),
                        pct(row["avg_gross_risk_pct_500k"]),
                        row["bet_direction"],
                        pct(row["pnl_pct_500k"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    (OUT / "position_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def print_key_tables(phase: pd.DataFrame, totals: pd.DataFrame) -> None:
    total_view = totals[totals["window"].isin([*TOTAL_WINDOWS, *PHASES])].pivot(
        index="window", columns="strategy", values="pnl_pct_500k"
    )
    total_view = total_view.loc[[*TOTAL_WINDOWS, *PHASES], STRATEGY_ORDER]
    print("\nTotal P&L / 500k")
    print(total_view.to_string(formatters={strategy: pct for strategy in STRATEGY_ORDER}))

    for window in ["Dotcom 2002", "2008 H1", "2008 H2", "2022 H1", "2022 H2"]:
        print(f"\n{window}: sector position and P&L")
        subset = phase[
            phase["window"].eq(window)
            & phase["sector"].isin(["Fixed Income", "Currency", "Commodity"])
        ].copy()
        subset["sector_order"] = subset["sector"].map({"Fixed Income": 0, "Currency": 1, "Commodity": 2})
        subset["strategy_order"] = subset["strategy"].map({strategy: idx for idx, strategy in enumerate(STRATEGY_ORDER)})
        subset = subset.sort_values(["sector_order", "strategy_order"])
        print(
            subset[
                [
                    "sector",
                    "strategy",
                    "median_futures_price_change",
                    "avg_net_risk_pct_500k",
                    "avg_gross_risk_pct_500k",
                    "bet_direction",
                    "pnl_pct_500k",
                ]
            ].to_string(
                index=False,
                formatters={
                    "median_futures_price_change": pct,
                    "avg_net_risk_pct_500k": pct,
                    "avg_gross_risk_pct_500k": pct,
                    "pnl_pct_500k": pct,
                },
            )
        )


def main() -> None:
    all_results = self_strategy_results()
    all_results["KMLM-like"] = kmlm_result()
    phase, instruments, totals = build_phase_rows(all_results)
    monthly = build_monthly_risk(all_results)
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / "monthly_sector_risk.csv", index=False)
    plot_phase_comparison(phase, totals)
    write_report(phase, instruments, totals)
    print(f"Wrote crisis position analysis to {OUT}")
    print_key_tables(phase, totals)


if __name__ == "__main__":
    main()
