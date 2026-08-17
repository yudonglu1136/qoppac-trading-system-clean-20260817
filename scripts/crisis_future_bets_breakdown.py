#!/usr/bin/env python3
"""Break down futures moves, strategy bets, and P&L in crisis windows."""

from __future__ import annotations

import math

import pandas as pd

import run_rob_style_backtest as bt
import run_rob_style_no_equity_40_backtest as no40
import run_rob_style_us_rates_selected_no_vol_backtest as selected17


OUT = bt.ROOT / "backtests" / "crisis_future_bets_breakdown"
WINDOWS = {
    "Dot-com bear": ("2000-03-24", "2002-10-09"),
    "2008 calendar": ("2008-01-02", "2008-12-31"),
    "GFC peak-to-trough": ("2007-10-09", "2009-03-09"),
}
STRATEGIES = {
    "17 selected": {
        "module": selected17,
        "config_fn": selected17.config_with_custom_weights,
    },
    "40 no-equity": {
        "module": no40,
        "config_fn": no40.config_with_no_equity_weights,
    },
}
MULTIPLE_PRICE_DIR = (
    bt.ROOT / "github" / "pysystemtrade" / "data" / "futures" / "multiple_prices_csv"
)


def run_strategy(module, config_fn) -> dict:
    config = config_fn(bt.load_rob_config())
    meta = bt.load_meta()
    instruments = module.selected_instruments()

    price = bt.load_price_matrix(instruments)
    price_vol = bt.mixed_vol(price.diff())
    fx = bt.load_fx_matrix(instruments, meta, price.index)
    raw_carry = bt.load_raw_carry_matrix(instruments, price_vol, price.index)
    forecasts = bt.build_rule_forecasts(config, instruments, meta, price, price_vol, raw_carry)
    combined_forecast, _rule_weight_used = bt.combine_forecasts(forecasts, config, instruments, price.index)
    target, daily_weights, unit_daily_cash_vol = bt.initial_target_positions(
        config, instruments, meta, price, price_vol, fx, combined_forecast
    )
    risk_multiplier = bt.risk_multiplier_for_targets(config, target, unit_daily_cash_vol, price, fx, meta, instruments)
    target = target.mul(risk_multiplier, axis=0)
    costs = bt.cost_matrix(instruments, meta, price, fx)

    idm = float(config.get("instrument_div_multiplier", 1.0))
    subsystem_position = (
        float(config["notional_trading_capital"])
        * (float(config["percentage_vol_target"]) / 100.0)
        / (bt.BUSINESS_DAYS**0.5)
        / unit_daily_cash_vol
    )
    buffer = (subsystem_position.abs() * daily_weights * idm * bt.BUFFER_SIZE).mul(risk_multiplier, axis=0)
    buffered_target = bt.apply_position_buffer(target.fillna(0.0), buffer.fillna(0.0))
    integer_positions = bt.optimise_integer_positions(buffered_target, unit_daily_cash_vol, config)
    integer_daily, integer_by_instrument = bt.pnl_from_positions(
        integer_positions, price, fx, meta, instruments, costs
    )
    return {
        "config": config,
        "meta": meta,
        "instruments": instruments,
        "price": price,
        "unit_daily_cash_vol": unit_daily_cash_vol,
        "positions": integer_positions,
        "daily": integer_daily,
        "by_instrument": integer_by_instrument,
    }


def price_change(price: pd.Series, start: str, end: str) -> float:
    window = price.loc[start:end].dropna()
    if len(window) < 2 or window.iloc[0] == 0:
        return float("nan")
    return window.iloc[-1] / window.iloc[0] - 1.0


def traded_price_change(instrument: str, start: str, end: str) -> float:
    path = MULTIPLE_PRICE_DIR / f"{instrument}.csv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path, parse_dates=["DATETIME"]).set_index("DATETIME")
    if "PRICE" not in df:
        return float("nan")
    return price_change(df["PRICE"], start, end)


def classify_position(value: float) -> str:
    if value > 0.01:
        return "net long"
    if value < -0.01:
        return "net short"
    return "near flat"


def build_breakdowns() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asset_rows = []
    instrument_rows = []
    strategy_rows = []

    for strategy_name, spec in STRATEGIES.items():
        result = run_strategy(spec["module"], spec["config_fn"])
        meta = result["meta"]
        instruments = result["instruments"]
        price = result["price"]
        positions = result["positions"]
        by_instrument = result["by_instrument"]
        unit_ann_risk_pct = (
            result["unit_daily_cash_vol"] * math.sqrt(bt.BUSINESS_DAYS) / float(result["config"]["notional_trading_capital"])
        )
        signed_risk = positions * unit_ann_risk_pct
        gross_risk = positions.abs() * unit_ann_risk_pct

        for window_name, (start, end) in WINDOWS.items():
            daily = result["daily"].loc[start:end]
            total_pnl = daily["net_pnl"].sum()
            strategy_rows.append(
                {
                    "strategy": strategy_name,
                    "window": window_name,
                    "start": str(daily.index.min().date()),
                    "end": str(daily.index.max().date()),
                    "net_pnl_usd": total_pnl,
                    "pnl_pct_500k": total_pnl / float(result["config"]["notional_trading_capital"]),
                    "nav_return": daily["net_pnl"].cumsum().iloc[-1] / daily["equity"].iloc[0]
                    if "equity" in daily
                    else float("nan"),
                    "costs_usd": daily["costs"].sum(),
                }
            )

            for instrument in instruments:
                asset_class = meta[instrument].asset_class
                instr_positions = positions[instrument].loc[start:end].fillna(0.0)
                instr_net_pnl = by_instrument["net_pnl"][instrument].loc[start:end].sum()
                instr_costs = by_instrument["costs"][instrument].loc[start:end].sum()
                instr_price_change = traded_price_change(instrument, start, end)
                instr_adjusted_price_change = price_change(price[instrument], start, end)
                instr_signed_risk = signed_risk[instrument].loc[start:end].fillna(0.0)
                instrument_rows.append(
                    {
                        "strategy": strategy_name,
                        "window": window_name,
                        "asset_class": asset_class,
                        "instrument": instrument,
                        "futures_price_change": instr_price_change,
                        "adjusted_price_change": instr_adjusted_price_change,
                        "avg_position_contracts": instr_positions.mean(),
                        "pct_days_long": (instr_positions > 0).mean(),
                        "pct_days_short": (instr_positions < 0).mean(),
                        "avg_signed_risk_pct_capital": instr_signed_risk.mean(),
                        "avg_abs_risk_pct_capital": instr_signed_risk.abs().mean(),
                        "net_pnl_usd": instr_net_pnl,
                        "pnl_pct_500k": instr_net_pnl / float(result["config"]["notional_trading_capital"]),
                        "costs_usd": instr_costs,
                    }
                )

            instr_df = pd.DataFrame([row for row in instrument_rows if row["strategy"] == strategy_name and row["window"] == window_name])
            for asset_class, group in instr_df.groupby("asset_class"):
                names = group["instrument"].tolist()
                class_signed_risk = signed_risk[names].loc[start:end].sum(axis=1)
                class_gross_risk = gross_risk[names].loc[start:end].sum(axis=1)
                class_positions = positions[names].loc[start:end].fillna(0.0)
                class_pnl = by_instrument["net_pnl"][names].loc[start:end].sum().sum()
                class_costs = by_instrument["costs"][names].loc[start:end].sum().sum()
                asset_rows.append(
                    {
                        "strategy": strategy_name,
                        "window": window_name,
                        "asset_class": asset_class,
                        "instrument_count": len(names),
                        "median_futures_price_change": group["futures_price_change"].median(),
                        "mean_futures_price_change": group["futures_price_change"].mean(),
                        "median_adjusted_price_change": group["adjusted_price_change"].median(),
                        "mean_adjusted_price_change": group["adjusted_price_change"].mean(),
                        "avg_net_risk_pct_capital": class_signed_risk.mean(),
                        "avg_gross_risk_pct_capital": class_gross_risk.mean(),
                        "bet_direction": classify_position(class_signed_risk.mean()),
                        "avg_long_markets": (class_positions > 0).sum(axis=1).mean(),
                        "avg_short_markets": (class_positions < 0).sum(axis=1).mean(),
                        "net_pnl_usd": class_pnl,
                        "pnl_pct_500k": class_pnl / float(result["config"]["notional_trading_capital"]),
                        "costs_usd": class_costs,
                    }
                )

    return pd.DataFrame(asset_rows), pd.DataFrame(instrument_rows), pd.DataFrame(strategy_rows)


def write_report(asset: pd.DataFrame, instrument: pd.DataFrame, strategy: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    asset.to_csv(OUT / "asset_class_breakdown.csv", index=False)
    instrument.to_csv(OUT / "instrument_breakdown.csv", index=False)
    strategy.to_csv(OUT / "strategy_window_summary.csv", index=False)

    def pct(value: float) -> str:
        return f"{value:.1%}"

    def money(value: float) -> str:
        return f"${value:,.0f}"

    lines = [
        "# Crisis Futures Bets Breakdown",
        "",
        "Definitions:",
        "",
        "- Futures price change is the percent change in the local `multiple_prices_csv` PRICE column. It is closer to the traded contract price than the back-adjusted trend input, but still reflects a rolled futures series, not a single contract total return.",
        "- Avg net risk is average signed annualised contract risk divided by USD 500k capital. Positive means net long futures; negative means net short futures.",
        "- P&L is actual buffered-integer strategy P&L, after trading costs.",
        "",
    ]

    for window in WINDOWS:
        lines.extend(["", f"## {window}", ""])
        for strategy_name in STRATEGIES:
            srow = strategy[strategy["strategy"].eq(strategy_name) & strategy["window"].eq(window)].iloc[0]
            lines.extend(
                [
                    f"### {strategy_name}",
                    "",
                    f"- Net P&L: {money(srow['net_pnl_usd'])}, or {pct(srow['pnl_pct_500k'])} of USD 500k risk capital.",
                    f"- Costs: {money(srow['costs_usd'])}.",
                    "",
                    "| Asset | Futures move median | Futures move mean | Avg net risk | Bet | Avg long mkts | Avg short mkts | P&L | P&L / 500k |",
                    "|---|---:|---:|---:|---|---:|---:|---:|---:|",
                ]
            )
            rows = asset[asset["strategy"].eq(strategy_name) & asset["window"].eq(window)].sort_values(
                "pnl_pct_500k", ascending=False
            )
            for row in rows.to_dict("records"):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row["asset_class"],
                            pct(row["median_futures_price_change"]),
                            pct(row["mean_futures_price_change"]),
                            pct(row["avg_net_risk_pct_capital"]),
                            row["bet_direction"],
                            f"{row['avg_long_markets']:.1f}",
                            f"{row['avg_short_markets']:.1f}",
                            money(row["net_pnl_usd"]),
                            pct(row["pnl_pct_500k"]),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    (OUT / "crisis_future_bets_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    asset, instrument, strategy = build_breakdowns()
    write_report(asset, instrument, strategy)
    print(f"Wrote breakdowns to {OUT}")
    for window in ["Dot-com bear", "2008 calendar"]:
        print(f"\n## {window}")
        view = asset[asset["window"].eq(window)][
            [
                "strategy",
                "asset_class",
                "median_futures_price_change",
                "avg_net_risk_pct_capital",
                "bet_direction",
                "avg_long_markets",
                "avg_short_markets",
                "net_pnl_usd",
                "pnl_pct_500k",
            ]
        ].sort_values(["strategy", "pnl_pct_500k"], ascending=[True, False])
        print(
            view.to_string(
                index=False,
                formatters={
                    "median_futures_price_change": lambda x: f"{x:.1%}",
                    "avg_net_risk_pct_capital": lambda x: f"{x:.1%}",
                    "avg_long_markets": lambda x: f"{x:.1f}",
                    "avg_short_markets": lambda x: f"{x:.1f}",
                    "net_pnl_usd": lambda x: f"${x:,.0f}",
                    "pnl_pct_500k": lambda x: f"{x:.1%}",
                },
            )
        )


if __name__ == "__main__":
    main()
