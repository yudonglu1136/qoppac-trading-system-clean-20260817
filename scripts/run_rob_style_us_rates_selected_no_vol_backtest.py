#!/usr/bin/env python3
"""Run a concentrated no-vol Rob-style futures backtest requested by the user."""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import pandas as pd

import run_rob_style_backtest as bt


OUT = bt.ROOT / "backtests" / "rob_style_us_rates_selected_no_vol"

CUSTOM_UNIVERSE = {
    "Bond": ["US2", "US5", "US10", "US20", "SOFR"],
    "FX": ["AUD", "GBP", "JPY"],
    "Ags": ["CORN", "SOYBEAN_mini", "WHEAT", "LIVECOW"],
    "Metals": ["COPPER-micro", "GOLD_micro", "IRON"],
    "OilGas": ["CRUDE_W", "GASOILINE"],
}

# Start from the no-equity risk budget, remove Vol, and rescale the remaining
# buckets to 100%. This preserves the prior ex-ante asset-class mix.
BASE_NO_EQUITY_WEIGHTS_EX_VOL = {
    "Bond": 0.30,
    "FX": 0.20,
    "Ags": 0.20,
    "Metals": 0.10,
    "OilGas": 0.15,
}
WEIGHT_TOTAL = sum(BASE_NO_EQUITY_WEIGHTS_EX_VOL.values())
ASSET_CLASS_RISK_WEIGHTS = {
    asset_class: weight / WEIGHT_TOTAL for asset_class, weight in BASE_NO_EQUITY_WEIGHTS_EX_VOL.items()
}


def selected_instruments() -> list[str]:
    return [instrument for instruments in CUSTOM_UNIVERSE.values() for instrument in instruments]


def config_with_custom_weights(config: dict) -> dict:
    updated = copy.deepcopy(config)
    updated["instrument_weights"] = {
        instrument: ASSET_CLASS_RISK_WEIGHTS[asset_class] / len(instruments)
        for asset_class, instruments in CUSTOM_UNIVERSE.items()
        for instrument in instruments
    }
    return updated


def capacity_summary(
    instruments: list[str],
    meta: dict[str, bt.InstrumentMeta],
    price: pd.DataFrame,
    unit_daily_cash_vol: pd.DataFrame,
    daily_weights: pd.DataFrame,
    costs: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    daily_cash_vol_target = (
        float(config["notional_trading_capital"])
        * (float(config["percentage_vol_target"]) / 100.0)
        / (bt.BUSINESS_DAYS**0.5)
    )
    idm = float(config.get("instrument_div_multiplier", 1.0))
    forecast_10_contracts = daily_cash_vol_target / unit_daily_cash_vol * daily_weights * idm

    rows = []
    for instrument in instruments:
        valid_price = price[instrument].dropna()
        unit_ann_risk = unit_daily_cash_vol[instrument] * (bt.BUSINESS_DAYS**0.5)
        bucket = next(bucket for bucket, names in CUSTOM_UNIVERSE.items() if instrument in names)
        rows.append(
            {
                "instrument": instrument,
                "asset_class": meta[instrument].asset_class,
                "selection_bucket": bucket,
                "target_asset_weight": ASSET_CLASS_RISK_WEIGHTS[bucket],
                "instrument_weight": config["instrument_weights"][instrument],
                "first_price": valid_price.index.min(),
                "last_price": valid_price.index.max(),
                "median_contracts_at_forecast_10": forecast_10_contracts[instrument].median(),
                "p25_contracts_at_forecast_10": forecast_10_contracts[instrument].quantile(0.25),
                "median_contract_annual_risk_usd": unit_ann_risk.median(),
                "median_trade_cost_usd": costs[instrument].median(),
                "median_cost_as_pct_annual_risk": costs[instrument].median() / unit_ann_risk.median(),
                "nonzero_config_rules": sum(
                    float(weight) != 0.0 for weight in config["forecast_weights"].get(instrument, {}).values()
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["first_price"] = out["first_price"].astype(str)
    out["last_price"] = out["last_price"].astype(str)
    return out.sort_values(["selection_bucket", "instrument"])


def plot_results(
    continuous_daily: pd.DataFrame,
    integer_daily: pd.DataFrame,
    asset_daily: pd.DataFrame,
    capital: float,
) -> None:
    continuous_equity = capital + continuous_daily["net_pnl"].cumsum()
    integer_equity = capital + integer_daily["net_pnl"].cumsum()
    integer_drawdown = integer_equity / integer_equity.cummax() - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    continuous_equity.plot(ax=axes[0], label="Continuous")
    integer_equity.plot(ax=axes[0], label="Buffered integer")
    axes[0].set_title("Rob-Style US Rates / Selected FX & Commodities, No Vol")
    axes[0].set_ylabel("USD")
    axes[0].legend()

    integer_drawdown.plot(ax=axes[1], title="Buffered Integer Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

    asset_daily.cumsum().plot(ax=axes[2], title="Buffered Integer Cumulative P&L by Asset Class")
    axes[2].set_ylabel("USD")
    fig.tight_layout()
    fig.savefig(OUT / "equity_drawdown_assetclass.png", dpi=160)


def write_summary(
    config: dict,
    instruments: list[str],
    continuous_stats: dict[str, float | str],
    integer_stats: dict[str, float | str],
    asset_summary: pd.DataFrame,
    instr_summary: pd.DataFrame,
    cap_summary: pd.DataFrame,
    risk_multiplier: pd.Series,
) -> None:
    capacity_ok = (cap_summary["median_contracts_at_forecast_10"] >= 0.50).sum()
    low_capacity = cap_summary.nsmallest(8, "median_contracts_at_forecast_10")
    lines = [
        "# Rob-Style US Rates / Selected FX & Commodities, No Vol Backtest",
        "",
        "## Design",
        "",
        "- Keeps only US rates, selected FX, selected agricultural futures, selected metals, and crude/gasoline.",
        "- Excludes equity, non-US bonds, most FX, extra ags/metals/oil, and volatility futures.",
        "- Uses the same Rob rule set, forecast scalars, forecast weights, FDM values, buffer, and integer optimiser.",
        "- Asset-class risk budgets are the prior no-equity weights excluding Vol and rescaled to 100%.",
        "",
        "## Universe And Risk Budgets",
        "",
    ]
    for bucket, names in CUSTOM_UNIVERSE.items():
        per_instrument = ASSET_CLASS_RISK_WEIGHTS[bucket] / len(names)
        lines.append(
            f"- {bucket} ({len(names)}, asset weight {ASSET_CLASS_RISK_WEIGHTS[bucket]:.1%}, "
            f"per instrument {per_instrument:.2%}): {', '.join(names)}"
        )
    lines.extend(
        [
            "",
            "## Capacity Diagnostics",
            "",
            f"- Instruments with median forecast=10 position >= 0.50 contracts: {capacity_ok} / {len(instruments)}",
            f"- Instruments with median forecast=10 position >= 1.00 contracts: {(cap_summary['median_contracts_at_forecast_10'] >= 1.00).sum()} / {len(instruments)}",
            f"- Median forecast=10 position across selected instruments: {cap_summary['median_contracts_at_forecast_10'].median():.2f} contracts",
            f"- Lowest median forecast=10 position: {cap_summary['median_contracts_at_forecast_10'].min():.2f} contracts",
            f"- Median trade cost / annual contract risk: {cap_summary['median_cost_as_pct_annual_risk'].median():.2%}",
            "",
            "## Portfolio Settings",
            "",
            f"- Capital: ${float(config['notional_trading_capital']):,.0f}",
            f"- Annual volatility target: {float(config['percentage_vol_target']):.1f}%",
            f"- Instrument diversification multiplier: {float(config.get('instrument_div_multiplier', 1.0)):.2f}",
            f"- Position buffer size: {bt.BUFFER_SIZE:.1%}",
            f"- Tracking error buffer: {bt.TRACKING_ERROR_BUFFER:.2%}",
            f"- Average risk multiplier: {risk_multiplier.mean():.2f}",
            f"- Minimum risk multiplier: {risk_multiplier.min():.2f}",
            "",
            "## Continuous Position Results",
            "",
        ]
    )
    for key, value in continuous_stats.items():
        lines.append(f"- {key}: {bt.format_stat(key, value)}")
    lines.extend(["", "## Buffered Integer Results", ""])
    for key, value in integer_stats.items():
        lines.append(f"- {key}: {bt.format_stat(key, value)}")
    lines.extend(["", "## Asset Class P&L, Buffered Integer", ""])
    for row in asset_summary.itertuples(index=False):
        lines.append(
            f"- {row.asset_class}: net ${row.net_pnl:,.0f}; gross ${row.gross_pnl:,.0f}; costs ${row.costs:,.0f}"
        )
    lines.extend(["", "## Lowest Capacity Instruments", ""])
    for row in low_capacity.itertuples(index=False):
        lines.append(
            f"- {row.instrument} ({row.asset_class}): median forecast=10 position "
            f"{row.median_contracts_at_forecast_10:.2f}; median cost/risk "
            f"{row.median_cost_as_pct_annual_risk:.2%}"
        )
    lines.extend(["", "## Highest Cost Instruments", ""])
    for row in instr_summary.sort_values("costs", ascending=False).head(8).itertuples(index=False):
        lines.append(
            f"- {row.instrument} ({row.asset_class}): costs ${row.costs:,.0f}; "
            f"net ${row.net_pnl:,.0f}; trades {row.total_traded_contracts:,.0f}"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- With only 17 instruments, the original IDM of 2.75 may overstate diversification; inspect realised volatility and risk overlay before live use.",
            "- This is still a compact local CSV implementation, not the full pysystemtrade production stack.",
            "- Asset-class relative rules are computed inside this reduced universe.",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = config_with_custom_weights(bt.load_rob_config())
    meta = bt.load_meta()
    instruments = selected_instruments()

    missing = [instrument for instrument in instruments if not bt.has_required_files(instrument, meta)]
    if missing:
        raise RuntimeError(f"Selected instruments missing local data: {missing}")

    price = bt.load_price_matrix(instruments)
    price_vol = bt.mixed_vol(price.diff())
    fx = bt.load_fx_matrix(instruments, meta, price.index)
    raw_carry = bt.load_raw_carry_matrix(instruments, price_vol, price.index)
    forecasts = bt.build_rule_forecasts(config, instruments, meta, price, price_vol, raw_carry)
    combined_forecast, rule_weight_used = bt.combine_forecasts(forecasts, config, instruments, price.index)
    target, daily_weights, unit_daily_cash_vol = bt.initial_target_positions(
        config, instruments, meta, price, price_vol, fx, combined_forecast
    )
    risk_multiplier = bt.risk_multiplier_for_targets(config, target, unit_daily_cash_vol, price, fx, meta, instruments)
    target = target.mul(risk_multiplier, axis=0)
    costs = bt.cost_matrix(instruments, meta, price, fx)

    continuous_daily, _continuous_by_instrument = bt.pnl_from_positions(
        target.fillna(0.0), price, fx, meta, instruments, costs
    )

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
    integer_daily, integer_by_instrument = bt.pnl_from_positions(integer_positions, price, fx, meta, instruments, costs)

    capital = float(config["notional_trading_capital"])
    continuous_stats = bt.portfolio_stats(continuous_daily, capital)
    integer_stats = bt.portfolio_stats(integer_daily, capital)
    base_weights = bt.base_instrument_weights(config, instruments)
    instr_summary = bt.instrument_summary(
        integer_by_instrument,
        instruments,
        meta,
        base_weights,
        combined_forecast,
        rule_weight_used,
        config,
    )
    cap_summary = capacity_summary(instruments, meta, price, unit_daily_cash_vol, daily_weights, costs, config)
    instr_summary = instr_summary.merge(
        cap_summary[
            [
                "instrument",
                "selection_bucket",
                "target_asset_weight",
                "instrument_weight",
                "median_contracts_at_forecast_10",
                "median_contract_annual_risk_usd",
                "median_trade_cost_usd",
                "median_cost_as_pct_annual_risk",
            ]
        ],
        on="instrument",
        how="left",
    )
    asset_summary = (
        instr_summary.groupby("asset_class")[["net_pnl", "gross_pnl", "costs"]]
        .sum()
        .reset_index()
        .sort_values("asset_class")
    )
    asset_daily = pd.DataFrame(index=price.index)
    for asset_class in asset_summary["asset_class"]:
        columns = [instrument for instrument in instruments if meta[instrument].asset_class == asset_class]
        asset_daily[asset_class] = integer_by_instrument["net_pnl"][columns].sum(axis=1)

    continuous_out = continuous_daily.copy()
    continuous_out["daily_return"] = continuous_out["net_pnl"] / capital
    continuous_out["equity"] = capital + continuous_out["net_pnl"].cumsum()
    continuous_out["drawdown"] = continuous_out["equity"] / continuous_out["equity"].cummax() - 1.0
    integer_out = integer_daily.copy()
    integer_out["daily_return"] = integer_out["net_pnl"] / capital
    integer_out["equity"] = capital + integer_out["net_pnl"].cumsum()
    integer_out["drawdown"] = integer_out["equity"] / integer_out["equity"].cummax() - 1.0

    pd.concat({"continuous": continuous_out, "buffered_integer": integer_out}, axis=1).to_csv(
        OUT / "portfolio_daily.csv"
    )
    instr_summary.to_csv(OUT / "instrument_summary.csv", index=False)
    cap_summary.to_csv(OUT / "capacity_summary.csv", index=False)
    asset_summary.to_csv(OUT / "asset_class_summary.csv", index=False)
    asset_daily.to_csv(OUT / "asset_class_daily_pnl.csv")
    daily_weights.to_csv(OUT / "daily_instrument_weights.csv")
    risk_multiplier.to_csv(OUT / "risk_multiplier.csv")

    write_summary(
        config,
        instruments,
        continuous_stats,
        integer_stats,
        asset_summary,
        instr_summary,
        cap_summary,
        risk_multiplier,
    )
    plot_results(continuous_daily, integer_daily, asset_daily, capital)

    print(f"Wrote results to {OUT}")
    print(f"selected instruments: {len(instruments)}")
    print(
        "capacity >= 0.5 contracts:",
        int((cap_summary["median_contracts_at_forecast_10"] >= 0.50).sum()),
        "/",
        len(instruments),
    )
    print(
        "capacity >= 1.0 contracts:",
        int((cap_summary["median_contracts_at_forecast_10"] >= 1.00).sum()),
        "/",
        len(instruments),
    )
    print("continuous:")
    for key, value in continuous_stats.items():
        print(f"  {key}: {bt.format_stat(key, value)}")
    print("buffered_integer:")
    for key, value in integer_stats.items():
        print(f"  {key}: {bt.format_stat(key, value)}")


if __name__ == "__main__":
    main()
