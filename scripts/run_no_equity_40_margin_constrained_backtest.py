#!/usr/bin/env python3
"""Run the no-equity 40 futures system with explicit futures margin constraints.

This keeps the existing Rob-style forecast/risk/position engine unchanged. The
only additional layer is a post-sizing futures margin check applied to the final
buffered integer contract positions.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_rob_style_backtest as bt  # noqa: E402
import run_rob_style_no_equity_40_backtest as no40  # noqa: E402


OUT = ROOT / "backtests" / "rob_style_no_equity_40_margin_constrained"
IBKR_MARGIN_URL = (
    "https://brokerage.ibkr.com/en/trading/margin-futures-fops.php"
    "?ex=us&hm=us&pm=0&relp=49560&rgt=0&rsk=1&rst=101004110808"
)
START_DATE = "1970-01-01"
MARGIN_LIMITS = (1.00, 0.75, 0.50)


@dataclass(frozen=True)
class MarginMapping:
    exchange: str
    trading_class: str
    note: str


MARGIN_MAPPING: dict[str, MarginMapping] = {
    "JGB-SGX-mini": MarginMapping("SGX", "JB", "SGX Mini JGB"),
    "KR10": MarginMapping("KSE", "67", "Korea 10-year treasury bond"),
    "KR3": MarginMapping("KSE", "65", "Korea 3-year treasury bond"),
    "US2": MarginMapping("CBOT", "ZT", "2-year US Treasury note"),
    "US5": MarginMapping("CBOT", "ZF", "5-year US Treasury note"),
    "SOFR": MarginMapping("CME", "SR3", "3-month SOFR"),
    "BTP3": MarginMapping("EUREX", "FBTS", "Short-term Euro-BTP"),
    "OAT": MarginMapping("EUREX", "FOAT", "Euro-OAT"),
    "US10": MarginMapping("CBOT", "ZN", "10-year US Treasury note"),
    "US20": MarginMapping("CBOT", "TWE", "20-year US Treasury bond"),
    "BUND": MarginMapping("EUREX", "FGBL", "Euro Bund"),
    "SHATZ": MarginMapping("EUREX", "FGBS", "Euro Schatz"),
    "EUR_micro": MarginMapping("CME", "M6E", "Micro EUR/USD"),
    "AUD": MarginMapping("CME", "6A", "AUD/USD"),
    "CAD": MarginMapping("CME", "6C", "CAD/USD"),
    "GBP": MarginMapping("CME", "6B", "GBP/USD"),
    "CNH": MarginMapping("SGX", "UC", "SGX USD/CNH"),
    "MXP": MarginMapping("CME", "6M", "Mexican peso"),
    "JPY": MarginMapping("CME", "6J", "JPY/USD"),
    "NZD": MarginMapping("CME", "6N", "NZD/USD"),
    "LEANHOG": MarginMapping("CME", "HE", "Lean hogs"),
    "LIVECOW": MarginMapping("CME", "LE", "Live cattle"),
    "CORN": MarginMapping("CBOT", "ZC", "Corn"),
    "SOYBEAN_mini": MarginMapping("CBOT", "XK", "Mini soybean"),
    "SOYMEAL": MarginMapping("CBOT", "ZM", "Soybean meal"),
    "WHEAT": MarginMapping("CBOT", "ZW", "Wheat"),
    "SOYOIL": MarginMapping("CBOT", "ZL", "Soybean oil"),
    "FEEDCOW": MarginMapping("CME", "GF", "Feeder cattle"),
    "GOLD_micro": MarginMapping("COMEX", "MGC", "Micro gold"),
    "COPPER-micro": MarginMapping("COMEX", "MHG", "Micro copper"),
    "SILVER": MarginMapping("COMEX", "SIL", "1,000 oz silver"),
    "PLAT": MarginMapping("NYMEX", "PL", "Platinum"),
    "CRUDE_W": MarginMapping("NYMEX", "CL", "WTI crude"),
    "GAS_US_mini": MarginMapping("NYMEX", "QG", "Mini natural gas"),
    "HEATOIL": MarginMapping("NYMEX", "HO", "Heating oil"),
    "GASOILINE": MarginMapping("NYMEX", "RB", "RBOB gasoline"),
    "VIX": MarginMapping("CFE", "VX", "CBOE VIX"),
    "V2X": MarginMapping("EUREX", "FVS", "VSTOXX"),
}

FALLBACK_MARGIN_RATES = {
    "Bond": 0.03,
    "FX": 0.04,
    "Ags": 0.10,
    "Metals": 0.12,
    "OilGas": 0.14,
    "Vol": 0.45,
}


def parse_numeric(value: object) -> float:
    text = str(value)
    text = re.sub(r"[^0-9.\\-]", "", text)
    return float(text) if text else np.nan


def load_ibkr_margin_table() -> pd.DataFrame:
    frames = []
    for table_id, table in enumerate(pd.read_html(IBKR_MARGIN_URL)):
        table = table.copy()
        first_col = str(table.columns[0])
        if first_col.startswith("Exchange "):
            table = table.rename(columns={table.columns[0]: "Exchange"})
        required = {"Exchange", "Underlying", "Trading Class", "Overnight Initial", "Currency"}
        if required.issubset(table.columns):
            table["ibkr_table_id"] = table_id
            frames.append(table)
    if not frames:
        raise RuntimeError("Could not parse any IBKR futures margin tables")
    margin = pd.concat(frames, ignore_index=True)
    for column in [
        "Overnight Initial",
        "Overnight Maintenance",
        "Short Overnight Initial",
        "Short Overnight Maintenance",
    ]:
        if column in margin.columns:
            margin[column] = margin[column].map(parse_numeric)
    return margin


def margin_row(margin: pd.DataFrame, mapping: MarginMapping) -> pd.Series:
    mask = (
        margin["Exchange"].astype(str).str.upper().eq(mapping.exchange.upper())
        & margin["Trading Class"].astype(str).str.upper().eq(mapping.trading_class.upper())
    )
    matches = margin.loc[mask]
    if matches.empty:
        raise KeyError(f"No IBKR margin row for {mapping.exchange} {mapping.trading_class}")
    return matches.iloc[0]


def margin_currency_fx(currency: str, index: pd.Index) -> pd.Series:
    currency = str(currency).upper()
    if currency == "USD":
        return pd.Series(1.0, index=index)
    fx_path = bt.FX / f"{currency}USD.csv"
    if not fx_path.exists():
        raise FileNotFoundError(f"Missing FX file for margin currency {currency}: {fx_path}")
    data = pd.read_csv(fx_path, parse_dates=["DATETIME"])
    fx = data.set_index("DATETIME")["PRICE"].sort_index().resample("1B").last().ffill()
    return fx.reindex(index, method="ffill").ffill()


def latest_valid(series: pd.Series) -> float:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(valid.iloc[-1]) if not valid.empty else np.nan


def build_margin_schedule(
    instruments: list[str],
    meta: dict[str, bt.InstrumentMeta],
    price: pd.DataFrame,
    fx: pd.DataFrame,
    margin: pd.DataFrame,
) -> pd.DataFrame:
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    current_notional = price.abs().mul(point_sizes, axis=1).mul(fx).apply(latest_valid)
    rows = []

    for instrument in instruments:
        asset_class = meta[instrument].asset_class
        mapping = MARGIN_MAPPING.get(instrument)
        if mapping is None:
            rate = FALLBACK_MARGIN_RATES[asset_class]
            rows.append(
                {
                    "instrument": instrument,
                    "asset_class": asset_class,
                    "source": "fallback_notional_rate",
                    "exchange": "",
                    "trading_class": "",
                    "product_description": "",
                    "margin_currency": "USD",
                    "long_initial_native": np.nan,
                    "short_initial_native": np.nan,
                    "latest_margin_fx": 1.0,
                    "latest_contract_notional_usd": current_notional[instrument],
                    "long_initial_usd_latest": current_notional[instrument] * rate,
                    "short_initial_usd_latest": current_notional[instrument] * rate,
                    "long_margin_to_notional": rate,
                    "short_margin_to_notional": rate,
                    "mapping_note": "No current IBKR row found; conservative notional-rate fallback",
                }
            )
            continue

        row = margin_row(margin, mapping)
        currency = str(row["Currency"]).upper()
        margin_fx = margin_currency_fx(currency, price.index)
        latest_margin_fx = latest_valid(margin_fx)
        long_native = float(row["Overnight Initial"])
        short_native = float(row.get("Short Overnight Initial", long_native))
        if not np.isfinite(short_native):
            short_native = long_native
        long_usd = long_native * latest_margin_fx
        short_usd = short_native * latest_margin_fx
        rows.append(
            {
                "instrument": instrument,
                "asset_class": asset_class,
                "source": "ibkr_current_overnight_initial",
                "exchange": row["Exchange"],
                "trading_class": row["Trading Class"],
                "product_description": row.get("Product description", ""),
                "margin_currency": currency,
                "long_initial_native": long_native,
                "short_initial_native": short_native,
                "latest_margin_fx": latest_margin_fx,
                "latest_contract_notional_usd": current_notional[instrument],
                "long_initial_usd_latest": long_usd,
                "short_initial_usd_latest": short_usd,
                "long_margin_to_notional": long_usd / current_notional[instrument],
                "short_margin_to_notional": short_usd / current_notional[instrument],
                "mapping_note": mapping.note,
            }
        )

    return pd.DataFrame(rows).sort_values(["asset_class", "instrument"])


def margin_per_contract_matrices(
    schedule: pd.DataFrame,
    instruments: list[str],
    meta: dict[str, bt.InstrumentMeta],
    price: pd.DataFrame,
    fx: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_margin = pd.DataFrame(index=price.index, columns=instruments, dtype=float)
    short_margin = pd.DataFrame(index=price.index, columns=instruments, dtype=float)
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    notional = price.abs().mul(point_sizes, axis=1).mul(fx)
    schedule_by_instrument = schedule.set_index("instrument")

    for instrument in instruments:
        row = schedule_by_instrument.loc[instrument]
        if mode == "notional_scaled":
            long_margin[instrument] = notional[instrument] * float(row["long_margin_to_notional"])
            short_margin[instrument] = notional[instrument] * float(row["short_margin_to_notional"])
        elif mode == "current_static":
            if row["source"] == "fallback_notional_rate":
                long_margin[instrument] = notional[instrument] * float(row["long_margin_to_notional"])
                short_margin[instrument] = notional[instrument] * float(row["short_margin_to_notional"])
            else:
                currency_fx = margin_currency_fx(row["margin_currency"], price.index)
                long_margin[instrument] = float(row["long_initial_native"]) * currency_fx
                short_margin[instrument] = float(row["short_initial_native"]) * currency_fx
        else:
            raise ValueError(f"Unknown margin mode: {mode}")
    return long_margin.ffill(), short_margin.ffill()


def total_margin(position: pd.Series, long_margin: pd.Series, short_margin: pd.Series) -> float:
    margin = position.abs() * np.where(position >= 0.0, long_margin, short_margin)
    return float(pd.Series(margin, index=position.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).sum())


def constrain_position_to_margin(
    desired: pd.Series,
    long_margin: pd.Series,
    short_margin: pd.Series,
    equity: float,
    limit: float,
) -> tuple[pd.Series, float, float]:
    desired = desired.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    long_margin = long_margin.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    short_margin = short_margin.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    desired_requirement = total_margin(desired, long_margin, short_margin)
    allowed = max(0.0, equity * limit)
    if equity <= 0.0 or allowed <= 0.0:
        return desired * 0.0, desired_requirement, 0.0
    if desired_requirement <= allowed or desired_requirement == 0.0:
        return desired, desired_requirement, 1.0
    scale = allowed / desired_requirement
    constrained = np.sign(desired) * np.floor(desired.abs() * scale)
    constrained = constrained.astype(float)
    return constrained, desired_requirement, scale


def run_margin_constrained_pnl(
    desired_positions: pd.DataFrame,
    price: pd.DataFrame,
    fx: pd.DataFrame,
    meta: dict[str, bt.InstrumentMeta],
    instruments: list[str],
    costs_per_contract: pd.DataFrame,
    long_margin: pd.DataFrame,
    short_margin: pd.DataFrame,
    capital: float,
    margin_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_sizes = pd.Series({instrument: meta[instrument].point_size for instrument in instruments})
    positions = pd.DataFrame(0.0, index=price.index, columns=instruments)
    gross_by_instr = pd.DataFrame(0.0, index=price.index, columns=instruments)
    cost_by_instr = pd.DataFrame(0.0, index=price.index, columns=instruments)
    margin_used = pd.Series(0.0, index=price.index)
    desired_margin = pd.Series(0.0, index=price.index)
    margin_scale = pd.Series(1.0, index=price.index)
    current_equity = float(capital)
    previous_position = pd.Series(0.0, index=instruments)
    previous_price = None

    for date in price.index:
        price_today = price.loc[date].reindex(instruments)
        fx_today = fx.loc[date].reindex(instruments).fillna(0.0)
        if previous_price is None:
            price_change = pd.Series(0.0, index=instruments)
        else:
            price_change = (price_today - previous_price).fillna(0.0)
        gross_vec = previous_position * price_change * point_sizes * fx_today
        equity_before_rebalance = current_equity + float(gross_vec.sum())

        desired = desired_positions.loc[date].reindex(instruments).fillna(0.0)
        constrained, wanted_margin, scale = constrain_position_to_margin(
            desired,
            long_margin.loc[date].reindex(instruments),
            short_margin.loc[date].reindex(instruments),
            equity_before_rebalance,
            margin_limit,
        )
        if previous_price is None:
            trades = constrained * 0.0
        else:
            trades = (constrained - previous_position).abs()
        cost_vec = trades * costs_per_contract.loc[date].reindex(instruments).fillna(0.0)
        net_vec = gross_vec - cost_vec
        current_equity += float(net_vec.sum())

        positions.loc[date] = constrained
        gross_by_instr.loc[date] = gross_vec
        cost_by_instr.loc[date] = cost_vec
        margin_used.loc[date] = total_margin(
            constrained,
            long_margin.loc[date].reindex(instruments),
            short_margin.loc[date].reindex(instruments),
        )
        desired_margin.loc[date] = wanted_margin
        margin_scale.loc[date] = scale

        previous_position = constrained
        previous_price = price_today

    trades = positions.diff().abs().fillna(0.0)
    net_by_instr = gross_by_instr - cost_by_instr
    daily = pd.DataFrame(
        {
            "gross_pnl": gross_by_instr.sum(axis=1),
            "costs": cost_by_instr.sum(axis=1),
            "net_pnl": net_by_instr.sum(axis=1),
            "margin_used": margin_used,
            "desired_margin": desired_margin,
            "margin_scale": margin_scale,
        },
        index=price.index,
    )
    daily["equity"] = capital + daily["net_pnl"].cumsum()
    daily["daily_return"] = daily["net_pnl"] / capital
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    daily["margin_to_equity"] = daily["margin_used"] / daily["equity"].replace(0.0, np.nan)
    daily["desired_margin_to_equity"] = daily["desired_margin"] / daily["equity"].replace(0.0, np.nan)
    by_instrument = pd.concat(
        {
            "gross_pnl": gross_by_instr,
            "costs": cost_by_instr,
            "net_pnl": net_by_instr,
            "position": positions,
            "trades": trades,
        },
        axis=1,
    )
    return daily, by_instrument


def stats_from_daily(daily: pd.DataFrame, capital: float) -> dict[str, float | str]:
    daily = daily.copy()
    equity = daily["equity"] if "equity" in daily else capital + daily["net_pnl"].cumsum()
    returns = daily["net_pnl"] / capital
    years = len(daily) / bt.BUSINESS_DAYS
    ann_return = returns.mean() * bt.BUSINESS_DAYS
    ann_vol = returns.std() * math.sqrt(bt.BUSINESS_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0.0 else np.nan
    total_return = equity.iloc[-1] / capital - 1.0 if not equity.empty else np.nan
    cagr = (equity.iloc[-1] / capital) ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "start": str(daily.index.min().date()) if len(daily) else "",
        "end": str(daily.index.max().date()) if len(daily) else "",
        "years": years,
        "ann_return": ann_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "total_return": total_return,
        "total_costs": float(daily["costs"].sum()),
        "median_margin_to_equity": float(daily.get("margin_to_equity", pd.Series(dtype=float)).median()),
        "p95_margin_to_equity": float(daily.get("margin_to_equity", pd.Series(dtype=float)).quantile(0.95)),
        "max_margin_to_equity": float(daily.get("margin_to_equity", pd.Series(dtype=float)).max()),
        "pct_days_scaled": float((daily.get("margin_scale", pd.Series(1.0, index=daily.index)) < 0.999).mean()),
    }


def format_pct(value: float) -> str:
    return "nan" if pd.isna(value) else f"{value:.2%}"


def format_float(value: float) -> str:
    return "nan" if pd.isna(value) else f"{value:.2f}"


def build_base_system() -> tuple[
    dict,
    list[str],
    dict[str, bt.InstrumentMeta],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    bt.START_DATE = START_DATE
    config = no40.config_with_no_equity_weights(bt.load_rob_config())
    meta = bt.load_meta()
    instruments = no40.selected_instruments()
    missing = [instrument for instrument in instruments if not bt.has_required_files(instrument, meta)]
    if missing:
        raise RuntimeError(f"Selected instruments missing local data: {missing}")

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
        / math.sqrt(bt.BUSINESS_DAYS)
        / unit_daily_cash_vol
    )
    buffer = (subsystem_position.abs() * daily_weights * idm * bt.BUFFER_SIZE).mul(risk_multiplier, axis=0)
    buffered_target = bt.apply_position_buffer(target.fillna(0.0), buffer.fillna(0.0))
    integer_positions = bt.optimise_integer_positions(buffered_target, unit_daily_cash_vol, config)
    return config, instruments, meta, price, fx, costs, integer_positions


def plot_results(daily_by_name: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for name, daily in daily_by_name.items():
        daily["equity"].plot(ax=axes[0], label=name)
    axes[0].set_title("Rob 40 No-Equity With Futures Margin Constraints")
    axes[0].set_ylabel("USD equity")
    axes[0].legend(fontsize=8)

    for name, daily in daily_by_name.items():
        daily["drawdown"].plot(ax=axes[1], label=name)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")

    for name, daily in daily_by_name.items():
        if "margin_to_equity" in daily:
            daily["margin_to_equity"].rolling(20).mean().plot(ax=axes[2], label=name)
    axes[2].set_title("20-Day Average Initial Margin / Equity")
    axes[2].set_ylabel("Margin / equity")
    axes[2].yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "margin_constrained_equity.png", dpi=170)


def write_summary(metrics: pd.DataFrame, schedule: pd.DataFrame, daily_by_name: dict[str, pd.DataFrame]) -> None:
    mapped = int((schedule["source"] == "ibkr_current_overnight_initial").sum())
    fallback = int((schedule["source"] != "ibkr_current_overnight_initial").sum())
    base = metrics[(metrics["period"] == "2000+") & (metrics["strategy"] == "unconstrained")].iloc[0]
    static100 = metrics[
        (metrics["period"] == "2000+") & (metrics["strategy"] == "current_static_margin_cap_100")
    ].iloc[0]
    scaled100 = metrics[
        (metrics["period"] == "2000+") & (metrics["strategy"] == "notional_scaled_margin_cap_100")
    ].iloc[0]
    worst_margin = schedule.sort_values("long_margin_to_notional", ascending=False).head(10)
    lines = [
        "# Rob 40 No-Equity Futures Margin Constraint Test",
        "",
        "## Setup",
        "",
        "- Base strategy: existing Rob-style no-equity 40 universe, buffered integer contracts.",
        "- Unchanged: forecast generation, forecast scaling, volatility target, instrument weights, IDM/FDM, position buffer, integer optimizer, costs, and P&L mechanics.",
        "- Added layer: post-sizing futures initial-margin cap applied after integer positions are produced.",
        f"- IBKR current margin rows mapped: {mapped} instruments.",
        f"- Fallback notional-rate margins: {fallback} instruments.",
        "- Margin modes:",
        "  - `current_static`: current IBKR overnight initial margin per contract, translated with daily FX; fallback instruments use a notional-rate proxy.",
        "  - `notional_scaled`: current margin-to-notional ratio applied to each historical date's contract notional.",
        "",
        "## Key 2000+ Result",
        "",
        "| Strategy | CAGR | Ann Return | Vol | Sharpe | Max DD | Median Margin/Equity | P95 Margin/Equity | Days Scaled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Unconstrained | {format_pct(base.cagr)} | {format_pct(base.ann_return)} | "
            f"{format_pct(base.ann_vol)} | {format_float(base.sharpe)} | {format_pct(base.max_drawdown)} | "
            f"{format_pct(base.median_margin_to_equity)} | {format_pct(base.p95_margin_to_equity)} | "
            f"{format_pct(base.pct_days_scaled)} |"
        ),
        (
            f"| Current static cap 100% | {format_pct(static100.cagr)} | {format_pct(static100.ann_return)} | "
            f"{format_pct(static100.ann_vol)} | {format_float(static100.sharpe)} | {format_pct(static100.max_drawdown)} | "
            f"{format_pct(static100.median_margin_to_equity)} | {format_pct(static100.p95_margin_to_equity)} | "
            f"{format_pct(static100.pct_days_scaled)} |"
        ),
        (
            f"| Notional scaled cap 100% | {format_pct(scaled100.cagr)} | {format_pct(scaled100.ann_return)} | "
            f"{format_pct(scaled100.ann_vol)} | {format_float(scaled100.sharpe)} | {format_pct(scaled100.max_drawdown)} | "
            f"{format_pct(scaled100.median_margin_to_equity)} | {format_pct(scaled100.p95_margin_to_equity)} | "
            f"{format_pct(scaled100.pct_days_scaled)} |"
        ),
        "",
        "## Highest Current Margin / Notional",
        "",
    ]
    for row in worst_margin.itertuples(index=False):
        lines.append(
            f"- {row.instrument}: long {row.long_margin_to_notional:.1%}, "
            f"short {row.short_margin_to_notional:.1%}; source {row.source}; "
            f"{row.exchange} {row.trading_class} {row.product_description}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This test treats futures margin as collateral usage, not as financing borrowed on the full futures notional.",
            "- True historical SPAN margins are not in the local dataset. CME provides historical margin datasets separately, and IBKR says margins can change frequently.",
            "- Therefore this is a real-current-margin capacity stress, not a perfect historical broker ledger.",
            "",
            "## Files",
            "",
            "- `margin_schedule.csv`",
            "- `metrics.csv`",
            "- `all_daily_2000_plus.csv`",
            "- `positions_*.csv`",
            "- `margin_constrained_equity.png`",
            "",
        ]
    )
    for name, daily in daily_by_name.items():
        final_equity = daily["equity"].iloc[-1]
        lines.append(f"- {name}: final equity ${final_equity:,.0f}")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config, instruments, meta, price, fx, costs, desired_positions = build_base_system()
    capital = float(config["notional_trading_capital"])

    margin_table = load_ibkr_margin_table()
    margin_table.to_csv(OUT / "ibkr_current_margin_table_snapshot.csv", index=False)
    schedule = build_margin_schedule(instruments, meta, price, fx, margin_table)
    schedule.to_csv(OUT / "margin_schedule.csv", index=False)

    metrics_rows = []
    periods = {"full": None, "1993+": "1993-01-01", "2000+": "2000-01-01"}
    daily_for_plot: dict[str, pd.DataFrame] = {}
    first_unconstrained_by_instr = None

    for mode in ("current_static", "notional_scaled"):
        long_margin, short_margin = margin_per_contract_matrices(schedule, instruments, meta, price, fx, mode=mode)
        for period_name, start in periods.items():
            period_price = price if start is None else price.loc[start:]
            period_fx = fx.loc[period_price.index]
            period_costs = costs.loc[period_price.index]
            period_desired = desired_positions.loc[period_price.index]
            period_long_margin = long_margin.loc[period_price.index]
            period_short_margin = short_margin.loc[period_price.index]

            if mode == "current_static":
                unconstrained_daily, unconstrained_by_instr = bt.pnl_from_positions(
                    period_desired, period_price, period_fx, meta, instruments, period_costs
                )
                unconstrained_daily = unconstrained_daily.copy()
                unconstrained_daily["daily_return"] = unconstrained_daily["net_pnl"] / capital
                unconstrained_daily["equity"] = capital + unconstrained_daily["net_pnl"].cumsum()
                unconstrained_daily["drawdown"] = (
                    unconstrained_daily["equity"] / unconstrained_daily["equity"].cummax() - 1.0
                )
                unconstrained_daily["margin_used"] = np.nan
                unconstrained_daily["desired_margin"] = np.nan
                unconstrained_daily["margin_scale"] = 1.0
                unconstrained_daily["margin_to_equity"] = np.nan
                unconstrained_daily["desired_margin_to_equity"] = np.nan
                row = stats_from_daily(unconstrained_daily, capital)
                row.update(
                    {
                        "strategy": "unconstrained",
                        "period": period_name,
                        "margin_mode": "none",
                        "margin_cap": np.nan,
                    }
                )
                metrics_rows.append(row)
                if period_name == "2000+":
                    daily_for_plot["unconstrained"] = unconstrained_daily
                    unconstrained_daily.to_csv(OUT / "daily_2000_unconstrained.csv")
                    unconstrained_by_instr["position"].to_csv(OUT / "positions_2000_unconstrained.csv")
                if first_unconstrained_by_instr is None:
                    first_unconstrained_by_instr = unconstrained_by_instr

            for limit in MARGIN_LIMITS:
                daily, by_instr = run_margin_constrained_pnl(
                    period_desired,
                    period_price,
                    period_fx,
                    meta,
                    instruments,
                    period_costs,
                    period_long_margin,
                    period_short_margin,
                    capital,
                    margin_limit=limit,
                )
                name = f"{mode}_margin_cap_{int(limit * 100)}"
                row = stats_from_daily(daily, capital)
                row.update({"strategy": name, "period": period_name, "margin_mode": mode, "margin_cap": limit})
                metrics_rows.append(row)
                safe_period = period_name.replace("+", "plus").replace(":", "").replace("/", "_")
                daily.to_csv(OUT / f"daily_{safe_period}_{name}.csv")
                by_instr["position"].to_csv(OUT / f"positions_{safe_period}_{name}.csv")
                if period_name == "2000+":
                    daily_for_plot[name] = daily

    all_daily_2000 = pd.concat(daily_for_plot, axis=1)
    all_daily_2000.to_csv(OUT / "all_daily_2000_plus.csv")
    all_daily_2000.to_csv(OUT / "all_daily.csv")
    if first_unconstrained_by_instr is not None:
        first_unconstrained_by_instr["position"].to_csv(OUT / "positions_unconstrained_full.csv")
    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[
        [
            "period",
            "strategy",
            "margin_mode",
            "margin_cap",
            "start",
            "end",
            "years",
            "ann_return",
            "cagr",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "total_return",
            "total_costs",
            "median_margin_to_equity",
            "p95_margin_to_equity",
            "max_margin_to_equity",
            "pct_days_scaled",
        ]
    ]
    metrics.to_csv(OUT / "metrics.csv", index=False)
    plot_results(
        {
            "unconstrained": daily_for_plot["unconstrained"],
            "static cap 100%": daily_for_plot["current_static_margin_cap_100"],
            "scaled cap 100%": daily_for_plot["notional_scaled_margin_cap_100"],
            "scaled cap 75%": daily_for_plot["notional_scaled_margin_cap_75"],
            "scaled cap 50%": daily_for_plot["notional_scaled_margin_cap_50"],
        }
    )
    write_summary(metrics, schedule, daily_for_plot)

    display = metrics[metrics["period"].eq("2000+")].copy()
    for column in ["ann_return", "cagr", "ann_vol", "max_drawdown", "median_margin_to_equity", "p95_margin_to_equity", "pct_days_scaled"]:
        display[column] = display[column].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    display["sharpe"] = display["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(f"Wrote margin constrained results to {OUT}")
    print(display[["strategy", "cagr", "ann_return", "ann_vol", "sharpe", "max_drawdown", "median_margin_to_equity", "p95_margin_to_equity", "pct_days_scaled"]].to_string(index=False))


if __name__ == "__main__":
    main()
