#!/usr/bin/env python3
"""Run a fresh 2016 IBKR-style USD 500k Rob 40 account and compare intervals."""

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


OUT = ROOT / "backtests" / "ibkr_500k_rob40_2016_intervals"
MARGIN_DIR = ROOT / "backtests" / "rob_style_no_equity_40_margin_constrained"
KMLM_DIR = ROOT / "backtests" / "kmlm_40_universe_comparison"
CAPITAL = 500_000.0
START = "2016-01-04"
LOCAL_END = "2024-03-28"
BUSINESS_DAYS = bt.BUSINESS_DAYS


PERIODS = [
    ("2016-full local", "2016-01-04", "2024-03-28"),
    ("2016-2019 pre-COVID", "2016-01-04", "2019-12-31"),
    ("2020 COVID/rebound", "2020-01-01", "2020-12-31"),
    ("2021 bull", "2021-01-01", "2021-12-31"),
    ("2022 inflation bear", "2022-01-01", "2022-12-31"),
    ("2023-local end", "2023-01-01", "2024-03-28"),
    ("KMLM ETF overlap", "2020-12-03", "2024-03-28"),
]


def nav_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def metrics_from_returns(
    period: str,
    name: str,
    returns: pd.Series,
    spy: pd.Series | None = None,
    margin_daily: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    returns = returns.dropna()
    if len(returns) < 3:
        return {
            "period": period,
            "series": name,
            "start": "",
            "end": "",
            "years": np.nan,
            "total_return": np.nan,
            "cagr": np.nan,
            "ann_return_arithmetic": np.nan,
            "ann_vol": np.nan,
            "sharpe_0rf": np.nan,
            "max_drawdown": np.nan,
            "corr_to_spy": np.nan,
            "pct_days_scaled": np.nan,
            "median_margin_to_equity": np.nan,
            "p95_margin_to_equity": np.nan,
        }
    nav = nav_from_returns(returns)
    years = (returns.index.max() - returns.index.min()).days / 365.25
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * np.sqrt(BUSINESS_DAYS)
    drawdown = nav / nav.cummax() - 1.0
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0.0 else np.nan
    corr_to_spy = np.nan
    if spy is not None:
        aligned = pd.concat({name: returns, "SPY": spy}, axis=1, sort=True).dropna()
        if len(aligned) > 3:
            corr_to_spy = aligned[name].corr(aligned["SPY"])
    if margin_daily is not None and name == "Rob 40 IBKR 500k reset 2016":
        pct_days_scaled = float((margin_daily["margin_scale"] < 0.999).mean())
        median_margin = float(margin_daily["margin_to_equity"].median())
        p95_margin = float(margin_daily["margin_to_equity"].quantile(0.95))
    else:
        pct_days_scaled = np.nan
        median_margin = np.nan
        p95_margin = np.nan
    return {
        "period": period,
        "series": name,
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "years": years,
        "total_return": nav.iloc[-1] - 1.0,
        "cagr": cagr,
        "ann_return_arithmetic": ann_return,
        "ann_vol": ann_vol,
        "sharpe_0rf": ann_return / ann_vol if ann_vol > 0.0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "corr_to_spy": corr_to_spy,
        "pct_days_scaled": pct_days_scaled,
        "median_margin_to_equity": median_margin,
        "p95_margin_to_equity": p95_margin,
    }


def add_equity_fields(daily: pd.DataFrame, capital: float = CAPITAL) -> pd.DataFrame:
    daily = daily.copy()
    daily["equity"] = capital + daily["net_pnl"].cumsum()
    daily["account_return"] = daily["equity"].pct_change().fillna(0.0)
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    return daily


def load_kmlm_streams() -> pd.DataFrame:
    frame = pd.read_csv(KMLM_DIR / "simulated_return_streams.csv", parse_dates=["date"])
    frame = frame.set_index("date").sort_index()
    return frame.rename(
        columns={
            "KMLM public-22 rule": "KMLM public-22 simulated",
            "KMLM rule on 40 no-equity": "KMLM rule on 40 no-equity",
            "KMLM actual": "KMLM actual ETF",
        }
    )


def build_2016_account() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    bt.START_DATE = "1970-01-01"
    instruments = no40.selected_instruments()
    meta = bt.load_meta()
    price = bt.load_price_matrix(instruments)
    fx = bt.load_fx_matrix(instruments, meta, price.index)
    costs = bt.cost_matrix(instruments, meta, price, fx)
    desired = pd.read_csv(MARGIN_DIR / "positions_unconstrained_full.csv", index_col=0, parse_dates=True)
    desired = desired.reindex(price.index).ffill().fillna(0.0)

    schedule = pd.read_csv(MARGIN_DIR / "margin_schedule.csv")
    long_margin, short_margin = margin_bt.margin_per_contract_matrices(
        schedule, instruments, meta, price, fx, mode="current_static"
    )

    period_price = price.loc[START:]
    period_fx = fx.loc[period_price.index]
    period_costs = costs.loc[period_price.index]
    period_desired = desired.loc[period_price.index]
    period_long_margin = long_margin.loc[period_price.index]
    period_short_margin = short_margin.loc[period_price.index]

    constrained_daily, constrained_by_instr = margin_bt.run_margin_constrained_pnl(
        period_desired,
        period_price,
        period_fx,
        meta,
        instruments,
        period_costs,
        period_long_margin,
        period_short_margin,
        CAPITAL,
        margin_limit=1.0,
    )
    constrained_daily["account_return"] = constrained_daily["equity"].pct_change().fillna(0.0)

    unconstrained_daily, _ = bt.pnl_from_positions(period_desired, period_price, period_fx, meta, instruments, period_costs)
    unconstrained_daily = add_equity_fields(unconstrained_daily)

    constrained_daily.to_csv(OUT / "rob40_ibkr_500k_2016_daily.csv")
    constrained_by_instr["position"].to_csv(OUT / "rob40_ibkr_500k_2016_positions.csv")
    unconstrained_daily.to_csv(OUT / "rob40_unconstrained_2016_daily.csv")
    return constrained_daily, unconstrained_daily, instruments


def period_slice(series: pd.Series, start: str, end: str) -> pd.Series:
    return series.loc[start:end].dropna()


def build_metrics(streams: dict[str, pd.Series], margin_daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    spy = streams["SPY"]
    full_actual_kmlm_periods = {"2021 bull", "2022 inflation bear", "2023-local end", "KMLM ETF overlap"}
    for period, start, end in PERIODS:
        for name, returns in streams.items():
            if name == "KMLM actual ETF" and period not in full_actual_kmlm_periods:
                continue
            sub = period_slice(returns, start, end)
            if name == "KMLM actual ETF" and sub.first_valid_index() is None:
                continue
            margin_sub = margin_daily.loc[start:end] if name == "Rob 40 IBKR 500k reset 2016" else None
            rows.append(metrics_from_returns(period, name, sub, period_slice(spy, start, end), margin_sub))
    return pd.DataFrame(rows)


def annual_returns(streams: dict[str, pd.Series]) -> pd.DataFrame:
    aligned = pd.concat(streams, axis=1, sort=True).loc[START:LOCAL_END]
    rows = []
    for year, frame in aligned.groupby(aligned.index.year):
        row = {"year": int(year)}
        for name in aligned.columns:
            series = frame[name].dropna()
            row[name] = nav_from_returns(series).iloc[-1] - 1.0 if len(series) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pct(value: float, digits: int = 1) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}%}"


def num(value: float, digits: int = 2) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def markdown_metrics(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for col in [
        "total_return",
        "cagr",
        "ann_return_arithmetic",
        "ann_vol",
        "max_drawdown",
        "corr_to_spy",
        "pct_days_scaled",
        "median_margin_to_equity",
        "p95_margin_to_equity",
    ]:
        if col in show:
            show[col] = show[col].map(pct)
    for col in ["sharpe_0rf"]:
        if col in show:
            show[col] = show[col].map(num)
    if "years" in show:
        show["years"] = show["years"].map(lambda value: "" if pd.isna(value) else f"{value:.1f}")
    return show.to_markdown(index=False)


def plot_outputs(streams: dict[str, pd.Series], metrics: pd.DataFrame, annual: pd.DataFrame, margin_daily: pd.DataFrame) -> None:
    colors = {
        "Rob 40 IBKR 500k reset 2016": "#0F766E",
        "Rob 40 unconstrained reset 2016": "#64748B",
        "KMLM public-22 simulated": "#7C3AED",
        "KMLM rule on 40 no-equity": "#B45309",
        "KMLM actual ETF": "#111827",
        "SPY": "#2563EB",
    }
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 18),
        gridspec_kw={"height_ratios": [1.25, 0.9, 1.1, 1.0, 1.05]},
    )
    plot_streams = {
        "Rob 40 IBKR 500k reset 2016": streams["Rob 40 IBKR 500k reset 2016"],
        "Rob 40 unconstrained reset 2016": streams["Rob 40 unconstrained reset 2016"],
        "KMLM public-22 simulated": streams["KMLM public-22 simulated"],
        "KMLM rule on 40 no-equity": streams["KMLM rule on 40 no-equity"],
        "SPY": streams["SPY"],
    }
    navs = pd.concat(plot_streams, axis=1, sort=True).loc[START:LOCAL_END].dropna(how="all").apply(nav_from_returns)
    for name in navs.columns:
        axes[0].plot(navs.index, navs[name], color=colors.get(name), label=name)
    axes[0].set_title("Fresh USD 500k Account From 2016")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left", ncols=2)
    axes[0].grid(alpha=0.25)

    drawdowns = navs / navs.cummax() - 1.0
    for name in drawdowns.columns:
        axes[1].plot(drawdowns.index, drawdowns[name], color=colors.get(name), label=name)
    axes[1].set_title("Drawdowns")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    axes[1].grid(alpha=0.25)

    actual_cols = ["Rob 40 IBKR 500k reset 2016", "KMLM actual ETF", "KMLM public-22 simulated", "SPY"]
    actual = pd.concat({name: streams[name] for name in actual_cols}, axis=1, sort=True).loc["2020-12-03":LOCAL_END].dropna()
    actual_navs = actual.apply(nav_from_returns)
    for name in actual_navs.columns:
        axes[2].plot(actual_navs.index, actual_navs[name], color=colors.get(name), label=name)
    axes[2].set_title("Actual KMLM ETF Overlap")
    axes[2].set_ylabel("Growth of $1")
    axes[2].legend(loc="upper left", ncols=2)
    axes[2].grid(alpha=0.25)

    annual_cols = ["Rob 40 IBKR 500k reset 2016", "KMLM public-22 simulated", "KMLM actual ETF", "SPY"]
    annual_cols = [col for col in annual_cols if col in annual.columns]
    x = np.arange(len(annual))
    width = min(0.75 / len(annual_cols), 0.18)
    offsets = np.linspace(-width * (len(annual_cols) - 1) / 2, width * (len(annual_cols) - 1) / 2, len(annual_cols))
    for offset, name in zip(offsets, annual_cols):
        axes[3].bar(x + offset, annual[name] * 100.0, width=width, color=colors.get(name), label=name)
    axes[3].axhline(0.0, color="#666666", linewidth=0.8)
    axes[3].set_title("Calendar-Year Returns")
    axes[3].set_ylabel("Return (%)")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(annual["year"].astype(int).astype(str))
    axes[3].legend(loc="upper left", ncols=2)
    axes[3].grid(axis="y", alpha=0.25)

    margin_daily["margin_to_equity"].rolling(20).mean().plot(ax=axes[4], label="used margin / equity", color="#0F766E")
    margin_daily["desired_margin_to_equity"].rolling(20).mean().plot(
        ax=axes[4], label="desired margin / equity", color="#DC2626", alpha=0.75
    )
    axes[4].axhline(1.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[4].set_title("IBKR Current Initial Margin Pressure")
    axes[4].set_ylabel("20D avg margin / equity")
    axes[4].yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "ibkr_500k_rob40_2016_intervals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ibkr_daily, unconstrained_daily, _instruments = build_2016_account()
    kmlm = load_kmlm_streams()
    streams = {
        "Rob 40 IBKR 500k reset 2016": ibkr_daily["account_return"].rename("Rob 40 IBKR 500k reset 2016"),
        "Rob 40 unconstrained reset 2016": unconstrained_daily["account_return"].rename(
            "Rob 40 unconstrained reset 2016"
        ),
        "KMLM public-22 simulated": kmlm["KMLM public-22 simulated"],
        "KMLM rule on 40 no-equity": kmlm["KMLM rule on 40 no-equity"],
        "KMLM actual ETF": kmlm["KMLM actual ETF"],
        "SPY": kmlm["SPY"],
    }
    metrics = build_metrics(streams, ibkr_daily)
    annual = annual_returns(streams)
    returns = pd.concat(streams, axis=1, sort=True).loc[START:LOCAL_END]
    corr_full = returns.dropna(subset=["Rob 40 IBKR 500k reset 2016", "KMLM public-22 simulated", "SPY"]).corr()
    actual_overlap = returns.loc["2020-12-03":LOCAL_END].dropna(
        subset=["Rob 40 IBKR 500k reset 2016", "KMLM actual ETF", "SPY"]
    )
    corr_actual = actual_overlap.corr()

    metrics.to_csv(OUT / "period_metrics.csv", index=False)
    annual.to_csv(OUT / "annual_returns.csv", index=False)
    returns.to_csv(OUT / "return_streams.csv", index_label="date")
    corr_full.to_csv(OUT / "correlation_2016_full.csv")
    corr_actual.to_csv(OUT / "correlation_kmlm_actual_overlap.csv")
    margin_summary = pd.DataFrame(
        [
            {
                "start": str(ibkr_daily.index.min().date()),
                "end": str(ibkr_daily.index.max().date()),
                "final_equity": ibkr_daily["equity"].iloc[-1],
                "median_margin_to_equity": ibkr_daily["margin_to_equity"].median(),
                "p95_margin_to_equity": ibkr_daily["margin_to_equity"].quantile(0.95),
                "max_margin_to_equity": ibkr_daily["margin_to_equity"].max(),
                "pct_days_scaled": (ibkr_daily["margin_scale"] < 0.999).mean(),
                "max_desired_margin_to_equity": ibkr_daily["desired_margin_to_equity"].max(),
            }
        ]
    )
    margin_summary.to_csv(OUT / "ibkr_margin_summary.csv", index=False)
    plot_outputs(streams, metrics, annual, ibkr_daily)

    key_periods = metrics[
        metrics["period"].isin(["2016-full local", "KMLM ETF overlap"])
        & metrics["series"].isin(["Rob 40 IBKR 500k reset 2016", "KMLM public-22 simulated", "KMLM actual ETF", "SPY"])
    ]
    lines = [
        "# Fresh 2016 IBKR 500k Rob 40 Account By Interval",
        "",
        "## Setup",
        "",
        "- Forecast and Rob risk system are unchanged.",
        "- Account starts fresh on 2016-01-04 with USD 500,000.",
        "- Uses existing Rob 40 target integer positions generated with all pre-2016 price history available.",
        "- Applies IBKR current overnight initial margin as a post-sizing constraint.",
        "- Returns are account-equity percentage returns, not fixed initial-capital P&L returns.",
        "",
        "## Key Metrics",
        "",
        markdown_metrics(key_periods),
        "",
        "## All Period Metrics",
        "",
        markdown_metrics(metrics),
        "",
        "## 2016-Full Correlation",
        "",
        corr_full.to_markdown(),
        "",
        "## KMLM Actual Overlap Correlation",
        "",
        corr_actual.to_markdown(),
        "",
        "## Margin Summary",
        "",
        margin_summary.to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `ibkr_500k_rob40_2016_intervals.png`",
        "- `period_metrics.csv`",
        "- `annual_returns.csv`",
        "- `return_streams.csv`",
        "- `rob40_ibkr_500k_2016_daily.csv`",
        "- `ibkr_margin_summary.csv`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT}")
    print(key_periods.to_string(index=False))
    print(margin_summary.to_string(index=False))


if __name__ == "__main__":
    main()
