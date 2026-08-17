#!/usr/bin/env python3
"""Summarise long-history Rob-style futures backtests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "rob_long_history_comparison"
BUSINESS_DAYS = 256.0


@dataclass(frozen=True)
class Stream:
    key: str
    label: str
    daily_path: Path
    weights_path: Path | None = None


STREAMS = [
    Stream(
        "rob_style_2000",
        "Rob-style buffered, 2000 start",
        ROOT / "backtests" / "rob_style_multirule" / "portfolio_daily.csv",
        ROOT / "backtests" / "rob_style_multirule" / "daily_instrument_weights.csv",
    ),
    Stream(
        "rob_style_long",
        "Rob-style buffered, 1970 start",
        ROOT / "backtests" / "rob_style_multirule_long" / "portfolio_daily.csv",
        ROOT / "backtests" / "rob_style_multirule_long" / "daily_instrument_weights.csv",
    ),
    Stream(
        "no_equity_40_2000",
        "No-equity 40 buffered, 2000 start",
        ROOT / "backtests" / "rob_style_no_equity_40" / "portfolio_daily.csv",
        ROOT / "backtests" / "rob_style_no_equity_40" / "daily_instrument_weights.csv",
    ),
    Stream(
        "no_equity_40_long",
        "No-equity 40 buffered, 1970 start",
        ROOT / "backtests" / "rob_style_no_equity_40_long" / "portfolio_daily.csv",
        ROOT / "backtests" / "rob_style_no_equity_40_long" / "daily_instrument_weights.csv",
    ),
]

DECADE_PERIODS = {
    "1970s": ("1970-01-01", "1979-12-31"),
    "1980s": ("1980-01-01", "1989-12-31"),
    "1990s": ("1990-01-01", "1999-12-31"),
    "2000s": ("2000-01-01", "2009-12-31"),
    "2010s": ("2010-01-01", "2019-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
}


def read_buffered_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    daily = frame["buffered_integer"].copy().sort_index()
    daily.index.name = "date"
    return daily


def stats_from_returns(returns: pd.Series, costs: pd.Series | None = None, equity: pd.Series | None = None) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    if equity is None:
        curve = 1.0 + returns.cumsum()
    else:
        curve = equity.loc[returns.index].dropna()
    drawdown = curve / curve.cummax() - 1.0
    years = len(returns) / BUSINESS_DAYS
    total_costs = float(costs.loc[returns.index].sum()) if costs is not None else np.nan
    return {
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "years": years,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else np.nan,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(returns.sum()),
        "total_costs": total_costs,
        "costs_per_year_pct_capital": total_costs / 500000.0 / years if years else np.nan,
    }


def full_sample_metrics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stream in STREAMS:
        daily = data[stream.key]
        rows.append(
            {
                "stream": stream.label,
                **stats_from_returns(daily["daily_return"], daily["costs"], daily["equity"]),
            }
        )
    return pd.DataFrame(rows)


def annual_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    years = sorted({year for daily in data.values() for year in daily.index.year})
    out = pd.DataFrame(index=pd.Index(years, name="year"))
    for stream in STREAMS:
        daily = data[stream.key]
        out[stream.key] = daily["daily_return"].groupby(daily.index.year).sum()
    return out


def decade_metrics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stream_key in ["rob_style_long", "no_equity_40_long"]:
        daily = data[stream_key]
        label = next(stream.label for stream in STREAMS if stream.key == stream_key)
        for period, (start, end) in DECADE_PERIODS.items():
            subset = daily.loc[start:end]
            if subset.empty:
                continue
            rows.append(
                {
                    "stream": label,
                    "period": period,
                    **stats_from_returns(subset["daily_return"], subset["costs"]),
                }
            )
    return pd.DataFrame(rows)


def active_counts_by_year() -> pd.DataFrame:
    rows = []
    for stream in STREAMS:
        if stream.weights_path is None or not stream.weights_path.exists():
            continue
        weights = pd.read_csv(stream.weights_path, index_col=0, parse_dates=True).sort_index()
        active = (weights.fillna(0.0) > 0.0).sum(axis=1)
        yearly = active.groupby(active.index.year).agg(["mean", "min", "max"])
        for year, row in yearly.iterrows():
            rows.append(
                {
                    "stream": stream.label,
                    "year": int(year),
                    "mean_active_instruments": row["mean"],
                    "min_active_instruments": row["min"],
                    "max_active_instruments": row["max"],
                }
            )
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.1%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def money(value: float) -> str:
    return "" if pd.isna(value) else f"${value:,.0f}"


def format_metrics_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    display["years"] = display["years"].map(lambda value: f"{value:.1f}")
    for column in ["ann_return", "ann_vol", "max_drawdown", "total_return", "costs_per_year_pct_capital"]:
        display[column] = display[column].map(pct)
    display["sharpe"] = display["sharpe"].map(num)
    display["total_costs"] = display["total_costs"].map(money)
    return display.to_markdown(index=False)


def plot_results(data: dict[str, pd.DataFrame], annual: pd.DataFrame, active: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [2.0, 1.0, 1.4]})
    colors = {
        "rob_style_long": "#4477aa",
        "no_equity_40_long": "#cc6677",
    }

    for key, label in [
        ("rob_style_long", "Rob-style 1970 start"),
        ("no_equity_40_long", "No-equity 40 1970 start"),
    ]:
        daily = data[key]
        cumulative = daily["daily_return"].cumsum() * 100.0
        axes[0].plot(cumulative.index, cumulative, label=label, color=colors[key])
    axes[0].axhline(0, color="#666666", linewidth=0.8)
    axes[0].set_title("Long-History Fixed-Capital Cumulative Return")
    axes[0].set_ylabel("Return (%)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    for label, color in [
        ("Rob-style buffered, 1970 start", "#4477aa"),
        ("No-equity 40 buffered, 1970 start", "#cc6677"),
    ]:
        subset = active[active["stream"] == label]
        axes[1].plot(subset["year"], subset["mean_active_instruments"], label=label, color=color)
    axes[1].set_title("Mean Active Instruments By Year")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    yearly = annual[["rob_style_long", "no_equity_40_long"]].dropna(how="all") * 100.0
    x = np.arange(len(yearly.index))
    axes[2].bar(x - 0.18, yearly["rob_style_long"], width=0.36, label="Rob-style", color="#4477aa")
    axes[2].bar(x + 0.18, yearly["no_equity_40_long"], width=0.36, label="No-equity 40", color="#cc6677")
    axes[2].axhline(0, color="#666666", linewidth=0.8)
    axes[2].set_title("Calendar-Year Returns")
    axes[2].set_ylabel("Return (%)")
    axes[2].set_xlabel("Calendar year")
    ticks = [i for i, year in enumerate(yearly.index) if year % 5 == 0]
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels([str(yearly.index[i]) for i in ticks], rotation=0)
    axes[2].legend(loc="upper left")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "long_history_equity_active_annual.png", dpi=180)
    plt.close(fig)


def write_markdown(metrics: pd.DataFrame, decades: pd.DataFrame, active: pd.DataFrame) -> None:
    metric_by_stream = metrics.set_index("stream")
    rob_2000 = metric_by_stream.loc["Rob-style buffered, 2000 start"]
    rob_long = metric_by_stream.loc["Rob-style buffered, 1970 start"]
    noeq_2000 = metric_by_stream.loc["No-equity 40 buffered, 2000 start"]
    noeq_long = metric_by_stream.loc["No-equity 40 buffered, 1970 start"]
    decade_by_stream_period = decades.set_index(["stream", "period"])
    rob_1970s = decade_by_stream_period.loc[("Rob-style buffered, 1970 start", "1970s")]
    noeq_1970s = decade_by_stream_period.loc[("No-equity 40 buffered, 1970 start", "1970s")]

    active_checkpoints = active[
        active["stream"].isin(["Rob-style buffered, 1970 start", "No-equity 40 buffered, 1970 start"])
        & active["year"].isin([1970, 1980, 1990, 2000, 2010, 2020, 2024])
    ].copy()
    active_checkpoints["mean_active_instruments"] = active_checkpoints["mean_active_instruments"].map(
        lambda value: f"{value:.1f}"
    )

    lines = [
        "# Long-History Rob-Style Futures Comparison",
        "",
        "## Scope",
        "",
        "- Re-ran the local Rob-style replica from `1970-02-03`, the first date available among the local Rob-config CSV files.",
        "- Re-ran the no-equity 40-instrument design from the same start date.",
        "- Forecast/risk logic is unchanged; data alignment was fixed so instruments require same-day price availability and vol/FX are not backfilled from the future.",
        "- Early history has a much smaller active universe, so 1970s/1980s results are less diversified than the 2000+ run.",
        "- Local back-adjusted futures prices can be negative in old history; this is normal for return/trend work but still a caveat for this compact risk-overlay implementation.",
        "",
        "## Full-Sample Metrics",
        "",
        format_metrics_table(metrics),
        "",
        "## Decade Metrics, Long Runs Only",
        "",
        format_metrics_table(decades),
        "",
        "## Active Instrument Checkpoints",
        "",
        active_checkpoints.to_markdown(index=False),
        "",
        "## Main Takeaways",
        "",
        f"- Extending Rob-style from 2000 to 1970 raises annual return from {pct(rob_2000['ann_return'])} to {pct(rob_long['ann_return'])} and Sharpe from {num(rob_2000['sharpe'])} to {num(rob_long['sharpe'])}, but worsens max drawdown from {pct(rob_2000['max_drawdown'])} to {pct(rob_long['max_drawdown'])}.",
        f"- Extending no-equity 40 from 2000 to 1970 raises annual return from {pct(noeq_2000['ann_return'])} to {pct(noeq_long['ann_return'])} and Sharpe from {num(noeq_2000['sharpe'])} to {num(noeq_long['sharpe'])}, but worsens max drawdown from {pct(noeq_2000['max_drawdown'])} to {pct(noeq_long['max_drawdown'])}.",
        f"- The 1970s are the dominant reason the long run looks so strong: Rob-style returns {pct(rob_1970s['ann_return'])} annualized and no-equity 40 returns {pct(noeq_1970s['ann_return'])}, while average active instruments in 1970 are only 3.3 and 2.6 respectively.",
        "- Treat the long-history result as stress/context, not deployable proof: early decades have few markets, local data is back-adjusted, and this compact implementation is still not Rob's production database/execution stack.",
        "",
        "## Files",
        "",
        "- `full_sample_metrics.csv`",
        "- `decade_metrics.csv`",
        "- `annual_returns.csv`",
        "- `active_counts_by_year.csv`",
        "- `long_history_equity_active_annual.png`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {stream.key: read_buffered_daily(stream.daily_path) for stream in STREAMS}
    metrics = full_sample_metrics(data)
    annual = annual_returns(data)
    decades = decade_metrics(data)
    active = active_counts_by_year()

    metrics.to_csv(OUT / "full_sample_metrics.csv", index=False)
    annual.to_csv(OUT / "annual_returns.csv")
    decades.to_csv(OUT / "decade_metrics.csv", index=False)
    active.to_csv(OUT / "active_counts_by_year.csv", index=False)
    plot_results(data, annual, active)
    write_markdown(metrics, decades, active)

    print(f"Wrote long-history comparison to {OUT}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
