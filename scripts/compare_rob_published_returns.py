#!/usr/bin/env python3
"""Compare local Rob-style futures backtests with Rob Carver's published returns.

The published series is Rob's "Me" column from the 2026 annual performance
update. Those figures are public live futures trading returns, reported on an
April-to-March month-end basis and on non-compounded account curves. The local
backtests use the same fixed-capital return convention, so annual returns are
computed as the sum of daily P&L divided by initial capital.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "rob_published_comparison"
ROB_STYLE_DAILY = ROOT / "backtests" / "rob_style_multirule" / "portfolio_daily.csv"
NO_EQUITY_40_DAILY = ROOT / "backtests" / "rob_style_no_equity_40" / "portfolio_daily.csv"

BUSINESS_DAYS = 256.0

# Rob published "Me" column, 2026 annual performance update, vol-adjusted
# benchmark comparison table. Values are return decimals, not percentages.
ROB_PUBLISHED_ME = {
    2015: 0.595,
    2016: 0.281,
    2017: 0.024,
    2018: 0.020,
    2019: 0.045,
    2020: 0.338,
    2021: -0.017,
    2022: 0.258,
    2023: -0.076,
    2024: 0.206,
    2025: -0.154,
    2026: 0.259,
}


@dataclass(frozen=True)
class Stream:
    name: str
    path: Path
    column: str
    label: str


STREAMS = [
    Stream("local_rob_style_buffered", ROB_STYLE_DAILY, "buffered_integer", "Local Rob-style buffered"),
    Stream("local_rob_style_continuous", ROB_STYLE_DAILY, "continuous", "Local Rob-style continuous"),
    Stream("no_equity_40_buffered", NO_EQUITY_40_DAILY, "buffered_integer", "No-equity 40 buffered"),
    Stream("no_equity_40_continuous", NO_EQUITY_40_DAILY, "continuous", "No-equity 40 continuous"),
]


def read_stream(path: Path, top_level_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest output: {path}")
    frame = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if top_level_column not in frame.columns.get_level_values(0):
        raise KeyError(f"{top_level_column!r} not found in {path}")
    out = frame[top_level_column].copy()
    out.index.name = "date"
    return out.sort_index()


def fixed_capital_annual_returns(daily_return: pd.Series, years: list[int]) -> pd.Series:
    rows = {}
    for year in years:
        start = pd.Timestamp(year=year - 1, month=4, day=1)
        end = pd.Timestamp(year=year, month=3, day=31)
        period = daily_return.loc[(daily_return.index >= start) & (daily_return.index <= end)].dropna()
        rows[year] = np.nan if period.empty else float(period.sum())
    return pd.Series(rows)


def full_sample_stats(daily: pd.DataFrame) -> dict[str, float | str]:
    returns = daily["daily_return"].dropna()
    equity = daily["equity"].dropna()
    drawdown = equity / equity.cummax() - 1.0
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    return {
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "years": len(returns) / BUSINESS_DAYS,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else np.nan,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "total_costs": float(daily["costs"].sum()),
    }


def pct(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def num(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def money(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    return f"${value:,.0f}"


def annual_markdown_table(annual: pd.DataFrame) -> str:
    display = annual.reset_index().copy()
    display["year"] = display["year"].astype(int)
    for column in display.columns:
        if column != "year":
            display[column] = display[column].map(pct)
    return display.to_markdown(index=False)


def summary_markdown_table(summary_metrics: pd.DataFrame) -> str:
    display = summary_metrics.copy()
    display["years"] = display["years"].map(lambda value: f"{value:.1f}")
    for column in ["ann_return", "ann_vol", "max_drawdown", "total_return"]:
        display[column] = display[column].map(pct)
    display["sharpe"] = display["sharpe"].map(num)
    display["total_costs"] = display["total_costs"].map(money)
    return display.to_markdown(index=False)


def overlap_markdown_table(overlap_metrics: pd.DataFrame) -> str:
    display = overlap_metrics.copy()
    display["years"] = display["years"].astype(int)
    for column in ["mean_return", "published_mean_return", "mean_difference", "mean_abs_difference"]:
        display[column] = display[column].map(pct)
    display["correlation"] = display["correlation"].map(num)
    display["same_sign_rate"] = display["same_sign_rate"].map(pct)
    return display.to_markdown(index=False)


def write_markdown(
    annual: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
) -> None:
    overlap_years = annual.dropna(subset=["published_rob_live", "local_rob_style_buffered"]).index
    lines = [
        "# Rob Published Returns Comparison",
        "",
        "## Source And Alignment",
        "",
        "- Rob strategy reference: `https://qoppac.blogspot.com/2021/12/my-trading-system.html`.",
        "- Published annual series: Rob's `Me` column from `https://qoppac.blogspot.com/2026/04/annual-performance-update-year-12.html`.",
        "- Published figures are live futures trading returns, not his 2021 in-sample backtest.",
        "- Annual periods are April 1 through March 31, matching Rob's annual update convention.",
        "- Local returns are fixed-capital, non-compounded daily P&L returns summed across each annual period.",
        "- Local CSV data ends at 2024-03-29, so 2025 and 2026 published years are not locally comparable.",
        "- Local Rob-style replica uses cloned `pysystemtrade` Rob config and local CSV data; it is not Rob's private production database.",
        "",
        "## Full-Sample Backtest Metrics",
        "",
        summary_markdown_table(summary_metrics),
        "",
        "## Overlap Fit Versus Rob Published Live Returns",
        "",
        f"Overlap years: {int(overlap_years.min())}-{int(overlap_years.max())}.",
        "",
        overlap_markdown_table(overlap_metrics),
        "",
        "## Annual Return Comparison",
        "",
        annual_markdown_table(annual),
        "",
        "## Interpretation",
        "",
        "- The local Rob-style result does not track Rob's published live annual returns closely; the main causes are data/database differences, missing instruments, simplified local implementation, and the fact that Rob's published series is live trading rather than this compact local replica.",
        "- The no-equity 40 version is a separate design, not a Rob replica. It has much higher local backtest returns and volatility, so it should be compared as an alternative portfolio, not as a fit to Rob's live record.",
        "- The 2021 blog aggregate backtest reports much higher annualized mean and volatility than the local compact run; that gap is expected because Rob explicitly says his full config cannot run on the standard supplied CSV data.",
        "",
        "## Files",
        "",
        "- `annual_returns_comparison.csv`",
        "- `overlap_fit_metrics.csv`",
        "- `full_sample_metrics.csv`",
        "- `published_vs_local_annual_returns.png`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_annual(annual: pd.DataFrame) -> None:
    columns = [
        "published_rob_live",
        "local_rob_style_buffered",
        "no_equity_40_buffered",
    ]
    labels = ["Rob published live", "Local Rob-style", "No-equity 40"]
    colors = ["#222222", "#4477aa", "#cc6677"]
    plot_data = annual[columns] * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.2, 1.1]})
    x = np.arange(len(plot_data.index))
    width = 0.25
    for offset, column, label, color in zip([-width, 0.0, width], columns, labels, colors):
        axes[0].bar(x + offset, plot_data[column], width=width, label=label, color=color)
    axes[0].axhline(0, color="#666666", linewidth=0.8)
    axes[0].set_title("Annual Returns: Rob Published Live vs Local Replicas")
    axes[0].set_ylabel("Annual return (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_data.index.astype(str), rotation=0)
    axes[0].legend(ncols=3, loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)

    cumulative = annual[columns].cumsum() * 100.0
    for column, label, color in zip(columns, labels, colors):
        axes[1].plot(cumulative.index, cumulative[column], marker="o", label=label, color=color)
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set_title("Cumulative Fixed-Capital Return Across Published Annual Windows")
    axes[1].set_ylabel("Cumulative return (%)")
    axes[1].set_xlabel("Year ending March")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "published_vs_local_annual_returns.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    years = list(ROB_PUBLISHED_ME)
    annual = pd.DataFrame(index=pd.Index(years, name="year"))
    annual["published_rob_live"] = pd.Series(ROB_PUBLISHED_ME)

    summary_rows = []
    for stream in STREAMS:
        daily = read_stream(stream.path, stream.column)
        annual[stream.name] = fixed_capital_annual_returns(daily["daily_return"], years)
        stats = full_sample_stats(daily)
        summary_rows.append({"stream": stream.label, **stats})

    annual["local_rob_style_minus_published"] = annual["local_rob_style_buffered"] - annual["published_rob_live"]
    annual["no_equity_40_minus_published"] = annual["no_equity_40_buffered"] - annual["published_rob_live"]

    overlap = annual.dropna(subset=["published_rob_live", "local_rob_style_buffered", "no_equity_40_buffered"])
    fit_rows = []
    for column, label in [
        ("local_rob_style_buffered", "Local Rob-style buffered"),
        ("no_equity_40_buffered", "No-equity 40 buffered"),
    ]:
        diff = overlap[column] - overlap["published_rob_live"]
        fit_rows.append(
            {
                "stream": label,
                "years": len(overlap),
                "mean_return": overlap[column].mean(),
                "published_mean_return": overlap["published_rob_live"].mean(),
                "mean_difference": diff.mean(),
                "mean_abs_difference": diff.abs().mean(),
                "correlation": overlap[[column, "published_rob_live"]].corr().iloc[0, 1],
                "same_sign_rate": (np.sign(overlap[column]) == np.sign(overlap["published_rob_live"])).mean(),
            }
        )

    summary_metrics = pd.DataFrame(summary_rows)
    overlap_metrics = pd.DataFrame(fit_rows)

    annual.to_csv(OUT / "annual_returns_comparison.csv")
    overlap_metrics.to_csv(OUT / "overlap_fit_metrics.csv", index=False)
    summary_metrics.to_csv(OUT / "full_sample_metrics.csv", index=False)
    plot_annual(annual)
    write_markdown(annual, overlap_metrics, summary_metrics)

    print(f"Wrote comparison to {OUT}")
    print(overlap_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
