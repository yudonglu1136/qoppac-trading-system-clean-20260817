#!/usr/bin/env python3
"""Compare the 17-instrument custom strategy, no-equity 40, and SPY."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {
    "17 selected": ROOT / "backtests" / "rob_style_us_rates_selected_no_vol" / "portfolio_daily.csv",
    "40 no-equity": ROOT / "backtests" / "rob_style_no_equity_40" / "portfolio_daily.csv",
}
OUT = ROOT / "backtests" / "selected_no_vol_vs_40_vs_spy"
YAHOO_HISTORY_PAGE = "https://finance.yahoo.com/quote/SPY/history/"
TRADING_DAYS = 252.0
ROLLING_WINDOW = 126

CRISIS_WINDOWS = [
    ("Dot-com bear", "2000-03-24", "2002-10-09"),
    ("Global financial crisis", "2007-10-09", "2009-03-09"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 inflation bear", "2022-01-03", "2022-10-12"),
]

COLORS = {
    "17 selected": "#1f4e79",
    "40 no-equity": "#6f8f2f",
    "SPY": "#c97917",
}


def yahoo_chart_url(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    start_dt = dt.datetime.combine(start.date(), dt.time(0, 0))
    end_dt = dt.datetime.combine((end + pd.Timedelta(days=1)).date(), dt.time(0, 0))
    params = urlencode(
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "history|div|splits",
            "includeAdjustedClose": "true",
        }
    )
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{params}"


def fetch_spy_adjusted_close(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, str]:
    url = yahoo_chart_url("SPY", start, end)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        data = json.loads(response.read())

    result = data["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s").normalize()
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]
    series = pd.Series(adj_close, index=dates, name="spy_adjusted_close", dtype=float).dropna()
    return series[~series.index.duplicated(keep="last")], url


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def load_strategy_equity(path: Path) -> pd.Series:
    portfolio = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    equity = portfolio[("buffered_integer", "equity")].dropna()
    return equity


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    equities = {name: load_strategy_equity(path) for name, path in STRATEGIES.items()}
    common_start = max(series.index.min() for series in equities.values())
    common_end = min(series.index.max() for series in equities.values())
    spy_close, source_url = fetch_spy_adjusted_close(common_start, common_end)
    comparison_index = spy_close.loc[common_start:common_end].index

    daily = pd.DataFrame(index=comparison_index)
    for name, equity in equities.items():
        on_dates = equity.reindex(comparison_index, method="ffill")
        daily[f"{name}_nav"] = on_dates / on_dates.dropna().iloc[0]
        daily[f"{name}_return"] = daily[f"{name}_nav"].pct_change()
        daily[f"{name}_drawdown"] = drawdown(daily[f"{name}_nav"])

    daily["SPY_nav"] = spy_close.reindex(comparison_index) / spy_close.reindex(comparison_index).dropna().iloc[0]
    daily["SPY_return"] = daily["SPY_nav"].pct_change()
    daily["SPY_drawdown"] = drawdown(daily["SPY_nav"])

    for name in STRATEGIES:
        daily[f"{name}_rolling_corr_126d"] = daily[f"{name}_return"].rolling(ROLLING_WINDOW).corr(
            daily["SPY_return"]
        )
        daily[f"{name}_rolling_beta_126d"] = daily[f"{name}_return"].rolling(ROLLING_WINDOW).cov(
            daily["SPY_return"]
        ) / daily["SPY_return"].rolling(ROLLING_WINDOW).var()
    daily.index.name = "date"

    metrics = pd.DataFrame(
        [{"series": name, **performance_metrics(daily[f"{name}_nav"])} for name in [*STRATEGIES.keys(), "SPY"]]
    )
    crisis = crisis_metrics(daily)
    tail = tail_day_metrics(daily)
    annual = annual_return_metrics(daily)
    return daily, metrics, crisis, tail, annual, source_url


def performance_metrics(nav: pd.Series) -> dict[str, float | str]:
    nav = nav.dropna()
    returns = nav.pct_change().dropna()
    elapsed_years = (nav.index[-1] - nav.index[0]).days / 365.25
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0
    annual_return = returns.mean() * TRADING_DAYS
    annual_vol = returns.std() * math.sqrt(TRADING_DAYS)
    max_dd = drawdown(nav).min()
    return {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "years": elapsed_years,
        "total_return": total_return,
        "cagr": cagr,
        "annual_return_arithmetic": annual_return,
        "annual_vol": annual_vol,
        "sharpe_0rf": annual_return / annual_vol if annual_vol else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd else np.nan,
    }


def crisis_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crisis_name, start_s, end_s in CRISIS_WINDOWS:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        window = daily.loc[start:end].dropna(subset=["SPY_return"])
        if window.empty:
            continue
        actual_start = window.index[0]
        actual_end = window.index[-1]
        spy_return = daily.loc[actual_end, "SPY_nav"] / daily.loc[actual_start, "SPY_nav"] - 1.0
        rows.append(
            {
                "crisis": crisis_name,
                "series": "SPY",
                "start": str(actual_start.date()),
                "end": str(actual_end.date()),
                "days": len(window),
                "period_return": spy_return,
                "hedge_spread_vs_spy": 0.0,
                "daily_correlation_to_spy": 1.0,
                "beta_to_spy": 1.0,
                "max_drawdown": window["SPY_drawdown"].min(),
            }
        )
        for name in STRATEGIES:
            strategy_window = window.dropna(subset=[f"{name}_return", "SPY_return"])
            strategy_return = daily.loc[actual_end, f"{name}_nav"] / daily.loc[actual_start, f"{name}_nav"] - 1.0
            corr = strategy_window[f"{name}_return"].corr(strategy_window["SPY_return"])
            beta = strategy_window[f"{name}_return"].cov(strategy_window["SPY_return"]) / strategy_window[
                "SPY_return"
            ].var()
            rows.append(
                {
                    "crisis": crisis_name,
                    "series": name,
                    "start": str(actual_start.date()),
                    "end": str(actual_end.date()),
                    "days": len(strategy_window),
                    "period_return": strategy_return,
                    "hedge_spread_vs_spy": strategy_return - spy_return,
                    "daily_correlation_to_spy": corr,
                    "beta_to_spy": beta,
                    "max_drawdown": strategy_window[f"{name}_drawdown"].min(),
                }
            )
    return pd.DataFrame(rows)


def tail_day_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buckets = [
        ("All days", daily["SPY_return"].notna()),
        ("SPY < 0%", daily["SPY_return"] < 0.0),
        ("SPY <= -1%", daily["SPY_return"] <= -0.01),
        ("SPY <= -2%", daily["SPY_return"] <= -0.02),
        ("SPY <= -3%", daily["SPY_return"] <= -0.03),
    ]
    for bucket, mask in buckets:
        window = daily.loc[mask].dropna(subset=["SPY_return"])
        rows.append(
            {
                "bucket": bucket,
                "series": "SPY",
                "days": len(window),
                "avg_daily_return": window["SPY_return"].mean(),
                "positive_hit_rate": (window["SPY_return"] > 0).mean(),
                "daily_correlation_to_spy": 1.0,
            }
        )
        for name in STRATEGIES:
            strategy_window = window.dropna(subset=[f"{name}_return", "SPY_return"])
            rows.append(
                {
                    "bucket": bucket,
                    "series": name,
                    "days": len(strategy_window),
                    "avg_daily_return": strategy_window[f"{name}_return"].mean(),
                    "positive_hit_rate": (strategy_window[f"{name}_return"] > 0).mean(),
                    "daily_correlation_to_spy": strategy_window[f"{name}_return"].corr(
                        strategy_window["SPY_return"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def annual_return_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in daily.groupby(daily.index.year):
        row = {
            "year": int(year),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "days": len(frame),
        }
        for name in [*STRATEGIES.keys(), "SPY"]:
            nav = frame[f"{name}_nav"].dropna()
            row[f"{name}_return"] = nav.iloc[-1] / nav.iloc[0] - 1.0 if len(nav) >= 2 else np.nan
        row["17 selected_minus_SPY"] = row["17 selected_return"] - row["SPY_return"]
        row["40 no-equity_minus_SPY"] = row["40 no-equity_return"] - row["SPY_return"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sample_note"] = ""
        out.loc[out.index.min(), "sample_note"] = "partial first year"
        out.loc[out.index.max(), "sample_note"] = "partial final year"
    return out


def pct_formatter(x: float, _pos: int) -> str:
    return f"{x:.0%}"


def nav_formatter(x: float, _pos: int) -> str:
    return f"{x:.0f}x"


def shade_crises(ax: plt.Axes, label_first: bool = False) -> None:
    for idx, (_name, start, end) in enumerate(CRISIS_WINDOWS):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#9aa3ad", alpha=0.15, lw=0)
        if label_first and idx == 0:
            ylim = ax.get_ylim()
            ax.text(
                pd.Timestamp(start),
                ylim[1],
                "crisis windows",
                color="#5f6872",
                fontsize=9,
                va="bottom",
            )


def plot_outputs(
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    crisis: pd.DataFrame,
    tail: pd.DataFrame,
    annual: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#c5cbd3",
            "axes.labelcolor": "#2f343b",
            "axes.titleweight": "bold",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    neutral = "#626b76"

    fig = plt.figure(figsize=(18, 21), constrained_layout=False)
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1.28, 0.95, 0.85, 1.0, 1.05], hspace=0.52, wspace=0.18)
    ax_nav = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_nav)
    ax_corr = fig.add_subplot(gs[2, 0], sharex=ax_nav)
    ax_beta = fig.add_subplot(gs[2, 1], sharex=ax_nav)
    ax_crisis = fig.add_subplot(gs[3, 0])
    ax_table = fig.add_subplot(gs[3, 1])
    ax_annual = fig.add_subplot(gs[4, :])

    ordered_series = ["17 selected", "40 no-equity", "SPY"]
    for name in ordered_series:
        ax_nav.plot(daily.index, daily[f"{name}_nav"], color=COLORS[name], lw=2.0, label=name)
    shade_crises(ax_nav, label_first=True)
    ax_nav.set_title("17 Selected Strategy vs 40 No-Equity Strategy vs SPY")
    ax_nav.set_ylabel("Growth of $1")
    ax_nav.yaxis.set_major_formatter(nav_formatter)
    ax_nav.grid(True, color="#e7ebef", lw=0.8)
    ax_nav.legend(loc="upper left", frameon=False, ncol=3)
    ax_nav.text(
        0.99,
        0.04,
        "Both strategies are buffered-integer backtests; SPY uses adjusted close.",
        transform=ax_nav.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=neutral,
    )

    for name in ordered_series:
        ax_dd.plot(daily.index, daily[f"{name}_drawdown"], color=COLORS[name], lw=1.6, label=name)
    shade_crises(ax_dd)
    ax_dd.axhline(0, color="#9aa3ad", lw=0.8)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_dd.grid(True, color="#e7ebef", lw=0.8)
    ax_dd.legend(loc="lower left", frameon=False, ncol=3)

    for name in STRATEGIES:
        ax_corr.plot(
            daily.index,
            daily[f"{name}_rolling_corr_126d"],
            color=COLORS[name],
            lw=1.6,
            label=f"{name} vs SPY",
        )
    shade_crises(ax_corr)
    ax_corr.axhline(0, color="#6b7280", lw=1.0)
    ax_corr.set_title("126-Day Rolling Correlation To SPY")
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_ylim(-0.8, 0.8)
    ax_corr.grid(True, color="#e7ebef", lw=0.8)
    ax_corr.legend(loc="lower left", frameon=False)

    for name in STRATEGIES:
        ax_beta.plot(
            daily.index,
            daily[f"{name}_rolling_beta_126d"],
            color=COLORS[name],
            lw=1.6,
            label=f"{name} vs SPY",
        )
    shade_crises(ax_beta)
    ax_beta.axhline(0, color="#6b7280", lw=1.0)
    ax_beta.set_title("126-Day Rolling Beta To SPY")
    ax_beta.set_ylabel("Beta")
    ax_beta.grid(True, color="#e7ebef", lw=0.8)
    ax_beta.legend(loc="lower left", frameon=False)

    crisis_pivot = crisis.pivot(index="crisis", columns="series", values="period_return").loc[
        [name for name, _start, _end in CRISIS_WINDOWS]
    ]
    y = np.arange(len(crisis_pivot))
    bar_h = 0.23
    offsets = {
        "17 selected": -bar_h,
        "40 no-equity": 0.0,
        "SPY": bar_h,
    }
    for name in ordered_series:
        ax_crisis.barh(
            y + offsets[name],
            crisis_pivot[name],
            height=bar_h,
            color=COLORS[name],
            label=name,
        )
    ax_crisis.axvline(0, color="#6b7280", lw=1.0)
    ax_crisis.set_yticks(y)
    ax_crisis.set_yticklabels(crisis_pivot.index)
    ax_crisis.invert_yaxis()
    ax_crisis.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_crisis.set_title("Crisis-Window Total Returns")
    ax_crisis.set_xlabel("Period return")
    ax_crisis.grid(True, axis="x", color="#e7ebef", lw=0.8)
    ax_crisis.legend(loc="lower right", frameon=False)

    ax_table.axis("off")
    table_rows = []
    for crisis_name, _start, _end in CRISIS_WINDOWS:
        c = crisis[crisis["crisis"].eq(crisis_name)].set_index("series")
        table_rows.append(
            [
                crisis_name,
                f"{c.loc['SPY', 'period_return']:.1%}",
                f"{c.loc['17 selected', 'period_return']:.1%}",
                f"{c.loc['40 no-equity', 'period_return']:.1%}",
                f"{c.loc['17 selected', 'daily_correlation_to_spy']:.2f}",
                f"{c.loc['40 no-equity', 'daily_correlation_to_spy']:.2f}",
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Crisis", "SPY", "17", "40", "Corr 17", "Corr 40"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.36, 0.12, 0.12, 0.12, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dde3")
        if row == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(weight="bold", color="#26313d")
        elif col == 0:
            cell.set_text_props(ha="left")
    ax_table.set_title("Crisis Returns And Correlation To SPY")

    x_annual = np.arange(len(annual))
    annual_width = 0.25
    annual_offsets = {
        "17 selected": -annual_width,
        "40 no-equity": 0.0,
        "SPY": annual_width,
    }
    for name in ordered_series:
        ax_annual.bar(
            x_annual + annual_offsets[name],
            annual[f"{name}_return"],
            width=annual_width,
            color=COLORS[name],
            label=name,
        )
    ax_annual.axhline(0, color="#6b7280", lw=1.0)
    ax_annual.set_xticks(x_annual)
    ax_annual.set_xticklabels(annual["year"].astype(str), rotation=45, ha="right")
    ax_annual.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_annual.set_title("Calendar-Year Returns")
    ax_annual.set_ylabel("Year return")
    ax_annual.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_annual.legend(loc="upper left", frameon=False, ncol=3)
    ax_annual.text(
        0.99,
        0.04,
        "2000 and 2024 are partial sample years.",
        transform=ax_annual.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=neutral,
    )

    for ax in [ax_nav, ax_dd, ax_corr, ax_beta]:
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "17 Selected Futures Strategy vs 40 No-Equity Strategy vs SPY",
        fontsize=17,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.02,
        0.967,
        "Daily comparison on SPY trading dates, 2000-01-19 to 2024-03-28. Crisis windows are shaded in the time-series panels.",
        fontsize=10,
        color=neutral,
        ha="left",
    )
    fig.subplots_adjust(top=0.935, left=0.13, right=0.98, bottom=0.07)
    fig.savefig(OUT / "selected_17_vs_40_vs_spy_big.png", dpi=180)
    fig.savefig(OUT / "selected_17_vs_40_vs_spy_big.pdf")
    plt.close(fig)


def format_metric(key: str, value: float | str) -> str:
    if key in {"start", "end"}:
        return str(value)
    if key == "years":
        return f"{value:.1f}"
    if key in {"sharpe_0rf", "calmar"}:
        return f"{value:.2f}"
    return f"{value:.2%}"


def write_summary(
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    crisis: pd.DataFrame,
    tail: pd.DataFrame,
    annual: pd.DataFrame,
    source_url: str,
) -> None:
    display_keys = [
        "start",
        "end",
        "years",
        "total_return",
        "cagr",
        "annual_vol",
        "sharpe_0rf",
        "max_drawdown",
        "calmar",
    ]
    corr_17 = daily["17 selected_return"].corr(daily["SPY_return"])
    corr_40 = daily["40 no-equity_return"].corr(daily["SPY_return"])
    beta_17 = daily["17 selected_return"].cov(daily["SPY_return"]) / daily["SPY_return"].var()
    beta_40 = daily["40 no-equity_return"].cov(daily["SPY_return"]) / daily["SPY_return"].var()
    years_17_wins_spy = (annual["17 selected_return"] > annual["SPY_return"]).sum()
    years_40_wins_spy = (annual["40 no-equity_return"] > annual["SPY_return"]).sum()
    years_40_wins_17 = (annual["40 no-equity_return"] > annual["17 selected_return"]).sum()

    lines = [
        "# 17 Selected Strategy vs 40 No-Equity Strategy vs SPY",
        "",
        "## Data",
        "",
        f"- 17 selected strategy: `{STRATEGIES['17 selected']}`",
        f"- 40 no-equity strategy: `{STRATEGIES['40 no-equity']}`",
        f"- SPY source: Yahoo Finance adjusted close, `{YAHOO_HISTORY_PAGE}`",
        f"- SPY chart endpoint used: `{source_url}`",
        "- Daily comparisons use SPY trading dates. Strategy values are forward-filled onto those dates.",
        "",
        "## Headline Metrics",
        "",
        "| Series | " + " | ".join(display_keys) + " |",
        "|---" * (len(display_keys) + 1) + "|",
    ]
    for row in metrics.to_dict("records"):
        values = [row["series"]] + [format_metric(key, row[key]) for key in display_keys]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## SPY Correlation",
            "",
            f"- 17 selected full-sample daily correlation to SPY: {corr_17:.2f}; beta: {beta_17:.2f}",
            f"- 40 no-equity full-sample daily correlation to SPY: {corr_40:.2f}; beta: {beta_40:.2f}",
            "",
            "## Crisis Windows",
            "",
            "| Crisis | SPY | 17 selected | 40 no-equity | Corr 17 | Corr 40 | Beta 17 | Beta 40 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for crisis_name, _start, _end in CRISIS_WINDOWS:
        c = crisis[crisis["crisis"].eq(crisis_name)].set_index("series")
        lines.append(
            "| "
            + " | ".join(
                [
                    crisis_name,
                    f"{c.loc['SPY', 'period_return']:.1%}",
                    f"{c.loc['17 selected', 'period_return']:.1%}",
                    f"{c.loc['40 no-equity', 'period_return']:.1%}",
                    f"{c.loc['17 selected', 'daily_correlation_to_spy']:.2f}",
                    f"{c.loc['40 no-equity', 'daily_correlation_to_spy']:.2f}",
                    f"{c.loc['17 selected', 'beta_to_spy']:.2f}",
                    f"{c.loc['40 no-equity', 'beta_to_spy']:.2f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Calendar-Year Read",
            "",
            f"- 17 selected beat SPY in {years_17_wins_spy:.0f} / {len(annual):.0f} calendar-year rows.",
            f"- 40 no-equity beat SPY in {years_40_wins_spy:.0f} / {len(annual):.0f} calendar-year rows.",
            f"- 40 no-equity beat 17 selected in {years_40_wins_17:.0f} / {len(annual):.0f} calendar-year rows.",
            "- 2000 and 2024 are partial sample years.",
            "",
            "## Outputs",
            "",
            "- Big chart: `selected_17_vs_40_vs_spy_big.png`",
            "- Daily data: `daily_comparison.csv`",
            "- Metrics: `metrics.csv`",
            "- Crisis metrics: `crisis_window_metrics.csv`",
            "- Tail-day metrics: `tail_day_metrics.csv`",
            "- Calendar-year returns: `annual_return_metrics.csv`",
            "",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily, metrics, crisis, tail, annual, source_url = build_comparison()
    daily.to_csv(OUT / "daily_comparison.csv")
    metrics.to_csv(OUT / "metrics.csv", index=False)
    crisis.to_csv(OUT / "crisis_window_metrics.csv", index=False)
    tail.to_csv(OUT / "tail_day_metrics.csv", index=False)
    annual.to_csv(OUT / "annual_return_metrics.csv", index=False)
    plot_outputs(daily, metrics, crisis, tail, annual)
    write_summary(daily, metrics, crisis, tail, annual, source_url)

    print(f"Wrote comparison to {OUT}")
    print(metrics.to_string(index=False))
    print()
    print(crisis.to_string(index=False))
    print()
    print(annual.to_string(index=False))


if __name__ == "__main__":
    main()
