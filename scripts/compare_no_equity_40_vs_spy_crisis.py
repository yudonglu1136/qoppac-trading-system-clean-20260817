#!/usr/bin/env python3
"""Compare the no-equity 40-instrument strategy with SPY and crisis windows."""

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
BACKTEST = ROOT / "backtests" / "rob_style_no_equity_40" / "portfolio_daily.csv"
OUT = ROOT / "backtests" / "rob_style_no_equity_40_spy_crisis_comparison"
YAHOO_HISTORY_PAGE = "https://finance.yahoo.com/quote/SPY/history/"
TRADING_DAYS = 252.0
ROLLING_WINDOW = 126

CRISIS_WINDOWS = [
    ("Dot-com bear", "2000-03-24", "2002-10-09"),
    ("Global financial crisis", "2007-10-09", "2009-03-09"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 inflation bear", "2022-01-03", "2022-10-12"),
]


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


def load_strategy() -> pd.DataFrame:
    portfolio = pd.read_csv(BACKTEST, header=[0, 1], index_col=0, parse_dates=True)
    out = pd.DataFrame(
        {
            "strategy_equity": portfolio[("buffered_integer", "equity")],
            "strategy_daily_return_on_500k": portfolio[("buffered_integer", "daily_return")],
        }
    ).dropna()
    return out


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    strategy = load_strategy()
    spy_close, source_url = fetch_spy_adjusted_close(strategy.index.min(), strategy.index.max())

    start = max(strategy.index.min(), spy_close.index.min())
    end = min(strategy.index.max(), spy_close.index.max())
    index = spy_close.loc[start:end].index
    strategy_on_spy_dates = strategy.reindex(index, method="ffill")
    spy_on_dates = spy_close.reindex(index).dropna()
    strategy_on_spy_dates = strategy_on_spy_dates.reindex(spy_on_dates.index, method="ffill")

    daily = pd.DataFrame(index=spy_on_dates.index)
    daily["strategy_equity"] = strategy_on_spy_dates["strategy_equity"]
    daily["strategy_nav"] = daily["strategy_equity"] / daily["strategy_equity"].iloc[0]
    daily["strategy_return"] = daily["strategy_nav"].pct_change()
    daily["strategy_return_on_500k"] = strategy_on_spy_dates["strategy_daily_return_on_500k"]
    daily["spy_adjusted_close"] = spy_on_dates
    daily["spy_nav"] = daily["spy_adjusted_close"] / daily["spy_adjusted_close"].iloc[0]
    daily["spy_return"] = daily["spy_nav"].pct_change()
    daily["strategy_drawdown"] = drawdown(daily["strategy_nav"])
    daily["spy_drawdown"] = drawdown(daily["spy_nav"])
    daily["rolling_corr_126d"] = daily["strategy_return"].rolling(ROLLING_WINDOW).corr(daily["spy_return"])
    daily["rolling_beta_126d"] = daily["strategy_return"].rolling(ROLLING_WINDOW).cov(
        daily["spy_return"]
    ) / daily["spy_return"].rolling(ROLLING_WINDOW).var()
    daily.index.name = "date"

    metrics = pd.DataFrame(
        [
            {"series": "No-equity strategy", **performance_metrics(daily["strategy_nav"])},
            {"series": "SPY", **performance_metrics(daily["spy_nav"])},
        ]
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
    for name, start_s, end_s in CRISIS_WINDOWS:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        window = daily.loc[start:end].dropna(subset=["strategy_return", "spy_return"])
        if window.empty:
            continue
        actual_start = window.index[0]
        actual_end = window.index[-1]
        strategy_return = daily.loc[actual_end, "strategy_nav"] / daily.loc[actual_start, "strategy_nav"] - 1.0
        spy_return = daily.loc[actual_end, "spy_nav"] / daily.loc[actual_start, "spy_nav"] - 1.0
        corr = window["strategy_return"].corr(window["spy_return"])
        beta = window["strategy_return"].cov(window["spy_return"]) / window["spy_return"].var()
        rows.append(
            {
                "crisis": name,
                "start": str(actual_start.date()),
                "end": str(actual_end.date()),
                "days": len(window),
                "strategy_return": strategy_return,
                "spy_return": spy_return,
                "hedge_spread": strategy_return - spy_return,
                "offset_ratio_vs_spy_loss": strategy_return / abs(spy_return) if spy_return < 0 else np.nan,
                "daily_correlation": corr,
                "beta_to_spy": beta,
                "strategy_max_drawdown": window["strategy_drawdown"].min(),
                "spy_max_drawdown": window["spy_drawdown"].min(),
            }
        )
    return pd.DataFrame(rows)


def tail_day_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buckets = [
        ("All days", daily["spy_return"].notna()),
        ("SPY < 0%", daily["spy_return"] < 0.0),
        ("SPY <= -1%", daily["spy_return"] <= -0.01),
        ("SPY <= -2%", daily["spy_return"] <= -0.02),
        ("SPY <= -3%", daily["spy_return"] <= -0.03),
    ]
    for label, mask in buckets:
        window = daily.loc[mask].dropna(subset=["strategy_return", "spy_return"])
        rows.append(
            {
                "bucket": label,
                "days": len(window),
                "strategy_avg_daily_return": window["strategy_return"].mean(),
                "spy_avg_daily_return": window["spy_return"].mean(),
                "strategy_positive_hit_rate": (window["strategy_return"] > 0).mean(),
                "daily_correlation": window["strategy_return"].corr(window["spy_return"]),
            }
        )
    return pd.DataFrame(rows)


def annual_return_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in daily.groupby(daily.index.year):
        strategy_nav = frame["strategy_nav"].dropna()
        spy_nav = frame["spy_nav"].dropna()
        if len(strategy_nav) < 2 or len(spy_nav) < 2:
            continue
        rows.append(
            {
                "year": int(year),
                "start": str(frame.index.min().date()),
                "end": str(frame.index.max().date()),
                "days": len(frame),
                "strategy_return": strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1.0,
                "spy_return": spy_nav.iloc[-1] / spy_nav.iloc[0] - 1.0,
                "spread": strategy_nav.iloc[-1] / strategy_nav.iloc[0] - spy_nav.iloc[-1] / spy_nav.iloc[0],
            }
        )
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
    for idx, (name, start, end) in enumerate(CRISIS_WINDOWS):
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
    strategy_color = "#1f4e79"
    spy_color = "#c97917"
    neutral = "#626b76"

    fig = plt.figure(figsize=(18, 20), constrained_layout=False)
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1.28, 0.95, 0.85, 1.0, 1.05], hspace=0.52, wspace=0.18)
    ax_nav = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_nav)
    ax_corr = fig.add_subplot(gs[2, 0], sharex=ax_nav)
    ax_beta = fig.add_subplot(gs[2, 1], sharex=ax_nav)
    ax_crisis = fig.add_subplot(gs[3, 0])
    ax_table = fig.add_subplot(gs[3, 1])
    ax_annual = fig.add_subplot(gs[4, :])

    ax_nav.plot(daily.index, daily["strategy_nav"], color=strategy_color, lw=2.2, label="No-equity strategy")
    ax_nav.plot(daily.index, daily["spy_nav"], color=spy_color, lw=1.9, label="SPY adjusted close")
    shade_crises(ax_nav, label_first=True)
    ax_nav.set_title("No-Equity 40-Instrument Strategy vs SPY")
    ax_nav.set_ylabel("Growth of $1")
    ax_nav.yaxis.set_major_formatter(nav_formatter)
    ax_nav.grid(True, color="#e7ebef", lw=0.8)
    ax_nav.legend(loc="upper left", frameon=False, ncol=2)
    ax_nav.text(
        0.99,
        0.04,
        "Strategy is the buffered-integer fixed-notional backtest; SPY uses adjusted close.",
        transform=ax_nav.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=neutral,
    )

    ax_dd.plot(daily.index, daily["strategy_drawdown"], color=strategy_color, lw=1.8, label="No-equity strategy")
    ax_dd.plot(daily.index, daily["spy_drawdown"], color=spy_color, lw=1.6, label="SPY")
    shade_crises(ax_dd)
    ax_dd.axhline(0, color="#9aa3ad", lw=0.8)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_dd.grid(True, color="#e7ebef", lw=0.8)
    ax_dd.legend(loc="lower left", frameon=False, ncol=2)

    ax_corr.plot(daily.index, daily["rolling_corr_126d"], color="#325f84", lw=1.7)
    shade_crises(ax_corr)
    ax_corr.axhline(0, color="#6b7280", lw=1.0)
    ax_corr.set_title("126-Day Rolling Correlation")
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_ylim(-0.8, 0.8)
    ax_corr.grid(True, color="#e7ebef", lw=0.8)

    ax_beta.plot(daily.index, daily["rolling_beta_126d"], color="#8a5a13", lw=1.7)
    shade_crises(ax_beta)
    ax_beta.axhline(0, color="#6b7280", lw=1.0)
    ax_beta.set_title("126-Day Rolling Beta To SPY")
    ax_beta.set_ylabel("Beta")
    ax_beta.grid(True, color="#e7ebef", lw=0.8)

    crisis_plot = crisis.copy()
    y = np.arange(len(crisis_plot))
    bar_h = 0.34
    ax_crisis.barh(y - bar_h / 2, crisis_plot["strategy_return"], height=bar_h, color=strategy_color, label="Strategy")
    ax_crisis.barh(y + bar_h / 2, crisis_plot["spy_return"], height=bar_h, color=spy_color, label="SPY")
    ax_crisis.axvline(0, color="#6b7280", lw=1.0)
    ax_crisis.set_yticks(y)
    ax_crisis.set_yticklabels(crisis_plot["crisis"])
    ax_crisis.invert_yaxis()
    ax_crisis.xaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_crisis.set_title("Crisis-Window Total Returns")
    ax_crisis.set_xlabel("Period return")
    ax_crisis.grid(True, axis="x", color="#e7ebef", lw=0.8)
    ax_crisis.legend(loc="lower right", frameon=False)

    ax_table.axis("off")
    table_df = crisis[
        ["crisis", "spy_return", "strategy_return", "daily_correlation", "beta_to_spy", "hedge_spread"]
    ].copy()
    table_df["spy_return"] = table_df["spy_return"].map(lambda x: f"{x:.1%}")
    table_df["strategy_return"] = table_df["strategy_return"].map(lambda x: f"{x:.1%}")
    table_df["daily_correlation"] = table_df["daily_correlation"].map(lambda x: f"{x:.2f}")
    table_df["beta_to_spy"] = table_df["beta_to_spy"].map(lambda x: f"{x:.2f}")
    table_df["hedge_spread"] = table_df["hedge_spread"].map(lambda x: f"{x:.1%}")
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=["Crisis", "SPY", "Strategy", "Corr", "Beta", "Spread"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.34, 0.12, 0.15, 0.1, 0.1, 0.13],
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
    ax_table.set_title("Crisis Correlation And Hedge Spread")

    annual_plot = annual.copy()
    x_annual = np.arange(len(annual_plot))
    annual_width = 0.38
    ax_annual.bar(
        x_annual - annual_width / 2,
        annual_plot["strategy_return"],
        width=annual_width,
        color=strategy_color,
        label="No-equity strategy",
    )
    ax_annual.bar(
        x_annual + annual_width / 2,
        annual_plot["spy_return"],
        width=annual_width,
        color=spy_color,
        label="SPY",
    )
    ax_annual.axhline(0, color="#6b7280", lw=1.0)
    ax_annual.set_xticks(x_annual)
    ax_annual.set_xticklabels(annual_plot["year"].astype(str), rotation=45, ha="right")
    ax_annual.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax_annual.set_title("Calendar-Year Returns: Strategy vs SPY")
    ax_annual.set_ylabel("Year return")
    ax_annual.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_annual.legend(loc="upper left", frameon=False, ncol=2)
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
        "No-Equity Futures Trend System vs SPY: Drawdowns, Correlation Regime, Crisis Hedge Effect, And Annual Returns",
        fontsize=17,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.02,
        0.965,
        "Daily comparison on SPY trading dates, 2000-01-19 to 2024-03-28. Crisis windows are shaded in the time-series panels.",
        fontsize=10,
        color=neutral,
        ha="left",
    )
    fig.subplots_adjust(top=0.935, left=0.13, right=0.98, bottom=0.07)
    fig.savefig(OUT / "strategy_vs_spy_crisis_big.png", dpi=180)
    fig.savefig(OUT / "strategy_vs_spy_crisis_big.pdf")

    tail_fig, tail_ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(tail))
    width = 0.36
    tail_ax.bar(x - width / 2, tail["spy_avg_daily_return"], width=width, color=spy_color, label="SPY avg day")
    tail_ax.bar(
        x + width / 2,
        tail["strategy_avg_daily_return"],
        width=width,
        color=strategy_color,
        label="Strategy avg day",
    )
    tail_ax.axhline(0, color="#6b7280", lw=1.0)
    tail_ax.set_xticks(x)
    tail_ax.set_xticklabels(tail["bucket"], rotation=0)
    tail_ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.2%}"))
    tail_ax.set_title("Average Strategy Return On SPY Down-Tail Days")
    tail_ax.set_ylabel("Average daily return")
    tail_ax.grid(True, axis="y", color="#e7ebef", lw=0.8)
    tail_ax.legend(frameon=False)
    tail_fig.tight_layout()
    tail_fig.savefig(OUT / "spy_tail_day_hedge_effect.png", dpi=180)
    plt.close(tail_fig)
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
    full_corr = daily[["strategy_return", "spy_return"]].corr().iloc[0, 1]
    full_beta = daily["strategy_return"].cov(daily["spy_return"]) / daily["spy_return"].var()
    negative_corr_days = (daily["rolling_corr_126d"] < 0).mean()

    lines = [
        "# No-Equity 40 Strategy vs SPY Crisis Comparison",
        "",
        "## Data",
        "",
        f"- Strategy: `{BACKTEST}`",
        f"- SPY source: Yahoo Finance adjusted close, `{YAHOO_HISTORY_PAGE}`",
        f"- SPY chart endpoint used: `{source_url}`",
        "- Daily comparisons use SPY trading dates. Strategy values are forward-filled onto those dates.",
        "- Correlation and beta use strategy NAV daily returns versus SPY daily returns.",
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
            "## Correlation And Hedge Read",
            "",
            f"- Full-sample daily correlation: {full_corr:.2f}",
            f"- Full-sample beta to SPY: {full_beta:.2f}",
            f"- Share of days where 126-day rolling correlation was negative: {negative_corr_days:.1%}",
            "- A lower or negative correlation during SPY stress windows is the key hedge signal.",
            "",
            "## Crisis Windows",
            "",
            "| Crisis | Start | End | SPY | Strategy | Hedge spread | Corr | Beta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in crisis.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["crisis"],
                    row["start"],
                    row["end"],
                    f"{row['spy_return']:.1%}",
                    f"{row['strategy_return']:.1%}",
                    f"{row['hedge_spread']:.1%}",
                    f"{row['daily_correlation']:.2f}",
                    f"{row['beta_to_spy']:.2f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## SPY Down-Tail Days",
            "",
            "| Bucket | Days | SPY avg day | Strategy avg day | Strategy positive hit rate | Corr |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tail.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["bucket"],
                    f"{row['days']:.0f}",
                    f"{row['spy_avg_daily_return']:.2%}",
                    f"{row['strategy_avg_daily_return']:.2%}",
                    f"{row['strategy_positive_hit_rate']:.1%}",
                    f"{row['daily_correlation']:.2f}",
                ]
            )
            + " |"
        )

    best_strategy_year = annual.loc[annual["strategy_return"].idxmax()]
    worst_strategy_year = annual.loc[annual["strategy_return"].idxmin()]
    strategy_won_years = (annual["strategy_return"] > annual["spy_return"]).sum()
    lines.extend(
        [
            "",
            "## Calendar-Year Return Comparison",
            "",
            f"- Strategy beat SPY in {strategy_won_years:.0f} / {len(annual):.0f} calendar-year rows.",
            f"- Best strategy year: {int(best_strategy_year['year'])}, {best_strategy_year['strategy_return']:.1%}.",
            f"- Worst strategy year: {int(worst_strategy_year['year'])}, {worst_strategy_year['strategy_return']:.1%}.",
            "- 2000 and 2024 are partial sample years.",
        ]
    )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- Big chart: `strategy_vs_spy_crisis_big.png`",
            "- Tail-day chart: `spy_tail_day_hedge_effect.png`",
            "- Daily data: `daily_comparison.csv`",
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

    print(f"Wrote crisis comparison to {OUT}")
    print(metrics.to_string(index=False))
    print()
    print(crisis.to_string(index=False))
    print()
    print(tail.to_string(index=False))
    print()
    print(annual.to_string(index=False))


if __name__ == "__main__":
    main()
