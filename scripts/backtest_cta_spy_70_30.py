#!/usr/bin/env python3
"""Backtest 30% no-equity CTA + 70% SPY with annual rebalancing."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "cta_30_spy_70_annual_rebalance"
CTA_DAILY = ROOT / "backtests" / "rob_style_no_equity_40_long" / "portfolio_daily.csv"
SPY_DAILY = ROOT / "data" / "sp500_yfinance" / "spy_adj_close.csv"
BUSINESS_DAYS = 256.0

CRISIS_WINDOWS = {
    "Dot-com bear": ("2000-03-24", "2002-10-09"),
    "GFC": ("2007-10-09", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 inflation bear": ("2022-01-03", "2022-10-12"),
}


def read_cta_returns() -> pd.Series:
    frame = pd.read_csv(CTA_DAILY, header=[0, 1], index_col=0, parse_dates=True)
    returns = frame[("buffered_integer", "daily_return")].rename("CTA_no_equity_40")
    return returns.sort_index()


def read_spy_returns() -> pd.Series:
    spy = pd.read_csv(SPY_DAILY, parse_dates=["Date"]).set_index("Date")["SPY"].sort_index()
    return spy.pct_change().rename("SPY")


def annual_rebalance_portfolio(returns: pd.DataFrame, target_weights: dict[str, float]) -> pd.DataFrame:
    returns = returns.dropna().copy()
    nav = pd.Series(index=returns.index, dtype=float)
    sleeve_values = {asset: target_weights[asset] for asset in target_weights}
    previous_total = sum(sleeve_values.values())
    current_year = None
    records = []

    for date, row in returns.iterrows():
        total_before = sum(sleeve_values.values())
        if current_year is None or date.year != current_year:
            total_before = sum(sleeve_values.values())
            sleeve_values = {asset: total_before * weight for asset, weight in target_weights.items()}
            current_year = date.year
        for asset in sleeve_values:
            sleeve_values[asset] *= 1.0 + float(row[asset])
        total_after = sum(sleeve_values.values())
        nav.loc[date] = total_after
        day_return = total_after / previous_total - 1.0
        previous_total = total_after
        total_weighted = sum(sleeve_values.values())
        records.append(
            {
                "date": date,
                "portfolio_return": day_return,
                "portfolio_nav": total_after,
                "cta_weight_drift": sleeve_values["CTA_no_equity_40"] / total_weighted,
                "spy_weight_drift": sleeve_values["SPY"] / total_weighted,
                "rebalanced": date.year == current_year and date == returns.loc[str(date.year)].index.min(),
            }
        )

    result = pd.DataFrame(records).set_index("date")
    result["portfolio_drawdown"] = result["portfolio_nav"] / result["portfolio_nav"].cummax() - 1.0
    return result


def nav_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def metrics(name: str, returns: pd.Series, nav: pd.Series | None = None) -> dict[str, float | str]:
    returns = returns.dropna()
    if nav is None:
        nav = nav_from_returns(returns)
    nav = nav.loc[returns.index]
    days = (returns.index.max() - returns.index.min()).days
    years = days / 365.25
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0 else np.nan
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    drawdown = nav / nav.cummax() - 1.0
    downside = returns[returns < 0].std() * math.sqrt(BUSINESS_DAYS)
    return {
        "series": name,
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "years": years,
        "total_return": nav.iloc[-1] - 1.0,
        "cagr": cagr,
        "annual_return_arithmetic": ann_return,
        "annual_vol": ann_vol,
        "sharpe_0rf": ann_return / ann_vol if ann_vol else np.nan,
        "sortino_0rf": ann_return / downside if downside else np.nan,
        "max_drawdown": float(drawdown.min()),
        "calmar": cagr / abs(float(drawdown.min())) if drawdown.min() < 0 else np.nan,
    }


def crisis_metrics(returns: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = returns.copy()
    full["CTA30_SPY70_annual"] = portfolio["portfolio_return"]
    for name, (start, end) in CRISIS_WINDOWS.items():
        subset = full.loc[start:end].dropna()
        if subset.empty:
            continue
        for column in ["CTA_no_equity_40", "SPY", "CTA30_SPY70_annual"]:
            r = subset[column]
            nav = nav_from_returns(r)
            rows.append(
                {
                    "window": name,
                    "series": column,
                    "start": str(r.index.min().date()),
                    "end": str(r.index.max().date()),
                    "total_return": nav.iloc[-1] - 1.0,
                    "annual_vol": r.std() * math.sqrt(BUSINESS_DAYS),
                    "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
                    "corr_to_spy": np.nan if column == "SPY" else r.corr(subset["SPY"]),
                }
            )
    return pd.DataFrame(rows)


def monthly_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rolling_corr_63d": returns["CTA_no_equity_40"].rolling(63).corr(returns["SPY"]),
            "rolling_corr_252d": returns["CTA_no_equity_40"].rolling(252).corr(returns["SPY"]),
        }
    )


def plot_results(returns: pd.DataFrame, portfolio: pd.DataFrame, rolling_corr: pd.DataFrame) -> None:
    navs = pd.DataFrame(
        {
            "CTA no-equity 40": nav_from_returns(returns["CTA_no_equity_40"]),
            "SPY": nav_from_returns(returns["SPY"]),
            "30 CTA / 70 SPY annual": portfolio["portfolio_nav"],
        }
    )
    drawdowns = navs / navs.cummax() - 1.0
    annual = pd.DataFrame(
        {
            "CTA": returns["CTA_no_equity_40"].groupby(returns.index.year).apply(lambda r: nav_from_returns(r).iloc[-1] - 1.0),
            "SPY": returns["SPY"].groupby(returns.index.year).apply(lambda r: nav_from_returns(r).iloc[-1] - 1.0),
            "30/70": portfolio["portfolio_return"].groupby(portfolio.index.year).apply(lambda r: nav_from_returns(r).iloc[-1] - 1.0),
        }
    )

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [2.0, 1.2, 1.0, 1.6]})
    colors = {"CTA no-equity 40": "#4477aa", "SPY": "#222222", "30 CTA / 70 SPY annual": "#cc6677"}
    for column in navs:
        axes[0].plot(navs.index, navs[column], label=column, color=colors[column])
    axes[0].set_title("30% CTA / 70% SPY, Annual Rebalance")
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    for column in drawdowns:
        axes[1].plot(drawdowns.index, drawdowns[column], label=column, color=colors[column])
    axes[1].set_title("Drawdowns")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].grid(alpha=0.25)

    axes[2].plot(rolling_corr.index, rolling_corr["rolling_corr_252d"], label="252D CTA-SPY corr", color="#117733")
    axes[2].plot(rolling_corr.index, rolling_corr["rolling_corr_63d"], label="63D CTA-SPY corr", color="#88ccee", alpha=0.8)
    axes[2].axhline(0.0, color="#666666", linewidth=0.8)
    axes[2].set_title("CTA vs SPY Rolling Correlation")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.25)

    annual = annual.dropna(how="all")
    x = np.arange(len(annual.index))
    width = 0.25
    axes[3].bar(x - width, annual["CTA"] * 100, width=width, label="CTA", color="#4477aa")
    axes[3].bar(x, annual["SPY"] * 100, width=width, label="SPY", color="#222222")
    axes[3].bar(x + width, annual["30/70"] * 100, width=width, label="30/70", color="#cc6677")
    axes[3].axhline(0.0, color="#666666", linewidth=0.8)
    axes[3].set_title("Calendar-Year Returns")
    axes[3].set_ylabel("Return (%)")
    ticks = [i for i, year in enumerate(annual.index) if year % 2 == 0]
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels([str(annual.index[i]) for i in ticks], rotation=0)
    axes[3].legend(loc="upper left", ncols=3)
    axes[3].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "cta30_spy70_annual_rebalance.png", dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.1%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def markdown(metrics_df: pd.DataFrame, crisis_df: pd.DataFrame, corr_summary: pd.DataFrame) -> str:
    metrics_show = metrics_df.copy()
    for col in ["total_return", "cagr", "annual_return_arithmetic", "annual_vol", "max_drawdown"]:
        metrics_show[col] = metrics_show[col].map(pct)
    metrics_show["sharpe_0rf"] = metrics_show["sharpe_0rf"].map(num)
    metrics_show["sortino_0rf"] = metrics_show["sortino_0rf"].map(num)
    metrics_show["calmar"] = metrics_show["calmar"].map(num)
    metrics_show["years"] = metrics_show["years"].map(lambda value: f"{value:.1f}")

    crisis_show = crisis_df.copy()
    for col in ["total_return", "annual_vol", "max_drawdown"]:
        crisis_show[col] = crisis_show[col].map(pct)
    crisis_show["corr_to_spy"] = crisis_show["corr_to_spy"].map(num)

    corr_show = corr_summary.copy()
    for col in corr_show.columns:
        if col != "metric":
            corr_show[col] = corr_show[col].map(num)

    lines = [
        "# CTA 30 / SPY 70 Annual Rebalance Backtest",
        "",
        "## Method",
        "",
        "- CTA sleeve: best current `40 no-equity` Rob-style futures system, buffered integer return stream from the long-history run.",
        "- SPY sleeve: yfinance adjusted close return stream.",
        "- Common live comparison period: dates where both CTA and SPY have data.",
        "- Rebalance: first trading day of each calendar year, target weights 30% CTA and 70% SPY.",
        "- CTA returns are already after local transaction costs from the Rob-style futures backtest.",
        "",
        "## Metrics",
        "",
        metrics_show.to_markdown(index=False),
        "",
        "## Correlation Summary",
        "",
        corr_show.to_markdown(index=False),
        "",
        "## Crisis Windows",
        "",
        crisis_show.to_markdown(index=False),
        "",
        "## Behavior",
        "",
        "- The 30/70 portfolio has lower drawdown than SPY because CTA is weakly or negatively correlated in several equity stress windows.",
        "- Annual rebalancing lets weights drift intra-year, so the portfolio captures large CTA trend years without daily forcing exposure back to 30%.",
        "- The main tradeoff is that the CTA sleeve can dominate behavior during strong commodity/rate trend regimes, while SPY still drives the long-run equity beta.",
        "",
        "## Files",
        "",
        "- `metrics.csv`",
        "- `daily_returns_nav.csv`",
        "- `annual_returns.csv`",
        "- `crisis_metrics.csv`",
        "- `cta30_spy70_annual_rebalance.png`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    returns = pd.concat([read_cta_returns(), read_spy_returns()], axis=1).dropna()
    portfolio = annual_rebalance_portfolio(returns, {"CTA_no_equity_40": 0.30, "SPY": 0.70})
    aligned = returns.loc[portfolio.index].copy()
    aligned["CTA_nav"] = nav_from_returns(aligned["CTA_no_equity_40"])
    aligned["SPY_nav"] = nav_from_returns(aligned["SPY"])
    aligned = aligned.join(portfolio)
    rolling_corr = monthly_correlation(returns.loc[portfolio.index])

    metrics_df = pd.DataFrame(
        [
            metrics("CTA no-equity 40", aligned["CTA_no_equity_40"], aligned["CTA_nav"]),
            metrics("SPY", aligned["SPY"], aligned["SPY_nav"]),
            metrics("30 CTA / 70 SPY annual rebalance", aligned["portfolio_return"], aligned["portfolio_nav"]),
        ]
    )
    crisis_df = crisis_metrics(aligned[["CTA_no_equity_40", "SPY"]], aligned)
    corr_summary = pd.DataFrame(
        [
            {
                "metric": "Daily return correlation",
                "CTA_vs_SPY": aligned["CTA_no_equity_40"].corr(aligned["SPY"]),
                "Portfolio_vs_SPY": aligned["portfolio_return"].corr(aligned["SPY"]),
                "Portfolio_vs_CTA": aligned["portfolio_return"].corr(aligned["CTA_no_equity_40"]),
            },
            {
                "metric": "Median 252D CTA-SPY corr",
                "CTA_vs_SPY": rolling_corr["rolling_corr_252d"].median(),
                "Portfolio_vs_SPY": np.nan,
                "Portfolio_vs_CTA": np.nan,
            },
            {
                "metric": "Worst 252D CTA-SPY corr",
                "CTA_vs_SPY": rolling_corr["rolling_corr_252d"].min(),
                "Portfolio_vs_SPY": np.nan,
                "Portfolio_vs_CTA": np.nan,
            },
            {
                "metric": "Best 252D CTA-SPY corr",
                "CTA_vs_SPY": rolling_corr["rolling_corr_252d"].max(),
                "Portfolio_vs_SPY": np.nan,
                "Portfolio_vs_CTA": np.nan,
            },
        ]
    )
    annual = pd.DataFrame(
        {
            "CTA_no_equity_40": aligned["CTA_no_equity_40"].groupby(aligned.index.year).apply(lambda r: nav_from_returns(r).iloc[-1] - 1.0),
            "SPY": aligned["SPY"].groupby(aligned.index.year).apply(lambda r: nav_from_returns(r).iloc[-1] - 1.0),
            "CTA30_SPY70_annual": aligned["portfolio_return"].groupby(aligned.index.year).apply(lambda r: nav_from_returns(r).iloc[-1] - 1.0),
        }
    )

    aligned.to_csv(OUT / "daily_returns_nav.csv")
    metrics_df.to_csv(OUT / "metrics.csv", index=False)
    crisis_df.to_csv(OUT / "crisis_metrics.csv", index=False)
    corr_summary.to_csv(OUT / "correlation_summary.csv", index=False)
    rolling_corr.to_csv(OUT / "rolling_correlation.csv")
    annual.to_csv(OUT / "annual_returns.csv")
    plot_results(aligned[["CTA_no_equity_40", "SPY"]], aligned, rolling_corr)
    (OUT / "summary.md").write_text(markdown(metrics_df, crisis_df, corr_summary), encoding="utf-8")

    print(f"Wrote 30/70 CTA-SPY backtest to {OUT}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
