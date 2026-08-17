#!/usr/bin/env python3
"""Compare the IBKR-style USD 500k Rob 40 no-equity account with KMLM."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "ibkr_500k_rob40_vs_kmlm"
MARGIN_DIR = ROOT / "backtests" / "rob_style_no_equity_40_margin_constrained"
KMLM_DIR = ROOT / "backtests" / "kmlm_40_universe_comparison"
BUSINESS_DAYS = 256.0
SPY_START = "1993-01-29"
IBKR_START = "2000-01-03"


def nav_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def metrics_from_returns(name: str, returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    nav = nav_from_returns(returns)
    years = (returns.index.max() - returns.index.min()).days / 365.25
    ann_return = returns.mean() * BUSINESS_DAYS
    ann_vol = returns.std() * np.sqrt(BUSINESS_DAYS)
    drawdown = nav / nav.cummax() - 1.0
    downside = returns[returns < 0.0].std() * np.sqrt(BUSINESS_DAYS)
    cagr = nav.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and nav.iloc[-1] > 0.0 else np.nan
    return {
        "series": name,
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "years": years,
        "total_return": nav.iloc[-1] - 1.0,
        "cagr": cagr,
        "ann_return_arithmetic": ann_return,
        "ann_vol": ann_vol,
        "sharpe_0rf": ann_return / ann_vol if ann_vol > 0.0 else np.nan,
        "sortino_0rf": ann_return / downside if downside > 0.0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "calmar": cagr / abs(float(drawdown.min())) if drawdown.min() < 0.0 else np.nan,
    }


def align_returns(streams: dict[str, pd.Series], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    aligned = pd.concat(streams, axis=1, sort=True).sort_index()
    if start is not None:
        aligned = aligned.loc[start:]
    if end is not None:
        aligned = aligned.loc[:end]
    return aligned.dropna(how="any")


def load_ibkr_rob40() -> pd.Series:
    path = MARGIN_DIR / "daily_2000plus_current_static_margin_cap_100.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame["equity"].pct_change().fillna(0.0).rename("Rob 40 no-equity IBKR 500k")


def load_margin_diagnostics() -> pd.DataFrame:
    path = MARGIN_DIR / "daily_2000plus_current_static_margin_cap_100.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_unconstrained() -> pd.Series:
    path = MARGIN_DIR / "daily_2000_unconstrained.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame["equity"].pct_change().fillna(0.0).rename("Rob 40 no-equity unconstrained")


def load_kmlm_streams() -> dict[str, pd.Series]:
    path = KMLM_DIR / "simulated_return_streams.csv"
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return {
        "KMLM public-22 simulated": frame["KMLM public-22 rule"].rename("KMLM public-22 simulated"),
        "KMLM rule on 40 no-equity": frame["KMLM rule on 40 no-equity"].rename("KMLM rule on 40 no-equity"),
        "KMLM actual ETF": frame["KMLM actual"].rename("KMLM actual ETF"),
        "SPY": frame["SPY"].rename("SPY"),
    }


def add_spy_corr(metrics: pd.DataFrame, streams: dict[str, pd.Series], spy: pd.Series, start: str, end: str) -> pd.DataFrame:
    metrics = metrics.copy()
    for row in metrics.index:
        name = str(metrics.at[row, "series"])
        if name == "SPY":
            metrics.at[row, "corr_to_spy"] = 1.0
            continue
        aligned = align_returns({name: streams[name], "SPY": spy}, start=start, end=end)
        metrics.at[row, "corr_to_spy"] = aligned[name].corr(aligned["SPY"]) if len(aligned) > 2 else np.nan
    return metrics


def annual_returns(streams: dict[str, pd.Series], start: str, end: str) -> pd.DataFrame:
    aligned = pd.concat(streams, axis=1, sort=True).sort_index().loc[start:end]
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
    for col in ["total_return", "cagr", "ann_return_arithmetic", "ann_vol", "max_drawdown", "corr_to_spy"]:
        if col in show:
            show[col] = show[col].map(pct)
    for col in ["sharpe_0rf", "sortino_0rf", "calmar"]:
        if col in show:
            show[col] = show[col].map(num)
    if "years" in show:
        show["years"] = show["years"].map(lambda value: f"{value:.1f}")
    return show.to_markdown(index=False)


def plot_comparison(
    long_frame: pd.DataFrame,
    actual_frame: pd.DataFrame,
    annual: pd.DataFrame,
    margin_daily: pd.DataFrame,
) -> None:
    colors = {
        "Rob 40 no-equity IBKR 500k": "#0F766E",
        "Rob 40 no-equity unconstrained": "#64748B",
        "KMLM public-22 simulated": "#7C3AED",
        "KMLM rule on 40 no-equity": "#B45309",
        "KMLM actual ETF": "#111827",
        "SPY": "#2563EB",
    }
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 18),
        sharex=False,
        gridspec_kw={"height_ratios": [1.4, 0.9, 1.1, 1.0, 1.1]},
    )

    long_nav = long_frame.apply(nav_from_returns)
    for name in long_nav.columns:
        axes[0].plot(long_nav.index, long_nav[name], label=name, color=colors.get(name))
    axes[0].set_yscale("log")
    axes[0].set_title("Long-Term Comparison From 2000, Local Futures Endpoint")
    axes[0].set_ylabel("Growth of $1, log")
    axes[0].legend(loc="upper left", ncols=2)
    axes[0].grid(alpha=0.25)

    long_dd = long_nav / long_nav.cummax() - 1.0
    for name in long_dd.columns:
        axes[1].plot(long_dd.index, long_dd[name], label=name, color=colors.get(name))
    axes[1].set_title("Long-Term Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    axes[1].grid(alpha=0.25)

    actual_nav = actual_frame.apply(nav_from_returns)
    for name in actual_nav.columns:
        axes[2].plot(actual_nav.index, actual_nav[name], label=name, color=colors.get(name))
    axes[2].set_title("Actual KMLM ETF Overlap")
    axes[2].set_ylabel("Growth of $1")
    axes[2].legend(loc="upper left", ncols=2)
    axes[2].grid(alpha=0.25)

    x = np.arange(len(annual))
    columns = ["Rob 40 no-equity IBKR 500k", "KMLM public-22 simulated", "KMLM actual ETF", "SPY"]
    columns = [column for column in columns if column in annual.columns]
    width = min(0.75 / len(columns), 0.18)
    offsets = np.linspace(-width * (len(columns) - 1) / 2, width * (len(columns) - 1) / 2, len(columns))
    for offset, name in zip(offsets, columns):
        axes[3].bar(x + offset, annual[name] * 100.0, width=width, label=name, color=colors.get(name))
    axes[3].axhline(0.0, color="#666666", linewidth=0.8)
    axes[3].set_title("Calendar-Year Returns In Actual KMLM Overlap")
    axes[3].set_ylabel("Return (%)")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(annual["year"].astype(int).astype(str), rotation=0)
    axes[3].legend(loc="upper left", ncols=2)
    axes[3].grid(axis="y", alpha=0.25)

    margin_daily["margin_to_equity"].rolling(20).mean().plot(ax=axes[4], label="used margin / equity", color="#0F766E")
    margin_daily["desired_margin_to_equity"].rolling(20).mean().plot(
        ax=axes[4], label="desired margin / equity", color="#DC2626", alpha=0.75
    )
    axes[4].axhline(1.0, color="#111827", linewidth=0.8, linestyle="--")
    axes[4].set_title("IBKR 500k Account Margin Pressure")
    axes[4].set_ylabel("20D avg margin / equity")
    axes[4].yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "ibkr_500k_rob40_vs_kmlm.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ibkr = load_ibkr_rob40()
    unconstrained = load_unconstrained()
    margin_daily = load_margin_diagnostics()
    kmlm = load_kmlm_streams()

    futures_end = min(ibkr.index.max(), kmlm["KMLM public-22 simulated"].dropna().index.max())
    long_streams = {
        "Rob 40 no-equity IBKR 500k": ibkr.loc[:futures_end],
        "Rob 40 no-equity unconstrained": unconstrained.loc[:futures_end],
        "KMLM public-22 simulated": kmlm["KMLM public-22 simulated"].loc[:futures_end],
        "KMLM rule on 40 no-equity": kmlm["KMLM rule on 40 no-equity"].loc[:futures_end],
        "SPY": kmlm["SPY"].loc[:futures_end],
    }
    long_frame = align_returns(long_streams, start=IBKR_START, end=str(futures_end.date()))
    long_metrics = pd.DataFrame([metrics_from_returns(name, long_frame[name]) for name in long_frame.columns])
    long_metrics = add_spy_corr(long_metrics, {name: long_frame[name] for name in long_frame.columns}, long_frame["SPY"], IBKR_START, str(futures_end.date()))

    actual_start = max(kmlm["KMLM actual ETF"].dropna().index.min(), ibkr.index.min())
    actual_end = min(kmlm["KMLM actual ETF"].dropna().index.max(), ibkr.index.max())
    actual_streams = {
        "Rob 40 no-equity IBKR 500k": ibkr,
        "Rob 40 no-equity unconstrained": unconstrained,
        "KMLM public-22 simulated": kmlm["KMLM public-22 simulated"],
        "KMLM rule on 40 no-equity": kmlm["KMLM rule on 40 no-equity"],
        "KMLM actual ETF": kmlm["KMLM actual ETF"],
        "SPY": kmlm["SPY"],
    }
    actual_frame = align_returns(actual_streams, start=str(actual_start.date()), end=str(actual_end.date()))
    actual_metrics = pd.DataFrame([metrics_from_returns(name, actual_frame[name]) for name in actual_frame.columns])
    actual_metrics = add_spy_corr(
        actual_metrics,
        {name: actual_frame[name] for name in actual_frame.columns},
        actual_frame["SPY"],
        str(actual_frame.index.min().date()),
        str(actual_frame.index.max().date()),
    )

    annual = annual_returns(actual_streams, str(actual_frame.index.min().date()), str(actual_frame.index.max().date()))
    plot_comparison(long_frame, actual_frame, annual, margin_daily.loc[IBKR_START:futures_end])

    combined = pd.concat(
        {
            "long_common": long_frame,
            "actual_kmlm_overlap": actual_frame,
        },
        axis=1,
        sort=True,
    )
    combined.to_csv(OUT / "return_streams.csv", index_label="date")
    long_corr = long_frame.corr()
    actual_corr = actual_frame.corr()
    long_corr.to_csv(OUT / "long_term_correlation_matrix.csv")
    actual_corr.to_csv(OUT / "actual_kmlm_overlap_correlation_matrix.csv")
    long_metrics.to_csv(OUT / "long_term_metrics.csv", index=False)
    actual_metrics.to_csv(OUT / "actual_kmlm_overlap_metrics.csv", index=False)
    annual.to_csv(OUT / "actual_kmlm_overlap_annual_returns.csv", index=False)

    margin_summary = pd.DataFrame(
        [
            {
                "start": str(margin_daily.index.min().date()),
                "end": str(margin_daily.index.max().date()),
                "median_margin_to_equity": margin_daily["margin_to_equity"].median(),
                "p95_margin_to_equity": margin_daily["margin_to_equity"].quantile(0.95),
                "max_margin_to_equity": margin_daily["margin_to_equity"].max(),
                "pct_days_scaled": (margin_daily["margin_scale"] < 0.999).mean(),
                "max_desired_margin_to_equity": margin_daily["desired_margin_to_equity"].max(),
            }
        ]
    )
    margin_summary.to_csv(OUT / "ibkr_margin_summary.csv", index=False)

    lines = [
        "# IBKR 500k Rob 40 No-Equity vs KMLM",
        "",
        "## Definition",
        "",
        "- `Rob 40 no-equity IBKR 500k` uses the existing Rob 40 no-equity buffered integer futures positions.",
        "- Account starts with USD 500,000 on 2000-01-03.",
        "- Futures positions are capped by IBKR current overnight initial margin, translated to USD. If desired margin exceeds account equity, integer contracts are scaled down.",
        "- Performance uses account-equity percentage returns, not `net_pnl / initial_capital` compounded as if it were an ETF return.",
        "- This treats futures margin as collateral usage, not as financing on full futures notional.",
        "",
        "## Long-Term Common Window",
        "",
        markdown_metrics(long_metrics),
        "",
        "## Actual KMLM ETF Overlap",
        "",
        markdown_metrics(actual_metrics),
        "",
        "## Correlation Snapshot",
        "",
        "Long-term common window:",
        "",
        long_corr.to_markdown(),
        "",
        "Actual KMLM ETF overlap:",
        "",
        actual_corr.to_markdown(),
        "",
        "## IBKR Margin Usage",
        "",
        margin_summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- The IBKR 500k constraint reduces the Rob 40 sleeve versus unconstrained, mainly in early years when account equity is still small.",
        "- Even after margin caps, Rob 40 remains much more aggressive than KMLM: higher return, higher volatility, and less ETF-like construction.",
        "- Actual KMLM remains the cleaner benchmark for an investable CTA ETF, while the simulated KMLM public-22 rule is the better long-history proxy.",
        "- Because true historical SPAN margins and IBKR historical interest schedules are not in the local data, this is a current-margin account-capacity simulation rather than a perfect historical broker statement.",
        "",
        "## Files",
        "",
        "- `ibkr_500k_rob40_vs_kmlm.png`",
        "- `long_term_metrics.csv`",
        "- `actual_kmlm_overlap_metrics.csv`",
        "- `actual_kmlm_overlap_annual_returns.csv`",
        "- `ibkr_margin_summary.csv`",
        "- `return_streams.csv`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT}")
    print(long_metrics.to_string(index=False))
    print(actual_metrics.to_string(index=False))
    print(margin_summary.to_string(index=False))


if __name__ == "__main__":
    main()
