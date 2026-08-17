#!/usr/bin/env python3
"""Compare Rob original-style portfolio with current custom systems and KMLM-like."""

from __future__ import annotations

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

import run_rob_style_backtest as bt
import run_rob_style_no_equity_40_backtest as self40
import run_rob_style_us_rates_selected_no_vol_backtest as self17


OUT = bt.ROOT / "backtests" / "rob_original_vs_current_kmlm"
CAPITAL = 500_000.0

RETURN_SERIES = {
    "Rob original-style": bt.ROOT / "backtests" / "rob_style_multirule" / "portfolio_daily.csv",
    "Self 17": bt.ROOT / "backtests" / "rob_style_us_rates_selected_no_vol" / "portfolio_daily.csv",
    "Self 40": bt.ROOT / "backtests" / "rob_style_no_equity_40" / "portfolio_daily.csv",
}

CRISIS_WINDOWS = {
    "Dotcom total": ("2000-03-24", "2002-10-09"),
    "Dotcom 2000": ("2000-03-24", "2000-12-29"),
    "Dotcom 2001": ("2001-01-02", "2001-12-31"),
    "Dotcom 2002": ("2002-01-02", "2002-10-09"),
    "2008 calendar": ("2008-01-02", "2008-12-31"),
    "2008 H1": ("2008-01-02", "2008-06-30"),
    "2008 H2": ("2008-07-01", "2008-12-31"),
    "2022 calendar": ("2022-01-03", "2022-12-30"),
    "2022 H1": ("2022-01-03", "2022-06-30"),
    "2022 H2": ("2022-07-01", "2022-12-30"),
}

SECTOR_MAP = {
    "Ags": "Commodity",
    "OilGas": "Commodity",
    "Metals": "Commodity",
    "FX": "Currency",
    "Bond": "Fixed Income",
    "Equity": "Equity",
    "Sector": "Equity",
    "Housing": "Equity",
    "Vol": "Vol",
    "Other": "Other",
}

COLORS = {
    "Rob original-style": "#111827",
    "Self 17": "#1f4e79",
    "Self 40": "#6f8f2f",
    "KMLM-like": "#8a3f73",
}


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def load_buffered_returns(path) -> pd.Series:
    portfolio = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return portfolio[("buffered_integer", "daily_return")].dropna()


def load_return_streams() -> pd.DataFrame:
    returns = {name: load_buffered_returns(path) for name, path in RETURN_SERIES.items()}
    kmlm_streams = pd.read_csv(
        bt.ROOT / "backtests" / "kmlm_like_2000" / "return_streams.csv",
        parse_dates=["date"],
    ).set_index("date")
    returns["KMLM-like"] = kmlm_streams["KMLM-like rolling 15V"]

    start = max(series.index.min() for series in returns.values())
    end = min(series.index.max() for series in returns.values())
    index = pd.date_range(start, end, freq="B")
    return pd.DataFrame({name: series.reindex(index).fillna(0.0) for name, series in returns.items()})


def metrics_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    elapsed_years = (returns.index[-1] - returns.index[0]).days / 365.25
    for name in returns.columns:
        series = returns[name]
        nav = (1.0 + series).cumprod()
        compound_total = nav.iloc[-1] - 1.0
        ann_return = series.mean() * 252.0
        ann_vol = series.std() * math.sqrt(252.0)
        rows.append(
            {
                "series": name,
                "start": str(returns.index[0].date()),
                "end": str(returns.index[-1].date()),
                "years": elapsed_years,
                "pnl_pct_500k": series.sum(),
                "compound_total_return": compound_total,
                "compound_cagr": nav.iloc[-1] ** (1.0 / elapsed_years) - 1.0,
                "annual_return_arithmetic": ann_return,
                "annual_vol": ann_vol,
                "sharpe_0rf": ann_return / ann_vol if ann_vol else np.nan,
                "compound_max_drawdown": (nav / nav.cummax() - 1.0).min(),
            }
        )
    return pd.DataFrame(rows)


def annual_returns(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, frame in returns.groupby(returns.index.year):
        row = {"year": int(year), "start": str(frame.index.min().date()), "end": str(frame.index.max().date())}
        for name in returns.columns:
            row[f"{name}_pnl_pct_500k"] = frame[name].sum()
            row[f"{name}_compound_return"] = (1.0 + frame[name]).prod() - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def crisis_returns(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, (start, end) in CRISIS_WINDOWS.items():
        frame = returns.loc[start:end]
        for name in returns.columns:
            series = frame[name].dropna()
            if len(series) < 2:
                continue
            nav = (1.0 + series).cumprod()
            rows.append(
                {
                    "window": window,
                    "series": name,
                    "start": str(series.index[0].date()),
                    "end": str(series.index[-1].date()),
                    "pnl_pct_500k": series.sum(),
                    "compound_return": nav.iloc[-1] - 1.0,
                    "annual_vol": series.std() * math.sqrt(252.0),
                    "compound_max_drawdown": (nav / nav.cummax() - 1.0).min(),
                }
            )
    return pd.DataFrame(rows)


def group_sector(asset_class: str) -> str:
    return SECTOR_MAP.get(asset_class, asset_class)


def rob_original_weights() -> pd.DataFrame:
    universe = pd.read_csv(bt.ROOT / "backtests" / "rob_style_multirule" / "universe.csv")
    universe["sector"] = universe["asset_class"].map(group_sector)
    grouped = universe.groupby("sector")["base_weight"].sum().rename("weight").reset_index()
    grouped["strategy"] = "Rob original-style"
    return grouped


def custom_weights(name: str, module, config_fn) -> pd.DataFrame:
    cfg = config_fn(bt.load_rob_config())
    meta = bt.load_meta()
    rows = []
    for instrument in module.selected_instruments():
        asset_class = meta[instrument].asset_class
        rows.append(
            {
                "sector": group_sector(asset_class),
                "weight": float(cfg["instrument_weights"][instrument]),
                "strategy": name,
            }
        )
    return pd.DataFrame(rows).groupby(["strategy", "sector"], as_index=False)["weight"].sum()


def kmlm_average_weights() -> pd.DataFrame:
    weights = pd.read_csv(
        bt.ROOT / "backtests" / "kmlm_like_2000" / "sector_weights.csv",
        parse_dates=["date"],
    ).set_index("date")
    weights = weights.loc["2000-01-19":"2024-03-28"]
    out = weights.mean().rename("weight").reset_index().rename(columns={"index": "sector"})
    out["strategy"] = "KMLM-like"
    return out


def design_weights() -> pd.DataFrame:
    weights = pd.concat(
        [
            rob_original_weights(),
            custom_weights("Self 17", self17, self17.config_with_custom_weights),
            custom_weights("Self 40", self40, self40.config_with_no_equity_weights),
            kmlm_average_weights(),
        ],
        ignore_index=True,
    )
    return weights


def load_asset_daily(path: str, strategy: str) -> pd.DataFrame:
    data = pd.read_csv(bt.ROOT / path, parse_dates=["DATETIME"]).set_index("DATETIME")
    out = pd.DataFrame(index=data.index)
    for column in data.columns:
        sector = group_sector(column)
        out[sector] = out.get(sector, 0.0) + data[column]
    return out / CAPITAL


def sector_crisis_pnl() -> pd.DataFrame:
    paths = {
        "Rob original-style": "backtests/rob_style_multirule/asset_class_daily_pnl.csv",
        "Self 17": "backtests/rob_style_us_rates_selected_no_vol/asset_class_daily_pnl.csv",
        "Self 40": "backtests/rob_style_no_equity_40/asset_class_daily_pnl.csv",
    }
    sector_daily = {name: load_asset_daily(path, name) for name, path in paths.items()}
    kmlm_asset = pd.read_csv(bt.ROOT / "backtests" / "kmlm_like_2000" / "asset_breakdown.csv")
    kmlm_map = {
        "Dotcom total": "Dot-com bear",
        "2008 calendar": "2008 calendar",
        "2022 calendar": "2022",
    }
    rows = []
    for window, (start, end) in CRISIS_WINDOWS.items():
        for name, data in sector_daily.items():
            frame = data.loc[start:end]
            for sector in frame.columns:
                rows.append(
                    {
                        "window": window,
                        "series": name,
                        "sector": sector,
                        "pnl_pct_500k": frame[sector].sum(),
                    }
                )
        if window in kmlm_map:
            subset = kmlm_asset[
                kmlm_asset["strategy"].eq("KMLM-like rolling 15V")
                & kmlm_asset["window"].eq(kmlm_map[window])
            ]
            for row in subset.to_dict("records"):
                rows.append(
                    {
                        "window": window,
                        "series": "KMLM-like",
                        "sector": group_sector(row["asset_class"]),
                        "pnl_pct_500k": row["pnl_pct_500k"],
                    }
                )
    return pd.DataFrame(rows)


def plot_outputs(returns: pd.DataFrame, metrics: pd.DataFrame, annual: pd.DataFrame, crisis: pd.DataFrame, weights: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0

    fig = plt.figure(figsize=(18, 18), constrained_layout=False)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.2, 0.8, 1.05, 0.95], hspace=0.38, wspace=0.18)
    ax_nav = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_nav)
    ax_annual = fig.add_subplot(gs[2, :])
    ax_crisis = fig.add_subplot(gs[3, 0])
    ax_weights = fig.add_subplot(gs[3, 1])

    for name in returns.columns:
        ax_nav.plot(nav.index, nav[name], label=name, color=COLORS[name], lw=2.0)
        ax_dd.plot(drawdown.index, drawdown[name], label=name, color=COLORS[name], lw=1.5)

    ax_nav.set_title("Rob Original-Style vs Current Systems and KMLM-like: Compounded Return Stream")
    ax_nav.set_ylabel("Growth of $1")
    ax_nav.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f}x"))
    ax_nav.grid(True, color="#e7ebef", lw=0.8)
    ax_nav.legend(frameon=False, loc="upper left", ncol=4)

    ax_dd.axhline(0, color="#8c96a0", lw=0.8)
    ax_dd.set_title("Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax_dd.grid(True, color="#e7ebef", lw=0.8)

    years = annual["year"].to_numpy()
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(returns.columns))
    for offset, name in zip(offsets, returns.columns):
        ax_annual.bar(years + offset, annual[f"{name}_compound_return"], width=width, color=COLORS[name], label=name)
    ax_annual.axhline(0, color="#8c96a0", lw=0.8)
    ax_annual.set_title("Calendar-Year Compound Returns")
    ax_annual.set_ylabel("Return")
    ax_annual.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax_annual.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_annual.legend(frameon=False, loc="upper left", ncol=4)

    crisis_order = ["Dotcom total", "2008 calendar", "2022 calendar"]
    x = np.arange(len(crisis_order))
    for offset, name in zip(offsets, returns.columns):
        vals = [
            crisis[crisis["window"].eq(window) & crisis["series"].eq(name)]["pnl_pct_500k"].sum()
            for window in crisis_order
        ]
        ax_crisis.bar(x + offset, vals, width=width, color=COLORS[name], label=name)
    ax_crisis.axhline(0, color="#8c96a0", lw=0.8)
    ax_crisis.set_title("Crisis Window P&L / 500k")
    ax_crisis.set_xticks(x)
    ax_crisis.set_xticklabels(crisis_order, rotation=20, ha="right")
    ax_crisis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax_crisis.grid(True, axis="y", color="#e7ebef", lw=0.8)

    sectors = ["Fixed Income", "Currency", "Commodity", "Equity", "Vol", "Other"]
    wx = np.arange(len(sectors))
    w_width = 0.18
    for offset, name in zip(offsets, returns.columns):
        vals = [
            weights[weights["strategy"].eq(name) & weights["sector"].eq(sector)]["weight"].sum()
            for sector in sectors
        ]
        ax_weights.bar(wx + offset, vals, width=w_width, color=COLORS[name], label=name)
    ax_weights.set_title("Design / Average Sector Weights")
    ax_weights.set_xticks(wx)
    ax_weights.set_xticklabels(sectors, rotation=30, ha="right")
    ax_weights.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax_weights.grid(True, axis="y", color="#e7ebef", lw=0.8)

    ax_dd.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Rob original-style uses local usable instruments from Rob config; KMLM-like is local rolling-15V replication.",
        y=0.992,
        fontsize=10,
        color="#626b76",
    )
    fig.savefig(OUT / "rob_original_vs_current_kmlm.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, crisis: pd.DataFrame, weights: pd.DataFrame, sector_pnl: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    crisis.to_csv(OUT / "crisis_returns.csv", index=False)
    weights.to_csv(OUT / "sector_design_weights.csv", index=False)
    sector_pnl.to_csv(OUT / "sector_crisis_pnl.csv", index=False)

    lines = [
        "# Rob Original-Style vs Current Systems and KMLM-like",
        "",
        "Scope:",
        "",
        "- Rob original-style: local approximation using Rob `rob_system/config.yaml`, 165 usable local markets out of 170 configured, original instrument weights, forecast weights, FDM, IDM and risk overlay.",
        "- Self 17 / Self 40: the custom no-equity systems built in this workspace.",
        "- KMLM-like: local rolling-15V 22-market monthly 12-month moving-average replication, not official KMLM ETF history.",
        "- Return metrics use daily P&L divided by original USD 500k and compound that return stream.",
        "",
        "## Long-Run Metrics",
        "",
        "| Series | P&L / 500k | Compound CAGR | Ann ret | Vol | Sharpe | Compound MDD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["series"],
                    pct(row["pnl_pct_500k"]),
                    pct(row["compound_cagr"]),
                    pct(row["annual_return_arithmetic"]),
                    pct(row["annual_vol"]),
                    f"{row['sharpe_0rf']:.2f}",
                    pct(row["compound_max_drawdown"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Key Crisis Windows", ""])
    for window in ["Dotcom total", "2008 calendar", "2022 calendar"]:
        lines.extend(
            [
                f"### {window}",
                "",
                "| Series | P&L / 500k | Compound return | Vol | MDD |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        rows = crisis[crisis["window"].eq(window)].sort_values("pnl_pct_500k", ascending=False)
        for row in rows.to_dict("records"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["series"],
                        pct(row["pnl_pct_500k"]),
                        pct(row["compound_return"]),
                        pct(row["annual_vol"]),
                        pct(row["compound_max_drawdown"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Sector Design Weights",
            "",
            "| Strategy | Fixed Income | Currency | Commodity | Equity | Vol | Other |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in ["Rob original-style", "Self 17", "Self 40", "KMLM-like"]:
        row = weights[weights["strategy"].eq(strategy)].set_index("sector")["weight"]
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    pct(row.get("Fixed Income", 0.0)),
                    pct(row.get("Currency", 0.0)),
                    pct(row.get("Commodity", 0.0)),
                    pct(row.get("Equity", 0.0)),
                    pct(row.get("Vol", 0.0)),
                    pct(row.get("Other", 0.0)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Crisis Sector P&L / 500k", ""])
    for window in ["Dotcom total", "2008 calendar", "2022 calendar"]:
        lines.extend(
            [
                f"### {window}",
                "",
                "| Series | Fixed Income | Currency | Commodity | Equity | Vol | Other |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy in ["Rob original-style", "Self 17", "Self 40", "KMLM-like"]:
            row = sector_pnl[sector_pnl["window"].eq(window) & sector_pnl["series"].eq(strategy)].set_index("sector")[
                "pnl_pct_500k"
            ]
            lines.append(
                "| "
                + " | ".join(
                    [
                        strategy,
                        pct(row.get("Fixed Income", 0.0)),
                        pct(row.get("Currency", 0.0)),
                        pct(row.get("Commodity", 0.0)),
                        pct(row.get("Equity", 0.0)),
                        pct(row.get("Vol", 0.0)),
                        pct(row.get("Other", 0.0)),
                    ]
                )
                + " |"
            )
        lines.append("")

    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    returns = load_return_streams()
    metrics = metrics_from_returns(returns)
    annual = annual_returns(returns)
    crisis = crisis_returns(returns)
    weights = design_weights()
    sector_pnl = sector_crisis_pnl()

    OUT.mkdir(parents=True, exist_ok=True)
    returns.to_csv(OUT / "return_streams.csv", index_label="date")
    annual.to_csv(OUT / "annual_returns.csv", index=False)
    plot_outputs(returns, metrics, annual, crisis, weights)
    write_report(metrics, crisis, weights, sector_pnl)

    print(f"Wrote comparison to {OUT}")
    print("\nMetrics")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "years": lambda value: f"{value:.1f}",
                "pnl_pct_500k": pct,
                "compound_total_return": pct,
                "compound_cagr": pct,
                "annual_return_arithmetic": pct,
                "annual_vol": pct,
                "sharpe_0rf": lambda value: f"{value:.2f}",
                "compound_max_drawdown": pct,
            },
        )
    )
    print("\nCrisis P&L / 500k")
    view = crisis[crisis["window"].isin(["Dotcom total", "2008 calendar", "2022 calendar"])].pivot(
        index="window", columns="series", values="pnl_pct_500k"
    )
    print(view[["Rob original-style", "Self 17", "Self 40", "KMLM-like"]].to_string(formatters={col: pct for col in view.columns}))
    print("\nSector design weights")
    print(
        weights.pivot(index="strategy", columns="sector", values="weight")
        .fillna(0.0)
        .to_string(formatters={col: pct for col in weights["sector"].unique()})
    )


if __name__ == "__main__":
    main()
