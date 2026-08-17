#!/usr/bin/env python3
"""Compare local CTA-style strategies with Federal Reserve QE/QT regimes."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import run_rob_style_backtest as bt


OUT = bt.ROOT / "backtests" / "cta_vs_fed_qe_qt"
FRED_WALCL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL"
RETURN_STREAMS = bt.ROOT / "backtests" / "kmlm_like_2000" / "return_streams.csv"

STRATEGIES = {
    "17 selected": "Self 17",
    "40 no-equity": "Self 40",
    "KMLM-like rolling 15V": "KMLM-like",
}

POLICY_PHASES = [
    ("Pre-QE / flat BS", "Other", "2002-12-31", "2008-08-31"),
    ("Crisis liquidity expansion", "Other", "2008-09-30", "2008-11-30"),
    ("QE1", "QE", "2008-12-31", "2010-03-31"),
    ("QE pause", "Other", "2010-04-30", "2010-10-31"),
    ("QE2", "QE", "2010-11-30", "2011-06-30"),
    ("Twist / reinvestment", "Other", "2011-07-31", "2012-08-31"),
    ("QE3 / taper", "QE", "2012-09-30", "2014-10-31"),
    ("Reinvestment / flat BS", "Other", "2014-11-30", "2017-09-30"),
    ("QT1", "QT", "2017-10-31", "2019-08-31"),
    ("Repo expansion", "Other", "2019-09-30", "2020-02-29"),
    ("Covid QE", "QE", "2020-03-31", "2022-03-31"),
    ("Transition to QT2", "Other", "2022-04-30", "2022-05-31"),
    ("QT2", "QT", "2022-06-30", "2024-03-31"),
]

PHASE_COLORS = {
    "QE": "#dbeafe",
    "QT": "#fee2e2",
    "Other": "#f3f4f6",
}

LINE_COLORS = {
    "Self 17": "#1f4e79",
    "Self 40": "#6f8f2f",
    "KMLM-like": "#8a3f73",
}


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def load_strategy_monthly_returns() -> pd.DataFrame:
    daily = pd.read_csv(RETURN_STREAMS, parse_dates=["date"]).set_index("date")
    daily = daily[list(STRATEGIES)].rename(columns=STRATEGIES)
    monthly = (1.0 + daily).resample("ME").prod() - 1.0
    return monthly


def load_walcl_monthly() -> pd.DataFrame:
    raw = pd.read_csv(FRED_WALCL_URL, parse_dates=["observation_date"])
    raw = raw.rename(columns={"observation_date": "date", "WALCL": "walcl_millions"})
    raw["walcl_millions"] = pd.to_numeric(raw["walcl_millions"], errors="coerce")
    weekly = raw.dropna().set_index("date").sort_index()
    monthly = weekly.resample("ME").last()
    monthly["walcl_trn"] = monthly["walcl_millions"] / 1_000_000.0
    monthly["walcl_1m_change_trn"] = monthly["walcl_trn"].diff()
    monthly["walcl_3m_change_trn"] = monthly["walcl_trn"].diff(3)
    monthly["walcl_3m_change_pct"] = monthly["walcl_trn"] / monthly["walcl_trn"].shift(3) - 1.0
    monthly["walcl_12m_change_trn"] = monthly["walcl_trn"].diff(12)
    monthly["walcl_12m_change_pct"] = monthly["walcl_trn"] / monthly["walcl_trn"].shift(12) - 1.0
    return monthly


def assign_policy_phase(index: pd.Index) -> pd.DataFrame:
    phase = pd.DataFrame(index=index, columns=["policy_phase", "policy_type"], dtype=object)
    phase[["policy_phase", "policy_type"]] = ["Unclassified", "Other"]
    for name, phase_type, start, end in POLICY_PHASES:
        mask = (phase.index >= pd.Timestamp(start)) & (phase.index <= pd.Timestamp(end))
        phase.loc[mask, "policy_phase"] = name
        phase.loc[mask, "policy_type"] = phase_type
    return phase


def assign_balance_sheet_regime(fed: pd.DataFrame) -> pd.Series:
    expansion = (fed["walcl_3m_change_trn"] > 0.075) | (fed["walcl_3m_change_pct"] > 0.015)
    contraction = (fed["walcl_3m_change_trn"] < -0.075) | (fed["walcl_3m_change_pct"] < -0.015)
    regime = pd.Series("Flat / noisy", index=fed.index, name="bs_regime")
    regime.loc[expansion] = "Balance sheet expanding"
    regime.loc[contraction] = "Balance sheet contracting"
    return regime


def strategy_stats(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "months": 0,
            "compound_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe_0rf": np.nan,
            "hit_rate": np.nan,
            "median_month": np.nan,
            "worst_month": np.nan,
            "best_month": np.nan,
        }
    compound = (1.0 + returns).prod() - 1.0
    months = len(returns)
    ann_return = (1.0 + compound) ** (12.0 / months) - 1.0 if compound > -1.0 else np.nan
    ann_vol = returns.std() * math.sqrt(12.0)
    return {
        "months": months,
        "compound_return": compound,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe_0rf": ann_return / ann_vol if ann_vol else np.nan,
        "hit_rate": (returns > 0.0).mean(),
        "median_month": returns.median(),
        "worst_month": returns.min(),
        "best_month": returns.max(),
    }


def build_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_returns = load_strategy_monthly_returns()
    fed = load_walcl_monthly()
    index = monthly_returns.index.intersection(fed.index)
    monthly = monthly_returns.reindex(index).join(fed.reindex(index))
    monthly = monthly.join(assign_policy_phase(monthly.index))
    monthly["bs_regime"] = assign_balance_sheet_regime(monthly)

    episode_rows = []
    for name, phase_type, start, end in POLICY_PHASES:
        frame = monthly.loc[start:end].dropna(subset=["walcl_trn"], how="all")
        frame = frame[[*STRATEGIES.values(), "walcl_trn"]].dropna(how="all")
        if frame.empty:
            continue
        row = {
            "policy_phase": name,
            "policy_type": phase_type,
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "months": len(frame),
            "fed_assets_start_trn": frame["walcl_trn"].dropna().iloc[0],
            "fed_assets_end_trn": frame["walcl_trn"].dropna().iloc[-1],
        }
        row["fed_assets_change_trn"] = row["fed_assets_end_trn"] - row["fed_assets_start_trn"]
        row["fed_assets_change_pct"] = row["fed_assets_end_trn"] / row["fed_assets_start_trn"] - 1.0
        for strategy in STRATEGIES.values():
            stats = strategy_stats(frame[strategy])
            for key, value in stats.items():
                row[f"{strategy}_{key}"] = value
        episode_rows.append(row)
    episode = pd.DataFrame(episode_rows)

    grouped_rows = []
    for group_col in ["policy_type", "bs_regime"]:
        for group, frame in monthly.groupby(group_col):
            row = {
                "grouping": group_col,
                "regime": group,
                "months": len(frame),
                "fed_assets_avg_3m_change_trn": frame["walcl_3m_change_trn"].mean(),
                "fed_assets_avg_3m_change_pct": frame["walcl_3m_change_pct"].mean(),
            }
            for strategy in STRATEGIES.values():
                stats = strategy_stats(frame[strategy])
                for key, value in stats.items():
                    row[f"{strategy}_{key}"] = value
            grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows)

    correlation_rows = []
    for strategy in STRATEGIES.values():
        frame = monthly[[strategy, "walcl_1m_change_trn", "walcl_3m_change_trn", "walcl_3m_change_pct"]].dropna()
        correlation_rows.append(
            {
                "strategy": strategy,
                "corr_return_vs_1m_fed_change": frame[strategy].corr(frame["walcl_1m_change_trn"]),
                "corr_return_vs_3m_fed_change": frame[strategy].corr(frame["walcl_3m_change_trn"]),
                "corr_return_vs_3m_fed_pct_change": frame[strategy].corr(frame["walcl_3m_change_pct"]),
            }
        )
    corr = pd.DataFrame(correlation_rows)
    return monthly, episode, grouped, corr, fed


def plot_outputs(monthly: pd.DataFrame, grouped: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nav = (1.0 + monthly[list(STRATEGIES.values())].fillna(0.0)).cumprod()

    fig = plt.figure(figsize=(18, 14), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.0, 0.95], hspace=0.28)
    ax_fed = fig.add_subplot(gs[0])
    ax_nav = fig.add_subplot(gs[1], sharex=ax_fed)
    ax_bar = fig.add_subplot(gs[2])

    for name, phase_type, start, end in POLICY_PHASES:
        start_ts = max(pd.Timestamp(start), monthly.index.min())
        end_ts = min(pd.Timestamp(end), monthly.index.max())
        if start_ts <= end_ts:
            ax_fed.axvspan(start_ts, end_ts, color=PHASE_COLORS[phase_type], alpha=0.65, lw=0)
            ax_nav.axvspan(start_ts, end_ts, color=PHASE_COLORS[phase_type], alpha=0.45, lw=0)

    ax_fed.plot(monthly.index, monthly["walcl_trn"], color="#111827", lw=2)
    ax_fed.set_title("Federal Reserve Balance Sheet and CTA-Style Strategy Returns")
    ax_fed.set_ylabel("Fed assets ($T)")
    ax_fed.grid(True, color="#e7ebef", lw=0.8)
    legend_patches = [
        Patch(facecolor=PHASE_COLORS["QE"], edgecolor="none", label="QE / expansion policy"),
        Patch(facecolor=PHASE_COLORS["QT"], edgecolor="none", label="QT"),
        Patch(facecolor=PHASE_COLORS["Other"], edgecolor="none", label="Other / flat"),
    ]
    ax_fed.legend(handles=legend_patches, frameon=False, loc="upper left", ncol=3)

    for strategy in STRATEGIES.values():
        ax_nav.plot(nav.index, nav[strategy], label=strategy, color=LINE_COLORS[strategy], lw=2)
    ax_nav.set_ylabel("Growth of $1")
    ax_nav.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f}x"))
    ax_nav.grid(True, color="#e7ebef", lw=0.8)
    ax_nav.legend(frameon=False, loc="upper left", ncol=3)

    policy = grouped[grouped["grouping"].eq("policy_type")].set_index("regime")
    regimes = [regime for regime in ["QE", "QT", "Other"] if regime in policy.index]
    x = np.arange(len(regimes))
    width = 0.22
    for offset, strategy in zip([-width, 0.0, width], STRATEGIES.values()):
        values = [policy.loc[regime, f"{strategy}_ann_return"] for regime in regimes]
        ax_bar.bar(x + offset, values, width=width, color=LINE_COLORS[strategy], label=strategy)
    ax_bar.axhline(0, color="#8c96a0", lw=0.8)
    ax_bar.set_title("Annualized Return by Manual Fed Policy Type")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(regimes)
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    ax_bar.grid(True, axis="y", color="#e7ebef", lw=0.8)
    ax_bar.legend(frameon=False, loc="upper left", ncol=3)

    ax_nav.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_nav.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(OUT / "cta_vs_fed_qe_qt.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(monthly: pd.DataFrame, episode: pd.DataFrame, grouped: pd.DataFrame, corr: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT / "monthly_strategy_fed_data.csv", index_label="date")
    episode.to_csv(OUT / "policy_episode_metrics.csv", index=False)
    grouped.to_csv(OUT / "regime_group_metrics.csv", index=False)
    corr.to_csv(OUT / "fed_balance_sheet_correlations.csv", index=False)

    lines = [
        "# CTA-Style Strategies vs Fed QE/QT",
        "",
        "Data and definitions:",
        "",
        f"- Fed balance sheet: FRED WALCL, source URL `{FRED_WALCL_URL}`.",
        "- Strategy returns: monthly compounding of daily return streams from the local backtests.",
        "- Manual policy type: QE, QT, or Other based on common Fed balance-sheet policy episodes.",
        "- Data-driven balance-sheet regime: expanding if 3-month WALCL change is above +$75bn or +1.5%; contracting if below -$75bn or -1.5%; otherwise flat/noisy.",
        "- KMLM-like is the local rolling-15V replication, not official KMLM ETF history.",
        "",
        "## Manual Policy-Type Summary",
        "",
        "| Type | Months | Fed 3m avg chg | Self 17 ann ret | Self 40 ann ret | KMLM-like ann ret |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    policy = grouped[grouped["grouping"].eq("policy_type")].set_index("regime")
    for regime in ["QE", "QT", "Other"]:
        if regime not in policy.index:
            continue
        row = policy.loc[regime]
        lines.append(
            "| "
            + " | ".join(
                [
                    regime,
                    f"{int(row['months'])}",
                    f"${row['fed_assets_avg_3m_change_trn']:.2f}T",
                    pct(row["Self 17_ann_return"]),
                    pct(row["Self 40_ann_return"]),
                    pct(row["KMLM-like_ann_return"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Data-Driven Balance-Sheet Regime Summary",
            "",
            "| Regime | Months | Fed 3m avg chg | Self 17 ann ret | Self 40 ann ret | KMLM-like ann ret |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    bs = grouped[grouped["grouping"].eq("bs_regime")].set_index("regime")
    for regime in ["Balance sheet expanding", "Balance sheet contracting", "Flat / noisy"]:
        if regime not in bs.index:
            continue
        row = bs.loc[regime]
        lines.append(
            "| "
            + " | ".join(
                [
                    regime,
                    f"{int(row['months'])}",
                    f"${row['fed_assets_avg_3m_change_trn']:.2f}T",
                    pct(row["Self 17_ann_return"]),
                    pct(row["Self 40_ann_return"]),
                    pct(row["KMLM-like_ann_return"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Policy Episode Detail",
            "",
            "| Episode | Type | Fed assets change | Self 17 | Self 40 | KMLM-like |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in episode.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["policy_phase"],
                    row["policy_type"],
                    f"${row['fed_assets_change_trn']:.2f}T / {pct(row['fed_assets_change_pct'])}",
                    pct(row["Self 17_compound_return"]),
                    pct(row["Self 40_compound_return"]),
                    pct(row["KMLM-like_compound_return"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Balance-Sheet Change Correlations",
            "",
            "| Strategy | Corr vs 1m WALCL chg | Corr vs 3m WALCL chg | Corr vs 3m WALCL % chg |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in corr.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["strategy"],
                    f"{row['corr_return_vs_1m_fed_change']:.2f}",
                    f"{row['corr_return_vs_3m_fed_change']:.2f}",
                    f"{row['corr_return_vs_3m_fed_pct_change']:.2f}",
                ]
            )
            + " |"
        )

    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(grouped: pd.DataFrame, episode: pd.DataFrame, corr: pd.DataFrame) -> None:
    print(f"Wrote QE/QT analysis to {OUT}")
    print("\nManual policy type summary")
    policy = grouped[grouped["grouping"].eq("policy_type")][
        [
            "regime",
            "months",
            "fed_assets_avg_3m_change_trn",
            "Self 17_ann_return",
            "Self 40_ann_return",
            "KMLM-like_ann_return",
            "Self 17_ann_vol",
            "Self 40_ann_vol",
            "KMLM-like_ann_vol",
        ]
    ]
    print(
        policy.to_string(
            index=False,
            formatters={
                "fed_assets_avg_3m_change_trn": lambda value: f"${value:.2f}T",
                "Self 17_ann_return": pct,
                "Self 40_ann_return": pct,
                "KMLM-like_ann_return": pct,
                "Self 17_ann_vol": pct,
                "Self 40_ann_vol": pct,
                "KMLM-like_ann_vol": pct,
            },
        )
    )
    print("\nEpisode returns")
    print(
        episode[
            [
                "policy_phase",
                "policy_type",
                "fed_assets_change_trn",
                "fed_assets_change_pct",
                "Self 17_compound_return",
                "Self 40_compound_return",
                "KMLM-like_compound_return",
            ]
        ].to_string(
            index=False,
            formatters={
                "fed_assets_change_trn": lambda value: f"${value:.2f}T",
                "fed_assets_change_pct": pct,
                "Self 17_compound_return": pct,
                "Self 40_compound_return": pct,
                "KMLM-like_compound_return": pct,
            },
        )
    )
    print("\nCorrelations")
    print(corr.to_string(index=False, formatters={
        "corr_return_vs_1m_fed_change": lambda value: f"{value:.2f}",
        "corr_return_vs_3m_fed_change": lambda value: f"{value:.2f}",
        "corr_return_vs_3m_fed_pct_change": lambda value: f"{value:.2f}",
    }))


def main() -> None:
    monthly, episode, grouped, corr, _fed = build_analysis()
    plot_outputs(monthly, grouped)
    write_report(monthly, episode, grouped, corr)
    print_summary(grouped, episode, corr)


if __name__ == "__main__":
    main()
