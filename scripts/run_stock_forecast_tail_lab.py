#!/usr/bin/env python3
"""Test top-decile, tail-only, and sector-neutral stock forecast portfolios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_benchmark_aware_stock_momentum as baw  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402
import run_stock_forecast_lab as lab  # noqa: E402


OUT = ROOT / "backtests" / "stock_forecast_tail_lab"
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-08-07"
DEFAULT_CAPITAL = 500_000.0
DEFAULT_VOL_TARGET = 0.25
DEFAULT_IDM = 2.75
DEFAULT_WEIGHT_MODE = "benchmark"
DEFAULT_IDM_METHOD = "fixed"

ROB_IDM_CORR_EW_LOOKBACK_WEEKS = 25
ROB_IDM_CORR_MIN_WEEKS = 20
ROB_IDM_EWMA_SPAN_DAYS = 125
ROB_IDM_MAX = 2.5

SELECTION_MODES = {
    "full": "all ranked active stocks",
    "top10_long": "top 10% only, long-only",
    "sector_top10_long": "top 10% within each sector, long-only",
    "tails5_each": "top 5% long and bottom 5% short, about 10% total names",
    "tails10_each": "top 10% long and bottom 10% short, about 20% total names",
}


def sector_frame_from_annual(annual: pd.DataFrame, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    snapshots = sorted(pd.to_datetime(annual["snapshot_date"].dropna().unique()))
    sectors = pd.DataFrame(pd.NA, index=index, columns=columns, dtype="object")
    if not snapshots or "sector" not in annual.columns:
        return sectors.fillna("Unknown")

    for i, snapshot_date in enumerate(snapshots):
        next_snapshot = snapshots[i + 1] if i + 1 < len(snapshots) else pd.Timestamp.max
        mask = (index >= snapshot_date) & (index < next_snapshot)
        if not mask.any():
            continue
        frame = annual[annual["snapshot_date"].eq(snapshot_date)].drop_duplicates("symbol").set_index("symbol")
        frame = frame[frame.index.isin(columns)]
        if frame.empty:
            continue
        values = frame["sector"].fillna("Unknown").astype(str)
        sectors.loc[mask, values.index] = np.repeat(
            values.to_numpy()[None, :],
            int(mask.sum()),
            axis=0,
        )
    return sectors.fillna("Unknown")


def apply_selection(
    forecast: pd.DataFrame,
    mode: str,
    *,
    sector_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if mode == "full":
        return forecast
    if mode == "sector_top10_long" and sector_frame is None:
        raise ValueError("sector_top10_long requires sector_frame")
    selected = pd.DataFrame(np.nan, index=forecast.index, columns=forecast.columns)
    for date, row in forecast.iterrows():
        scores = row.dropna()
        if scores.empty:
            continue
        if mode == "top10_long":
            top_n = max(1, int(np.ceil(len(scores) * 0.10)))
            names = scores.nlargest(top_n).index
        elif mode == "sector_top10_long":
            sector_row = sector_frame.loc[date, scores.index].fillna("Unknown").astype(str)
            chosen = []
            for _sector, sector_names in sector_row.groupby(sector_row).groups.items():
                sector_scores = scores.reindex(list(sector_names)).dropna()
                if sector_scores.empty:
                    continue
                top_n = max(1, int(np.ceil(len(sector_scores) * 0.10)))
                chosen.extend(sector_scores.nlargest(top_n).index.tolist())
            names = pd.Index(chosen)
        elif mode == "tails5_each":
            tail_n = max(1, int(np.ceil(len(scores) * 0.05)))
            names = scores.nlargest(tail_n).index.union(scores.nsmallest(tail_n).index)
        elif mode == "tails10_each":
            tail_n = max(1, int(np.ceil(len(scores) * 0.10)))
            names = scores.nlargest(tail_n).index.union(scores.nsmallest(tail_n).index)
        else:  # pragma: no cover
            raise ValueError(mode)
        selected.loc[date, names] = row.loc[names]
    return selected


def instrument_weights_for_forecast(
    price: pd.DataFrame,
    price_vol: pd.DataFrame,
    forecast: pd.DataFrame,
    annual: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    unit_daily_cash_vol = price_vol.abs().replace(0.0, np.nan)
    base = rob_stock.daily_base_weights(annual, price.columns, price.index, mode)
    valid = forecast.notna() & unit_daily_cash_vol.notna() & (unit_daily_cash_vol > 0.0) & price.notna()
    active_base = base.where(valid).fillna(0.0)
    active_sum = active_base.sum(axis=1).replace(0.0, np.nan)
    return active_base.div(active_sum, axis=0).fillna(0.0)


def cleaned_corr_matrix(history: pd.DataFrame, columns: pd.Index) -> pd.DataFrame | None:
    history = history.reindex(columns=columns).dropna(how="all")
    if len(history) < ROB_IDM_CORR_MIN_WEEKS:
        return None

    usable = history.columns[history.notna().sum() >= ROB_IDM_CORR_MIN_WEEKS]
    if len(usable) < 2:
        return None

    data = history[usable].tail(260).to_numpy(dtype=float)
    valid = np.isfinite(data)
    alpha = 2.0 / (ROB_IDM_CORR_EW_LOOKBACK_WEEKS + 1.0)
    row_weights = (1.0 - alpha) ** np.arange(len(data) - 1, -1, -1)
    row_weights = row_weights / row_weights.sum()

    weighted_valid = valid * row_weights[:, None]
    denom = weighted_valid.sum(axis=0)
    if np.any(denom <= 0.0):
        return None

    means = np.nansum(np.where(valid, data, 0.0) * row_weights[:, None], axis=0) / denom
    centred = np.where(valid, data - means, 0.0)
    covariance = (centred * row_weights[:, None]).T @ centred
    variance = np.diag(covariance)
    std = np.sqrt(np.maximum(variance, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_values = covariance / np.outer(std, std)

    corr_values = np.where(np.isfinite(corr_values), corr_values, np.nan)
    off_diag = corr_values.copy()
    np.fill_diagonal(off_diag, np.nan)
    fallback = np.nanmean(off_diag)
    if not np.isfinite(fallback):
        fallback = 0.99

    values = np.full((len(columns), len(columns)), fallback, dtype=float)
    loc = pd.Index(columns).get_indexer(usable)
    for i, target_i in enumerate(loc):
        if target_i < 0:
            continue
        for j, target_j in enumerate(loc):
            if target_j < 0:
                continue
            values[target_i, target_j] = corr_values[i, j]
    values = np.where(np.isfinite(values), values, fallback)
    values = np.maximum(values, 0.0)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=columns, columns=columns)


def rob_diversification_multiplier(weights: pd.Series, corr: pd.DataFrame | None) -> float:
    active = weights[weights > 0.0].dropna()
    if len(active) < 2 or corr is None:
        return 1.0
    corr = corr.reindex(index=active.index, columns=active.index)
    values = corr.to_numpy(dtype=float)
    weight_values = active.to_numpy(dtype=float)
    risk = float(np.sqrt(weight_values @ values @ weight_values.T))
    if not np.isfinite(risk) or risk < 0.0000001:
        return 1.0
    return float(min(1.0 / risk, ROB_IDM_MAX))


def estimate_rob_idm(
    price: pd.DataFrame,
    price_vol: pd.DataFrame,
    instrument_weights: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    normalised_returns = price.diff().div(price_vol.shift(1)).replace([np.inf, -np.inf], np.nan)
    weekly_returns = normalised_returns.cumsum().ffill().resample("W-FRI").last().diff()
    raw_weekly = pd.Series(dtype=float)
    weekly_index = weekly_returns.index
    weight_index = instrument_weights.index

    for weekly_date in weekly_index:
        weight_pos = weight_index.searchsorted(weekly_date, side="right") - 1
        if weight_pos < 0:
            continue
        date = weight_index[weight_pos]
        weights = instrument_weights.loc[date]
        active = weights[weights > 0.0].dropna()
        if len(active) < 2:
            continue
        history = weekly_returns.loc[weekly_returns.index < date, active.index].tail(260)
        corr = cleaned_corr_matrix(history, active.index)
        raw_weekly.loc[date] = rob_diversification_multiplier(active, corr)

    raw = raw_weekly.reindex(instrument_weights.index, method="ffill").fillna(1.0)
    smoothed = raw.ffill().fillna(1.0).ewm(span=ROB_IDM_EWMA_SPAN_DAYS).mean().clip(lower=1.0, upper=ROB_IDM_MAX)
    smoothed.name = "idm"
    raw.name = "raw_idm"
    return smoothed, raw


def selection_count_table(forecast: pd.DataFrame, selected: pd.DataFrame, label: str) -> dict[str, float | str]:
    active = forecast.notna().sum(axis=1)
    selected_count = selected.notna().sum(axis=1)
    selected_nonzero = selected.fillna(0.0).ne(0.0).sum(axis=1)
    valid = active[active > 0]
    return {
        "strategy": label,
        "avg_active_universe": float(valid.mean()),
        "avg_selected_names": float(selected_count.reindex(valid.index).mean()),
        "max_selected_names": int(selected_count.max()),
        "min_selected_names": int(selected_count[selected_count > 0].min()),
        "avg_nonzero_names": float(selected_nonzero.reindex(valid.index).mean()),
    }


def diagnostics_with_idm(label: str, daily: pd.DataFrame, capital: float) -> dict[str, float | str]:
    row = lab.diagnostics_row(label, daily, capital)
    if "idm" in daily.columns:
        active = rob_stock.trim_active_daily(daily)
        row["avg_idm"] = active["idm"].mean()
        row["min_idm"] = active["idm"].min()
        row["max_idm"] = active["idm"].max()
    return row


def run_universe(
    key: str,
    forecast_sets: list[str],
    selection_modes: list[str],
    start: str,
    end: str,
    *,
    weight_mode: str,
    capital: float,
    vol_target: float,
    idm: float,
    idm_method: str,
) -> dict[str, pd.DataFrame]:
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)

    annual = rob_stock.load_annual(key, start, end)
    price = rob_stock.load_price(key, annual, start, end)
    benchmark = baw.load_benchmark(key, start, end)
    library = lab.raw_signal_library(price, benchmark, annual)
    active = lab.active_mask(annual, price.columns, price.index)
    price_vol = rob_stock.rob.mixed_vol(price.diff())
    sectors = sector_frame_from_annual(annual, price.columns, price.index)

    stats_rows = []
    diag_rows = []
    count_rows = []
    daily_by_name: dict[str, pd.DataFrame] = {}

    for forecast_set in forecast_sets:
        forecast, _rule_table = lab.combine_forecast_set(library, lab.FORECAST_SETS[forecast_set], active)
        for selection_mode in selection_modes:
            selected_forecast = apply_selection(forecast, selection_mode, sector_frame=sectors)
            label = f"{forecast_set}__{selection_mode}"
            if idm_method == "rob_estimated":
                pre_idm_weights = instrument_weights_for_forecast(price, price_vol, selected_forecast, annual, weight_mode)
                idm_value, raw_idm = estimate_rob_idm(price, price_vol, pre_idm_weights)
                pd.concat([raw_idm, idm_value], axis=1).to_csv(out_dir / f"{label}_idm.csv")
            else:
                idm_value = idm
            positions, _target, _instrument_weights, risk = rob_stock.target_positions(
                price,
                price_vol,
                selected_forecast,
                annual,
                weight_mode,
                capital=capital,
                vol_target=vol_target,
                idm=idm_value,
            )
            daily, _by_instrument = rob_stock.pnl_from_stock_positions(
                positions,
                price,
                capital,
                rob_stock.DEFAULT_COST_PER_DOLLAR,
            )
            daily = daily.join(risk).loc[start:end]
            trimmed = rob_stock.trim_active_daily(daily)
            daily_by_name[label] = trimmed
            stats_rows.append({"strategy": label, **lab.performance_stats(trimmed, capital)})
            diag_rows.append(diagnostics_with_idm(label, trimmed, capital))
            count_rows.append(selection_count_table(forecast, selected_forecast, label))
            daily.to_csv(out_dir / f"{label}_daily.csv")

    benchmark_daily = rob_stock.benchmark_daily(key, start, end, capital)
    common_start = max(frame.index.min() for frame in daily_by_name.values())
    common_end = min(min(frame.index.max() for frame in daily_by_name.values()), benchmark_daily.index.max())
    daily_by_name = {name: frame.loc[common_start:common_end] for name, frame in daily_by_name.items()}
    benchmark_daily = rob_stock.rebase_daily(benchmark_daily.loc[common_start:common_end], capital, compound=True)
    daily_by_name[benchmark.name] = benchmark_daily
    stats_rows.append({"strategy": benchmark.name, **lab.performance_stats(benchmark_daily, capital)})

    stats = pd.DataFrame(stats_rows)
    diagnostics = pd.DataFrame(diag_rows)
    counts = pd.DataFrame(count_rows)
    annual_returns = rob_stock.yearly_returns_from_equity(
        {name: daily_by_name[name] for name in daily_by_name if name in stats["strategy"].values},
        capital,
    )

    stats.to_csv(out_dir / "stats.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    counts.to_csv(out_dir / "selection_counts.csv", index=False)
    annual_returns.to_csv(out_dir / "yearly_returns.csv")
    plot_best(key, daily_by_name, stats, out_dir, capital)
    write_universe_summary(key, stats, diagnostics, counts, out_dir, start, end, weight_mode, vol_target, idm, idm_method)
    return {
        "stats": stats.assign(universe=rob_stock.SUPPORTED_UNIVERSES[key]),
        "diagnostics": diagnostics.assign(universe=rob_stock.SUPPORTED_UNIVERSES[key]),
        "counts": counts.assign(universe=rob_stock.SUPPORTED_UNIVERSES[key]),
    }


def plot_best(
    key: str,
    daily_by_name: dict[str, pd.DataFrame],
    stats: pd.DataFrame,
    out_dir: Path,
    capital: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    benchmark = stats[stats["strategy"].str.startswith("Benchmark")]["strategy"].iloc[0]
    chosen = [benchmark]
    non_benchmark = stats[~stats["strategy"].eq(benchmark)].copy()
    non_benchmark["selection_mode"] = non_benchmark["strategy"].str.split("__").str[-1]
    for _mode, frame in non_benchmark.groupby("selection_mode"):
        chosen.append(frame.sort_values("sharpe", ascending=False)["strategy"].iloc[0])
    chosen = list(dict.fromkeys(chosen))

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.1], hspace=0.12)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    for name in chosen:
        daily = daily_by_name[name]
        equity = daily["equity"].dropna() / capital
        ax0.plot(equity.index, equity, label=name, linewidth=1.5)
        ax1.plot(equity.index, equity / equity.cummax() - 1.0, linewidth=1.0)
    ax0.set_title(f"{rob_stock.SUPPORTED_UNIVERSES[key]} Top-Decile Forecast Tests")
    ax0.set_yscale("log")
    ax0.set_ylabel("Growth of $1")
    ax1.set_ylabel("Drawdown")
    ax0.legend(loc="upper left", fontsize=8)
    ax1.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{key}_tail_lab_best.png", dpi=180)
    plt.close(fig)


def write_universe_summary(
    key: str,
    stats: pd.DataFrame,
    diagnostics: pd.DataFrame,
    counts: pd.DataFrame,
    out_dir: Path,
    start: str,
    end: str,
    weight_mode: str,
    vol_target: float,
    idm: float,
    idm_method: str,
) -> None:
    lines = [
        f"# {rob_stock.SUPPORTED_UNIVERSES[key]} Top-Decile Forecast Tests",
        "",
        f"- Sample requested: {start} to {end}.",
        f"- Portfolio engine: Rob-style stock sizing, `{weight_mode}` instrument weights, vol target {vol_target:.0%}, IDM method `{idm_method}`.",
        f"- Rob estimated IDM uses `1 / sqrt(W * Corr * W.T)`, max {ROB_IDM_MAX:.1f}, weekly EW correlation span {ROB_IDM_CORR_EW_LOOKBACK_WEEKS}, daily smoothing span {ROB_IDM_EWMA_SPAN_DAYS}.",
        "- Selection modes: full universe, top 10% long-only, sector-neutral top 10% long-only, top/bottom tails.",
        "",
        "## Performance",
        "",
        "| Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in stats.sort_values("sharpe", ascending=False).iterrows():
        lines.append(
            f"| {row['strategy']} | {row['start']} | {row['end']} | {lab.pct(row['cagr'])} | {lab.pct(row['ann_return'])} | {lab.pct(row['vol'])} | {lab.num(row['sharpe'])} | {lab.pct(row['mdd'])} | {lab.pct(row['total_return'])} |"
        )
    lines += [
        "",
        "## Name Counts",
        "",
        "| Strategy | Avg Active Universe | Avg Selected | Max Selected | Min Selected |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in counts.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['avg_active_universe']:.0f} | {row['avg_selected_names']:.0f} | {int(row['max_selected_names'])} | {int(row['min_selected_names'])} |"
        )
    lines += [
        "",
        "## Diagnostics",
        "",
        "| Strategy | Ann Turnover | Ann Cost | Avg Gross | Avg Net | Avg Long | Avg Short | Avg Names | Avg IDM | Min IDM | Max IDM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {row['strategy']} | {lab.pct(row['avg_turnover_annual'])} | {lab.pct(row['avg_cost_annual'])} | {lab.pct(row['avg_gross_exposure'])} | {lab.pct(row['avg_net_exposure'])} | {lab.pct(row['avg_long_exposure'])} | {lab.pct(row['avg_short_exposure'])} | {row['avg_active_names']:.0f} | {lab.num(row.get('avg_idm', np.nan))} | {lab.num(row.get('min_idm', np.nan))} | {lab.num(row.get('max_idm', np.nan))} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_summary(all_stats: pd.DataFrame, all_diag: pd.DataFrame, all_counts: pd.DataFrame) -> None:
    lines = [
        "# Stock Forecast Tail Lab",
        "",
        "Testing whether holding only the strongest ranked names, sector-neutral strongest names, or both tails improves the stock forecast system.",
        "",
        "## Performance",
        "",
        "| Universe | Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_stats.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {row['start']} | {row['end']} | {lab.pct(row['cagr'])} | {lab.pct(row['ann_return'])} | {lab.pct(row['vol'])} | {lab.num(row['sharpe'])} | {lab.pct(row['mdd'])} | {lab.pct(row['total_return'])} |"
        )
    lines += [
        "",
        "## Best By Universe And Mode",
        "",
        "| Universe | Selection Mode | Best Strategy | CAGR | Sharpe | MDD | Avg Selected |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    non_bench = all_stats[~all_stats["strategy"].str.startswith("Benchmark")].copy()
    non_bench["selection_mode"] = non_bench["strategy"].str.split("__").str[-1]
    for (universe, selection_mode), frame in non_bench.groupby(["universe", "selection_mode"]):
        best = frame.sort_values("sharpe", ascending=False).iloc[0]
        count_row = all_counts[(all_counts["universe"].eq(universe)) & (all_counts["strategy"].eq(best["strategy"]))].iloc[0]
        lines.append(
            f"| {universe} | {selection_mode} | {best['strategy']} | {lab.pct(best['cagr'])} | {lab.num(best['sharpe'])} | {lab.pct(best['mdd'])} | {count_row['avg_selected_names']:.0f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", nargs="+", default=["sp500", "eem", "efa"], choices=sorted(rob_stock.SUPPORTED_UNIVERSES))
    parser.add_argument("--forecast-sets", nargs="+", default=list(lab.FORECAST_SETS), choices=sorted(lab.FORECAST_SETS))
    parser.add_argument("--selection-modes", nargs="+", default=list(SELECTION_MODES), choices=sorted(SELECTION_MODES))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--weight-mode", default=DEFAULT_WEIGHT_MODE, choices=sorted(rob_stock.WEIGHT_MODES))
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--vol-target", type=float, default=DEFAULT_VOL_TARGET)
    parser.add_argument("--idm", type=float, default=DEFAULT_IDM)
    parser.add_argument("--idm-method", default=DEFAULT_IDM_METHOD, choices=["fixed", "rob_estimated"])
    parser.add_argument("--output-name", default="stock_forecast_tail_lab")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUT
    OUT = ROOT / "backtests" / args.output_name
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = []
    all_diag = []
    all_counts = []
    for key in args.universes:
        result = run_universe(
            key,
            args.forecast_sets,
            args.selection_modes,
            args.start,
            args.end,
            weight_mode=args.weight_mode,
            capital=args.capital,
            vol_target=args.vol_target,
            idm=args.idm,
            idm_method=args.idm_method,
        )
        all_stats.append(result["stats"])
        all_diag.append(result["diagnostics"])
        all_counts.append(result["counts"])
    stats = pd.concat(all_stats, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True)
    counts = pd.concat(all_counts, ignore_index=True)
    stats.to_csv(OUT / "all_stats.csv", index=False)
    diag.to_csv(OUT / "all_diagnostics.csv", index=False)
    counts.to_csv(OUT / "all_selection_counts.csv", index=False)
    write_combined_summary(stats, diag, counts)
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
