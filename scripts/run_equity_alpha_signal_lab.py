#!/usr/bin/env python3
"""Stage 02-04 close-only cross-sectional equity alpha signal lab."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402
import run_rob_style_backtest as rob  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402


OUT = ROOT / "research" / "equity_alpha" / "stage_02_04_signal_lab"
START = "2016-01-01"
DEV_END = "2023-12-31"
UNIVERSES = ("sp500", "eem", "efa")
HORIZONS = (5, 10, 20, 60, 120)
BUSINESS_DAYS = 252.0
COST_PER_DOLLAR = rob_stock.DEFAULT_COST_PER_DOLLAR
MIN_CROSS_SECTION = 50
EXECUTION_LAG_DAYS = 1
warnings.filterwarnings(
    "ignore",
    message="An input array is constant; the correlation coefficient is not defined.",
)


@dataclass(frozen=True)
class UniverseInputs:
    key: str
    label: str
    annual: pd.DataFrame
    price: pd.DataFrame
    active: pd.DataFrame
    sector: pd.DataFrame
    benchmark_price: pd.Series


def annualized_horizon_return(value: float, horizon: int) -> float:
    return float(value * BUSINESS_DAYS / horizon) if pd.notna(value) else math.nan


def fmt_pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def fmt_num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.3f}"


def load_benchmark_price(key: str, index: pd.DatetimeIndex) -> pd.Series:
    spec = next(spec for spec in pit.UNIVERSES if spec.key == key)
    path = pit.DATA_ROOT / key / "benchmark_adj_close.csv"
    price = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()[spec.benchmark_ticker]
    return price.reindex(index).ffill()


def sector_frame_from_annual(annual: pd.DataFrame, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    sectors = pd.DataFrame("Unknown", index=index, columns=columns, dtype="object")
    snapshots = sorted(pd.to_datetime(annual["snapshot_date"].dropna().unique()))
    for i, snapshot_date in enumerate(snapshots):
        next_snapshot = snapshots[i + 1] if i + 1 < len(snapshots) else pd.Timestamp.max
        mask = (index >= snapshot_date) & (index < next_snapshot)
        if not mask.any():
            continue
        frame = annual[annual["snapshot_date"].eq(snapshot_date)].drop_duplicates("symbol").set_index("symbol")
        frame = frame[frame.index.isin(columns)]
        if frame.empty:
            continue
        values = frame.get("sector", pd.Series("Unknown", index=frame.index)).fillna("Unknown").astype(str)
        sectors.loc[mask, values.index] = values
    return sectors


def load_universe(key: str) -> UniverseInputs:
    annual = rob_stock.load_annual(key, START, DEV_END)
    price = rob_stock.load_price(key, annual, START, DEV_END)
    price = price.loc[:DEV_END]
    active = rob_stock.daily_base_weights(annual, price.columns, price.index, "equal") > 0.0
    sector = sector_frame_from_annual(annual, price.columns, price.index)
    benchmark_price = load_benchmark_price(key, price.index)
    return UniverseInputs(
        key=key,
        label=rob_stock.SUPPORTED_UNIVERSES[key],
        annual=annual,
        price=price,
        active=active,
        sector=sector,
        benchmark_price=benchmark_price,
    )


def log_return(price: pd.DataFrame | pd.Series, horizon: int) -> pd.DataFrame | pd.Series:
    return np.log(price / price.shift(horizon))


def forward_excess_return(data: UniverseInputs, horizon: int) -> pd.DataFrame:
    entry = EXECUTION_LAG_DAYS
    exit_lag = EXECUTION_LAG_DAYS + horizon
    stock_forward = np.log(data.price.shift(-exit_lag) / data.price.shift(-entry))
    benchmark_forward = np.log(data.benchmark_price.shift(-exit_lag) / data.benchmark_price.shift(-entry))
    target = stock_forward.sub(benchmark_forward, axis=0)
    return target.where(data.active)


def trailing_vol(price: pd.DataFrame, window: int) -> pd.DataFrame:
    min_periods = min(window, max(5, window // 2))
    return np.log(price).diff().rolling(window, min_periods=min_periods).std() * math.sqrt(BUSINESS_DAYS)


def trailing_signal_return(price: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return np.log(price / price.shift(horizon))


def skip_recent_return(price: pd.DataFrame, lookback: int, skip: int = 21) -> pd.DataFrame:
    return np.log(price.shift(skip) / price.shift(lookback))


def breakout(price: pd.DataFrame, window: int) -> pd.DataFrame:
    high = price.shift(1).rolling(window, min_periods=max(20, window // 2)).max()
    low = price.shift(1).rolling(window, min_periods=max(20, window // 2)).min()
    return (2.0 * (price - low) / (high - low).replace(0.0, np.nan) - 1.0).clip(-3.0, 3.0)


def sector_group_mean(frame: pd.DataFrame, active: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    change = sector.ne(sector.shift()).any(axis=1)
    starts = list(sector.index[change])
    if not starts:
        return out
    starts.append(pd.Timestamp.max)
    for start, end in zip(starts[:-1], starts[1:]):
        period_index = frame.index[(frame.index >= start) & (frame.index < end)]
        if len(period_index) == 0:
            continue
        sector_row = sector.loc[start].fillna("Unknown").astype(str)
        sector_values = sector_row.to_numpy()
        for sector_name in pd.unique(sector_values):
            members = sector_row.index[sector_values == sector_name]
            if len(members) == 0:
                continue
            values = frame.loc[period_index, members].where(active.loc[period_index, members])
            group_mean = values.mean(axis=1)
            out.loc[period_index, members] = np.repeat(group_mean.to_numpy()[:, None], len(members), axis=1)
    return out


def rolling_residual_returns(data: UniverseInputs) -> pd.DataFrame:
    returns = data.price.pct_change(fill_method=None).where(data.active)
    market = data.benchmark_price.pct_change(fill_method=None).reindex(returns.index).fillna(0.0)
    sector_returns = sector_group_mean(returns, data.active, data.sector).fillna(0.0)
    returns = returns.mask(returns.abs() > rob_stock.MAX_ABS_DAILY_RETURN)

    window = 252
    min_periods = 126
    mean_y = returns.rolling(window, min_periods=min_periods).mean()
    mean_x1 = market.rolling(window, min_periods=min_periods).mean()
    mean_x2 = sector_returns.rolling(window, min_periods=min_periods).mean()

    cov11 = market.rolling(window, min_periods=min_periods).var()
    cov22 = sector_returns.rolling(window, min_periods=min_periods).var()
    cov12 = sector_returns.mul(market, axis=0).rolling(window, min_periods=min_periods).mean().sub(
        mean_x2.mul(mean_x1, axis=0),
        axis=0,
    )
    cov_y1 = returns.mul(market, axis=0).rolling(window, min_periods=min_periods).mean().sub(
        mean_y.mul(mean_x1, axis=0),
        axis=0,
    )
    cov_y2 = (returns * sector_returns).rolling(window, min_periods=min_periods).mean() - mean_y * mean_x2

    det = cov22.mul(cov11, axis=0) - cov12.pow(2)
    beta_market = (cov_y1 * cov22 - cov_y2 * cov12).div(det.replace(0.0, np.nan))
    beta_sector = (cov_y2.mul(cov11, axis=0) - cov_y1 * cov12).div(det.replace(0.0, np.nan))
    intercept = mean_y - beta_market.mul(mean_x1, axis=0) - beta_sector * mean_x2

    fitted = intercept.shift(1) + beta_market.shift(1).mul(market, axis=0) + beta_sector.shift(1) * sector_returns
    return (returns - fitted).where(data.active)


def build_signal_library(data: UniverseInputs) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    price = data.price
    active = data.active
    sigma_60 = trailing_vol(price, 60)
    signals: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for horizon in (20, 60, 120, 250):
        raw = trailing_signal_return(price, horizon)
        signals[f"mom_raw_{horizon}"] = raw.where(active)
        denom = sigma_60 * math.sqrt(horizon / BUSINESS_DAYS)
        signals[f"mom_voladj_{horizon}"] = (raw / denom.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)
        rows.append({"signal": f"mom_raw_{horizon}", "family": "momentum", "formula": f"log(P_t/P_t-{horizon})", "uses": "adjusted close", "lookahead_risk": "trailing close only"})
        rows.append({"signal": f"mom_voladj_{horizon}", "family": "momentum", "formula": f"log(P_t/P_t-{horizon}) / trailing sigma_60", "uses": "adjusted close", "lookahead_risk": "trailing close only"})

    for name, lookback in (("6_1", 126), ("12_1", 252)):
        raw = skip_recent_return(price, lookback, 21)
        signals[f"mom_raw_{name}"] = raw.where(active)
        denom = sigma_60 * math.sqrt(max(lookback - 21, 1) / BUSINESS_DAYS)
        signals[f"mom_voladj_{name}"] = (raw / denom.replace(0.0, np.nan)).clip(-10.0, 10.0).where(active)
        rows.append({"signal": f"mom_raw_{name}", "family": "momentum_skip_recent", "formula": f"log(P_t-21/P_t-{lookback})", "uses": "adjusted close", "lookahead_risk": "trailing close only"})
        rows.append({"signal": f"mom_voladj_{name}", "family": "momentum_skip_recent", "formula": f"log(P_t-21/P_t-{lookback}) / trailing sigma_60", "uses": "adjusted close", "lookahead_risk": "trailing close only"})

    for horizon in (1, 2, 5, 10, 20):
        signals[f"reversal_{horizon}"] = (-trailing_signal_return(price, horizon)).where(active)
        rows.append({"signal": f"reversal_{horizon}", "family": "short_reversal", "formula": f"-log(P_t/P_t-{horizon})", "uses": "adjusted close", "lookahead_risk": "trailing close only"})

    residual = rolling_residual_returns(data)
    for horizon in (20, 60, 120, 250):
        signal = residual.shift(1).rolling(horizon, min_periods=max(5, horizon // 2)).sum()
        signals[f"resid_mom_{horizon}"] = signal.where(active)
        rows.append({"signal": f"resid_mom_{horizon}", "family": "residual_momentum", "formula": f"{horizon}D sum of residuals from rolling stock ~ market + sector regression", "uses": "adjusted close, benchmark, sector", "lookahead_risk": "regression coefficients shifted one day"})

    price_vol = rob.mixed_vol(price.diff())
    for fast, slow in ((4, 16), (8, 32), (16, 64), (32, 128), (64, 256)):
        signals[f"ewmac_{fast}_{slow}"] = rob.ewmac(price, price_vol, fast, slow).clip(-20.0, 20.0).where(active)
        rows.append({"signal": f"ewmac_{fast}_{slow}", "family": "trend_ewmac", "formula": f"Rob EWMAC({fast},{slow})", "uses": "adjusted close", "lookahead_risk": "uses existing Rob trailing volatility framework"})

    for window in (20, 60, 120, 250):
        signals[f"breakout_{window}"] = breakout(price, window).where(active)
        rows.append({"signal": f"breakout_{window}", "family": "breakout", "formula": f"position in prior {window}D high-low range", "uses": "adjusted close", "lookahead_risk": "range uses price.shift(1)"})

    for window in (10, 20, 60, 120):
        signals[f"low_vol_{window}"] = (-trailing_vol(price, window)).where(active)
        rows.append({"signal": f"low_vol_{window}", "family": "low_vol", "formula": f"-rolling {window}D annualized volatility", "uses": "adjusted close", "lookahead_risk": "trailing close only"})

    return signals, pd.DataFrame(rows)


def rank_frame(frame: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    masked = frame.where(active).replace([np.inf, -np.inf], np.nan)
    counts = masked.notna().sum(axis=1)
    ranks = masked.rank(axis=1, pct=True)
    return ranks.where(counts >= MIN_CROSS_SECTION)


def turnover_and_persistence(signal: pd.DataFrame, active: pd.DataFrame) -> dict[str, float]:
    ranks = rank_frame(signal.loc[START:DEV_END], active.loc[START:DEV_END])
    counts = ranks.notna().sum(axis=1)
    q_top = ranks.gt(0.8)
    q_bottom = ranks.le(0.2)
    q_weights = q_top.div(q_top.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    q_weights = q_weights - q_bottom.div(q_bottom.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    d_top = ranks.gt(0.9)
    d_bottom = ranks.le(0.1)
    d_weights = d_top.div(d_top.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    d_weights = d_weights - d_bottom.div(d_bottom.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    q_turnover = q_weights.diff().abs().sum(axis=1).where(counts >= MIN_CROSS_SECTION)
    d_turnover = d_weights.diff().abs().sum(axis=1).where(counts >= MIN_CROSS_SECTION)
    rank_persistence_1d = ranks.corrwith(ranks.shift(1), axis=1, method="spearman")
    rank_persistence_5d = ranks.corrwith(ranks.shift(5), axis=1, method="spearman")
    return {
        "quintile_turnover_annual": float(q_turnover.mean() * BUSINESS_DAYS),
        "quintile_cost_annual": float(q_turnover.mean() * BUSINESS_DAYS * COST_PER_DOLLAR),
        "decile_turnover_annual": float(d_turnover.mean() * BUSINESS_DAYS),
        "decile_cost_annual": float(d_turnover.mean() * BUSINESS_DAYS * COST_PER_DOLLAR),
        "signal_persistence_1d": float(rank_persistence_1d.mean()),
        "signal_persistence_5d": float(rank_persistence_5d.mean()),
    }


def bucket_mean(target: pd.DataFrame, ranks: pd.DataFrame, lower: float, upper: float) -> pd.Series:
    mask = ranks.gt(lower) & ranks.le(upper)
    return target.where(mask).mean(axis=1)


def monotonicity(bucket_returns: list[float]) -> float:
    values = pd.Series(bucket_returns, index=np.arange(1, len(bucket_returns) + 1), dtype=float)
    if values.notna().sum() < 3:
        return math.nan
    return float(values.index.to_series().corr(values, method="spearman"))


def evaluate_signal(
    signal: pd.DataFrame,
    target: pd.DataFrame,
    active: pd.DataFrame,
    horizon: int,
    common_stats: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    signal = signal.loc[START:DEV_END]
    target = target.loc[START:DEV_END]
    active = active.loc[START:DEV_END]
    ranks = rank_frame(signal, active)
    target = target.where(active & ranks.notna())
    target_ranks = target.rank(axis=1)
    daily_ic = ranks.rank(axis=1).corrwith(target_ranks, axis=1, method="spearman").dropna()

    quintiles = [bucket_mean(target, ranks, i / 5.0, (i + 1) / 5.0) for i in range(5)]
    deciles = [bucket_mean(target, ranks, i / 10.0, (i + 1) / 10.0) for i in range(10)]
    q_means = [series.mean() for series in quintiles]
    d_means = [series.mean() for series in deciles]
    q5_q1 = quintiles[-1] - quintiles[0]
    d10_d1 = deciles[-1] - deciles[0]

    ic_std = float(daily_ic.std())
    mean_ic = float(daily_ic.mean()) if not daily_ic.empty else math.nan
    gross_q = annualized_horizon_return(float(q5_q1.mean()), horizon)
    gross_d = annualized_horizon_return(float(d10_d1.mean()), horizon)
    row = {
        "mean_rank_ic": mean_ic,
        "median_rank_ic": float(daily_ic.median()) if not daily_ic.empty else math.nan,
        "ic_std": ic_std,
        "daily_ic_ir": mean_ic / ic_std if ic_std else math.nan,
        "annualized_ic_ir": mean_ic / ic_std * math.sqrt(BUSINESS_DAYS) if ic_std else math.nan,
        "positive_ic_days": float((daily_ic > 0.0).mean()) if not daily_ic.empty else math.nan,
        "top_quintile_return_ann": annualized_horizon_return(float(quintiles[-1].mean()), horizon),
        "bottom_quintile_return_ann": annualized_horizon_return(float(quintiles[0].mean()), horizon),
        "q5_q1_spread_ann": gross_q,
        "q5_q1_net_spread_ann": gross_q - common_stats["quintile_cost_annual"],
        "top_decile_return_ann": annualized_horizon_return(float(deciles[-1].mean()), horizon),
        "bottom_decile_return_ann": annualized_horizon_return(float(deciles[0].mean()), horizon),
        "d10_d1_spread_ann": gross_d,
        "d10_d1_net_spread_ann": gross_d - common_stats["decile_cost_annual"],
        "quintile_monotonicity": monotonicity(q_means),
        "decile_monotonicity": monotonicity(d_means),
        "mean_daily_observations": float((active & ranks.notna() & target.notna()).sum(axis=1).mean()),
        "ic_days": int(len(daily_ic)),
        **common_stats,
    }

    yearly_rows = []
    years = sorted(set(daily_ic.index.year))
    for year in years:
        year_mask = daily_ic.index.year == year
        year_q = q5_q1.reindex(daily_ic.index).loc[year_mask]
        yearly_rows.append(
            {
                "year": int(year),
                "mean_rank_ic": float(daily_ic.loc[year_mask].mean()),
                "q5_q1_spread_ann": annualized_horizon_return(float(year_q.mean()), horizon),
                "q5_q1_net_spread_ann": annualized_horizon_return(float(year_q.mean()), horizon)
                - common_stats["quintile_cost_annual"],
                "ic_days": int(year_mask.sum()),
            }
        )
    return row, pd.DataFrame(yearly_rows)


def gate_row(row: pd.Series, yearly: pd.DataFrame) -> dict[str, object]:
    positive_ic_years = int((yearly["mean_rank_ic"] > 0.0).sum()) if not yearly.empty else 0
    positive_spread_years = int((yearly["q5_q1_net_spread_ann"] > 0.0).sum()) if not yearly.empty else 0
    tested_years = int(yearly["year"].nunique()) if not yearly.empty else 0
    checks = {
        "enough_test_years": tested_years >= 4,
        "mean_ic_positive": row["mean_rank_ic"] > 0.0,
        "ic_ir_positive": row["daily_ic_ir"] > 0.0,
        "spread_direction_correct": row["q5_q1_spread_ann"] > 0.0,
        "net_spread_positive": row["q5_q1_net_spread_ann"] > 0.0,
        "quintile_monotonicity_positive": row["quintile_monotonicity"] > 0.0,
        "positive_ic_years_ge_half": positive_ic_years >= max(1, math.ceil(tested_years / 2)),
        "positive_spread_years_ge_half": positive_spread_years >= max(1, math.ceil(tested_years / 2)),
    }
    score = int(sum(bool(value) for value in checks.values()))
    strict_pass = all(bool(value) for value in checks.values())
    return {
        "gate_score": score,
        "gate_status": "PASS_CANDIDATE" if strict_pass else "DROP",
        "positive_ic_years": positive_ic_years,
        "positive_spread_years": positive_spread_years,
        "tested_years": tested_years,
        **checks,
    }


def run_universe(key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Loading {key}", flush=True)
    data = load_universe(key)
    print(f"Building signals for {key}", flush=True)
    signals, dictionary = build_signal_library(data)
    dictionary.insert(0, "universe", data.label)
    targets = {horizon: forward_excess_return(data, horizon) for horizon in HORIZONS}

    rows = []
    yearly_rows = []
    gate_rows = []
    for signal_name, signal in signals.items():
        common_stats = turnover_and_persistence(signal, data.active)
        print(f"{key}: {signal_name}", flush=True)
        for horizon in HORIZONS:
            metrics, yearly = evaluate_signal(signal, targets[horizon], data.active, horizon, common_stats)
            row = {
                "universe": data.label,
                "universe_key": key,
                "signal": signal_name,
                "horizon": horizon,
                **metrics,
            }
            rows.append(row)
            if not yearly.empty:
                yearly.insert(0, "horizon", horizon)
                yearly.insert(0, "signal", signal_name)
                yearly.insert(0, "universe_key", key)
                yearly.insert(0, "universe", data.label)
                yearly_rows.append(yearly)
            gate = gate_row(pd.Series(row), yearly)
            gate_rows.append({"universe": data.label, "universe_key": key, "signal": signal_name, "horizon": horizon, **gate})

    matrix = pd.DataFrame(rows)
    yearly = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    gates = pd.DataFrame(gate_rows)
    return matrix, yearly, dictionary, gates


def plot_ic_heatmap(matrix: pd.DataFrame, out_dir: Path) -> None:
    pivot = matrix.pivot_table(index="signal", columns=["universe_key", "horizon"], values="mean_rank_ic", aggfunc="first")
    fig, ax = plt.subplots(figsize=(max(14, pivot.shape[1] * 0.7), max(8, pivot.shape[0] * 0.28)))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-0.03, vmax=0.03)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{u}-{h}" for u, h in pivot.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Mean Rank IC by Signal, Market, Horizon")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_rank_ic_heatmap.png", dpi=180)
    plt.close(fig)


def write_report(matrix: pd.DataFrame, gates: pd.DataFrame, dictionary: pd.DataFrame, out_dir: Path) -> None:
    top = matrix.sort_values(["q5_q1_net_spread_ann", "mean_rank_ic"], ascending=False).head(20)
    candidates = gates[gates["gate_status"].eq("PASS_CANDIDATE")].merge(
        matrix,
        on=["universe", "universe_key", "signal", "horizon"],
        how="left",
    )
    family_map = dictionary.drop_duplicates("signal").set_index("signal")["family"].to_dict()
    candidates["family"] = candidates["signal"].map(family_map)
    family_candidates = (
        candidates.groupby(["universe", "family"], as_index=False)
        .agg(
            candidates=("signal", "count"),
            best_mean_ic=("mean_rank_ic", "max"),
            best_net_spread=("q5_q1_net_spread_ann", "max"),
            best_horizon=("horizon", lambda values: int(values.iloc[0]) if len(values) else math.nan),
        )
        .sort_values(["universe", "best_mean_ic"], ascending=[True, False])
        if not candidates.empty
        else pd.DataFrame()
    )

    lines = [
        "# Stage 02-04 Alpha Signal Lab",
        "",
        "- Stage 02 framework: PASS. Standard IC, bucket, turnover, cost, persistence diagnostics generated for every predefined signal.",
        "- Stage 03 signal library: PASS. Only adjusted-close close-to-close signals are used; OHLC, volume, market cap, shares outstanding are excluded by Stage 01 gate.",
        "- Stage 04 horizon discovery: candidate signals are marked `PASS_CANDIDATE` only if every strict core check passes; dropped rows should not be rescued by ML.",
        "- Target alignment: features use information known at close t; forward excess return starts after a one-trading-day execution lag.",
        "- Development window: 2016-01-01 to 2023-12-31. Final holdout 2024-2026 remains untouched.",
        "",
        "## Candidate Count",
        "",
        "| Gate Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in gates["gate_status"].value_counts().items():
        lines.append(f"| {status} | {count} |")

    lines += [
        "",
        "## Best Rows By Net Quintile Spread",
        "",
        "| Universe | Signal | Horizon | Mean Rank IC | Daily IC IR | Q5-Q1 Net Ann | Q5-Q1 Gross Ann | Pos IC Days | Quintile Mono | Ann Turnover |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['universe']} | {row['signal']} | {int(row['horizon'])} | {fmt_num(row['mean_rank_ic'])} | "
            f"{fmt_num(row['daily_ic_ir'])} | {fmt_pct(row['q5_q1_net_spread_ann'])} | {fmt_pct(row['q5_q1_spread_ann'])} | "
            f"{fmt_pct(row['positive_ic_days'])} | {fmt_num(row['quintile_monotonicity'])} | {fmt_pct(row['quintile_turnover_annual'])} |"
        )

    if not family_candidates.empty:
        lines += [
            "",
            "## Candidate Families",
            "",
            "| Universe | Family | Candidate Rows | Best Mean IC | Best Net Spread |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for _, row in family_candidates.iterrows():
            lines.append(
                f"| {row['universe']} | {row['family']} | {int(row['candidates'])} | "
                f"{fmt_num(row['best_mean_ic'])} | {fmt_pct(row['best_net_spread'])} |"
            )

    lines += [
        "",
        "## Gate Rules",
        "",
        "A row is `PASS_CANDIDATE` only when every strict check passes: at least four tested years, positive mean Rank IC, positive IC IR, correct gross spread direction, positive net spread after base cost, positive quintile monotonicity, positive IC in at least half tested years, and positive net spread in at least half tested years.",
        "",
        "## Files",
        "",
        "- `signal_market_horizon_matrix.csv`: main Signal x Market x Horizon table.",
        "- `signal_yearly_diagnostics.csv`: year-by-year IC and spread checks.",
        "- `horizon_gate.csv`: pass/drop gate for every row.",
        "- `signal_dictionary.csv`: formulas and leakage notes.",
        "- `mean_rank_ic_heatmap.png`: visual overview.",
    ]
    (out_dir / "stage_02_04_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    matrix_parts = []
    yearly_parts = []
    dictionary_parts = []
    gate_parts = []
    for key in args.universes:
        matrix, yearly, dictionary, gates = run_universe(key)
        matrix_parts.append(matrix)
        yearly_parts.append(yearly)
        dictionary_parts.append(dictionary)
        gate_parts.append(gates)

    matrix = pd.concat(matrix_parts, ignore_index=True)
    yearly = pd.concat(yearly_parts, ignore_index=True)
    dictionary = pd.concat(dictionary_parts, ignore_index=True)
    gates = pd.concat(gate_parts, ignore_index=True)

    matrix.to_csv(OUT / "signal_market_horizon_matrix.csv", index=False)
    yearly.to_csv(OUT / "signal_yearly_diagnostics.csv", index=False)
    dictionary.to_csv(OUT / "signal_dictionary.csv", index=False)
    gates.to_csv(OUT / "horizon_gate.csv", index=False)
    plot_ic_heatmap(matrix, OUT)
    write_report(matrix, gates, dictionary, OUT)
    print(OUT / "stage_02_04_report.md")


if __name__ == "__main__":
    main()
