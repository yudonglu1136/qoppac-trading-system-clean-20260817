#!/usr/bin/env python3
"""Driver analysis for point-in-time annual ranked stock trend systems."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_point_in_time_annual_ranked_long_only as pit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "point_in_time_annual_ranked_long_only" / "driver_analysis"
START = "2016-01-01"
END = "2026-08-07"
BUSINESS_DAYS = 252.0

UNIVERSES = {
    "sp500": "SPY / S&P 500",
    "eem": "EM / EEM",
    "efa": "Developed / EFA",
}


def stats(returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    ann = returns.mean() * BUSINESS_DAYS
    vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
        "ann_return": ann,
        "vol": vol,
        "sharpe": ann / vol if vol else np.nan,
        "mdd": (equity / equity.cummax() - 1.0).min(),
        "total_return": equity.iloc[-1] - 1.0,
    }


def compound_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(how="all").groupby(pd.Grouper(freq="YE")).apply(lambda x: (1.0 + x).prod() - 1.0)


def load_price(key: str, annual: pd.DataFrame) -> pd.DataFrame:
    data_dir = pit.DATA_ROOT / key
    price = pd.read_csv(data_dir / "adj_close.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    if key in {"eem", "efa"}:
        price = pit.convert_em_prices_to_usd(price, annual, data_dir, False)
    price = price.loc[:END].ffill(limit=5)
    usable = [column for column in price.columns if price[column].notna().sum() >= pit.MIN_HISTORY_DAYS]
    return price[usable]


def load_benchmark(key: str, benchmark_ticker: str) -> pd.Series:
    path = pit.DATA_ROOT / key / "benchmark_adj_close.csv"
    bench = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()[benchmark_ticker]
    return bench.loc[START:END].pct_change().rename("benchmark")


def reconstruct_top_weights(price: pd.DataFrame, selections: pd.DataFrame, total_names: int) -> pd.DataFrame:
    rebalance = sorted(pd.to_datetime(selections["date"].unique()))
    target = pd.DataFrame(0.0, index=rebalance, columns=price.columns)
    subset = selections[selections["portfolio_size"].eq(total_names)]
    for date, frame in subset.groupby("date"):
        names = [ticker for ticker in frame["ticker"].astype(str) if ticker in target.columns]
        if names:
            target.loc[pd.Timestamp(date), names] = 1.0 / total_names
    return target.reindex(price.index).ffill().fillna(0.0)


def reconstruct_scored_equal_weights(
    price: pd.DataFrame,
    annual: pd.DataFrame,
    forecast: pd.DataFrame,
    rebalance_dates: pd.Index,
) -> pd.DataFrame:
    year_members = pit.membership_by_year(annual)
    target = pd.DataFrame(0.0, index=rebalance_dates, columns=price.columns)
    for date in rebalance_dates:
        allowed = list(year_members.get(int(date.year), set()))
        cols = [ticker for ticker in allowed if ticker in forecast.columns]
        scores = forecast.loc[date, cols].replace([np.inf, -np.inf], np.nan).dropna()
        if scores.empty:
            continue
        target.loc[date, scores.index] = 1.0 / len(scores)
    return target.reindex(price.index).ffill().fillna(0.0)


def return_from_weights(price: pd.DataFrame, weights: pd.DataFrame, *, max_abs_daily_return: float | None = None) -> pd.Series:
    returns = price.pct_change()
    if max_abs_daily_return is not None:
        returns = returns.mask(returns.abs() > max_abs_daily_return)
    returns = returns.fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    return (held * returns).sum(axis=1)


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def regime_stats(streams: pd.DataFrame) -> pd.DataFrame:
    daily = streams.dropna(how="any")
    rows = []
    benchmark = daily["benchmark"]
    for name in ["top40_net", "top40_gross", "scored_equal"]:
        series = daily[name]
        beta = np.cov(series, benchmark)[0, 1] / np.var(benchmark)
        corr = series.corr(benchmark)
        rows.append(
            {
                "stream": name,
                "daily_corr_to_benchmark": corr,
                "daily_beta_to_benchmark": beta,
                "avg_ret_when_benchmark_up": series[benchmark > 0].mean() * BUSINESS_DAYS,
                "avg_ret_when_benchmark_down": series[benchmark < 0].mean() * BUSINESS_DAYS,
                "hit_rate_when_benchmark_up": (series[benchmark > 0] > 0).mean(),
                "hit_rate_when_benchmark_down": (series[benchmark < 0] > 0).mean(),
            }
        )
    monthly = (1.0 + daily).resample("ME").prod() - 1.0
    up = monthly["benchmark"] > 0
    down = monthly["benchmark"] < 0
    for row in rows:
        name = row["stream"]
        row["monthly_up_capture"] = monthly.loc[up, name].mean() / monthly.loc[up, "benchmark"].mean()
        row["monthly_down_capture"] = monthly.loc[down, name].mean() / monthly.loc[down, "benchmark"].mean()
    return pd.DataFrame(rows)


def metadata_for_year(annual: pd.DataFrame, year: int) -> pd.DataFrame:
    cols = ["symbol"]
    for candidate in ["name", "sector", "location", "market_currency"]:
        if candidate in annual.columns:
            cols.append(candidate)
    meta = annual[annual["year"].eq(year)][cols].drop_duplicates("symbol").set_index("symbol")
    return meta


def contribution_tables(
    key: str,
    price: pd.DataFrame,
    annual: pd.DataFrame,
    top_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = price.pct_change().mask(lambda frame: frame.abs() > 0.8).fillna(0.0)
    contrib = top_weights.shift(1).fillna(0.0) * returns
    by_symbol = contrib.sum().sort_values(ascending=False)
    name_map = annual.drop_duplicates("symbol").set_index("symbol")["name"].to_dict()
    top_symbols = pd.DataFrame(
        {
            "symbol": by_symbol.index,
            "name": [name_map.get(symbol, "") for symbol in by_symbol.index],
            "contribution": by_symbol.values,
        }
    )

    group_rows = []
    for year, frame in contrib.groupby(contrib.index.year):
        meta = metadata_for_year(annual, int(year))
        if meta.empty:
            continue
        for group_col in ["sector", "location", "market_currency"]:
            if group_col not in meta.columns:
                continue
            mapper = meta[group_col].astype(str).to_dict()
            groups = pd.Series({column: mapper.get(column, "Unknown") for column in frame.columns})
            annual_group = frame.sum().groupby(groups).sum().sort_values(ascending=False)
            for group, value in annual_group.items():
                group_rows.append({"universe": key, "year": int(year), "group_type": group_col, "group": group, "contribution": value})
    group_contrib = pd.DataFrame(group_rows)
    if group_contrib.empty:
        return top_symbols.head(15), top_symbols.tail(15), group_contrib, group_contrib

    full_period = (
        group_contrib.groupby(["group_type", "group"])["contribution"]
        .sum()
        .reset_index()
        .sort_values(["group_type", "contribution"], ascending=[True, False])
    )
    worst_years = group_contrib.sort_values("contribution").head(30)
    return top_symbols.head(15), top_symbols.tail(15), full_period, worst_years


def forecast_ic_table(
    price: pd.DataFrame,
    annual: pd.DataFrame,
    forecast: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    """Measure whether each weekly forecast rank predicts next holding-period returns."""
    year_members = pit.membership_by_year(annual)
    rebalance_dates = pd.Index(sorted(pd.to_datetime(selections["date"].unique())))
    clean_returns = price.pct_change().mask(lambda frame: frame.abs() > 0.8)
    rows = []
    for index, date in enumerate(rebalance_dates):
        year = int(date.year)
        if year not in year_members:
            continue
        valid_dates = forecast.index[forecast.index >= date]
        if valid_dates.empty:
            continue
        signal_date = valid_dates[0]
        if index + 1 < len(rebalance_dates):
            next_valid_dates = forecast.index[forecast.index >= rebalance_dates[index + 1]]
            next_date = next_valid_dates[0] if len(next_valid_dates) else pd.Timestamp(END)
        else:
            next_date = pd.Timestamp(END)
        horizon = clean_returns[(clean_returns.index > signal_date) & (clean_returns.index <= next_date)]
        if horizon.empty:
            continue
        forward = (1.0 + horizon).prod(min_count=1) - 1.0
        allowed = [ticker for ticker in year_members[year] if ticker in forecast.columns]
        scores = forecast.loc[signal_date, allowed].replace([np.inf, -np.inf], np.nan).dropna()
        aligned = pd.concat([scores.rename("forecast"), forward.rename("forward_return")], axis=1).dropna()
        if len(aligned) < 50:
            continue
        top_n = min(40, len(aligned))
        ordered = aligned.sort_values("forecast", ascending=False)
        top = ordered.head(top_n)
        bottom = ordered.tail(top_n)
        rows.append(
            {
                "date": signal_date.date(),
                "year": year,
                "scored_members": len(aligned),
                "spearman_ic": aligned["forecast"].rank().corr(aligned["forward_return"].rank()),
                "top40_forward_return": top["forward_return"].mean(),
                "all_scored_forward_return": aligned["forward_return"].mean(),
                "bottom40_forward_return": bottom["forward_return"].mean(),
                "top40_minus_all": top["forward_return"].mean() - aligned["forward_return"].mean(),
                "top40_minus_bottom40": top["forward_return"].mean() - bottom["forward_return"].mean(),
                "forecast_spread_top_bottom": top["forecast"].mean() - bottom["forecast"].mean(),
            }
        )
    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return weekly
    annual = (
        weekly.groupby("year")
        .agg(
            rebalance_count=("date", "count"),
            avg_scored_members=("scored_members", "mean"),
            avg_spearman_ic=("spearman_ic", "mean"),
            median_spearman_ic=("spearman_ic", "median"),
            positive_ic_rate=("spearman_ic", lambda s: (s > 0).mean()),
            avg_top40_forward_return=("top40_forward_return", "mean"),
            avg_all_scored_forward_return=("all_scored_forward_return", "mean"),
            avg_bottom40_forward_return=("bottom40_forward_return", "mean"),
            avg_top40_minus_all=("top40_minus_all", "mean"),
            avg_top40_minus_bottom40=("top40_minus_bottom40", "mean"),
            avg_forecast_spread_top_bottom=("forecast_spread_top_bottom", "mean"),
        )
        .reset_index()
    )
    return annual


def benchmark_weight_coverage(annual: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    if "weight" not in annual.columns:
        return pd.DataFrame()
    selected = selections[selections["portfolio_size"].eq(40)].copy()
    selected["year"] = selected["date"].dt.year
    meta = annual[["year", "symbol", "weight"]].drop_duplicates(["year", "symbol"]).copy()
    meta["weight"] = pd.to_numeric(meta["weight"], errors="coerce")
    rows = []
    for (year, date), frame in selected.groupby(["year", "date"]):
        year_meta = meta[meta["year"].eq(year)].dropna(subset=["weight"]).copy()
        if year_meta.empty:
            continue
        ranked_weights = year_meta.sort_values("weight", ascending=False).reset_index(drop=True)
        year_meta = year_meta.set_index("symbol")
        selected_symbols = frame["ticker"].astype(str).tolist()
        selected_weights = year_meta.reindex(selected_symbols)["weight"].dropna()
        top10 = set(ranked_weights.head(10)["symbol"])
        top20 = set(ranked_weights.head(20)["symbol"])
        rows.append(
            {
                "year": int(year),
                "date": pd.Timestamp(date).date(),
                "selected_count_with_weight": int(selected_weights.notna().sum()),
                "selected_benchmark_weight_sum_pct": selected_weights.sum(),
                "benchmark_top40_weight_sum_pct": ranked_weights.head(40)["weight"].sum(),
                "selected_avg_benchmark_weight_pct": selected_weights.mean(),
                "universe_avg_benchmark_weight_pct": year_meta["weight"].mean(),
                "selected_top10_count": sum(symbol in top10 for symbol in selected_symbols),
                "selected_top20_count": sum(symbol in top20 for symbol in selected_symbols),
            }
        )
    weekly = pd.DataFrame(rows)
    if weekly.empty:
        return weekly
    return (
        weekly.groupby("year")
        .agg(
            rebalance_count=("date", "count"),
            avg_selected_count_with_weight=("selected_count_with_weight", "mean"),
            avg_selected_benchmark_weight_sum_pct=("selected_benchmark_weight_sum_pct", "mean"),
            avg_benchmark_top40_weight_sum_pct=("benchmark_top40_weight_sum_pct", "mean"),
            avg_selected_avg_benchmark_weight_pct=("selected_avg_benchmark_weight_pct", "mean"),
            avg_universe_avg_benchmark_weight_pct=("universe_avg_benchmark_weight_pct", "mean"),
            avg_selected_top10_count=("selected_top10_count", "mean"),
            avg_selected_top20_count=("selected_top20_count", "mean"),
        )
        .reset_index()
    )


def analyze_universe(key: str, label: str) -> dict[str, pd.DataFrame]:
    out_dir = ROOT / "backtests" / "point_in_time_annual_ranked_long_only" / key
    data_dir = pit.DATA_ROOT / key
    annual = pd.read_csv(out_dir / "annual_constituents_used.csv")
    annual = annual[(annual["year"] >= 2016) & (annual["year"] <= 2026)].copy()
    price = load_price(key, annual)

    spec = next(spec for spec in pit.UNIVERSES if spec.key == key)
    benchmark = load_benchmark(key, spec.benchmark_ticker)
    selections = pd.read_csv(out_dir / "rebalance_selections.csv", parse_dates=["date"])
    selections = selections[selections["date"].between(pd.Timestamp(START), pd.Timestamp(END))]
    top_weights = reconstruct_top_weights(price, selections, 40)
    daily = pd.read_csv(out_dir / "portfolio_daily_top40.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    top_net = daily["net_return"].loc[START:END].rename("top40_net")
    top_gross = daily["gross_return"].loc[START:END].rename("top40_gross")
    costs = daily["costs"].loc[START:END].rename("costs")

    forecast, _ = pit.build_forecasts_no_lookahead(price)
    rebalance_dates = pd.Index(sorted(selections["date"].unique()))
    eq_weights = reconstruct_scored_equal_weights(price, annual, forecast, rebalance_dates)
    scored_equal = return_from_weights(price, eq_weights, max_abs_daily_return=0.8).loc[START:END].rename("scored_equal")

    streams = pd.concat([top_net, top_gross, scored_equal, benchmark], axis=1).dropna(how="any")
    perf = pd.DataFrame([{"universe": label, "stream": column, **stats(streams[column])} for column in streams.columns])
    annual_returns = compound_by_year(streams)
    annual_returns.index = annual_returns.index.year
    annual_returns["top40_net_minus_benchmark"] = annual_returns["top40_net"] - annual_returns["benchmark"]
    annual_returns["selection_alpha_gross_vs_scored_equal"] = annual_returns["top40_gross"] - annual_returns["scored_equal"]
    annual_returns["scored_equal_minus_benchmark"] = annual_returns["scored_equal"] - annual_returns["benchmark"]
    annual_returns["cost_drag"] = annual_returns["top40_net"] - annual_returns["top40_gross"]

    drawdowns = pd.DataFrame({column: drawdown_series(streams[column]) for column in streams.columns})
    dd_windows = []
    for column in ["top40_net", "scored_equal", "benchmark"]:
        dd = drawdowns[column]
        trough = dd.idxmin()
        dd_windows.append({"stream": column, "trough": trough.date(), "max_drawdown": dd.loc[trough]})
    dd_table = pd.DataFrame(dd_windows)
    reg = regime_stats(streams)
    reg.insert(0, "universe", label)

    selected_forecast = selections[selections["portfolio_size"].eq(40)].groupby("date")["forecast"].agg(["mean", "min", "max"])
    selected_forecast.index = pd.to_datetime(selected_forecast.index)
    selected_forecast["year"] = selected_forecast.index.year
    forecast_by_year = selected_forecast.groupby("year").agg({"mean": "mean", "min": "mean", "max": "mean"})

    top_symbols, bottom_symbols, group_contrib, worst_group_years = contribution_tables(key, price, annual, top_weights)
    forecast_ic = forecast_ic_table(price, annual, forecast, selections)
    weight_coverage = benchmark_weight_coverage(annual, selections)

    result = {
        "performance": perf,
        "annual_decomposition": annual_returns.reset_index(names="year"),
        "drawdowns": dd_table,
        "regime": reg,
        "forecast_by_year": forecast_by_year.reset_index(),
        "forecast_ic": forecast_ic,
        "benchmark_weight_coverage": weight_coverage,
        "top_symbol_contributors": top_symbols,
        "bottom_symbol_contributors": bottom_symbols,
        "group_contribution": group_contrib,
        "worst_group_years": worst_group_years,
    }
    for name, frame in result.items():
        frame.to_csv(OUT / f"{key}_{name}.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_perf = []
    all_annual = []
    all_regime = []
    all_ic = []
    all_weight_coverage = []
    for key, label in UNIVERSES.items():
        print(f"Analyzing {label}", flush=True)
        result = analyze_universe(key, label)
        all_perf.append(result["performance"])
        annual = result["annual_decomposition"].copy()
        annual.insert(0, "universe", label)
        all_annual.append(annual)
        all_regime.append(result["regime"])
        if not result["forecast_ic"].empty:
            ic = result["forecast_ic"].copy()
            ic.insert(0, "universe", label)
            all_ic.append(ic)
        if not result["benchmark_weight_coverage"].empty:
            coverage = result["benchmark_weight_coverage"].copy()
            coverage.insert(0, "universe", label)
            all_weight_coverage.append(coverage)

    pd.concat(all_perf, ignore_index=True).to_csv(OUT / "all_performance.csv", index=False)
    pd.concat(all_annual, ignore_index=True).to_csv(OUT / "all_annual_decomposition.csv", index=False)
    pd.concat(all_regime, ignore_index=True).to_csv(OUT / "all_regime_stats.csv", index=False)
    if all_ic:
        pd.concat(all_ic, ignore_index=True).to_csv(OUT / "all_forecast_ic.csv", index=False)
    if all_weight_coverage:
        pd.concat(all_weight_coverage, ignore_index=True).to_csv(OUT / "all_benchmark_weight_coverage.csv", index=False)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
