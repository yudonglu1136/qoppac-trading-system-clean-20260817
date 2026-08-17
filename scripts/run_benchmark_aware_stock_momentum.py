#!/usr/bin/env python3
"""Benchmark-aware stock momentum tests using point-in-time ETF holdings.

The current local SPY/QQQ point-in-time universes do not include historical
index weights, so this script intentionally starts with EEM/EFA where archived
iShares holdings provide point-in-time constituent weights and sectors.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402


OUT = ROOT / "backtests" / "benchmark_aware_stock_momentum"
SOURCE_OUT = ROOT / "backtests" / "point_in_time_annual_ranked_long_only"
BUSINESS_DAYS = 252.0
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2026-08-07"

UNIVERSES = {
    "sp500": "SPY / S&P 500",
    "eem": "Emerging Markets / EEM",
    "efa": "Developed Markets ex-US / EFA",
}

SPY_HOLDINGS_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"


def performance_stats(returns: pd.Series) -> dict[str, float | str]:
    returns = returns.dropna()
    equity = (1.0 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    ann = returns.mean() * BUSINESS_DAYS
    vol = returns.std() * math.sqrt(BUSINESS_DAYS)
    return {
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "years": years,
        "total_return": equity.iloc[-1] - 1.0,
        "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
        "ann_return": ann,
        "vol": vol,
        "sharpe": ann / vol if vol else np.nan,
        "mdd": (equity / equity.cummax() - 1.0).min(),
    }


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 qoppac benchmark-aware stock momentum"})
    return session


def spy_cdx_captures(start: str, end: str) -> list[dict[str, str]]:
    from_year = str(max(2020, pd.Timestamp(start).year - 1))
    params = {
        "url": SPY_HOLDINGS_URL,
        "matchType": "exact",
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "filter": "statuscode:200",
        "from": from_year,
        "to": str(pd.Timestamp(end).year),
        "limit": "500",
    }
    response = pit.get_with_retries(
        requests_session(),
        "https://web.archive.org/cdx/",
        params=params,
        timeout=60,
    )
    data = response.json()
    rows = [
        {"timestamp": row[0], "original": row[1], "mimetype": row[3], "digest": row[4]}
        for row in data[1:]
    ]
    unique = {row["timestamp"]: row for row in rows}
    return [unique[key] for key in sorted(unique)]


def parse_spy_holdings_file(path: Path, capture: dict[str, str]) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    asof = ""
    for value in raw.iloc[:, 1].dropna().astype(str):
        if "As of" in value:
            asof = value.replace("As of", "").strip()
            break
    holding_asof = pd.to_datetime(asof, errors="coerce")

    header_row = None
    for idx, row in raw.iterrows():
        labels = [str(value).strip().lower() for value in row.tolist()]
        if "weight" in labels and "sector" in labels:
            header_row = idx
            break
    if header_row is None:
        raise ValueError(f"Could not find SPY holdings header in {path}")

    frame = pd.read_excel(path, sheet_name=0, header=header_row)
    frame.columns = [str(column).strip() for column in frame.columns]
    ticker_col = "Ticker" if "Ticker" in frame.columns else "Identifier"
    required = {"Name", ticker_col, "Weight", "Sector"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"SPY holdings missing columns {sorted(missing)} in {path}")

    out = frame[["Name", ticker_col, "Weight", "Sector"]].copy()
    out.columns = ["name", "symbol_raw", "weight", "sector"]
    out["symbol"] = out["symbol_raw"].map(lambda value: pit.clean_symbol(value, ""))
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["symbol", "weight"])
    out = out[out["symbol"].astype(str).str.match(r"^[A-Z0-9-]+$")]
    out = out[out["weight"] > 0.0]

    capture_ts = pd.to_datetime(capture["timestamp"], format="%Y%m%d%H%M%S")
    out["year"] = capture_ts.year
    out["snapshot_date"] = capture_ts.date().isoformat()
    out["holding_asof"] = "" if pd.isna(holding_asof) else holding_asof.date().isoformat()
    out["archive_timestamp"] = capture["timestamp"]
    out["source_url"] = f"https://web.archive.org/web/{capture['timestamp']}if_/{capture['original']}"
    return out[
        [
            "symbol",
            "name",
            "symbol_raw",
            "weight",
            "sector",
            "year",
            "snapshot_date",
            "holding_asof",
            "archive_timestamp",
            "source_url",
        ]
    ]


def load_spy_holdings(start: str, end: str) -> pd.DataFrame:
    data_dir = pit.DATA_ROOT / "sp500"
    cache = data_dir / "spy_historical_holdings_weights.csv"
    audit_cache = data_dir / "spy_historical_holdings_audit.csv"
    if cache.exists() and audit_cache.exists():
        annual = pd.read_csv(cache)
        annual["snapshot_date"] = pd.to_datetime(annual["snapshot_date"])
        return annual[annual["snapshot_date"].le(pd.Timestamp(end))].copy()

    snapshots_dir = data_dir / "spy_holdings_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    captures = spy_cdx_captures(start, end)
    if not captures:
        raise RuntimeError("No SPY holdings captures found in Wayback")

    session = requests_session()
    frames = []
    audit_rows = []
    for capture in captures:
        suffix = ".xlsx" if "spreadsheetml" in capture["mimetype"] else ".xls"
        local = snapshots_dir / f"spy_holdings_{capture['timestamp']}{suffix}"
        url = f"https://web.archive.org/web/{capture['timestamp']}if_/{capture['original']}"
        if not local.exists():
            response = session.get(url, timeout=60)
            response.raise_for_status()
            local.write_bytes(response.content)
            time.sleep(1.0)
        try:
            parsed = parse_spy_holdings_file(local, capture)
            frames.append(parsed)
            audit_rows.append(
                {
                    "archive_timestamp": capture["timestamp"],
                    "snapshot_date": parsed["snapshot_date"].iloc[0],
                    "holding_asof": parsed["holding_asof"].iloc[0],
                    "rows": len(parsed),
                    "status": "ok",
                    "source_url": url,
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "archive_timestamp": capture["timestamp"],
                    "snapshot_date": pd.to_datetime(capture["timestamp"], format="%Y%m%d%H%M%S").date().isoformat(),
                    "holding_asof": "",
                    "rows": 0,
                    "status": f"error: {exc}",
                    "source_url": url,
                }
            )

    if not frames:
        raise RuntimeError("No SPY holdings snapshots parsed")
    annual = pd.concat(frames, ignore_index=True)
    annual.to_csv(cache, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_cache, index=False)
    annual["snapshot_date"] = pd.to_datetime(annual["snapshot_date"])
    return annual[annual["snapshot_date"].le(pd.Timestamp(end))].copy()


def load_annual(key: str, start: str, end: str) -> pd.DataFrame:
    if key == "sp500":
        return load_spy_holdings(start, end)
    path = SOURCE_OUT / key / "annual_constituents_used.csv"
    if path.exists():
        annual = pd.read_csv(path)
    else:
        annual = pd.read_csv(pit.DATA_ROOT / key / "annual_constituents.csv")
    annual = annual[(annual["year"] >= pd.Timestamp(start).year) & (annual["year"] <= pd.Timestamp(end).year)].copy()
    required = {"symbol", "year", "weight", "sector"}
    missing = required.difference(annual.columns)
    if missing:
        raise ValueError(f"{key} annual constituents missing required columns: {sorted(missing)}")
    annual["weight"] = pd.to_numeric(annual["weight"], errors="coerce")
    return annual.dropna(subset=["symbol", "weight"])


def load_price(key: str, annual: pd.DataFrame, end: str) -> pd.DataFrame:
    data_dir = pit.DATA_ROOT / key
    price = pd.read_csv(data_dir / "adj_close.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    symbols = pd.Index(sorted(annual["symbol"].dropna().astype(str).unique()))
    price = price.loc[:, ~price.columns.duplicated()].reindex(columns=symbols)
    if key in {"eem", "efa"}:
        price = pit.convert_em_prices_to_usd(price, annual, data_dir, False).reindex(columns=symbols)
    price = price.loc[:end].ffill(limit=5)
    return price


def load_benchmark(key: str, start: str, end: str) -> pd.Series:
    spec = next(spec for spec in pit.UNIVERSES if spec.key == key)
    path = pit.DATA_ROOT / key / "benchmark_adj_close.csv"
    price = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()[spec.benchmark_ticker]
    return price.loc[start:end].pct_change().rename(f"Benchmark: {spec.benchmark_label}").dropna()


def year_metadata(annual: pd.DataFrame, date: pd.Timestamp, columns: pd.Index) -> pd.DataFrame:
    annual = annual.copy()
    annual["snapshot_date"] = pd.to_datetime(annual["snapshot_date"])
    available_dates = annual.loc[annual["snapshot_date"].le(date), "snapshot_date"]
    if available_dates.empty:
        return pd.DataFrame()
    snapshot_date = available_dates.max()
    frame = annual[annual["snapshot_date"].eq(snapshot_date)].copy()
    frame = frame.drop_duplicates("symbol").set_index("symbol")
    frame = frame[frame.index.isin(columns)]
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["weight"])
    frame = frame[frame["weight"] > 0.0]
    return frame


def normalized_base_weights(meta: pd.DataFrame, scores: pd.Series) -> pd.Series:
    base = meta["weight"].reindex(scores.index).dropna()
    base = base[base > 0.0]
    if base.empty:
        return base
    return base / base.sum()


def positive_signal_sleeve(base: pd.Series, scores: pd.Series, power: float = 1.0) -> pd.Series:
    aligned = pd.concat([base.rename("base"), scores.rename("score")], axis=1).dropna()
    if aligned.empty:
        return base
    positive = aligned["score"].clip(lower=0.0).pow(power)
    weighted = aligned["base"] * positive
    if weighted.sum() <= 0.0:
        return aligned["base"] / aligned["base"].sum()
    return weighted / weighted.sum()


def sector_neutral_sleeve(base: pd.Series, scores: pd.Series, meta: pd.DataFrame) -> pd.Series:
    sector = meta["sector"].astype(str).replace({"nan": "Unknown"}).reindex(base.index).fillna("Unknown")
    sector_weights = base.groupby(sector).sum()
    pieces = []
    for sector_name, sector_weight in sector_weights.items():
        names = sector[sector.eq(sector_name)].index
        sector_base = base.reindex(names).dropna()
        sector_scores = scores.reindex(sector_base.index)
        sleeve = positive_signal_sleeve(sector_base / sector_base.sum(), sector_scores)
        pieces.append(sleeve * sector_weight)
    if not pieces:
        return base
    out = pd.concat(pieces).groupby(level=0).sum()
    return out / out.sum()


def build_weights(
    forecast: pd.DataFrame,
    annual: pd.DataFrame,
    strategy: str,
    *,
    core_weight: float,
    min_active: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rebalance_dates = pit.stock_base.last_rebalance_dates(forecast.index)
    target = pd.DataFrame(0.0, index=rebalance_dates, columns=forecast.columns)
    rows: list[dict[str, float | int | str]] = []

    for date in rebalance_dates:
        meta = year_metadata(annual, date, forecast.columns)
        if meta.empty:
            continue
        scores = forecast.loc[date, meta.index].replace([np.inf, -np.inf], np.nan).dropna()
        if len(scores) < min_active:
            continue
        base = normalized_base_weights(meta, scores)
        if base.empty:
            continue
        scores = scores.reindex(base.index)
        signal = positive_signal_sleeve(base, scores)
        convex_signal = positive_signal_sleeve(base, scores, power=2.0)

        if strategy == "core80_signal20":
            weights = core_weight * base + (1.0 - core_weight) * signal
        elif strategy == "benchmark_x_signal":
            weights = signal
        elif strategy == "convex_signal_power2":
            weights = convex_signal
        elif strategy == "sector_neutral_signal":
            weights = sector_neutral_sleeve(base, scores, meta)
        else:  # pragma: no cover
            raise ValueError(f"Unknown strategy: {strategy}")

        weights = weights[weights > 0.0]
        weights = weights / weights.sum()
        target.loc[date, weights.index] = weights
        rows.append(
            {
                "date": date,
                "year": int(date.year),
                "strategy": strategy,
                "scored_members": len(scores),
                "active_names": int((weights > 0.0).sum()),
                "effective_names": 1.0 / float((weights**2).sum()),
                "max_weight": float(weights.max()),
                "top10_weight": float(weights.nlargest(10).sum()),
                "positive_forecast_share": float((scores > 0.0).mean()),
                "avg_forecast": float(scores.mean()),
                "avg_selected_forecast": float((weights * scores.reindex(weights.index)).sum()),
            }
        )

    weights = target.reindex(forecast.index).ffill().fillna(0.0)
    return weights, pd.DataFrame(rows)


def run_weighted_portfolio(price: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    returns = price.pct_change().mask(lambda frame: frame.abs() > 0.8).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * pit.COST_PER_DOLLAR_TRADED
    net = gross - costs
    equity = (1.0 + net).cumprod()
    return pd.DataFrame(
        {
            "gross_return": gross,
            "costs": costs,
            "net_return": net,
            "equity": equity,
            "drawdown": equity / equity.cummax() - 1.0,
            "turnover": turnover,
            "gross_exposure": weights.abs().sum(axis=1),
            "net_exposure": weights.sum(axis=1),
            "active_names": (weights > 0.0).sum(axis=1),
            "effective_names": 1.0 / (weights.pow(2).sum(axis=1).replace(0.0, np.nan)),
            "max_weight": weights.max(axis=1),
            "top10_weight": weights.apply(lambda row: row.nlargest(10).sum(), axis=1),
        },
        index=price.index,
    )


def trim_active(series: pd.Series, weights: pd.DataFrame) -> pd.Series:
    active = weights.abs().sum(axis=1)
    active = active[active > 0.0]
    if active.empty:
        return series.dropna()
    return series.loc[active.index[0] :].dropna()


def yearly_returns(streams: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(streams).dropna(how="all").groupby(pd.Grouper(freq="YE")).apply(
        lambda frame: (1.0 + frame).prod() - 1.0
    )


def crisis_table(streams: dict[str, pd.Series]) -> pd.DataFrame:
    windows = {
        "covid_2020": ("2020-02-19", "2020-03-23"),
        "inflation_2022": ("2022-01-03", "2022-10-12"),
    }
    benchmark_name = next(name for name in streams if name.startswith("Benchmark"))
    benchmark = streams[benchmark_name]
    rows = []
    for window, (start, end) in windows.items():
        for name, series in streams.items():
            period = series.loc[start:end].dropna()
            if period.empty:
                continue
            equity = (1.0 + period).cumprod()
            row = {
                "window": window,
                "strategy": name,
                "return": equity.iloc[-1] - 1.0,
                "vol": period.std() * math.sqrt(BUSINESS_DAYS),
                "mdd": (equity / equity.cummax() - 1.0).min(),
            }
            if name != benchmark_name:
                row["corr_to_benchmark"] = series.loc[start:end].corr(benchmark.loc[start:end])
            rows.append(row)
    return pd.DataFrame(rows)


def diagnostics_table(daily_by_strategy: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for strategy, daily in daily_by_strategy.items():
        active = daily[daily["gross_exposure"] > 0.0]
        rows.append(
            {
                "strategy": strategy,
                "avg_turnover_annual": active["turnover"].mean() * BUSINESS_DAYS,
                "avg_cost_annual": active["costs"].mean() * BUSINESS_DAYS,
                "avg_effective_names": active["effective_names"].mean(),
                "avg_active_names": active["active_names"].mean(),
                "avg_max_weight": active["max_weight"].mean(),
                "avg_top10_weight": active["top10_weight"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_universe(key: str, streams: dict[str, pd.Series], annual: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    frame = pd.DataFrame(streams).dropna(how="all")
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.2, 1.6], hspace=0.18)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[2, 0])
    for name in frame:
        series = frame[name].dropna()
        equity = (1.0 + series).cumprod()
        ax0.plot(equity.index, equity, label=name, linewidth=1.6)
        ax1.plot(equity.index, equity / equity.cummax() - 1.0, linewidth=1.0)
    annual_plot = annual.copy()
    annual_plot.index = annual_plot.index.year
    annual_plot = annual_plot.dropna(how="all")
    x = np.arange(len(annual_plot.index))
    width = 0.14
    for i, name in enumerate(annual_plot.columns):
        ax2.bar(x + (i - (len(annual_plot.columns) - 1) / 2) * width, annual_plot[name], width=width, label=name)
    ax2.axhline(0.0, color="#555555", linewidth=0.8)
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([str(year) for year in annual_plot.index[::2]], rotation=45, ha="right")
    ax0.set_title(f"{UNIVERSES[key]} Benchmark-Aware Momentum Tests")
    ax0.set_yscale("log")
    ax0.set_ylabel("Growth of $1")
    ax1.set_ylabel("Drawdown")
    ax2.set_ylabel("Year return")
    ax0.legend(loc="upper left", ncol=3)
    ax2.legend(loc="upper left", ncol=3, fontsize=8)
    ax1.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{key}_benchmark_aware_momentum.png", dpi=180)
    plt.close(fig)


def run_universe(key: str, start: str, end: str, core_weight: float, min_active: int) -> dict[str, pd.DataFrame]:
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    annual = load_annual(key, start, end)
    price = load_price(key, annual, end)
    forecast, rule_table = pit.build_forecasts_no_lookahead(price)
    benchmark = load_benchmark(key, start, end)

    strategies = ["core80_signal20", "benchmark_x_signal", "convex_signal_power2", "sector_neutral_signal"]
    streams: dict[str, pd.Series] = {}
    daily_by_strategy: dict[str, pd.DataFrame] = {}
    selections = []
    weights_sample = {}
    for strategy in strategies:
        weights, strategy_rows = build_weights(forecast, annual, strategy, core_weight=core_weight, min_active=min_active)
        daily = run_weighted_portfolio(price, weights)
        label = strategy
        streams[label] = trim_active(daily["net_return"].rename(label), weights).loc[start:end]
        daily_by_strategy[label] = daily
        strategy_rows.insert(0, "universe", UNIVERSES[key])
        selections.append(strategy_rows)
        weights_sample[strategy] = weights.iloc[::5]
        daily.to_csv(out_dir / f"{strategy}_daily.csv")
        weights.iloc[::5].to_csv(out_dir / f"{strategy}_weekly_weights.csv")

    legacy_path = SOURCE_OUT / key / "portfolio_daily_top40.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path, parse_dates=["Date"]).set_index("Date").sort_index()["net_return"]
        streams["legacy_top40_equal"] = legacy.loc[start:end].dropna()

    streams[benchmark.name] = benchmark
    common_start = max(series.index.min() for series in streams.values())
    common_end = min(series.index.max() for series in streams.values())
    streams = {name: series.loc[common_start:common_end] for name, series in streams.items()}
    aligned = pd.DataFrame(streams).dropna(how="any")
    streams = {name: aligned[name] for name in aligned.columns}

    stats = pd.DataFrame([{"strategy": name, **performance_stats(series)} for name, series in streams.items()])
    annual_returns = yearly_returns(streams)
    crisis = crisis_table(streams)
    diagnostics = diagnostics_table(daily_by_strategy)
    diagnostics.insert(0, "universe", UNIVERSES[key])
    rule_table.to_csv(out_dir / "rule_scalars.csv", index=False)
    stats.to_csv(out_dir / "stats.csv", index=False)
    annual_returns.to_csv(out_dir / "yearly_returns.csv")
    crisis.to_csv(out_dir / "crisis_windows.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(out_dir / "rebalance_diagnostics.csv", index=False)
    plot_universe(key, streams, annual_returns, out_dir)
    write_universe_summary(key, stats, diagnostics, crisis, out_dir, start, end, core_weight)
    return {
        "stats": stats.assign(universe=UNIVERSES[key]),
        "diagnostics": diagnostics,
        "crisis": crisis.assign(universe=UNIVERSES[key]),
        "annual_returns": annual_returns.assign(universe=UNIVERSES[key]),
    }


def write_universe_summary(
    key: str,
    stats: pd.DataFrame,
    diagnostics: pd.DataFrame,
    crisis: pd.DataFrame,
    out_dir: Path,
    start: str,
    end: str,
    core_weight: float,
) -> None:
    spec = next(spec for spec in pit.UNIVERSES if spec.key == key)
    benchmark = f"Benchmark: {spec.benchmark_label}"
    if key == "sp500":
        source_line = "- Source weights: archived State Street/SPDR SPY holdings xlsx snapshots from Wayback; sparse captures use the latest available snapshot known before each weekly rebalance."
    else:
        source_line = "- Source weights: archived point-in-time iShares holdings already cached locally."
    lines = [
        f"# {UNIVERSES[key]} Benchmark-Aware Momentum Tests",
        "",
        f"- Sample: {start} to {end}.",
        source_line,
        "- Rebalance: weekly.",
        f"- Cost assumption: {pit.COST_PER_DOLLAR_TRADED:.2%} of notional traded.",
        "- Bad free-data daily stock returns above 80% absolute are treated as missing before portfolio aggregation.",
        f"- `core80_signal20`: {core_weight:.0%} benchmark-weight core plus {1.0 - core_weight:.0%} benchmark-weighted positive-forecast sleeve.",
        "- `benchmark_x_signal`: full portfolio proportional to point-in-time ETF weight times positive forecast.",
        "- `convex_signal_power2`: full portfolio proportional to point-in-time ETF weight times squared positive forecast, so stronger trends receive disproportionately larger weights.",
        "- `sector_neutral_signal`: preserves point-in-time ETF sector weights and applies weight times positive forecast inside each sector.",
        "",
        "## Performance",
        "",
        "| Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in stats.iterrows():
        lines.append(
            f"| {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )
    lines += [
        "",
        "## Diagnostics",
        "",
        "| Strategy | Ann Turnover | Ann Cost | Effective Names | Active Names | Avg Max Weight | Avg Top10 Weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| {row['strategy']} | {pct(row['avg_turnover_annual'])} | {pct(row['avg_cost_annual'])} | {row['avg_effective_names']:.1f} | {row['avg_active_names']:.1f} | {pct(row['avg_max_weight'])} | {pct(row['avg_top10_weight'])} |"
        )
    lines += [
        "",
        "## Crisis Windows",
        "",
        "| Window | Strategy | Return | Vol | MDD | Corr To Benchmark |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in crisis.iterrows():
        corr = "" if pd.isna(row.get("corr_to_benchmark", np.nan)) else f"{row['corr_to_benchmark']:.2f}"
        lines.append(
            f"| {row['window']} | {row['strategy']} | {pct(row['return'])} | {pct(row['vol'])} | {pct(row['mdd'])} | {corr} |"
        )
    lines += [
        "",
        f"- Benchmark comparison row: `{benchmark}`.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_summary(all_stats: pd.DataFrame, all_diag: pd.DataFrame) -> None:
    lines = [
        "# Benchmark-Aware Stock Momentum Tests",
        "",
        "These tests use only point-in-time index weights available locally or from archived issuer holdings files.",
        "EEM/EFA use cached archived iShares holdings; SPY uses archived State Street/SPDR holdings snapshots when requested.",
        "QQQ is intentionally excluded because the current local Wikipedia constituent snapshots do not include historical index weights.",
        "",
        "## Performance",
        "",
        "| Universe | Strategy | Start | End | CAGR | Ann Ret | Vol | Sharpe | MDD | Total |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_stats.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {row['start']} | {row['end']} | {pct(row['cagr'])} | {pct(row['ann_return'])} | {pct(row['vol'])} | {num(row['sharpe'])} | {pct(row['mdd'])} | {pct(row['total_return'])} |"
        )
    lines += [
        "",
        "## Diagnostics",
        "",
        "| Universe | Strategy | Ann Turnover | Ann Cost | Effective Names | Active Names | Avg Max Weight | Avg Top10 Weight |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_diag.iterrows():
        lines.append(
            f"| {row['universe']} | {row['strategy']} | {pct(row['avg_turnover_annual'])} | {pct(row['avg_cost_annual'])} | {row['avg_effective_names']:.1f} | {row['avg_active_names']:.1f} | {pct(row['avg_max_weight'])} | {pct(row['avg_top10_weight'])} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", nargs="+", default=["eem", "efa"], choices=sorted(UNIVERSES))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--core-weight", type=float, default=0.80)
    parser.add_argument("--min-active", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = []
    all_diag = []
    all_crisis = []
    for key in args.universes:
        print(f"Running {UNIVERSES[key]}", flush=True)
        result = run_universe(key, args.start, args.end, args.core_weight, args.min_active)
        all_stats.append(result["stats"])
        all_diag.append(result["diagnostics"])
        all_crisis.append(result["crisis"])
    stats = pd.concat(all_stats, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True)
    crisis = pd.concat(all_crisis, ignore_index=True)
    stats.to_csv(OUT / "stats_all.csv", index=False)
    diag.to_csv(OUT / "diagnostics_all.csv", index=False)
    crisis.to_csv(OUT / "crisis_all.csv", index=False)
    write_combined_summary(stats, diag)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
