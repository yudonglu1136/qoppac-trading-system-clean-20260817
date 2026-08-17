#!/usr/bin/env python3
"""Fetch China A-share point-in-time universes and daily price data.

The metadata source is BaoStock's stock basic table, which includes IPO and
delisting dates. Daily OHLCV prices are fetched from Eastmoney's kline endpoint
and cached per symbol. The output mirrors the existing stock backtest data
layout: annual_constituents.csv, adj_close.csv, and benchmark_adj_close.csv.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import math
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import baostock as bs
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
CHINA_ROOT = ROOT / "data" / "china_a_shares"
PIT_ROOT = ROOT / "data" / "point_in_time_index_members"
PRICE_CACHE = CHINA_ROOT / "eastmoney_daily_qfq"
BENCHMARK_CACHE = CHINA_ROOT / "eastmoney_index_daily"
SUMMARY_PATH = CHINA_ROOT / "china_data_summary.md"

DEFAULT_START = "2000-01-01"
DEFAULT_END = date.today().isoformat()
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
USER_AGENT = "Mozilla/5.0 qoppac-china-a-share-data"

KLINE_FIELDS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "change",
    "turnover",
]


@dataclass(frozen=True)
class ChinaUniverse:
    key: str
    title: str
    benchmark_code: str
    benchmark_label: str
    selector: Callable[[pd.DataFrame], pd.Series]


def is_chinext(frame: pd.DataFrame) -> pd.Series:
    return frame["code"].str.match(r"^sz\.30[01]\d{3}$")


def is_sse(frame: pd.DataFrame) -> pd.Series:
    return frame["code"].str.match(r"^sh\.(60[0135]\d{3}|68[89]\d{3})$")


def is_szse(frame: pd.DataFrame) -> pd.Series:
    return frame["code"].str.match(r"^sz\.(00[0123]\d{3}|30[01]\d{3})$")


def is_szse_ex_chinext(frame: pd.DataFrame) -> pd.Series:
    return frame["code"].str.match(r"^sz\.00[0123]\d{3}$")


UNIVERSES: dict[str, ChinaUniverse] = {
    "cn_chinext": ChinaUniverse(
        key="cn_chinext",
        title="China ChiNext",
        benchmark_code="sz.399006",
        benchmark_label="ChiNext Index / 399006",
        selector=is_chinext,
    ),
    "cn_sse": ChinaUniverse(
        key="cn_sse",
        title="China Shanghai A Shares",
        benchmark_code="sh.000001",
        benchmark_label="SSE Composite / 000001",
        selector=is_sse,
    ),
    "cn_szse": ChinaUniverse(
        key="cn_szse",
        title="China Shenzhen A Shares",
        benchmark_code="sz.399001",
        benchmark_label="SZSE Component / 399001",
        selector=is_szse,
    ),
    "cn_szse_ex_chinext": ChinaUniverse(
        key="cn_szse_ex_chinext",
        title="China Shenzhen A Shares ex-ChiNext",
        benchmark_code="sz.399001",
        benchmark_label="SZSE Component / 399001",
        selector=is_szse_ex_chinext,
    ),
}


def baostock_login() -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {result.error_code} {result.error_msg}")


def fetch_stock_basic() -> pd.DataFrame:
    baostock_login()
    try:
        result = bs.query_stock_basic()
        rows: list[list[str]] = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        if result.error_code != "0":
            raise RuntimeError(f"query_stock_basic failed: {result.error_code} {result.error_msg}")
        return pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()


def load_or_fetch_stock_basic(refresh: bool) -> pd.DataFrame:
    CHINA_ROOT.mkdir(parents=True, exist_ok=True)
    path = CHINA_ROOT / "stock_basic_baostock.csv"
    if path.exists() and not refresh:
        basic = pd.read_csv(path, dtype=str)
    else:
        basic = fetch_stock_basic()
        basic.to_csv(path, index=False)
    return normalize_basic(basic)


def normalize_basic(raw: pd.DataFrame) -> pd.DataFrame:
    basic = raw.copy()
    for column in ["code", "code_name", "ipoDate", "outDate", "type", "status"]:
        if column not in basic.columns:
            raise ValueError(f"BaoStock stock basic missing {column}")
        basic[column] = basic[column].fillna("").astype(str)
    basic = basic[basic["type"].eq("1")].copy()
    basic["ipo_date"] = pd.to_datetime(basic["ipoDate"], errors="coerce")
    basic["out_date"] = pd.to_datetime(basic["outDate"].replace("", np.nan), errors="coerce")
    basic = basic.dropna(subset=["ipo_date"])
    basic = basic[is_sse(basic) | is_szse(basic)].copy()
    basic["symbol"] = basic["code"]
    basic["name"] = basic["code_name"]
    basic["exchange"] = np.where(basic["code"].str.startswith("sh."), "SSE", "SZSE")
    basic["board"] = "Unknown"
    basic.loc[is_sse(basic) & basic["code"].str.match(r"^sh\.68[89]"), "board"] = "STAR"
    basic.loc[is_sse(basic) & ~basic["code"].str.match(r"^sh\.68[89]"), "board"] = "SSE Main"
    basic.loc[is_chinext(basic), "board"] = "ChiNext"
    basic.loc[is_szse_ex_chinext(basic), "board"] = "SZSE Main"
    return basic.sort_values("symbol").reset_index(drop=True)


def annual_constituents_for_universe(
    basic: pd.DataFrame,
    universe: ChinaUniverse,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    source = basic[universe.selector(basic)].copy()
    rows: list[pd.DataFrame] = []
    for year in range(start_ts.year, end_ts.year + 1):
        snapshot = pd.Timestamp(year=year, month=1, day=1)
        if snapshot > end_ts:
            continue
        active = source[
            source["ipo_date"].le(snapshot)
            & (source["out_date"].isna() | source["out_date"].ge(snapshot))
        ].copy()
        if active.empty:
            continue
        active["year"] = year
        active["snapshot_date"] = snapshot.date().isoformat()
        active["weight"] = 1.0
        active["sector"] = active["board"]
        active["asset_class"] = "Equity"
        active["source"] = "BaoStock query_stock_basic IPO/outDate listing filter"
        rows.append(
            active[
                [
                    "symbol",
                    "name",
                    "code",
                    "weight",
                    "sector",
                    "asset_class",
                    "exchange",
                    "board",
                    "year",
                    "snapshot_date",
                    "ipoDate",
                    "outDate",
                    "status",
                    "source",
                ]
            ].rename(columns={"code": "symbol_raw"})
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["snapshot_date", "symbol"])


def eastmoney_secid(symbol: str) -> str:
    exchange, code = symbol.split(".")
    if exchange == "sh":
        return f"1.{code}"
    if exchange == "sz":
        return f"0.{code}"
    raise ValueError(symbol)


def eastmoney_kline_params(symbol: str, start: str, end: str, adjusted: bool) -> dict[str, str]:
    return {
        "secid": eastmoney_secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1" if adjusted else "0",
        "beg": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
    }


def parse_eastmoney_klines(symbol: str, payload: dict) -> pd.DataFrame:
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame(columns=["symbol", *KLINE_FIELDS])
    rows = [line.split(",") for line in klines]
    frame = pd.DataFrame(rows, columns=KLINE_FIELDS)
    frame.insert(0, "symbol", symbol)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in KLINE_FIELDS:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    return frame.sort_values("date")


def request_eastmoney_kline(symbol: str, start: str, end: str, adjusted: bool, retries: int = 4) -> pd.DataFrame:
    params = eastmoney_kline_params(symbol, start, end, adjusted)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                EASTMONEY_KLINE_URL,
                params=params,
                headers={"User-Agent": USER_AGENT, "Referer": "https://quote.eastmoney.com/"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("rc") not in {0, None}:
                raise RuntimeError(f"Eastmoney rc={payload.get('rc')}")
            return parse_eastmoney_klines(symbol, payload)
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{symbol} failed after {retries} retries: {last_error}")


def safe_symbol_filename(symbol: str) -> str:
    return symbol.replace(".", "_")


def price_cache_path(symbol: str) -> Path:
    return PRICE_CACHE / f"{safe_symbol_filename(symbol)}.parquet"


def benchmark_cache_path(symbol: str) -> Path:
    return BENCHMARK_CACHE / f"{safe_symbol_filename(symbol)}.parquet"


def fetch_one_price(symbol: str, start: str, end: str, force: bool) -> dict[str, str | int | float]:
    path = price_cache_path(symbol)
    if path.exists() and not force:
        try:
            frame = pd.read_parquet(path, columns=["date", "close"])
            if not frame.empty:
                return {
                    "symbol": symbol,
                    "status": "cached",
                    "rows": int(len(frame)),
                    "start": str(pd.to_datetime(frame["date"]).min().date()),
                    "end": str(pd.to_datetime(frame["date"]).max().date()),
                    "path": str(path),
                }
        except Exception:
            pass
    try:
        frame = request_eastmoney_kline(symbol, start, end, adjusted=True)
        if frame.empty:
            return {"symbol": symbol, "status": "empty", "rows": 0, "start": "", "end": "", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return {
            "symbol": symbol,
            "status": "downloaded",
            "rows": int(len(frame)),
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "path": str(path),
        }
    except Exception as exc:
        return {"symbol": symbol, "status": f"error: {exc}", "rows": 0, "start": "", "end": "", "path": str(path)}


def download_prices(symbols: list[str], start: str, end: str, force: bool, workers: int) -> pd.DataFrame:
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, str | int | float]] = []
    total = len(symbols)
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(fetch_one_price, symbol, start, end, force): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(futures.as_completed(future_map), start=1):
            row = future.result()
            audit_rows.append(row)
            if completed % 100 == 0 or completed == total:
                ok = sum(str(item["status"]) in {"cached", "downloaded"} for item in audit_rows)
                empty = sum(str(item["status"]) == "empty" for item in audit_rows)
                errors = completed - ok - empty
                print(f"prices {completed}/{total}: ok={ok} empty={empty} errors={errors}", flush=True)
    audit = pd.DataFrame(audit_rows).sort_values("symbol")
    audit.to_csv(CHINA_ROOT / "price_download_audit.csv", index=False)
    return audit


def fetch_benchmark(symbol: str, start: str, end: str, force: bool) -> pd.DataFrame:
    path = benchmark_cache_path(symbol)
    if path.exists() and not force:
        return pd.read_parquet(path)
    frame = request_eastmoney_kline(symbol, start, end, adjusted=False)
    if frame.empty:
        raise RuntimeError(f"No benchmark data for {symbol}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def read_close_series(symbol: str) -> pd.Series | None:
    path = price_cache_path(symbol)
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, columns=["date", "close"])
    except Exception:
        return None
    if frame.empty:
        return None
    series = frame.dropna(subset=["date", "close"]).set_index("date")["close"].sort_index()
    if series.empty:
        return None
    series.name = symbol
    return series


def write_wide_adj_close(symbols: list[str], path: Path) -> pd.DataFrame:
    pieces = []
    for symbol in symbols:
        series = read_close_series(symbol)
        if series is not None:
            pieces.append(series)
    if not pieces:
        raise RuntimeError(f"No close series available for {path}")
    wide = pd.concat(pieces, axis=1).sort_index()
    wide = wide.loc[:, ~wide.columns.duplicated()]
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.reset_index(names="Date").to_csv(path, index=False)
    return wide


def write_benchmark_close(symbol: str, column_name: str, path: Path, start: str, end: str, force: bool) -> pd.DataFrame:
    frame = fetch_benchmark(symbol, start, end, force)
    series = frame.dropna(subset=["date", "close"]).set_index("date")["close"].sort_index()
    out = series.rename(column_name).to_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index(names="Date").to_csv(path, index=False)
    return out


def write_universe_outputs(
    basic: pd.DataFrame,
    universe: ChinaUniverse,
    start: str,
    end: str,
    force_benchmark: bool,
) -> dict[str, str | int | float]:
    out_dir = PIT_ROOT / universe.key
    annual = annual_constituents_for_universe(basic, universe, start, end)
    if annual.empty:
        raise RuntimeError(f"No annual constituents for {universe.key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    annual.to_csv(out_dir / "annual_constituents.csv", index=False)

    symbols = sorted(annual["symbol"].dropna().unique())
    price = write_wide_adj_close(symbols, out_dir / "adj_close.csv")
    benchmark = write_benchmark_close(
        universe.benchmark_code,
        universe.benchmark_code,
        out_dir / "benchmark_adj_close.csv",
        start,
        end,
        force_benchmark,
    )
    audit = pd.DataFrame(
        {
            "symbol": price.columns,
            "rows": price.notna().sum().to_numpy(),
            "first_date": [str(price[column].first_valid_index().date()) if price[column].first_valid_index() else "" for column in price.columns],
            "last_date": [str(price[column].last_valid_index().date()) if price[column].last_valid_index() else "" for column in price.columns],
        }
    )
    audit.to_csv(out_dir / "price_coverage_audit.csv", index=False)
    snapshot_counts = annual.groupby("year")["symbol"].nunique().rename("constituents").reset_index()
    snapshot_counts.to_csv(out_dir / "annual_snapshot_audit.csv", index=False)
    return {
        "universe": universe.key,
        "title": universe.title,
        "annual_rows": int(len(annual)),
        "unique_symbols": int(len(symbols)),
        "price_columns": int(price.shape[1]),
        "price_rows": int(price.shape[0]),
        "price_start": str(price.index.min().date()),
        "price_end": str(price.index.max().date()),
        "benchmark_rows": int(len(benchmark)),
        "benchmark_start": str(benchmark.index.min().date()),
        "benchmark_end": str(benchmark.index.max().date()),
        "out_dir": str(out_dir),
    }


def write_summary(rows: list[dict[str, str | int | float]], audit: pd.DataFrame, start: str, end: str) -> None:
    ok = audit["status"].astype(str).isin(["cached", "downloaded"])
    empty = audit["status"].astype(str).eq("empty")
    lines = [
        "# China A-share Data Fetch",
        "",
        f"- Requested sample: {start} to {end}.",
        "- Metadata: BaoStock `query_stock_basic`, filtered by IPO date and delisting date.",
        "- Prices: Eastmoney daily kline endpoint, forward-adjusted (`fqt=1`) OHLCV cached per symbol.",
        "- Universe snapshots: annual Jan 1 listing membership, not current-constituent backfill.",
        "",
        "## Download Audit",
        "",
        f"- Symbols requested: {len(audit)}",
        f"- Cached/downloaded with price rows: {int(ok.sum())}",
        f"- Empty responses: {int(empty.sum())}",
        f"- Errors: {int((~ok & ~empty).sum())}",
        "",
        "## Universe Outputs",
        "",
        "| Universe | Unique Symbols | Price Columns | Price Rows | Price Start | Price End | Benchmark Rows | Output |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['title']} | {row['unique_symbols']} | {row['price_columns']} | {row['price_rows']} | {row['price_start']} | {row['price_end']} | {row['benchmark_rows']} | `{row['out_dir']}` |"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--universes", nargs="+", default=["cn_chinext", "cn_sse", "cn_szse"], choices=sorted(UNIVERSES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh-basic", action="store_true")
    parser.add_argument("--force-price", action="store_true")
    parser.add_argument("--force-benchmark", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--limit-symbols", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CHINA_ROOT.mkdir(parents=True, exist_ok=True)
    PIT_ROOT.mkdir(parents=True, exist_ok=True)
    basic = load_or_fetch_stock_basic(args.refresh_basic)
    basic.to_csv(CHINA_ROOT / "stock_basic_normalized.csv", index=False)

    selected_universes = [UNIVERSES[key] for key in args.universes]
    annual_by_key = {
        universe.key: annual_constituents_for_universe(basic, universe, args.start, args.end)
        for universe in selected_universes
    }
    requested_symbols = sorted(
        {
            symbol
            for annual in annual_by_key.values()
            for symbol in annual["symbol"].dropna().unique()
        }
    )
    if args.limit_symbols:
        requested_symbols = requested_symbols[: args.limit_symbols]
    print(f"selected universes={args.universes} requested_symbols={len(requested_symbols)}", flush=True)

    if args.metadata_only:
        audit = pd.DataFrame({"symbol": requested_symbols, "status": "metadata_only", "rows": 0})
    else:
        audit = download_prices(
            requested_symbols,
            args.start,
            args.end,
            force=args.force_price,
            workers=max(1, args.workers),
        )

    rows = []
    if not args.metadata_only:
        for universe in selected_universes:
            rows.append(write_universe_outputs(basic, universe, args.start, args.end, args.force_benchmark))
        write_summary(rows, audit, args.start, args.end)
        print(SUMMARY_PATH)
    else:
        for key, annual in annual_by_key.items():
            out_dir = PIT_ROOT / key
            out_dir.mkdir(parents=True, exist_ok=True)
            annual.to_csv(out_dir / "annual_constituents.csv", index=False)
        print("metadata only complete")


if __name__ == "__main__":
    main()
