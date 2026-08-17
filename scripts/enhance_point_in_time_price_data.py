#!/usr/bin/env python3
"""Improve cached point-in-time stock price data from 2016 onward.

The repair is intentionally conservative:

- Remove non-trading class markers such as the Mexican "*" marker.
- Fill original point-in-time ticker columns from verified Yahoo aliases for
  same-company ticker changes.
- Retry low-coverage tickers from 2016 onward with yfinance.

It does not replace delisted companies with acquirers or today's index members.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

import run_point_in_time_annual_ranked_long_only as pit


START = "2016-01-01"
MIN_OBS = 260


def backup_once(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".pre_enhance")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def clean_cached_symbol(symbol: object) -> object:
    if pd.isna(symbol):
        return symbol
    text = str(symbol).strip()
    if "*" in text:
        text = text.replace("*", "")
    return text


def clean_symbol_file(path: Path) -> int:
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    if "symbol" not in frame.columns:
        return 0
    old = frame["symbol"].astype(str)
    frame["symbol"] = frame["symbol"].map(clean_cached_symbol)
    changed = int((old != frame["symbol"].astype(str)).sum())
    if changed:
        backup_once(path)
        frame.to_csv(path, index=False)
    return changed


def download_close(request_symbols: list[str], chunk_size: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    request_symbols = sorted(set(request_symbols))
    for start in range(0, len(request_symbols), chunk_size):
        chunk = request_symbols[start : start + chunk_size]
        print(f"download {start + 1}-{start + len(chunk)} / {len(request_symbols)}", flush=True)
        data = yf.download(
            chunk,
            start=START,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=30,
        )
        close = pit.close_from_download(data, chunk)
        if not close.empty:
            chunks.append(close)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, axis=1).sort_index()
    out = out.loc[:, ~out.columns.duplicated()].dropna(how="all")
    out.index.name = "Date"
    return out


def enhance_universe(key: str, chunk_size: int) -> pd.DataFrame:
    data_dir = pit.DATA_ROOT / key
    annual_path = data_dir / "annual_constituents.csv"
    price_path = data_dir / "adj_close.csv"
    if not annual_path.exists() or not price_path.exists():
        raise FileNotFoundError(f"Missing annual constituents or prices for {key}")

    changed_annual = clean_symbol_file(annual_path)
    changed_snapshots = clean_symbol_file(data_dir / "holding_snapshots.csv")
    annual = pd.read_csv(annual_path)
    annual = annual[(annual["year"] >= 2016) & (annual["year"] <= 2026)].copy()
    targets = sorted(annual["symbol"].dropna().astype(str).unique())

    price = pd.read_csv(price_path, parse_dates=["Date"]).set_index("Date").sort_index()
    backup_once(price_path)
    price = price.rename(columns={column: clean_cached_symbol(column) for column in price.columns})
    price = price.loc[:, ~price.columns.duplicated()]
    price16 = price.loc[START:]
    current_obs = {symbol: int(price16[symbol].notna().sum()) if symbol in price16 else 0 for symbol in targets}

    request_map: dict[str, str] = {}
    for symbol in targets:
        if symbol in pit.PRICE_ALIASES:
            request_map[symbol] = pit.PRICE_ALIASES[symbol]
        elif current_obs.get(symbol, 0) < MIN_OBS:
            request_map[symbol] = symbol

    if not request_map:
        return pd.DataFrame()

    downloaded = download_close(list(request_map.values()), chunk_size)
    rows = []
    for symbol, request_symbol in sorted(request_map.items()):
        old_obs = current_obs.get(symbol, 0)
        if request_symbol not in downloaded:
            rows.append(
                {
                    "universe": key,
                    "symbol": symbol,
                    "download_symbol": request_symbol,
                    "old_obs_since_2016": old_obs,
                    "new_obs_since_2016": old_obs,
                    "action": "not_downloaded",
                }
            )
            continue
        candidate = downloaded[request_symbol].dropna()
        new_obs = int(candidate.loc[START:].notna().sum())
        if new_obs <= old_obs:
            rows.append(
                {
                    "universe": key,
                    "symbol": symbol,
                    "download_symbol": request_symbol,
                    "old_obs_since_2016": old_obs,
                    "new_obs_since_2016": old_obs,
                    "action": "kept_existing",
                }
            )
            continue

        aligned = candidate.reindex(price.index.union(candidate.index)).sort_index()
        price = price.reindex(aligned.index)
        price[symbol] = aligned
        rows.append(
            {
                "universe": key,
                "symbol": symbol,
                "download_symbol": request_symbol,
                "old_obs_since_2016": old_obs,
                "new_obs_since_2016": new_obs,
                "action": "replaced_with_alias" if request_symbol != symbol else "refreshed_symbol",
            }
        )

    price.index.name = "Date"
    price.to_csv(price_path)
    audit = pd.DataFrame(rows)
    audit["annual_symbol_rows_cleaned"] = changed_annual
    audit["snapshot_symbol_rows_cleaned"] = changed_snapshots
    audit.to_csv(data_dir / "data_enhancement_audit.csv", index=False)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", nargs="+", default=["sp500", "eem", "efa"])
    parser.add_argument("--chunk-size", type=int, default=80)
    args = parser.parse_args()

    all_rows = []
    for key in args.universes:
        print(f"\nEnhancing {key}", flush=True)
        audit = enhance_universe(key, args.chunk_size)
        if not audit.empty:
            print(audit["action"].value_counts().to_string(), flush=True)
            all_rows.append(audit)

    if all_rows:
        out = pit.DATA_ROOT / f"data_enhancement_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
        print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
