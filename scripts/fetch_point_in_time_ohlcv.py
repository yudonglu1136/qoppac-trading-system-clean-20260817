#!/usr/bin/env python3
"""Fetch yfinance OHLCV panels for existing point-in-time stock universes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402


FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def field_from_download(data: pd.DataFrame, request_tickers: list[str], field: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        if len(request_tickers) == 1 and field in data.columns:
            return data[[field]].rename(columns={field: request_tickers[0]})
        return pd.DataFrame()

    values: dict[str, pd.Series] = {}
    for ticker in request_tickers:
        if (ticker, field) in data.columns:
            values[ticker] = data[(ticker, field)]
        elif (field, ticker) in data.columns:
            values[ticker] = data[(field, ticker)]
    return pd.DataFrame(values)


def load_symbols(key: str, limit: int | None = None) -> list[str]:
    annual = pd.read_csv(pit.DATA_ROOT / key / "annual_constituents.csv")
    symbols = sorted(annual["symbol"].dropna().astype(str).unique())
    return symbols[:limit] if limit else symbols


def read_field_panels(out_dir: Path) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for field_name, file_name in FIELDS.items():
        path = out_dir / f"{file_name}.csv"
        if path.exists():
            panels[field_name] = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            panels[field_name] = pd.DataFrame()
    return panels


def concat_field_frames(frames_by_field: dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for field_name, frames in frames_by_field.items():
        if not frames:
            panels[field_name] = pd.DataFrame()
            continue
        panel = pd.concat(frames, axis=1, sort=True).sort_index()
        panels[field_name] = panel.loc[:, ~panel.columns.duplicated()].dropna(how="all")
    return panels


def write_field_panels(panels: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for field_name, panel in panels.items():
        if panel.empty:
            continue
        panel = panel.loc[:, ~panel.columns.duplicated()].sort_index()
        panel.index.name = "Date"
        panel.to_csv(out_dir / f"{FIELDS[field_name]}.csv")


def symbols_with_close_data(panels: dict[str, pd.DataFrame]) -> set[str]:
    close = panels.get("Close", pd.DataFrame())
    if close.empty:
        return set()
    return {str(column) for column in close.columns if close[column].notna().any()}


def retry_empty_symbols(
    key: str,
    symbols: list[str],
    panels: dict[str, pd.DataFrame],
    *,
    start: str,
    end: str | None,
    sleep: float,
) -> pd.DataFrame:
    available = symbols_with_close_data(panels)
    missing_symbols = [symbol for symbol in symbols if symbol not in available]
    if not missing_symbols:
        return pd.DataFrame()

    audit_rows = []
    for idx, symbol in enumerate(missing_symbols, start=1):
        request_ticker = pit.PRICE_ALIASES.get(symbol, symbol)
        print(f"{key}: retrying empty OHLCV {idx}-{len(missing_symbols)}: {symbol}", flush=True)
        try:
            data = yf.download(
                request_ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - network edge
            audit_rows.append(
                {
                    "kind": "retry",
                    "symbol": symbol,
                    "request_ticker": request_ticker,
                    "requested": 1,
                    "downloaded": 0,
                    "status": f"download_error: {exc}",
                    "rows": 0,
                }
            )
            time.sleep(sleep)
            continue

        close = field_from_download(data, [request_ticker], "Close")
        has_close = request_ticker in close and close[request_ticker].notna().any()
        rows = int(close[request_ticker].notna().sum()) if has_close else 0
        if has_close:
            for field_name in FIELDS:
                downloaded = field_from_download(data, [request_ticker], field_name)
                if request_ticker not in downloaded:
                    continue
                series = downloaded[request_ticker]
                if panels[field_name].empty:
                    panels[field_name] = pd.DataFrame(index=series.index)
                panels[field_name][symbol] = series

        audit_rows.append(
            {
                "kind": "retry",
                "symbol": symbol,
                "request_ticker": request_ticker,
                "requested": 1,
                "downloaded": int(has_close),
                "status": "ok" if has_close else "empty",
                "rows": rows,
            }
        )
        time.sleep(sleep)
    return pd.DataFrame(audit_rows)


def append_audit(audit_path: Path, audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.read_csv(audit_path) if audit_path.exists() else audit
    if audit_path.exists():
        existing = pd.read_csv(audit_path)
        audit = pd.concat([existing, audit], ignore_index=True, sort=False)
    audit.to_csv(audit_path, index=False)
    return audit


def download_universe_ohlcv(
    key: str,
    *,
    start: str,
    end: str | None,
    refresh: bool,
    retry_empty: bool,
    retry_empty_only: bool,
    chunk_size: int,
    limit: int | None,
    sleep: float,
) -> pd.DataFrame:
    data_dir = pit.DATA_ROOT / key
    out_dir = data_dir / "ohlcv"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "download_audit.csv"
    field_paths = {field_name: out_dir / f"{file_name}.csv" for field_name, file_name in FIELDS.items()}
    symbols = load_symbols(key, limit)
    if retry_empty_only:
        panels = read_field_panels(out_dir)
        retry_audit = retry_empty_symbols(key, symbols, panels, start=start, end=end, sleep=sleep) if retry_empty else pd.DataFrame()
        write_field_panels(panels, out_dir)
        return append_audit(audit_path, retry_audit)

    if all(path.exists() for path in field_paths.values()) and audit_path.exists() and not refresh:
        if retry_empty:
            panels = read_field_panels(out_dir)
            retry_audit = retry_empty_symbols(key, symbols, panels, start=start, end=end, sleep=sleep)
            write_field_panels(panels, out_dir)
            return append_audit(audit_path, retry_audit)
        return pd.read_csv(audit_path)

    panels = {field_name: [] for field_name in FIELDS}
    audit_rows = []
    for start_idx in range(0, len(symbols), chunk_size):
        chunk = symbols[start_idx : start_idx + chunk_size]
        request_map = {symbol: pit.PRICE_ALIASES.get(symbol, symbol) for symbol in chunk}
        request_tickers = sorted(set(request_map.values()))
        print(f"{key}: downloading OHLCV {start_idx + 1}-{start_idx + len(chunk)} / {len(symbols)}", flush=True)
        try:
            data = yf.download(
                request_tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - network edge
            audit_rows.append(
                {
                    "kind": "chunk",
                    "chunk_start": start_idx + 1,
                    "chunk_end": start_idx + len(chunk),
                    "requested": len(chunk),
                    "downloaded": 0,
                    "missing": ",".join(chunk),
                    "status": f"download_error: {exc}",
                    "rows": 0,
                }
            )
            time.sleep(sleep)
            continue

        downloaded_by_field = {field_name: field_from_download(data, request_tickers, field_name) for field_name in FIELDS}
        chunk_rows = 0
        missing = set(chunk)
        close_downloaded = downloaded_by_field["Close"]
        for field_name, downloaded in downloaded_by_field.items():
            if downloaded.empty:
                continue
            panel = pd.DataFrame(index=downloaded.index)
            for symbol, request_ticker in request_map.items():
                if request_ticker in downloaded:
                    panel[symbol] = downloaded[request_ticker]
                    if (
                        request_ticker in close_downloaded
                        and close_downloaded[request_ticker].notna().any()
                    ):
                        missing.discard(symbol)
            if not panel.empty:
                panels[field_name].append(panel)
                chunk_rows = max(chunk_rows, len(panel))
        audit_rows.append(
            {
                "kind": "chunk",
                "chunk_start": start_idx + 1,
                "chunk_end": start_idx + len(chunk),
                "requested": len(chunk),
                "downloaded": len(chunk) - len(missing),
                "missing": ",".join(sorted(missing)),
                "status": "ok" if len(missing) < len(chunk) else "empty",
                "rows": chunk_rows,
            }
        )
        time.sleep(sleep)

    field_panels = concat_field_frames(panels)
    if retry_empty:
        retry_audit = retry_empty_symbols(key, symbols, field_panels, start=start, end=end, sleep=sleep)
        if not retry_audit.empty:
            audit_rows.extend(retry_audit.to_dict("records"))
    write_field_panels(field_panels, out_dir)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False)
    return audit


def count_usable_symbols(key: str, limit: int | None = None) -> tuple[int, int]:
    symbols = load_symbols(key, limit)
    close_path = pit.DATA_ROOT / key / "ohlcv" / "close.csv"
    if not close_path.exists():
        return len(symbols), 0
    close = pd.read_csv(close_path, index_col=0)
    available = {str(column) for column in close.columns if close[column].notna().any()}
    return len(symbols), sum(1 for symbol in symbols if symbol in available)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universes", nargs="+", default=["sp500", "eem", "efa"], choices=["sp500", "eem", "efa"])
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per universe.")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--retry-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-empty-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for key in args.universes:
        audit = download_universe_ohlcv(
            key,
            start=args.start,
            end=args.end,
            refresh=args.refresh,
            retry_empty=args.retry_empty,
            retry_empty_only=args.retry_empty_only,
            chunk_size=args.chunk_size,
            limit=args.limit,
            sleep=args.sleep,
        )
        requested, usable = count_usable_symbols(key, args.limit)
        summaries.append(
            {
                "universe": key,
                "chunks": len(audit),
                "requested": requested,
                "downloaded": usable,
                "empty": requested - usable,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(pit.DATA_ROOT / "ohlcv_download_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
