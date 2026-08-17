#!/usr/bin/env python3
"""Build a non-destructive live-data overlay for the no-equity futures system.

The source pysystemtrade CSVs currently stop around 2024-03.  This script
leaves those files untouched and writes an overlay directory with:

- adjusted_prices_csv: original adjusted prices plus recent continuous futures
  proxy closes where Yahoo has coverage.
- multiple_prices_csv: original multiple prices plus PRICE=CARRY=FORWARD in the
  appended proxy segment so carry forecasts become unavailable instead of using
  stale 2024 carry.
- fx_prices_csv: original FX plus Yahoo FX updates.

This is a pragmatic paper-trading data bridge, not a replacement for a proper
continuous futures database with full contract rolls and term structure.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from ib_insync import Contract, IB

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_rob_style_backtest as bt  # noqa: E402
import run_rob_style_no_equity_40_backtest as no40  # noqa: E402


DEFAULT_CONTRACTS = ROOT / "output" / "ibkr_contract_qualification" / "all_40_contracts.csv"
DEFAULT_OUT = ROOT / "output" / "live_futures_overlay"
DOWNLOAD_START = "2024-01-01"

YAHOO_FUTURES = {
    "US2": "ZT=F",
    "US5": "ZF=F",
    "US10": "ZN=F",
    "SOFR": "SR3=F",
    "EUR_micro": "M6E=F",
    "AUD": "6A=F",
    "CAD": "6C=F",
    "GBP": "6B=F",
    "CNH": "CNH=F",
    "MXP": "6M=F",
    "JPY": "6J=F",
    "NZD": "6N=F",
    "LEANHOG": "HE=F",
    "LIVECOW": "LE=F",
    "CORN": "ZC=F",
    # Yahoo does not expose XK mini soybean reliably; full-size soybeans are the
    # closest same-underlying price proxy and share the same quote units.
    "SOYBEAN_mini": "ZS=F",
    "SOYMEAL": "ZM=F",
    "WHEAT": "ZW=F",
    "SOYOIL": "ZL=F",
    "FEEDCOW": "GF=F",
    "GOLD_micro": "MGC=F",
    # Micro copper uses the same quote unit as full-size HG.
    "COPPER-micro": "HG=F",
    "SILVER": "SI=F",
    "PLAT": "PL=F",
    "CRUDE_W": "CL=F",
    # QG mini natural gas is proxied by full-size NG because Yahoo has no stable
    # QG continuous ticker.
    "GAS_US_mini": "NG=F",
    "HEATOIL": "HO=F",
    "GASOILINE": "RB=F",
}

FX_TICKERS = {
    "EUR": ("EURUSD=X", False),
    "JPY": ("JPY=X", True),
    "KRW": ("KRW=X", True),
    "CNH": ("CNH=X", True),
}


@dataclass(frozen=True)
class Config:
    contracts: Path
    output_dir: Path
    refresh: bool
    max_age_days: int
    max_gap_days: int
    min_download_rows: int
    ibkr_current_contract_fallback: bool
    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int
    min_ibkr_rows: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Build live futures data overlay without changing source CSVs")
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-age-days", type=int, default=5)
    parser.add_argument("--max-gap-days", type=int, default=10)
    parser.add_argument("--min-download-rows", type=int, default=200)
    parser.add_argument(
        "--ibkr-current-contract-fallback",
        action="store_true",
        help="Use current IBKR contract daily history when Yahoo has no usable continuous proxy.",
    )
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=4002)
    parser.add_argument("--ibkr-client-id", type=int, default=121)
    parser.add_argument("--min-ibkr-rows", type=int, default=60)
    args = parser.parse_args()
    return Config(
        args.contracts,
        args.output_dir,
        args.refresh,
        args.max_age_days,
        args.max_gap_days,
        args.min_download_rows,
        args.ibkr_current_contract_fallback,
        args.ibkr_host,
        args.ibkr_port,
        args.ibkr_client_id,
        args.min_ibkr_rows,
    )


def latest_allowed(max_age_days: int) -> pd.Timestamp:
    return pd.Timestamp(date.today() - timedelta(days=max_age_days))


def normalize_yahoo_download(data: pd.DataFrame, ticker: str, field: str = "Close") -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(data.columns, pd.MultiIndex):
        if (field, ticker) in data.columns:
            series = data[(field, ticker)]
        else:
            matches = data.xs(field, axis=1, level=0, drop_level=False)
            series = matches.iloc[:, 0]
    else:
        series = data[field]
    index = pd.to_datetime(series.index).tz_localize(None).normalize()
    return pd.Series(pd.to_numeric(series.values, errors="coerce"), index=index, name=ticker).dropna()


def download_close(ticker: str, cache_path: Path, refresh: bool) -> pd.Series:
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        return cached.set_index("Date")["Close"].sort_index().dropna()
    try:
        data = yf.download(
            ticker,
            start=DOWNLOAD_START,
            end=(date.today() + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            timeout=30,
            threads=False,
        )
    except Exception as exc:
        if not cache_path.exists():
            raise
        print(f"YAHOO_REFRESH_FALLBACK ticker={ticker} error={exc!r}")
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        return cached.set_index("Date")["Close"].sort_index().dropna()
    close = normalize_yahoo_download(data, ticker, "Close")
    if close.empty and cache_path.exists():
        print(f"YAHOO_REFRESH_FALLBACK ticker={ticker} error=empty_download")
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        return cached.set_index("Date")["Close"].sort_index().dropna()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = close.rename("Close").reset_index()
    cache.columns = ["Date", "Close"]
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache.to_csv(tmp, index=False)
    tmp.replace(cache_path)
    return close


def max_calendar_gap_days(series: pd.Series) -> float:
    if len(series) < 2:
        return np.inf
    return float(series.index.to_series().sort_values().diff().dt.days.max())


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    result = str(value).strip()
    if result.endswith(".0") and result[:-2].isdigit():
        return result[:-2]
    return result


def contract_from_qualification(row: pd.Series) -> Contract:
    return Contract(
        secType="FUT",
        conId=int(float(row["con_id"])),
        exchange=text(row["exchange"]),
        currency=text(row["currency"]),
        localSymbol=text(row["local_symbol"]),
        tradingClass=text(row["trading_class"]),
    )


def download_ibkr_current_contract_close(
    ib: IB,
    instrument: str,
    row: pd.Series,
    output_dir: Path,
) -> pd.Series:
    contract = contract_from_qualification(row)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        keepUpToDate=False,
    )
    records = []
    for bar in bars:
        records.append({"Date": pd.Timestamp(bar.date).normalize(), "Close": float(bar.close)})
    out = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol = text(row["local_symbol"]).replace(" ", "_")
    path = output_dir / f"{instrument}_{symbol}.csv"
    out.to_csv(path, index=False)
    if out.empty:
        return pd.Series(dtype=float, name=instrument)
    return out.set_index("Date")["Close"].sort_index().dropna().rename(instrument)


def load_qualified_contracts(path: Path) -> pd.DataFrame:
    contracts = pd.read_csv(path).fillna("")
    contracts["is_qualified"] = contracts["status"].astype(str).str.startswith("qualified")
    return contracts.set_index("instrument", drop=False)


def source_daily_adjusted(instrument: str) -> pd.Series:
    source = pd.read_csv(bt.ADJUSTED / f"{instrument}.csv", parse_dates=["DATETIME"])
    series = source.set_index("DATETIME")["price"].sort_index()
    daily = series.resample("1D").last().dropna()
    daily.index = daily.index.normalize()
    return daily


def append_adjusted_prices(instrument: str, yahoo_close: pd.Series, output_path: Path) -> tuple[int, str, float]:
    source_path = bt.ADJUSTED / f"{instrument}.csv"
    source = pd.read_csv(source_path, parse_dates=["DATETIME"]).sort_values("DATETIME")
    source_daily = source_daily_adjusted(instrument)
    last_source_day = source_daily.index.max()
    overlap = pd.concat([source_daily.rename("source"), yahoo_close.rename("yahoo")], axis=1).dropna()
    recent_overlap = overlap.loc[overlap.index >= pd.Timestamp(DOWNLOAD_START)]

    if len(recent_overlap) >= 5:
        offset = float((recent_overlap["source"] - recent_overlap["yahoo"]).tail(20).median())
        anchor_status = "median_overlap_offset"
    else:
        future = yahoo_close.loc[yahoo_close.index > last_source_day]
        if future.empty:
            shutil.copy2(source_path, output_path)
            return 0, "no_new_yahoo_rows", np.nan
        offset = float(source_daily.iloc[-1] - future.iloc[0])
        anchor_status = "first_new_row_offset_no_overlap"

    appended = yahoo_close.loc[yahoo_close.index > last_source_day].copy()
    if appended.empty:
        shutil.copy2(source_path, output_path)
        return 0, "already_current", offset

    adjusted = appended + offset
    appended_df = pd.DataFrame(
        {
            "DATETIME": adjusted.index + pd.Timedelta(hours=23),
            "price": adjusted.values,
        }
    )
    combined = pd.concat([source, appended_df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return len(appended_df), anchor_status, offset


def contract_code_from_row(row: pd.Series | None) -> int:
    if row is None:
        return 0
    raw = str(row.get("last_trade_date_or_contract_month", "")).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    if len(raw) >= 6 and raw[:6].isdigit():
        return int(raw[:6] + "00")
    return 0


def append_multiple_prices(
    instrument: str,
    yahoo_close: pd.Series,
    contracts: pd.DataFrame,
    output_path: Path,
) -> tuple[int, str]:
    source_path = bt.MULTIPLE / f"{instrument}.csv"
    source = pd.read_csv(source_path, parse_dates=["DATETIME"]).sort_values("DATETIME")
    last_source_day = pd.to_datetime(source["DATETIME"]).max().normalize()
    appended = yahoo_close.loc[yahoo_close.index > last_source_day].copy()
    last_source = source.iloc[-1]
    if appended.empty:
        shutil.copy2(source_path, output_path)
        return 0, "already_current"

    row = contracts.loc[instrument] if instrument in contracts.index else None
    code = contract_code_from_row(row)
    disable_day = last_source_day + pd.Timedelta(days=1)
    disable_rows = []
    if disable_day < appended.index.min():
        last_price = float(last_source["PRICE"])
        disable_rows.append(
            {
                "DATETIME": disable_day + pd.Timedelta(hours=23),
                "CARRY": last_price,
                "CARRY_CONTRACT": code,
                "PRICE": last_price,
                "PRICE_CONTRACT": code,
                "FORWARD": last_price,
                "FORWARD_CONTRACT": code,
            }
        )
    appended_df = pd.DataFrame(
        {
            "DATETIME": appended.index + pd.Timedelta(hours=23),
            "CARRY": appended.values,
            "CARRY_CONTRACT": code,
            "PRICE": appended.values,
            "PRICE_CONTRACT": code,
            "FORWARD": appended.values,
            "FORWARD_CONTRACT": code,
        }
    )
    if disable_rows:
        appended_df = pd.concat([pd.DataFrame(disable_rows), appended_df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([source, appended_df], ignore_index=True)
    combined.to_csv(output_path, index=False)
    return len(appended_df), "carry_disabled_price_equals_carry"


def build_fx_overlay(output_dir: Path, refresh: bool) -> pd.DataFrame:
    rows = []
    fx_out = output_dir / "fx_prices_csv"
    fx_out.mkdir(parents=True, exist_ok=True)
    for source_path in bt.FX.glob("*.csv"):
        shutil.copy2(source_path, fx_out / source_path.name)

    for currency, (ticker, invert) in FX_TICKERS.items():
        source_path = bt.FX / f"{currency}USD.csv"
        if not source_path.exists():
            continue
        cache_path = output_dir / "raw_yahoo" / "fx" / f"{currency}_{ticker.replace('=', '_')}.csv"
        close = download_close(ticker, cache_path, refresh)
        if invert:
            close = (1.0 / close.replace(0.0, np.nan)).dropna()
        source = pd.read_csv(source_path, parse_dates=["DATETIME"]).sort_values("DATETIME")
        last_source_day = pd.to_datetime(source["DATETIME"]).max().normalize()
        appended = close.loc[close.index > last_source_day].copy()
        status = "updated" if not appended.empty else "already_current_or_no_new_rows"
        if not appended.empty:
            appended_df = pd.DataFrame(
                {"DATETIME": appended.index + pd.Timedelta(hours=23), "PRICE": appended.values}
            )
            combined = pd.concat([source, appended_df], ignore_index=True)
            combined.to_csv(fx_out / source_path.name, index=False)
        rows.append(
            {
                "currency": currency,
                "ticker": ticker,
                "invert": invert,
                "download_rows": len(close),
                "download_first": close.index.min().date().isoformat() if len(close) else "",
                "download_last": close.index.max().date().isoformat() if len(close) else "",
                "appended_rows": len(appended),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    config = parse_args()
    instruments = no40.selected_instruments()
    contracts = load_qualified_contracts(config.contracts)
    adjusted_out = config.output_dir / "adjusted_prices_csv"
    multiple_out = config.output_dir / "multiple_prices_csv"
    raw_yahoo_dir = config.output_dir / "raw_yahoo" / "futures"
    adjusted_out.mkdir(parents=True, exist_ok=True)
    multiple_out.mkdir(parents=True, exist_ok=True)
    ib = IB()
    if config.ibkr_current_contract_fallback:
        ib.RequestTimeout = 25

        def on_error(req_id, error_code, error_string, _contract):
            if error_code not in {2104, 2106, 2158, 2107, 2108, 10167}:
                print(f"IB_ERROR reqId={req_id} code={error_code} msg={error_string}")

        ib.errorEvent += on_error
        ib.client.connect(config.ibkr_host, config.ibkr_port, clientId=config.ibkr_client_id, timeout=30)

    rows = []
    minimum_latest = latest_allowed(config.max_age_days)
    try:
        for instrument in instruments:
            ticker = YAHOO_FUTURES.get(instrument, "")
            contract_row = contracts.loc[instrument] if instrument in contracts.index else None
            is_qualified = bool(contract_row is not None and contract_row["is_qualified"])
            source_last = source_daily_adjusted(instrument).index.max()
            row = {
                "instrument": instrument,
                "ticker": ticker,
                "data_source": "",
                "data_symbol": "",
                "ibkr_qualified": is_qualified,
                "source_last": source_last.date().isoformat(),
                "download_rows": 0,
                "download_first": "",
                "download_last": "",
                "max_gap_days": np.nan,
                "bridge_gap_days": np.nan,
                "adjusted_appended_rows": 0,
                "multiple_appended_rows": 0,
                "anchor_status": "",
                "multiple_status": "",
                "offset": np.nan,
                "trade_ready": False,
                "status": "",
            }
            if not is_qualified:
                shutil.copy2(bt.ADJUSTED / f"{instrument}.csv", adjusted_out / f"{instrument}.csv")
                shutil.copy2(bt.MULTIPLE / f"{instrument}.csv", multiple_out / f"{instrument}.csv")
                row["status"] = "blocked_unqualified_ibkr_contract"
                rows.append(row)
                continue

            close = pd.Series(dtype=float)
            failure_status = "blocked_no_yahoo_proxy_mapping"
            if ticker:
                close = download_close(
                    ticker, raw_yahoo_dir / f"{instrument}_{ticker.replace('=', '_')}.csv", config.refresh
                )
                failure_status = "blocked_yahoo_unknown"
                if len(close) < config.min_download_rows:
                    failure_status = "blocked_insufficient_yahoo_history"
                elif close.index.max() < minimum_latest:
                    failure_status = "blocked_stale_yahoo_history"
                elif max_calendar_gap_days(close) > config.max_gap_days:
                    failure_status = "blocked_gappy_yahoo_history"
                else:
                    row["data_source"] = "yahoo_nearby_proxy"
                    row["data_symbol"] = ticker

            use_fallback = not row["data_source"] and config.ibkr_current_contract_fallback
            if use_fallback:
                try:
                    ibkr_close = download_ibkr_current_contract_close(
                        ib,
                        instrument,
                        contract_row,
                        config.output_dir / "raw_ibkr_current_contract",
                    )
                except Exception as exc:
                    ibkr_close = pd.Series(dtype=float)
                    failure_status = f"blocked_ibkr_fallback_error:{str(exc)[:80]}"
                if not ibkr_close.empty:
                    if len(ibkr_close) < config.min_ibkr_rows:
                        failure_status = "blocked_insufficient_ibkr_current_contract_history"
                    elif ibkr_close.index.max() < minimum_latest:
                        failure_status = "blocked_stale_ibkr_current_contract_history"
                    elif max_calendar_gap_days(ibkr_close) > config.max_gap_days:
                        failure_status = "blocked_gappy_ibkr_current_contract_history"
                    else:
                        close = ibkr_close
                        row["data_source"] = "ibkr_current_contract_proxy"
                        row["data_symbol"] = text(contract_row["local_symbol"])

            if not row["data_source"]:
                shutil.copy2(bt.ADJUSTED / f"{instrument}.csv", adjusted_out / f"{instrument}.csv")
                shutil.copy2(bt.MULTIPLE / f"{instrument}.csv", multiple_out / f"{instrument}.csv")
                row["status"] = failure_status
                rows.append(row)
                continue

            source_days = source_daily_adjusted(instrument).index
            overlap_days = source_days.intersection(close.index)
            first_new = close.loc[close.index > source_last].index.min()
            bridge_gap_days = (
                float((first_new - source_last).days) if pd.notna(first_new) else 0.0
            )
            row["bridge_gap_days"] = bridge_gap_days
            if len(overlap_days) == 0 and bridge_gap_days > config.max_gap_days:
                shutil.copy2(bt.ADJUSTED / f"{instrument}.csv", adjusted_out / f"{instrument}.csv")
                shutil.copy2(bt.MULTIPLE / f"{instrument}.csv", multiple_out / f"{instrument}.csv")
                row["data_source"] = ""
                row["data_symbol"] = ""
                row["status"] = f"blocked_source_bridge_gap_{int(bridge_gap_days)}d"
                rows.append(row)
                continue

            row["download_rows"] = len(close)
            row["download_first"] = close.index.min().date().isoformat() if len(close) else ""
            row["download_last"] = close.index.max().date().isoformat() if len(close) else ""
            row["max_gap_days"] = max_calendar_gap_days(close)
            adjusted_rows, anchor_status, offset = append_adjusted_prices(
                instrument, close, adjusted_out / f"{instrument}.csv"
            )
            multiple_rows, multiple_status = append_multiple_prices(
                instrument, close, contracts, multiple_out / f"{instrument}.csv"
            )
            row.update(
                {
                    "adjusted_appended_rows": adjusted_rows,
                    "multiple_appended_rows": multiple_rows,
                    "anchor_status": anchor_status,
                    "multiple_status": multiple_status,
                    "offset": offset,
                    "trade_ready": True,
                    "status": "ready_proxy_trend_only_carry_disabled"
                    if row["data_source"] == "yahoo_nearby_proxy"
                    else "ready_degraded_ibkr_current_contract_proxy_carry_disabled",
                }
            )
            rows.append(row)
    finally:
        if ib.isConnected():
            ib.disconnect()

    fx_coverage = build_fx_overlay(config.output_dir, config.refresh)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(config.output_dir / "coverage.csv", index=False)
    fx_coverage.to_csv(config.output_dir / "fx_coverage.csv", index=False)

    ready = int(coverage["trade_ready"].sum())
    qualified = int(coverage["ibkr_qualified"].sum())
    print(f"wrote_overlay={config.output_dir}")
    print(f"qualified={qualified}/{len(coverage)} ready_proxy={ready}/{len(coverage)}")
    print(coverage.groupby("status").size().to_string())
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
