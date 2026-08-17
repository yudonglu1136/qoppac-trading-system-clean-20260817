#!/usr/bin/env python3
"""Qualify IBKR futures contract mappings without placing orders."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import Contract, IB

ROOT = Path(__file__).resolve().parents[1]
MARGIN_DIR = ROOT / "backtests" / "rob_style_no_equity_40_margin_constrained"
DEFAULT_OUT = ROOT / "output" / "ibkr_contract_qualification" / "sample_contracts.csv"

DEFAULT_INSTRUMENTS = [
    "US10",
    "US2",
    "GOLD_micro",
    "COPPER-micro",
    "CRUDE_W",
    "CORN",
    "AUD",
    "GBP",
    "JPY",
    "VIX",
]


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    client_id: int
    instruments: list[str]
    all_instruments: bool
    min_days_to_expiry: int
    timeout: float
    output: Path
    min_qualified: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Qualify IBKR futures contracts without order placement")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002, help="Paper IB Gateway is commonly 4002")
    parser.add_argument("--client-id", type=int, default=51)
    parser.add_argument("--instrument", action="append", dest="instruments", default=[])
    parser.add_argument("--all", action="store_true", dest="all_instruments")
    parser.add_argument("--min-days-to-expiry", type=int, default=14)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-qualified",
        type=int,
        default=0,
        help="Publish only when at least this many instruments qualify; 0 requires every requested instrument.",
    )
    args = parser.parse_args()
    instruments = args.instruments if args.instruments else DEFAULT_INSTRUMENTS
    return Config(
        args.host,
        args.port,
        args.client_id,
        instruments,
        args.all_instruments,
        args.min_days_to_expiry,
        args.timeout,
        args.output,
        args.min_qualified,
    )


def load_schedule() -> pd.DataFrame:
    schedule_path = MARGIN_DIR / "margin_schedule.csv"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Missing {schedule_path}")
    schedule = pd.read_csv(schedule_path).fillna("")
    return schedule


def load_margin_snapshot_underlyings() -> pd.DataFrame:
    snapshot_path = MARGIN_DIR / "ibkr_current_margin_table_snapshot.csv"
    if not snapshot_path.exists():
        return pd.DataFrame()
    snapshot = pd.read_csv(snapshot_path).fillna("")
    keep = ["Exchange", "Underlying", "Trading Class", "Product description", "Currency"]
    return snapshot[[column for column in keep if column in snapshot.columns]].drop_duplicates()


def normalize_text(value: object) -> str:
    return str(value).strip()


def candidate_symbols(row: pd.Series, snapshot: pd.DataFrame) -> list[str]:
    symbols: list[str] = []
    exchange = normalize_text(row["exchange"]).upper()
    trading_class = normalize_text(row["trading_class"]).upper()
    currency = normalize_text(row["margin_currency"]).upper()

    if not snapshot.empty:
        matches = snapshot[
            snapshot["Exchange"].astype(str).str.upper().eq(exchange)
            & snapshot["Trading Class"].astype(str).str.upper().eq(trading_class)
            & snapshot["Currency"].astype(str).str.upper().eq(currency)
        ]
        symbols.extend(matches["Underlying"].astype(str).str.strip().tolist())

    symbols.append(trading_class)
    if trading_class == "VX":
        symbols.append("VIX")
    if trading_class == "FVS":
        symbols.extend(["V2TX", "V2X"])
    if trading_class in {"6A", "6B", "6C", "6J", "6M", "6N"}:
        symbols.append(normalize_text(row["instrument"]).upper())

    seen: set[str] = set()
    deduped = []
    for symbol in symbols:
        symbol = normalize_text(symbol).upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            deduped.append(symbol)
    return deduped


def candidate_exchanges(exchange: str) -> list[str]:
    exchange = normalize_text(exchange).upper()
    extras = {
        "CBOT": ["CBOT", "ECBOT"],
        "CME": ["CME", "GLOBEX"],
        "COMEX": ["COMEX", "NYMEX"],
        "NYMEX": ["NYMEX"],
        "CFE": ["CFE"],
        "EUREX": ["EUREX"],
        "SGX": ["SGX"],
        "KSE": ["KSE"],
    }
    return extras.get(exchange, [exchange])


def parse_expiry(value: str) -> date | None:
    value = normalize_text(value)
    if not value:
        return None
    for fmt in ("%Y%m%d", "%Y%m"):
        try:
            parsed = datetime.strptime(value[: len(datetime.now().strftime(fmt))], fmt)
            if fmt == "%Y%m":
                return date(parsed.year, parsed.month, 1)
            return parsed.date()
        except ValueError:
            continue
    return None


def choose_trade_contract(details, min_days_to_expiry: int) -> tuple[object | None, object | None]:
    today = date.today()
    min_expiry = today + timedelta(days=min_days_to_expiry)
    dated = []
    for detail in details:
        contract = detail.contract
        expiry = parse_expiry(contract.lastTradeDateOrContractMonth)
        if expiry is None:
            continue
        if expiry >= today:
            dated.append((expiry, detail))
    if not dated:
        return None, None
    dated.sort(key=lambda item: item[0])
    front = dated[0][1]
    eligible = [item for item in dated if item[0] >= min_expiry]
    if not eligible:
        return front, None
    return front, eligible[0][1]


def describe_contract(detail) -> dict[str, object]:
    contract = detail.contract
    expiry = parse_expiry(contract.lastTradeDateOrContractMonth)
    return {
        "con_id": contract.conId,
        "symbol": contract.symbol,
        "sec_type": contract.secType,
        "exchange": contract.exchange,
        "primary_exchange": contract.primaryExchange,
        "currency": contract.currency,
        "trading_class": contract.tradingClass,
        "local_symbol": contract.localSymbol,
        "last_trade_date_or_contract_month": contract.lastTradeDateOrContractMonth,
        "days_to_expiry": (expiry - date.today()).days if expiry else "",
        "multiplier": contract.multiplier,
        "min_tick": getattr(detail, "minTick", ""),
        "time_zone_id": getattr(detail, "timeZoneId", ""),
        "long_name": getattr(detail, "longName", ""),
    }


def query_details(ib: IB, symbol: str, exchange: str, currency: str, trading_class: str):
    contract = Contract(
        secType="FUT",
        symbol=symbol,
        exchange=exchange,
        currency=currency,
        tradingClass=trading_class,
    )
    try:
        return ib.reqContractDetails(contract)
    except Exception:
        return []


def qualify_instrument(ib: IB, row: pd.Series, snapshot: pd.DataFrame, min_days_to_expiry: int) -> dict[str, object]:
    instrument = normalize_text(row["instrument"])
    source = normalize_text(row["source"])
    exchange = normalize_text(row["exchange"]).upper()
    trading_class = normalize_text(row["trading_class"]).upper()
    currency = normalize_text(row["margin_currency"]).upper()

    base = {
        "instrument": instrument,
        "asset_class": normalize_text(row["asset_class"]),
        "source": source,
        "configured_exchange": exchange,
        "configured_trading_class": trading_class,
        "configured_currency": currency,
        "status": "unqualified",
        "query_symbol": "",
        "query_exchange": "",
        "detail_count": 0,
        "candidate_expiries": "",
        "warning": "",
    }

    if source != "ibkr_current_overnight_initial" or not exchange or not trading_class:
        base["status"] = "skipped_fallback_mapping"
        base["warning"] = "No verified IBKR margin mapping; do not trade until manually mapped."
        return base

    for symbol in candidate_symbols(row, snapshot):
        for query_exchange in candidate_exchanges(exchange):
            details = query_details(ib, symbol, query_exchange, currency, trading_class)
            if not details:
                continue
            expiries = sorted(
                {
                    normalize_text(detail.contract.lastTradeDateOrContractMonth)
                    for detail in details
                    if normalize_text(detail.contract.lastTradeDateOrContractMonth)
                }
            )
            front, chosen = choose_trade_contract(details, min_days_to_expiry)
            result = {
                **base,
                "status": "qualified_trade_candidate_found" if chosen else "details_found_no_trade_candidate",
                "query_symbol": symbol,
                "query_exchange": query_exchange,
                "detail_count": len(details),
                "candidate_expiries": ";".join(expiries[:12]),
            }
            if front:
                result.update({f"front_{key}": value for key, value in describe_contract(front).items()})
            if chosen:
                result.update(describe_contract(chosen))
            else:
                result["warning"] = f"No contract expires at least {min_days_to_expiry} days from today."
            return result

    base["warning"] = "No IBKR contract details returned for generated symbol/exchange candidates."
    return base


def main() -> int:
    config = parse_args()
    schedule = load_schedule()
    snapshot = load_margin_snapshot_underlyings()
    instruments = schedule["instrument"].tolist() if config.all_instruments else config.instruments
    schedule = schedule[schedule["instrument"].isin(instruments)].copy()

    missing = sorted(set(instruments) - set(schedule["instrument"]))
    if missing:
        print(f"missing_from_margin_schedule={missing}")

    ib = IB()
    ib.RequestTimeout = config.timeout
    errors: list[tuple[int, int, str]] = []

    def on_error(req_id, error_code, error_string, _contract):
        errors.append((req_id, error_code, error_string))
        if error_code not in {2104, 2106, 2158}:
            print(f"IB_ERROR reqId={req_id} code={error_code} msg={error_string}")

    ib.errorEvent += on_error
    try:
        ib.client.connect(config.host, config.port, clientId=config.client_id, timeout=10)
        accounts = ib.client.getAccounts()
        print(f"connected server_version={ib.client.serverVersion()} account_count={len(accounts)}")
        rows = []
        for _, row in schedule.iterrows():
            result = qualify_instrument(ib, row, snapshot, config.min_days_to_expiry)
            rows.append(result)
            print(
                f"{result['instrument']}: {result['status']} "
                f"{result.get('local_symbol', '')} {result.get('exchange', '')} "
                f"{result.get('last_trade_date_or_contract_month', '')}"
            )
        output = config.output
        output.parent.mkdir(parents=True, exist_ok=True)
        qualified_count = sum(row["status"].startswith("qualified") for row in rows)
        required_count = config.min_qualified if config.min_qualified > 0 else len(rows)
        if rows and qualified_count >= required_count:
            tmp = output.with_suffix(output.suffix + ".tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.replace(output)
            print(f"wrote={output} qualified={qualified_count}/{len(rows)} required={required_count}")
            return 0
        rejected = output.with_name(
            f"{output.stem}.rejected.{datetime.now().strftime('%Y%m%dT%H%M%S')}{output.suffix}"
        )
        pd.DataFrame(rows).to_csv(rejected, index=False)
        print(f"publish_blocked={rejected} qualified={qualified_count}/{len(rows)} required={required_count}")
        return 1
    finally:
        if ib.client.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
