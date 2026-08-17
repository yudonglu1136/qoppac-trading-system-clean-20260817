#!/usr/bin/env python3
"""Check IBKR market data availability for qualified futures contracts.

The script only requests market data. It does not place orders.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ib_insync import Contract, IB

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT / "output" / "ibkr_contract_qualification" / "sample_contracts.csv"
DEFAULT_OUT = ROOT / "output" / "ibkr_market_data_gate" / "sample_market_data.csv"

MARKET_DATA_TYPES = {
    "live": 1,
    "frozen": 2,
    "delayed": 3,
    "delayed_frozen": 4,
}

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
    contracts: Path
    output: Path
    market_data_type: str
    wait_seconds: float
    instruments: list[str]
    all_instruments: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Check IBKR futures market data availability")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=61)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--market-data-type", choices=MARKET_DATA_TYPES, default="delayed")
    parser.add_argument("--wait-seconds", type=float, default=12.0)
    parser.add_argument("--instrument", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_instruments")
    args = parser.parse_args()
    instruments = args.instrument if args.instrument else DEFAULT_INSTRUMENTS
    return Config(
        args.host,
        args.port,
        args.client_id,
        args.contracts,
        args.output,
        args.market_data_type,
        args.wait_seconds,
        instruments,
        args.all_instruments,
    )


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def clean_int(value: object) -> int | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def load_contract_rows(config: Config) -> pd.DataFrame:
    if not config.contracts.exists():
        raise FileNotFoundError(f"Missing contract qualification file: {config.contracts}")
    contracts = pd.read_csv(config.contracts)
    contracts = contracts[contracts["status"].astype(str).str.startswith("qualified")].copy()
    if not config.all_instruments:
        contracts = contracts[contracts["instrument"].isin(config.instruments)].copy()
    return contracts


def contract_from_row(row: pd.Series) -> Contract | None:
    con_id = clean_int(row.get("con_id"))
    if con_id is None:
        return None
    return Contract(
        secType="FUT",
        conId=con_id,
        symbol=clean_text(row.get("symbol")),
        exchange=clean_text(row.get("exchange")),
        currency=clean_text(row.get("currency")),
        localSymbol=clean_text(row.get("local_symbol")),
        tradingClass=clean_text(row.get("trading_class")),
        lastTradeDateOrContractMonth=clean_text(row.get("last_trade_date_or_contract_month")),
        multiplier=clean_text(row.get("multiplier")),
    )


def quote_status(ticker) -> str:
    bid_ok = finite_positive(ticker.bid)
    ask_ok = finite_positive(ticker.ask)
    last_ok = finite_positive(ticker.last)
    close_ok = finite_positive(ticker.close)
    mark_ok = finite_positive(ticker.markPrice)
    if bid_ok and ask_ok:
        return "bid_ask"
    if last_ok:
        return "last_only"
    if mark_ok:
        return "mark_only"
    if close_ok:
        return "close_only"
    return "no_quote"


def gate_status(status: str) -> str:
    if status == "bid_ask":
        return "pass_bid_ask"
    if status in {"last_only", "mark_only", "close_only"}:
        return "pass_price_only"
    return "fail_no_market_data"


def numeric_or_blank(value: object) -> object:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return number


def main() -> int:
    config = parse_args()
    contracts = load_contract_rows(config)
    if contracts.empty:
        print("No qualified contracts selected.")
        return 2

    errors_by_req_id: dict[int, list[str]] = {}
    req_id_to_instrument: dict[int, str] = {}
    ticker_rows = []
    ib = IB()

    def on_error(req_id, error_code, error_string, _contract):
        if error_code in {2104, 2106, 2158}:
            return
        errors_by_req_id.setdefault(req_id, []).append(f"{error_code}: {error_string}")
        instrument = req_id_to_instrument.get(req_id, "global")
        print(f"IB_ERROR instrument={instrument} reqId={req_id} code={error_code} msg={error_string}")

    ib.errorEvent += on_error
    try:
        ib.client.connect(config.host, config.port, clientId=config.client_id, timeout=10)
        accounts = ib.client.getAccounts()
        print(f"connected server_version={ib.client.serverVersion()} account_count={len(accounts)}")
        ib.reqMarketDataType(MARKET_DATA_TYPES[config.market_data_type])
        print(f"market_data_type={config.market_data_type}")

        for _, row in contracts.iterrows():
            contract = contract_from_row(row)
            if contract is None:
                continue
            ticker = ib.reqMktData(contract, genericTickList="", snapshot=False, regulatorySnapshot=False)
            req_id = ib.wrapper.ticker2ReqId["mktData"][ticker]
            req_id_to_instrument[req_id] = clean_text(row["instrument"])
            ticker_rows.append((row, contract, ticker, req_id))

        ib.sleep(config.wait_seconds)

        rows = []
        for row, contract, ticker, req_id in ticker_rows:
            status = quote_status(ticker)
            result = {
                "instrument": clean_text(row["instrument"]),
                "asset_class": clean_text(row.get("asset_class")),
                "local_symbol": clean_text(row.get("local_symbol")),
                "con_id": clean_int(row.get("con_id")),
                "exchange": clean_text(row.get("exchange")),
                "currency": clean_text(row.get("currency")),
                "trading_class": clean_text(row.get("trading_class")),
                "expiry": clean_text(row.get("last_trade_date_or_contract_month")),
                "market_data_type_requested": config.market_data_type,
                "market_data_type_returned": ticker.marketDataType,
                "quote_status": status,
                "gate_status": gate_status(status),
                "bid": numeric_or_blank(ticker.bid),
                "ask": numeric_or_blank(ticker.ask),
                "last": numeric_or_blank(ticker.last),
                "close": numeric_or_blank(ticker.close),
                "mark_price": numeric_or_blank(ticker.markPrice),
                "bid_size": numeric_or_blank(ticker.bidSize),
                "ask_size": numeric_or_blank(ticker.askSize),
                "last_size": numeric_or_blank(ticker.lastSize),
                "volume": numeric_or_blank(ticker.volume),
                "snapshot_permissions": clean_text(ticker.snapshotPermissions),
                "errors": " | ".join(errors_by_req_id.get(req_id, [])),
            }
            rows.append(result)
            print(
                f"{result['instrument']}: {result['gate_status']} "
                f"bid={result['bid']} ask={result['ask']} last={result['last']} "
                f"close={result['close']} errors={result['errors']}"
            )

        for _, contract, _ticker, _req_id in ticker_rows:
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass

        output = config.output
        output.parent.mkdir(parents=True, exist_ok=True)
        result_frame = pd.DataFrame(rows)
        result_frame.to_csv(output, index=False)
        print(f"wrote={output}")
        failures = result_frame[result_frame["gate_status"].eq("fail_no_market_data")]
        return 0 if failures.empty else 1
    finally:
        if ib.client.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
