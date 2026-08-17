#!/usr/bin/env python3
"""Place a tiny paper market-order round trip through IB Gateway.

This is an execution smoke test, not a strategy runner. It refuses to run unless
an explicit confirmation flag is provided and the connected account looks like
an IBKR paper account.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ib_insync import Contract, IB, MarketOrder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT / "output" / "ibkr_contract_qualification" / "sample_contracts.csv"
DEFAULT_OUT = ROOT / "output" / "ibkr_execution_smoke"


def default_expected_account() -> str:
    return os.environ.get("IBKR_EXPECTED_ACCOUNT") or os.environ.get("IBKR_PAPER_ACCOUNT", "")


def mask_account(account: str) -> str:
    value = str(account)
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}...{value[-2:]}"


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    client_id: int
    contracts: Path
    output_dir: Path
    expected_account: str
    instrument: str
    action: str
    quantity: int
    wait_seconds: float
    round_trip: bool
    confirm: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Paper market-order smoke test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=81)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-account", default=default_expected_account())
    parser.add_argument("--instrument", default="US10")
    parser.add_argument("--action", choices=["BUY", "SELL"], default="BUY")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--round-trip", action="store_true")
    parser.add_argument("--confirm-paper-market-order", action="store_true", dest="confirm")
    args = parser.parse_args()
    if not args.expected_account:
        parser.error("--expected-account is required, or set IBKR_EXPECTED_ACCOUNT / IBKR_PAPER_ACCOUNT")
    return Config(
        args.host,
        args.port,
        args.client_id,
        args.contracts,
        args.output_dir,
        args.expected_account,
        args.instrument,
        args.action,
        args.quantity,
        args.wait_seconds,
        args.round_trip,
        args.confirm,
    )


def fail(code: int, message: str) -> int:
    print(f"FAIL {message}")
    return code


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    result = str(value).strip()
    if result.endswith(".0") and result[:-2].isdigit():
        return result[:-2]
    return result


def load_contract(config: Config) -> tuple[pd.Series, Contract]:
    if not config.contracts.exists():
        raise FileNotFoundError(f"Missing contract qualification file: {config.contracts}")
    contracts = pd.read_csv(config.contracts)
    contracts = contracts[contracts["status"].astype(str).str.startswith("qualified")].copy()
    matches = contracts[contracts["instrument"].eq(config.instrument)]
    if matches.empty:
        raise ValueError(f"No qualified contract row for {config.instrument}")
    row = matches.iloc[0]
    contract = Contract(
        secType="FUT",
        conId=int(float(row["con_id"])),
        symbol=text(row["symbol"]),
        exchange=text(row["exchange"]),
        currency=text(row["currency"]),
        localSymbol=text(row["local_symbol"]),
        tradingClass=text(row["trading_class"]),
    )
    return row, contract


def number_or_blank(value: object) -> object:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return result if math.isfinite(result) else ""


def get_account_values(ib: IB, account: str, wait_seconds: float) -> dict[str, str]:
    req_id = ib.client.getReqId()
    tags = "NetLiquidation,TotalCashValue,InitMarginReq,MaintMarginReq,ExcessLiquidity,AvailableFunds,BuyingPower"
    ib.client.reqAccountSummary(req_id, "All", tags)
    ib.sleep(wait_seconds)
    values = {
        value.tag: value.value
        for value in ib.wrapper.acctSummary.values()
        if value.account == account and value.currency in {"USD", ""}
    }
    try:
        ib.client.cancelAccountSummary(req_id)
    except Exception:
        pass
    return values


def get_quote(ib: IB, contract: Contract, wait_seconds: float) -> dict[str, object]:
    ib.reqMarketDataType(3)
    ticker = ib.reqMktData(contract, snapshot=False, regulatorySnapshot=False)
    ib.sleep(wait_seconds)
    quote = {
        "market_data_type": ticker.marketDataType,
        "bid": number_or_blank(ticker.bid),
        "ask": number_or_blank(ticker.ask),
        "last": number_or_blank(ticker.last),
        "close": number_or_blank(ticker.close),
        "bid_size": number_or_blank(ticker.bidSize),
        "ask_size": number_or_blank(ticker.askSize),
    }
    try:
        ib.cancelMktData(contract)
    except Exception:
        pass
    return quote


def wait_for_terminal(ib: IB, trade, wait_seconds: float) -> str:
    terminal = {"Filled", "Cancelled", "Inactive", "ApiCancelled"}
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        ib.sleep(0.25)
        status = trade.orderStatus.status
        if status in terminal:
            return status
    return trade.orderStatus.status


def fills_to_rows(label: str, trade) -> list[dict[str, object]]:
    rows = []
    for fill in trade.fills:
        rows.append(
            {
                "leg": label,
                "time": fill.time,
                "side": fill.execution.side,
                "shares": fill.execution.shares,
                "price": fill.execution.price,
                "exchange": fill.execution.exchange,
                "commission": getattr(fill.commissionReport, "commission", ""),
                "currency": getattr(fill.commissionReport, "currency", ""),
                "exec_id": fill.execution.execId,
            }
        )
    return rows


def place_market_order(
    ib: IB,
    account: str,
    contract: Contract,
    action: str,
    quantity: int,
    order_ref: str,
    wait_seconds: float,
):
    order = MarketOrder(action, quantity)
    order.account = account
    order.orderRef = order_ref
    order.tif = "DAY"
    trade = ib.placeOrder(contract, order)
    status = wait_for_terminal(ib, trade, wait_seconds)
    return trade, status


def write_failure_artifacts(
    config: Config,
    timestamp: str,
    instrument: str,
    fills: list[dict[str, object]],
    errors: list[dict[str, object]],
    reason: str,
) -> None:
    fill_path = config.output_dir / f"{timestamp}_{instrument}_fills.csv"
    error_path = config.output_dir / f"{timestamp}_{instrument}_errors.csv"
    summary_path = config.output_dir / f"{timestamp}_{instrument}_summary.csv"
    pd.DataFrame(fills).to_csv(fill_path, index=False)
    pd.DataFrame(errors).to_csv(error_path, index=False)
    pd.DataFrame([{"timestamp_utc": timestamp, "instrument": instrument, "status": "fail", "reason": reason}]).to_csv(
        summary_path, index=False
    )
    print(f"wrote_fills={fill_path}")
    print(f"wrote_errors={error_path}")
    print(f"wrote_summary={summary_path}")


def main() -> int:
    config = parse_args()
    if not config.confirm:
        return fail(2, "missing --confirm-paper-market-order")
    if config.port != 4002:
        return fail(2, "refusing market-order smoke test unless port is paper Gateway 4002")
    if config.quantity < 1 or config.quantity > 1:
        return fail(2, "quantity must be exactly 1 for the first paper market-order smoke test")

    row, contract = load_contract(config)
    ib = IB()
    errors: list[dict[str, object]] = []

    def on_error(req_id, error_code, error_string, _contract):
        if error_code in {2104, 2106, 2107, 2158, 10167}:
            return
        errors.append({"req_id": req_id, "code": error_code, "message": error_string})
        print(f"IB_ERROR reqId={req_id} code={error_code} msg={error_string}")

    ib.errorEvent += on_error
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ib.MaxSyncedSubAccounts = 0
        ib.RequestTimeout = 10
        ib.connect(
            config.host,
            config.port,
            clientId=config.client_id,
            timeout=15,
            readonly=False,
            account=config.expected_account,
        )
        accounts = ib.managedAccounts()
        if not accounts:
            return fail(3, "no IBKR account returned by Gateway")
        account = config.expected_account if config.expected_account in accounts else accounts[0]
        if config.expected_account and account != config.expected_account:
            return fail(3, f"expected configured account, got account_count={len(accounts)}")
        if not account.startswith("D"):
            return fail(3, f"refusing to trade non-paper-looking account: {mask_account(account)}")

        print(f"connected account={mask_account(account)} instrument={config.instrument} local_symbol={contract.localSymbol}")
        before = get_account_values(ib, account, 3.0)
        print(f"before_account={before}")
        quote = get_quote(ib, contract, 5.0)
        print(f"delayed_quote={quote}")

        order_ref = f"codex_paper_smoke_{timestamp}_{config.instrument}"
        trade, status = place_market_order(
            ib, account, contract, config.action, config.quantity, order_ref, config.wait_seconds
        )
        print(
            f"entry_status={status} filled={trade.orderStatus.filled} "
            f"avg_fill_price={trade.orderStatus.avgFillPrice}"
        )
        fills = fills_to_rows("entry", trade)
        if status != "Filled" or float(trade.orderStatus.filled or 0) < config.quantity:
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass
            write_failure_artifacts(config, timestamp, config.instrument, fills, errors, "entry order did not fill")
            return fail(4, "entry order did not fill; check Read-Only API, trading permissions, and order warnings")

        if config.round_trip:
            exit_action = "SELL" if config.action == "BUY" else "BUY"
            exit_trade, exit_status = place_market_order(
                ib,
                account,
                contract,
                exit_action,
                config.quantity,
                f"{order_ref}_flatten",
                config.wait_seconds,
            )
            print(
                f"exit_status={exit_status} filled={exit_trade.orderStatus.filled} "
                f"avg_fill_price={exit_trade.orderStatus.avgFillPrice}"
            )
            fills.extend(fills_to_rows("exit", exit_trade))
            if exit_status != "Filled" or float(exit_trade.orderStatus.filled or 0) < config.quantity:
                write_failure_artifacts(config, timestamp, config.instrument, fills, errors, "exit order did not fill")
                return fail(5, "exit order did not fill; manually verify and flatten the paper position")

        after = get_account_values(ib, account, 3.0)
        print(f"after_account={after}")

        fill_path = config.output_dir / f"{timestamp}_{config.instrument}_fills.csv"
        pd.DataFrame(fills).to_csv(fill_path, index=False)
        error_path = config.output_dir / f"{timestamp}_{config.instrument}_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        summary_path = config.output_dir / f"{timestamp}_{config.instrument}_summary.csv"
        pd.DataFrame(
            [
                {
                    "timestamp_utc": timestamp,
                    "account": account,
                    "instrument": config.instrument,
                    "local_symbol": text(row["local_symbol"]),
                    "action": config.action,
                    "quantity": config.quantity,
                    "round_trip": config.round_trip,
                    "entry_status": status,
                    "before_net_liquidation": before.get("NetLiquidation", ""),
                    "after_net_liquidation": after.get("NetLiquidation", ""),
                    **{f"quote_{key}": value for key, value in quote.items()},
                    "fill_file": str(fill_path),
                    "error_file": str(error_path),
                }
            ]
        ).to_csv(summary_path, index=False)
        print(f"wrote_fills={fill_path}")
        print(f"wrote_summary={summary_path}")
        return 0
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
