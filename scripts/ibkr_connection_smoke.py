#!/usr/bin/env python3
"""Read-only IBKR Gateway smoke test.

This script deliberately avoids order placement. It checks whether the local
IB Gateway socket is open, whether the API handshake works, and whether account
state is available through the API.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import NoReturn

from ib_insync import IB


@dataclass(frozen=True)
class SmokeConfig:
    host: str
    port: int
    client_id: int
    wait_seconds: float


def parse_args() -> SmokeConfig:
    parser = argparse.ArgumentParser(description="Read-only IBKR Gateway smoke test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002, help="Paper IB Gateway is commonly 4002")
    parser.add_argument("--client-id", type=int, default=17)
    parser.add_argument("--wait-seconds", type=float, default=8.0)
    args = parser.parse_args()
    return SmokeConfig(args.host, args.port, args.client_id, args.wait_seconds)


def fail(code: int, reason: str, fix: str) -> NoReturn:
    print(f"FAIL {reason}")
    print(f"Fix: {fix}")
    sys.exit(code)


def main() -> int:
    config = parse_args()
    ib = IB()
    try:
        print(f"api_connect host={config.host} port={config.port} client_id={config.client_id}")
        try:
            ib.client.connect(config.host, config.port, clientId=config.client_id, timeout=10)
        except TimeoutError:
            fail(
                2,
                "api_handshake_timeout",
                "restart IB Gateway, confirm paper mode, API socket clients enabled, and port 4002.",
            )
        except OSError as exc:
            fail(
                2,
                "api_socket_unreachable",
                f"check IB Gateway is running and listening on {config.host}:{config.port}; detail={exc}",
            )
        print(f"PASS api_handshake server_version={ib.client.serverVersion()}")
        accounts = ib.client.getAccounts()
        print(f"account_count={len(accounts)}")
        if not accounts:
            fail(3, "no_managed_accounts", "confirm the Gateway login is a trading or paper-trading user.")

        current_time = ib.reqCurrentTime()
        print(f"PASS server_time={current_time}")

        ib.client.reqPositions()
        ib.sleep(config.wait_seconds)
        positions = ib.positions()
        print(f"positions_count={len(positions)}")
        try:
            ib.client.cancelPositions()
        except Exception:
            pass

        req_id = ib.client.getReqId()
        tags = (
            "NetLiquidation,TotalCashValue,InitMarginReq,MaintMarginReq,"
            "ExcessLiquidity,AvailableFunds,BuyingPower"
        )
        ib.client.reqAccountSummary(req_id, "All", tags)
        ib.sleep(config.wait_seconds)
        summary_values = list(ib.wrapper.acctSummary.values())
        print(f"account_summary_count={len(summary_values)}")
        for value in summary_values:
            print(f"summary {value.account} {value.tag} {value.value} {value.currency}")
        try:
            ib.client.cancelAccountSummary(req_id)
        except Exception:
            pass

        account = accounts[0]
        ib.client.reqAccountUpdates(True, account)
        ib.sleep(config.wait_seconds)
        account_values = list(ib.wrapper.accountValues.values())
        print(f"account_updates_count={len(account_values)}")
        wanted = {
            "NetLiquidation",
            "TotalCashValue",
            "InitMarginReq",
            "MaintMarginReq",
            "ExcessLiquidity",
            "AvailableFunds",
            "BuyingPower",
        }
        for value in account_values:
            if value.tag in wanted:
                print(f"account_value {value.account} {value.tag} {value.value} {value.currency}")
        try:
            ib.client.reqAccountUpdates(False, account)
        except Exception:
            pass

        has_account_state = bool(summary_values or account_values)
        if not has_account_state:
            fail(
                4,
                "no_account_state",
                "restart Gateway and retry; if it persists, open TWS once to confirm the paper account has Account/Portfolio data.",
            )

        print("PASS account_state_available")
        return 0
    finally:
        if ib.client.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
