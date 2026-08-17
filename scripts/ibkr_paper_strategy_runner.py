#!/usr/bin/env python3
"""Execute approved paper strategy orders and persist IBKR state locally.

This is the paper-trading bridge for the Rob-style futures strategy.  It only
trades when the latest dry-run summary says transmission is allowed, and it
refuses to run outside the IBKR paper gateway/account.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from ib_insync import Contract, IB, MarketOrder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIEW = ROOT / "output" / "ibkr_strategy_order_dry_run" / "latest_1x_order_preview.csv"
DEFAULT_DRY_RUN_SUMMARY = ROOT / "output" / "ibkr_strategy_order_dry_run" / "latest_1x_order_summary.md"
DEFAULT_CONTRACTS = ROOT / "output" / "ibkr_contract_qualification" / "all_40_contracts.csv"
DEFAULT_BAR_GATE = ROOT / "output" / "ibkr_market_data_gate" / "all_40_15min_historical.csv"
DEFAULT_COVERAGE = ROOT / "output" / "live_futures_overlay" / "coverage.csv"
DEFAULT_DB = ROOT / "data" / "ibkr_paper_trading" / "ibkr_paper_trading.sqlite"
DEFAULT_OUT = ROOT / "output" / "ibkr_paper_strategy_runs"
DEFAULT_GUARDRAILS = ROOT / "config" / "ibkr_paper_live_guardrails.yaml"


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
    expected_account: str
    preview: Path
    dry_run_summary: Path
    contracts: Path
    bar_gate: Path
    coverage: Path
    database: Path
    output_dir: Path
    wait_seconds: float
    fetch_bar_history: bool
    bar_duration: str
    bar_size: str
    bar_scope: str
    continue_after_unfilled: bool
    skip_state_snapshot: bool
    execute_orders: bool
    confirm: bool
    allow_repeat_target: bool
    guardrails: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Execute approved IBKR paper strategy market orders")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=151)
    parser.add_argument("--expected-account", default=default_expected_account())
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--dry-run-summary", type=Path, default=DEFAULT_DRY_RUN_SUMMARY)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--bar-gate", type=Path, default=DEFAULT_BAR_GATE)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--wait-seconds", type=float, default=45.0)
    parser.add_argument("--fetch-bar-history", action="store_true")
    parser.add_argument("--bar-duration", default="2 D")
    parser.add_argument("--bar-size", default="15 mins")
    parser.add_argument("--bar-scope", choices=["actionable", "all_qualified"], default="all_qualified")
    parser.add_argument(
        "--continue-after-unfilled",
        action="store_true",
        help="Cancel unfilled/held market orders and continue with the remaining paper orders.",
    )
    parser.add_argument(
        "--skip-state-snapshot",
        action="store_true",
        help="Persist orders/fills/targets but skip account, position, NAV, holding, coverage, and bar snapshot tables.",
    )
    parser.add_argument("--execute-orders", action="store_true")
    parser.add_argument("--confirm-paper-strategy-market-orders", action="store_true", dest="confirm")
    parser.add_argument(
        "--allow-repeat-target",
        action="store_true",
        help="Permit a second automatic execution attempt for the same target hash.",
    )
    parser.add_argument("--guardrails", type=Path, default=DEFAULT_GUARDRAILS)
    args = parser.parse_args()
    if not args.expected_account:
        parser.error("--expected-account is required, or set IBKR_EXPECTED_ACCOUNT / IBKR_PAPER_ACCOUNT")
    return Config(
        args.host,
        args.port,
        args.client_id,
        args.expected_account,
        args.preview,
        args.dry_run_summary,
        args.contracts,
        args.bar_gate,
        args.coverage,
        args.database,
        args.output_dir,
        args.wait_seconds,
        args.fetch_bar_history,
        args.bar_duration,
        args.bar_size,
        args.bar_scope,
        args.continue_after_unfilled,
        args.skip_state_snapshot,
        args.execute_orders,
        args.confirm,
        args.allow_repeat_target,
        args.guardrails,
    )


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    result = str(value).strip()
    if result.endswith(".0") and result[:-2].isdigit():
        return result[:-2]
    return result


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def fail(message: str, code: int = 2) -> int:
    print(f"FAIL {message}")
    return code


@contextmanager
def execution_lock(path: Path, enabled: bool):
    if not enabled:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another execution process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            timestamp_utc TEXT NOT NULL,
            account TEXT,
            mode TEXT,
            target_date TEXT,
            preview_path TEXT,
            dry_run_summary_path TEXT,
            order_count INTEGER,
            projected_margin_usd REAL,
            projected_margin_to_equity REAL,
            status TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS strategy_targets (
            run_id TEXT,
            timestamp_utc TEXT,
            instrument TEXT,
            target_contracts_1x INTEGER,
            ib_local_symbol TEXT,
            ib_exchange TEXT,
            ib_con_id INTEGER,
            qualified INTEGER,
            broker_position_contracts REAL,
            order_action TEXT,
            order_quantity INTEGER,
            signed_order_quantity INTEGER,
            estimated_initial_margin_usd REAL,
            row_status TEXT
        );

        CREATE TABLE IF NOT EXISTS orders (
            run_id TEXT,
            timestamp_utc TEXT,
            instrument TEXT,
            local_symbol TEXT,
            con_id INTEGER,
            action TEXT,
            quantity INTEGER,
            signed_order_quantity INTEGER,
            order_id INTEGER,
            perm_id INTEGER,
            order_ref TEXT,
            status TEXT,
            filled REAL,
            remaining REAL,
            avg_fill_price REAL,
            estimated_initial_margin_usd REAL
        );

        CREATE TABLE IF NOT EXISTS fills (
            run_id TEXT,
            timestamp_utc TEXT,
            instrument TEXT,
            local_symbol TEXT,
            fill_time TEXT,
            side TEXT,
            shares REAL,
            price REAL,
            exchange TEXT,
            commission REAL,
            commission_currency TEXT,
            exec_id TEXT
        );

        CREATE TABLE IF NOT EXISTS account_values (
            run_id TEXT,
            timestamp_utc TEXT,
            snapshot_label TEXT,
            account TEXT,
            tag TEXT,
            value TEXT,
            currency TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_nav (
            run_id TEXT,
            snapshot_date TEXT,
            timestamp_utc TEXT,
            account TEXT,
            net_liquidation REAL,
            total_cash_value REAL,
            init_margin_req REAL,
            maint_margin_req REAL,
            excess_liquidity REAL,
            available_funds REAL,
            buying_power REAL
        );

        CREATE TABLE IF NOT EXISTS position_snapshots (
            run_id TEXT,
            timestamp_utc TEXT,
            snapshot_label TEXT,
            account TEXT,
            instrument TEXT,
            local_symbol TEXT,
            con_id INTEGER,
            sec_type TEXT,
            exchange TEXT,
            currency TEXT,
            trading_class TEXT,
            multiplier TEXT,
            position REAL,
            avg_cost REAL
        );

        CREATE TABLE IF NOT EXISTS daily_holdings (
            run_id TEXT,
            snapshot_date TEXT,
            timestamp_utc TEXT,
            account TEXT,
            instrument TEXT,
            local_symbol TEXT,
            con_id INTEGER,
            position REAL,
            avg_cost REAL,
            last_15m_close REAL,
            last_15m_bar_time TEXT
        );

        CREATE TABLE IF NOT EXISTS ibkr_bars (
            run_id TEXT,
            request_timestamp_utc TEXT,
            instrument TEXT,
            local_symbol TEXT,
            con_id INTEGER,
            bar_size TEXT,
            duration TEXT,
            bar_time TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            average REAL,
            bar_count INTEGER
        );

        CREATE TABLE IF NOT EXISTS bar_requests (
            run_id TEXT,
            timestamp_utc TEXT,
            instrument TEXT,
            local_symbol TEXT,
            con_id INTEGER,
            bar_size TEXT,
            duration TEXT,
            status TEXT,
            bar_count INTEGER,
            first_bar_time TEXT,
            last_bar_time TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS bar_gate_snapshots (
            run_id TEXT,
            timestamp_utc TEXT,
            instrument TEXT,
            local_symbol TEXT,
            exchange TEXT,
            bar_size TEXT,
            duration TEXT,
            what_to_show TEXT,
            status TEXT,
            bar_count INTEGER,
            first_bar_time TEXT,
            last_bar_time TEXT,
            last_open REAL,
            last_high REAL,
            last_low REAL,
            last_close REAL,
            last_volume REAL,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS coverage_snapshots (
            run_id TEXT,
            timestamp_utc TEXT,
            instrument TEXT,
            ticker TEXT,
            data_source TEXT,
            data_symbol TEXT,
            ibkr_qualified INTEGER,
            source_last TEXT,
            download_rows INTEGER,
            download_first TEXT,
            download_last TEXT,
            max_gap_days REAL,
            adjusted_appended_rows INTEGER,
            multiple_appended_rows INTEGER,
            anchor_status TEXT,
            multiple_status TEXT,
            offset REAL,
            trade_ready INTEGER,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS errors (
            run_id TEXT,
            timestamp_utc TEXT,
            req_id INTEGER,
            code INTEGER,
            message TEXT
        );
        """
    )
    ensure_column(conn, "runs", "target_hash", "TEXT")
    ensure_column(conn, "runs", "target_row_count", "INTEGER")
    ensure_column(conn, "strategy_targets", "contract_expiry", "TEXT")
    ensure_column(conn, "strategy_targets", "days_to_expiry", "INTEGER")
    ensure_column(conn, "orders", "target_hash", "TEXT")
    ensure_column(conn, "orders", "target_date", "TEXT")
    ensure_column(conn, "orders", "target_position", "REAL")
    ensure_column(conn, "orders", "broker_position_before", "REAL")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS orders_order_ref_unique "
        "ON orders(order_ref) WHERE order_ref IS NOT NULL AND order_ref <> ''"
    )
    conn.commit()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def append_rows(conn: sqlite3.Connection, table: str, rows: Iterable[dict[str, object]]) -> None:
    data = list(rows)
    if not data:
        return
    pd.DataFrame(data).to_sql(table, conn, if_exists="append", index=False)


def load_preview(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing order preview: {path}")
    preview = pd.read_csv(path)
    preview["order_quantity"] = pd.to_numeric(preview["order_quantity"], errors="coerce").fillna(0).astype(int)
    preview["target_contracts_1x"] = (
        pd.to_numeric(preview["target_contracts_1x"], errors="coerce").fillna(0).astype(int)
    )
    preview["signed_order_quantity"] = (
        pd.to_numeric(preview["signed_order_quantity"], errors="coerce").fillna(0).astype(int)
    )
    return preview


def target_records(preview: pd.DataFrame) -> list[dict[str, object]]:
    fields = ["instrument", "target_contracts_1x", "ib_local_symbol", "ib_con_id"]
    rows = preview[fields].copy()
    rows["instrument"] = rows["instrument"].astype(str)
    rows["target_contracts_1x"] = pd.to_numeric(rows["target_contracts_1x"], errors="coerce").fillna(0).astype(int)
    rows["ib_local_symbol"] = rows["ib_local_symbol"].astype(str)
    rows["ib_con_id"] = pd.to_numeric(rows["ib_con_id"], errors="coerce").fillna(0).astype(int)
    return rows.sort_values(["instrument", "ib_local_symbol"]).to_dict("records")


def compute_target_hash(preview: pd.DataFrame, target_date: str) -> str:
    payload = {
        "target_date": target_date,
        "rows": target_records(preview),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def previous_execution_for_target(conn: sqlite3.Connection, target_date: str, target_hash: str) -> str:
    executed_statuses = {
        "orders_submitted",
        "orders_filled",
        "orders_filled_with_skips",
        "partial_or_failed",
        "duplicate_target_skipped",
    }
    conn.row_factory = sqlite3.Row
    runs = conn.execute(
        """
        select run_id, status, target_hash
        from runs
        where mode = 'execute_orders'
          and target_date = ?
        order by timestamp_utc desc
        limit 200
        """,
        (target_date,),
    ).fetchall()
    for run in runs:
        if str(run["status"]) not in executed_statuses:
            continue
        stored_hash = run["target_hash"]
        if stored_hash:
            if stored_hash == target_hash:
                return str(run["run_id"])
            continue
        rows = pd.read_sql_query(
            """
            select instrument, target_contracts_1x, ib_local_symbol, ib_con_id
            from strategy_targets
            where run_id = ?
            """,
            conn,
            params=(run["run_id"],),
        )
        if not rows.empty and compute_target_hash(rows, target_date) == target_hash:
            return str(run["run_id"])
    return ""


def parse_dry_run_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dry-run summary: {path}")
    body = path.read_text(encoding="utf-8")

    def match(pattern: str, default: str = "") -> str:
        found = re.search(pattern, body)
        return found.group(1).strip() if found else default

    margin_text = match(r"Projected initial margin from local schedule: \$([0-9,]+\.\d+)", "0")
    ratio_text = match(r"Projected margin/equity: ([0-9.]+)%", "nan")
    return {
        "body": body,
        "transmission_allowed": match(r"Transmission allowed: (True|False)", "False") == "True",
        "target_date": match(r"Target date: ([0-9-]+)", ""),
        "target_hash": match(r"Target hash: `([0-9a-f]{64})`", ""),
        "projected_margin_usd": float(margin_text.replace(",", "")),
        "projected_margin_to_equity": float(ratio_text) / 100.0,
    }


def load_guardrails(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def account_positions_by_con_id(ib: IB, account: str) -> dict[int, float]:
    positions: dict[int, float] = {}
    for position in ib.positions():
        if position.account != account:
            continue
        con_id = int(position.contract.conId)
        positions[con_id] = positions.get(con_id, 0.0) + float(position.position)
    return {con_id: value for con_id, value in positions.items() if value != 0.0}


def account_open_orders(ib: IB, account: str) -> list[str]:
    ib.reqAllOpenOrders()
    ib.sleep(0.5)
    rows: list[str] = []
    for trade in ib.openTrades():
        order_account = text(trade.order.account)
        if order_account and order_account != account:
            continue
        rows.append(
            f"{text(trade.contract.localSymbol)} {text(trade.order.action)} "
            f"{float(trade.orderStatus.remaining or 0.0):g} {text(trade.orderStatus.status)} "
            f"ref={text(trade.order.orderRef)}"
        )
    return rows


def reconcile_preview_with_broker(preview: pd.DataFrame, broker_positions: dict[int, float]) -> pd.DataFrame:
    reconciled = preview.copy()
    reconciled["ib_con_id"] = pd.to_numeric(reconciled["ib_con_id"], errors="coerce").fillna(0).astype(int)
    preview_con_ids = set(reconciled.loc[reconciled["ib_con_id"].gt(0), "ib_con_id"].tolist())
    orphans = {con_id: position for con_id, position in broker_positions.items() if con_id not in preview_con_ids}
    if orphans:
        raise RuntimeError(f"broker positions absent from approved preview: {orphans}")

    reconciled["broker_position_contracts"] = reconciled["ib_con_id"].map(broker_positions).fillna(0.0)
    reconciled["signed_order_quantity"] = (
        reconciled["target_contracts_1x"] - reconciled["broker_position_contracts"]
    ).round().astype(int)
    reconciled["order_quantity"] = reconciled["signed_order_quantity"].abs().astype(int)
    reconciled["order_action"] = reconciled["signed_order_quantity"].map(
        lambda value: "BUY" if value > 0 else ("SELL" if value < 0 else "")
    )
    return reconciled


def retry_block_reason(
    conn: sqlite3.Connection,
    target_hash: str,
    instrument: str,
    max_attempts: int,
    cooldown_seconds: int,
) -> str:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        select o.timestamp_utc, o.status
        from orders o
        join runs r on r.run_id = o.run_id
        where r.target_hash = ? and o.instrument = ?
        order by o.timestamp_utc desc
        """,
        (target_hash, instrument),
    ).fetchall()
    if len(rows) >= max_attempts:
        return f"{instrument}: {len(rows)} attempts reached max {max_attempts} for this target"
    if rows:
        latest = pd.to_datetime(rows[0]["timestamp_utc"], errors="coerce", utc=True)
        if pd.notna(latest):
            elapsed = datetime.now(timezone.utc) - latest.to_pydatetime()
            if elapsed < timedelta(seconds=cooldown_seconds):
                remaining = cooldown_seconds - int(elapsed.total_seconds())
                return f"{instrument}: retry cooldown has {remaining}s remaining"
    return ""


def load_contracts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing contracts: {path}")
    contracts = pd.read_csv(path).fillna("")
    contracts["con_id"] = pd.to_numeric(contracts["con_id"], errors="coerce")
    return contracts


def contract_from_row(row: pd.Series) -> Contract:
    return Contract(
        secType="FUT",
        conId=int(float(row["con_id"])),
        symbol=text(row["symbol"]),
        exchange=text(row["exchange"]),
        currency=text(row["currency"]),
        localSymbol=text(row["local_symbol"]),
        tradingClass=text(row["trading_class"]),
    )


def get_account_rows(ib: IB, account: str, run_id: str, label: str) -> list[dict[str, object]]:
    timestamp = iso_now()
    rows: list[dict[str, object]] = []
    for item in ib.accountSummary(account=account):
        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "snapshot_label": label,
                "account": item.account,
                "tag": item.tag,
                "value": item.value,
                "currency": item.currency,
            }
        )
    return rows


def account_value_map(rows: list[dict[str, object]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        if row["currency"] not in {"USD", ""}:
            continue
        try:
            values[str(row["tag"])] = float(str(row["value"]))
        except ValueError:
            pass
    return values


def get_position_rows(
    ib: IB,
    account: str,
    run_id: str,
    label: str,
    instrument_by_local_symbol: dict[str, str],
) -> list[dict[str, object]]:
    timestamp = iso_now()
    rows = []
    for position in ib.positions():
        if position.account != account:
            continue
        contract = position.contract
        local_symbol = text(contract.localSymbol)
        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "snapshot_label": label,
                "account": account,
                "instrument": instrument_by_local_symbol.get(local_symbol, ""),
                "local_symbol": local_symbol,
                "con_id": contract.conId,
                "sec_type": contract.secType,
                "exchange": contract.exchange,
                "currency": contract.currency,
                "trading_class": contract.tradingClass,
                "multiplier": contract.multiplier,
                "position": float(position.position),
                "avg_cost": float(position.avgCost),
            }
        )
    return rows


def wait_for_terminal(ib: IB, trade, wait_seconds: float) -> str:
    terminal = {"Filled", "Cancelled", "Inactive", "ApiCancelled"}
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        ib.sleep(0.25)
        status = trade.orderStatus.status
        if status in terminal:
            return status
    return trade.orderStatus.status


def order_fill_rows(run_id: str, instrument: str, local_symbol: str, trade) -> list[dict[str, object]]:
    timestamp = iso_now()
    rows = []
    for fill in trade.fills:
        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "instrument": instrument,
                "local_symbol": local_symbol,
                "fill_time": str(fill.time),
                "side": fill.execution.side,
                "shares": fill.execution.shares,
                "price": fill.execution.price,
                "exchange": fill.execution.exchange,
                "commission": getattr(fill.commissionReport, "commission", None),
                "commission_currency": getattr(fill.commissionReport, "currency", ""),
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
    order.outsideRth = True
    trade = ib.placeOrder(contract, order)
    status = wait_for_terminal(ib, trade, wait_seconds)
    return trade, status


def cancel_and_wait(ib: IB, trade, wait_seconds: float = 10.0) -> str:
    try:
        ib.cancelOrder(trade.order)
    except Exception:
        return text(trade.orderStatus.status)
    return wait_for_terminal(ib, trade, wait_seconds)


def store_daily_nav(conn: sqlite3.Connection, run_id: str, account: str, account_rows: list[dict[str, object]]) -> None:
    values = account_value_map(account_rows)
    append_rows(
        conn,
        "daily_nav",
        [
            {
                "run_id": run_id,
                "snapshot_date": date.today().isoformat(),
                "timestamp_utc": iso_now(),
                "account": account,
                "net_liquidation": values.get("NetLiquidation"),
                "total_cash_value": values.get("TotalCashValue"),
                "init_margin_req": values.get("InitMarginReq"),
                "maint_margin_req": values.get("MaintMarginReq"),
                "excess_liquidity": values.get("ExcessLiquidity"),
                "available_funds": values.get("AvailableFunds"),
                "buying_power": values.get("BuyingPower"),
            }
        ],
    )


def load_bar_gate_last_close(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    gate = pd.read_csv(path).fillna("")
    out = {}
    for row in gate.to_dict("records"):
        out[str(row.get("instrument", ""))] = row
    return out


def store_daily_holdings(
    conn: sqlite3.Connection,
    run_id: str,
    account: str,
    positions: list[dict[str, object]],
    bar_gate_by_instrument: dict[str, dict[str, object]],
) -> None:
    timestamp = iso_now()
    rows = []
    for position in positions:
        instrument = str(position.get("instrument", ""))
        gate = bar_gate_by_instrument.get(instrument, {})
        rows.append(
            {
                "run_id": run_id,
                "snapshot_date": date.today().isoformat(),
                "timestamp_utc": timestamp,
                "account": account,
                "instrument": instrument,
                "local_symbol": position.get("local_symbol", ""),
                "con_id": position.get("con_id"),
                "position": position.get("position"),
                "avg_cost": position.get("avg_cost"),
                "last_15m_close": pd.to_numeric(pd.Series([gate.get("last_close", None)]), errors="coerce").iloc[0],
                "last_15m_bar_time": gate.get("last_bar_time", ""),
            }
        )
    append_rows(conn, "daily_holdings", rows)


def store_bar_gate(conn: sqlite3.Connection, run_id: str, path: Path) -> None:
    if not path.exists():
        return
    timestamp = iso_now()
    gate = pd.read_csv(path).fillna("")
    gate.insert(0, "timestamp_utc", timestamp)
    gate.insert(0, "run_id", run_id)
    append_rows(conn, "bar_gate_snapshots", gate.to_dict("records"))


def store_coverage(conn: sqlite3.Connection, run_id: str, path: Path) -> None:
    if not path.exists():
        return
    timestamp = iso_now()
    coverage = pd.read_csv(path).fillna("")
    coverage.insert(0, "timestamp_utc", timestamp)
    coverage.insert(0, "run_id", run_id)
    append_rows(conn, "coverage_snapshots", coverage.to_dict("records"))


def fetch_and_store_bars(
    ib: IB,
    conn: sqlite3.Connection,
    run_id: str,
    contracts: pd.DataFrame,
    instruments: list[str],
    duration: str,
    bar_size: str,
) -> list[dict[str, object]]:
    timestamp = iso_now()
    contracts_by_instrument = contracts.set_index("instrument", drop=False)
    request_rows = []
    bar_rows = []
    for instrument in instruments:
        if instrument not in contracts_by_instrument.index:
            continue
        contract_row = contracts_by_instrument.loc[instrument]
        if not str(contract_row.get("status", "")).startswith("qualified"):
            continue
        contract = contract_from_row(contract_row)
        try:
            bars = list(
                ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                    keepUpToDate=False,
                    timeout=25,
                )
            )
            status = "pass" if bars else "fail_no_bars"
            error = ""
        except Exception as exc:
            bars = []
            status = "fail_exception"
            error = repr(exc)

        request_rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "instrument": instrument,
                "local_symbol": text(contract_row["local_symbol"]),
                "con_id": int(float(contract_row["con_id"])),
                "bar_size": bar_size,
                "duration": duration,
                "status": status,
                "bar_count": len(bars),
                "first_bar_time": str(bars[0].date) if bars else "",
                "last_bar_time": str(bars[-1].date) if bars else "",
                "error": error,
            }
        )
        for bar in bars:
            bar_rows.append(
                {
                    "run_id": run_id,
                    "request_timestamp_utc": timestamp,
                    "instrument": instrument,
                    "local_symbol": text(contract_row["local_symbol"]),
                    "con_id": int(float(contract_row["con_id"])),
                    "bar_size": bar_size,
                    "duration": duration,
                    "bar_time": str(bar.date),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "average": float(getattr(bar, "average", 0.0)),
                    "bar_count": int(getattr(bar, "barCount", 0)),
                }
            )
        print(f"bars {instrument}: {status} count={len(bars)}")
    append_rows(conn, "bar_requests", request_rows)
    append_rows(conn, "ibkr_bars", bar_rows)
    return request_rows


def write_run_artifacts(
    output_dir: Path,
    run_id: str,
    orders: list[dict[str, object]],
    fills: list[dict[str, object]],
    account_rows: list[dict[str, object]],
    position_rows: list[dict[str, object]],
    errors: list[dict[str, object]],
) -> None:
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orders).to_csv(run_dir / "orders.csv", index=False)
    pd.DataFrame(fills).to_csv(run_dir / "fills.csv", index=False)
    pd.DataFrame(account_rows).to_csv(run_dir / "account_values.csv", index=False)
    pd.DataFrame(position_rows).to_csv(run_dir / "positions.csv", index=False)
    pd.DataFrame(errors).to_csv(run_dir / "errors.csv", index=False)


def run_strategy(config: Config) -> int:
    if config.execute_orders and not config.confirm:
        return fail("missing --confirm-paper-strategy-market-orders")
    if config.port != 4002:
        return fail("refusing to run strategy market orders unless port is paper Gateway 4002")

    preview = load_preview(config.preview)
    dry_run = parse_dry_run_summary(config.dry_run_summary)
    target_date = str(dry_run["target_date"])
    target_hash = compute_target_hash(preview, target_date)
    target_row_count = len(target_records(preview))
    if not dry_run["target_hash"] or dry_run["target_hash"] != target_hash:
        return fail("dry-run summary target hash does not match preview")
    if config.execute_orders and not dry_run["transmission_allowed"]:
        return fail("latest dry-run summary does not allow transmission")

    contracts = load_contracts(config.contracts)
    contracts_by_instrument = contracts.set_index("instrument", drop=False)
    actionable = preview[preview["order_quantity"].gt(0)].copy()
    bad_rows = actionable[~actionable["row_status"].astype(str).eq("ready")]
    if config.execute_orders and not bad_rows.empty:
        return fail(f"actionable rows not ready: {bad_rows['instrument'].tolist()}")

    run_id = run_id_now()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.database)
    init_db(conn)
    guardrails = load_guardrails(config.guardrails)
    retry_policy = guardrails.get("execution_retry", {}) or {}
    max_attempts = int(retry_policy.get("max_attempts_per_target_instrument", 3))
    cooldown_seconds = int(retry_policy.get("cooldown_seconds", 3600))

    ib = IB()
    errors: list[dict[str, object]] = []

    def on_error(req_id, error_code, error_string, _contract):
        if error_code in {2104, 2106, 2107, 2108, 2158, 10167}:
            return
        row = {"run_id": run_id, "timestamp_utc": iso_now(), "req_id": req_id, "code": error_code, "message": error_string}
        errors.append(row)
        print(f"IB_ERROR reqId={req_id} code={error_code} msg={error_string}")

    ib.errorEvent += on_error

    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    account_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    status = "snapshot_only"
    notes: list[str] = []
    account = config.expected_account
    run_intent_persisted = False

    try:
        ib.MaxSyncedSubAccounts = 0
        ib.RequestTimeout = 15
        ib.connect(
            config.host,
            config.port,
            clientId=config.client_id,
            timeout=20,
            readonly=not config.execute_orders,
            account=config.expected_account,
        )
        accounts = ib.managedAccounts()
        if config.expected_account not in accounts:
            return fail(f"expected configured paper account, got account_count={len(accounts)}", code=3)
        if not config.expected_account.startswith("D"):
            return fail(f"refusing non-paper-looking account {mask_account(config.expected_account)}", code=3)

        instrument_by_local_symbol = {
            text(row["local_symbol"]): text(row["instrument"]) for _, row in contracts.iterrows() if text(row["local_symbol"])
        }

        if config.execute_orders:
            open_orders = account_open_orders(ib, account)
            if open_orders:
                return fail(f"broker has open orders; reconcile before execution: {open_orders}", code=3)
            preview = reconcile_preview_with_broker(preview, account_positions_by_con_id(ib, account))
            actionable = preview[preview["order_quantity"].gt(0)].copy()
            bad_rows = actionable[~actionable["row_status"].astype(str).eq("ready")]
            if not bad_rows.empty:
                return fail(f"freshly reconciled actionable rows not ready: {bad_rows['instrument'].tolist()}", code=3)
            retry_blocks = [
                retry_block_reason(
                    conn,
                    target_hash,
                    text(row.instrument),
                    max_attempts=max_attempts,
                    cooldown_seconds=cooldown_seconds,
                )
                for row in actionable.itertuples(index=False)
            ]
            retry_blocks = [reason for reason in retry_blocks if reason]
            if retry_blocks and not config.allow_repeat_target:
                return fail(f"execution retry gate blocked: {retry_blocks}", code=3)

            append_rows(
                conn,
                "runs",
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": iso_now(),
                        "account": account,
                        "mode": "execute_orders",
                        "target_date": target_date,
                        "preview_path": str(config.preview),
                        "dry_run_summary_path": str(config.dry_run_summary),
                        "order_count": int(len(actionable)),
                        "projected_margin_usd": dry_run["projected_margin_usd"],
                        "projected_margin_to_equity": dry_run["projected_margin_to_equity"],
                        "status": "execution_started",
                        "notes": "fresh broker positions reconciled before transmission",
                        "target_hash": target_hash,
                        "target_row_count": target_row_count,
                    }
                ],
            )
            targets = preview.copy()
            targets.insert(0, "timestamp_utc", iso_now())
            targets.insert(0, "run_id", run_id)
            append_rows(conn, "strategy_targets", targets.to_dict("records"))
            conn.commit()
            run_intent_persisted = True

        if not config.skip_state_snapshot:
            before_account = get_account_rows(ib, account, run_id, "before")
            before_positions = get_position_rows(ib, account, run_id, "before", instrument_by_local_symbol)
            account_rows.extend(before_account)
            position_rows.extend(before_positions)
        print(f"connected paper account={mask_account(account)} orders={len(actionable)} execute={config.execute_orders}")

        if config.execute_orders:
            status = "orders_submitted"
            for order_index, row in enumerate(actionable.itertuples(index=False), start=1):
                instrument = text(row.instrument)
                contract_row = contracts_by_instrument.loc[instrument]
                contract = contract_from_row(contract_row)
                quantity = int(row.order_quantity)
                action = text(row.order_action)
                local_symbol = text(row.ib_local_symbol)
                order_ref = f"codex_paper_rob40_{run_id}_{order_index:02d}_{instrument}"
                planned_order = {
                    "run_id": run_id,
                    "timestamp_utc": iso_now(),
                    "instrument": instrument,
                    "local_symbol": local_symbol,
                    "con_id": int(row.ib_con_id),
                    "action": action,
                    "quantity": quantity,
                    "signed_order_quantity": int(row.signed_order_quantity),
                    "order_id": None,
                    "perm_id": None,
                    "order_ref": order_ref,
                    "status": "planned",
                    "filled": 0.0,
                    "remaining": float(quantity),
                    "avg_fill_price": 0.0,
                    "estimated_initial_margin_usd": float(row.estimated_initial_margin_usd),
                    "target_hash": target_hash,
                    "target_date": target_date,
                    "target_position": float(row.target_contracts_1x),
                    "broker_position_before": float(row.broker_position_contracts),
                }
                append_rows(conn, "orders", [planned_order])
                conn.commit()
                print(f"placing {order_index}/{len(actionable)} {instrument} {action} {quantity} {local_symbol}")
                trade, terminal_status = place_market_order(
                    ib, account, contract, action, quantity, order_ref, config.wait_seconds
                )
                if terminal_status != "Filled" or float(trade.orderStatus.remaining or 0.0) != 0.0:
                    terminal_status = cancel_and_wait(ib, trade)
                final_order = {
                    **planned_order,
                    "timestamp_utc": iso_now(),
                    "order_id": trade.order.orderId,
                    "perm_id": trade.orderStatus.permId,
                    "status": terminal_status,
                    "filled": float(trade.orderStatus.filled or 0.0),
                    "remaining": float(trade.orderStatus.remaining or 0.0),
                    "avg_fill_price": float(trade.orderStatus.avgFillPrice or 0.0),
                }
                conn.execute(
                    """
                    update orders
                    set timestamp_utc = ?, order_id = ?, perm_id = ?, status = ?, filled = ?, remaining = ?, avg_fill_price = ?
                    where order_ref = ?
                    """,
                    (
                        final_order["timestamp_utc"],
                        final_order["order_id"],
                        final_order["perm_id"],
                        final_order["status"],
                        final_order["filled"],
                        final_order["remaining"],
                        final_order["avg_fill_price"],
                        order_ref,
                    ),
                )
                order_rows.append(final_order)
                new_fills = order_fill_rows(run_id, instrument, local_symbol, trade)
                fill_rows.extend(new_fills)
                append_rows(conn, "fills", new_fills)
                conn.commit()
                if terminal_status != "Filled" or float(trade.orderStatus.remaining or 0.0) != 0.0:
                    message = f"{instrument} not filled; status {terminal_status}; order cancelled."
                    notes.append(message)
                    print(f"SKIP {message}")
                    status = "partial_or_failed"
                    if not config.continue_after_unfilled:
                        notes.append(f"Stopped after {instrument}.")
                        break
            if status != "partial_or_failed":
                status = "orders_filled"

        ib.sleep(5.0)
        if not config.skip_state_snapshot:
            after_account = get_account_rows(ib, account, run_id, "after")
            after_positions = get_position_rows(ib, account, run_id, "after", instrument_by_local_symbol)
            account_rows.extend(after_account)
            position_rows.extend(after_positions)
        else:
            after_account = []
            after_positions = []

        run_notes = list(notes)
        if config.skip_state_snapshot:
            run_notes.append("state snapshot skipped")

        if run_intent_persisted:
            conn.execute(
                "update runs set timestamp_utc = ?, order_count = ?, status = ?, notes = ? where run_id = ?",
                (iso_now(), int(len(actionable)), status, " | ".join(run_notes), run_id),
            )
        else:
            append_rows(
                conn,
                "runs",
                [
                    {
                        "run_id": run_id,
                        "timestamp_utc": iso_now(),
                        "account": account,
                        "mode": "snapshot_only",
                        "target_date": dry_run["target_date"],
                        "preview_path": str(config.preview),
                        "dry_run_summary_path": str(config.dry_run_summary),
                        "order_count": 0,
                        "projected_margin_usd": dry_run["projected_margin_usd"],
                        "projected_margin_to_equity": dry_run["projected_margin_to_equity"],
                        "status": status,
                        "notes": " | ".join(run_notes),
                        "target_hash": target_hash,
                        "target_row_count": target_row_count,
                    }
                ],
            )
            targets = preview.copy()
            targets.insert(0, "timestamp_utc", iso_now())
            targets.insert(0, "run_id", run_id)
            append_rows(conn, "strategy_targets", targets.to_dict("records"))
        append_rows(conn, "errors", errors)
        if not config.skip_state_snapshot:
            append_rows(conn, "account_values", account_rows)
            append_rows(conn, "position_snapshots", position_rows)
            store_daily_nav(conn, run_id, account, after_account)
            store_bar_gate(conn, run_id, config.bar_gate)
            store_coverage(conn, run_id, config.coverage)

            bar_gate_by_instrument = load_bar_gate_last_close(config.bar_gate)
            store_daily_holdings(conn, run_id, account, after_positions, bar_gate_by_instrument)

        if config.fetch_bar_history and not config.skip_state_snapshot:
            if config.bar_scope == "actionable":
                bar_instruments = sorted(actionable["instrument"].astype(str).unique().tolist())
            else:
                bar_instruments = sorted(
                    contracts[contracts["status"].astype(str).str.startswith("qualified")]["instrument"].astype(str).tolist()
                )
            fetch_and_store_bars(
                ib,
                conn,
                run_id,
                contracts,
                bar_instruments,
                config.bar_duration,
                config.bar_size,
            )

        conn.commit()
        write_run_artifacts(config.output_dir, run_id, order_rows, fill_rows, account_rows, position_rows, errors)
        print(f"run_id={run_id} status={status}")
        print(f"database={config.database}")
        if config.skip_state_snapshot:
            print(f"orders={len(order_rows)} fills={len(fill_rows)} positions_after=skipped")
        else:
            print(f"orders={len(order_rows)} fills={len(fill_rows)} positions_after={len(after_positions)}")
        return 0 if status in {"orders_filled", "snapshot_only"} else 4
    except Exception as exc:
        if run_intent_persisted:
            conn.execute(
                "update runs set timestamp_utc = ?, status = ?, notes = ? where run_id = ?",
                (iso_now(), "execution_exception", repr(exc), run_id),
            )
            append_rows(conn, "errors", errors)
            conn.commit()
        raise
    finally:
        if ib.isConnected():
            ib.disconnect()
        conn.close()


def main() -> int:
    config = parse_args()
    lock_path = config.database.with_suffix(config.database.suffix + ".execution.lock")
    try:
        with execution_lock(lock_path, config.execute_orders):
            return run_strategy(config)
    except RuntimeError as exc:
        return fail(str(exc), code=5)


if __name__ == "__main__":
    sys.exit(main())
