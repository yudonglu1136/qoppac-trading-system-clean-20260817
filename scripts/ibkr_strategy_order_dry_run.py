#!/usr/bin/env python3
"""Build a paper strategy order preview from current IBKR positions.

This script does not place orders. It reads the latest local strategy target
positions, maps them to qualified IBKR futures contracts, compares them with the
paper broker account, and writes a blocked/allowed order preview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml
from ib_insync import IB

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITIONS = ROOT / "backtests" / "rob_style_no_equity_40_margin_constrained" / "positions_unconstrained_full.csv"
DEFAULT_CONTRACTS = ROOT / "output" / "ibkr_contract_qualification" / "all_40_contracts.csv"
DEFAULT_MARGIN = ROOT / "backtests" / "rob_style_no_equity_40_margin_constrained" / "margin_schedule.csv"
DEFAULT_OUT = ROOT / "output" / "ibkr_strategy_order_dry_run"
DEFAULT_GUARDRAILS = ROOT / "config" / "ibkr_paper_live_guardrails.yaml"
DEFAULT_DATABASE = ROOT / "data" / "ibkr_paper_trading" / "ibkr_paper_trading.sqlite"


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
    positions: Path
    contracts: Path
    margin_schedule: Path
    output_dir: Path
    model_scale: float
    max_target_age_days: int
    guardrails: Path
    target_manifest: Path
    database: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="IBKR strategy order dry run; no order placement")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=91)
    parser.add_argument("--expected-account", default=default_expected_account())
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--margin-schedule", type=Path, default=DEFAULT_MARGIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--max-target-age-days", type=int, default=3)
    parser.add_argument("--guardrails", type=Path, default=DEFAULT_GUARDRAILS)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    if not args.expected_account:
        parser.error("--expected-account is required, or set IBKR_EXPECTED_ACCOUNT / IBKR_PAPER_ACCOUNT")
    target_manifest = args.target_manifest or args.positions.parent / "target_manifest.json"
    return Config(
        args.host,
        args.port,
        args.client_id,
        args.expected_account,
        args.positions,
        args.contracts,
        args.margin_schedule,
        args.output_dir,
        args.model_scale,
        args.max_target_age_days,
        args.guardrails,
        target_manifest,
        args.database,
    )


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    result = str(value).strip()
    if result.endswith(".0") and result[:-2].isdigit():
        return result[:-2]
    return result


def latest_target_positions(path: Path, model_scale: float) -> tuple[pd.Timestamp, pd.Series]:
    positions = pd.read_csv(path, index_col=0, parse_dates=True)
    latest_date = positions.index.max()
    latest = positions.loc[latest_date].fillna(0.0)
    scaled = (latest * model_scale).round().astype(int)
    return latest_date, scaled


def broker_positions_by_con_id(ib: IB, account: str) -> dict[int, dict[str, object]]:
    positions: dict[int, dict[str, object]] = {}
    for position in ib.positions():
        if position.account != account or float(position.position) == 0.0:
            continue
        con_id = int(position.contract.conId)
        local_symbol = text(position.contract.localSymbol)
        current = positions.setdefault(
            con_id,
            {
                "con_id": con_id,
                "local_symbol": local_symbol,
                "position": 0.0,
                "sec_type": text(position.contract.secType),
                "exchange": text(position.contract.exchange),
            },
        )
        current["position"] = float(str(current["position"])) + float(position.position)
    return positions


def broker_open_orders(ib: IB, account: str) -> list[dict[str, object]]:
    ib.reqAllOpenOrders()
    ib.sleep(0.5)
    rows: list[dict[str, object]] = []
    for trade in ib.openTrades():
        order_account = text(trade.order.account)
        if order_account and order_account != account:
            continue
        rows.append(
            {
                "con_id": int(trade.contract.conId),
                "local_symbol": text(trade.contract.localSymbol),
                "action": text(trade.order.action),
                "quantity": float(trade.order.totalQuantity or 0.0),
                "remaining": float(trade.orderStatus.remaining or 0.0),
                "status": text(trade.orderStatus.status),
                "order_ref": text(trade.order.orderRef),
            }
        )
    return rows


def get_account_values(ib: IB, account: str) -> dict[str, float]:
    values = {}
    for item in ib.accountSummary(account=account):
        if item.tag in {"NetLiquidation", "ExcessLiquidity", "InitMarginReq", "MaintMarginReq", "AvailableFunds"}:
            try:
                values[item.tag] = float(item.value)
            except ValueError:
                pass
    return values


def load_contracts(path: Path) -> pd.DataFrame:
    contracts = pd.read_csv(path)
    contracts["is_qualified"] = contracts["status"].astype(str).str.startswith("qualified")
    return contracts


def load_margin(path: Path) -> pd.DataFrame:
    margin = pd.read_csv(path)
    for column in ["long_initial_usd_latest", "short_initial_usd_latest"]:
        margin[column] = pd.to_numeric(margin[column], errors="coerce")
    return margin


def load_guardrails(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_target_hash(latest_targets: pd.Series, target_date: pd.Timestamp) -> str:
    rows = [
        {"instrument": str(instrument), "buffered_integer_target_1x": int(target)}
        for instrument, target in latest_targets.sort_index().items()
    ]
    payload = {"target_date": str(target_date.date()), "rows": rows}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_target_manifest(
    path: Path,
    positions_path: Path,
    latest_targets: pd.Series,
    target_date: pd.Timestamp,
    model_scale: float,
) -> tuple[dict[str, object], list[str]]:
    if not path.exists():
        return {}, [f"Target manifest is missing: {path}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"Target manifest cannot be read: {exc}"]

    failures: list[str] = []
    if manifest.get("status") != "published":
        failures.append(f"Manifest status is {manifest.get('status')!r}, expected 'published'.")
    if manifest.get("gate_pass") is not True:
        failures.append("Manifest gate_pass is not true.")
    if str(manifest.get("strategy_date", "")) != str(target_date.date()):
        failures.append(
            f"Manifest strategy_date {manifest.get('strategy_date')!r} does not match target {target_date.date()}."
        )
    active = int(manifest.get("active_instruments", 0) or 0)
    required = int(manifest.get("required_instruments", 0) or 0)
    if active < required or active != len(latest_targets):
        failures.append(
            f"Manifest instrument counts are inconsistent: active={active}, required={required}, target_columns={len(latest_targets)}."
        )
    try:
        manifest_scale = float(manifest.get("model_scale"))
    except (TypeError, ValueError):
        manifest_scale = float("nan")
    if pd.isna(manifest_scale) or abs(manifest_scale - model_scale) > 1e-12:
        failures.append(f"Manifest model_scale {manifest.get('model_scale')!r} does not match {model_scale:g}.")

    expected_target_hash = source_target_hash(latest_targets, target_date)
    if str(manifest.get("target_hash", "")) != expected_target_hash:
        failures.append("Manifest target_hash does not match the target matrix.")
    expected_file_hash = str(manifest.get("positions_file_sha256", ""))
    if not expected_file_hash:
        failures.append("Manifest does not contain positions_file_sha256.")
    elif expected_file_hash != file_sha256(positions_path):
        failures.append("Manifest positions_file_sha256 does not match the positions file.")
    return manifest, failures


def business_day_age(target_date: pd.Timestamp, today: date) -> int:
    target = pd.Timestamp(target_date).normalize()
    current = pd.Timestamp(today).normalize()
    if target > current:
        return -max(1, len(pd.bdate_range(current, target)) - 1)
    return max(0, len(pd.bdate_range(target, current)) - 1)


def contract_expiry(value: object) -> date | None:
    digits = "".join(character for character in text(value) if character.isdigit())
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date()
    except ValueError:
        return None


def nav_loss_blocks(database: Path, current_nav: float, guardrails: dict) -> list[str]:
    triggers = guardrails.get("manual_review_triggers", {}) or {}
    daily_limit = triggers.get("daily_loss_to_netliq")
    five_day_limit = triggers.get("five_day_loss_to_netliq")
    if not database.exists() or not pd.notna(current_nav) or current_nav <= 0:
        return []
    try:
        with sqlite3.connect(database) as conn:
            nav = pd.read_sql_query(
                "select snapshot_date, timestamp_utc, net_liquidation from daily_nav where net_liquidation is not null order by timestamp_utc",
                conn,
                parse_dates=["timestamp_utc"],
            )
    except (sqlite3.Error, ValueError):
        return []
    if nav.empty:
        return []
    nav["snapshot_date"] = nav["snapshot_date"].astype(str)
    first_by_day = nav.groupby("snapshot_date", sort=True).first(numeric_only=False)
    blocks: list[str] = []
    today_key = date.today().isoformat()
    if daily_limit is not None and today_key in first_by_day.index:
        start_nav = float(first_by_day.loc[today_key, "net_liquidation"])
        daily_return = current_nav / start_nav - 1.0 if start_nav > 0 else 0.0
        if daily_return <= float(daily_limit):
            blocks.append(f"Daily NAV return {daily_return:.2%} breached {float(daily_limit):.2%}.")
    if five_day_limit is not None:
        recent = first_by_day.tail(5)
        if not recent.empty:
            start_nav = float(recent.iloc[0]["net_liquidation"])
            five_day_return = current_nav / start_nav - 1.0 if start_nav > 0 else 0.0
            if five_day_return <= float(five_day_limit):
                blocks.append(f"Five-day NAV return {five_day_return:.2%} breached {float(five_day_limit):.2%}.")
    return blocks


def resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_bar_gate(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path).fillna("")
    data["instrument"] = data["instrument"].astype(str)
    return data


def scoped_instruments_for_bar_gate(preview: pd.DataFrame, scope: str) -> list[str]:
    if scope == "actionable_orders":
        scoped = preview[preview["order_quantity"].gt(0)]
    elif scope == "nonzero_targets":
        scoped = preview[preview["target_contracts_1x"].ne(0)]
    else:
        raise ValueError(f"Unknown paper market-order override gate_scope: {scope}")
    return sorted(scoped["instrument"].astype(str).unique().tolist())


def parse_gate_time(value: object) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(str(value), errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_convert(None)


def check_15min_bar_gate(
    gate: pd.DataFrame,
    instruments: list[str],
    required_bar_size: str,
    min_bar_count: int,
    max_age_days: int,
) -> tuple[bool, list[str], list[str]]:
    if gate.empty:
        return False, [], ["15-minute historical bar gate file is missing or empty."]
    by_instrument = gate.set_index("instrument", drop=False)
    passed: list[str] = []
    failures: list[str] = []
    today = pd.Timestamp(date.today()).normalize()
    oldest_allowed = today - pd.Timedelta(days=max_age_days)

    for instrument in instruments:
        if instrument not in by_instrument.index:
            failures.append(f"{instrument}: missing from 15-minute gate file")
            continue
        row = by_instrument.loc[instrument]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        status = str(row.get("status", "")).lower()
        if status != "pass":
            failures.append(f"{instrument}: gate status {row.get('status', '')}")
            continue
        if required_bar_size and str(row.get("bar_size", "")) != required_bar_size:
            failures.append(f"{instrument}: bar size {row.get('bar_size', '')}, expected {required_bar_size}")
            continue
        bar_count = pd.to_numeric(pd.Series([row.get("bar_count", 0)]), errors="coerce").iloc[0]
        if pd.isna(bar_count) or int(bar_count) < min_bar_count:
            failures.append(f"{instrument}: bar_count {row.get('bar_count', '')}, expected >= {min_bar_count}")
            continue
        last_bar = parse_gate_time(row.get("last_bar_time", ""))
        if last_bar is None:
            failures.append(f"{instrument}: missing last_bar_time")
            continue
        if last_bar.normalize() < oldest_allowed:
            failures.append(
                f"{instrument}: last 15-minute bar {last_bar} older than {max_age_days} calendar days"
            )
            continue
        passed.append(instrument)
    return not failures, passed, failures


def margin_for_target(target: int, row: pd.Series | None) -> float:
    if row is None or target == 0:
        return 0.0
    column = "long_initial_usd_latest" if target > 0 else "short_initial_usd_latest"
    value = row.get(column)
    return abs(float(target)) * float(value) if pd.notna(value) else float("nan")


def build_preview(
    latest_targets: pd.Series,
    contracts: pd.DataFrame,
    margin: pd.DataFrame,
    broker_positions: dict[int, dict[str, object]],
    roll_window_days: int = 14,
    today: date | None = None,
) -> pd.DataFrame:
    today = today or date.today()
    contracts_by_instrument = contracts.set_index("instrument", drop=False)
    margin_by_instrument = margin.set_index("instrument", drop=False)
    contract_instrument_by_con_id = {
        int(float(row["con_id"])): str(row["instrument"])
        for _, row in contracts.iterrows()
        if pd.notna(row.get("con_id")) and str(row.get("con_id", "")).strip()
    }
    rows: list[dict[str, object]] = []
    matched_con_ids: set[int] = set()
    for instrument, target in latest_targets.sort_index().items():
        contract_row = contracts_by_instrument.loc[instrument] if instrument in contracts_by_instrument.index else None
        margin_row = margin_by_instrument.loc[instrument] if instrument in margin_by_instrument.index else None
        is_qualified = bool(contract_row is not None and contract_row["is_qualified"])
        local_symbol = text(contract_row["local_symbol"]) if contract_row is not None else ""
        con_id = int(float(str(contract_row["con_id"]))) if is_qualified and contract_row is not None else 0
        broker_record = broker_positions.get(con_id, {}) if con_id else {}
        broker_position = float(str(broker_record.get("position", 0.0)))
        if con_id and con_id in broker_positions:
            matched_con_ids.add(con_id)
        order_quantity = int(target - broker_position)
        if order_quantity > 0:
            action = "BUY"
        elif order_quantity < 0:
            action = "SELL"
        else:
            action = ""
        row_status = "ready" if is_qualified else "blocked_unqualified"
        if int(target) != 0 and not is_qualified:
            row_status = "blocked_unqualified_nonzero_target"
        estimated_margin = margin_for_target(int(target), margin_row)
        if int(target) != 0 and pd.isna(estimated_margin):
            row_status = "blocked_missing_margin"
        expiry = contract_expiry(contract_row.get("last_trade_date_or_contract_month")) if contract_row is not None else None
        days_to_expiry = (expiry - today).days if expiry is not None else None
        if int(target) != 0 and days_to_expiry is not None and days_to_expiry <= roll_window_days:
            row_status = "blocked_contract_expiry_window"
        rows.append(
            {
                "instrument": instrument,
                "target_contracts_1x": int(target),
                "ib_local_symbol": local_symbol,
                "ib_exchange": text(contract_row["exchange"]) if contract_row is not None else "",
                "ib_con_id": con_id if is_qualified else "",
                "qualified": is_qualified,
                "broker_position_contracts": broker_position,
                "order_action": action,
                "order_quantity": abs(order_quantity),
                "signed_order_quantity": order_quantity,
                "estimated_initial_margin_usd": estimated_margin,
                "contract_expiry": expiry.isoformat() if expiry is not None else "",
                "days_to_expiry": days_to_expiry,
                "row_status": row_status,
            }
        )

    for con_id, broker_record in sorted(broker_positions.items()):
        if con_id in matched_con_ids or float(str(broker_record.get("position", 0.0))) == 0.0:
            continue
        local_symbol = text(broker_record.get("local_symbol", ""))
        mapped_instrument = contract_instrument_by_con_id.get(con_id, "")
        instrument = mapped_instrument or f"UNMAPPED:{local_symbol or con_id}"
        broker_position = float(str(broker_record["position"]))
        signed_order_quantity = int(-broker_position)
        rows.append(
            {
                "instrument": instrument,
                "target_contracts_1x": 0,
                "ib_local_symbol": local_symbol,
                "ib_exchange": text(broker_record.get("exchange", "")),
                "ib_con_id": con_id,
                "qualified": bool(mapped_instrument),
                "broker_position_contracts": broker_position,
                "order_action": "BUY" if signed_order_quantity > 0 else "SELL",
                "order_quantity": abs(signed_order_quantity),
                "signed_order_quantity": signed_order_quantity,
                "estimated_initial_margin_usd": float("nan"),
                "contract_expiry": "",
                "days_to_expiry": None,
                "row_status": "blocked_orphan_broker_position",
            }
        )
    return pd.DataFrame(rows)


def target_records(preview: pd.DataFrame) -> list[dict[str, object]]:
    fields = ["instrument", "target_contracts_1x", "ib_local_symbol", "ib_con_id"]
    rows = preview[fields].copy()
    rows["instrument"] = rows["instrument"].astype(str)
    rows["target_contracts_1x"] = pd.to_numeric(rows["target_contracts_1x"], errors="coerce").fillna(0).astype(int)
    rows["ib_local_symbol"] = rows["ib_local_symbol"].astype(str)
    rows["ib_con_id"] = pd.to_numeric(rows["ib_con_id"], errors="coerce").fillna(0).astype(int)
    return rows.sort_values(["instrument", "ib_local_symbol"]).to_dict("records")


def target_hash(preview: pd.DataFrame, target_date: pd.Timestamp) -> str:
    payload = {
        "target_date": str(target_date.date()),
        "rows": target_records(preview),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    latest_date, latest_targets = latest_target_positions(config.positions, config.model_scale)
    age_days = business_day_age(latest_date, date.today())
    contracts = load_contracts(config.contracts)
    margin = load_margin(config.margin_schedule)
    guardrails = load_guardrails(config.guardrails)
    manifest, manifest_failures = validate_target_manifest(
        config.target_manifest,
        config.positions,
        latest_targets,
        latest_date,
        config.model_scale,
    )
    contract_guards = guardrails.get("contract_guards", {}) or {}
    roll_window_days = int(contract_guards.get("roll_window_calendar_days", 14))

    ib = IB()
    ib.MaxSyncedSubAccounts = 0
    try:
        ib.connect(
            config.host,
            config.port,
            clientId=config.client_id,
            timeout=15,
            readonly=True,
            account=config.expected_account,
        )
        accounts = ib.managedAccounts()
        if config.expected_account not in accounts:
            raise RuntimeError(f"Expected configured account, got account_count={len(accounts)}")
        account_values = get_account_values(ib, config.expected_account)
        broker_positions = broker_positions_by_con_id(ib, config.expected_account)
        open_orders = broker_open_orders(ib, config.expected_account)
    finally:
        if ib.isConnected():
            ib.disconnect()

    preview = build_preview(
        latest_targets,
        contracts,
        margin,
        broker_positions,
        roll_window_days=roll_window_days,
    )
    hash_value = target_hash(preview, latest_date)
    actionable = preview[preview["order_quantity"].gt(0)].copy()
    nonzero_unqualified = preview[preview["row_status"].eq("blocked_unqualified_nonzero_target")]
    blocking_preview_rows = preview[
        ~preview["row_status"].astype(str).eq("ready")
        & (
            preview["target_contracts_1x"].ne(0)
            | preview["broker_position_contracts"].ne(0)
            | preview["order_quantity"].gt(0)
        )
    ]
    future_target_block = age_days < 0
    stale_block = age_days > config.max_target_age_days
    unqualified_block = not nonzero_unqualified.empty
    net_liq = account_values.get("NetLiquidation", float("nan"))
    projected_margin = preview["estimated_initial_margin_usd"].sum(min_count=1)
    projected_margin_to_equity = projected_margin / net_liq if net_liq and pd.notna(net_liq) else float("nan")
    guardrail_blocks: list[str] = []
    guardrail_warnings: list[str] = []
    guardrail_blocks.extend(manifest_failures)
    if not blocking_preview_rows.empty:
        blocked = ", ".join(
            f"{row.instrument}={row.row_status}" for row in blocking_preview_rows.itertuples(index=False)
        )
        guardrail_blocks.append(f"Position/contract reconciliation is blocked: {blocked}.")
    if open_orders:
        details = ", ".join(
            f"{row['local_symbol']} {row['action']} {row['remaining']} ({row['status']})" for row in open_orders[:8]
        )
        guardrail_blocks.append(f"Broker has open orders that must be reconciled first: {details}.")
    paper_override = guardrails.get("paper_market_order_override", {}) or {}
    paper_override_enabled = bool(paper_override.get("enabled"))
    paper_override_active = False
    paper_bar_gate_ok = False
    paper_bar_gate_file = resolve_repo_path(
        paper_override.get(
            "historical_bar_gate_file",
            guardrails.get("market_data_gate", {}).get("latest_15min_historical_gate_file", ""),
        )
    )
    paper_gate_scope = str(paper_override.get("gate_scope", "nonzero_targets"))
    paper_gate_instruments: list[str] = []
    paper_gate_passed: list[str] = []
    paper_gate_failures: list[str] = []

    if paper_override_enabled:
        paper_port = int(paper_override.get("paper_port", 4002))
        paper_override_active = config.port == paper_port
        if paper_override_active:
            paper_gate_instruments = scoped_instruments_for_bar_gate(preview, paper_gate_scope)
            gate = load_bar_gate(paper_bar_gate_file)
            paper_bar_gate_ok, paper_gate_passed, paper_gate_failures = check_15min_bar_gate(
                gate,
                paper_gate_instruments,
                str(paper_override.get("required_bar_size", "15 mins")),
                int(paper_override.get("min_bar_count", 1)),
                int(paper_override.get("max_last_bar_age_calendar_days", 2)),
            )
        else:
            guardrail_warnings.append(
                f"Paper market-order override configured but inactive on port {config.port}; expected paper port {paper_port}."
            )

    margin_guards = guardrails.get("margin_guards", {})
    hard_margin = margin_guards.get("hard_block_projected_margin_to_equity")
    if hard_margin is not None and pd.notna(projected_margin_to_equity) and projected_margin_to_equity > float(hard_margin):
        guardrail_blocks.append(
            f"Projected margin/equity {projected_margin_to_equity:.2%} exceeds hard block {float(hard_margin):.2%}."
        )
    stressed_multiplier = margin_guards.get("hard_block_stressed_margin_multiplier")
    stressed_limit = margin_guards.get("hard_block_stressed_margin_to_equity")
    if (
        stressed_multiplier is not None
        and stressed_limit is not None
        and pd.notna(projected_margin_to_equity)
        and projected_margin_to_equity * float(stressed_multiplier) > float(stressed_limit)
    ):
        guardrail_blocks.append(
            f"Stressed margin/equity {projected_margin_to_equity * float(stressed_multiplier):.2%} exceeds hard block {float(stressed_limit):.2%}."
        )
    warning_margin = margin_guards.get("warning_projected_margin_to_equity")
    if warning_margin is not None and pd.notna(projected_margin_to_equity) and projected_margin_to_equity > float(warning_margin):
        guardrail_warnings.append(
            f"Projected margin/equity {projected_margin_to_equity:.2%} exceeds warning level {float(warning_margin):.2%}."
        )
    single_margin_limit = margin_guards.get("max_single_instrument_margin_to_equity_initial_live")
    if single_margin_limit is not None and pd.notna(net_liq) and net_liq > 0:
        single_margin = pd.to_numeric(preview["estimated_initial_margin_usd"], errors="coerce") / net_liq
        breaches = preview.loc[single_margin.gt(float(single_margin_limit)), "instrument"].astype(str).tolist()
        if breaches:
            guardrail_blocks.append(
                f"Single-instrument projected margin exceeds {float(single_margin_limit):.2%}: {', '.join(breaches)}."
            )

    actual_margin = account_values.get("InitMarginReq", float("nan"))
    margin_difference_limit = (guardrails.get("manual_review_triggers", {}) or {}).get(
        "broker_vs_local_margin_difference"
    )
    if (
        margin_difference_limit is not None
        and pd.notna(actual_margin)
        and pd.notna(projected_margin)
        and max(actual_margin, projected_margin) > 0
    ):
        margin_difference = abs(actual_margin - projected_margin) / max(actual_margin, projected_margin)
        if margin_difference > float(margin_difference_limit):
            guardrail_blocks.append(
                f"Broker/local margin mismatch {margin_difference:.2%} exceeds {float(margin_difference_limit):.2%} "
                f"(broker ${actual_margin:,.0f}, projected ${projected_margin:,.0f})."
            )
    guardrail_blocks.extend(nav_loss_blocks(config.database, net_liq, guardrails))

    max_contract_file_age_hours = contract_guards.get("max_contract_file_age_hours")
    if max_contract_file_age_hours is not None:
        age_hours = (datetime.now().timestamp() - config.contracts.stat().st_mtime) / 3600.0
        if age_hours > float(max_contract_file_age_hours):
            guardrail_blocks.append(
                f"Contract qualification file is {age_hours:.1f}h old; maximum is {float(max_contract_file_age_hours):.1f}h."
            )

    disabled = set(map(str, guardrails.get("disabled_until_verified", []) or []))
    disabled_nonzero = preview[preview["instrument"].astype(str).isin(disabled) & preview["target_contracts_1x"].ne(0)]
    if not disabled_nonzero.empty:
        names = ", ".join(disabled_nonzero["instrument"].astype(str).tolist())
        disabled_allowed = (
            paper_override_active
            and paper_bar_gate_ok
            and bool(paper_override.get("allow_disabled_until_verified_if_15min_gate_passes"))
        )
        if disabled_allowed:
            guardrail_warnings.append(
                f"Paper override allows disabled-until-verified instruments after 15-minute gate pass: {names}."
            )
        else:
            guardrail_blocks.append(f"Nonzero target in disabled-until-verified instruments: {names}.")

    execution_blocks = guardrails.get("execution_blocks", {})
    market_gate = guardrails.get("market_data_gate", {})
    live_status = str(market_gate.get("live_subscription_gate_status", "")).lower()
    live_data_bypassed_for_paper = (
        paper_override_active
        and paper_bar_gate_ok
        and bool(paper_override.get("bypass_live_subscription_gate_if_15min_gate_passes"))
    )
    if execution_blocks.get("block_order_transmission_until_live_market_data_gate_passes") and live_status != "pass":
        if live_data_bypassed_for_paper:
            guardrail_warnings.append(
                "Paper override bypasses live market-data subscription block because 15-minute historical gate passed."
            )
        else:
            guardrail_blocks.append(
                f"Live market-data subscription gate is `{market_gate.get('live_subscription_gate_status')}`; order transmission is blocked."
            )
    if execution_blocks.get("require_market_data_subscription_or_disable_instrument") and live_status != "pass":
        if live_data_bypassed_for_paper:
            guardrail_warnings.append("Paper override treats the 15-minute historical gate as the market-data check.")
        else:
            guardrail_blocks.append("Guardrail requires live market-data subscription or explicit per-instrument disable.")

    if paper_override_active and not paper_bar_gate_ok:
        failed_preview = "; ".join(paper_gate_failures[:8])
        if len(paper_gate_failures) > 8:
            failed_preview += f"; ... +{len(paper_gate_failures) - 8} more"
        guardrail_blocks.append(f"Paper 15-minute historical bar gate failed: {failed_preview}")

    transmission_allowed = (
        not future_target_block
        and not stale_block
        and not unqualified_block
        and not guardrail_blocks
    )

    preview_path = config.output_dir / "latest_1x_order_preview.csv"
    summary_path = config.output_dir / "latest_1x_order_summary.md"
    account_path = config.output_dir / "latest_account_values.csv"
    open_orders_path = config.output_dir / "latest_open_orders.csv"
    preview.to_csv(preview_path, index=False)
    pd.DataFrame([account_values]).to_csv(account_path, index=False)
    pd.DataFrame(open_orders).to_csv(open_orders_path, index=False)

    lines = [
        "# IBKR Strategy Order Dry Run",
        "",
        f"- Target source: `{config.positions}`",
        f"- Target date: {latest_date.date()}",
        f"- Target hash: `{hash_value}`",
        f"- Target age trading days: {age_days}",
        f"- Target manifest: `{config.target_manifest}`",
        f"- Manifest status: `{manifest.get('status', 'missing')}`",
        f"- Model scale: {config.model_scale:g}x",
        f"- Account: `{mask_account(config.expected_account)}`",
        f"- NetLiquidation: ${net_liq:,.2f}" if pd.notna(net_liq) else "- NetLiquidation: unavailable",
        f"- Projected initial margin from local schedule: ${projected_margin:,.2f}",
        f"- Projected margin/equity: {projected_margin_to_equity:.2%}",
        f"- Actionable order rows: {len(actionable)}",
        f"- Broker open orders: {len(open_orders)}",
        f"- Reconciliation-blocked rows: {len(blocking_preview_rows)}",
        f"- Nonzero unqualified targets: {len(nonzero_unqualified)}",
        f"- Transmission allowed: {transmission_allowed}",
        "",
        "## Paper 15m Market-Order Override",
        "",
        f"- Override enabled: {paper_override_enabled}",
        f"- Override active for this run: {paper_override_active}",
        f"- Gate file: `{paper_bar_gate_file}`",
        f"- Gate scope: {paper_gate_scope}",
        f"- Instruments checked: {len(paper_gate_instruments)}",
        f"- Instruments passed: {len(paper_gate_passed)}",
        f"- Gate pass: {paper_bar_gate_ok}",
        "",
        "## Gate",
        "",
    ]
    if stale_block:
        lines.append(
            f"- FAIL: target positions are stale. Max allowed age is {config.max_target_age_days} days; actual age is {age_days} days."
        )
    if future_target_block:
        lines.append(
            f"- FAIL: target positions are future-dated versus local date {date.today().isoformat()}; target date is {latest_date.date()}."
        )
    if unqualified_block:
        lines.append("- FAIL: at least one nonzero target lacks a qualified IBKR contract.")
    for item in guardrail_blocks:
        lines.append(f"- FAIL: {item}")
    if not future_target_block and not stale_block and not unqualified_block:
        if guardrail_blocks:
            lines.append("- BLOCKED: base dry-run checks passed, but guardrails blocked transmission.")
        else:
            lines.append("- PASS: order preview is eligible for a separate explicit transmission step.")
    if guardrail_warnings:
        lines.extend(["", "## Warnings", ""])
        for item in guardrail_warnings:
            lines.append(f"- WARNING: {item}")
    lines.extend(
        [
            "",
            "## Nonzero Orders Preview",
            "",
            actionable[
                [
                    "instrument",
                    "target_contracts_1x",
                    "ib_local_symbol",
                    "order_action",
                    "order_quantity",
                    "estimated_initial_margin_usd",
                    "row_status",
                ]
            ].to_markdown(index=False),
            "",
            "## Nonzero Unqualified Targets",
            "",
            nonzero_unqualified[
                ["instrument", "target_contracts_1x", "order_action", "order_quantity", "row_status"]
            ].to_markdown(index=False)
            if not nonzero_unqualified.empty
            else "None.",
            "",
            "## Files",
            "",
            f"- `{preview_path}`",
            f"- `{account_path}`",
            f"- `{open_orders_path}`",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"target_date={latest_date.date()} age_days={age_days}")
    print(f"target_hash={hash_value}")
    print(f"net_liq={net_liq} projected_margin={projected_margin} margin_to_equity={projected_margin_to_equity}")
    print(f"actionable_orders={len(actionable)} nonzero_unqualified={len(nonzero_unqualified)}")
    print(f"transmission_allowed={transmission_allowed}")
    print(f"wrote={preview_path}")
    print(f"wrote={summary_path}")
    return 0 if transmission_allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
