#!/usr/bin/env python3
"""Export the local IBKR paper trading state for the monitor prototype."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "ibkr_paper_trading" / "ibkr_paper_trading.sqlite"
OUT = ROOT / "ibkr-monitor-prototype" / "public" / "monitor-data.json"
TARGET_SUMMARY = ROOT / "backtests" / "rob_style_no_equity_live_overlay" / "latest_target_summary.csv"
FORECASTS = ROOT / "backtests" / "rob_style_no_equity_live_overlay" / "combined_forecast_live_overlay.csv"
POSITIONS = ROOT / "backtests" / "rob_style_no_equity_live_overlay" / "positions_live_overlay_1x.csv"
LATEST_BAR_GATE = ROOT / "output" / "ibkr_market_data_gate" / "all_40_15min_historical.csv"
LATEST_ACCOUNT_VALUES = ROOT / "output" / "ibkr_strategy_order_dry_run" / "latest_account_values.csv"
LATEST_OPEN_ORDERS = ROOT / "output" / "ibkr_strategy_order_dry_run" / "latest_open_orders.csv"
ADJUSTED_DIR = ROOT / "output" / "live_futures_overlay" / "adjusted_prices_csv"
FX_DIR = ROOT / "output" / "live_futures_overlay" / "fx_prices_csv"
NATIVE_PAPER_DIR = ROOT / "github" / "pysystemtrade" / "private" / "paper_15m"
RUNTIME_GUARDRAIL_DIR = (
    Path.home()
    / "Library"
    / "Application Support"
    / "pysystemtrade-paper"
    / "private"
    / "data"
    / "guardrails"
)
NATIVE_GUARDRAIL_DIR = (
    RUNTIME_GUARDRAIL_DIR
    if RUNTIME_GUARDRAIL_DIR.exists()
    else NATIVE_PAPER_DIR / "data" / "guardrails"
)
NATIVE_GATE_STATE = NATIVE_GUARDRAIL_DIR / "trading_gate.json"
NATIVE_TARGET = NATIVE_GUARDRAIL_DIR / "rob_no_equity_38_targets.json"


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_latest_csv_map(path: Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        if row.get(key):
            out[str(row[key])] = row
    return out


def file_timestamp_utc(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat()


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    raw = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timestamp_age_seconds(value: Any) -> float:
    parsed = parse_timestamp(value)
    if parsed is None:
        return math.inf
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def latest_price_from_csv(path: Path) -> float | None:
    rows = read_csv_rows(path)
    for row in reversed(rows):
        value = to_float(row.get("PRICE", row.get("price")), math.nan)
        if not math.isnan(value):
            return value
    return None


def latest_fx_rates() -> dict[str, float]:
    rates = {"USD": 1.0}
    if not FX_DIR.exists():
        return rates
    for path in FX_DIR.glob("*USD.csv"):
        currency = path.stem.replace("USD", "")
        price = latest_price_from_csv(path)
        if price and price > 0:
            rates[currency] = price
    return rates


def latest_account_values() -> dict[str, Any]:
    rows = read_csv_rows(LATEST_ACCOUNT_VALUES)
    return rows[-1] if rows else {}


def latest_bar_gate() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = read_csv_rows(LATEST_BAR_GATE)
    latest: dict[str, dict[str, Any]] = {}
    pass_count = 0
    max_bar_time = ""
    max_bar_datetime: datetime | None = None
    for row in rows:
        instrument = str(row.get("instrument") or "")
        if not instrument:
            continue
        status = str(row.get("status") or "")
        close = to_float(row.get("last_close"), math.nan)
        bar_time = str(row.get("last_bar_time") or "")
        if status == "pass" and not math.isnan(close):
            pass_count += 1
            latest[instrument] = {
                "last_close": close,
                "last_bar_time": bar_time,
                "status": status,
            }
            parsed_bar_time = parse_timestamp(bar_time)
            if parsed_bar_time is not None and (max_bar_datetime is None or parsed_bar_time > max_bar_datetime):
                max_bar_time = bar_time
                max_bar_datetime = parsed_bar_time
    return latest, {
        "fileTimestampUtc": file_timestamp_utc(LATEST_BAR_GATE),
        "rows": len(rows),
        "passCount": pass_count,
        "maxBarTime": max_bar_time,
    }


def connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10)


def table_rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    return [{key: clean(row[key]) for key in row.keys()} for row in rows]


def latest_run_id(conn: sqlite3.Connection, table: str) -> str:
    row = conn.execute(f"select run_id from {table} order by timestamp_utc desc limit 1").fetchone()
    return str(row[0]) if row else ""


def latest_snapshot_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("select run_id from daily_nav order by timestamp_utc desc limit 1").fetchone()
    return str(row[0]) if row else ""


def load_daily_series(
    path: Path,
    instrument: str,
    fills: list[dict[str, Any]],
    lookback: int = 130,
) -> list[dict[str, Any]]:
    price_rows = read_csv_rows(ADJUSTED_DIR / f"{instrument}.csv")
    forecast_rows = read_csv_rows(FORECASTS)
    position_rows = read_csv_rows(POSITIONS)

    forecast_by_date = {row.get("DATETIME", "")[:10]: to_float(row.get(instrument), math.nan) for row in forecast_rows}
    position_by_date = {row.get("DATETIME", "")[:10]: to_float(row.get(instrument), 0.0) for row in position_rows}

    fill_action_by_date: dict[str, str] = {}
    for fill in reversed(fills):
        if str(fill.get("instrument") or "") != instrument:
            continue
        fill_date = str(fill.get("fill_time") or fill.get("timestamp_utc") or "")[:10]
        side = str(fill.get("side") or "").upper()
        if fill_date and side:
            fill_action_by_date[fill_date] = "LONG" if side in {"BOT", "BUY"} else "SHORT"

    series: list[dict[str, Any]] = []
    for row in price_rows[-lookback:]:
        date = row.get("DATETIME", "")[:10]
        price = to_float(row.get("PRICE", row.get("price")), math.nan)
        if not date or math.isnan(price):
            continue
        position = position_by_date.get(date, 0.0)
        forecast = forecast_by_date.get(date, math.nan)
        series.append(
            {
                "date": date,
                "price": clean(price),
                "forecast": clean(forecast),
                "position": clean(position),
                "action": fill_action_by_date.get(date, ""),
            }
        )
    return series


def append_latest_gate_point(
    series: list[dict[str, Any]],
    instrument: str,
    gate_by_instrument: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gate = gate_by_instrument.get(instrument, {})
    bar_time = str(gate.get("last_bar_time") or "")
    price = to_float(gate.get("last_close"), math.nan)
    if not bar_time or math.isnan(price):
        return series
    if series and str(series[-1].get("date")) == bar_time:
        series[-1]["price"] = clean(price)
        series[-1]["source"] = "latest_15m_gate"
        return series
    out = list(series)
    last_forecast = out[-1].get("forecast") if out else None
    last_position = out[-1].get("position") if out else None
    out.append(
        {
            "date": bar_time,
            "price": clean(price),
            "forecast": last_forecast,
            "position": last_position,
            "action": "",
            "source": "latest_15m_gate",
        }
    )
    return out


def build_payload() -> dict[str, Any]:
    if not DB.exists():
        raise FileNotFoundError(DB)

    fx = latest_fx_rates()
    target_summary = read_latest_csv_map(TARGET_SUMMARY, "instrument")
    latest_gate_csv, gate_meta = latest_bar_gate()
    latest_account = latest_account_values()

    with connect() as conn:
        snapshot_run = latest_snapshot_run_id(conn)
        latest_run = latest_run_id(conn, "runs")
        nav_rows = table_rows(
            conn,
            """
            select * from daily_nav
            order by timestamp_utc
            """,
        )
        latest_nav = nav_rows[-1] if nav_rows else {}
        latest_run_rows = table_rows(conn, "select * from runs where run_id = ?", (latest_run,))
        latest_run_row = latest_run_rows[0] if latest_run_rows else {}

        holdings = table_rows(
            conn,
            """
            select h.*, p.currency, p.multiplier, p.trading_class, p.exchange
            from daily_holdings h
            left join position_snapshots p
              on p.run_id = h.run_id
             and p.instrument = h.instrument
             and p.snapshot_label = 'after'
            where h.run_id = ?
            order by abs(h.position) desc, h.instrument
            """,
            (snapshot_run,),
        )

        targets = table_rows(
            conn,
            "select * from strategy_targets where run_id = ? order by instrument",
            (latest_run,),
        )
        latest_orders = table_rows(
            conn,
            """
            select * from orders
            order by timestamp_utc desc, rowid desc
            limit 20
            """,
        )
        latest_fills = table_rows(
            conn,
            """
            select * from fills
            order by timestamp_utc desc, rowid desc
            limit 500
            """,
        )
        latest_gate = table_rows(
            conn,
            """
            select instrument, last_close, last_bar_time
            from bar_gate_snapshots
            where run_id = ?
            """,
            (snapshot_run,),
        )

    target_by_instrument = {str(row["instrument"]): row for row in targets}
    snapshot_gate_by_instrument = {str(row["instrument"]): row for row in latest_gate}
    gate_by_instrument = {**snapshot_gate_by_instrument, **latest_gate_csv}

    positions: list[dict[str, Any]] = []
    seen_instruments: set[str] = set()
    has_broker_snapshot = len(holdings) > 0
    for row in holdings:
        instrument = str(row.get("instrument") or "")
        seen_instruments.add(instrument)
        position = to_float(row.get("position"))
        gate = latest_gate_csv.get(instrument, {})
        gate_price = to_float(gate.get("last_close"), math.nan)
        last_price = gate_price if not math.isnan(gate_price) else to_float(row.get("last_15m_close"))
        price_source = "latest_15m_gate" if not math.isnan(gate_price) else "state_snapshot"
        summary = target_summary.get(instrument, {})
        if last_price == 0:
            last_price = to_float(summary.get("latest_price"))
            price_source = "target_summary"
        multiplier = to_float(row.get("multiplier"), 1.0) or 1.0
        currency = str(row.get("currency") or "USD")
        fx_rate = fx.get(currency, 1.0)
        avg_cost = to_float(row.get("avg_cost"))
        estimated_pnl = (last_price * multiplier - avg_cost) * position * fx_rate
        target = target_by_instrument.get(instrument, {})
        forecast = to_float(summary.get("combined_forecast"), to_float(target.get("target_contracts_1x")))
        target_contracts = to_float(summary.get("buffered_integer_target_1x"), to_float(target.get("target_contracts_1x")))
        direction = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"
        positions.append(
            {
                "instrument": instrument,
                "localSymbol": row.get("local_symbol"),
                "position": clean(position),
                "target": clean(target_contracts),
                "direction": direction,
                "lastPrice": clean(last_price),
                "avgEntry": clean(avg_cost / multiplier if multiplier else avg_cost),
                "unrealizedPnl": clean(estimated_pnl),
                "unrealizedPctNav": clean(estimated_pnl / to_float(latest_nav.get("net_liquidation"), 1.0)),
                "forecast": clean(forecast),
                "margin": clean(to_float(target.get("estimated_initial_margin_usd"))),
                "currency": currency,
                "lastBarTime": gate.get("last_bar_time") or row.get("last_15m_bar_time"),
                "priceSource": price_source,
            }
        )

    for instrument, target in target_by_instrument.items():
        target_contracts = to_float(target.get("target_contracts_1x"))
        broker_position = 0.0 if has_broker_snapshot else to_float(target.get("broker_position_contracts"))
        if instrument in seen_instruments or (target_contracts == 0 and broker_position == 0):
            continue
        summary = target_summary.get(instrument, {})
        gate = gate_by_instrument.get(instrument, {})
        last_price = to_float(gate.get("last_close"), to_float(summary.get("latest_price")))
        direction_source = broker_position if broker_position != 0 else target_contracts
        direction = "LONG" if direction_source > 0 else "SHORT" if direction_source < 0 else "FLAT"
        forecast = to_float(summary.get("combined_forecast"), target_contracts)
        positions.append(
            {
                "instrument": instrument,
                "localSymbol": target.get("ib_local_symbol"),
                "position": clean(broker_position),
                "target": clean(target_contracts),
                "direction": direction,
                "lastPrice": clean(last_price),
                "avgEntry": clean(last_price),
                "unrealizedPnl": 0.0,
                "unrealizedPctNav": 0.0,
                "forecast": clean(forecast),
                "margin": clean(to_float(target.get("estimated_initial_margin_usd"))),
                "currency": "USD",
                "lastBarTime": gate.get("last_bar_time", ""),
                "priceSource": "latest_15m_gate" if instrument in latest_gate_csv else "state_snapshot",
            }
        )

    positions.sort(
        key=lambda item: (
            abs(to_float(item.get("target")) - to_float(item.get("position"))) > 0,
            abs(to_float(item.get("unrealizedPnl"))),
            abs(to_float(item.get("position"))),
        ),
        reverse=True,
    )
    nonzero_positions = [p for p in positions if to_float(p["position"]) != 0]
    selected = "BTP3" if any(p["instrument"] == "BTP3" for p in positions) else (positions[0]["instrument"] if positions else "")
    chart_instruments = [p["instrument"] for p in nonzero_positions[:12]]
    for must in [selected, "KR3", "SOYBEAN_mini", "AUD", "MXP", "VIX", "COPPER-micro", "GAS_US_mini"]:
        if must and must not in chart_instruments:
            chart_instruments.append(must)

    series_by_instrument = {
        instrument: append_latest_gate_point(
            load_daily_series(ADJUSTED_DIR, instrument, latest_fills),
            instrument,
            latest_gate_csv,
        )
        for instrument in chart_instruments
    }

    nav_series = [
        {
            "timestamp": row.get("timestamp_utc"),
            "nav": row.get("net_liquidation"),
            "availableFunds": row.get("available_funds"),
            "margin": row.get("init_margin_req"),
        }
        for row in nav_rows
    ]
    account_file_time = parse_timestamp(file_timestamp_utc(LATEST_ACCOUNT_VALUES))
    latest_nav_time = parse_timestamp(latest_nav.get("timestamp_utc"))
    account_file_is_newer = bool(
        account_file_time is not None
        and (latest_nav_time is None or account_file_time >= latest_nav_time)
    )
    current_account = latest_account if account_file_is_newer else latest_nav
    latest_nav_value = to_float(
        current_account.get("NetLiquidation", current_account.get("net_liquidation")),
        to_float(latest_nav.get("net_liquidation")),
    )
    today_key = date.today().isoformat()
    current_year = today_key[:4]
    today_nav_rows = [row for row in nav_rows if str(row.get("snapshot_date")) == today_key]
    ytd_nav_rows = [row for row in nav_rows if str(row.get("snapshot_date", "")).startswith(current_year)]
    first_today = to_float(
        today_nav_rows[0].get("net_liquidation") if today_nav_rows else latest_nav_value,
        latest_nav_value,
    )
    first_ytd = to_float(
        ytd_nav_rows[0].get("net_liquidation") if ytd_nav_rows else latest_nav_value,
        latest_nav_value,
    )
    daily_pnl = latest_nav_value - to_float(first_today)
    ytd_pnl = latest_nav_value - first_ytd
    total_unrealized = sum(to_float(p.get("unrealizedPnl")) for p in nonzero_positions)
    margin_used = to_float(
        current_account.get("InitMarginReq", current_account.get("init_margin_req")),
        to_float(latest_nav.get("init_margin_req")),
    )
    available_funds = to_float(
        current_account.get("AvailableFunds", current_account.get("available_funds")),
        to_float(latest_nav.get("available_funds")),
    )
    excess_liquidity = to_float(
        current_account.get("ExcessLiquidity", current_account.get("excess_liquidity")),
        to_float(latest_nav.get("excess_liquidity")),
    )

    guard_state = read_json(NATIVE_GATE_STATE)
    native_target = read_json(NATIVE_TARGET)
    gate_generated_at = guard_state.get("generatedAtUtc")
    gate_fresh = timestamp_age_seconds(gate_generated_at) <= 180
    account_values_fresh = timestamp_age_seconds(file_timestamp_utc(LATEST_ACCOUNT_VALUES)) <= 5400
    snapshot_fresh = timestamp_age_seconds(latest_nav.get("timestamp_utc")) <= 5400
    daemon_phase = str(guard_state.get("evaluationSource", "guardrail_missing"))
    connection_healthy = gate_fresh and bool(guard_state.get("ibkr", {}).get("connected"))

    data_audit = guard_state.get("data", {})
    production_data = data_audit.get("productionUniverse", {})
    research_data = data_audit.get("researchUniverse", {})
    continuous_data = data_audit.get("continuousContracts", {})
    target_state = guard_state.get("target", {})
    launch_state = guard_state.get("launchAgents", {})
    intent_state = guard_state.get("intentLog", {})
    gate_reasons = list(guard_state.get("reasons", []))
    native_instruments = data_audit.get("instruments", [])
    native_dates = [
        row.get("latestAdjustedDate")
        for row in native_instruments
        if row.get("latestAdjustedDate")
    ]
    latest_native_price = max(native_dates) if native_dates else None

    native_targets = native_target.get("targets", [])
    remaining_actionable = sum(
        1 for row in native_targets if abs(to_float(row.get("trade"))) > 0
    )
    if not native_targets:
        remaining_actionable = sum(
            1 for row in targets if to_float(row.get("order_quantity")) > 0
        )
    open_orders = len(read_csv_rows(LATEST_OPEN_ORDERS))
    utc_today = datetime.now(timezone.utc).date().isoformat()
    filled_today = sum(
        to_float(row.get("shares"))
        for row in latest_fills
        if str(row.get("timestamp_utc") or row.get("fill_time") or "")[:10] == utc_today
    )

    return {
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "account": latest_nav.get("account") or latest_run_row.get("account"),
        "strategy": "RC Trend Following 38 No-Equity 1x",
        "mode": "IBKR Paper",
        "selectedInstrument": selected,
        "summary": {
            "nav": clean(latest_nav_value),
            "dailyPnl": clean(daily_pnl),
            "dailyPnlPct": clean(daily_pnl / first_today if first_today else 0),
            "ytdPnl": clean(ytd_pnl),
            "ytdPnlPct": clean(ytd_pnl / first_ytd if first_ytd else 0),
            "unrealizedPnlEstimate": clean(total_unrealized),
            "marginUsed": clean(margin_used),
            "marginUsedPct": clean(margin_used / latest_nav_value if latest_nav_value else 0),
            "availableFunds": clean(available_funds),
            "buyingPower": latest_nav.get("buying_power"),
            "excessLiquidity": clean(excess_liquidity),
            "projectedMargin": latest_run_row.get("projected_margin_usd"),
            "projectedMarginPct": latest_run_row.get("projected_margin_to_equity"),
        },
        "status": {
            "ibkrConnection": "Connected" if connection_healthy else "Disconnected",
            "accountStatus": "Paper",
            "daemonPhase": daemon_phase,
            "daemonHeartbeatFresh": gate_fresh,
            "accountValuesFresh": account_values_fresh,
            "snapshotFresh": snapshot_fresh,
            "overlayHealthy": bool(data_audit.get("hardPass")),
            "contractsHealthy": not bool(continuous_data.get("activeTailFailureCount")),
            "daemonHeartbeatUtc": gate_generated_at,
            "lastSnapshotUtc": latest_nav.get("timestamp_utc"),
            "latestAccountValuesUtc": file_timestamp_utc(LATEST_ACCOUNT_VALUES),
            "latestBarGateUtc": gate_meta["fileTimestampUtc"],
            "latestBarGateMaxBarTime": gate_meta["maxBarTime"],
            "latestBarGatePassCount": gate_meta["passCount"],
            "latestNativePriceUtc": latest_native_price,
            "tradingCheckCadence": "15 min",
            "stateSnapshotCadence": "1 h",
            "remainingActionableOrders": remaining_actionable,
            "openOrders": open_orders,
            "filledToday": clean(filled_today),
            "latestRunStatus": latest_run_row.get("status"),
            "tradingGate": guard_state.get("status", "BLOCKED"),
            "gateReasons": gate_reasons or (["GATE_STATE_MISSING"] if not guard_state else []),
            "gateStateFresh": gate_fresh,
            "gateFreshness": "Fresh" if gate_fresh else "Stale",
            "accountValuesFreshness": "Fresh" if account_values_fresh else "Stale",
            "fullSnapshotFreshness": "Fresh" if snapshot_fresh else "Stale",
            "nativeDataGate": "PASS" if data_audit.get("hardPass") else "BLOCKED",
            "businessDate": guard_state.get("businessDate"),
            "productionFiles": f"{production_data.get('filesComplete', 0)}/{production_data.get('expected', 38)}",
            "productionTradable": f"{production_data.get('tradablePass', 0)}/{production_data.get('tradable', 0)}",
            "frozenInstruments": production_data.get("frozen", 0),
            "researchFiles": f"{research_data.get('filesComplete', 0)}/{research_data.get('expected', 40)}",
            "researchCurrent": f"{research_data.get('current', 0)}/{research_data.get('expected', 40)}",
            "continuousGapFailures": continuous_data.get("activeTailFailureCount", 0),
            "targetSha256": target_state.get("sha256"),
            "targetSha256Verified": bool(target_state.get("sha256Verified")),
            "targetBusinessDate": target_state.get("businessDate"),
            "targetBusinessDateVerified": bool(target_state.get("businessDateVerified")),
            "targetAgeHours": target_state.get("ageHours"),
            "targetUniverseCount": target_state.get("universeCount"),
            "launchAgentsLoaded": launch_state.get(
                "running", launch_state.get("loaded", 0)
            ),
            "launchAgentsExpected": launch_state.get("expected", 7),
            "autoRecoveryConfigured": launch_state.get("autoRecoveryConfigured", 0),
            "intentRunCount": intent_state.get("runCount", 0),
            "intentCount": intent_state.get("intentCount", 0),
            "lastIntentUtc": intent_state.get("lastIntentUtc"),
        },
        "navSeries": nav_series,
        "positions": positions,
        "orders": latest_orders,
        "fills": latest_fills,
        "series": series_by_instrument,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export monitor data JSON")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if args.stdout:
        print(body)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
