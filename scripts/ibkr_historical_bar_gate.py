#!/usr/bin/env python3
"""Check IBKR historical intraday bar availability for qualified futures.

This script requests historical bars only. It does not place orders.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
try:
    from ib_async import Contract, IB
except ModuleNotFoundError:
    from ib_insync import Contract, IB

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT / "output" / "ibkr_contract_qualification" / "sample_contracts.csv"
DEFAULT_OUT = ROOT / "output" / "ibkr_market_data_gate" / "sample_15min_historical.csv"
DEFAULT_DATABASE = ROOT / "data" / "ibkr_market_data" / "ibkr_market_data.sqlite"

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
    database: Path
    bar_size: str
    duration: str
    what_to_show: str
    timeout: float
    instruments: list[str]
    all_instruments: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Check IBKR historical bar availability")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=71)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--bar-size", default="15 mins")
    parser.add_argument("--duration", default="2 D")
    parser.add_argument("--what-to-show", default="TRADES")
    parser.add_argument("--timeout", type=float, default=20.0)
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
        args.database,
        args.bar_size,
        args.duration,
        args.what_to_show,
        args.timeout,
        instruments,
        args.all_instruments,
    )


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    result = str(value).strip()
    if result.endswith(".0") and result[:-2].isdigit():
        return result[:-2]
    return result


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


def load_contracts(config: Config) -> pd.DataFrame:
    if not config.contracts.exists():
        raise FileNotFoundError(f"Missing contract file: {config.contracts}")
    contracts = pd.read_csv(config.contracts)
    contracts = contracts[contracts["status"].astype(str).str.startswith("qualified")].copy()
    if not config.all_instruments:
        contracts = contracts[contracts["instrument"].isin(config.instruments)].copy()
    return contracts


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def bar_time_text(value: object) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def initialise_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        create table if not exists market_data_runs (
            run_id text primary key,
            retrieved_at_utc text not null,
            status text not null,
            requested_contracts integer not null,
            passed_contracts integer not null,
            failed_contracts integer not null,
            contracts_path text not null,
            output_path text not null,
            bar_size text not null,
            duration text not null,
            what_to_show text not null
        );

        create table if not exists futures_15m_bars (
            instrument text not null,
            con_id integer not null,
            local_symbol text not null,
            exchange text not null,
            bar_size text not null,
            what_to_show text not null,
            bar_time text not null,
            open real,
            high real,
            low real,
            close real,
            volume real,
            average real,
            source_bar_count integer,
            retrieved_at_utc text not null,
            run_id text not null,
            primary key (instrument, con_id, bar_size, what_to_show, bar_time)
        );

        create table if not exists market_data_instrument_status (
            instrument text not null,
            con_id integer not null,
            local_symbol text not null,
            exchange text not null,
            bar_size text not null,
            what_to_show text not null,
            status text not null,
            bar_count integer not null,
            first_bar_time text,
            last_bar_time text,
            error text,
            retrieved_at_utc text not null,
            run_id text not null,
            primary key (instrument, con_id, bar_size, what_to_show)
        );

        create index if not exists idx_futures_15m_bars_time
            on futures_15m_bars (bar_time);
        create index if not exists idx_futures_15m_bars_instrument_time
            on futures_15m_bars (instrument, bar_time);
        """
    )


def persist_run(
    database: Path,
    run_id: str,
    retrieved_at_utc: str,
    config: Config,
    rows: list[dict[str, object]],
    bar_rows: list[dict[str, object]],
) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        initialise_database(connection)
        connection.executemany(
            """
            insert into futures_15m_bars (
                instrument, con_id, local_symbol, exchange, bar_size,
                what_to_show, bar_time, open, high, low, close, volume,
                average, source_bar_count, retrieved_at_utc, run_id
            ) values (
                :instrument, :con_id, :local_symbol, :exchange, :bar_size,
                :what_to_show, :bar_time, :open, :high, :low, :close, :volume,
                :average, :source_bar_count, :retrieved_at_utc, :run_id
            )
            on conflict (instrument, con_id, bar_size, what_to_show, bar_time)
            do update set
                local_symbol = excluded.local_symbol,
                exchange = excluded.exchange,
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                average = excluded.average,
                source_bar_count = excluded.source_bar_count,
                retrieved_at_utc = excluded.retrieved_at_utc,
                run_id = excluded.run_id
            """,
            bar_rows,
        )
        status_rows = [
            {
                **row,
                "con_id": int(row["con_id"]),
                "retrieved_at_utc": retrieved_at_utc,
                "run_id": run_id,
            }
            for row in rows
        ]
        connection.executemany(
            """
            insert into market_data_instrument_status (
                instrument, con_id, local_symbol, exchange, bar_size,
                what_to_show, status, bar_count, first_bar_time, last_bar_time,
                error, retrieved_at_utc, run_id
            ) values (
                :instrument, :con_id, :local_symbol, :exchange, :bar_size,
                :what_to_show, :status, :bar_count, :first_bar_time,
                :last_bar_time, :error, :retrieved_at_utc, :run_id
            )
            on conflict (instrument, con_id, bar_size, what_to_show)
            do update set
                local_symbol = excluded.local_symbol,
                exchange = excluded.exchange,
                status = excluded.status,
                bar_count = excluded.bar_count,
                first_bar_time = excluded.first_bar_time,
                last_bar_time = excluded.last_bar_time,
                error = excluded.error,
                retrieved_at_utc = excluded.retrieved_at_utc,
                run_id = excluded.run_id
            """,
            status_rows,
        )
        passed = sum(row["status"] == "pass" for row in rows)
        statuses = {str(row["status"]) for row in rows}
        if rows and statuses == {"fail_connection"}:
            run_status = "connection_failed"
        elif rows and passed == len(rows):
            run_status = "pass"
        else:
            run_status = "partial_or_failed"
        connection.execute(
            """
            insert into market_data_runs values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                retrieved_at_utc,
                run_status,
                len(rows),
                passed,
                len(rows) - passed,
                str(config.contracts),
                str(config.output),
                config.bar_size,
                config.duration,
                config.what_to_show,
            ),
        )


def connection_failure_rows(
    contracts: pd.DataFrame, config: Config, error: str
) -> list[dict[str, object]]:
    return [
        {
            "instrument": text(row["instrument"]),
            "con_id": int(float(row["con_id"])),
            "local_symbol": text(row["local_symbol"]),
            "exchange": text(row["exchange"]),
            "bar_size": config.bar_size,
            "duration": config.duration,
            "what_to_show": config.what_to_show,
            "status": "fail_connection",
            "bar_count": 0,
            "first_bar_time": "",
            "last_bar_time": "",
            "last_open": "",
            "last_high": "",
            "last_low": "",
            "last_close": "",
            "last_volume": "",
            "error": error,
        }
        for _, row in contracts.iterrows()
    ]


def main() -> int:
    config = parse_args()
    contracts = load_contracts(config)
    if contracts.empty:
        print("No qualified contracts selected.")
        return 2

    ib = IB()

    def on_error(req_id, error_code, error_string, _contract):
        if error_code not in {2104, 2106, 2107, 2158}:
            print(f"IB_ERROR reqId={req_id} code={error_code} msg={error_string}")

    ib.errorEvent += on_error
    rows = []
    bar_rows = []
    retrieved_at_utc = iso_utc_now()
    run_id = retrieved_at_utc.replace(":", "").replace("+00:00", "Z")
    try:
        try:
            ib.client.connect(
                config.host, config.port, clientId=config.client_id, timeout=10
            )
        except Exception as exc:
            rows = connection_failure_rows(contracts, config, repr(exc))
            persist_run(
                config.database,
                run_id,
                retrieved_at_utc,
                config,
                rows,
                bar_rows,
            )
            print(f"connection_failed={exc!r}")
            print(f"database={config.database} status=fail_connection")
            return 3
        accounts = ib.client.getAccounts()
        print(f"connected server_version={ib.client.serverVersion()} account_count={len(accounts)}")
        for _, row in contracts.iterrows():
            instrument = text(row["instrument"])
            contract = contract_from_row(row)
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=config.duration,
                    barSizeSetting=config.bar_size,
                    whatToShow=config.what_to_show,
                    useRTH=False,
                    formatDate=2,
                    keepUpToDate=False,
                    timeout=config.timeout,
                )
                result = {
                    "instrument": instrument,
                    "con_id": int(float(row["con_id"])),
                    "local_symbol": text(row["local_symbol"]),
                    "exchange": text(row["exchange"]),
                    "bar_size": config.bar_size,
                    "duration": config.duration,
                    "what_to_show": config.what_to_show,
                    "status": "pass" if bars else "fail_no_bars",
                    "bar_count": len(bars),
                    "first_bar_time": bar_time_text(bars[0].date) if bars else "",
                    "last_bar_time": bar_time_text(bars[-1].date) if bars else "",
                    "last_open": bars[-1].open if bars else "",
                    "last_high": bars[-1].high if bars else "",
                    "last_low": bars[-1].low if bars else "",
                    "last_close": bars[-1].close if bars else "",
                    "last_volume": bars[-1].volume if bars else "",
                    "error": "",
                }
                for bar in bars:
                    bar_rows.append(
                        {
                            "instrument": instrument,
                            "con_id": int(float(row["con_id"])),
                            "local_symbol": text(row["local_symbol"]),
                            "exchange": text(row["exchange"]),
                            "bar_size": config.bar_size,
                            "what_to_show": config.what_to_show,
                            "bar_time": bar_time_text(bar.date),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                            "average": float(getattr(bar, "average", 0.0)),
                            "source_bar_count": int(getattr(bar, "barCount", 0)),
                            "retrieved_at_utc": retrieved_at_utc,
                            "run_id": run_id,
                        }
                    )
            except Exception as exc:
                result = {
                    "instrument": instrument,
                    "con_id": int(float(row["con_id"])),
                    "local_symbol": text(row["local_symbol"]),
                    "exchange": text(row["exchange"]),
                    "bar_size": config.bar_size,
                    "duration": config.duration,
                    "what_to_show": config.what_to_show,
                    "status": "fail_exception",
                    "bar_count": 0,
                    "first_bar_time": "",
                    "last_bar_time": "",
                    "last_open": "",
                    "last_high": "",
                    "last_low": "",
                    "last_close": "",
                    "last_volume": "",
                    "error": repr(exc),
                }
            rows.append(result)
            print(
                f"{result['instrument']}: {result['status']} bars={result['bar_count']} "
                f"last={result['last_bar_time']} close={result['last_close']}"
            )

        output = config.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output_tmp = output.with_suffix(output.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(output_tmp, index=False)
        output_tmp.replace(output)
        persist_run(config.database, run_id, retrieved_at_utc, config, rows, bar_rows)
        print(f"wrote={output}")
        print(f"database={config.database} bars_upserted={len(bar_rows)}")
        return 0 if rows and all(row["status"] == "pass" for row in rows) else 1
    finally:
        if ib.client.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
