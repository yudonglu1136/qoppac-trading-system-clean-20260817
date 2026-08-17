from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ibkr_historical_bar_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ibkr_historical_bar_gate", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_persist_run_upserts_overlapping_bars(tmp_path):
    module = load_module()
    database = tmp_path / "bars.sqlite"
    config = module.Config(
        host="127.0.0.1",
        port=4002,
        client_id=1701,
        contracts=tmp_path / "contracts.csv",
        output=tmp_path / "latest.csv",
        database=database,
        bar_size="15 mins",
        duration="2 D",
        what_to_show="TRADES",
        timeout=20.0,
        instruments=[],
        all_instruments=True,
    )
    status_rows = [
        {
            "instrument": "BUND",
            "con_id": 123,
            "local_symbol": "FGBLU6",
            "exchange": "EUREX",
            "bar_size": "15 mins",
            "duration": "2 D",
            "what_to_show": "TRADES",
            "status": "pass",
            "bar_count": 1,
            "first_bar_time": "2026-08-14T12:00:00+00:00",
            "last_bar_time": "2026-08-14T12:00:00+00:00",
            "last_open": 125.0,
            "last_high": 125.1,
            "last_low": 124.9,
            "last_close": 125.0,
            "last_volume": 10,
            "error": "",
        }
    ]
    bar = {
        "instrument": "BUND",
        "con_id": 123,
        "local_symbol": "FGBLU6",
        "exchange": "EUREX",
        "bar_size": "15 mins",
        "what_to_show": "TRADES",
        "bar_time": "2026-08-14T12:00:00+00:00",
        "open": 125.0,
        "high": 125.1,
        "low": 124.9,
        "close": 125.0,
        "volume": 10.0,
        "average": 125.0,
        "source_bar_count": 2,
        "retrieved_at_utc": "2026-08-14T12:05:00+00:00",
        "run_id": "run-1",
    }

    module.persist_run(
        database,
        "run-1",
        "2026-08-14T12:05:00+00:00",
        config,
        status_rows,
        [bar],
    )
    module.persist_run(
        database,
        "run-2",
        "2026-08-14T13:05:00+00:00",
        config,
        status_rows,
        [{**bar, "close": 125.2, "run_id": "run-2"}],
    )

    with sqlite3.connect(database) as connection:
        assert connection.execute("select count(*) from futures_15m_bars").fetchone()[0] == 1
        assert connection.execute("select close from futures_15m_bars").fetchone()[0] == 125.2
        assert connection.execute("select count(*) from market_data_runs").fetchone()[0] == 2
        assert connection.execute("select count(*) from market_data_instrument_status").fetchone()[0] == 1
