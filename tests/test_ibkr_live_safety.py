from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

import pandas as pd

import ibkr_paper_strategy_runner as runner
import ibkr_strategy_order_dry_run as dry_run
import run_rob_style_backtest as backtest


def test_causal_volatility_rank_is_prefix_invariant() -> None:
    prefix = pd.Series([1.0, 3.0, 2.0, 4.0], index=pd.date_range("2020-01-01", periods=4))
    extended = pd.concat([prefix, pd.Series([1000.0], index=[pd.Timestamp("2020-01-05")])])

    expected = backtest.causal_quantile_of_points(prefix)
    actual = backtest.causal_quantile_of_points(extended).iloc[: len(prefix)]

    pd.testing.assert_series_equal(actual, expected, check_freq=False)


def test_price_alignment_does_not_carry_old_price_to_today() -> None:
    price = pd.DataFrame({"A": [100.0]}, index=[pd.Timestamp("2026-08-10")])

    aligned = backtest.align_prices_to_as_of(price, "2026-08-13", max_stale_business_days=1)

    assert aligned.loc["2026-08-11", "A"] == 100.0
    assert pd.isna(aligned.loc["2026-08-12", "A"])
    assert pd.isna(aligned.loc["2026-08-13", "A"])


def test_manifest_binds_published_positions_file(tmp_path) -> None:
    target_date = pd.Timestamp("2026-08-13")
    targets = pd.Series({"A": 1, "B": -2})
    positions_path = tmp_path / "positions_live_overlay_1x.csv"
    pd.DataFrame([targets], index=[target_date]).to_csv(positions_path)
    manifest_path = tmp_path / "target_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "published",
                "gate_pass": True,
                "strategy_date": "2026-08-13",
                "active_instruments": 2,
                "required_instruments": 2,
                "model_scale": 1.0,
                "target_hash": dry_run.source_target_hash(targets, target_date),
                "positions_file_sha256": dry_run.file_sha256(positions_path),
            }
        ),
        encoding="utf-8",
    )

    _, failures = dry_run.validate_target_manifest(
        manifest_path, positions_path, targets, target_date, model_scale=1.0
    )
    assert failures == []

    positions_path.write_text(positions_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _, failures = dry_run.validate_target_manifest(
        manifest_path, positions_path, targets, target_date, model_scale=1.0
    )
    assert any("positions_file_sha256" in failure for failure in failures)


def test_preview_blocks_orphan_position_and_expiring_target() -> None:
    expiry = (date.today() + timedelta(days=10)).strftime("%Y%m%d")
    contracts = pd.DataFrame(
        [
            {
                "instrument": "A",
                "status": "qualified_trade_candidate_found",
                "local_symbol": "AU6",
                "exchange": "TEST",
                "con_id": 101,
                "last_trade_date_or_contract_month": expiry,
            }
        ]
    )
    contracts["is_qualified"] = True
    margin = pd.DataFrame(
        [{"instrument": "A", "long_initial_usd_latest": 1000.0, "short_initial_usd_latest": 1100.0}]
    )
    broker = {
        999: {
            "con_id": 999,
            "local_symbol": "OLDU6",
            "position": -2.0,
            "exchange": "TEST",
        }
    }

    preview = dry_run.build_preview(pd.Series({"A": 1}), contracts, margin, broker, roll_window_days=14)

    status = dict(zip(preview["instrument"], preview["row_status"], strict=True))
    assert status["A"] == "blocked_contract_expiry_window"
    assert status["UNMAPPED:OLDU6"] == "blocked_orphan_broker_position"


def test_execution_reconciles_delta_from_fresh_broker_positions() -> None:
    preview = pd.DataFrame(
        [
            {
                "instrument": "A",
                "target_contracts_1x": 5,
                "ib_con_id": 101,
                "broker_position_contracts": 0,
                "signed_order_quantity": 5,
                "order_quantity": 5,
                "order_action": "BUY",
                "row_status": "ready",
            }
        ]
    )

    reconciled = runner.reconcile_preview_with_broker(preview, {101: 3.0})

    assert reconciled.loc[0, "broker_position_contracts"] == 3.0
    assert reconciled.loc[0, "signed_order_quantity"] == 2
    assert reconciled.loc[0, "order_quantity"] == 2
    assert reconciled.loc[0, "order_action"] == "BUY"


def test_retry_gate_blocks_recent_attempt() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        create table runs (run_id text, target_hash text);
        create table orders (run_id text, timestamp_utc text, instrument text, status text);
        insert into runs values ('r1', 'hash1');
        """
    )
    conn.execute(
        "insert into orders values (?, ?, ?, ?)",
        ("r1", runner.iso_now(), "A", "Cancelled"),
    )

    reason = runner.retry_block_reason(conn, "hash1", "A", max_attempts=3, cooldown_seconds=3600)

    assert "cooldown" in reason
