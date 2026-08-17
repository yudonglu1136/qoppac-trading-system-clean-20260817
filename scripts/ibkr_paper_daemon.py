#!/usr/bin/env python3
"""Run the IBKR paper strategy workflow continuously with fail-closed gates."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output" / "ibkr_paper_daemon"
DEFAULT_CONTRACTS = ROOT / "output" / "ibkr_contract_qualification" / "all_40_contracts.csv"
DEFAULT_POSITIONS = ROOT / "backtests" / "rob_style_no_equity_live_overlay" / "positions_live_overlay_1x.csv"
DEFAULT_BAR_GATE = ROOT / "output" / "ibkr_market_data_gate" / "all_40_15min_historical.csv"
DEFAULT_DB = ROOT / "data" / "ibkr_paper_trading" / "ibkr_paper_trading.sqlite"
DEFAULT_COVERAGE = ROOT / "output" / "live_futures_overlay" / "coverage.csv"


def default_expected_account() -> str:
    return os.environ.get("IBKR_EXPECTED_ACCOUNT") or os.environ.get("IBKR_PAPER_ACCOUNT", "")


@dataclass(frozen=True)
class Config:
    output_dir: Path
    duration_hours: float
    interval_seconds: int
    trade_interval_seconds: int
    state_snapshot_interval_seconds: int
    overlay_interval_seconds: int
    full_bar_interval_seconds: int
    client_id_base: int
    expected_account: str
    enable_trading: bool
    confirm: bool
    run_once: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run the IBKR paper strategy daemon")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--duration-hours", type=float, default=0.0, help="0 runs until stopped")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--trade-interval-seconds", type=int, default=900)
    parser.add_argument("--state-snapshot-interval-seconds", type=int, default=3600)
    parser.add_argument("--overlay-interval-seconds", type=int, default=3600)
    parser.add_argument("--full-bar-interval-seconds", type=int, default=3600)
    parser.add_argument("--client-id-base", type=int, default=300)
    parser.add_argument("--expected-account", default=default_expected_account())
    parser.add_argument("--enable-trading", action="store_true")
    parser.add_argument("--confirm-paper-24h-daemon", action="store_true", dest="confirm")
    parser.add_argument("--once", action="store_true", dest="run_once")
    args = parser.parse_args()
    if not args.expected_account:
        parser.error("--expected-account is required, or set IBKR_EXPECTED_ACCOUNT / IBKR_PAPER_ACCOUNT")
    return Config(
        args.output_dir,
        args.duration_hours,
        args.interval_seconds,
        args.trade_interval_seconds,
        args.state_snapshot_interval_seconds,
        args.overlay_interval_seconds,
        args.full_bar_interval_seconds,
        args.client_id_base,
        args.expected_account,
        args.enable_trading,
        args.confirm,
        args.run_once,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    tmp.replace(path)


def existing_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.log_path = config.output_dir / "daemon.jsonl"
        self.heartbeat_path = config.output_dir / "heartbeat.json"
        self.stop_path = config.output_dir / "STOP"
        self.pid_path = config.output_dir / "daemon.pid"
        self.lock_path = config.output_dir / "daemon.lock"
        self.command_counter = 0
        self.stop_requested = False
        self.last_overlay = existing_mtime(DEFAULT_POSITIONS)
        self.last_overlay_attempt = existing_mtime(DEFAULT_COVERAGE)
        self.last_contract_refresh = existing_mtime(DEFAULT_CONTRACTS)
        self.last_contract_attempt = 0.0
        self.last_trade = 0.0
        self.last_state_snapshot = existing_mtime(DEFAULT_DB)
        self.last_full_bars = existing_mtime(DEFAULT_BAR_GATE)
        self.overlay_healthy = self.published_target_is_valid()
        self.contracts_healthy = (
            self.last_contract_refresh > 0 and time.time() - self.last_contract_refresh <= 24 * 3600
        )
        self.lock_handle: TextIO | None = None
        self.last_command_stdout: dict[str, str] = {}

    def next_client_id(self) -> int:
        self.command_counter += 1
        return self.config.client_id_base + (self.command_counter % 10)

    def published_target_is_valid(self) -> bool:
        manifest_path = DEFAULT_POSITIONS.parent / "target_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("status") == "published" and manifest.get("gate_pass") is True

    def acquire_lock(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_handle.close()
            self.lock_handle = None
            raise RuntimeError(f"another daemon holds {self.lock_path}") from exc

    def release_lock(self) -> None:
        if self.lock_handle is None:
            return
        fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        self.lock_handle.close()
        self.lock_handle = None

    def log(self, event: str, **payload: object) -> None:
        append_jsonl(self.log_path, {"timestamp_utc": utc_now(), "event": event, **payload})

    def heartbeat(self, **payload: object) -> None:
        write_json(
            self.heartbeat_path,
            {
                "timestamp_utc": utc_now(),
                "pid": os.getpid(),
                "stop_file": str(self.stop_path),
                "enable_trading": self.config.enable_trading,
                **payload,
            },
        )

    def command(self, name: str, args: list[str], timeout: int = 600, allow_fail: bool = False) -> int:
        started = time.time()
        self.log("command_start", name=name, args=args, timeout=timeout)
        try:
            proc = subprocess.run(
                [sys.executable, "-u", *args],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - started
            self.log(
                "command_timeout",
                name=name,
                elapsed_seconds=round(elapsed, 2),
                timeout=timeout,
                stdout_tail=(exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                stderr_tail=(exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            )
            if allow_fail:
                self.last_command_stdout[name] = ""
                return 124
            raise
        elapsed = time.time() - started
        self.last_command_stdout[name] = proc.stdout
        self.log(
            "command_finish",
            name=name,
            returncode=proc.returncode,
            elapsed_seconds=round(elapsed, 2),
            stdout_tail=proc.stdout[-4000:],
            stderr_tail=proc.stderr[-4000:],
        )
        if proc.returncode != 0 and not allow_fail:
            raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
        return proc.returncode

    def run_overlay_refresh(self) -> bool:
        client_id = self.next_client_id()
        started = time.time()
        self.last_overlay_attempt = started
        build_rc = self.command(
            "build_live_futures_overlay",
            [
                "scripts/build_live_futures_overlay.py",
                "--refresh",
                "--ibkr-current-contract-fallback",
                "--ibkr-client-id",
                str(client_id),
                "--max-gap-days",
                "15",
            ],
            timeout=900,
            allow_fail=True,
        )
        if build_rc != 0:
            self.log("overlay_refresh_failed", stage="build_live_futures_overlay", returncode=build_rc)
            return False
        positions_rc = self.command(
            "run_live_overlay_no_equity_positions",
            [
                "scripts/run_live_overlay_no_equity_positions.py",
                "--min-instruments",
                "38",
                "--as-of-date",
                date.today().isoformat(),
            ],
            timeout=900,
            allow_fail=True,
        )
        if positions_rc != 0:
            self.log("overlay_refresh_failed", stage="run_live_overlay_no_equity_positions", returncode=positions_rc)
            return False
        self.last_overlay = started
        self.log("overlay_refresh_done")
        return True

    def run_contract_refresh(self) -> bool:
        started = time.time()
        self.last_contract_attempt = started
        rc = self.command(
            "ibkr_contract_qualification",
            [
                "scripts/ibkr_contract_qualification.py",
                "--all",
                "--min-qualified",
                "38",
                "--min-days-to-expiry",
                "21",
                "--output",
                str(DEFAULT_CONTRACTS),
                "--client-id",
                str(self.next_client_id()),
            ],
            timeout=600,
            allow_fail=True,
        )
        if rc != 0:
            self.log("contract_refresh_failed", returncode=rc)
            return False
        self.last_contract_refresh = started
        self.log("contract_refresh_done")
        return True

    def run_bar_gate(self) -> int:
        return self.command(
            "ibkr_historical_bar_gate",
            [
                "scripts/ibkr_historical_bar_gate.py",
                "--contracts",
                str(DEFAULT_CONTRACTS),
                "--all",
                "--output",
                str(DEFAULT_BAR_GATE),
                "--client-id",
                str(self.next_client_id()),
                "--bar-size",
                "15 mins",
                "--duration",
                "2 D",
            ],
            timeout=180,
            allow_fail=True,
        )

    def run_dry_run(self) -> int:
        return self.command(
            "ibkr_strategy_order_dry_run",
            [
                "scripts/ibkr_strategy_order_dry_run.py",
                "--positions",
                str(DEFAULT_POSITIONS),
                "--model-scale",
                "1.0",
                "--max-target-age-days",
                "1",
                "--client-id",
                str(self.next_client_id()),
                "--expected-account",
                self.config.expected_account,
            ],
            timeout=300,
            allow_fail=True,
        )

    def run_snapshot(self, fetch_full_bars: bool) -> int:
        args = [
            "scripts/ibkr_paper_strategy_runner.py",
            "--client-id",
            str(self.next_client_id()),
            "--expected-account",
            self.config.expected_account,
        ]
        if fetch_full_bars:
            args.extend(["--fetch-bar-history", "--bar-scope", "all_qualified"])
            self.last_full_bars = time.time()
        rc = self.command("ibkr_paper_strategy_runner_snapshot", args, timeout=1200, allow_fail=True)
        if rc == 0:
            self.last_state_snapshot = time.time()
        return rc

    def run_trade_attempt(self, persist_state_snapshot: bool = False) -> int:
        args = [
            "scripts/ibkr_paper_strategy_runner.py",
            "--execute-orders",
            "--confirm-paper-strategy-market-orders",
            "--continue-after-unfilled",
            "--client-id",
            str(self.next_client_id()),
            "--expected-account",
            self.config.expected_account,
            "--wait-seconds",
            "8",
        ]
        if not persist_state_snapshot:
            args.append("--skip-state-snapshot")
        rc = self.command("ibkr_paper_strategy_runner_execute", args, timeout=900, allow_fail=True)
        self.last_trade = time.time()
        return rc

    def one_cycle(self, cycle: int, started_at: float) -> None:
        now = time.time()
        self.heartbeat(cycle=cycle, phase="cycle_start", uptime_seconds=round(now - started_at, 1))
        self.log("cycle_start", cycle=cycle)

        contract_due = self.last_contract_refresh == 0.0 or now - self.last_contract_refresh >= 24 * 3600
        contract_retry_due = self.last_contract_attempt == 0.0 or now - self.last_contract_attempt >= 3600
        if contract_due and contract_retry_due:
            self.contracts_healthy = self.run_contract_refresh()

        overlay_due = self.last_overlay == 0.0 or now - self.last_overlay >= self.config.overlay_interval_seconds
        overlay_retry_due = (
            self.last_overlay_attempt == 0.0
            or now - self.last_overlay_attempt >= self.config.overlay_interval_seconds
        )
        if self.contracts_healthy and overlay_due and overlay_retry_due:
            self.overlay_healthy = self.run_overlay_refresh()
        elif not self.contracts_healthy:
            self.overlay_healthy = False

        bar_gate_rc = self.run_bar_gate()
        dry_run_rc = self.run_dry_run() if self.overlay_healthy else 1
        if not self.overlay_healthy:
            self.log("dry_run_skip", reason="overlay_refresh_not_healthy")
        due_state_snapshot = (
            self.last_state_snapshot == 0.0
            or now - self.last_state_snapshot >= self.config.state_snapshot_interval_seconds
        )
        if due_state_snapshot:
            fetch_full_bars = (
                self.last_full_bars == 0.0 or now - self.last_full_bars >= self.config.full_bar_interval_seconds
            )
            self.run_snapshot(fetch_full_bars=fetch_full_bars)
        else:
            self.log(
                "state_snapshot_skip",
                cycle=cycle,
                seconds_until_next=round(
                    self.config.state_snapshot_interval_seconds - (now - self.last_state_snapshot), 1
                ),
            )

        due_trade = self.last_trade == 0.0 or now - self.last_trade >= self.config.trade_interval_seconds
        if self.config.enable_trading and dry_run_rc == 0 and due_trade:
            self.run_trade_attempt(persist_state_snapshot=False)
        elif self.config.enable_trading:
            self.log("trade_skip", dry_run_rc=dry_run_rc, due_trade=due_trade)

        self.heartbeat(
            cycle=cycle,
            phase="cycle_done",
            uptime_seconds=round(time.time() - started_at, 1),
            overlay_healthy=self.overlay_healthy,
            contracts_healthy=self.contracts_healthy,
            ibkr_healthy="connected server_version="
            in self.last_command_stdout.get("ibkr_historical_bar_gate", ""),
            bar_gate_rc=bar_gate_rc,
            dry_run_rc=dry_run_rc,
        )
        self.log("cycle_done", cycle=cycle)

    def request_stop(self, signum, _frame) -> None:
        self.stop_requested = True
        self.log("signal_stop", signal=signum)

    def run(self) -> int:
        if self.config.enable_trading and not self.config.confirm:
            print("FAIL missing --confirm-paper-24h-daemon for trading-enabled daemon")
            return 2
        try:
            self.acquire_lock()
        except RuntimeError as exc:
            print(f"FAIL {exc}")
            return 3
        try:
            return self._run_locked()
        finally:
            self.release_lock()

    def _run_locked(self) -> int:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

        started_at = time.time()
        deadline = (
            float("inf")
            if self.config.duration_hours <= 0
            else started_at + self.config.duration_hours * 3600.0
        )
        cycle = 0
        self.log(
            "daemon_start",
            pid=os.getpid(),
            duration_hours=self.config.duration_hours,
            interval_seconds=self.config.interval_seconds,
            state_snapshot_interval_seconds=self.config.state_snapshot_interval_seconds,
            enable_trading=self.config.enable_trading,
        )
        self.heartbeat(cycle=cycle, phase="started", uptime_seconds=0)

        while not self.stop_requested and time.time() < deadline:
            if self.stop_path.exists():
                self.log("stop_file_seen", path=str(self.stop_path))
                break
            cycle += 1
            try:
                self.one_cycle(cycle, started_at)
            except Exception as exc:
                self.log("cycle_error", cycle=cycle, error=repr(exc))
                self.heartbeat(cycle=cycle, phase="cycle_error", error=repr(exc))

            if self.config.run_once:
                break

            sleep_until = min(time.time() + self.config.interval_seconds, deadline)
            while not self.stop_requested and time.time() < sleep_until:
                if self.stop_path.exists():
                    self.log("stop_file_seen", path=str(self.stop_path))
                    self.stop_requested = True
                    break
                time.sleep(min(5, sleep_until - time.time()))

        self.heartbeat(cycle=cycle, phase="stopped", uptime_seconds=round(time.time() - started_at, 1))
        self.log("daemon_stop", cycle=cycle)
        self.pid_path.unlink(missing_ok=True)
        return 0


def main() -> int:
    return Runner(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
