#!/usr/bin/env python3
"""Run current no-equity futures positions from the live-data overlay.

This script intentionally reuses the existing Rob-style forecasting and risk
engine.  The only differences versus the normal local backtest are:

- data directories are pointed at the overlay built by
  build_live_futures_overlay.py;
- instruments without fresh, qualified overlay data are removed by gate; and
- the output is designed for IBKR dry-run/order preview.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

import run_rob_style_backtest as bt  # noqa: E402
import run_rob_style_no_equity_40_backtest as no40  # noqa: E402


DEFAULT_OVERLAY = ROOT / "output" / "live_futures_overlay"
DEFAULT_OUT = ROOT / "backtests" / "rob_style_no_equity_live_overlay"


@dataclass(frozen=True)
class Config:
    overlay_dir: Path
    output_dir: Path
    min_instruments: int
    require_all_qualified: bool
    model_scale: float
    start_date: str
    as_of_date: str
    force_republish_same_date: bool
    max_stale_business_days: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run Rob no-equity positions from live overlay data")
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-instruments", type=int, default=38)
    parser.add_argument("--allow-partial", action="store_true", help="Allow fewer than min-instruments for diagnostics")
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--start-date", default="2000-01-19")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--max-stale-business-days", type=int, default=1)
    parser.add_argument(
        "--force-republish-same-date",
        action="store_true",
        help="Overwrite an already published target for the same strategy date.",
    )
    args = parser.parse_args()
    return Config(
        args.overlay_dir,
        args.output_dir,
        args.min_instruments,
        not args.allow_partial,
        args.model_scale,
        args.start_date,
        args.as_of_date,
        args.force_republish_same_date,
        args.max_stale_business_days,
    )


def set_overlay_paths(overlay_dir: Path) -> None:
    bt.ADJUSTED = overlay_dir / "adjusted_prices_csv"
    bt.MULTIPLE = overlay_dir / "multiple_prices_csv"
    bt.FX = overlay_dir / "fx_prices_csv"


def load_trade_ready_instruments(overlay_dir: Path) -> tuple[list[str], pd.DataFrame]:
    coverage_path = overlay_dir / "coverage.csv"
    if not coverage_path.exists():
        raise FileNotFoundError(f"Missing overlay coverage file: {coverage_path}")
    coverage = pd.read_csv(coverage_path)
    selected = no40.selected_instruments()
    ready = coverage[
        coverage["instrument"].isin(selected)
        & coverage["trade_ready"].astype(bool)
        & coverage["ibkr_qualified"].astype(bool)
    ].copy()
    instruments = [instrument for instrument in selected if instrument in set(ready["instrument"])]
    return instruments, coverage


def build_positions(
    config: dict,
    instruments: list[str],
    meta: dict[str, bt.InstrumentMeta],
    model_scale: float,
    max_stale_business_days: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    price = bt.load_price_matrix(instruments)
    price = bt.align_prices_to_as_of(
        price,
        config.get("_as_of_date", price.index.max()),
        max_stale_business_days=max_stale_business_days,
    )
    price_vol = bt.mixed_vol(price.diff())
    fx = bt.load_fx_matrix(instruments, meta, price.index)
    raw_carry = bt.load_raw_carry_matrix(instruments, price_vol, price.index)
    forecasts = bt.build_rule_forecasts(config, instruments, meta, price, price_vol, raw_carry)
    combined_forecast, rule_weight_used = bt.combine_forecasts(forecasts, config, instruments, price.index)
    target, daily_weights, unit_daily_cash_vol = bt.initial_target_positions(
        config, instruments, meta, price, price_vol, fx, combined_forecast
    )
    risk_multiplier = bt.risk_multiplier_for_targets(config, target, unit_daily_cash_vol, price, fx, meta, instruments)
    target = target.mul(risk_multiplier, axis=0) * model_scale

    idm = float(config.get("instrument_div_multiplier", 1.0))
    subsystem_position = (
        float(config["notional_trading_capital"])
        * (float(config["percentage_vol_target"]) / 100.0)
        / math.sqrt(bt.BUSINESS_DAYS)
        / unit_daily_cash_vol
    )
    buffer = (
        (subsystem_position.abs() * daily_weights * idm * bt.BUFFER_SIZE)
        .mul(risk_multiplier, axis=0)
        * model_scale
    )
    buffered_target = bt.apply_position_buffer(target.fillna(0.0), buffer.fillna(0.0))
    integer_positions = bt.optimise_integer_positions(buffered_target, unit_daily_cash_vol, config)
    return (
        price,
        fx,
        raw_carry,
        combined_forecast,
        rule_weight_used,
        unit_daily_cash_vol,
        risk_multiplier,
        target,
        integer_positions,
    )


def build_summary_text(
    config: Config,
    coverage: pd.DataFrame,
    instruments: list[str],
    price: pd.DataFrame,
    raw_carry: pd.DataFrame,
    combined_forecast: pd.DataFrame,
    rule_weight_used: pd.DataFrame,
    target: pd.DataFrame,
    integer_positions: pd.DataFrame,
    risk_multiplier: pd.Series,
) -> str:
    latest = price.index.max()
    selected = no40.selected_instruments()
    disabled = coverage[coverage["instrument"].isin(selected) & ~coverage["trade_ready"].astype(bool)]
    nonzero = integer_positions.loc[latest][integer_positions.loc[latest].round().astype(int).ne(0)].sort_index()
    carry_available_latest = raw_carry.loc[latest].notna() if latest in raw_carry.index else pd.Series(False, index=instruments)
    avg_rule_weight_latest = rule_weight_used.loc[latest].mean()
    gate_pass = len(instruments) >= config.min_instruments

    lines = [
        "# Live Overlay No-Equity Position Run",
        "",
        "## Gate",
        "",
        f"- Overlay: `{config.overlay_dir}`",
        f"- Latest strategy date: {latest.date()}",
        f"- Active trade-ready instruments: {len(instruments)} / {len(selected)}",
        f"- Required instruments: {config.min_instruments}",
        f"- Gate pass for full 38-instrument paper execution: {gate_pass}",
        f"- Model scale: {config.model_scale:g}x",
        "",
        "## Important Data Caveat",
        "",
        "- This overlay uses Yahoo nearby futures proxy closes where available.",
        "- It does not reconstruct true contract-level carry/roll term structure after the local CSV end date.",
        "- Appended multiple-price rows set `PRICE=CARRY`, so carry rules are unavailable in the live overlay segment rather than stale.",
        "- The original Rob risk system is unchanged, but forecast information is reduced where carry is disabled.",
        "",
        "## Latest Diagnostics",
        "",
        f"- Risk multiplier: {risk_multiplier.loc[latest]:.3f}",
        f"- Average available rule weight: {avg_rule_weight_latest:.3f}",
        f"- Instruments with latest carry value: {int(carry_available_latest.sum())} / {len(instruments)}",
        f"- Nonzero buffered integer targets: {len(nonzero)}",
        "",
        "## Nonzero Targets",
        "",
        nonzero.rename("target_contracts").to_frame().to_markdown(),
        "",
        "## Disabled Instruments",
        "",
        disabled[["instrument", "status", "ticker", "ibkr_qualified", "download_rows", "download_last"]].to_markdown(
            index=False
        ),
        "",
        "## Files",
        "",
        "- `positions_live_overlay_1x.csv`",
        "- `target_continuous_live_overlay_1x.csv`",
        "- `combined_forecast_live_overlay.csv`",
        "- `rule_weight_used_live_overlay.csv`",
        "- `latest_target_summary.csv`",
    ]
    return "\n".join(lines)


def latest_published_target_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path, index_col=0, parse_dates=True, usecols=[0])
    except Exception:
        return None
    if len(data.index) == 0:
        return None
    return pd.Timestamp(data.index.max()).normalize()


def has_valid_published_manifest(output_dir: Path, positions_path: Path) -> bool:
    manifest_path = output_dir / "target_manifest.json"
    if not manifest_path.exists() or not positions_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "published"
        and manifest.get("gate_pass") is True
        and manifest.get("positions_file_sha256") == file_sha256(positions_path)
    )


def target_hash(latest_summary: pd.DataFrame, latest_date: pd.Timestamp) -> str:
    fields = ["instrument", "buffered_integer_target_1x"]
    payload = {
        "target_date": str(latest_date.date()),
        "rows": latest_summary[fields].sort_values("instrument").to_dict("records"),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output_dir: Path,
    config: Config,
    latest: pd.Timestamp,
    instruments: list[str],
    gate_pass: bool,
    hash_value: str,
    status: str,
    positions_file_sha256: str = "",
) -> None:
    payload = {
        "status": status,
        "published_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "strategy_date": str(latest.date()),
        "target_hash": hash_value,
        "active_instruments": len(instruments),
        "required_instruments": config.min_instruments,
        "gate_pass": gate_pass,
        "model_scale": config.model_scale,
        "positions_file": "positions_live_overlay_1x.csv",
        "positions_file_sha256": positions_file_sha256,
    }
    path = output_dir / "target_manifest.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def publish_outputs(
    output_dir: Path,
    latest_summary: pd.DataFrame,
    integer_positions: pd.DataFrame,
    target: pd.DataFrame,
    combined_forecast: pd.DataFrame,
    rule_weight_used: pd.DataFrame,
    coverage: pd.DataFrame,
    summary_text_path: Path,
    summary_text: str,
) -> None:
    outputs = {
        output_dir / "latest_target_summary.csv": latest_summary,
        output_dir / "positions_live_overlay_1x.csv": integer_positions.round().astype(int),
        output_dir / "target_continuous_live_overlay_1x.csv": target,
        output_dir / "combined_forecast_live_overlay.csv": combined_forecast,
        output_dir / "rule_weight_used_live_overlay.csv": rule_weight_used,
        output_dir / "coverage_used.csv": coverage,
    }
    temp_paths: list[tuple[Path, Path]] = []
    for path, frame in outputs.items():
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(tmp, index=False if path.name in {"latest_target_summary.csv", "coverage_used.csv"} else True)
        temp_paths.append((tmp, path))
    summary_tmp = summary_text_path.with_suffix(summary_text_path.suffix + ".tmp")
    summary_tmp.write_text(summary_text, encoding="utf-8")
    temp_paths.append((summary_tmp, summary_text_path))
    for tmp, path in temp_paths:
        tmp.replace(path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_overlay_paths(args.overlay_dir)
    bt.START_DATE = args.start_date

    instruments, coverage = load_trade_ready_instruments(args.overlay_dir)
    selected_count = len(no40.selected_instruments())
    gate_pass = len(instruments) >= args.min_instruments
    if args.require_all_qualified and not gate_pass:
        print(
            f"gate_fail_active_instruments={len(instruments)}/{selected_count} "
            f"required={args.min_instruments}; rerun with --allow-partial for diagnostics only"
        )

    config = no40.config_with_no_equity_weights(bt.load_rob_config())
    config["_as_of_date"] = args.as_of_date
    meta = bt.load_meta()
    missing = [instrument for instrument in instruments if not bt.has_required_files(instrument, meta)]
    if missing:
        raise RuntimeError(f"Overlay missing required files for active instruments: {missing}")

    (
        price,
        _fx,
        raw_carry,
        combined_forecast,
        rule_weight_used,
        _unit_daily_cash_vol,
        risk_multiplier,
        target,
        integer_positions,
    ) = build_positions(
        config,
        instruments,
        meta,
        model_scale=args.model_scale,
        max_stale_business_days=args.max_stale_business_days,
    )

    latest = price.index.max()
    latest_price_valid = price.loc[latest].reindex(instruments).notna()
    latest_forecast_valid = combined_forecast.loc[latest].reindex(instruments).notna()
    latest_target_ready = latest_price_valid & latest_forecast_valid
    target_date_ready_count = int(latest_target_ready.sum())
    target_date_gate_pass = target_date_ready_count >= args.min_instruments
    gate_pass = gate_pass and target_date_gate_pass
    latest_summary = pd.DataFrame(
        {
            "instrument": instruments,
            "latest_price": price.loc[latest].reindex(instruments).values,
            "combined_forecast": combined_forecast.loc[latest].reindex(instruments).values,
            "continuous_target_1x": target.loc[latest].reindex(instruments).values,
            "buffered_integer_target_1x": integer_positions.loc[latest].reindex(instruments).round().astype(int).values,
            "available_rule_weight": rule_weight_used.loc[latest].reindex(instruments).values,
            "carry_available": raw_carry.loc[latest].reindex(instruments).notna().values,
        }
    )
    hash_value = target_hash(latest_summary, latest)
    current_target_date = latest_published_target_date(args.output_dir / "positions_live_overlay_1x.csv")
    published_positions_path = args.output_dir / "positions_live_overlay_1x.csv"
    same_date_already_published = (
        current_target_date == latest.normalize()
        and has_valid_published_manifest(args.output_dir, published_positions_path)
    )

    if args.require_all_qualified and not gate_pass:
        rejected_dir = args.output_dir / "rejected" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rejected_dir.mkdir(parents=True, exist_ok=True)
        latest_summary.to_csv(rejected_dir / "latest_target_summary.csv", index=False)
        coverage.to_csv(rejected_dir / "coverage_used.csv", index=False)
        write_manifest(rejected_dir, args, latest, instruments, gate_pass, hash_value, "rejected_gate_fail")
        (rejected_dir / "target_date_readiness.csv").write_text(
            pd.DataFrame(
                {
                    "instrument": instruments,
                    "price_valid": latest_price_valid.values,
                    "forecast_valid": latest_forecast_valid.values,
                    "target_ready": latest_target_ready.values,
                }
            ).to_csv(index=False),
            encoding="utf-8",
        )
        print(
            f"target_date_gate_fail={target_date_ready_count}/{len(instruments)} "
            f"strategy_date={latest.date()} required={args.min_instruments}"
        )
        print(f"rejected_diagnostics={rejected_dir}")
        return 1

    if same_date_already_published and not args.force_republish_same_date:
        print(
            f"publish_skip_same_date strategy_date={latest.date()} "
            f"existing={current_target_date.date()} target_hash={hash_value}"
        )
        return 0

    summary_text = build_summary_text(
        args,
        coverage,
        instruments,
        price,
        raw_carry,
        combined_forecast,
        rule_weight_used,
        target,
        integer_positions,
        risk_multiplier,
    )
    summary_path = args.output_dir / "summary.md"
    publish_outputs(
        args.output_dir,
        latest_summary,
        integer_positions,
        target,
        combined_forecast,
        rule_weight_used,
        coverage,
        summary_path,
        summary_text,
    )
    positions_path = args.output_dir / "positions_live_overlay_1x.csv"
    write_manifest(
        args.output_dir,
        args,
        latest,
        instruments,
        gate_pass,
        hash_value,
        "published",
        positions_file_sha256=file_sha256(positions_path),
    )

    print(f"wrote={args.output_dir}")
    print(f"latest_date={latest.date()} active_instruments={len(instruments)}/{selected_count} gate_pass={gate_pass}")
    print("nonzero_targets")
    print(
        latest_summary[latest_summary["buffered_integer_target_1x"].ne(0)][
            ["instrument", "combined_forecast", "continuous_target_1x", "buffered_integer_target_1x"]
        ].to_string(index=False)
    )
    return 0 if gate_pass or not args.require_all_qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
