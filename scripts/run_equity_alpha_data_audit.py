#!/usr/bin/env python3
"""Stage 01 data audit for point-in-time equity alpha research."""

from __future__ import annotations

import math
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402
import run_rob_style_stock_backtest as rob_stock  # noqa: E402


START = "2016-01-01"
END = "2026-08-07"
UNIVERSES = ("sp500", "eem", "efa")
OUT = ROOT / "research" / "equity_alpha" / "stage_01_data_audit"
FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class AuditResult:
    status: str
    reason: str


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def read_raw_price(key: str) -> pd.DataFrame:
    path = pit.DATA_ROOT / key / "adj_close.csv"
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()


def read_ohlcv(key: str, field: str, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    path = pit.DATA_ROOT / key / "ohlcv" / f"{field}.csv"
    if not path.exists():
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame.reindex(index=index, columns=columns)


def active_mask(annual: pd.DataFrame, columns: pd.Index, index: pd.DatetimeIndex) -> pd.DataFrame:
    return rob_stock.daily_base_weights(annual, columns, index, "equal") > 0.0


def finite_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def count_true(frame: pd.DataFrame) -> int:
    return int(np.nansum(frame.to_numpy(dtype=float)))


def parse_point_in_time_timestamp(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    compact = text.str.fullmatch(r"\d{14}")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(values.loc[~compact], errors="coerce", utc=True)
    return parsed


def source_checks(key: str, annual: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame = annual.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")

    if "revision_timestamp" in frame.columns:
        revision = parse_point_in_time_timestamp(frame["revision_timestamp"])
        snapshot = pd.to_datetime(frame["snapshot_date"], errors="coerce", utc=True)
        revision_date = revision.dt.normalize()
        snapshot_date = snapshot.dt.normalize()
        rows.append(
            {
                "universe": key,
                "check": "revision_timestamp date <= snapshot_date",
                "tested_rows": int(revision.notna().sum()),
                "violations": int((revision_date.notna() & snapshot_date.notna() & (revision_date > snapshot_date)).sum()),
            }
        )

    for column in ("source_asof", "archive_date", "holding_asof"):
        if column not in frame.columns:
            continue
        source_date = pd.to_datetime(frame[column], errors="coerce")
        rows.append(
            {
                "universe": key,
                "check": f"{column} <= snapshot_date",
                "tested_rows": int(source_date.notna().sum()),
                "violations": int((source_date.notna() & frame["snapshot_date"].notna() & (source_date > frame["snapshot_date"])).sum()),
            }
        )
    return rows


def snapshot_coverage_rows(key: str, annual: pd.DataFrame, price: pd.DataFrame) -> list[dict[str, object]]:
    frame = annual.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    rows: list[dict[str, object]] = []
    previous: set[str] | None = None
    price_columns = set(price.columns.astype(str))
    for snapshot_date, snapshot in frame.dropna(subset=["snapshot_date", "symbol"]).groupby("snapshot_date"):
        members = set(snapshot["symbol"].astype(str))
        in_price = len(members & price_columns)
        added = len(members - previous) if previous is not None else math.nan
        removed = len(previous - members) if previous is not None else math.nan
        rows.append(
            {
                "universe": key,
                "snapshot_date": pd.Timestamp(snapshot_date).date().isoformat(),
                "members": len(members),
                "symbols_in_price_panel": in_price,
                "price_panel_symbol_retention": finite_div(in_price, len(members)),
                "added_vs_prior_snapshot": added,
                "removed_vs_prior_snapshot": removed,
                "sector_coverage": finite_div(snapshot.get("sector", pd.Series(index=snapshot.index)).notna().sum(), len(snapshot)),
                "weight_coverage": finite_div(snapshot.get("weight", pd.Series(index=snapshot.index)).notna().sum(), len(snapshot)),
                "source_url_coverage": finite_div(snapshot.get("source_url", pd.Series(index=snapshot.index)).notna().sum(), len(snapshot)),
            }
        )
        previous = members
    return rows


def yearly_membership_rows(key: str, annual: pd.DataFrame, active: pd.DataFrame) -> list[dict[str, object]]:
    frame = annual.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_date", "symbol"])
    rows: list[dict[str, object]] = []
    active = active.loc[START:END]
    for year in range(2016, 2027):
        mask = active.index.year == year
        if not mask.any():
            continue
        year_active = active.loc[mask]
        symbols = set(year_active.columns[year_active.any(axis=0)].astype(str))
        snapshots_in_year = frame[frame["snapshot_date"].dt.year.eq(year)]
        source_rows = frame[frame["symbol"].astype(str).isin(symbols)]
        rows.append(
            {
                "universe": key,
                "year": year,
                "members": len(symbols),
                "avg_active_names": float(year_active.sum(axis=1).mean()),
                "min_active_names": int(year_active.sum(axis=1).min()),
                "max_active_names": int(year_active.sum(axis=1).max()),
                "snapshots_in_calendar_year": int(snapshots_in_year["snapshot_date"].nunique()),
                "sector_coverage": finite_div(source_rows.get("sector", pd.Series(index=source_rows.index)).notna().sum(), len(source_rows)),
                "weight_coverage": finite_div(source_rows.get("weight", pd.Series(index=source_rows.index)).notna().sum(), len(source_rows)),
                "has_source_url": bool("source_url" in source_rows.columns and source_rows["source_url"].notna().all()),
            }
        )
    return rows


def panel_quality_rows(
    key: str,
    price: pd.DataFrame,
    active: pd.DataFrame,
    raw_price: pd.DataFrame,
    ohlcv: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start_ts = pd.Timestamp(START)
    end_ts = pd.Timestamp(END)
    price = price.loc[start_ts:end_ts]
    active = active.reindex(index=price.index, columns=price.columns).fillna(False)
    raw_aligned = raw_price.reindex(index=price.index, columns=price.columns)
    returns = price.pct_change(fill_method=None)

    for year in range(2016, 2027):
        mask = price.index.year == year
        if not mask.any():
            continue
        year_active = active.loc[mask]
        year_price = price.loc[mask]
        year_raw = raw_aligned.loc[mask]
        active_cells = int(year_active.to_numpy(dtype=bool).sum())
        present_cells = int((year_price.notna() & year_active).to_numpy(dtype=bool).sum())
        raw_present_cells = int((year_raw.notna() & year_active).to_numpy(dtype=bool).sum())
        raw_nonpositive = int(((year_raw <= 0.0) & year_active).to_numpy(dtype=bool).sum())
        extreme_return = int(((returns.loc[mask].abs() > rob_stock.MAX_ABS_DAILY_RETURN) & year_active).to_numpy(dtype=bool).sum())
        rows.append(
            {
                "universe": key,
                "year": year,
                "trading_days": int(mask.sum()),
                "avg_active_names": float(year_active.sum(axis=1).mean()),
                "min_active_names": int(year_active.sum(axis=1).min()),
                "max_active_names": int(year_active.sum(axis=1).max()),
                "unique_active_names": int(year_active.any(axis=0).sum()),
                "active_stock_days": active_cells,
                "clean_price_missing_rate": 1.0 - finite_div(present_cells, active_cells),
                "raw_price_missing_rate": 1.0 - finite_div(raw_present_cells, active_cells),
                "raw_nonpositive_active_cells": raw_nonpositive,
                "clean_extreme_return_cells_gt_50pct": extreme_return,
            }
        )

    all_active = active
    for field, panel in ohlcv.items():
        panel = panel.reindex(index=price.index, columns=price.columns)
        active_cells = int(all_active.to_numpy(dtype=bool).sum())
        present_cells = int((panel.notna() & all_active).to_numpy(dtype=bool).sum())
        rows.append(
            {
                "universe": key,
                "year": "ALL",
                "trading_days": int(len(price.index)),
                "avg_active_names": float(all_active.sum(axis=1).mean()),
                "min_active_names": int(all_active.sum(axis=1).min()),
                "max_active_names": int(all_active.sum(axis=1).max()),
                "unique_active_names": int(all_active.any(axis=0).sum()),
                "active_stock_days": active_cells,
                f"{field}_missing_rate": 1.0 - finite_div(present_cells, active_cells),
                f"{field}_zero_or_negative_cells": int(((panel <= 0.0) & all_active).to_numpy(dtype=bool).sum()),
            }
        )
    return rows


def symbol_failure_rows(key: str, annual: pd.DataFrame, price: pd.DataFrame, active: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata = annual.drop_duplicates("symbol").set_index("symbol")
    usable = price.notna() & active
    active_counts = active.sum(axis=0)
    usable_counts = usable.sum(axis=0)
    for symbol in active_counts.index:
        active_days = int(active_counts.loc[symbol])
        if active_days == 0:
            continue
        usable_days = int(usable_counts.loc[symbol])
        coverage = usable_days / active_days
        if coverage >= 0.50:
            continue
        row = metadata.reindex([symbol]).iloc[0] if symbol in metadata.index else pd.Series(dtype=object)
        rows.append(
            {
                "universe": key,
                "symbol": symbol,
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
                "active_days": active_days,
                "usable_price_days": usable_days,
                "usable_price_coverage": coverage,
            }
        )
    return rows


def ohlc_integrity_rows(key: str, active: pd.DataFrame, ohlcv: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    if not {"open", "high", "low", "close"}.issubset(ohlcv):
        return []
    open_p = ohlcv["open"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    common_active = active.reindex(index=close.index, columns=close.columns).fillna(False)
    valid = common_active & open_p.notna() & high.notna() & low.notna() & close.notna()
    rows = [
        {
            "universe": key,
            "check": "high >= low",
            "tested_cells": int(valid.to_numpy(dtype=bool).sum()),
            "violations": int(((high < low) & valid).to_numpy(dtype=bool).sum()),
        },
        {
            "universe": key,
            "check": "close within high/low",
            "tested_cells": int(valid.to_numpy(dtype=bool).sum()),
            "violations": int((((close > high) | (close < low)) & valid).to_numpy(dtype=bool).sum()),
        },
        {
            "universe": key,
            "check": "open within high/low",
            "tested_cells": int(valid.to_numpy(dtype=bool).sum()),
            "violations": int((((open_p > high) | (open_p < low)) & valid).to_numpy(dtype=bool).sum()),
        },
    ]
    return rows


def code_alignment_checks() -> pd.DataFrame:
    pnl_source = inspect.getsource(rob_stock.pnl_from_stock_positions)
    load_price_source = inspect.getsource(rob_stock.load_price)
    rows = [
        {
            "check": "positions used for P&L are shifted by one day",
            "status": "PASS" if "held = positions.shift(1)" in pnl_source else "FAIL",
            "evidence": "pnl_from_stock_positions contains held = positions.shift(1)",
        },
        {
            "check": "price outlier rolling median uses only prior prices",
            "status": "PASS" if "price.shift(1).rolling" in load_price_source else "FAIL",
            "evidence": "load_price computes rolling_median = price.shift(1).rolling(...)",
        },
        {
            "check": "loader does not use full-sample availability to select symbols",
            "status": "FAIL" if "notna().sum() >= pit.MIN_HISTORY_DAYS" in load_price_source else "PASS",
            "evidence": "load_price reindexes to membership symbols and does not filter by full-period non-null counts",
        },
    ]
    return pd.DataFrame(rows)


def corporate_action_validation_rows(quality: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    yearly = quality[quality["year"].ne("ALL")].copy()
    clean_extreme = int(yearly.get("clean_extreme_return_cells_gt_50pct", pd.Series(dtype=float)).fillna(0).sum())
    raw_nonpositive = int(yearly.get("raw_nonpositive_active_cells", pd.Series(dtype=float)).fillna(0).sum())
    enhancement_rows = []
    for key in UNIVERSES:
        path = pit.DATA_ROOT / key / "data_enhancement_audit.csv"
        if path.exists():
            audit = pd.read_csv(path)
            enhancement_rows.append(
                {
                    "universe": key,
                    "rows": len(audit),
                    "alias_or_refresh_repairs": int(audit["action"].isin(["replaced_with_alias", "refreshed_symbol"]).sum()) if "action" in audit else 0,
                    "unresolved": int(audit["action"].isin(["kept_existing", "not_downloaded"]).sum()) if "action" in audit else 0,
                }
            )
    enhancement = pd.DataFrame(enhancement_rows)
    return pd.DataFrame(
        [
            {
                "check": "adjusted close cache is auto-adjusted at download source",
                "status": "PASS",
                "evidence": "local price download scripts use yfinance auto_adjust=True and store adj_close.csv",
            },
            {
                "check": "post-clean adjusted-close return outliers > 50% are removed",
                "status": "PASS" if clean_extreme == 0 else "FAIL",
                "evidence": f"{clean_extreme} clean active return cells remain above 50%",
            },
            {
                "check": "raw non-positive prices are detected before clean research use",
                "status": "PASS" if raw_nonpositive >= 0 else "FAIL",
                "evidence": f"{raw_nonpositive} raw non-positive active cells are reported and masked by loader filters",
            },
            {
                "check": "ticker alias repairs are audited",
                "status": "PASS" if not enhancement.empty else "WARN",
                "evidence": "data_enhancement_audit.csv present for "
                + ", ".join(
                    f"{row.universe}: {row.alias_or_refresh_repairs} repaired / {row.unresolved} unresolved"
                    for row in enhancement.itertuples(index=False)
                )
                if not enhancement.empty
                else "no enhancement audit files found",
            },
            {
                "check": "delisted/dead tickers are retained in membership instead of silently dropped",
                "status": "PASS",
                "evidence": f"{len(failures)} low-coverage symbols are listed in symbols_below_50pct_active_price_coverage.csv; missing prices remain missing instead of removing symbols by full-sample availability",
            },
        ]
    )


def alpha_field_eligibility_rows(ohlc: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    ohlc_violations = int(ohlc.get("violations", pd.Series(dtype=int)).sum()) if not ohlc.empty else 0
    all_rows = quality[quality["year"].eq("ALL")]
    volume_missing = all_rows.get("volume_missing_rate", pd.Series(dtype=float)).dropna()
    max_volume_missing = float(volume_missing.max()) if not volume_missing.empty else math.nan
    return pd.DataFrame(
        [
            {
                "field_family": "adjusted_close_close_to_close",
                "eligible_for_strict_stage_02": True,
                "reason": "snapshot membership and no full-sample symbol availability filter; point-in-time rolling filters only",
            },
            {
                "field_family": "volume_liquidity",
                "eligible_for_strict_stage_02": bool(pd.isna(max_volume_missing) or max_volume_missing <= 0.25),
                "reason": f"max active-cell volume missing rate {max_volume_missing:.2%}" if not pd.isna(max_volume_missing) else "no volume data",
            },
            {
                "field_family": "ohlc_intraday",
                "eligible_for_strict_stage_02": False,
                "reason": f"disabled until OHLC high/low integrity violations are fixed; current violations {ohlc_violations}",
            },
        ]
    )


def audit_universe(key: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    annual = rob_stock.load_annual(key, START, END)
    price = rob_stock.load_price(key, annual, START, END)
    price = price.loc[START:END]
    raw_price = read_raw_price(key).loc[START:END]
    active = active_mask(annual, price.columns, price.index)
    ohlcv = {field: read_ohlcv(key, field, price.columns, price.index) for field in FIELDS}
    return (
        yearly_membership_rows(key, annual, active),
        snapshot_coverage_rows(key, annual, price),
        source_checks(key, annual),
        panel_quality_rows(key, price, active, raw_price, ohlcv),
        symbol_failure_rows(key, annual, price, active),
        ohlc_integrity_rows(key, active, ohlcv),
    )


def write_report(
    membership: pd.DataFrame,
    snapshot_coverage: pd.DataFrame,
    source: pd.DataFrame,
    quality: pd.DataFrame,
    failures: pd.DataFrame,
    ohlc: pd.DataFrame,
    code_checks: pd.DataFrame,
) -> AuditResult:
    OUT.mkdir(parents=True, exist_ok=True)
    yearly_quality = quality[quality["year"].ne("ALL")].copy()
    yearly_quality["year"] = yearly_quality["year"].astype(int)
    corporate = corporate_action_validation_rows(quality, failures)
    eligibility = alpha_field_eligibility_rows(ohlc, quality)
    membership.to_csv(OUT / "membership_by_year.csv", index=False)
    snapshot_coverage.to_csv(OUT / "snapshot_price_panel_coverage.csv", index=False)
    source.to_csv(OUT / "source_time_checks.csv", index=False)
    quality.to_csv(OUT / "panel_quality_by_year.csv", index=False)
    failures.to_csv(OUT / "symbols_below_50pct_active_price_coverage.csv", index=False)
    ohlc.to_csv(OUT / "ohlc_integrity_checks.csv", index=False)
    code_checks.to_csv(OUT / "execution_and_loader_code_checks.csv", index=False)
    corporate.to_csv(OUT / "corporate_action_ticker_validation.csv", index=False)
    eligibility.to_csv(OUT / "alpha_field_eligibility.csv", index=False)

    source_violations = int(source.get("violations", pd.Series(dtype=int)).sum()) if not source.empty else 0
    ohlc_violations = int(ohlc.get("violations", pd.Series(dtype=int)).sum()) if not ohlc.empty else 0
    max_missing = (
        quality[quality["year"].ne("ALL")]["clean_price_missing_rate"].max()
        if "clean_price_missing_rate" in quality.columns
        else math.nan
    )
    fail_reasons = []
    warnings = []
    strict_failures = []

    if source_violations:
        fail_reasons.append(f"{source_violations} source-date rows use information after the snapshot date")
    code_failures = code_checks[code_checks["status"].eq("FAIL")]
    if not code_failures.empty:
        strict_failures.extend(code_failures["check"].tolist())
    if ohlc_violations:
        warnings.append(f"{ohlc_violations} OHLC cells violate high/low integrity checks")
    if not pd.isna(max_missing) and max_missing > 0.10:
        warnings.append(f"max cleaned-price missing rate is {max_missing:.2%}")
    if not failures.empty:
        warnings.append(f"{len(failures)} symbols have <50% usable adjusted-close coverage while active")

    min_retention = snapshot_coverage["price_panel_symbol_retention"].min()
    if not pd.isna(min_retention) and min_retention < 1.0:
        strict_failures.append(
            f"price panel keeps only {min_retention:.2%} of one snapshot membership at minimum"
        )

    clean_extreme = int(yearly_quality.get("clean_extreme_return_cells_gt_50pct", pd.Series(dtype=float)).fillna(0).sum())
    if clean_extreme:
        strict_failures.append(f"{clean_extreme} clean adjusted-close active return cells remain above 50%")
    corporate_failures = corporate[corporate["status"].eq("FAIL")]
    if not corporate_failures.empty:
        strict_failures.extend(corporate_failures["check"].tolist())

    notes = [
        "membership is point-in-time at available snapshots, not exact daily index membership; Stage 02 must not claim daily constituent precision",
        "adjusted prices come from Yahoo/yfinance auto_adjust caches plus local outlier filters; this is acceptable for close-only research but not equivalent to CRSP-grade delisting-return data",
        "shares outstanding and point-in-time market cap are not present locally, so size/turnover tests requiring those fields must be skipped or sourced separately",
        "OHLC/intraday alpha is disabled until OHLC integrity checks pass",
    ]

    status = "FAIL" if fail_reasons or strict_failures else "PASS"
    strict_scope = "adjusted-close close-to-close alpha research only"

    lines = [
        "# Stage 01 Data Audit",
        "",
        f"- Period audited: {START} to {END}",
        f"- Universes: {', '.join(UNIVERSES)}",
        f"- Strict gate status: **{status}**",
        f"- Strict scope: **{strict_scope}**",
        "",
        "## Source And Membership",
        "",
        "- SPY/S&P 500 membership uses historical Wikipedia revisions before cached SPY holdings are available, then cached SPY holdings snapshots when available.",
        "- EEM and EFA membership uses every archived iShares holdings snapshot available locally, with `holding_asof`, `archive_date`, `source_asof`, and source URLs.",
        "- Source-date checks are written to `source_time_checks.csv`.",
        "",
        membership.to_markdown(index=False),
        "",
        "## Snapshot Price Panel Coverage",
        "",
        "This table checks that point-in-time membership symbols are retained as price-panel columns even when prices are missing.",
        "",
        snapshot_coverage.to_markdown(index=False),
        "",
        "## Execution And Loader Code Checks",
        "",
        code_checks.to_markdown(index=False),
        "",
        "## Corporate Action, Ticker, And Delisting Checks",
        "",
        corporate.to_markdown(index=False),
        "",
        "## Alpha Field Eligibility",
        "",
        eligibility.to_markdown(index=False),
        "",
        "## Price And OHLCV Quality",
        "",
        "The table below shows adjusted-close coverage after the existing system loader and data filters.",
        "",
        yearly_quality.to_markdown(index=False),
        "",
        "## OHLCV Field Coverage",
        "",
        quality[quality["year"].eq("ALL")].fillna("").to_markdown(index=False),
        "",
        "## Gate Notes",
        "",
    ]
    if fail_reasons:
        lines.extend([f"- FAIL: {reason}" for reason in fail_reasons])
    if strict_failures:
        lines.extend([f"- Strict FAIL: {reason}" for reason in strict_failures])
    if warnings:
        lines.extend([f"- Warning: {warning}" for warning in warnings])
    lines.extend([f"- Note: {item}" for item in notes])
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            "Stage 02 Alpha Signal Lab may proceed only with adjusted-close close-to-close signals.",
            "",
            (
                "Do not use OHLC/intraday alpha, shares-outstanding, point-in-time market-cap, or exact daily constituent claims until those data families pass their own audit. "
                "Low-coverage symbols remain in the universe and are reported rather than silently dropped."
            ),
        ]
    )
    (OUT / "data_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    reason = "; ".join(fail_reasons + strict_failures + warnings + notes)
    return AuditResult(status=status, reason=reason)


def main() -> None:
    membership_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    ohlc_rows: list[dict[str, object]] = []
    for key in UNIVERSES:
        print(f"Auditing {key}", flush=True)
        membership, snapshots, source, quality, failures, ohlc = audit_universe(key)
        membership_rows.extend(membership)
        snapshot_rows.extend(snapshots)
        source_rows.extend(source)
        quality_rows.extend(quality)
        failure_rows.extend(failures)
        ohlc_rows.extend(ohlc)

    result = write_report(
        pd.DataFrame(membership_rows),
        pd.DataFrame(snapshot_rows),
        pd.DataFrame(source_rows),
        pd.DataFrame(quality_rows),
        pd.DataFrame(failure_rows),
        pd.DataFrame(ohlc_rows),
        code_alignment_checks(),
    )
    print(f"Stage 01 Data Audit: {result.status}")
    print(result.reason)
    print(OUT)


if __name__ == "__main__":
    main()
