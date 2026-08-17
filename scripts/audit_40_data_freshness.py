#!/usr/bin/env python3
"""Audit freshness of the 40-instrument capacity universe CSV inputs.

This script is read-only. It reports where each pysystemtrade-style CSV input
currently stops so that live/provider data can be appended into a mirror tree
without changing the original repository data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rob_style_backtest as bt  # noqa: E402
import run_rob_style_40_capacity_backtest as cap40  # noqa: E402


ROLL_CALENDARS = bt.FUTURES_DATA / "roll_calendars_csv"
REPORT_DIR = ROOT / "reports"
REPORT_CSV = REPORT_DIR / "data_freshness_40.csv"
REPORT_MD = REPORT_DIR / "data_freshness_40.md"


def last_timestamp(path: Path) -> tuple[pd.Timestamp | None, int, str | None]:
    if not path.exists():
        return None, 0, None

    frame = pd.read_csv(path)
    if frame.empty:
        return None, 0, None

    date_columns = [column for column in ("DATETIME", "DATE_TIME", "date", "datetime") if column in frame.columns]
    if not date_columns:
        return None, len(frame), None

    date_column = date_columns[0]
    timestamp = pd.to_datetime(frame[date_column], errors="coerce").max()
    if pd.isna(timestamp):
        return None, len(frame), date_column
    return timestamp, len(frame), date_column


def fmt_timestamp(value: pd.Timestamp | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def stale_days(value: pd.Timestamp | None, audit_date: pd.Timestamp) -> int | None:
    if value is None:
        return None
    return int((audit_date.normalize() - value.normalize()).days)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit_date = pd.Timestamp.today().normalize()
    meta = bt.load_meta()
    instruments = cap40.selected_instruments()

    rows: list[dict[str, object]] = []
    for bucket, names in cap40.CAPACITY_UNIVERSE.items():
        for instrument in names:
            instrument_meta = meta.get(instrument)
            currency = instrument_meta.currency if instrument_meta is not None else ""

            adjusted_last, adjusted_rows, adjusted_date_col = last_timestamp(bt.ADJUSTED / f"{instrument}.csv")
            multiple_last, multiple_rows, multiple_date_col = last_timestamp(bt.MULTIPLE / f"{instrument}.csv")
            roll_last, roll_rows, roll_date_col = last_timestamp(ROLL_CALENDARS / f"{instrument}.csv")

            fx_path = bt.FX / f"{currency}USD.csv"
            if currency == "USD":
                fx_last, fx_rows, fx_date_col = audit_date, 0, "synthetic"
            else:
                fx_last, fx_rows, fx_date_col = last_timestamp(fx_path)

            rows.append(
                {
                    "bucket": bucket,
                    "instrument": instrument,
                    "asset_class": instrument_meta.asset_class if instrument_meta is not None else "",
                    "currency": currency,
                    "fx_required": currency != "USD",
                    "adjusted_last": fmt_timestamp(adjusted_last),
                    "adjusted_stale_days": stale_days(adjusted_last, audit_date),
                    "adjusted_rows": adjusted_rows,
                    "adjusted_date_column": adjusted_date_col or "",
                    "multiple_last": fmt_timestamp(multiple_last),
                    "multiple_stale_days": stale_days(multiple_last, audit_date),
                    "multiple_rows": multiple_rows,
                    "multiple_date_column": multiple_date_col or "",
                    "roll_calendar_last": fmt_timestamp(roll_last),
                    "roll_calendar_stale_days": stale_days(roll_last, audit_date),
                    "roll_calendar_rows": roll_rows,
                    "roll_calendar_date_column": roll_date_col or "",
                    "fx_last": fmt_timestamp(fx_last),
                    "fx_stale_days": stale_days(fx_last, audit_date),
                    "fx_rows": fx_rows,
                    "fx_date_column": fx_date_col or "",
                }
            )

    freshness = pd.DataFrame(rows).sort_values(["bucket", "instrument"])
    freshness.to_csv(REPORT_CSV, index=False)

    adjusted_stale = freshness["adjusted_stale_days"].dropna()
    multiple_stale = freshness["multiple_stale_days"].dropna()
    roll_stale = freshness["roll_calendar_stale_days"].dropna()
    fx_stale = freshness["fx_stale_days"].dropna()
    required_fx_stale = freshness.loc[freshness["fx_required"], "fx_stale_days"].dropna()

    lines = [
        "# 40-Instrument Data Freshness Audit",
        "",
        f"- Audit date: {audit_date.date()}",
        f"- Instruments checked: {len(instruments)}",
        f"- Adjusted price max last date: {freshness['adjusted_last'].max()}",
        f"- Multiple price max last date: {freshness['multiple_last'].max()}",
        f"- Roll calendar max last date: {freshness['roll_calendar_last'].max()}",
        f"- FX max last date: {freshness['fx_last'].max()}",
        "",
        "## Staleness",
        "",
        f"- Adjusted prices: median {adjusted_stale.median():.0f} days stale, max {adjusted_stale.max():.0f} days stale",
        f"- Multiple prices: median {multiple_stale.median():.0f} days stale, max {multiple_stale.max():.0f} days stale",
        f"- Roll calendars: median {roll_stale.median():.0f} days stale, max {roll_stale.max():.0f} days stale",
        f"- FX all instruments: median {fx_stale.median():.0f} days stale, max {fx_stale.max():.0f} days stale",
        f"- FX required non-USD instruments: median {required_fx_stale.median():.0f} days stale, max {required_fx_stale.max():.0f} days stale",
        "",
        "## Most Stale Adjusted Prices",
        "",
    ]

    for row in freshness.sort_values("adjusted_stale_days", ascending=False).head(12).itertuples(index=False):
        lines.append(
            f"- {row.instrument}: adjusted {row.adjusted_last or 'missing'}, multiple {row.multiple_last or 'missing'}, "
            f"roll {row.roll_calendar_last or 'missing'}, FX {row.fx_last or 'missing'}"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- CSV report: `{REPORT_CSV}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {REPORT_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
