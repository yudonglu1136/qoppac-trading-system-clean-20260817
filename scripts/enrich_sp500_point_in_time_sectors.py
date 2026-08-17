#!/usr/bin/env python3
"""Reparse historical S&P 500 Wikipedia snapshots to add point-in-time sectors."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_point_in_time_annual_ranked_long_only as pit  # noqa: E402


DATA_DIR = pit.DATA_ROOT / "sp500"
DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2026


def load_sp500_spec() -> pit.UniverseSpec:
    return next(spec for spec in pit.UNIVERSES if spec.key == "sp500")


def reparse_sector_snapshots(start_year: int, end_year: int, sleep: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = load_sp500_spec()
    audit = pd.read_csv(DATA_DIR / "annual_snapshot_audit.csv")
    audit = audit[(audit["year"] >= start_year) & (audit["year"] <= end_year)].copy()
    annual_existing = pd.read_csv(DATA_DIR / "annual_constituents.csv")
    annual_existing = annual_existing[
        (annual_existing["year"] >= start_year) & (annual_existing["year"] <= end_year)
    ].copy()

    session = requests.Session()
    frames = []
    audit_rows = []
    for row in audit.itertuples(index=False):
        year = int(row.year)
        source_url = getattr(row, "source_url", "")
        if not isinstance(source_url, str) or not source_url.startswith("http"):
            audit_rows.append(
                {
                    "year": year,
                    "snapshot_date": row.snapshot_date,
                    "status": "missing_source_url",
                    "rows": 0,
                    "sector_coverage": 0.0,
                    "source_url": source_url,
                }
            )
            continue

        try:
            response = pit.get_with_retries(session, source_url, timeout=30)
            parsed, status = pit.parse_constituent_table(response.text, spec)
        except Exception as exc:  # pragma: no cover - network edge
            parsed = None
            status = f"error: {exc}"

        if parsed is None:
            fallback = annual_existing[annual_existing["year"].eq(year)].copy()
            fallback["sector"] = fallback.get("sector", pd.NA)
            parsed = fallback
        else:
            parsed["year"] = year
            parsed["snapshot_date"] = row.snapshot_date
            parsed["revision_timestamp"] = getattr(row, "revision_timestamp", pd.NA)
            parsed["revision_id"] = getattr(row, "revision_id", pd.NA)
            parsed["source_url"] = source_url

        parsed["sector"] = parsed["sector"].replace("", pd.NA).fillna("Unknown")
        frames.append(parsed)
        coverage = parsed["sector"].ne("Unknown").mean() if "sector" in parsed else 0.0
        audit_rows.append(
            {
                "year": year,
                "snapshot_date": row.snapshot_date,
                "status": status,
                "rows": len(parsed),
                "sector_coverage": coverage,
                "source_url": source_url,
            }
        )
        print(f"sp500 {year}: {status}, rows={len(parsed)}, sector_coverage={coverage:.1%}", flush=True)
        time.sleep(sleep)

    if not frames:
        raise RuntimeError("No SPY sector snapshots parsed")
    annual = pd.concat(frames, ignore_index=True, sort=False)
    return annual, pd.DataFrame(audit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--write", action="store_true", help="Overwrite annual_constituents.csv with sector-enriched rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annual, audit = reparse_sector_snapshots(args.start_year, args.end_year, args.sleep)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    enriched_path = DATA_DIR / "annual_constituents_with_sectors.csv"
    audit_path = DATA_DIR / "sector_enrichment_audit.csv"
    annual.to_csv(enriched_path, index=False)
    audit.to_csv(audit_path, index=False)
    if args.write:
        backup = DATA_DIR / "annual_constituents.csv.pre_sector_enrichment"
        original = DATA_DIR / "annual_constituents.csv"
        if original.exists() and not backup.exists():
            backup.write_bytes(original.read_bytes())
        annual.to_csv(original, index=False)
    print(enriched_path)
    print(audit_path)


if __name__ == "__main__":
    main()
