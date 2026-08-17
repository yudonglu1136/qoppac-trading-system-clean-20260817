from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_stock_forecast_tail_lab as tail_lab  # noqa: E402


def test_sector_top10_uses_point_in_time_sector_groups() -> None:
    columns = pd.Index([f"A{i}" for i in range(10)] + [f"B{i}" for i in range(10)])
    index = pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
    annual = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2020-01-01")] * 20,
            "symbol": columns,
            "sector": ["Tech"] * 10 + ["Financials"] * 10,
        }
    )
    sectors = tail_lab.sector_frame_from_annual(annual, columns, index)

    scores = pd.DataFrame(
        [[*range(100, 90, -1), *range(10, 0, -1)]],
        index=index,
        columns=columns,
        dtype=float,
    )
    top = tail_lab.apply_selection(scores, "top10_long", sector_frame=sectors)
    sector_top = tail_lab.apply_selection(scores, "sector_top10_long", sector_frame=sectors)

    assert sectors.loc[index[0], "A0"] == "Tech"
    assert sectors.loc[index[0], "B0"] == "Financials"
    assert set(top.loc[index[0]].dropna().index) == {"A0", "A1"}
    assert set(sector_top.loc[index[0]].dropna().index) == {"A0", "B0"}
