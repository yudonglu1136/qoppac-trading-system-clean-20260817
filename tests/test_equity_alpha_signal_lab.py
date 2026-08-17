from __future__ import annotations

import math

import pandas as pd

import run_equity_alpha_signal_lab as lab


def test_forward_excess_return_starts_after_execution_lag() -> None:
    index = pd.bdate_range("2020-01-01", periods=8)
    price = pd.DataFrame({"AAA": [10, 11, 12, 14, 15, 16, 17, 18]}, index=index, dtype=float)
    benchmark = pd.Series([20, 21, 22, 23, 24, 25, 26, 27], index=index, dtype=float)
    active = pd.DataFrame(True, index=index, columns=price.columns)
    data = lab.UniverseInputs(
        key="test",
        label="Test",
        annual=pd.DataFrame(),
        price=price,
        active=active,
        sector=pd.DataFrame("Unknown", index=index, columns=price.columns),
        benchmark_price=benchmark,
    )

    target = lab.forward_excess_return(data, horizon=2)

    expected = math.log(price.loc[index[3], "AAA"] / price.loc[index[1], "AAA"]) - math.log(
        benchmark.loc[index[3]] / benchmark.loc[index[1]]
    )
    assert math.isclose(target.loc[index[0], "AAA"], expected)


def test_horizon_gate_requires_positive_net_spread() -> None:
    yearly = pd.DataFrame(
        {
            "year": [2020, 2021, 2022, 2023],
            "mean_rank_ic": [0.01, 0.02, 0.01, 0.03],
            "q5_q1_net_spread_ann": [0.01, 0.02, 0.01, 0.03],
        }
    )
    row = pd.Series(
        {
            "mean_rank_ic": 0.01,
            "daily_ic_ir": 0.1,
            "q5_q1_spread_ann": 0.02,
            "q5_q1_net_spread_ann": 0.01,
            "quintile_monotonicity": 0.5,
        }
    )
    assert lab.gate_row(row, yearly)["gate_status"] == "PASS_CANDIDATE"

    row["q5_q1_net_spread_ann"] = -0.001
    assert lab.gate_row(row, yearly)["gate_status"] == "DROP"
