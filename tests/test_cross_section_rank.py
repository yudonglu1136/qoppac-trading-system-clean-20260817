from __future__ import annotations

import pandas as pd

from cross_sectional_nn.forecast_conversion import raw_alpha_to_forecast


def test_raw_alpha_forecast_uses_exact_daily_percentile_rank() -> None:
    predictions = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 3,
            "Stock": ["A", "B", "C"],
            "MarketKey": ["sp500"] * 3,
            "Market": ["US"] * 3,
            "raw_alpha": [0.1, 0.3, -0.2],
        }
    )
    converted = raw_alpha_to_forecast(predictions).set_index("Stock")
    assert converted.loc["C", "forecast"] == -20.0
    assert converted.loc["A", "forecast"] == 0.0
    assert converted.loc["B", "forecast"] == 20.0


def test_top_long_forecast_only_keeps_top_selection_fraction() -> None:
    predictions = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 10,
            "Stock": [f"S{i}" for i in range(10)],
            "MarketKey": ["sp500"] * 10,
            "raw_alpha": list(range(10)),
        }
    )
    converted = raw_alpha_to_forecast(
        predictions,
        forecast_mode="top_long",
        selection_frac=0.10,
        selected_forecast=10.0,
    ).set_index("Stock")

    assert converted["forecast"].notna().sum() == 1
    assert converted.loc["S9", "forecast"] == 10.0
    assert converted.drop(index="S9")["forecast"].isna().all()


def test_top_bottom_forecast_keeps_top_and_bottom_selection_fraction() -> None:
    predictions = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 10,
            "Stock": [f"S{i}" for i in range(10)],
            "MarketKey": ["sp500"] * 10,
            "raw_alpha": list(range(10)),
        }
    )
    converted = raw_alpha_to_forecast(
        predictions,
        forecast_mode="top_bottom",
        selection_frac=0.10,
        selected_forecast=10.0,
    ).set_index("Stock")

    assert converted["forecast"].notna().sum() == 2
    assert converted.loc["S9", "forecast"] == 10.0
    assert converted.loc["S0", "forecast"] == -10.0
    assert converted.drop(index=["S0", "S9"])["forecast"].isna().all()


def test_score_smoothing_is_point_in_time() -> None:
    predictions = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")] * 2,
            "Stock": ["A", "A", "A", "B", "B", "B"],
            "MarketKey": ["sp500"] * 6,
            "raw_alpha": [0.0, 0.0, 100.0, 1.0, 1.0, 1.0],
        }
    )
    converted = raw_alpha_to_forecast(
        predictions,
        forecast_mode="top_long",
        selection_frac=0.50,
        score_smoothing_span=3,
    )
    first_day = converted[converted["Date"].eq(pd.Timestamp("2024-01-02"))].set_index("Stock")

    assert first_day.loc["B", "forecast"] == 10.0
    assert pd.isna(first_day.loc["A", "forecast"])


def test_exit_buffer_keeps_existing_name_until_it_leaves_buffer() -> None:
    predictions = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 10 + [pd.Timestamp("2024-01-03")] * 10,
            "Stock": [f"S{i}" for i in range(10)] * 2,
            "MarketKey": ["sp500"] * 20,
            "raw_alpha": list(range(10)) + [0, 1, 2, 3, 4, 5, 6, 7, 10, 9],
        }
    )
    converted = raw_alpha_to_forecast(
        predictions,
        forecast_mode="top_long",
        selection_frac=0.10,
        exit_frac=0.20,
        selected_forecast=10.0,
    )
    second_day = converted[converted["Date"].eq(pd.Timestamp("2024-01-03"))].set_index("Stock")

    assert second_day["forecast"].notna().sum() == 1
    assert second_day.loc["S9", "forecast"] == 10.0
    assert pd.isna(second_day.loc["S8", "forecast"])


def test_weekly_selection_holds_until_next_week() -> None:
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-08")]
    predictions = pd.DataFrame(
        {
            "Date": [date for date in dates for _ in range(10)],
            "Stock": [f"S{i}" for _date in dates for i in range(10)],
            "MarketKey": ["sp500"] * 30,
            "raw_alpha": list(range(10)) + [0, 1, 2, 3, 4, 5, 6, 7, 10, 9] + [0, 1, 2, 3, 4, 5, 6, 7, 10, 9],
        }
    )
    converted = raw_alpha_to_forecast(
        predictions,
        forecast_mode="top_long",
        selection_frac=0.10,
        selected_forecast=10.0,
        selection_interval="weekly",
    )
    same_week = converted[converted["Date"].eq(pd.Timestamp("2024-01-03"))].set_index("Stock")
    next_week = converted[converted["Date"].eq(pd.Timestamp("2024-01-08"))].set_index("Stock")

    assert same_week.loc["S9", "forecast"] == 10.0
    assert pd.isna(same_week.loc["S8", "forecast"])
    assert next_week.loc["S8", "forecast"] == 10.0
    assert pd.isna(next_week.loc["S9", "forecast"])
