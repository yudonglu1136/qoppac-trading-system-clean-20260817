import pandas as pd

from cross_sectional_nn.models import add_cross_sectional_training_targets


def test_top_decile_training_target_marks_highest_future_alpha() -> None:
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 10,
            "Stock": [f"S{i}" for i in range(10)],
            "MarketKey": ["sp500"] * 10,
            "target": list(range(10)),
        }
    )

    labelled = add_cross_sectional_training_targets(frame, selection_frac=0.10, rank_bins=10).set_index("Stock")

    assert labelled["target_top_decile"].sum() == 1
    assert labelled.loc["S9", "target_top_decile"] == 1
    assert labelled.drop(index="S9")["target_top_decile"].eq(0).all()
    assert labelled.loc["S0", "target_rank_grade"] == 0
    assert labelled.loc["S9", "target_rank_grade"] == 9


def test_top_decile_training_target_is_grouped_by_market_and_date() -> None:
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-02")] * 6 + [pd.Timestamp("2024-01-03")] * 6,
            "Stock": [f"S{i}" for i in range(6)] * 2,
            "MarketKey": ["sp500"] * 6 + ["eem"] * 6,
            "target": [0, 1, 2, 3, 4, 5, 10, 9, 8, 7, 6, 5],
        }
    )

    labelled = add_cross_sectional_training_targets(frame, selection_frac=0.20, rank_bins=5)

    selected = labelled[labelled["target_top_decile"].eq(1)]
    assert set(selected["Stock"]) == {"S5", "S4", "S0", "S1"}
    assert selected.groupby(["MarketKey", "Date"]).size().eq(2).all()
