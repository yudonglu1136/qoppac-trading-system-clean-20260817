from __future__ import annotations

import pandas as pd

from cross_sectional_nn.walk_forward import FoldSpec, purged_train_mask, validation_mask


def test_purge_removes_training_labels_overlapping_validation_window() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2019-12-26", "2019-12-30", "2019-12-31", "2020-01-02"]),
            "label_end_date": pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-31"]),
        }
    )
    fold = FoldSpec(
        "test",
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2019-12-31"),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-12-31"),
        20,
    )
    mask = purged_train_mask(frame, fold)
    assert mask.tolist() == [True, False, False, False]
    assert validation_mask(frame, fold).tolist() == [False, False, False, True]

