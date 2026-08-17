from __future__ import annotations

import numpy as np
import pandas as pd

from cross_sectional_nn.targets import forward_log_return, label_end_dates


def test_forward_log_return_uses_future_horizon_only_for_target() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    price = pd.Series([100.0, 101.0, 104.0, 108.0, 109.0], index=index)
    target = forward_log_return(price, 2)
    assert np.isclose(target.iloc[0], np.log(104.0 / 100.0))
    assert np.isclose(target.iloc[1], np.log(108.0 / 101.0))
    assert pd.isna(target.iloc[-1])


def test_label_end_dates_match_forward_horizon() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    labels = label_end_dates(index, 2)
    assert labels.iloc[0] == index[2]
    assert labels.iloc[2] == index[4]
    assert pd.isna(labels.iloc[3])

