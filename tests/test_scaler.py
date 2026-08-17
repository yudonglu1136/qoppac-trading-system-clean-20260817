from __future__ import annotations

import numpy as np
import pandas as pd

from cross_sectional_nn.normalization import standard_preprocessor


def test_standard_preprocessor_is_fit_on_train_only() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    validation = pd.DataFrame({"x": [100.0]})
    scaler = standard_preprocessor()
    scaler.fit(train)
    transformed = scaler.transform(validation)
    expected = (100.0 - 2.0) / np.std([1.0, 2.0, 3.0], ddof=0)
    assert np.isclose(transformed[0, 0], expected)

