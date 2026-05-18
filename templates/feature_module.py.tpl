from __future__ import annotations

import numpy as np
import pandas as pd

from quantmaster.features.utils import get_price_series, validate_positive_int


def {{feature_name}}(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = {{default_window}},
    price_col: str = "close",
) -> pd.Series:
    window = validate_positive_int(window, name="window")

    price = get_price_series(data, price_col=price_col).astype(float)
    price = price.where(price > 0)
    rets = np.log(price).diff()

    # TODO: replace with final feature logic.
    out = rets.rolling(window).mean()
    out.name = "{{feature_name}}_{window}"
    return out

