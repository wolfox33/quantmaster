from __future__ import annotations

import numpy as np
import pandas as pd

from tests.helpers import assert_no_lookahead
from quantmaster.features.{{module_name}} import {{feature_name}}


def test_{{feature_name}}_shape_name_and_lookahead() -> None:
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.random.normal(0, 0.5, n)), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = {{feature_name}}(df, window={{default_window}})

    assert len(out) == len(df)
    assert out.index.equals(df.index)
    assert out.name == "{{feature_name}}_{{default_window}}"

    valid = out.dropna()
    assert np.isfinite(valid).all()

    assert_no_lookahead(
        feature_fn={{feature_name}},
        data=df,
        t=80,
        feature_kwargs={"window": {{default_window}}},
    )

