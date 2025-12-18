from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def assert_no_lookahead(
    *,
    feature_fn: Callable[..., pd.Series | pd.DataFrame],
    data: pd.DataFrame,
    t: int,
    feature_kwargs: dict[str, Any] | None = None,
) -> None:
    if t < 0:
        raise ValueError("t must be >= 0")
    if t >= len(data):
        raise ValueError("t must be < len(data)")

    kwargs = feature_kwargs or {}

    out1 = feature_fn(data.copy(), **kwargs)

    mutated = data.copy()
    if t + 1 < len(mutated):
        numeric_cols = [c for c in mutated.columns if pd.api.types.is_numeric_dtype(mutated[c])]
        mutated.loc[mutated.index[t + 1 :], numeric_cols] = mutated.loc[mutated.index[t + 1 :], numeric_cols] * 10.0

    out2 = feature_fn(mutated, **kwargs)

    if isinstance(out1, pd.Series):
        if not isinstance(out2, pd.Series):
            raise TypeError("feature_fn returned different types across calls")
        pd.testing.assert_series_equal(out1.iloc[: t + 1], out2.iloc[: t + 1], check_names=True)
        return

    if not isinstance(out2, pd.DataFrame):
        raise TypeError("feature_fn returned different types across calls")
    pd.testing.assert_frame_equal(out1.iloc[: t + 1], out2.iloc[: t + 1], check_names=True)
