from __future__ import annotations

import numpy as np
import pandas as pd

from quantmaster.features.momentum import momentum_volatility_state
from quantmaster.features.utils import create_all
from quantmaster.features.volatility import relative_jump_contribution
from tests.helpers import assert_no_lookahead


def test_btc_dataset_contract(btc_1d_df: pd.DataFrame) -> None:
    required = ["open", "high", "low", "close", "volume"]
    assert all(c in btc_1d_df.columns for c in required)
    assert btc_1d_df.index.is_monotonic_increasing
    assert btc_1d_df.index.duplicated().sum() == 0
    assert np.isfinite(btc_1d_df[required].to_numpy()).all()


def test_btc_relative_jump_contribution_no_lookahead_and_range(btc_1d_df: pd.DataFrame) -> None:
    out = relative_jump_contribution(btc_1d_df, window=20)

    assert isinstance(out, pd.Series)
    assert out.index.equals(btc_1d_df.index)
    assert out.name == "relative_jump_contribution_20"

    valid = out.dropna()
    assert not valid.empty
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()

    t = min(500, len(btc_1d_df) - 2)
    assert_no_lookahead(
        feature_fn=relative_jump_contribution,
        data=btc_1d_df,
        t=t,
        feature_kwargs={"window": 20},
    )


def test_btc_relative_jump_contribution_no_repaint(btc_1d_df: pd.DataFrame) -> None:
    base = btc_1d_df.iloc[:1200].copy()
    extended = btc_1d_df.iloc[:1500].copy()

    out_base = relative_jump_contribution(base, window=20)
    out_extended = relative_jump_contribution(extended, window=20)

    pd.testing.assert_series_equal(
        out_base,
        out_extended.loc[out_base.index],
        check_names=True,
    )


def test_btc_create_all_includes_momentum_volatility_state_for_experimental_status(
    btc_1d_df: pd.DataFrame,
) -> None:
    feature_col = "momentum_volatility_state_20_20_60"
    base = btc_1d_df.iloc[:1200].copy()
    extended = btc_1d_df.iloc[:1500].copy()

    out_base = create_all(
        base,
        include=["momentum_volatility_state"],
        include_statuses=["experimental"],
        errors="raise",
    )
    out_extended = create_all(
        extended,
        include=["momentum_volatility_state"],
        include_statuses=["experimental"],
        errors="raise",
    )

    assert feature_col in out_base.columns
    assert feature_col in out_extended.columns

    series_base = out_base[feature_col]
    series_extended = out_extended.loc[series_base.index, feature_col]
    valid = series_base.dropna()
    assert not valid.empty
    assert np.isfinite(valid.to_numpy()).all()

    pd.testing.assert_series_equal(series_base, series_extended, check_names=True)

    t = min(500, len(btc_1d_df) - 2)
    assert_no_lookahead(
        feature_fn=momentum_volatility_state,
        data=btc_1d_df,
        t=t,
        feature_kwargs={"mom_window": 20, "vol_window": 20, "state_window": 60},
    )
