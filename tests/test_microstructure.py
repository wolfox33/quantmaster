from __future__ import annotations

import numpy as np
import pandas as pd

from quantmaster.features.microstructure import (
    amihud_illiquidity,
    corwin_schultz_spread,
    order_flow_imbalance_range,
    relative_spread_proxy,
    roll_spread,
    vpin_proxy,
    vwap_deviation,
)
from tests.helpers import assert_no_lookahead


def test_microstructure_module_importable() -> None:
    import quantmaster.features.microstructure as microstructure

    assert microstructure is not None


def test_amihud_illiquidity_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    close = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    volume = pd.Series([1000] * len(idx), index=idx)
    df = pd.DataFrame({"close": close, "volume": volume}, index=idx)

    out = amihud_illiquidity(df, window=20)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "amihud_illiquidity_20"

    valid = out.dropna()
    assert np.isfinite(valid).all()

    assert_no_lookahead(feature_fn=amihud_illiquidity, data=df, t=30, feature_kwargs={"window": 20})


def test_roll_spread_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = roll_spread(df, window=30)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "roll_spread_30"

    valid = out.dropna()
    assert np.isfinite(valid).all()
    assert (valid >= 0).all()

    assert_no_lookahead(feature_fn=roll_spread, data=df, t=40, feature_kwargs={"window": 30})


def test_corwin_schultz_spread_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    high = close * 1.01
    low = close * 0.99
    df = pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)

    out = corwin_schultz_spread(df, window=10)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "corwin_schultz_spread_10"

    valid = out.dropna()
    assert np.isfinite(valid).all()
    assert (valid >= 0).all()

    assert_no_lookahead(feature_fn=corwin_schultz_spread, data=df, t=40, feature_kwargs={"window": 10})


def test_relative_spread_proxy_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    out = relative_spread_proxy(df, window=14)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "relative_spread_proxy_14"

    valid = out.dropna()
    assert np.isfinite(valid).all()
    assert (valid >= 0).all()

    assert_no_lookahead(feature_fn=relative_spread_proxy, data=df, t=40, feature_kwargs={"window": 14})


def test_vwap_deviation_shape_name_and_values() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "close": np.full(100, 110.0), # Constant price
            "volume": np.full(100, 1000.0),
        },
        index=idx,
    )
    
    # VWAP should be 110.0. Deviation should be 0.
    out = vwap_deviation(df, window=20)
    assert np.allclose(out.dropna(), 0.0)
    assert out.name == "vwap_deviation_20"
    
    # Rising price
    df["close"] = np.linspace(100, 200, 100)
    # Price is rising, so Price > VWAP (since VWAP lags). Deviation should be positive.
    out_rising = vwap_deviation(df, window=20)
    assert (out_rising.dropna() > 0).all()
    
    assert_no_lookahead(feature_fn=vwap_deviation, data=df, t=40, feature_kwargs={"window": 20})


def test_order_flow_imbalance_range_shape_name() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 200, 100),
            "close": np.linspace(101, 201, 100), # Always close > open (bullish)
            "high": np.linspace(102, 202, 100),
            "low": np.linspace(99, 199, 100),
            "volume": np.full(100, 1000.0),
        },
        index=idx,
    )
    # OFI should be positive
    out = order_flow_imbalance_range(df)
    assert (out.dropna() > 0).all()
    
    # Check name
    assert out.name == "order_flow_imbalance_range"


def test_vpin_proxy_shape_name() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "open": np.full(100, 100.0),
            "close": np.full(100, 101.0), # Always up
            "volume": np.full(100, 1000.0),
        },
        index=idx,
    )
    # Everything is buy volume. V_buy = 1000, V_sell = 0.
    # |1000 - 0| / (1000 + 0) = 1.0. 
    # VPIN should be 1.0
    out = vpin_proxy(df, window=20)
    assert np.allclose(out.dropna(), 1.0)
    
    # Mixed scenario
    df.iloc[::2, df.columns.get_loc("close")] = 99.0 # Alternating up/down
    # close < open (100)
    
    # V_buy = 1000 (even indices? no odd indices 1, 3 etc remain 101)
    # Actually wait:
    # 0: close 99 (sell)
    # 1: close 101 (buy)
    # Window 2: |V_buy - V_sell| = |1000 - 1000| = 0.
    # VPIN ~ 0.
    
    out_mixed = vpin_proxy(df, window=2)
    # Should be close to 0 (skipping first few)
    assert np.allclose(out_mixed.dropna(), 0.0)
    
    assert_no_lookahead(feature_fn=vpin_proxy, data=df, t=40, feature_kwargs={"window": 20})
