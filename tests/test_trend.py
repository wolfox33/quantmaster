from __future__ import annotations

import numpy as np
import pandas as pd

from quantmaster.features.trend import (
    kaufman_efficiency_ratio,
    price_acceleration,
    trend_intensity,
    trend_strength_autocorr,
    trend_strength_indicator,
)
from tests.helpers import assert_no_lookahead


def test_trend_module_importable() -> None:
    import quantmaster.features.trend as trend

    assert trend is not None


def test_trend_intensity_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.linspace(-0.2, 0.4, len(idx))), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = trend_intensity(df, window=20)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "trend_intensity_20"

    valid = out.dropna()
    assert np.isfinite(valid).all()
    assert (valid >= 0).all()
    assert (valid <= 1.0 + 1e-12).all()

    assert_no_lookahead(feature_fn=trend_intensity, data=df, t=60, feature_kwargs={"window": 20})


def test_kaufman_efficiency_ratio_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=160, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.linspace(-0.2, 0.4, len(idx))), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = kaufman_efficiency_ratio(df, window=10)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "kaufman_efficiency_ratio_10"

    valid = out.dropna()
    assert np.isfinite(valid).all()
    assert (valid >= 0).all()
    assert (valid <= 1.0 + 1e-12).all()

    assert_no_lookahead(
        feature_fn=kaufman_efficiency_ratio,
        data=df,
        t=60,
        feature_kwargs={"window": 10},
    )


def test_price_acceleration_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=220, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.linspace(-0.2, 0.4, len(idx))), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = price_acceleration(df, window=20)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "price_acceleration_20"

    valid = out.dropna()
    assert np.isfinite(valid).all()

    assert_no_lookahead(feature_fn=price_acceleration, data=df, t=80, feature_kwargs={"window": 20})


def test_trend_strength_indicator_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=420, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.linspace(-0.2, 0.6, len(idx))), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = trend_strength_indicator(df, windows=[21, 63])

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "trend_strength_indicator_21_63"

    valid = out.dropna()
    assert np.isfinite(valid).all()

    assert_no_lookahead(feature_fn=trend_strength_indicator, data=df, t=150, feature_kwargs={"windows": [21, 63]})


def test_trend_strength_autocorr_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=240, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.linspace(-0.1, 0.3, len(idx))), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = trend_strength_autocorr(df, window=60, lag=1)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "trend_strength_autocorr_60_1"

    valid = out.dropna()
    assert np.isfinite(valid).all()

    assert_no_lookahead(
        feature_fn=trend_strength_autocorr,
        data=df,
        t=120,
        feature_kwargs={"window": 60, "lag": 1},
    )
