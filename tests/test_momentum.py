import numpy as np
import pandas as pd

from quantmaster.features.momentum import (
    chande_momentum_oscillator,
    momentum_volatility_state,
    rsi,
    time_series_momentum,
)
from tests.helpers import assert_no_lookahead


def _assert_no_repaint_series(
    *,
    feature_fn,
    data: pd.DataFrame,
    feature_kwargs: dict | None = None,
    cutoff: int,
) -> None:
    kwargs = feature_kwargs or {}
    out1 = feature_fn(data.copy(), **kwargs)

    future_idx = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=20, freq="D")
    tail = float(data["close"].iloc[-1])
    future_close = pd.Series(np.linspace(tail, tail + 4.0, len(future_idx)), index=future_idx)
    extended = pd.concat([data, pd.DataFrame({"close": future_close}, index=future_idx)])

    out2 = feature_fn(extended, **kwargs)
    pd.testing.assert_series_equal(out1.iloc[: cutoff + 1], out2.iloc[: cutoff + 1], check_names=True)


def test_rsi_shape_and_name() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "open": range(1, 51),
            "high": range(2, 52),
            "low": range(0, 50),
            "close": range(1, 51),
            "volume": [100] * 50,
        },
        index=idx,
    )

    out = rsi(df, window=14)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "rsi_14"

    bounded = out.dropna()
    assert (bounded >= 0).all()
    assert (bounded <= 100).all()


def test_time_series_momentum_shape_name_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=160, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.linspace(-0.1, 0.3, len(idx))), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = time_series_momentum(df, lookback=30, volatility_window=10)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "time_series_momentum_30_10"

    valid = out.dropna()
    assert np.isfinite(valid).all()

    assert_no_lookahead(
        feature_fn=time_series_momentum,
        data=df,
        t=60,
        feature_kwargs={"lookback": 30, "volatility_window": 10},
    )


def test_chande_momentum_oscillator_shape_bounds_and_no_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(100.0 + np.cumsum(np.linspace(-0.3, 0.6, len(idx))), index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = chande_momentum_oscillator(df, window=14)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "chande_momentum_oscillator_14"

    valid = out.dropna()
    assert np.isfinite(valid).all()
    assert (valid >= -100).all()
    assert (valid <= 100).all()

    assert_no_lookahead(feature_fn=chande_momentum_oscillator, data=df, t=50, feature_kwargs={"window": 14})


def test_momentum_volatility_state_shape_no_lookahead_no_repaint() -> None:
    idx = pd.date_range("2022-01-01", periods=300, freq="D")
    base = np.linspace(100.0, 180.0, len(idx))
    noise = np.sin(np.linspace(0.0, 24.0, len(idx))) * 1.5
    close = pd.Series(base + noise, index=idx)
    df = pd.DataFrame({"close": close}, index=idx)

    out = momentum_volatility_state(df, mom_window=20, vol_window=20, state_window=60)

    assert isinstance(out, pd.Series)
    assert out.index.equals(df.index)
    assert out.name == "momentum_volatility_state_20_20_60"

    valid = out.dropna()
    assert np.isfinite(valid).all()

    assert_no_lookahead(
        feature_fn=momentum_volatility_state,
        data=df,
        t=180,
        feature_kwargs={"mom_window": 20, "vol_window": 20, "state_window": 60},
    )
    _assert_no_repaint_series(
        feature_fn=momentum_volatility_state,
        data=df,
        feature_kwargs={"mom_window": 20, "vol_window": 20, "state_window": 60},
        cutoff=180,
    )
