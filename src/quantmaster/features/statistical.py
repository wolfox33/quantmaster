from __future__ import annotations

import numpy as np
import pandas as pd

from quantmaster.features.utils import get_price_series, validate_positive_int


def _fracdiff_weights(d: float, *, thresh: float, max_lags: int) -> np.ndarray:
    weights: list[float] = [1.0]

    if max_lags < 1:
        return np.asarray(weights, dtype=float)

    w_prev = 1.0
    for k in range(1, max_lags + 1):
        w_k = -w_prev * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        weights.append(w_k)
        w_prev = w_k

    return np.asarray(weights, dtype=float)


def fracdiff(
    data: pd.DataFrame | pd.Series,
    *,
    d: float = 0.5,
    thresh: float = 1e-5,
    max_lags: int | None = None,
    price_col: str = "close",
) -> pd.Series:
    try:
        d = float(d)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"d must be float, got {type(d).__name__}") from exc

    if not (0.0 <= d <= 1.0):
        raise ValueError(f"d must be between 0 and 1 (inclusive), got {d}")

    try:
        thresh = float(thresh)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"thresh must be float, got {type(thresh).__name__}") from exc

    if thresh <= 0:
        raise ValueError(f"thresh must be > 0, got {thresh}")

    if max_lags is not None:
        max_lags = validate_positive_int(max_lags, name="max_lags")

    x = get_price_series(data, price_col=price_col).astype(float)

    out_name = f"fracdiff_{d:g}"

    if len(x) < 2:
        out = pd.Series(np.nan, index=x.index, name=out_name, dtype=float)
        if d == 0.0:
            out = x.astype(float).copy()
            out.name = out_name
        return out

    max_allowed_lags = len(x) - 1
    max_lags_eff = max_allowed_lags if max_lags is None else min(max_lags, max_allowed_lags)

    weights = _fracdiff_weights(d, thresh=thresh, max_lags=max_lags_eff)
    k = len(weights) - 1

    y = np.convolve(x.to_numpy(dtype=float), weights, mode="full")[: len(x)]
    if k > 0:
        y[:k] = np.nan

    out = pd.Series(y, index=x.index)
    out.name = out_name
    return out


def hurst_dfa(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 240,
    price_col: str = "close",
    min_scale: int = 4,
    max_scale: int | None = None,
    n_scales: int = 10,
    log_price: bool = True,
) -> pd.Series:
    window = validate_positive_int(window, name="window")
    min_scale = validate_positive_int(min_scale, name="min_scale")
    n_scales = validate_positive_int(n_scales, name="n_scales")

    if window < 8:
        raise ValueError(f"window must be >= 8, got {window}")

    if max_scale is None:
        max_scale_eff = max(min_scale, window // 4)
    else:
        max_scale_eff = validate_positive_int(max_scale, name="max_scale")

    max_scale_eff = min(max_scale_eff, window // 2)
    if max_scale_eff < min_scale:
        raise ValueError(f"max_scale must be >= min_scale, got max_scale={max_scale_eff}, min_scale={min_scale}")

    x = get_price_series(data, price_col=price_col).astype(float)
    if log_price:
        x = np.log(x.where(x > 0))

    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.name = f"hurst_dfa_{window}"

    if len(x) < window:
        return out

    scales = np.unique(
        np.floor(
            np.logspace(
                np.log10(float(min_scale)),
                np.log10(float(max_scale_eff)),
                int(n_scales),
            )
        ).astype(int)
    )
    scales = scales[(scales >= min_scale) & (scales <= max_scale_eff)]
    if scales.size < 2:
        return out

    x_arr = x.to_numpy(dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(x_arr, window_shape=window)

    mean_win = np.nanmean(windows, axis=1, keepdims=True)
    prof = np.nancumsum(windows - mean_win, axis=1)

    log_s = np.log(scales.astype(float))
    f_mat = np.full((prof.shape[0], scales.size), np.nan, dtype=float)

    for j, s in enumerate(scales.tolist()):
        m = window // int(s)
        if m < 2:
            continue

        y = prof[:, : m * s].reshape(prof.shape[0], m, s)

        t = np.arange(s, dtype=float)
        t_mean = t.mean()
        t_centered = t - t_mean
        denom = float(np.sum(t_centered**2))
        if denom == 0.0:
            continue

        y_mean = np.nanmean(y, axis=2, keepdims=True)
        slope = np.nansum(t_centered[None, None, :] * (y - y_mean), axis=2, keepdims=True) / denom
        intercept = y_mean - slope * t_mean

        trend = slope * t[None, None, :] + intercept
        res = y - trend
        f_mat[:, j] = np.sqrt(np.nanmean(res**2, axis=(1, 2)))

    log_f = np.log(f_mat)
    valid_scale = np.isfinite(log_s)
    valid_scale &= np.isfinite(log_f).all(axis=0)

    if valid_scale.sum() < 2:
        return out

    xs = log_s[valid_scale]
    ys = log_f[:, valid_scale]

    x_mean = xs.mean()
    x_centered = xs - x_mean
    x_var = float(np.sum(x_centered**2))
    if x_var == 0.0:
        return out

    y_mean = ys.mean(axis=1, keepdims=True)
    cov = np.sum((ys - y_mean) * x_centered[None, :], axis=1)
    hurst = cov / x_var

    out.iloc[window - 1 :] = hurst
    return out
