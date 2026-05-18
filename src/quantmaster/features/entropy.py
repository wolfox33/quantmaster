from __future__ import annotations

import numpy as np
import pandas as pd

from quantmaster.features.utils import get_price_series, validate_positive_int


def shannon_entropy(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    bins: int = 10,
    price_col: str = "close",
    log_returns: bool = True,
) -> pd.Series:
    """
    Rolling Shannon Entropy of returns.

    Measures the uncertainty/disorder in the distribution of returns.
    High entropy indicates random/noise regime.
    Low entropy indicates predictable/trend regime.
    """
    window = validate_positive_int(window, name="window")
    bins = validate_positive_int(bins, name="bins")

    price = get_price_series(data, price_col=price_col).astype(float)
    price = price.where(price > 0)

    if log_returns:
        x = np.log(price).diff()
    else:
        x = price.pct_change()

    # Pre-allocate output
    out = pd.Series(np.nan, index=price.index, dtype=float)
    out.name = f"shannon_entropy_{window}_{bins}"

    if len(x) < window:
        return out

    x_arr = x.to_numpy(dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(x_arr, window_shape=window)

    entropies = np.full(windows.shape[0], np.nan, dtype=float)

    for i in range(windows.shape[0]):
        w = windows[i]
        w = w[np.isfinite(w)]
        if w.size < 2:
            continue
        
        hist, _ = np.histogram(w, bins=bins, density=True)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        
        if probs.size > 0:
            entropies[i] = -np.sum(probs * np.log2(probs))

    out.iloc[window - 1 :] = entropies
    return out


def permutation_entropy(
    data: pd.DataFrame | pd.Series,
    *,
    window: int = 60,
    order: int = 3,
    delay: int = 1,
    price_col: str = "close",
    log_returns: bool = True,
) -> pd.Series:
    """
    Rolling Permutation Entropy (Bandt & Pompe).

    Measures the complexity of the time series based on ordinal patterns.
    Robust to outliers and captures non-linear dynamics.
    """
    window = validate_positive_int(window, name="window")
    order = validate_positive_int(order, name="order")
    delay = validate_positive_int(delay, name="delay")

    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    
    # Require window to be large enough for patterns
    # Although technically possible, very small windows yield poor estimates
    
    price = get_price_series(data, price_col=price_col).astype(float)
    price = price.where(price > 0)

    if log_returns:
        x = np.log(price).diff()
    else:
        x = price.pct_change()

    out = pd.Series(np.nan, index=price.index, dtype=float)
    out.name = f"permutation_entropy_{window}_{order}_{delay}"

    if len(x) < window:
        return out

    x_arr = x.to_numpy(dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(x_arr, window_shape=window)

    pe_vals = np.full(windows.shape[0], np.nan, dtype=float)

    for i in range(windows.shape[0]):
        w = windows[i]
        # Drop NaNs for this window calculation? 
        # PE usually requires contiguous data. If we drop NaNs, we change time structure.
        # We'll forward fill or just check finite.
        # If too many NaNs, skip.
        
        if not np.isfinite(w).all():
             # Basic handling: if any NaN, result is NaN. 
             # Or could try to interpolate/fill. For now, strict.
             continue

        # Extract patterns
        n = w.size
        if n < order + (order - 1) * (delay - 1):
             continue

        # Create partitions
        # We need sub-windows of length 'order' with stride 'delay'
        # partitions shape: (num_patterns, order)
        # stride for sliding window within 'w'
        
        # Optimized way to get patterns:
        # We want w[j], w[j+delay], ..., w[j+(order-1)*delay]
        # Total span required = 1 + (order-1)*delay
        
        span = 1 + (order - 1) * delay
        num_patterns = n - span + 1
        
        if num_patterns < 1:
            continue
            
        # Use simple loop or stride_tricks again for efficiency
        # Construct matrix of patterns
        pattern_matrix = np.empty((num_patterns, order), dtype=w.dtype)
        for k in range(order):
            pattern_matrix[:, k] = w[k*delay : k*delay + num_patterns]
            
        # Argsort to get ordinal patterns (indices of sorted elements)
        # axis=1
        ordinal_patterns = np.argsort(pattern_matrix, axis=1)
        
        # Convert to tuple or unique representation to count
        # We can view the rows as void type for uniqueness
        dtype_view = np.dtype((np.void, ordinal_patterns.dtype.itemsize * ordinal_patterns.shape[1]))
        patterns_void = np.ascontiguousarray(ordinal_patterns).view(dtype_view)
        
        _, counts = np.unique(patterns_void, return_counts=True)
        probs = counts / num_patterns
        probs = probs[probs > 0]
        
        pe = -np.sum(probs * np.log2(probs))
        
        # Normalize? Usually PE is normalized by log2(factorial(order))
        # But standard def is unnormalized. Let's return unnormalized to match basic entropy unit (bits).
        # Or normalize to [0, 1]. The prompt implies standard entropy formulation.
        # "SE_t = ...", "PE_t = ..."
        # I'll stick to raw bits entropy.
        
        pe_vals[i] = pe

    out.iloc[window - 1 :] = pe_vals
    return out
