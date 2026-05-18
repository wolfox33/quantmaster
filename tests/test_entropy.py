import numpy as np
import pandas as pd

from quantmaster.features.entropy import permutation_entropy, shannon_entropy


def test_shannon_entropy_basic() -> None:
    # Create a predictable series (low entropy) and a random series (high entropy)
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    
    # Constant prices -> 0 returns -> 0 entropy
    df_const = pd.DataFrame({"close": np.full(100, 100.0)}, index=idx)
    # returns will be constant 0.0
    
    out_const = shannon_entropy(df_const, window=20, bins=5, log_returns=True)
    # Entropy of constant value is 0 (all points in one bin)
    assert np.isclose(out_const.iloc[-1], 0.0, atol=0.1) 
    
    # Random returns
    np.random.seed(42)
    # returns ~ N(0, 1)
    rets = np.random.normal(0, 0.01, 100)
    prices = 100 * np.exp(np.cumsum(rets))
    df_rand = pd.DataFrame({"close": prices}, index=idx)
    
    out_rand = shannon_entropy(df_rand, window=50, bins=10)
    # Entropy should be > 0. Max entropy for 10 bins is log2(10) ≈ 3.32
    # Uniform dist has max entropy. Normal has less.
    assert out_rand.iloc[-1] > 1.0


def test_permutation_entropy_basic() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    
    # Note: permutation_entropy uses diff/returns by default? 
    # Yes, checks log_returns=True default.
    # log(x).diff() for monotonic x is positive, but not necessarily constant.
    # Let's use x directly with log_returns=False if we want to test prices
    # But usually applied to returns.
    
    # If returns are monotonic increasing?
    # prices = exp(t^2)? returns = 2t. Monotonic increasing returns.
    # Pattern is always (0, 1, 2). Entropy 0.
    
    # Simpler: pass Series directly?
    # function takes data frame and computes returns inside.
    # to control input to PE calculation exactly, we can pass pre-computed returns as `data` and ensure `log_returns=False` (if input is returns)
    # Wait, function computes returns from `price_col`.
    
    # Let's just use log_returns=False and pass a series that is strictly monotonic
    # returns = [1, 2, 3, 4...]
    
    df_test = pd.DataFrame({"close": range(1, 51)}, index=idx)
    # returns: 1/1, 1/2, 1/3... decreasing returns!
    # so pattern of returns is monotonic decreasing.
    # (2, 1, 0). 1 distinct pattern. Entropy 0.
    
    out = permutation_entropy(df_test, window=10, order=3, log_returns=False)
    # First few are NaNs
    assert np.isnan(out.iloc[0])
    # Last one should be 0 because returns are monotonic decreasing
    assert np.isclose(out.iloc[-1], 0.0, atol=1e-5)


def test_entropy_inputs() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "close": np.random.rand(20) + 10,
            "volume": np.random.rand(20) * 100,
        },
        index=idx,
    )
    
    out = shannon_entropy(df, window=10)
    assert len(out) == 20
    assert out.name.startswith("shannon_entropy")
    
    out_pe = permutation_entropy(df, window=10)
    assert len(out_pe) == 20
    assert out_pe.name.startswith("permutation_entropy")
