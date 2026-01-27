import time
import inspect
import logging
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import quantmaster.features as qf

# Functions that require two series (asset, benchmark)
DUAL_SERIES_FEATURES = {
    "rolling_beta",
    "downside_beta",
    "spread_zscore",
    # "volatility_ratio", # Actually takes 1 arg (df usually)
}

# Functions that might return a DataFrame or have specific complex inputs
# We will try to pass the full DataFrame to these first.
DATAFRAME_FEATURES = {
    "path_signature_features", 
    "ornstein_uhlenbeck",
    "fracdiff",
}

def setup_logger():
    log_path = Path(__file__).parent / "feature_performance.log"
    logger = logging.getLogger("feature_benchmark")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_path, mode='w') # Overwrite each time
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger, log_path

@pytest.fixture
def benchmark_data():
    """Generates a sample DataFrame for benchmarking."""
    n_rows = 1000
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    np.random.seed(42)
    
    # Random walk for close
    rets = np.random.normal(0, 0.01, n_rows)
    close = 100 * np.exp(np.cumsum(rets))
    
    # Synthetic OHLCV
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_rows)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_rows)))
    open_ = close + np.random.normal(0, 0.002, n_rows) # Simplified
    volume = np.random.poisson(1000, n_rows) * np.abs(np.random.normal(1, 0.5, n_rows))
    
    # Asset and Benchmark for dual-series features
    bench_rets = np.random.normal(0, 0.008, n_rows)
    bench = 100 * np.exp(np.cumsum(bench_rets))
    
    # Additional column for cross-entropy etc
    y = close * 1.05 + np.random.normal(0, 1, n_rows)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "asset": close, # Alias for clarity in logs/logic
        "bench": bench,
        "x": close,
        "y": y
    }, index=idx)
    
    return df

def test_log_feature_performance(benchmark_data, capsys):
    """
    Benchmarks all features exposed in quantmaster.features
    and logs their execution time.
    """
    logger, log_path = setup_logger()
    logger.info("Starting Feature Benchmark")
    logger.info(f"Data shape: {benchmark_data.shape}")
    
    print(f"\n{'='*60}")
    print(f"Running Feature Benchmarks (Log: {log_path})")
    print(f"{'='*60}")
    print(f"{'Feature Name':<40} | {'Time (s)':<15}")
    print(f"{'-'*40}-|-{'-'*15}")

    all_features = qf.__all__
    
    for feature_name in sorted(all_features):
        feature_fn = getattr(qf, feature_name)
        
        # Determine arguments
        kwargs = {} 
        # Add minimal required args for known functions if they don't have defaults
        # This part is 'best effort' based on common params
        
        # Inspection could resolve defaults, but let's try calling with standard sets
        
        start_time = time.monotonic()
        try:
            # Dispatch based on known types or try-fail strategy
            if feature_name in DUAL_SERIES_FEATURES:
                # Expects two series
                feature_fn(benchmark_data["asset"], benchmark_data["bench"], **kwargs)

            else:
                # Try DataFrame first (preferred for OHLCV features)
                try:
                    feature_fn(benchmark_data, **kwargs)
                except (TypeError, ValueError, KeyError):
                    # Fallback to Series (usually close)
                    try:
                        feature_fn(benchmark_data["close"], **kwargs)
                    except Exception as e:
                        # Some might need volume or other specific columns if passed a Series? 
                        # Unlikely, usually if it needs volume it takes a DataFrame.
                        # Let's try passing arguments explicitly if simple call failed.
                         raise e

            end_time = time.monotonic()
            duration = end_time - start_time
            
            logger.info(f"{feature_name}: {duration:.6f}s")
            print(f"{feature_name:<40} | {duration:.6f}")

        except Exception as e:
            logger.error(f"FAILED {feature_name}: {e}")
            print(f"{feature_name:<40} | FAILED: {str(e)[:40]}...")

    print(f"{'='*60}")
    assert log_path.exists()
