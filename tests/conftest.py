from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


def pytest_configure() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


@pytest.fixture(scope="session")
def btc_1d_df() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "tests" / "data" / "btc_1d.csv"
    if not csv_path.exists():
        pytest.skip(f"Historical dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in btc_1d.csv: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    return df
