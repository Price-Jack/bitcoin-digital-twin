"""Data ingestion utilities."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_data(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance via yfinance.

    Returns a DataFrame indexed by timestamp with columns:
    Open, High, Low, Close, Adj Close, Volume
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df.dropna()
