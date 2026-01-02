"""Feature engineering for BTC digital twin."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame, horizon: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Create feature matrix and targets.

    Targets:
    - y_reg: future return over `horizon`
    - y_class: 1 if future return > 0 else 0

    Returns:
    (x, y_reg, y_class, df_features)
    """
    data = df.copy()

    # If Close somehow becomes a DataFrame, squeeze to Series
    if isinstance(data["Close"], pd.DataFrame):
        data["Close"] = data["Close"].iloc[:, 0]

    # basic return (one step)
    data["return_1"] = data["Close"].pct_change()

    # targets
    data["future_close"] = data["Close"].shift(-horizon)
    data["future_return"] = (data["future_close"] - data["Close"]) / data["Close"]
    data["future_up"] = (data["future_return"] > 0).astype(int)

    # simple moving averages
    data["sma10"] = data["Close"].rolling(window=10).mean()
    data["sma20"] = data["Close"].rolling(window=20).mean()
    data["ema10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["ema20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # volatility
    data["vol10"] = data["return_1"].rolling(window=10).std()
    data["vol20"] = data["return_1"].rolling(window=20).std()

    # momentum
    data["mom5"] = data["return_1"].rolling(window=5).mean()
    data["mom10"] = data["return_1"].rolling(window=10).mean()

    # RSI
    data["rsi14"] = rsi(data["Close"], window=14)

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

    # Drop rows with NaNs created by rolling windows and horizon shift
    data = data.dropna()

    feature_cols = [
        "return_1",
        "sma10", "sma20", "ema10", "ema20",
        "vol10", "vol20",
        "mom5", "mom10",
        "rsi14",
        "macd", "macd_signal",
        "Volume",
    ]

    x = data[feature_cols].values
    y_reg = data["future_return"].values
    y_class = data["future_up"].values

    return x, y_reg, y_class, data


def make_seqs(x_arr: np.ndarray, y_arr: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert (N, D) features into (N-lookback, lookback, D) sequences."""
    seq_x, seq_y = [], []
    for i in range(lookback, len(x_arr)):
        seq_x.append(x_arr[i - lookback : i])
        seq_y.append(y_arr[i])
    return np.array(seq_x), np.array(seq_y)
