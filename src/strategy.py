"""Simple strategy/backtest utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def confidence_weighted_positions(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Map probabilities to long/short/flat position.

    Matches the notebook's style:
    - If prob >= threshold: long with size proportional to confidence
    - If prob <= (1-threshold): short with size proportional to confidence
    - Else: flat
    """
    positions = np.zeros_like(probs, dtype=float)
    for i, p in enumerate(probs):
        if p >= threshold:
            positions[i] = (p - threshold) / max(1e-8, (1 - threshold))  # 0..1
        elif p <= 1 - threshold:
            positions[i] = -((1 - threshold) - p) / max(1e-8, (1 - threshold))  # 0..-1
        else:
            positions[i] = 0.0
    return positions


def equity_curves_from_positions(
    prices: pd.Series,
    positions: np.ndarray,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Compute strategy and buy&hold equity curves.

    Positions are shifted by 1 step to avoid trading on the same bar prediction.
    """
    returns = prices.pct_change().fillna(0).values

    positions_shifted = np.roll(positions, 1)
    positions_shifted[0] = 0.0

    strategy_returns = positions_shifted * returns
    baseline_returns = returns

    strategy_equity = pd.Series((1 + strategy_returns).cumprod(), index=prices.index, name="Strategy")
    baseline_equity = pd.Series((1 + baseline_returns).cumprod(), index=prices.index, name="Buy & Hold")

    return strategy_equity, baseline_equity, strategy_returns, baseline_returns


def compare_returns(strategy_returns: np.ndarray, baseline_returns: np.ndarray) -> dict:
    """Welch's t-test to compare mean returns."""
    t_stat, p_value = ttest_ind(strategy_returns, baseline_returns, equal_var=False)
    return {
        "strategy_avg": float(np.mean(strategy_returns)),
        "strategy_vol": float(np.std(strategy_returns)),
        "baseline_avg": float(np.mean(baseline_returns)),
        "baseline_vol": float(np.std(baseline_returns)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }
