"""Rule-based baseline strategies.

Use these as benchmarks: an AI model is only interesting if it beats simple rules
after costs. Each function takes OHLC data and returns a signal Series in {-1, 0, 1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger, ema, rsi


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index)


def ema_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    return pd.Series(np.where(f > s, 1, -1), index=df.index)


def rsi_reversion(df: pd.DataFrame, n: int = 14, lower: float = 30, upper: float = 70,
                  trend_filter: int = 200) -> pd.Series:
    """Buy oversold dips in an uptrend, sell overbought rallies in a downtrend."""
    r = rsi(df["close"], n)
    trend = ema(df["close"], trend_filter)
    sig = pd.Series(0, index=df.index)
    sig[(r < lower) & (df["close"] > trend)] = 1
    sig[(r > upper) & (df["close"] < trend)] = -1
    return sig


def donchian_breakout(df: pd.DataFrame, n: int = 20, adx_min: float = 20) -> pd.Series:
    """Trade breakouts of the n-bar range when ADX says the market is trending; hold until opposite break."""
    hi = df["high"].rolling(n).max().shift()
    lo = df["low"].rolling(n).min().shift()
    strong = adx(df) > adx_min
    raw = pd.Series(np.nan, index=df.index)
    raw[(df["close"] > hi) & strong] = 1
    raw[(df["close"] < lo) & strong] = -1
    return raw.ffill().fillna(0).astype(int)


def bollinger_reversion(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    lo, mid, hi = bollinger(df["close"], n, k)
    raw = pd.Series(np.nan, index=df.index)
    raw[df["close"] < lo] = 1
    raw[df["close"] > hi] = -1
    raw[((df["close"] > mid) & (df["close"].shift() <= mid.shift())) |
        ((df["close"] < mid) & (df["close"].shift() >= mid.shift()))] = 0
    return raw.ffill().fillna(0).astype(int)


STRATEGIES = {
    "buy_hold": buy_and_hold,
    "ema_cross": ema_cross,
    "rsi_reversion": rsi_reversion,
    "donchian": donchian_breakout,
    "bollinger": bollinger_reversion,
}
