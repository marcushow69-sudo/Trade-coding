"""Technical indicators and the ML feature set.

Every value at row t only uses information available at the close of bar t,
so there is no look-ahead when a signal at t is executed at the open of t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return mid - k * sd, mid, mid + k * sd


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    a = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features for the ML model (no raw prices, so it generalises across price levels)."""
    c = df["close"]
    a = atr(df, 14)
    f = pd.DataFrame(index=df.index)

    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["vol_12"] = f["ret_1"].rolling(12).std()
    f["vol_48"] = f["ret_1"].rolling(48).std()
    f["vol_ratio"] = f["vol_12"] / f["vol_48"]

    f["atr_pct"] = a / c
    f["rsi_14"] = rsi(c, 14) / 100
    f["rsi_4"] = rsi(c, 4) / 100
    line, sig, hist = macd(c)
    f["macd_hist"] = hist / a
    f["macd_line"] = line / a
    for n in (20, 50, 200):
        f[f"dist_ema{n}"] = (c - ema(c, n)) / a
    f["ema20_slope"] = ema(c, 20).diff(5) / a
    lo, mid, hi = bollinger(c)
    f["bb_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f["adx"] = adx(df) / 100
    f["range_pos_24"] = (c - df["low"].rolling(24).min()) / (
        df["high"].rolling(24).max() - df["low"].rolling(24).min()
    ).replace(0, np.nan)
    body = (c - df["open"]) / a
    f["body"] = body
    f["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / a
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / a

    # Calendar / session effects (gold moves differently in Asia vs London/NY).
    idx = pd.DatetimeIndex(df.index)
    if (idx.hour != 0).any():
        f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow"] = idx.dayofweek / 6
    return f.replace([np.inf, -np.inf], np.nan)
