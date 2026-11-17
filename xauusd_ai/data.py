"""Loading XAUUSD price data.

Three sources:
  * CSV files (generic, MT4 or MT5 exports)       -> load_csv()
  * Yahoo Finance (gold futures GC=F as a proxy)  -> load_yahoo()
  * Synthetic random-walk data for offline tests  -> synthetic_xauusd()

Every loader returns a DataFrame indexed by a sorted DatetimeIndex with
lower-case columns: open, high, low, close, volume.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[OHLC + ["volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=OHLC)
    # Repair rows where high/low don't bracket open/close (bad ticks).
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df.index.name = "time"
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV in generic, MT4 or MT5 format (auto-detected)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text else ""

    # MT5: tab separated with <DATE> <TIME> headers
    if "<DATE>" in first.upper():
        sep = "\t" if "\t" in first else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
        df.columns = [c.strip("<>").lower() for c in df.columns]
        if "time" in df.columns:
            ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
            if ts.isna().all():
                ts = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
        else:
            ts = pd.to_datetime(df["date"], format="%Y.%m.%d", errors="coerce")
        vol = df["tickvol"] if "tickvol" in df.columns else df.get("vol", 0)
        out = pd.DataFrame(
            {"open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"], "volume": vol}
        )
        out.index = ts
        return _finalise(out)

    # MT4: no header, "2024.01.02,01:00,o,h,l,c,v"
    if first[:4].isdigit() and first[4:5] == ".":
        df = pd.read_csv(io.StringIO(text), header=None)
        ts = pd.to_datetime(df[0] + " " + df[1], format="%Y.%m.%d %H:%M")
        out = pd.DataFrame(
            {"open": df[2], "high": df[3], "low": df[4], "close": df[5], "volume": df[6] if 6 in df else 0}
        )
        out.index = ts
        return _finalise(out)

    # Generic CSV with a header row
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in ("time", "datetime", "date", "timestamp", "gmt time") if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Could not find a time column in {path}. Columns: {list(df.columns)}")
    df.index = pd.to_datetime(df[time_col], dayfirst=time_col == "gmt time")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return _finalise(df)


def load_yahoo(start: str = "2015-01-01", end: str | None = None, interval: str = "1d",
               ticker: str = "GC=F") -> pd.DataFrame:
    """Download gold prices from Yahoo Finance.

    Yahoo has no clean spot XAUUSD feed, so COMEX gold futures (GC=F) are used
    as a proxy. Intraday intervals (1h, 15m...) are limited to recent history.
    """
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No data returned from Yahoo for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    return _finalise(raw)


def synthetic_xauusd(n_bars: int = 5000, freq: str = "1h", start_price: float = 1800.0,
                     seed: int = 42) -> pd.DataFrame:
    """Generate gold-like OHLC data with trending/ranging regimes and volatility clustering.

    Useful for testing the pipeline offline. Results on synthetic data say
    nothing about the real market.
    """
    rng = np.random.default_rng(seed)
    regime = np.zeros(n_bars)
    drift = 0.0
    for i in range(n_bars):
        if rng.random() < 0.01:  # regime switch
            drift = rng.choice([-1, 0, 0, 1]) * 0.00012
        regime[i] = drift
    vol = np.empty(n_bars)
    vol[0] = 0.0025
    shocks = rng.standard_normal(n_bars)
    for i in range(1, n_bars):  # simple GARCH(1,1)-ish volatility
        vol[i] = np.sqrt(0.000001 + 0.08 * (vol[i - 1] * shocks[i - 1]) ** 2 + 0.9 * vol[i - 1] ** 2)
    rets = regime + vol * shocks
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.0002, n_bars))
    wick = np.abs(rng.normal(0, 1, (2, n_bars))) * vol * close * 0.6
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    idx = pd.date_range("2020-01-01", periods=n_bars, freq=freq)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.integers(100, 5000, n_bars)}, index=idx)
    return _finalise(df)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe, e.g. '4h' or '1D'."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=OHLC)
