"""Download real spot XAUUSD history from Dukascopy's free datafeed.

Dukascopy publishes one LZMA-compressed file of 1-minute BID candles per day:
    https://datafeed.dukascopy.com/datafeed/XAUUSD/YYYY/MM/DD/BID_candles_min_1.bi5
(MM is zero-based: January = 00). Each record is 24 bytes, big-endian:
    uint32 seconds-from-midnight-UTC, uint32 open, uint32 close, uint32 low, uint32 high, float32 volume
Prices are integers scaled by 1000 for XAUUSD.

Downloaded days are cached under data/cache/ so re-runs only fetch new days.
"""

from __future__ import annotations

import lzma
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
POINT = {"XAUUSD": 1000.0, "XAGUSD": 1000.0}
REC = struct.Struct(">5If")


def _fetch(url: str, retries: int = 12) -> bytes:
    """GET with patient backoff: Dukascopy throttles (HTTP 429/503, dropped connections)."""
    err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            err = e
            if e.code in (429, 503):
                time.sleep(20 + 10 * attempt)
                continue
        except Exception as e:  # noqa: BLE001 - network hiccups, retry
            err = e
        time.sleep(min(3 * 2 ** attempt, 60))
    raise RuntimeError(f"failed to download {url}: {err}")


def _decode(raw: bytes, day: date, point: float) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    buf = lzma.decompress(raw)
    arr = np.frombuffer(buf, dtype=np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"),
                                              ("h", ">u4"), ("v", ">f4")]))
    base = pd.Timestamp(day)
    df = pd.DataFrame({
        "open": arr["o"] / point, "high": arr["h"] / point, "low": arr["l"] / point,
        "close": arr["c"] / point, "volume": arr["v"].astype(float),
    }, index=base + pd.to_timedelta(arr["t"].astype(np.int64), unit="s"))
    return df[df["volume"] > 0]  # drop filler minutes when the market is closed


def download_minutes(start: str, end: str | None = None, symbol: str = "XAUUSD",
                     cache_dir: str | Path = "data/cache", workers: int = 2, verbose: bool = True) -> pd.DataFrame:
    """Return 1-minute candles for [start, end], downloading missing days in parallel."""
    point = POINT.get(symbol, 100000.0)
    cache = Path(cache_dir) / symbol
    cache.mkdir(parents=True, exist_ok=True)
    d0 = pd.Timestamp(start).date()
    d1 = pd.Timestamp(end).date() if end else (datetime.now(timezone.utc).date() - timedelta(days=1))
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    days = [d for d in days if d.weekday() != 5]  # gold doesn't trade on Saturdays

    def path(d: date) -> Path:
        return cache / f"{d:%Y%m%d}.bi5"

    todo = [d for d in days if not path(d).exists() or d >= d1 - timedelta(days=2)]
    if verbose:
        print(f"Dukascopy {symbol}: {len(days)} days requested, {len(todo)} to download")
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch, URL.format(sym=symbol, y=d.year, m=d.month - 1, d=d.day)): d for d in todo}
        for f in as_completed(futs):
            try:
                path(futs[f]).write_bytes(f.result())
            except RuntimeError as e:
                print(f"  skipped {futs[f]} ({e}); re-run to retry")
            done += 1
            if verbose and done % 20 == 0:
                print(f"  downloaded {done}/{len(todo)} days")

    frames = [_decode(path(d).read_bytes(), d, point) for d in days if path(d).exists()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("no data downloaded")
    df = pd.concat(frames).sort_index()
    df.index.name = "time"
    return df


HOUR_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/BID_candles_hour_1.bi5"


def _decode_hours(raw: bytes, month_start: pd.Timestamp, point: float) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    arr = np.frombuffer(lzma.decompress(raw), dtype=np.dtype(
        [("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")]))
    df = pd.DataFrame({
        "open": arr["o"] / point, "high": arr["h"] / point, "low": arr["l"] / point,
        "close": arr["c"] / point, "volume": arr["v"].astype(float),
    }, index=month_start + pd.to_timedelta(arr["t"].astype(np.int64), unit="s"))
    return df[df["volume"] > 0]


def download_hours(start: str = "2015-01-01", end: str | None = None, symbol: str = "XAUUSD",
                   cache_dir: str | Path = "data/cache", workers: int = 1, verbose: bool = True) -> pd.DataFrame:
    """Hourly candles: one file per completed month, plus minute files for the current month."""
    point = POINT.get(symbol, 100000.0)
    cache = Path(cache_dir) / symbol / "hourly"
    cache.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    this_month = pd.Timestamp(today.year, today.month, 1)
    months = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), this_month, freq="MS")
    full = [m for m in months if m < this_month]

    def path(m):
        return cache / f"{m:%Y%m}.bi5"

    todo = [m for m in full if not path(m).exists()]
    if verbose:
        print(f"Dukascopy {symbol} hourly: {len(full)} months, {len(todo)} to download")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch, HOUR_URL.format(sym=symbol, y=m.year, m=m.month - 1)): m for m in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                path(futs[f]).write_bytes(f.result())
            except RuntimeError as e:
                print(f"  skipped {futs[f]:%Y-%m} ({e}); re-run to retry")
            if verbose:
                print(f"  {i}/{len(todo)} months", flush=True)

    frames = [_decode_hours(path(m).read_bytes(), m, point) for m in full if path(m).exists()]
    # current, incomplete month: build hours from daily minute files
    if this_month >= pd.Timestamp(start):
        try:
            m1 = download_minutes(str(this_month.date()), None, symbol, cache_dir, workers, verbose=False)
            from .data import resample
            frames.append(resample(m1, "1h"))
        except RuntimeError:
            pass
    df = pd.concat([f for f in frames if not f.empty]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    df.index.name = "time"
    return df


def get_xauusd(timeframe: str = "1h", start: str = "2015-01-01", end: str | None = None,
               data_dir: str | Path = "data") -> pd.DataFrame:
    """Download (or load cached) real XAUUSD and resample to ``timeframe`` (1h, 4h, 1D ...).

    The result is also saved as data/XAUUSD_<timeframe>.csv.
    """
    from .data import _finalise, resample

    h1 = download_hours(start, end, cache_dir=Path(data_dir) / "cache")
    df = _finalise(h1 if timeframe.lower() == "1h" else resample(h1, timeframe))
    df.to_csv(Path(data_dir) / f"XAUUSD_{timeframe}.csv")
    return df
