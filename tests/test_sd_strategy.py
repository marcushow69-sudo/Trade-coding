import numpy as np
import pandas as pd
import pytest

from xauusd_ai.data import synthetic_xauusd
from xauusd_ai.sd_strategy import SDConfig, h1_trend, run_sd_backtest


def test_h1_trend_hh_hl_and_ll_lh():
    # zig-zag up: swing highs 10,12 ; swing lows 5,7  -> bullish
    up = [5, 6, 8, 10, 8, 7, 7.5, 9, 12, 9, 8.5, 9.5, 10, 11, 11.5]
    df = pd.DataFrame({"high": np.array(up) + 0.5, "low": np.array(up) - 0.5, "close": up},
                      index=pd.date_range("2024-01-01", periods=len(up), freq="1h"))
    tr = h1_trend(df, strength=2, break_cancels=False)
    assert tr.iloc[-1] == 1
    down = pd.DataFrame({"high": -df["low"], "low": -df["high"], "close": -df["close"]}, index=df.index)
    assert h1_trend(down, strength=2, break_cancels=False).iloc[-1] == -1


def _flat(n, price, start):
    idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({"open": price, "high": price + 0.2, "low": price - 0.2, "close": price}, index=idx)


def test_demand_zone_trade_hits_target(monkeypatch):
    # force a bullish H1 trend so the test isolates the M5 logic
    import xauusd_ai.sd_strategy as sd
    monkeypatch.setattr(sd, "trend_on_entry_bars", lambda m5, cfg: np.ones(len(m5), dtype=int))
    base = _flat(30, 2000.0, "2024-01-02")
    t = base.index[-1]
    rows = [
        (2000.0, 2000.3, 1998.0, 1998.5),   # zone candle (down), low 1998.0 high 2000.3
        (1998.5, 2004.0, 1998.4, 2003.8),   # impulse up
        (2003.8, 2008.0, 2003.5, 2007.5),   # impulse up (closes far above zone high)
        (2007.5, 2007.6, 2003.0, 2003.5),
        (2003.5, 2003.6, 2000.0, 2000.5),   # touches zone top 2000.3 -> buy
        (2000.5, 2010.0, 2000.4, 2009.0),   # passes 1R -> BE
        (2009.0, 2016.0, 2008.0, 2015.0),   # hits 2R target
    ]
    idx = pd.date_range(t + pd.Timedelta(minutes=5), periods=len(rows), freq="5min")
    df = pd.concat([base, pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)])
    df = pd.concat([df, _flat(5, 2015.0, idx[-1] + pd.Timedelta(minutes=5))]).assign(volume=1.0)
    cfg = SDConfig(spread=0.0, commission_per_lot=0.0, impulse_atr=1.5)
    r = run_sd_backtest(df, cfg)
    assert len(r.trades) == 1
    tr = r.trades.iloc[0]
    assert tr.direction == 1
    assert tr.entry_price == pytest.approx(2000.3)          # zone top
    assert tr.stop == pytest.approx(1998.0 - 0.5)            # wick - 5 pips
    assert tr.target == pytest.approx(2000.3 + 2 * 2.8)
    assert tr.exit_reason == "target"
    assert tr.r_multiple == pytest.approx(2.0, abs=0.05)
    assert tr.pnl == pytest.approx(10_000 * 0.01 * 2, rel=0.05)   # ~2% of balance


def test_runs_on_synthetic_m5():
    m5 = synthetic_xauusd(20000, freq="5min", seed=3)
    r = run_sd_backtest(m5, SDConfig())
    assert r.zones_found > 0
    assert np.isfinite(r.equity).all()
    if len(r.trades):
        assert set(r.trades.exit_reason) <= {"stop", "target", "breakeven", "end"}
        assert r.trades.r_multiple.max() <= 2.05
