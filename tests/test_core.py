import numpy as np
import pandas as pd
import pytest

from xauusd_ai.backtest import BacktestConfig, run_backtest
from xauusd_ai.data import load_csv, synthetic_xauusd
from xauusd_ai.indicators import build_features
from xauusd_ai.ml import MLConfig, triple_barrier_labels, walk_forward
from xauusd_ai.strategies import STRATEGIES


def bars(rows, freq="1h"):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=0.0)


def no_cost(**kw):
    base = dict(spread=0.0, commission_per_lot=0.0, slippage=0.0, sl_atr=None, tp_atr=None,
                fixed_lots=1.0, initial_capital=100_000)
    base.update(kw)
    return BacktestConfig(**base)


def test_signal_executes_next_open_and_pnl_per_lot():
    df = bars([(2000, 2001, 1999, 2000), (2000, 2011, 1999, 2010), (2010, 2021, 2009, 2020), (2020, 2021, 2019, 2020)])
    sig = pd.Series([1, 1, 0, 0], index=df.index)
    r = run_backtest(df, sig, no_cost())
    t = r.trades.iloc[0]
    assert t.entry_time == df.index[1] and t.entry_price == 2000      # next bar's open
    assert t.exit_time == df.index[3] and t.exit_price == 2020         # flat signal -> exit next open
    assert t.pnl == pytest.approx(20 * 100)                            # $20 * 100 oz * 1 lot
    assert r.equity.iloc[-1] == pytest.approx(102_000)


def test_costs_reduce_pnl():
    df = bars([(2000, 2001, 1999, 2000)] * 2 + [(2010, 2011, 2009, 2010)] * 2)
    sig = pd.Series([1, 1, 0, 0], index=df.index)
    r = run_backtest(df, sig, no_cost(spread=0.4, commission_per_lot=7))
    # spread 0.4 * 100 oz = 40, commission 7
    assert r.trades.iloc[0].pnl == pytest.approx(10 * 100 - 40 - 7)


def test_stop_beats_target_when_both_hit_same_bar():
    rows = [(2000, 2001, 1999, 2000)] * 20 + [(2000, 2100, 1900, 2000)] + [(2000, 2001, 1999, 2000)] * 3
    df = bars(rows)
    sig = pd.Series(0, index=df.index)
    sig.iloc[19] = 1
    r = run_backtest(df, sig, no_cost(sl_atr=2, tp_atr=2, exit_on_flat=False))
    assert r.trades.iloc[0].exit_reason == "stop"
    assert r.trades.iloc[0].pnl < 0


def test_features_have_no_lookahead():
    df = synthetic_xauusd(600)
    full = build_features(df)
    cut = build_features(df.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400].dropna(), cut.dropna(), check_freq=False)


def test_triple_barrier_labels():
    rows = [(100, 100.5, 99.5, 100)] * 30 + [(100, 120, 99.9, 119)] + [(119, 119.5, 118.5, 119)] * 5
    df = bars(rows)
    y = triple_barrier_labels(df, horizon=3, tp_atr=2, sl_atr=2)
    assert y["long_win"].iloc[29] == 1 and y["short_win"].iloc[29] == 0
    assert y["long_win"].iloc[-1:].isna().all()                     # future unknown


def test_walk_forward_is_out_of_sample():
    df = synthetic_xauusd(2500)
    cfg = MLConfig(train_bars=1000, test_bars=300, horizon=12, model="logreg")
    wf = walk_forward(df, cfg, verbose=False)
    first = wf.proba.dropna(how="all").index[0]
    assert df.index.get_loc(first) >= cfg.train_bars + cfg.horizon
    assert set(np.unique(wf.signals)) <= {-1, 0, 1}


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_run(name):
    df = synthetic_xauusd(1500)
    r = run_backtest(df, STRATEGIES[name](df), BacktestConfig())
    assert np.isfinite(r.equity).all()


def test_csv_formats(tmp_path):
    mt4 = tmp_path / "mt4.csv"
    mt4.write_text("2024.01.02,01:00,2063.1,2065.0,2061.2,2064.4,1234\n2024.01.02,02:00,2064.4,2066,2063,2065,999\n")
    mt5 = tmp_path / "mt5.csv"
    mt5.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
                   "2024.01.02\t01:00:00\t2063.1\t2065.0\t2061.2\t2064.4\t1234\t0\t20\n")
    gen = tmp_path / "gen.csv"
    gen.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,2063.1,2065.0,2061.2,2064.4,1\n")
    for p in (mt4, mt5, gen):
        df = load_csv(p)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["close"].iloc[0] == pytest.approx(2064.4)
