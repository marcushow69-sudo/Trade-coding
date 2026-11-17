# XAUUSD AI Backtester

A Python toolkit for testing whether a machine-learning model can trade gold (XAUUSD)
profitably **after realistic costs**. It runs side by side with simple rule-based strategies
so you can see whether the AI actually adds anything.

## Quick start

```bash
pip install -r requirements.txt

# 1. Offline demo on synthetic data (checks everything works)
python -m xauusd_ai --source synthetic

# 2. Your own data: export XAUUSD H1 from MetaTrader (or any CSV) into data/
python -m xauusd_ai --csv data/XAUUSD_H1.csv

# 3. Daily gold futures from Yahoo Finance (GC=F as a proxy for spot)
python -m xauusd_ai --source yahoo --interval 1d --start 2008-01-01 \
    --train-bars 1000 --test-bars 125 --horizon 10
```

Each run prints a comparison table and saves to `results/<timestamp>/`:
`metrics.csv`, `metrics.json`, `trades_<strategy>.csv` and `equity.png` (equity, drawdown, price).

## How the AI works (`xauusd_ai/ml.py`)

1. **Features**: about 30 scale-free indicators: multi-period returns, volatility, ATR %, RSI, MACD,
   distance from EMAs 20/50/200, Bollinger position, ADX, candle shape, and session hour / weekday.
2. **Labels (triple barrier)**: for every bar, would a long (or short) entered at the next open
   hit **take-profit (3×ATR) before stop-loss (2×ATR)** within `horizon` bars? The labels match the
   trades the backtester actually takes.
3. **Walk-forward training**: train on the last `train_bars`, predict the next `test_bars`,
   slide forward and retrain. A purge gap of `horizon` bars stops future information from leaking
   into training. **Every AI trade in the report is out-of-sample.**
4. **Trading**: go long when P(long wins) > `--threshold` (default 0.55) and beats P(short wins),
   and the reverse for shorts. Exits come from the stop, the target or a time stop after `horizon` bars.

Models: `--model gbm` (gradient boosting, default), `rf` (random forest), `logreg` (logistic regression).

## Backtester (`xauusd_ai/backtest.py`)

* Signals are executed at the **next bar's open** (no look-ahead).
* 1 lot = 100 oz. Spread (default $0.30), commission ($7/lot) and slippage are all modelled.
* Risk-based position sizing (default 1% of equity per trade, using the stop distance).
* If a bar touches both the stop and the target, the **stop is assumed to fill first**.
  Gaps through the stop fill at the open.
* Metrics: total return, CAGR, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor,
  expectancy, average R-multiple.

## Baseline strategies (`xauusd_ai/strategies.py`)

`buy_hold`, `ema_cross`, `rsi_reversion`, `donchian` (breakout + ADX filter), `bollinger`.
Pick a subset with `--strategies ai,ema_cross,donchian`. To add your own, write a function
`df -> Series of {-1,0,1}` and register it in `STRATEGIES`.

## Useful options

| Option | Default | Meaning |
|---|---|---|
| `--resample 4h` | – | convert data to another timeframe |
| `--risk 0.01` | 1% | equity risked per trade |
| `--spread 0.30` | $0.30 | your broker's typical XAUUSD spread |
| `--sl-atr / --tp-atr` | 2 / 3 | stop and target in ATR multiples (used by the AI labels too) |
| `--threshold` | 0.55 | higher means fewer, more confident AI trades |
| `--horizon` | 24 | bars the AI looks ahead, and its maximum holding time |

## Tests

```bash
python -m pytest -q
```

## Honest caveats

* Synthetic data has no real edge, so a losing AI on it is expected. It only proves the plumbing works.
* Use at least several years of H1 data. Tune parameters on one period and confirm on a later one you
  never touched; otherwise you are just curve-fitting.
* Yahoo's GC=F is futures, not spot: there are roll gaps and different trading hours. Broker data is better.
* Past performance does not guarantee future results. This is a research tool, not financial advice.
