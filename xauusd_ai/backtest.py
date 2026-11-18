"""Bar-by-bar backtester tuned for XAUUSD CFDs.

Conventions
-----------
* ``signals`` is a Series aligned with the price data: +1 long, -1 short, 0 flat.
  A signal on bar t is decided at the close of t and executed at the OPEN of t+1.
* 1 standard lot = 100 troy ounces, so a $1 move = $100 per lot.
* Spread is in price units (e.g. 0.30 = 30 cents). Longs buy at ask, shorts sell at bid.
* Stop-loss / take-profit are ATR multiples set at entry. If both are touched
  inside the same bar the stop is assumed to fill first (conservative).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .indicators import atr as atr_indicator


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.01        # fraction of equity risked per trade (needs a stop)
    fixed_lots: float | None = None     # if set, overrides risk-based sizing
    contract_size: float = 100.0        # oz per lot
    spread: float = 0.30                # in USD per oz
    commission_per_lot: float = 7.0     # round-turn, USD
    slippage: float = 0.05              # USD per oz, applied on entry and stop exits
    sl_atr: float | None = 2.0          # stop distance in ATR multiples (None = no stop)
    tp_atr: float | None = 3.0          # take-profit in ATR multiples (None = no target)
    atr_period: int = 14
    max_lots: float = 50.0
    max_leverage: float = 20.0          # cap position notional at N x equity
    min_lots: float = 0.01
    exit_on_flat: bool = True           # close when the signal goes to 0
    max_hold_bars: int | None = None    # time stop: close after N bars


@dataclass
class Trade:
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    lots: float
    stop: float | None
    target: float | None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    r_multiple: float = float("nan")


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    config: BacktestConfig = field(repr=False)

    def summary(self) -> str:
        lines = [f"{k:>22}: {v}" for k, v in self.metrics.items()]
        return "\n".join(lines)


def run_backtest(df: pd.DataFrame, signals: pd.Series, config: BacktestConfig | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    signals = signals.reindex(df.index).fillna(0).astype(int).clip(-1, 1)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr_indicator(df, cfg.atr_period).to_numpy()
    sig = signals.to_numpy()
    times = df.index
    half_spread = cfg.spread / 2

    cash = cfg.initial_capital
    equity = np.empty(len(df))
    trades: list[Trade] = []
    pos: Trade | None = None
    entry_bar = 0

    def close(trade: Trade, price: float, t, reason: str):
        nonlocal cash
        # exit at bid for longs, ask for shorts
        fill = price - half_spread * trade.direction
        gross = (fill - trade.entry_price) * trade.direction * trade.lots * cfg.contract_size
        trade.pnl = gross - cfg.commission_per_lot * trade.lots
        trade.exit_time, trade.exit_price, trade.exit_reason = t, fill, reason
        if trade.stop is not None:
            risk = abs(trade.entry_price - trade.stop) * trade.lots * cfg.contract_size
            trade.r_multiple = trade.pnl / risk if risk > 0 else float("nan")
        cash += trade.pnl
        trades.append(trade)

    for i in range(len(df)):
        # 1) act on yesterday's signal at today's open
        if i > 0:
            want = sig[i - 1]
            if pos is not None and (want == -pos.direction or (want == 0 and cfg.exit_on_flat)):
                close(pos, o[i], times[i], "signal")
                pos = None
            if pos is None and want != 0 and np.isfinite(a[i - 1]) and cash > 0:
                d = int(want)
                entry = o[i] + (half_spread + cfg.slippage) * d
                stop = entry - d * cfg.sl_atr * a[i - 1] if cfg.sl_atr else None
                target = entry + d * cfg.tp_atr * a[i - 1] if cfg.tp_atr else None
                if cfg.fixed_lots is not None:
                    lots = cfg.fixed_lots
                else:
                    # risk-based sizing; without a stop, assume a 2-ATR adverse move as the risk unit.
                    # Round-trip costs count as part of the risk so tiny stops can't create huge positions.
                    risk_dist = abs(entry - stop) if stop is not None else 2 * a[i - 1]
                    risk_dist += cfg.spread + 2 * cfg.slippage + cfg.commission_per_lot / cfg.contract_size
                    lots = cash * cfg.risk_per_trade / (risk_dist * cfg.contract_size)
                    lots = min(lots, cash * cfg.max_leverage / (entry * cfg.contract_size))
                lots = float(min(np.floor(lots * 100) / 100, cfg.max_lots))
                if lots >= cfg.min_lots:
                    pos = Trade(d, times[i], entry, lots, stop, target)
                    entry_bar = i

        # 2) intrabar stop / target checks (bid for longs, ask for shorts)
        if pos is not None:
            d = pos.direction
            bar_lo = l[i] - half_spread if d == 1 else l[i] + half_spread
            bar_hi = h[i] - half_spread if d == 1 else h[i] + half_spread
            hit_stop = pos.stop is not None and ((d == 1 and bar_lo <= pos.stop) or (d == -1 and bar_hi >= pos.stop))
            hit_tp = pos.target is not None and ((d == 1 and bar_hi >= pos.target) or (d == -1 and bar_lo <= pos.target))
            if hit_stop:
                # gap through the stop -> fill at the open
                level = pos.stop
                if (d == 1 and o[i] - half_spread < level) or (d == -1 and o[i] + half_spread > level):
                    level = o[i] - half_spread * d
                close(pos, level + half_spread * d - cfg.slippage * d, times[i], "stop")
                pos = None
            elif hit_tp:
                close(pos, pos.target + half_spread * d, times[i], "target")
                pos = None
            elif cfg.max_hold_bars and i - entry_bar + 1 >= cfg.max_hold_bars:
                close(pos, c[i], times[i], "time")
                pos = None

        # 3) mark to market at the close
        open_pnl = 0.0
        if pos is not None:
            mark = c[i] - half_spread * pos.direction
            open_pnl = (mark - pos.entry_price) * pos.direction * pos.lots * cfg.contract_size
        equity[i] = cash + open_pnl

    if pos is not None:
        close(pos, c[-1], times[-1], "end")
        equity[-1] = cash

    eq = pd.Series(equity, index=df.index, name="equity")
    tdf = pd.DataFrame([asdict(t) for t in trades])
    return BacktestResult(eq, tdf, compute_metrics(eq, tdf, cfg), cfg)


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    years = (index[-1] - index[0]).days / 365.25
    return len(index) / years if years > 0 else 252.0


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, cfg: BacktestConfig) -> dict:
    rets = equity.pct_change().dropna()
    ppy = _periods_per_year(pd.DatetimeIndex(equity.index))
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total_ret = equity.iloc[-1] / cfg.initial_capital - 1
    if equity.iloc[-1] <= 0:
        cagr = -1.0
    elif years < 0.1:  # annualising a few days is meaningless
        cagr = total_ret
    else:
        cagr = (equity.iloc[-1] / cfg.initial_capital) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(ppy) if rets.std() > 0 else 0.0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(ppy) if downside and downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    m = {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_%": round(total_ret * 100, 2),
        "cagr_%": round(cagr * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "max_drawdown_%": round(float(max_dd) * 100, 2),
        "calmar": round(float(cagr / abs(max_dd)), 2) if max_dd < 0 else 0.0,
        "trades": int(len(trades)),
    }
    if len(trades):
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        gp, gl = wins.pnl.sum(), -losses.pnl.sum()
        m.update({
            "win_rate_%": round(len(wins) / len(trades) * 100, 2),
            "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            "avg_win": round(float(wins.pnl.mean()), 2) if len(wins) else 0.0,
            "avg_loss": round(float(losses.pnl.mean()), 2) if len(losses) else 0.0,
            "expectancy": round(float(trades.pnl.mean()), 2),
            "avg_R": round(float(trades.r_multiple.mean()), 2) if trades.r_multiple.notna().any() else None,
            "long_trades": int((trades.direction == 1).sum()),
            "short_trades": int((trades.direction == -1).sum()),
        })
    return m
