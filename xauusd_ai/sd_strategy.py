"""Python twin of mt5/SupplyDemandStructure.mq5 for backtesting outside MetaTrader.

Strategy (user's rules)
-----------------------
* Trend on H1 from market structure: last two confirmed swing highs/lows.
  HH + HL = bullish, LH + LL = bearish, otherwise no trades. An H1 close beyond the
  last HL (bull) / LH (bear) cancels the trend until structure re-forms.
* Zones on M5: demand = last down candle before an up-candle impulse (1-3 candles)
  that closes >= ``impulse_atr`` x ATR above that candle's high. Supply mirrors it.
* Entry the moment price first touches the zone, if the H1 trend agrees.
  First touch only: the zone is consumed whether a trade is taken or not.
* SL 5 pips beyond the zone candle's wick, TP at 2R (full close),
  SL to breakeven once price reaches 1R. Risk 1% of balance per trade.

Fill assumptions (conservative, bar data only)
---------------------------------------------
* Prices are BID candles. Longs buy at ask (= bid + spread), shorts are stopped/targeted on ask.
* Touch fill = the zone edge, or the bar's open if it gapped into the zone.
* On the entry bar only the stop is checked (the high/low that would hit TP may have come first).
* If a bar touches both the stop and the target, the stop wins. Breakeven applies from the next bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, compute_metrics
from .data import resample


@dataclass
class SDConfig:
    swing_strength: int = 3
    break_cancels_trend: bool = True
    impulse_max_candles: int = 3
    impulse_atr: float = 1.5
    atr_period: int = 14
    zone_use_body: bool = False
    zone_max_height_atr: float = 3.0
    zone_max_age_bars: int = 288
    pip: float = 0.10
    sl_buffer_pips: float = 5.0
    rr: float = 2.0
    breakeven_at_r: float = 1.0
    risk_pct: float = 1.0
    max_open: int = 0
    session: tuple[int, int] = (0, 24)
    spread: float = 0.30
    commission_per_lot: float = 7.0
    contract_size: float = 100.0
    initial_capital: float = 10_000.0
    min_lots: float = 0.01
    max_lots: float = 50.0


# ---------------------------------------------------------------- H1 trend
def h1_trend(h1: pd.DataFrame, strength: int = 3, break_cancels: bool = True) -> pd.Series:
    """Trend known at the CLOSE of each H1 bar (+1 / -1 / 0)."""
    hi, lo, cl = h1["high"].to_numpy(), h1["low"].to_numpy(), h1["close"].to_numpy()
    n, s = len(h1), strength
    trend = np.zeros(n, dtype=int)
    highs: list[float] = []
    lows: list[float] = []
    for k in range(n):
        j = k - s                       # candidate swing whose right side just closed
        if j - s >= 0:
            # same tie rules as the EA: newer bars strictly lower, older bars not higher
            if all(hi[j - m] <= hi[j] for m in range(1, s + 1)) and all(hi[j + m] < hi[j] for m in range(1, s + 1)):
                highs.append(hi[j])
            if all(lo[j - m] >= lo[j] for m in range(1, s + 1)) and all(lo[j + m] > lo[j] for m in range(1, s + 1)):
                lows.append(lo[j])
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                t = 1
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                t = -1
            else:
                t = 0
            if break_cancels and ((t == 1 and cl[k] < lows[-1]) or (t == -1 and cl[k] > highs[-1])):
                t = 0
            trend[k] = t
    return pd.Series(trend, index=h1.index)


def trend_on_entry_bars(m5: pd.DataFrame, cfg: SDConfig) -> np.ndarray:
    h1 = resample(m5, "1h")
    tr = h1_trend(h1, cfg.swing_strength, cfg.break_cancels_trend)
    # An H1 bar stamped T closes at T+1h, so its trend is usable from T+1h onwards.
    tr.index = tr.index + pd.Timedelta(hours=1)
    return tr.reindex(m5.index, method="ffill").fillna(0).astype(int).to_numpy()


def mt5_atr(df: pd.DataFrame, n: int) -> np.ndarray:
    """MetaTrader's iATR: simple average of true range."""
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


# ---------------------------------------------------------------- simulation
@dataclass
class _Pos:
    dir: int
    entry: float
    sl: float
    tp: float
    lots: float
    risk: float
    time: pd.Timestamp
    bar: int
    be_done: bool = False
    zone_top: float = 0.0
    zone_bottom: float = 0.0


@dataclass
class SDResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    zones_found: int
    zones_touched: int
    cfg: SDConfig = field(repr=False)


def run_sd_backtest(m5: pd.DataFrame, cfg: SDConfig | None = None) -> SDResult:
    cfg = cfg or SDConfig()
    o, h, l, c = (m5[k].to_numpy() for k in ("open", "high", "low", "close"))
    t = m5.index
    n = len(m5)
    atr = mt5_atr(m5, cfg.atr_period)
    trend = trend_on_entry_bars(m5, cfg)
    hours = t.hour.to_numpy()
    sp, buf = cfg.spread, cfg.sl_buffer_pips * cfg.pip
    s0, s1 = cfg.session

    balance = cfg.initial_capital
    zones: list[dict] = []           # active zones
    seen: set[tuple[int, int]] = set()
    open_pos: list[_Pos] = []
    trades: list[dict] = []
    zones_found = zones_touched = 0

    def close(p: _Pos, exit_px: float, i: int, reason: str):
        nonlocal balance
        pnl = (exit_px - p.entry) * p.dir * p.lots * cfg.contract_size - cfg.commission_per_lot * p.lots
        balance += pnl
        trades.append(dict(direction=p.dir, entry_time=p.time, entry_price=round(p.entry, 3),
                           stop=round(p.sl, 3), target=round(p.tp, 3), lots=p.lots,
                           exit_time=t[i], exit_price=round(exit_px, 3), exit_reason=reason,
                           breakeven=p.be_done, pnl=round(pnl, 2),
                           r_multiple=round(pnl / (p.risk * p.lots * cfg.contract_size), 3),
                           bars_held=i - p.bar, balance=round(balance, 2)))

    def in_session(i):
        if s0 <= 0 and s1 >= 24:
            return True
        return s0 <= hours[i] < s1 if s0 <= s1 else (hours[i] >= s0 or hours[i] < s1)

    for i in range(1, n):
        # ---- 1) new bar opened: detect zones from bars closed up to i-1 (EA: DetectZones)
        k = i - 1                                     # last closed bar
        a = atr[k]
        if np.isfinite(a) and a > 0:
            for b in range(2, cfg.impulse_max_candles + 2):
                j = k - (b - 1)                      # zone-forming candle
                if j < 0:
                    break
                imp = range(j + 1, k + 1)
                if c[j] <= o[j] and all(c[m] > o[m] for m in imp) and c[k] - h[j] >= cfg.impulse_atr * a:
                    _add_zone(zones, seen, 1, o[j] if cfg.zone_use_body else h[j], l[j], j, i, a, cfg)
                if c[j] >= o[j] and all(c[m] < o[m] for m in imp) and l[j] - c[k] >= cfg.impulse_atr * a:
                    _add_zone(zones, seen, -1, h[j], o[j] if cfg.zone_use_body else l[j], j, i, a, cfg)
            zones_found = len(seen)
        zones[:] = [z for z in zones if i - z["created"] <= cfg.zone_max_age_bars]

        # ---- 2) manage open positions on this bar (stop first, then target, then breakeven)
        still = []
        for p in open_pos:
            done = False
            if p.dir == 1:
                if l[i] <= p.sl:
                    close(p, min(o[i], p.sl), i, "breakeven" if p.be_done else "stop"); done = True
                elif h[i] >= p.tp:
                    close(p, max(o[i], p.tp), i, "target"); done = True
                elif not p.be_done and cfg.breakeven_at_r > 0 and h[i] >= p.entry + cfg.breakeven_at_r * p.risk:
                    p.sl, p.be_done = p.entry, True
            else:
                if h[i] + sp >= p.sl:
                    close(p, max(o[i] + sp, p.sl), i, "breakeven" if p.be_done else "stop"); done = True
                elif l[i] + sp <= p.tp:
                    close(p, min(o[i] + sp, p.tp), i, "target"); done = True
                elif not p.be_done and cfg.breakeven_at_r > 0 and l[i] + sp <= p.entry - cfg.breakeven_at_r * p.risk:
                    p.sl, p.be_done = p.entry, True
            if not done:
                still.append(p)
        open_pos = still

        # ---- 3) first touches of zones on this bar (EA: CheckZoneTouches)
        remaining = []
        for z in zones:
            d = z["dir"]
            touched = (d == 1 and l[i] <= z["top"]) or (d == -1 and h[i] >= z["bottom"])
            if not touched:
                remaining.append(z)
                continue
            zones_touched += 1
            fill_bid = min(o[i], z["top"]) if d == 1 else max(o[i], z["bottom"])
            gapped = (d == 1 and fill_bid < z["bottom"]) or (d == -1 and fill_bid > z["top"])
            if gapped or trend[i] != d or not in_session(i):
                continue
            if cfg.max_open and len(open_pos) >= cfg.max_open:
                continue
            entry = fill_bid + sp if d == 1 else fill_bid
            sl = z["bottom"] - buf if d == 1 else z["top"] + buf
            risk = abs(entry - sl)
            if risk <= 0:
                continue
            lots = np.floor(balance * cfg.risk_pct / 100 / (risk * cfg.contract_size) * 100) / 100
            if lots < cfg.min_lots:
                continue
            lots = min(lots, cfg.max_lots)
            p = _Pos(d, entry, sl, entry + d * cfg.rr * risk, lots, risk, t[i], i,
                     zone_top=z["top"], zone_bottom=z["bottom"])
            # entry bar: only the adverse side is checked (conservative)
            if (d == 1 and l[i] <= p.sl) or (d == -1 and h[i] + sp >= p.sl):
                close(p, p.sl, i, "stop")
            else:
                open_pos.append(p)
        zones[:] = remaining

    for p in open_pos:
        close(p, c[-1] if p.dir == 1 else c[-1] + sp, n - 1, "end")

    tdf = pd.DataFrame(trades)
    eq = _daily_equity(tdf, m5.index, cfg.initial_capital)
    bt_cfg = BacktestConfig(initial_capital=cfg.initial_capital)
    metrics = compute_metrics(eq, tdf.rename(columns={}) if len(tdf) else tdf, bt_cfg)
    if len(tdf):
        metrics["breakeven_exits"] = int((tdf.exit_reason == "breakeven").sum())
        metrics["target_exits"] = int((tdf.exit_reason == "target").sum())
        metrics["stop_exits"] = int((tdf.exit_reason == "stop").sum())
    metrics["zones_found"] = zones_found
    metrics["zones_touched"] = zones_touched
    return SDResult(tdf, eq, metrics, zones_found, zones_touched, cfg)


def _add_zone(zones, seen, d, top, bottom, j, created, a, cfg):
    key = (d, j)
    if key in seen or top <= bottom:
        return
    if cfg.zone_max_height_atr and top - bottom > cfg.zone_max_height_atr * a:
        return
    seen.add(key)
    zones.append(dict(dir=d, top=top, bottom=bottom, base=j, created=created))


def _daily_equity(trades: pd.DataFrame, index: pd.DatetimeIndex, start: float) -> pd.Series:
    days = pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D")
    if trades.empty:
        return pd.Series(start, index=days, name="equity")
    pnl = trades.groupby(pd.to_datetime(trades.exit_time).dt.normalize())["pnl"].sum()
    return (start + pnl.reindex(days, fill_value=0).cumsum()).rename("equity")


# ---------------------------------------------------------------- command line
def main(argv=None):
    """python -m xauusd_ai.sd_strategy --start 2025-11-01 [--csv m5.csv]"""
    import argparse
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .data import load_csv

    p = argparse.ArgumentParser(prog="xauusd_ai.sd_strategy")
    p.add_argument("--csv", help="M1 or M5 XAUUSD file (default: download from Dukascopy)")
    p.add_argument("--start", default="2025-11-01")
    p.add_argument("--end", default=None)
    p.add_argument("--spread", type=float, default=0.30)
    p.add_argument("--commission", type=float, default=7.0)
    p.add_argument("--out", default="reports/sd_strategy")
    a = p.parse_args(argv)

    if a.csv:
        m1 = load_csv(a.csv)
        desc = a.csv
    else:
        from .dukascopy import download_minutes
        m1 = download_minutes(a.start, a.end)
        desc = "Dukascopy spot XAUUSD (BID) 1-minute candles"
    m5 = resample(m1, "5min")
    m5 = m5[m5.index >= a.start]

    rows = {}
    results = {}
    for label, comm in (("with $%g/lot commission" % a.commission, a.commission), ("spread only", 0.0)):
        r = run_sd_backtest(m5, SDConfig(spread=a.spread, commission_per_lot=comm))
        rows[label] = r.metrics
        results[label] = r

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main_r = next(iter(results.values()))
    main_r.trades.to_csv(out / "trades.csv", index=False)
    table = pd.DataFrame(rows).T
    table.to_csv(out / "metrics.csv")
    (out / "metrics.json").write_text(json.dumps(rows, indent=2, default=str))

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for label, r in results.items():
        ax[0].plot(r.equity.index, r.equity.values, lw=1.4, label=label)
    ax[0].axhline(main_r.cfg.initial_capital, color="grey", lw=0.8, ls="--")
    ax[0].set_title("H1 structure + M5 supply/demand, 1:2 RR, breakeven at 1R, 1% risk")
    ax[0].set_ylabel("Balance ($)")
    ax[0].legend(frameon=False)
    ax[1].plot(m5.index, m5.close, color="goldenrod", lw=0.8)
    ax[1].set_ylabel("XAUUSD")
    for x in ax:
        x.grid(alpha=0.25)
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=120)
    plt.close(fig)

    keys = ["total_return_%", "max_drawdown_%", "sharpe", "trades", "win_rate_%", "profit_factor", "avg_R",
            "target_exits", "breakeven_exits", "stop_exits", "zones_found", "zones_touched"]
    md = [
        "# Supply/demand + market structure strategy: backtest",
        "",
        f"Data: {desc}, M5 bars {m5.index[0]:%Y-%m-%d} to {m5.index[-1]:%Y-%m-%d}. "
        f"Spread ${a.spread}/oz. Start balance ${main_r.cfg.initial_capital:,.0f}, 1% risk per trade.",
        "",
        table[[k for k in keys if k in table.columns]].T.to_markdown(),
        "",
        "![equity](equity.png)",
        "",
        "Every trade is in `trades.csv`. Fill rules are conservative: stop beats target inside one bar, "
        "and take-profit is not checked on the entry bar.",
    ]
    (out / "REPORT.md").write_text("\n".join(md))
    print(table[[k for k in keys if k in table.columns]].T.to_string())
    print(f"\nReport: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
