"""Fully automatic research run: download data, search strategies, validate on unseen data, write a report.

    python -m xauusd_ai.auto                 # real XAUUSD from Dukascopy (default)
    python -m xauusd_ai.auto --csv my.csv    # your own hourly data
    python -m xauusd_ai.auto --quick         # smaller grid, for a fast check

How it avoids fooling itself
----------------------------
* The AI is trained walk-forward, so every prediction it trades on is out-of-sample.
* All settings (timeframe, model, horizon, stop/target, confidence threshold, baseline
  parameters) are chosen ONLY on the development period (default 2017-2023).
* The winners are then scored on the holdout period (default 2024 -> today), which played
  no part in any choice. The holdout numbers are the ones to believe.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestConfig, run_backtest  # noqa: E402
from .data import load_csv, resample, synthetic_xauusd  # noqa: E402
from .indicators import build_features  # noqa: E402
from .ml import MLConfig, signals_from_proba, triple_barrier_labels, walk_forward  # noqa: E402
from .strategies import STRATEGIES  # noqa: E402

TIMEFRAMES = {
    # walk-forward windows: ~1 year of training, retrain roughly monthly
    "1h": dict(train=6000, test=500, horizons=[12, 24, 48]),
    "4h": dict(train=1500, test=125, horizons=[6, 12, 30]),
    "1D": dict(train=500, test=21, horizons=[5, 10, 20]),
}
EXITS = [(2.0, 3.0), (1.5, 1.5), (1.0, 2.0)]            # (stop ATR, target ATR)
MODELS = ["gbm", "logreg"]
EDGE_MARGINS = [0.0, 0.05, 0.10, 0.15]                    # required win prob above break-even

BASELINE_GRID = {
    "buy_hold": [{}],
    "ema_cross": [dict(fast=10, slow=30), dict(fast=20, slow=50), dict(fast=50, slow=200)],
    "rsi_reversion": [dict(lower=30, upper=70), dict(lower=25, upper=75)],
    "donchian": [dict(n=20), dict(n=55)],
    "bollinger": [dict(k=2.0), dict(k=2.5)],
}
BASELINE_EXITS = [(2.0, 3.0), (None, None)]
MIN_DEV_TRADES = 30


def _bt(df, sig, cfg, start, end):
    m = (df.index >= start) & (df.index < end)
    if m.sum() < 50:
        return None
    return run_backtest(df[m], sig[m], cfg)


def research(data: dict[str, pd.DataFrame], dev: tuple[str, str], hold: tuple[str, str],
             base: BacktestConfig, quick: bool = False, log=print):
    rows, keep = [], {}
    models = MODELS[:1] if quick else MODELS
    exits = EXITS[:1] if quick else EXITS

    for tf, df in data.items():
        spec = TIMEFRAMES[tf]
        horizons = spec["horizons"][1:2] if quick else spec["horizons"]
        X = build_features(df)
        log(f"\n[{tf}] {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")

        # ---------- AI ----------
        for horizon, (sl, tp) in itertools.product(horizons, exits):
            y = triple_barrier_labels(df, horizon, tp, sl)
            for model in models:
                t0 = time.time()
                mlc = MLConfig(model=model, horizon=horizon, tp_atr=tp, sl_atr=sl,
                               train_bars=spec["train"], test_bars=spec["test"])
                wf = walk_forward(df, mlc, verbose=False, X=X, y=y)
                breakeven = sl / (sl + tp)
                for margin in EDGE_MARGINS:
                    thr = round(breakeven + margin, 3)
                    sig = signals_from_proba(wf.proba, thr)
                    cfg = replace(base, sl_atr=sl, tp_atr=tp, exit_on_flat=False, max_hold_bars=horizon)
                    name = f"ai|{tf}|{model}|h{horizon}|sl{sl}/tp{tp}|p>{thr}"
                    _record(rows, keep, name, "ai", tf, df, sig, cfg, dev, hold)
                log(f"  AI {model:6} h={horizon:<3} sl/tp={sl}/{tp}  ({time.time() - t0:.0f}s)")

        # ---------- rule-based baselines ----------
        for strat, grid in BASELINE_GRID.items():
            for params in grid:
                sig = STRATEGIES[strat](df, **params)
                for sl, tp in ([(None, None)] if strat == "buy_hold" else BASELINE_EXITS):
                    cfg = replace(base, sl_atr=sl, tp_atr=tp)
                    if strat == "buy_hold":
                        cfg = replace(cfg, fixed_lots=None, risk_per_trade=0.0)
                    pstr = ",".join(f"{k}={v}" for k, v in params.items())
                    name = f"{strat}|{tf}|{pstr}|sl{sl}/tp{tp}"
                    _record(rows, keep, name, strat, tf, df, sig, cfg, dev, hold)
        log(f"  baselines done")

    board = pd.DataFrame(rows).set_index("name")
    return board, keep


def _record(rows, keep, name, family, tf, df, sig, cfg, dev, hold):
    if family == "buy_hold":  # 1x notional, sized at the start of each period
        res = {}
        for key, (s, e) in (("dev", dev), ("hold", hold)):
            m = (df.index >= s) & (df.index < e)
            if m.sum() < 50:
                return
            lots = round(cfg.initial_capital / (df.loc[m, "open"].iloc[0] * cfg.contract_size), 2)
            res[key] = run_backtest(df[m], sig[m], replace(cfg, fixed_lots=lots))
    else:
        res = {"dev": _bt(df, sig, cfg, *dev), "hold": _bt(df, sig, cfg, *hold)}
    if res["dev"] is None or res["hold"] is None:
        return
    d, h = res["dev"].metrics, res["hold"].metrics
    rows.append({
        "name": name, "family": family, "timeframe": tf,
        "dev_sharpe": d["sharpe"], "dev_return_%": d["total_return_%"], "dev_maxdd_%": d["max_drawdown_%"],
        "dev_trades": d["trades"], "dev_pf": d.get("profit_factor"),
        "hold_sharpe": h["sharpe"], "hold_return_%": h["total_return_%"], "hold_cagr_%": h["cagr_%"],
        "hold_maxdd_%": h["max_drawdown_%"], "hold_trades": h["trades"], "hold_win_%": h.get("win_rate_%"),
        "hold_pf": h.get("profit_factor"),
    })
    keep[name] = res


def pick_winners(board: pd.DataFrame) -> pd.DataFrame:
    """Best configuration per strategy family, chosen by DEVELOPMENT Sharpe only."""
    ok = board[(board.dev_trades >= MIN_DEV_TRADES) | (board.family == "buy_hold")]
    winners = ok.sort_values("dev_sharpe", ascending=False).groupby("family", sort=False).head(1)
    return winners.sort_values("hold_sharpe", ascending=False)


def write_report(board, winners, keep, out: Path, dev, hold, data_desc, base: BacktestConfig, elapsed):
    out.mkdir(parents=True, exist_ok=True)
    board.sort_values("dev_sharpe", ascending=False).to_csv(out / "leaderboard.csv")
    winners.to_csv(out / "winners.csv")

    # holdout equity chart of every family winner
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for name in winners.index:
        eq = keep[name]["hold"].equity
        ax[0].plot(eq.index, eq.values, lw=1.4, label=name.split("|")[0] + f" ({name.split('|')[1]})")
        ax[1].plot(eq.index, (eq / eq.cummax() - 1) * 100, lw=1)
    ax[0].axhline(base.initial_capital, color="grey", lw=0.8, ls="--")
    ax[0].set_title(f"Holdout {hold[0]} -> {hold[1][:10]}: settings were chosen on {dev[0]} -> {dev[1]} only")
    ax[0].set_ylabel("Equity ($)")
    ax[1].set_ylabel("Drawdown (%)")
    ax[0].legend(frameon=False, loc="upper left")
    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "holdout_equity.png", dpi=120)
    plt.close(fig)

    ai_best = winners[winners.family == "ai"]
    ai_top = board[(board.family == "ai") & (board.dev_trades >= MIN_DEV_TRADES)].sort_values(
        "dev_sharpe", ascending=False).head(10)
    for name in winners.index:
        tr = keep[name]["hold"].trades
        if not tr.empty:
            tr.to_csv(out / f"holdout_trades_{name.split('|')[0]}.csv", index=False)

    cols = ["timeframe", "dev_sharpe", "dev_return_%", "hold_sharpe", "hold_return_%", "hold_cagr_%",
            "hold_maxdd_%", "hold_trades", "hold_win_%", "hold_pf"]
    verdict = _verdict(winners)
    md = [
        "# XAUUSD automatic backtest report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} UTC in {elapsed / 60:.1f} min. Data: {data_desc}.",
        "",
        f"* **Development period** (all settings chosen here): {dev[0]} to {dev[1]}",
        f"* **Holdout period** (never used for any choice): {hold[0]} to {hold[1][:10]}",
        f"* Costs: spread ${base.spread}/oz, commission ${base.commission_per_lot}/lot, "
        f"slippage ${base.slippage}/oz; risk {base.risk_per_trade:.0%} of equity per trade; "
        f"starting capital ${base.initial_capital:,.0f}",
        f"* Configurations tested: {len(board)} ({(board.family == 'ai').sum()} AI, "
        f"{(board.family != 'ai').sum()} rule-based)",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Winner of each strategy family on the holdout",
        "",
        "Each row is the configuration with the best *development* Sharpe in its family.",
        "",
        winners[cols].to_markdown(),
        "",
        "![holdout equity](holdout_equity.png)",
        "",
        "## Top 10 AI configurations (ranked on development) and how they held up",
        "",
        "If the holdout numbers collapse compared with development, the edge was probably curve-fitted.",
        "",
        ai_top[cols].to_markdown(),
        "",
        "## Files",
        "",
        "* `leaderboard.csv`: every configuration tested, development and holdout metrics",
        "* `winners.csv`: the table above",
        "* `holdout_trades_<strategy>.csv`: every holdout trade of each winner",
        "",
        "_Backtests are not a guarantee of future results. This is research, not financial advice._",
    ]
    (out / "REPORT.md").write_text("\n".join(md))
    (out / "summary.json").write_text(json.dumps({
        "dev": dev, "hold": hold, "verdict": verdict,
        "ai_winner": ai_best.reset_index().to_dict("records"),
        "winners": winners.reset_index().to_dict("records"),
    }, indent=2, default=str))
    return out


def _verdict(w: pd.DataFrame) -> str:
    if "ai" not in set(w.family):
        return "No AI configuration made enough trades in development to be evaluated."
    ai = w[w.family == "ai"].iloc[0]
    bh = w[w.family == "buy_hold"].iloc[0] if "buy_hold" in set(w.family) else None
    rules = w[~w.family.isin(["ai", "buy_hold"])]
    best_rule = rules.iloc[0] if len(rules) else None
    s = [f"The AI's chosen setup (`{ai.name}`) returned **{ai['hold_return_%']:+.1f}%** on the holdout "
         f"(Sharpe {ai.hold_sharpe:.2f}, max drawdown {ai['hold_maxdd_%']:.1f}%, {int(ai.hold_trades)} trades), "
         f"compared with Sharpe {ai.dev_sharpe:.2f} in development."]
    if best_rule is not None:
        s.append(f"The best rule-based strategy, `{best_rule.name}`, returned {best_rule['hold_return_%']:+.1f}% "
                 f"(Sharpe {best_rule.hold_sharpe:.2f}).")
    if bh is not None:
        s.append(f"Simply buying and holding gold returned {bh['hold_return_%']:+.1f}% "
                 f"(Sharpe {bh.hold_sharpe:.2f}, max drawdown {bh['hold_maxdd_%']:.1f}%).")
    if ai.hold_sharpe <= 0:
        s.append("**The AI did not find an edge that survived on unseen data.**")
    elif bh is not None and ai.hold_sharpe < bh.hold_sharpe:
        s.append("The AI was profitable on unseen data but less efficient than holding gold.")
    else:
        s.append("The AI kept a positive risk-adjusted edge on unseen data. Paper-trade it before risking money.")
    return " ".join(s)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xauusd_ai.auto", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="hourly XAUUSD CSV/MT4/MT5 file (default: download from Dukascopy)")
    p.add_argument("--synthetic", action="store_true", help="use synthetic data (pipeline test)")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--dev", default="2017-01-01:2024-01-01", help="development period start:end")
    p.add_argument("--holdout-start", default="2024-01-01")
    p.add_argument("--timeframes", default="1h,4h,1D")
    p.add_argument("--quick", action="store_true", help="small grid for a fast run")
    p.add_argument("--spread", type=float, default=0.30)
    p.add_argument("--commission", type=float, default=7.0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--capital", type=float, default=10_000)
    p.add_argument("--out", default="reports/latest")
    a = p.parse_args(argv)

    t0 = time.time()
    if a.synthetic:
        h1 = synthetic_xauusd(100000, seed=7)
        h1.index = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=len(h1), freq="1h")
        desc = "SYNTHETIC (pipeline test only)"
    elif a.csv:
        h1 = load_csv(a.csv)
        desc = f"{a.csv}"
    else:
        from .dukascopy import get_xauusd
        h1 = get_xauusd("1h", a.start)
        desc = "Dukascopy spot XAUUSD (BID), hourly candles resampled as needed"
    h1 = h1[h1.index >= a.start]
    data = {tf: (h1 if tf == "1h" else resample(h1, tf)) for tf in a.timeframes.split(",")}

    dev = tuple(a.dev.split(":"))
    hold = (a.holdout_start, str(h1.index[-1] + pd.Timedelta(days=1)))
    base = BacktestConfig(initial_capital=a.capital, spread=a.spread, commission_per_lot=a.commission,
                          risk_per_trade=a.risk)
    board, keep = research(data, dev, hold, base, quick=a.quick)
    winners = pick_winners(board)
    out = write_report(board, winners, keep, Path(a.out), dev, hold, desc, base, time.time() - t0)

    print("\n" + "=" * 80)
    print(_verdict(winners))
    print("\n" + winners[["timeframe", "dev_sharpe", "hold_sharpe", "hold_return_%", "hold_maxdd_%",
                          "hold_trades"]].to_string())
    print(f"\nFull report: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
