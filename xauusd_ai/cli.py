"""Command-line interface.

Examples
--------
  # Offline demo on synthetic data (no download needed)
  python -m xauusd_ai --source synthetic

  # Your own MT4/MT5/CSV export, H1 bars
  python -m xauusd_ai --csv data/XAUUSD_H1.csv

  # Daily gold futures from Yahoo, compare AI vs every baseline
  python -m xauusd_ai --source yahoo --interval 1d --start 2010-01-01 --train-bars 750 --test-bars 125 --horizon 10
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from . import data as data_mod
from .backtest import BacktestConfig, run_backtest
from .ml import MLConfig, walk_forward
from .report import save_report
from .strategies import STRATEGIES


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="xauusd_ai", description="Backtest AI and rule-based strategies on XAUUSD.")
    src = p.add_argument_group("data")
    src.add_argument("--source", choices=["csv", "yahoo", "synthetic"], default=None)
    src.add_argument("--csv", help="path to a CSV / MT4 / MT5 export")
    src.add_argument("--start", default="2015-01-01")
    src.add_argument("--end", default=None)
    src.add_argument("--interval", default="1d", help="Yahoo interval (1d, 1h, ...)")
    src.add_argument("--resample", default=None, help="resample loaded data, e.g. 4h or 1D")
    src.add_argument("--bars", type=int, default=8000, help="number of synthetic bars")

    st = p.add_argument_group("strategies")
    st.add_argument("--strategies", default="ai," + ",".join(STRATEGIES),
                    help="comma list: ai," + ",".join(STRATEGIES))

    ml = p.add_argument_group("AI model")
    ml.add_argument("--model", choices=["gbm", "rf", "logreg"], default="gbm")
    ml.add_argument("--horizon", type=int, default=24, help="bars the model looks ahead / max holding time")
    ml.add_argument("--train-bars", type=int, default=3000)
    ml.add_argument("--test-bars", type=int, default=500)
    ml.add_argument("--threshold", type=float, default=0.55, help="min predicted win probability to trade")

    bt = p.add_argument_group("costs & risk")
    bt.add_argument("--capital", type=float, default=10_000)
    bt.add_argument("--risk", type=float, default=0.01, help="fraction of equity risked per trade")
    bt.add_argument("--spread", type=float, default=0.30, help="USD per oz")
    bt.add_argument("--commission", type=float, default=7.0, help="USD per lot round turn")
    bt.add_argument("--sl-atr", type=float, default=2.0)
    bt.add_argument("--tp-atr", type=float, default=3.0)
    p.add_argument("--out", default=None, help="output folder (default results/<timestamp>)")
    return p.parse_args(argv)


def load_data(args) -> pd.DataFrame:
    source = args.source or ("csv" if args.csv else "synthetic")
    if source == "csv":
        if not args.csv:
            raise SystemExit("--csv PATH is required with --source csv")
        df = data_mod.load_csv(args.csv)
    elif source == "yahoo":
        df = data_mod.load_yahoo(args.start, args.end, args.interval)
    else:
        df = data_mod.synthetic_xauusd(args.bars)
        print("NOTE: using synthetic data - results only prove the pipeline works, not that a strategy has an edge.")
    if args.resample:
        df = data_mod.resample(df, args.resample)
    if args.start and source == "csv":
        df = df[df.index >= args.start]
    if args.end and source == "csv":
        df = df[df.index <= args.end]
    return df


def main(argv=None):
    args = parse_args(argv)
    df = load_data(args)
    print(f"Loaded {len(df):,} bars: {df.index[0]} -> {df.index[-1]}")

    base = BacktestConfig(initial_capital=args.capital, risk_per_trade=args.risk, spread=args.spread,
                          commission_per_lot=args.commission, sl_atr=args.sl_atr, tp_atr=args.tp_atr)
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    signals: dict[str, pd.Series] = {}
    eval_start = df.index[0]

    if "ai" in names:
        mlc = MLConfig(model=args.model, horizon=args.horizon, tp_atr=args.tp_atr, sl_atr=args.sl_atr,
                       train_bars=args.train_bars, test_bars=args.test_bars, threshold=args.threshold)
        print(f"\nWalk-forward training ({args.model}) ...")
        wf = walk_forward(df, mlc)
        first_pred = wf.proba.dropna(how="all").index
        if len(first_pred) == 0:
            raise SystemExit("The model produced no out-of-sample predictions - load more data.")
        eval_start = first_pred[0]
        signals["ai"] = wf.signals
        if wf.feature_importance is not None:
            print("\nTop features:\n" + wf.feature_importance.head(8).round(4).to_string())

    for n in names:
        if n == "ai":
            continue
        if n not in STRATEGIES:
            raise SystemExit(f"unknown strategy {n!r}. Choose from: ai, {', '.join(STRATEGIES)}")
        signals[n] = STRATEGIES[n](df)

    # Compare everything over the same out-of-sample window as the AI.
    test_df = df[df.index >= eval_start]
    results = {}
    for n, sig in signals.items():
        cfg = BacktestConfig(**{**base.__dict__})
        if n == "ai":
            cfg.exit_on_flat = False           # let stop / target / time stop manage AI trades
            cfg.max_hold_bars = args.horizon
        if n == "buy_hold":                    # 1x notional, no stops
            cfg.sl_atr = cfg.tp_atr = None
            cfg.fixed_lots = round(cfg.initial_capital / (test_df["open"].iloc[0] * cfg.contract_size), 2)
        results[n] = run_backtest(test_df, sig.loc[test_df.index], cfg)

    table = pd.DataFrame({n: r.metrics for n, r in results.items()}).T
    cols = ["total_return_%", "cagr_%", "sharpe", "max_drawdown_%", "trades", "win_rate_%", "profit_factor", "avg_R"]
    print(f"\nOut-of-sample results {test_df.index[0]} -> {test_df.index[-1]}:\n")
    print(table[[c for c in cols if c in table.columns]].to_string())

    out = args.out or f"results/{datetime.now():%Y%m%d_%H%M%S}"
    save_report(results, out, test_df)
    print(f"\nSaved metrics, trades and equity chart to {out}/")


if __name__ == "__main__":
    main()
