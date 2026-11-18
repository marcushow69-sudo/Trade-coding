"""The "AI" part: a machine-learning model that predicts the next move and trades on it.

Approach
--------
1. Features: scale-free technical features (``indicators.build_features``).
2. Labels ("triple barrier"): for every bar, look forward up to ``horizon`` bars and
   ask which happens first — price rises ``tp_atr`` x ATR (label 1), falls
   ``sl_atr`` x ATR (label -1 for the short side), or neither (label 0).
   We train two classifiers: P(long trade wins) and P(short trade wins).
3. Walk-forward: train on a rolling window of past data, predict the next block,
   slide forward, retrain. A ``gap`` of ``horizon`` bars between train and test
   prevents label leakage. Every prediction is genuinely out-of-sample.
4. Signals: go long if P(long wins) > threshold and > P(short wins), and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import atr, build_features


@dataclass
class MLConfig:
    model: str = "gbm"            # gbm | rf | logreg
    horizon: int = 24             # bars to look ahead for the label
    tp_atr: float = 3.0           # should match BacktestConfig.tp_atr
    sl_atr: float = 2.0           # should match BacktestConfig.sl_atr
    train_bars: int = 3000        # rolling training window
    test_bars: int = 500          # retrain every N bars
    threshold: float = 0.55       # min win probability to take a trade
    min_train_samples: int = 500
    seed: int = 0


def make_model(name: str, seed: int = 0):
    if name == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                              l2_regularization=1.0, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20,
                                      n_jobs=-1, random_state=seed)
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    raise ValueError(f"unknown model {name!r}")


def triple_barrier_labels(df: pd.DataFrame, horizon: int, tp_atr: float, sl_atr: float,
                          atr_period: int = 14) -> pd.DataFrame:
    """For each bar t, would a long / short entered at the next open hit its target before its stop?

    Returns columns ``long_win`` and ``short_win`` (1/0, NaN where the future is unknown).
    """
    o, h, l = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    a = atr(df, atr_period).to_numpy()
    n = len(df)
    long_win = np.full(n, np.nan)
    short_win = np.full(n, np.nan)
    for t in range(n - 1):
        end = t + 1 + horizon
        if end > n:
            break
        entry = o[t + 1]
        hh, ll = h[t + 1:end], l[t + 1:end]
        lt, ls = entry + tp_atr * a[t], entry - sl_atr * a[t]
        st, ss = entry - tp_atr * a[t], entry + sl_atr * a[t]
        long_win[t] = _first_hit(hh >= lt, ll <= ls)
        short_win[t] = _first_hit(ll <= st, hh >= ss)
    return pd.DataFrame({"long_win": long_win, "short_win": short_win}, index=df.index)


def _first_hit(target_hits: np.ndarray, stop_hits: np.ndarray) -> float:
    ti = np.argmax(target_hits) if target_hits.any() else np.inf
    si = np.argmax(stop_hits) if stop_hits.any() else np.inf
    # same-bar ties count as a loss (conservative, matches the backtester)
    return 1.0 if ti < si else 0.0


@dataclass
class WalkForwardOutput:
    signals: pd.Series
    proba: pd.DataFrame           # out-of-sample P(long win), P(short win)
    feature_importance: pd.Series | None
    folds: int


def walk_forward(df: pd.DataFrame, cfg: MLConfig | None = None, verbose: bool = True,
                 X: pd.DataFrame | None = None, y: pd.DataFrame | None = None) -> WalkForwardOutput:
    """Rolling train/predict. Pass precomputed features ``X`` / labels ``y`` to save time in grid searches."""
    cfg = cfg or MLConfig()
    X = build_features(df) if X is None else X
    y = triple_barrier_labels(df, cfg.horizon, cfg.tp_atr, cfg.sl_atr) if y is None else y
    valid = X.notna().all(axis=1)

    proba = pd.DataFrame(np.nan, index=df.index, columns=["p_long", "p_short"])
    importances = []
    n = len(df)
    start = cfg.train_bars + cfg.horizon
    if start >= n:
        raise ValueError(f"Need more than {start} bars for walk-forward (have {n}). "
                         "Lower --train-bars or load more data.")
    folds = 0
    for test_start in range(start, n, cfg.test_bars):
        test_end = min(test_start + cfg.test_bars, n)
        train_end = test_start - cfg.horizon          # purge: labels must be fully known
        train_start = max(0, train_end - cfg.train_bars)
        tr = slice(train_start, train_end)
        te = slice(test_start, test_end)

        Xtr = X.iloc[tr][valid.iloc[tr]]
        Xte = X.iloc[te][valid.iloc[te]]
        if len(Xtr) < cfg.min_train_samples or Xte.empty:
            continue
        for side, col in (("long_win", "p_long"), ("short_win", "p_short")):
            ytr = y[side].reindex(Xtr.index)
            mask = ytr.notna()
            if ytr[mask].nunique() < 2:
                continue
            model = make_model(cfg.model, cfg.seed)
            model.fit(Xtr[mask], ytr[mask].astype(int))
            proba.loc[Xte.index, col] = model.predict_proba(Xte)[:, 1]
            if side == "long_win" and hasattr(model, "feature_importances_"):
                importances.append(pd.Series(model.feature_importances_, index=X.columns))
        folds += 1
        if verbose:
            print(f"  fold {folds:>3}: train {X.index[train_start]} -> {X.index[train_end - 1]} | "
                  f"test {X.index[test_start]} -> {X.index[test_end - 1]}")

    signals = signals_from_proba(proba, cfg.threshold)
    fi = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) if importances else None
    return WalkForwardOutput(signals, proba, fi, folds)


def signals_from_proba(proba: pd.DataFrame, threshold: float) -> pd.Series:
    pl, ps = proba["p_long"].fillna(0), proba["p_short"].fillna(0)
    sig = pd.Series(0, index=proba.index)
    sig[(pl > threshold) & (pl > ps)] = 1
    sig[(ps > threshold) & (ps > pl)] = -1
    return sig

