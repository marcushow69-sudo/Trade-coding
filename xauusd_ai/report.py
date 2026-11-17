"""Save backtest results: metrics JSON, trade list CSV, equity/drawdown chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult  # noqa: E402


def save_report(results: dict[str, BacktestResult], out_dir: str | Path, prices: pd.DataFrame | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame({name: r.metrics for name, r in results.items()}).T
    table.to_csv(out / "metrics.csv")
    (out / "metrics.json").write_text(json.dumps({k: r.metrics for k, r in results.items()}, indent=2, default=str))
    for name, r in results.items():
        if not r.trades.empty:
            r.trades.to_csv(out / f"trades_{name}.csv", index=False)

    rows = 3 if prices is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3.2 * rows), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1] + ([1.4] if prices is not None else [])})
    ax_eq, ax_dd = axes[0], axes[1]
    for name, r in results.items():
        ax_eq.plot(r.equity.index, r.equity.values, label=name, linewidth=1.3)
        dd = (r.equity / r.equity.cummax() - 1) * 100
        ax_dd.plot(dd.index, dd.values, linewidth=1)
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.legend(loc="upper left", frameon=False)
    ax_eq.set_title("XAUUSD backtest")
    ax_dd.set_ylabel("Drawdown (%)")
    if prices is not None:
        axes[2].plot(prices.index, prices["close"], color="goldenrod", linewidth=1)
        axes[2].set_ylabel("XAUUSD")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=120)
    plt.close(fig)
    return out
