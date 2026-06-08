"""
Improvement D — asset selection / weighting.

The diagnosis showed BTC and SOL barely contribute (BTC +4.3%, SOL +22% at
WR 36%) while AVAX/XLM/DOGE carry the book. Two data-driven (hence overfit-prone)
treatments — ALWAYS compared against the original on OOS, never assumed better:

    D1  drop the two weakest assets (BTC, SOL); trade the other 7
    D2  keep all assets but weight each by historical strength tier

D1 is a universe change (re-backtest on the reduced dict). D2 is an aggregation
change (weighted cross-asset mean of the same trades). Neither touches signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_v3 import MIN_TRADES

WEAK_ASSETS = ("BTC", "SOL")

# tier weights from the diagnosis table (v4 spec §Improvement D)
ASSET_WEIGHTS = {
    "AVAX": 1.5, "XLM": 1.5,            # ★★★
    "DOGE": 1.0, "XRP": 1.0,            # ★★☆
    "ETH": 0.5, "OP": 0.5, "SUI": 0.5,  # ★☆☆
    "SOL": 0.25, "BTC": 0.25,           # ☆☆☆
}


def reduced_universe(data: "dict[str, pd.DataFrame]",
                     drop=WEAK_ASSETS) -> "dict[str, pd.DataFrame]":
    """The universe minus the weakest assets (Improvement D1)."""
    return {k: v for k, v in data.items() if k not in drop}


def weighted_cross_asset_return(per_asset: pd.DataFrame, strategy_id: str,
                                weights=ASSET_WEIGHTS,
                                min_trades: int = MIN_TRADES) -> float:
    """Tier-weighted mean of per-asset total returns for one strategy
    (Improvement D2). Falls back to NaN if no asset clears `min_trades`."""
    g = per_asset[(per_asset["strategy_id"] == strategy_id)
                  & (per_asset["n_trades"] >= min_trades)]
    if g.empty:
        return float("nan")
    w = g["symbol"].map(lambda s: weights.get(s, 1.0)).to_numpy(float)
    r = g["total_return"].to_numpy(float)
    if w.sum() <= 0:
        return float("nan")
    return float(np.average(r, weights=w))
