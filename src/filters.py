"""
filters.py — Filter Library (F1–F10) for the v3 strategy engine.

A filter is an extra boolean condition AND-ed onto an entry signal: the signal
only fires on days where the filter is also True. Like signals, filters come in
per-asset and cross-asset (market-level) flavours. All are look-ahead safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from indicators import _ta
from indicators import _ta_extended as tae


@dataclass(frozen=True)
class FilterSpec:
    id: str
    name: str
    fn: Callable
    cross_asset: bool = False
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# per-asset filters
# ---------------------------------------------------------------------------
def f1_uptrend(df, ema_period=50):
    return df["close"] > _ta.ema(df["close"], ema_period)


def f2_downtrend(df, ema_period=50):
    return df["close"] < _ta.ema(df["close"], ema_period)


def f3_ranging(df, period=14, thresh=20):
    _, _, adx_val = _ta.adx(df["high"], df["low"], df["close"], period)
    return adx_val < thresh


def f4_trending(df, period=14, thresh=25):
    _, _, adx_val = _ta.adx(df["high"], df["low"], df["close"], period)
    return adx_val > thresh


def f5_high_vol(df, period=14, ma=60, mult=1.5):
    a = _ta.atr(df["high"], df["low"], df["close"], period)
    return a > _ta.sma(a, ma) * mult


def f6_low_vol(df, period=14, ma=60, mult=0.7):
    a = _ta.atr(df["high"], df["low"], df["close"], period)
    return a < _ta.sma(a, ma) * mult


def f7_volume_confirm(df, ma=20, mult=1.5):
    return df["volume"] > _ta.sma(df["volume"], ma) * mult


def f10_high_regime(df, period=60, pctile=0.70):
    hv = tae.hist_vol(df["close"], period)
    rank = tae.rolling_percentile(hv, 252)
    return rank > pctile


# ---------------------------------------------------------------------------
# cross-asset (market-level) filters
# ---------------------------------------------------------------------------
def _breadth_above_ema(data, ema_period=20):
    """Fraction of assets whose close is above their EMA(ema_period), per date."""
    cols = {}
    for sym, df in data.items():
        cols[sym] = (df["close"] > _ta.ema(df["close"], ema_period)).astype(float)
    wide = pd.DataFrame(cols).sort_index()
    return wide.mean(axis=1)


def f8_weak_breadth(data, symbol, ema_period=20, thresh=0.40):
    breadth = _breadth_above_ema(data, ema_period)
    out = breadth < thresh
    return out.reindex(data[symbol].index)


def f9_btc_direction(data, symbol, period=5, positive=True):
    btc = data.get("BTC")
    if btc is None:
        return pd.Series(True, index=data[symbol].index)
    ret = btc["close"].pct_change(period)
    out = ret > 0 if positive else ret < 0
    return out.reindex(data[symbol].index)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
FILTERS = {
    "F1":  FilterSpec("F1", "Uptrend (close>EMA50)", f1_uptrend),
    "F2":  FilterSpec("F2", "Downtrend (close<EMA50)", f2_downtrend),
    "F3":  FilterSpec("F3", "Ranging (ADX<20)", f3_ranging),
    "F4":  FilterSpec("F4", "Trending (ADX>25)", f4_trending),
    "F5":  FilterSpec("F5", "High volatility", f5_high_vol),
    "F6":  FilterSpec("F6", "Low volatility", f6_low_vol),
    "F7":  FilterSpec("F7", "Volume confirmation", f7_volume_confirm),
    "F8":  FilterSpec("F8", "Weak market breadth", f8_weak_breadth, cross_asset=True),
    "F9":  FilterSpec("F9", "BTC direction", f9_btc_direction, cross_asset=True),
    "F10": FilterSpec("F10", "High-vol regime", f10_high_regime),
}


def filter_mask(filter_id: str, data: "dict[str, pd.DataFrame]", symbol: str,
                params: "dict | None" = None, direction: "str | None" = None) -> pd.Series:
    """Boolean mask for one filter on (symbol), aligned to data[symbol].index.
    `direction` makes direction-aware filters (F9 BTC direction) align with the
    trade side: LONG wants BTC up, SHORT wants BTC down."""
    spec = FILTERS[filter_id]
    p = {**spec.params, **(params or {})}
    if filter_id == "F9" and direction in ("LONG", "SHORT") and "positive" not in (params or {}):
        p["positive"] = (direction == "LONG")
    df = data[symbol]
    if spec.cross_asset:
        raw = spec.fn(data, symbol, **p)
    else:
        raw = spec.fn(df, **p)
    return pd.Series(raw).reindex(df.index).fillna(False).astype(bool)


def apply_filters(entries: pd.Series, filter_ids, data, symbol,
                  direction: "str | None" = None) -> pd.Series:
    """AND a list of filters onto an entry series (direction-aware where relevant)."""
    out = entries
    for fid in filter_ids:
        out = out & filter_mask(fid, data, symbol, direction=direction)
    return out.fillna(False)
