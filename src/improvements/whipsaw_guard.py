"""
Improvement C — whipsaw / chop guards.

Five extra entry conditions that try to keep the strategy out of the regimes
that produced the loss streaks in the diagnosis (range-bound chop, dead vol,
back-to-back losers):

    FC1  ADX(14) > 15          minimal directionality present
    FC2  ATR(5) > ATR(20)      short-term vol expanding vs long-term
    FC3  loss-streak skip       skip 1 signal after 3 straight losing closes
    FC4  60d realised-vol pct > 30%   avoid ultra-low-vol regimes
    FC5  Efficiency Ratio > 0.3       price move is reasonably directional

FC1/FC2/FC4/FC5 are pure look-ahead-safe boolean masks and are registered into
the v3 FILTERS table so the normal `apply_filters` path picks them up. FC3 is
path-dependent (it needs realised trade outcomes) so it rides on the exit config
(ExitConfig.skip_after_n_losses) and the FC3 branch in the simulator.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from strategy import Strategy
import filters as flt
from filters import FilterSpec
from indicators import _ta
from indicators import _ta_extended as tae


# ---------------------------------------------------------------------------
# mask filters
# ---------------------------------------------------------------------------
def fc1_min_trend(df, period=14, thresh=15):
    _, _, adx_val = _ta.adx(df["high"], df["low"], df["close"], period)
    return adx_val > thresh


def fc2_atr_expansion(df, short=5, long=20):
    a_s = _ta.atr(df["high"], df["low"], df["close"], short)
    a_l = _ta.atr(df["high"], df["low"], df["close"], long)
    return a_s > a_l


def fc4_vol_regime(df, period=60, lookback=252, pctile=0.30):
    hv = tae.hist_vol(df["close"], period)
    rank = tae.rolling_percentile(hv, lookback)
    return rank > pctile


def efficiency_ratio(close: pd.Series, n: int = 10) -> pd.Series:
    """Kaufman ER = |close - close[-n]| / sum(|close.diff()|, n). 1 = perfectly
    straight move, ~0 = pure chop. Look-ahead safe (trailing only)."""
    change = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n, min_periods=n).sum()
    return change / vol.replace(0.0, np.nan)


def fc5_efficiency(df, n=10, thresh=0.30):
    return efficiency_ratio(df["close"], n) > thresh


# register the mask filters into the shared table (idempotent)
_FC_FILTERS = {
    "FC1": FilterSpec("FC1", "Min trend (ADX>15)", fc1_min_trend),
    "FC2": FilterSpec("FC2", "ATR expansion (ATR5>ATR20)", fc2_atr_expansion),
    "FC4": FilterSpec("FC4", "Vol regime (60d pct>30%)", fc4_vol_regime),
    "FC5": FilterSpec("FC5", "Efficiency ratio>0.3", fc5_efficiency),
}
flt.FILTERS.update(_FC_FILTERS)

# Which mask filters to try per base (FC3 handled separately below).
MASK_FILTERS = ("FC1", "FC2", "FC4", "FC5")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _with_filter(base: Strategy, fid: str) -> Strategy:
    return replace(base,
                   id=f"{base.id}+{fid}",
                   filters=tuple(base.filters) + (fid,))


def apply_fc3(base: Strategy, n: int = 3) -> Strategy:
    return replace(base,
                   id=f"{base.id}+FC3",
                   exit=replace(base.exit, skip_after_n_losses=n,
                                label=f"{base.exit.label}+FC3"))


def variants(base: Strategy) -> "list[Strategy]":
    """FC1/2/4/5 mask variants plus the FC3 loss-streak variant."""
    out = [_with_filter(base, fid) for fid in MASK_FILTERS]
    out.append(apply_fc3(base))
    return out
