"""S3 — Alternative trend indicators (12 signals).

Noise-robust / adaptive trend tools used instead of plain SMA/EMA crosses.
All signals are LOOK-AHEAD SAFE: the value at index t uses only data at t' <= t.
A True at day t means "enter at t+1 open" (the backtester handles entry timing).
"""
from __future__ import annotations

from indicators import _ta
from indicators import _ta_extended as tae
from .registry import signal, LONG, SHORT, BOTH

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# S3.01 — Supertrend trend flip
# ---------------------------------------------------------------------------
@signal("S3.01", "Supertrend Flip", "S3", BOTH, period=10, mult=3.0)
def supertrend_flip(df, direction, period=10, mult=3.0):
    _, dirn = tae.supertrend(df["high"], df["low"], df["close"], period, mult)
    prev = dirn.shift(1)
    if direction == "LONG":
        # trend reverses down -> up (price > supertrend)
        return (dirn > 0) & (prev <= 0)
    # trend reverses up -> down (price < supertrend)
    return (dirn < 0) & (prev >= 0)


# ---------------------------------------------------------------------------
# S3.02 — Hull MA cross
# ---------------------------------------------------------------------------
@signal("S3.02", "Hull MA Cross", "S3", BOTH, n=20)
def hull_ma_cross(df, direction, n=20):
    hma = tae.hull_ma(df["close"], n)
    if direction == "LONG":
        return _ta.crosses_above(df["close"], hma)
    return _ta.crosses_below(df["close"], hma)


# ---------------------------------------------------------------------------
# S3.03 — DEMA fast/slow cross
# ---------------------------------------------------------------------------
@signal("S3.03", "DEMA Cross", "S3", BOTH, fast=10, slow=30)
def dema_cross(df, direction, fast=10, slow=30):
    fast_line = tae.dema(df["close"], fast)
    slow_line = tae.dema(df["close"], slow)
    if direction == "LONG":
        return _ta.crosses_above(fast_line, slow_line)
    return _ta.crosses_below(fast_line, slow_line)


# ---------------------------------------------------------------------------
# S3.04 — TEMA fast/slow cross
# ---------------------------------------------------------------------------
@signal("S3.04", "TEMA Cross", "S3", BOTH, fast=10, slow=30)
def tema_cross(df, direction, fast=10, slow=30):
    fast_line = tae.tema(df["close"], fast)
    slow_line = tae.tema(df["close"], slow)
    if direction == "LONG":
        return _ta.crosses_above(fast_line, slow_line)
    return _ta.crosses_below(fast_line, slow_line)


# ---------------------------------------------------------------------------
# S3.05 — KAMA (Kaufman Adaptive Moving Average) cross
# ---------------------------------------------------------------------------
@signal("S3.05", "KAMA Cross", "S3", BOTH, er_period=10, fast=2, slow=30)
def kama_cross(df, direction, er_period=10, fast=2, slow=30):
    k = tae.kama(df["close"], er_period, fast, slow)
    if direction == "LONG":
        return _ta.crosses_above(df["close"], k)
    return _ta.crosses_below(df["close"], k)


# ---------------------------------------------------------------------------
# S3.06 — McGinley Dynamic cross
# ---------------------------------------------------------------------------
@signal("S3.06", "McGinley Dynamic Cross", "S3", BOTH, n=14)
def mcginley_cross(df, direction, n=14):
    md = tae.mcginley(df["close"], n)
    if direction == "LONG":
        return _ta.crosses_above(df["close"], md)
    return _ta.crosses_below(df["close"], md)


# ---------------------------------------------------------------------------
# S3.07 — VIDYA cross
# ---------------------------------------------------------------------------
@signal("S3.07", "VIDYA Cross", "S3", BOTH, cmo_period=9, n=12)
def vidya_cross(df, direction, cmo_period=9, n=12):
    v = tae.vidya(df["close"], cmo_period, n)
    if direction == "LONG":
        return _ta.crosses_above(df["close"], v)
    return _ta.crosses_below(df["close"], v)


# ---------------------------------------------------------------------------
# S3.08 — Linear Regression Slope sign flip
# ---------------------------------------------------------------------------
@signal("S3.08", "Linear Regression Slope Flip", "S3", BOTH, n=20)
def linreg_slope_flip(df, direction, n=20):
    slope = tae.linreg_slope(df["close"], n)
    prev = slope.shift(1)
    if direction == "LONG":
        # slope flips negative -> positive
        return (slope > 0) & (prev <= 0)
    # slope flips positive -> negative
    return (slope < 0) & (prev >= 0)


# ---------------------------------------------------------------------------
# S3.09 — Linear Regression Channel re-entry (mean reversion)
# ---------------------------------------------------------------------------
@signal("S3.09", "Linear Regression Channel Re-entry", "S3", BOTH, n=20, k=2.0)
def linreg_channel_reentry(df, direction, n=20, k=2.0):
    mid, upper, lower = tae.linreg_channel(df["close"], n, k)
    close = df["close"]
    if direction == "LONG":
        # broke below lower band, then crosses back up into the channel
        return _ta.crosses_above(close, lower)
    # broke above upper band, then crosses back down into the channel
    return _ta.crosses_below(close, upper)


# ---------------------------------------------------------------------------
# S3.10 — Heikin-Ashi trend reversal (3+ bars then flip)
# ---------------------------------------------------------------------------
@signal("S3.10", "Heikin-Ashi Trend Reversal", "S3", BOTH, run=3)
def heikin_ashi_reversal(df, direction, run=3):
    ha_open, _, _, ha_close = tae.heikin_ashi(df["open"], df["high"],
                                              df["low"], df["close"])
    bull = ha_close > ha_open
    bear = ha_close < ha_open
    if direction == "LONG":
        # >= `run` consecutive HA bearish bars ending at t-1, flip bullish at t
        prior_bear = pd.Series(True, index=df.index)
        for k in range(1, run + 1):
            prior_bear &= bear.shift(k).fillna(False)
        return (bull & prior_bear).fillna(False)
    # >= `run` consecutive HA bullish bars ending at t-1, flip bearish at t
    prior_bull = pd.Series(True, index=df.index)
    for k in range(1, run + 1):
        prior_bull &= bull.shift(k).fillna(False)
    return (bear & prior_bull).fillna(False)


# ---------------------------------------------------------------------------
# S3.11 — Ichimoku Tenkan/Kijun cross (cloud-filtered)
# ---------------------------------------------------------------------------
@signal("S3.11", "Ichimoku TK Cross", "S3", BOTH, tenkan=9, kijun=26, senkou=52)
def ichimoku_tk_cross(df, direction, tenkan=9, kijun=26, senkou=52):
    conv, base, span_a, span_b = _ta.ichimoku(df["high"], df["low"],
                                              tenkan, kijun, senkou)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    close = df["close"]
    if direction == "LONG":
        # Tenkan crosses above Kijun while price is above the cloud
        return _ta.crosses_above(conv, base) & (close > cloud_top)
    # Tenkan crosses below Kijun while price is below the cloud
    return _ta.crosses_below(conv, base) & (close < cloud_bot)


# ---------------------------------------------------------------------------
# S3.12 — Pivot Point breakout (R1 / S1 from previous bar)
# ---------------------------------------------------------------------------
@signal("S3.12", "Pivot Point Breakout", "S3", BOTH)
def pivot_point_breakout(df, direction):
    _, r1, s1 = tae.pivot_points(df["high"], df["low"], df["close"])
    close = df["close"]
    if direction == "LONG":
        # close breaks above R1
        return _ta.crosses_above(close, r1)
    # close breaks below S1
    return _ta.crosses_below(close, s1)
