"""
alt_momentum.py — signal group S2 (alternative / improved momentum oscillators).

Sixteen entry-signal generators built on the S2 helpers in
``indicators._ta_extended``. Each function is per-asset:

    def fn(df, direction, **params) -> boolean pd.Series  (aligned to df.index)

A True at day t means "enter at the NEXT bar's open (t+1)". Every computation
is LOOK-AHEAD SAFE: the value at t depends only on data at indices <= t
(rolling / ewm / shift(+k) and the recursive helpers, which are all backward
looking). Cross-style ("크로스업/크로스다운") wording is implemented with
``_ta.crosses_above`` / ``_ta.crosses_below``; div-by-zero is guarded with
``.replace(0, np.nan)``.
"""
from __future__ import annotations

from indicators import _ta
from indicators import _ta_extended as tae
from .registry import signal, LONG, SHORT, BOTH

import numpy as np
import pandas as pd


# ===========================================================================
# S2.01 — Fisher Transform
# ===========================================================================
@signal("S2.01", "Fisher Transform turn", "S2", BOTH, period=10)
def fisher_transform(df, direction, period=10):
    """LONG: fisher turns up from below -1 (fisher > prev AND prev < -1).
    SHORT: fisher turns down from above +1 (fisher < prev AND prev > 1)."""
    fish = tae.fisher_transform(df["high"], df["low"], period)
    prev = fish.shift(1)
    if direction == LONG:
        return (fish > prev) & (prev < -1.0)
    return (fish < prev) & (prev > 1.0)


# ===========================================================================
# S2.02 — Inverse Fisher Transform of RSI
# ===========================================================================
@signal("S2.02", "Inverse Fisher RSI cross", "S2", BOTH, rsi_period=14)
def inverse_fisher_rsi(df, direction, rsi_period=14):
    """LONG: IFT crosses up through -0.5.  SHORT: IFT crosses down through +0.5."""
    ift = tae.inverse_fisher_rsi(df["close"], rsi_period)
    if direction == LONG:
        return _ta.crosses_above(ift, -0.5)
    return _ta.crosses_below(ift, 0.5)


# ===========================================================================
# S2.03 — Connors RSI
# ===========================================================================
@signal("S2.03", "Connors RSI zone", "S2", BOTH,
        rsi_period=3, streak_period=2, rank_period=100)
def connors_rsi(df, direction, rsi_period=3, streak_period=2, rank_period=100):
    """LONG: CRSI < 15 (oversold).  SHORT: CRSI > 85 (overbought)."""
    crsi = tae.connors_rsi(df["close"], rsi_period, streak_period, rank_period)
    if direction == LONG:
        return crsi < 15.0
    return crsi > 85.0


# ===========================================================================
# S2.04 — TSI (True Strength Index)
# ===========================================================================
@signal("S2.04", "TSI signal-line cross", "S2", BOTH, r=25, s=13, signal_period=7)
def tsi(df, direction, r=25, s=13, signal_period=7):
    """LONG: TSI crosses up through signal while TSI < 0 (oversold region).
    SHORT: TSI crosses down through signal while TSI > 0."""
    tsi_val, sig = tae.tsi(df["close"], r, s, signal_period)
    if direction == LONG:
        return _ta.crosses_above(tsi_val, sig) & (tsi_val < 0.0)
    return _ta.crosses_below(tsi_val, sig) & (tsi_val > 0.0)


# ===========================================================================
# S2.05 — Coppock Curve
# ===========================================================================
@signal("S2.05", "Coppock sign flip", "S2", BOTH, roc1=14, roc2=11, wma_period=10)
def coppock(df, direction, roc1=14, roc2=11, wma_period=10):
    """LONG: Coppock turns from negative to positive.
    SHORT: Coppock turns from positive to negative."""
    cop = tae.coppock(df["close"], roc1, roc2, wma_period)
    prev = cop.shift(1)
    if direction == LONG:
        return (cop > 0.0) & (prev <= 0.0)
    return (cop < 0.0) & (prev >= 0.0)


# ===========================================================================
# S2.06 — KST (Know Sure Thing)
# ===========================================================================
@signal("S2.06", "KST signal-line cross", "S2", BOTH)
def kst(df, direction):
    """LONG: KST crosses up through its signal.  SHORT: crosses down."""
    kst_val, sig = tae.kst(df["close"])
    if direction == LONG:
        return _ta.crosses_above(kst_val, sig)
    return _ta.crosses_below(kst_val, sig)


# ===========================================================================
# S2.07 — Aroon Oscillator
# ===========================================================================
@signal("S2.07", "Aroon Oscillator cross", "S2", BOTH, period=25)
def aroon_oscillator(df, direction, period=25):
    """LONG: oscillator crosses up through +50.
    SHORT: oscillator crosses down through -50."""
    _, _, osc = tae.aroon(df["high"], df["low"], period)
    if direction == LONG:
        return _ta.crosses_above(osc, 50.0)
    return _ta.crosses_below(osc, -50.0)


# ===========================================================================
# S2.08 — Vortex Indicator
# ===========================================================================
@signal("S2.08", "Vortex +VI/-VI cross", "S2", BOTH, n=14)
def vortex(df, direction, n=14):
    """LONG: +VI crosses above -VI.  SHORT: -VI crosses above +VI."""
    plus_vi, minus_vi = tae.vortex(df["high"], df["low"], df["close"], n)
    if direction == LONG:
        return _ta.crosses_above(plus_vi, minus_vi)
    return _ta.crosses_above(minus_vi, plus_vi)


# ===========================================================================
# S2.09 — Elder Ray (Bull / Bear Power)  [asymmetric]
# ===========================================================================
@signal("S2.09", "Elder Ray power turn", "S2", BOTH, period=13)
def elder_ray(df, direction, period=13):
    """LONG: BearPower < 0 and turning up (rising vs prior bar) with EMA rising.
    SHORT: BullPower > 0 and turning down (falling vs prior bar) with EMA falling."""
    bull, bear, ema_ = tae.elder_ray(df["high"], df["low"], df["close"], period)
    ema_up = ema_ > ema_.shift(1)
    ema_down = ema_ < ema_.shift(1)
    if direction == LONG:
        return (bear < 0.0) & (bear > bear.shift(1)) & ema_up
    return (bull > 0.0) & (bull < bull.shift(1)) & ema_down


# ===========================================================================
# S2.10 — CMO (Chande Momentum Oscillator)
# ===========================================================================
@signal("S2.10", "CMO threshold cross", "S2", BOTH, period=14)
def cmo(df, direction, period=14):
    """LONG: CMO crosses up through -50.  SHORT: CMO crosses down through +50."""
    c = tae.cmo(df["close"], period)
    if direction == LONG:
        return _ta.crosses_above(c, -50.0)
    return _ta.crosses_below(c, 50.0)


# ===========================================================================
# S2.11 — DPO (Detrended Price Oscillator)
# ===========================================================================
@signal("S2.11", "DPO sign flip", "S2", BOTH, period=20)
def dpo(df, direction, period=20):
    """LONG: DPO turns from negative to positive.
    SHORT: DPO turns from positive to negative."""
    d = tae.dpo(df["close"], period)
    prev = d.shift(1)
    if direction == LONG:
        return (d > 0.0) & (prev <= 0.0)
    return (d < 0.0) & (prev >= 0.0)


# ===========================================================================
# S2.12 — Ultimate Oscillator (with divergence)  [asymmetric]
# ===========================================================================
@signal("S2.12", "Ultimate Oscillator divergence", "S2", BOTH,
        p1=7, p2=14, p3=28, lookback=14)
def ultimate_oscillator(df, direction, p1=7, p2=14, p3=28, lookback=14):
    """LONG: UO < 30 (oversold) with bullish divergence — price makes a lower
    low over `lookback` while UO makes a higher low.
    SHORT: UO > 70 (overbought) with bearish divergence — price makes a higher
    high while UO makes a lower high."""
    uo = tae.ultimate_oscillator(df["high"], df["low"], df["close"], p1, p2, p3)
    close = df["close"]
    prior_low = close.rolling(lookback, min_periods=lookback).min().shift(1)
    prior_high = close.rolling(lookback, min_periods=lookback).max().shift(1)
    uo_prior_low = uo.rolling(lookback, min_periods=lookback).min().shift(1)
    uo_prior_high = uo.rolling(lookback, min_periods=lookback).max().shift(1)
    if direction == LONG:
        price_lower_low = close < prior_low
        uo_higher_low = uo > uo_prior_low
        return (uo < 30.0) & price_lower_low & uo_higher_low
    price_higher_high = close > prior_high
    uo_lower_high = uo < uo_prior_high
    return (uo > 70.0) & price_higher_high & uo_lower_high


# ===========================================================================
# S2.13 — Stochastic RSI
# ===========================================================================
@signal("S2.13", "Stochastic RSI extreme cross", "S2", BOTH,
        rsi_period=14, stoch_period=14)
def stoch_rsi(df, direction, rsi_period=14, stoch_period=14):
    """LONG: StochRSI was < 0.1 (recently oversold) and crosses up through 0.2.
    SHORT: StochRSI was > 0.9 (recently overbought) and crosses down through 0.8."""
    sr = tae.stoch_rsi(df["close"], rsi_period, stoch_period)
    if direction == LONG:
        was_oversold = (sr.shift(1) < 0.1)
        return was_oversold & _ta.crosses_above(sr, 0.2)
    was_overbought = (sr.shift(1) > 0.9)
    return was_overbought & _ta.crosses_below(sr, 0.8)


# ===========================================================================
# S2.14 — RVI (Relative Vigor Index)
# ===========================================================================
@signal("S2.14", "RVI signal-line cross", "S2", BOTH, period=4)
def rvi(df, direction, period=4):
    """LONG: RVI crosses up through its signal while RVI < 0 (negative region).
    SHORT: RVI crosses down through its signal while RVI > 0."""
    rvi_val, sig = tae.rvi(df["open"], df["high"], df["low"], df["close"], period)
    if direction == LONG:
        return _ta.crosses_above(rvi_val, sig) & (rvi_val < 0.0)
    return _ta.crosses_below(rvi_val, sig) & (rvi_val > 0.0)


# ===========================================================================
# S2.15 — Mass Index  [asymmetric]
# ===========================================================================
@signal("S2.15", "Mass Index reversal bulge", "S2", BOTH,
        ema_period=9, sum_period=25, trend_period=9, bulge_window=15)
def mass_index(df, direction, ema_period=9, sum_period=25, trend_period=9,
               bulge_window=15):
    """Reversal "bulge": Mass Index rose above 27 then falls back below 26.5.
    LONG: bulge while price is in a downtrend (EMA falling) -> expect reversal up.
    SHORT: bulge while price is in an uptrend (EMA rising) -> expect reversal down."""
    mi = tae.mass_index(df["high"], df["low"], ema_period, sum_period)
    # "MI > 27 후 < 26.5": MI poked above 27 at some point in the recent window,
    # then now crosses back down through 26.5 (the reversal bulge completing).
    recent_above_27 = (mi > 27.0).rolling(
        bulge_window, min_periods=1).max().shift(1).fillna(0.0).astype(bool)
    bulge = recent_above_27 & _ta.crosses_below(mi, 26.5)
    trend_ema = tae.ema(df["close"], trend_period)
    uptrend = trend_ema > trend_ema.shift(1)
    downtrend = trend_ema < trend_ema.shift(1)
    if direction == LONG:
        return bulge & downtrend
    return bulge & uptrend


# ===========================================================================
# S2.16 — Random Walk Index  [asymmetric]
# ===========================================================================
@signal("S2.16", "Random Walk Index rebound", "S2", BOTH, n=14, window=10)
def random_walk_index(df, direction, n=14, window=10):
    """LONG: a recent strong down-leg (RWI_low > 1.0 within the trailing window)
    followed by RWI_high rebounding above RWI_low (crosses above) -> upside turn.
    SHORT: a recent strong up-leg (RWI_high > 1.0 within the window) followed by
    RWI_low rebounding above RWI_high (crosses above) -> downside turn."""
    rwi_high, rwi_low = tae.random_walk_index(df["high"], df["low"], df["close"], n)
    if direction == LONG:
        recent_down = (rwi_low > 1.0).rolling(
            window, min_periods=1).max().shift(1).fillna(0.0).astype(bool)
        return recent_down & _ta.crosses_above(rwi_high, rwi_low)
    recent_up = (rwi_high > 1.0).rolling(
        window, min_periods=1).max().shift(1).fillna(0.0).astype(bool)
    return recent_up & _ta.crosses_above(rwi_low, rwi_high)
