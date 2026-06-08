"""
statistical.py — group S6 signal module (statistical / time-series signals).

15 signals borrowed from statistics / time-series analysis rather than textbook
TA. Every function is per-asset:

    def fn(df, direction, **params) -> boolean pd.Series aligned to df.index

A True at day t means "enter at t+1 open" — the value at t depends only on data
at indices <= t (rolling / ewm / shift(+k)). Registration happens at import time
via the @signal decorator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import _ta
from indicators import _ta_extended as tae
from .registry import signal, LONG, SHORT, BOTH


# ---------------------------------------------------------------------------
# S6.01 — Return Z-Score
# ---------------------------------------------------------------------------
@signal("S6.01", "Return Z-Score", "S6", BOTH, n=1, lookback=60, thr=2.0)
def return_zscore(df, direction, n=1, lookback=60, thr=2.0):
    """z = (ret_N - mean(ret_N, lookback)) / std(ret_N, lookback). N=1.
    LONG: z < -thr (extreme drop, mean reversion). SHORT: z > +thr."""
    ret = df["close"].pct_change(n)
    z = tae.zscore(ret, lookback)
    if direction == LONG:
        return z < -thr
    return z > thr


# ---------------------------------------------------------------------------
# S6.02 — Cumulative Return Z-Score
# ---------------------------------------------------------------------------
@signal("S6.02", "Cumulative Return Z-Score", "S6", BOTH, cum=5, lookback=120, thr=2.0)
def cumulative_return_zscore(df, direction, cum=5, lookback=120, thr=2.0):
    """z of the `cum`-day cumulative return, lookback=120.
    LONG: z < -thr. SHORT: z > +thr."""
    cum_ret = df["close"].pct_change(cum)
    z = tae.zscore(cum_ret, lookback)
    if direction == LONG:
        return z < -thr
    return z > thr


# ---------------------------------------------------------------------------
# S6.03 — Price Percentile Rank
# ---------------------------------------------------------------------------
@signal("S6.03", "Price Percentile Rank", "S6", BOTH, n=252, low_pct=0.10, high_pct=0.90)
def price_percentile_rank(df, direction, n=252, low_pct=0.10, high_pct=0.90):
    """Percentile rank (0..1) of the current close within the trailing n bars.
    LONG: pctile < 10%. SHORT: pctile > 90%."""
    pct = tae.rolling_percentile(df["close"], n)
    if direction == LONG:
        return pct < low_pct
    return pct > high_pct


# ---------------------------------------------------------------------------
# S6.04 — Hurst Exponent (간이)
# ---------------------------------------------------------------------------
@signal("S6.04", "Hurst Exponent", "S6", BOTH, period=60, low_h=0.4, high_h=0.6)
def hurst_exponent_regime(df, direction, period=60, low_h=0.4, high_h=0.6):
    """Rolling Hurst exponent (R/S-style), period=60.
    LONG: Hurst < 0.4 (mean-reverting) AND price falling today.
    SHORT: Hurst > 0.6 (trending) AND price starting to fall today."""
    h = tae.hurst_exponent(df["close"], n=period)
    falling = df["close"] < df["close"].shift(1)
    if direction == LONG:
        return (h < low_h) & falling
    return (h > high_h) & falling


# ---------------------------------------------------------------------------
# S6.05 — Autocorrelation Flip
# ---------------------------------------------------------------------------
@signal("S6.05", "Autocorrelation Flip", "S6", BOTH, n=20, lag=1, thr=0.3)
def autocorrelation_flip(df, direction, n=20, lag=1, thr=0.3):
    """lag-1 autocorrelation of returns, rolling 20.
    LONG: AC < -0.3 (excess reversal -> trend flip soon).
    SHORT: AC > +0.3 (trend persisting -> follow)."""
    ret = df["close"].pct_change()
    ac = tae.rolling_autocorr(ret, n=n, lag=lag)
    if direction == LONG:
        return ac < -thr
    return ac > thr


# ---------------------------------------------------------------------------
# S6.06 — Skewness Shift
# ---------------------------------------------------------------------------
@signal("S6.06", "Skewness Shift", "S6", BOTH, n=20, thr=1.5)
def skewness_shift(df, direction, n=20, thr=1.5):
    """Rolling 20-day return skewness.
    LONG: skew < -1.5 (left-skewed extreme -> bounce).
    SHORT: skew > +1.5 (right-skewed -> drop)."""
    skew = df["close"].pct_change().rolling(n, min_periods=n).skew()
    if direction == LONG:
        return skew < -thr
    return skew > thr


# ---------------------------------------------------------------------------
# S6.07 — Kurtosis Spike
# ---------------------------------------------------------------------------
@signal("S6.07", "Kurtosis Spike", "S6", BOTH, n=20, thr=6.0)
def kurtosis_spike(df, direction, n=20, thr=6.0):
    """Rolling 20-day return kurtosis.
    LONG: kurtosis > 6 (fat tail) AND a down day -> vol exhaustion bounce.
    SHORT: kurtosis > 6 AND after a rise -> reversal."""
    kurt = df["close"].pct_change().rolling(n, min_periods=n).kurt()
    up_day = df["close"] > df["close"].shift(1)
    down_day = df["close"] < df["close"].shift(1)
    if direction == LONG:
        return (kurt > thr) & down_day
    return (kurt > thr) & up_day


# ---------------------------------------------------------------------------
# S6.08 — Entropy of Returns
# ---------------------------------------------------------------------------
@signal("S6.08", "Entropy of Returns", "S6", BOTH, n=30, bins=10, pct_n=120, low_pct=0.20)
def entropy_of_returns(df, direction, n=30, bins=10, pct_n=120, low_pct=0.20):
    """Shannon entropy of the binned return distribution, rolling 30.
    Entropy plunge (< 20th percentile of its own trailing window) signals an
    extreme market bias -> reversal. Same condition for LONG and SHORT."""
    ret = df["close"].pct_change()
    ent = tae.return_entropy(ret, n=n, bins=bins)
    ent_pct = tae.rolling_percentile(ent, pct_n)
    return ent_pct < low_pct


# ---------------------------------------------------------------------------
# S6.09 — Variance Ratio
# ---------------------------------------------------------------------------
@signal("S6.09", "Variance Ratio", "S6", BOTH, k=5, n=60, low_vr=0.7, high_vr=1.3)
def variance_ratio_regime(df, direction, k=5, n=60, low_vr=0.7, high_vr=1.3):
    """VR = var(ret_k) / (k * var(ret_1d)), rolling 60.
    LONG: VR < 0.7 (excess reversal, mean-revert) AND price falling.
    SHORT: VR > 1.3 (excess trend) AND price rising (trend follow)."""
    vr = tae.variance_ratio(df["close"], k=k, n=n)
    falling = df["close"] < df["close"].shift(1)
    rising = df["close"] > df["close"].shift(1)
    if direction == LONG:
        return (vr < low_vr) & falling
    return (vr > high_vr) & rising


# ---------------------------------------------------------------------------
# S6.10 — Distance from MA (Detrended)
# ---------------------------------------------------------------------------
@signal("S6.10", "Distance from MA", "S6", BOTH, n=50, thr=15.0)
def distance_from_ma(df, direction, n=50, thr=15.0):
    """(close - SMA(50)) / SMA(50) * 100.
    LONG: distance < -15% (excess undershoot, mean-revert).
    SHORT: distance > +15%."""
    dist = tae.distance_from_ma(df["close"], n=n)
    if direction == LONG:
        return dist < -thr
    return dist > thr


# ---------------------------------------------------------------------------
# S6.11 — Consecutive Directional Days
# ---------------------------------------------------------------------------
@signal("S6.11", "Consecutive Directional Days", "S6", BOTH, run=5)
def consecutive_directional_days(df, direction, run=5):
    """Signed run length of consecutive up/down closes.
    LONG: 5+ consecutive down days -> reversal expected.
    SHORT: 5+ consecutive up days -> reversal expected."""
    streak = tae.consecutive_directional(df["close"])
    if direction == LONG:
        return streak <= -run
    return streak >= run


# ---------------------------------------------------------------------------
# S6.12 — High-Low Range Ratio
# ---------------------------------------------------------------------------
@signal("S6.12", "High-Low Range Ratio", "S6", BOTH, n=20, thr=2.5)
def high_low_range_ratio(df, direction, n=20, thr=2.5):
    """today_range / avg_range(20).
    LONG: ratio > 2.5 AND a down day -> excess fear bounce.
    SHORT: ratio > 2.5 AND an up day -> overheated."""
    today_range = df["high"] - df["low"]
    avg_range = tae.sma(today_range, n)
    ratio = today_range / avg_range.replace(0.0, np.nan)
    up_day = df["close"] > df["close"].shift(1)
    down_day = df["close"] < df["close"].shift(1)
    if direction == LONG:
        return (ratio > thr) & down_day
    return (ratio > thr) & up_day


# ---------------------------------------------------------------------------
# S6.13 — Close Location Value (CLV)
# ---------------------------------------------------------------------------
@signal("S6.13", "Close Location Value", "S6", BOTH, n=5, thr=0.7)
def close_location_value(df, direction, n=5, thr=0.7):
    """5-day average of CLV = (2C - H - L) / (H - L).
    LONG: CLV_avg < -0.7 (closing near lows repeatedly -> exhaustion).
    SHORT: CLV_avg > +0.7."""
    c = tae.clv(df["high"], df["low"], df["close"])
    clv_avg = c.rolling(n, min_periods=n).mean()
    if direction == LONG:
        return clv_avg < -thr
    return clv_avg > thr


# ---------------------------------------------------------------------------
# S6.14 — Median Reversion
# ---------------------------------------------------------------------------
@signal("S6.14", "Median Reversion", "S6", BOTH, n=50, thr=2.5)
def median_reversion(df, direction, n=50, thr=2.5):
    """(close - rolling_median(50)) / rolling_mad(50).
    LONG: < -2.5 (extreme undervaluation vs median).
    SHORT: > +2.5."""
    mr = tae.median_reversion(df["close"], n=n)
    if direction == LONG:
        return mr < -thr
    return mr > thr


# ---------------------------------------------------------------------------
# S6.15 — Return Regime (Hidden State)
# ---------------------------------------------------------------------------
@signal("S6.15", "Return Regime", "S6", BOTH, window=10, thr=7)
def return_regime(df, direction, window=10, thr=7):
    """Count of return sign flips over the last `window` days.
    flips > 7 (choppy -> direction decision imminent), then:
    LONG: last candle is bullish (close > open).
    SHORT: last candle is bearish (close < open)."""
    ret = df["close"].pct_change()
    sign = np.sign(ret)
    # a "flip" = today's sign differs from yesterday's (both non-zero)
    flip = (sign != sign.shift(1)) & (sign != 0) & (sign.shift(1) != 0)
    flip_count = flip.rolling(window, min_periods=window).sum()
    bullish = df["close"] > df["open"]
    bearish = df["close"] < df["open"]
    if direction == LONG:
        return (flip_count > thr) & bullish
    return (flip_count > thr) & bearish
