"""nonstandard.py — group S8: non-standard reuse of textbook indicators.

Ten signals that take ordinary indicators (RSI, MACD, Bollinger, OBV, ATR,
Stochastic, ROC) and feed them an *unusual* input series — RSI of volume,
MACD of an ATR series, Bollinger of an RSI series, Stochastic of OBV, etc.

Contract (see docs/v3_signal_contract.md): each function takes ONE asset's
OHLCV frame plus a direction and returns a boolean Series aligned to df.index.
True at day t => enter at t+1 open. Only data at indices <= t may be used.
"""
from __future__ import annotations

from indicators import _ta
from indicators import _ta_extended as tae
from .registry import signal, LONG, SHORT, BOTH

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# S8.01 — RSI of Volume
# --------------------------------------------------------------------------
@signal("S8.01", "RSI of Volume", "S8", BOTH, period=14)
def rsi_of_volume(df, direction, period=14):
    """RSI applied to the volume series instead of price.

    LONG : VolRSI < 20  -> volume exhausted, price bounce expected.
    SHORT: VolRSI > 80  -> volume climax.
    """
    vol_rsi = _ta.rsi(df["volume"], period)
    if direction == LONG:
        return vol_rsi < 20
    return vol_rsi > 80


# --------------------------------------------------------------------------
# S8.02 — MACD of ATR
# --------------------------------------------------------------------------
@signal("S8.02", "MACD of ATR", "S8", BOTH,
        atr_period=14, fast=12, slow=26, signal=9)
def macd_of_atr(df, direction, atr_period=14, fast=12, slow=26, signal=9):
    """MACD computed on an ATR (volatility) series.

    LONG : ATR-MACD dead cross (line crosses below signal) -> volatility
           starting to fall, escape from the low.
    SHORT: ATR-MACD golden cross (line crosses above signal) -> volatility
           spiking up.

    Note the deliberate asymmetry: falling volatility is the bullish cue here.
    """
    atr = _ta.atr(df["high"], df["low"], df["close"], atr_period)
    line, sig, _ = _ta.macd(atr, fast, slow, signal)
    if direction == LONG:
        return _ta.crosses_below(line, sig)
    return _ta.crosses_above(line, sig)


# --------------------------------------------------------------------------
# S8.03 — Bollinger Bands of RSI
# --------------------------------------------------------------------------
@signal("S8.03", "Bollinger Bands of RSI", "S8", BOTH,
        rsi_period=14, bb_period=20, bb_k=2.0)
def bb_of_rsi(df, direction, rsi_period=14, bb_period=20, bb_k=2.0):
    """Bollinger Bands applied to the RSI series (dynamic over/under-bought).

    LONG : RSI < RSI_BB_lower -> dynamically oversold.
    SHORT: RSI > RSI_BB_upper -> dynamically overbought.
    """
    r = _ta.rsi(df["close"], rsi_period)
    _, upper, lower, _ = _ta.bollinger(r, bb_period, bb_k)
    if direction == LONG:
        return r < lower
    return r > upper


# --------------------------------------------------------------------------
# S8.04 — OBV EMA cross
# --------------------------------------------------------------------------
@signal("S8.04", "OBV EMA Cross", "S8", BOTH, fast=10, slow=30)
def obv_ema_cross(df, direction, fast=10, slow=30):
    """EMA(10)/EMA(30) crossover applied to the OBV line.

    LONG : OBV EMA golden cross (fast crosses above slow).
    SHORT: OBV EMA dead cross   (fast crosses below slow).
    """
    obv = _ta.obv(df["close"], df["volume"])
    fast_ema = _ta.ema(obv, fast)
    slow_ema = _ta.ema(obv, slow)
    if direction == LONG:
        return _ta.crosses_above(fast_ema, slow_ema)
    return _ta.crosses_below(fast_ema, slow_ema)


# --------------------------------------------------------------------------
# S8.05 — ATR ratio (short / long)
# --------------------------------------------------------------------------
@signal("S8.05", "ATR Ratio (short/long)", "S8", BOTH,
        short_period=5, long_period=20, high_thr=1.5, low_thr=0.8,
        trend_period=20, spike_lookback=10)
def atr_ratio(df, direction, short_period=5, long_period=20,
              high_thr=1.5, low_thr=0.8, trend_period=20, spike_lookback=10):
    """ATR(5)/ATR(20) — short-term vs long-term volatility.

    LONG : ratio was > 1.5 within the recent lookback (vol spike) AND has now
           settled below 0.8 -> post-spike stabilisation, bottom forming.
    SHORT: ratio > 1.5 AND price in an uptrend -> volatility surging while
           rising (blow-off risk).

    Asymmetric by design (read the spec carefully).
    """
    atr_s = _ta.atr(df["high"], df["low"], df["close"], short_period)
    atr_l = _ta.atr(df["high"], df["low"], df["close"], long_period)
    ratio = atr_s / atr_l.replace(0, np.nan)
    if direction == LONG:
        # a spike happened in the trailing window (strictly before / up to now),
        # and right now the ratio has dropped into the calm zone.
        recent_spike = ((ratio > high_thr).rolling(
            spike_lookback, min_periods=1).max().shift(1) > 0).fillna(False)
        calm_now = ratio < low_thr
        return (recent_spike & calm_now).fillna(False).astype(bool)
    # SHORT: high ratio + uptrend
    uptrend = df["close"] > _ta.sma(df["close"], trend_period)
    return ((ratio > high_thr) & uptrend).fillna(False)


# --------------------------------------------------------------------------
# S8.06 — Dual-timeframe RSI
# --------------------------------------------------------------------------
@signal("S8.06", "Dual-Timeframe RSI", "S8", BOTH,
        daily_period=14, weekly_days=5, weekly_period=14,
        level_lo=30, level_hi=70)
def dual_timeframe_rsi(df, direction, daily_period=14, weekly_days=5,
                       weekly_period=14, level_lo=30, level_hi=70):
    """Higher-timeframe RSI confirmation + daily RSI trigger.

    The "weekly" RSI is RSI(`weekly_period`) computed on a `weekly_days`-day
    resample of close. The resample is RIGHT-labelled / RIGHT-closed, so each
    aggregated bar is stamped on its closing day and forward-filled onto the
    daily grid: a daily bar at t therefore only ever reads a weekly value whose
    window closed on or before t (look-ahead safe — close[t] is known at end of
    day t and entry is at t+1 open).

    LONG : weekly RSI < 30 (both oversold) AND daily RSI(14) crosses up
           through 30 -> rebound after confirmed oversold.
    SHORT: weekly RSI > 70 (both overbought) AND daily RSI(14) crosses down
           through 70 -> decline after confirmed overbought.
    """
    rsi_d = _ta.rsi(df["close"], daily_period)
    wk_close = df["close"].resample(f"{weekly_days}D",
                                    label="right", closed="right").last()
    rsi_w = _ta.rsi(wk_close, weekly_period).reindex(df.index, method="ffill")
    if direction == LONG:
        return ((rsi_w < level_lo) &
                _ta.crosses_above(rsi_d, level_lo)).fillna(False)
    return ((rsi_w > level_hi) &
            _ta.crosses_below(rsi_d, level_hi)).fillna(False)


# --------------------------------------------------------------------------
# S8.07 — MACD-histogram hidden divergence
# --------------------------------------------------------------------------
@signal("S8.07", "MACD-Hist Hidden Divergence", "S8", BOTH,
        fast=12, slow=26, signal=9, window=10, lookback=10)
def macd_hist_hidden_divergence(df, direction, fast=12, slow=26, signal=9,
                                window=10, lookback=10):
    """Hidden divergence between price and the MACD histogram.

    LONG  (hidden bullish): price makes a lower low while the MACD histogram
          makes a higher low -> trend-continuation buy.
    SHORT (hidden bearish): price makes a higher high while the MACD histogram
          makes a lower high.

    Implemented with backward-looking rolling extrema: today's recent extreme
    is compared against the same extreme from a prior, non-overlapping window
    (via .shift(window)). No future bars are referenced.
    """
    _, _, hist = _ta.macd(df["close"], fast, slow, signal)
    close = df["close"]

    if direction == LONG:
        # price lower low: current rolling-min low < prior window's rolling-min low
        cur_low = df["low"].rolling(window, min_periods=window).min()
        prev_low = cur_low.shift(lookback)
        price_ll = cur_low < prev_low
        # hist higher low
        cur_hl = hist.rolling(window, min_periods=window).min()
        prev_hl = cur_hl.shift(lookback)
        hist_hl = cur_hl > prev_hl
        return (price_ll & hist_hl).fillna(False)

    # SHORT
    cur_high = df["high"].rolling(window, min_periods=window).max()
    prev_high = cur_high.shift(lookback)
    price_hh = cur_high > prev_high
    cur_hh = hist.rolling(window, min_periods=window).max()
    prev_hh = cur_hh.shift(lookback)
    hist_lh = cur_hh < prev_hh
    return (price_hh & hist_lh).fillna(False)


# --------------------------------------------------------------------------
# S8.08 — Volume-weighted RSI
# --------------------------------------------------------------------------
@signal("S8.08", "Volume-Weighted RSI", "S8", BOTH, period=14)
def volume_weighted_rsi(df, direction, period=14):
    """RSI in which each bar's price change is weighted by its volume.

    Wilder-style averaging is applied to (gain * volume) and (loss * volume)
    instead of plain gain/loss, so high-volume up/down moves dominate.

    LONG : VW-RSI < 25.
    SHORT: VW-RSI > 75.
    """
    delta = df["close"].diff()
    vol = df["volume"]
    gain = delta.clip(lower=0.0) * vol
    loss = (-delta).clip(lower=0.0) * vol
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    vw_rsi = 100.0 - 100.0 / (1.0 + rs)
    vw_rsi[avg_loss == 0.0] = 100.0
    if direction == LONG:
        return vw_rsi < 25
    return vw_rsi > 75


# --------------------------------------------------------------------------
# S8.09 — Stochastic of OBV
# --------------------------------------------------------------------------
@signal("S8.09", "Stochastic of OBV", "S8", BOTH, lookback=20)
def stochastic_of_obv(df, direction, lookback=20):
    """Stochastic oscillator formula applied to the OBV line (20-day lookback).

    StochOBV = 100 * (OBV - min(OBV, n)) / (max(OBV, n) - min(OBV, n)).

    LONG : StochOBV < 10  -> accumulation bottom.
    SHORT: StochOBV > 90  -> distribution top.
    """
    obv = _ta.obv(df["close"], df["volume"])
    ll = obv.rolling(lookback, min_periods=lookback).min()
    hh = obv.rolling(lookback, min_periods=lookback).max()
    rng = (hh - ll).replace(0.0, np.nan)
    stoch_obv = 100.0 * (obv - ll) / rng
    if direction == LONG:
        return stoch_obv < 10
    return stoch_obv > 90


# --------------------------------------------------------------------------
# S8.10 — ROC of RSI (RSI rate-of-change)
# --------------------------------------------------------------------------
@signal("S8.10", "ROC of RSI", "S8", BOTH, rsi_period=14, roc_period=5)
def roc_of_rsi(df, direction, rsi_period=14, roc_period=5):
    """Rate-of-change of RSI(14) over `roc_period` bars (absolute point change).

    LONG : RSI_ROC < -30 (RSI crashing -> over-reaction) AND RSI < 40.
    SHORT: RSI_ROC >  30 (RSI surging)               AND RSI > 60.
    """
    r = _ta.rsi(df["close"], rsi_period)
    rsi_roc = r - r.shift(roc_period)   # change in RSI points over the window
    if direction == LONG:
        return ((rsi_roc < -30) & (r < 40)).fillna(False)
    return ((rsi_roc > 30) & (r > 60)).fillna(False)
