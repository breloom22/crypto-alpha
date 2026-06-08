"""
volatility.py — signal group S4 (Volatility / Range based).

Implements the 12 S4 strategies from
``crypto_alpha_v3_300plus_strategies.md`` (lines 105-121). Every signal is a
per-asset function returning a boolean entry Series aligned to ``df.index``:
a True at day t means "enter at t+1 open". All logic is LOOK-AHEAD SAFE — the
value at t uses only data at indices <= t (rolling / ewm / shift(+k)).

Foundation used:
  * ``tae.*``  — ttm_squeeze, chaikin_volatility, hist_vol, natr,
                 yang_zhang_vol, parkinson_vol, ulcer_index,
                 atr_trailing_stop, chandelier_exit, rolling_percentile.
  * ``_ta.bollinger`` / ``_ta.keltner`` — for BB %B and Keltner-BB spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import _ta
from indicators import _ta_extended as tae
from .registry import signal, LONG, SHORT, BOTH


# ===========================================================================
# S4.01 — Squeeze Momentum (TTM concept)
# ===========================================================================
@signal("S4.01", "Squeeze Momentum (TTM)", "S4", BOTH,
        bb_n=20, bb_k=2.0, kc_n=20, kc_mult=1.5)
def squeeze_momentum(df, direction, bb_n=20, bb_k=2.0, kc_n=20, kc_mult=1.5):
    """Squeeze release (BB exits Keltner) + momentum sign follow-through.

    LONG : squeeze released this bar AND momentum > 0.
    SHORT: squeeze released this bar AND momentum < 0.
    """
    squeeze_on, momentum = tae.ttm_squeeze(
        df["high"], df["low"], df["close"], bb_n, bb_k, kc_n, kc_mult)
    # squeeze "released" = was on yesterday, off today.
    released = squeeze_on.shift(1).fillna(False) & (~squeeze_on)
    if direction == LONG:
        return released & (momentum > 0)
    return released & (momentum < 0)


# ===========================================================================
# S4.02 — Chaikin Volatility
# ===========================================================================
@signal("S4.02", "Chaikin Volatility Extreme Reversal", "S4", BOTH,
        n=10, level=50.0, loc_n=20)
def chaikin_volatility_rev(df, direction, n=10, level=50.0, loc_n=20):
    """CV extreme high (>level) then turning down, combined with price extreme.

    LONG : CV was extreme (>level) and turns down  + price at a recent LOW.
    SHORT: CV was extreme (>level) and turns down  + price at a recent HIGH.
    """
    cv = tae.chaikin_volatility(df["high"], df["low"], n)
    # extreme high yesterday, falling today (peak roll-over after a spike).
    cv_extreme_prev = cv.shift(1) > level
    cv_turning_down = cv < cv.shift(1)
    cv_signal = cv_extreme_prev & cv_turning_down
    close = df["close"]
    prior_low = df["low"].rolling(loc_n, min_periods=loc_n).min().shift(1)
    prior_high = df["high"].rolling(loc_n, min_periods=loc_n).max().shift(1)
    if direction == LONG:
        price_low = close <= prior_low          # price making/at a recent trough
        return cv_signal & price_low
    price_high = close >= prior_high            # price making/at a recent peak
    return cv_signal & price_high


# ===========================================================================
# S4.03 — Historical Vol Percentile (compression breakout)
# ===========================================================================
@signal("S4.03", "Historical Vol Percentile Breakout", "S4", BOTH,
        hv_n=20, pct_n=252, low_thr=0.10, brk_n=20)
def hist_vol_percentile(df, direction, hv_n=20, pct_n=252, low_thr=0.10, brk_n=20):
    """20d return-std percentile rank in 252d window. Extreme-low vol
    (< low_thr) primes a breakout; direction follows the break.

    LONG : HV pctile < low_thr AND close breaks ABOVE prior brk_n-day high.
    SHORT: HV pctile < low_thr AND close breaks BELOW prior brk_n-day low.
    """
    hv = tae.hist_vol(df["close"], hv_n)
    pct = tae.rolling_percentile(hv, pct_n)
    low_vol = pct < low_thr
    close = df["close"]
    prior_high = df["high"].rolling(brk_n, min_periods=brk_n).max().shift(1)
    prior_low = df["low"].rolling(brk_n, min_periods=brk_n).min().shift(1)
    if direction == LONG:
        return low_vol & _ta.crosses_above(close, prior_high)
    return low_vol & _ta.crosses_below(close, prior_low)


# ===========================================================================
# S4.04 — Normalized ATR (overreaction reversal)
# ===========================================================================
@signal("S4.04", "Normalized ATR Overreaction Reversal", "S4", BOTH,
        n=14, pct_n=60, high_thr=0.90, trend_n=5)
def natr_overreaction(df, direction, n=14, pct_n=60, high_thr=0.90, trend_n=5):
    """NATR = ATR(14)/close*100. Extreme NATR (> high_thr pctile, 60d) flags an
    over-extended, exhausted move.

    LONG : NATR > 90th pctile AND a down day  -> over-reaction snap-back up.
    SHORT: NATR > 90th pctile AND an up trend that is rolling over (price down
           after an up run)  -> exhaustion / trend end.
    """
    natr = tae.natr(df["high"], df["low"], df["close"], n)
    pct = tae.rolling_percentile(natr, pct_n)
    natr_spike = pct > high_thr
    close = df["close"]
    down_day = close < close.shift(1)
    if direction == LONG:
        return natr_spike & down_day
    # uptrend ending: price was rising over trend_n but turns down on the spike.
    uptrend = close.shift(1) > close.shift(1 + trend_n)
    return natr_spike & uptrend & down_day


# ===========================================================================
# S4.05 — Yang-Zhang Volatility (spike reversal)
# ===========================================================================
@signal("S4.05", "Yang-Zhang Volatility Spike Reversal", "S4", BOTH,
        n=20, z_n=60, z_thr=2.0)
def yang_zhang_spike(df, direction, n=20, z_n=60, z_thr=2.0):
    """Yang-Zhang vol spike (> z_thr sigma over z_n days) + day direction.

    LONG : YZ vol spike AND a down day  -> capitulation reversal up.
    SHORT: YZ vol spike AND an up day   -> blow-off top reversal down.
    """
    yz = tae.yang_zhang_vol(df["open"], df["high"], df["low"], df["close"], n)
    z = _ta.zscore(yz, z_n)
    spike = z > z_thr
    close = df["close"]
    down_day = close < close.shift(1)
    up_day = close > close.shift(1)
    if direction == LONG:
        return spike & down_day
    return spike & up_day


# ===========================================================================
# S4.06 — Parkinson Volatility Ratio
# ===========================================================================
@signal("S4.06", "Parkinson Volatility Ratio Extreme", "S4", BOTH,
        n=20, ratio_thr=2.0)
def parkinson_ratio(df, direction, n=20, ratio_thr=2.0):
    """Parkinson(intraday) / close-to-close vol ratio. ratio > thr means
    excessive intraday range relative to realised close-to-close vol.

    LONG : ratio > 2.0 AND a down day  -> oversold bounce.
    SHORT: ratio > 2.0 AND an up day   -> overheated.
    """
    pk = tae.parkinson_vol(df["high"], df["low"], n)
    cc = tae.hist_vol(df["close"], n)
    ratio = pk / cc.replace(0.0, np.nan)
    extreme = ratio > ratio_thr
    close = df["close"]
    down_day = close < close.shift(1)
    up_day = close > close.shift(1)
    if direction == LONG:
        return extreme & down_day
    return extreme & up_day


# ===========================================================================
# S4.07 — BB %B extreme
# ===========================================================================
@signal("S4.07", "Bollinger %B Extreme", "S4", BOTH, n=20, k=2.0)
def bb_percent_b(df, direction, n=20, k=2.0):
    """%B = (close - lower) / (upper - lower).

    LONG : %B < 0  (close below the lower band).
    SHORT: %B > 1  (close above the upper band).
    """
    _, upper, lower, _ = _ta.bollinger(df["close"], n, k)
    rng = (upper - lower).replace(0.0, np.nan)
    pct_b = (df["close"] - lower) / rng
    if direction == LONG:
        return pct_b < 0.0
    return pct_b > 1.0


# ===========================================================================
# S4.08 — Keltner-BB Spread (squeeze release)
# ===========================================================================
@signal("S4.08", "Keltner-BB Spread Squeeze Release", "S4", BOTH,
        bb_n=20, bb_k=2.0, kc_n=20, kc_mult=1.5, mom_n=20)
def keltner_bb_spread(df, direction, bb_n=20, bb_k=2.0, kc_n=20, kc_mult=1.5,
                      mom_n=20):
    """spread = BB width - Keltner width. When BB sits inside Keltner the
    spread is negative (squeeze); a flip to positive = squeeze release.

    LONG : spread flips negative -> positive (release) AND momentum up.
    SHORT: spread flips negative -> positive (release) AND momentum down.
    """
    _, bb_up, bb_low, _ = _ta.bollinger(df["close"], bb_n, bb_k)
    _, kc_up, kc_low = _ta.keltner(df["high"], df["low"], df["close"],
                                   kc_n, kc_mult)
    spread = (bb_up - bb_low) - (kc_up - kc_low)
    # negative -> positive transition (squeeze on -> off).
    release = (spread > 0) & (spread.shift(1) <= 0)
    close = df["close"]
    up = close > close.shift(mom_n)
    down = close < close.shift(mom_n)
    if direction == LONG:
        return release & up
    return release & down


# ===========================================================================
# S4.09 — ATR Trailing Stop reversal
# ===========================================================================
@signal("S4.09", "ATR Trailing Stop Reversal", "S4", BOTH, n=14, mult=3.0)
def atr_trailing_reversal(df, direction, n=14, mult=3.0):
    """ATR trailing stop direction flip (Chande/Kroll). direction = +1 when
    close is above the stop, -1 below.

    LONG : direction flips -1 -> +1 (downtrend stop reverses up).
    SHORT: direction flips +1 -> -1 (uptrend stop reverses down).
    """
    _, dir_ = tae.atr_trailing_stop(df["high"], df["low"], df["close"], n, mult)
    if direction == LONG:
        return (dir_ > 0) & (dir_.shift(1) < 0)
    return (dir_ < 0) & (dir_.shift(1) > 0)


# ===========================================================================
# S4.10 — Chandelier Exit reversal
# ===========================================================================
@signal("S4.10", "Chandelier Exit Reversal", "S4", BOTH, n=22, mult=3.0)
def chandelier_reversal(df, direction, n=22, mult=3.0):
    """22d Chandelier stops: long_stop = HH(22) - 3*ATR, short_stop = LL(22)
    + 3*ATR.

    LONG : close recovers back ABOVE the long stop after having been below it
           (down-break -> recovery).
    SHORT: close breaks DOWN through the long stop after having been above it,
           i.e. loses the trailing support.
    """
    long_stop, short_stop = tae.chandelier_exit(
        df["high"], df["low"], df["close"], n, mult)
    close = df["close"]
    if direction == LONG:
        # was below long stop yesterday, crosses back above today.
        return _ta.crosses_above(close, long_stop)
    # was above long stop, breaks below today (upward chandelier breached).
    return _ta.crosses_below(close, long_stop)


# ===========================================================================
# S4.11 — Volatility Contraction Pattern (VCP)
# ===========================================================================
@signal("S4.11", "Volatility Contraction Pattern (VCP)", "S4", BOTH,
        contractions=3, brk_n=10)
def vcp(df, direction, contractions=3, brk_n=10):
    """Successively shrinking daily ranges (>= `contractions` in a row), then
    a break of the most recent contraction box.

    LONG : range contracted `contractions`+ times in a row AND close breaks
           ABOVE the prior brk_n-day high.
    SHORT: same contraction AND close breaks BELOW the prior brk_n-day low.
    """
    rng = df["high"] - df["low"]
    contracting = rng < rng.shift(1)                 # today's range < prior
    # require `contractions` consecutive contractions ending on the PRIOR bar:
    # the breakout bar itself is an expansion, so the squeeze must precede it.
    run = contracting.astype(int)
    acc = run.rolling(contractions, min_periods=contractions).sum().shift(1)
    contracted = acc >= contractions
    close = df["close"]
    prior_high = df["high"].rolling(brk_n, min_periods=brk_n).max().shift(1)
    prior_low = df["low"].rolling(brk_n, min_periods=brk_n).min().shift(1)
    if direction == LONG:
        return contracted & _ta.crosses_above(close, prior_high)
    return contracted & _ta.crosses_below(close, prior_low)


# ===========================================================================
# S4.12 — Ulcer Index based
# ===========================================================================
@signal("S4.12", "Ulcer Index Extreme", "S4", BOTH,
        n=14, pct_n=60, high_thr=0.90, low_thr=0.10, trend_n=5)
def ulcer_index_sig(df, direction, n=14, pct_n=60, high_thr=0.90,
                    low_thr=0.10, trend_n=5):
    """Ulcer Index = sqrt(mean(drawdown^2)), period 14.

    LONG : UI > 90th pctile (60d)  -> deep-drawdown oversold bounce.
    SHORT: UI was at 10th pctile (calm) after a rise, and UI now starts
           spiking  -> regime change / top forming.
    """
    ui = tae.ulcer_index(df["close"], n)
    pct = tae.rolling_percentile(ui, pct_n)
    close = df["close"]
    if direction == LONG:
        return pct > high_thr
    # was calm (UI low pctile) recently AND price had risen, now UI turns up.
    was_calm = (pct.shift(1) < low_thr)
    ui_rising = ui > ui.shift(1)
    prior_up = close.shift(1) > close.shift(1 + trend_n)
    return was_calm & ui_rising & prior_up
