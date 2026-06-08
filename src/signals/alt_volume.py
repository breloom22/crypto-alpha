"""
alt_volume.py — Group S5: Alternative volume indicators (10 signals).

Per-asset signals. A True at day t means "enter at t+1 open"; every value at t
depends only on data at indices <= t (look-ahead safe). Foundation indicators in
`indicators._ta_extended` are reused; "크로스" wording maps to
`_ta.crosses_above` / `_ta.crosses_below`.
"""
from __future__ import annotations

from indicators import _ta
from indicators import _ta_extended as tae
from .registry import signal, LONG, SHORT, BOTH
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# S5.01 — MFI (Money Flow Index): volume-weighted RSI.
# LONG: MFI < 20 (oversold).  SHORT: MFI > 80 (overbought).
# ---------------------------------------------------------------------------
@signal("S5.01", "MFI (Money Flow Index)", "S5", BOTH, n=14)
def mfi(df, direction, n=14):
    m = tae.mfi(df["high"], df["low"], df["close"], df["volume"], n)
    if direction == LONG:
        return m < 20
    return m > 80


# ---------------------------------------------------------------------------
# S5.02 — Ease of Movement: SMA(EMV, 14) sign flip.
# LONG: EMV SMA flips negative -> positive.  SHORT: positive -> negative.
# ---------------------------------------------------------------------------
@signal("S5.02", "Ease of Movement", "S5", BOTH, n=14)
def ease_of_movement(df, direction, n=14):
    _, emv_sma = tae.ease_of_movement(df["high"], df["low"], df["volume"], n)
    prev = emv_sma.shift(1)
    if direction == LONG:
        return (emv_sma > 0) & (prev <= 0)
    return (emv_sma < 0) & (prev >= 0)


# ---------------------------------------------------------------------------
# S5.03 — Klinger Volume Oscillator: KVO vs its signal line.
# LONG: KVO crosses up signal while in negative zone (KVO < 0).
# SHORT: KVO crosses down signal while in positive zone (KVO > 0).
# ---------------------------------------------------------------------------
@signal("S5.03", "Klinger Volume Oscillator", "S5", BOTH,
        fast=34, slow=55, signal=13)
def klinger(df, direction, fast=34, slow=55, signal=13):
    kvo, sig = tae.klinger(df["high"], df["low"], df["close"], df["volume"],
                           fast, slow, signal)
    if direction == LONG:
        return _ta.crosses_above(kvo, sig) & (kvo < 0)
    return _ta.crosses_below(kvo, sig) & (kvo > 0)


# ---------------------------------------------------------------------------
# S5.04 — Negative Volume Index: NVI vs EMA(NVI, 255) signal.
# LONG: NVI > signal (smart-money accumulation).  SHORT: NVI < signal.
# ---------------------------------------------------------------------------
@signal("S5.04", "Negative Volume Index", "S5", BOTH, signal=255)
def nvi(df, direction, signal=255):
    series, sig = tae.nvi(df["close"], df["volume"], signal)
    if direction == LONG:
        return series > sig
    return series < sig


# ---------------------------------------------------------------------------
# S5.05 — Positive Volume Index: contrarian cross of EMA(PVI, 255) signal.
# LONG: PVI crosses DOWN signal (crowd exits -> fade).
# SHORT: PVI crosses UP signal (crowd chases).
# (Asymmetric/inverted relative to a naive trend reading — read carefully.)
# ---------------------------------------------------------------------------
@signal("S5.05", "Positive Volume Index", "S5", BOTH, signal=255)
def pvi(df, direction, signal=255):
    series, sig = tae.pvi(df["close"], df["volume"], signal)
    if direction == LONG:
        return _ta.crosses_below(series, sig)
    return _ta.crosses_above(series, sig)


# ---------------------------------------------------------------------------
# S5.06 — Volume Oscillator: VO = (EMA(V,5)-EMA(V,20))/EMA(V,20)*100.
# LONG: VO < -30% AND price falling (selling exhaustion).
# SHORT: VO > 50% AND price rising (climax).
# ---------------------------------------------------------------------------
@signal("S5.06", "Volume Oscillator", "S5", BOTH, fast=5, slow=20)
def volume_oscillator(df, direction, fast=5, slow=20):
    vo = tae.volume_oscillator(df["volume"], fast, slow)
    chg = df["close"] - df["close"].shift(1)
    if direction == LONG:
        return (vo < -30) & (chg < 0)
    return (vo > 50) & (chg > 0)


# ---------------------------------------------------------------------------
# S5.07 — VWAP 괴리율 (deviation): zscore of (close - VWAP) / VWAP over 20d.
# LONG: deviation < -2sigma.  SHORT: deviation > +2sigma.
# ---------------------------------------------------------------------------
@signal("S5.07", "VWAP Deviation", "S5", BOTH, n=20, z_n=20, k=2.0)
def vwap_deviation(df, direction, n=20, z_n=20, k=2.0):
    vwap = tae.vwap_rolling(df["high"], df["low"], df["close"], df["volume"], n)
    dev = (df["close"] - vwap) / vwap.replace(0.0, np.nan)
    z = _ta.zscore(dev, z_n)
    if direction == LONG:
        return z < -k
    return z > k


# ---------------------------------------------------------------------------
# S5.08 — A/D Oscillator: Osc = EMA(AD,3) - EMA(AD,10), with price divergence.
# LONG: Osc flips neg -> pos WHILE price is falling (bullish divergence).
# SHORT: Osc flips pos -> neg WHILE price is rising (bearish divergence).
# ---------------------------------------------------------------------------
@signal("S5.08", "A/D Oscillator", "S5", BOTH, fast=3, slow=10)
def ad_oscillator(df, direction, fast=3, slow=10):
    osc = tae.ad_oscillator(df["high"], df["low"], df["close"], df["volume"],
                            fast, slow)
    prev = osc.shift(1)
    chg = df["close"] - df["close"].shift(1)
    if direction == LONG:
        return (osc > 0) & (prev <= 0) & (chg < 0)
    return (osc < 0) & (prev >= 0) & (chg > 0)


# ---------------------------------------------------------------------------
# S5.09 — Volume Price Confirmation: N-day price direction vs volume trend.
# LONG: price down + volume down (N=5) -> selling exhaustion.
# SHORT: price up + volume down -> buying exhaustion.
# ---------------------------------------------------------------------------
@signal("S5.09", "Volume Price Confirmation", "S5", BOTH, n=5)
def volume_price_confirmation(df, direction, n=5):
    price_chg = df["close"] - df["close"].shift(n)
    vol_chg = df["volume"] - df["volume"].shift(n)
    if direction == LONG:
        return (price_chg < 0) & (vol_chg < 0)
    return (price_chg > 0) & (vol_chg < 0)


# ---------------------------------------------------------------------------
# S5.10 — Relative Volume (RVOL) = volume / SMA(volume, 20).
# LONG: RVOL > 3 AND bullish candle (close > open) — heavy buying.
# SHORT: RVOL > 3 AND bearish candle (close < open) — heavy selling.
# ---------------------------------------------------------------------------
@signal("S5.10", "Relative Volume (RVOL)", "S5", BOTH, n=20, mult=3.0)
def relative_volume(df, direction, n=20, mult=3.0):
    rv = tae.rvol(df["volume"], n)
    if direction == LONG:
        return (rv > mult) & (df["close"] > df["open"])
    return (rv > mult) & (df["close"] < df["open"])
