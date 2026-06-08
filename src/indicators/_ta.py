"""
_ta.py — vectorized technical-analysis primitives.

Every function here is LOOK-AHEAD SAFE: the value at index t depends only on
data at indices <= t. We rely exclusively on pandas rolling / ewm / shift(+k),
all of which look backward. (pandas_ta is intentionally NOT used — it is broken
on numpy>=2.0, which imports the removed numpy.NaN.)

All functions take/return pandas objects indexed by date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- moving averages -------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA) — equivalent to ewm(alpha=1/n)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


# ---- momentum --------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, n)
    avg_loss = _wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[avg_loss == 0.0] = 100.0          # no losses -> RSI 100
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(high, low, close, k: int = 14, d: int = 3):
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    rng = (hh - ll).replace(0.0, np.nan)
    pct_k = 100.0 * (close - ll) / rng
    pct_d = pct_k.rolling(d, min_periods=d).mean()
    return pct_k, pct_d


def cci(high, low, close, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(n, min_periods=n).mean()
    mad = (tp - sma_tp).abs().rolling(n, min_periods=n).mean()
    return (tp - sma_tp) / (0.015 * mad.replace(0.0, np.nan))


def williams_r(high, low, close, n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    rng = (hh - ll).replace(0.0, np.nan)
    return -100.0 * (hh - close) / rng


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    return close.pct_change(n) * 100.0


# ---- volatility / range ----------------------------------------------------
def true_range(high, low, close) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def atr(high, low, close, n: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return mid, upper, lower, width


def keltner(high, low, close, n: int = 20, atr_mult: float = 2.0):
    mid = ema(close, n)
    rng = atr(high, low, close, n)
    return mid, mid + atr_mult * rng, mid - atr_mult * rng


def garman_klass_vol(open_, high, low, close, n: int = 20) -> pd.Series:
    """Rolling Garman-Klass volatility (per-day variance proxy, smoothed)."""
    hl = np.log(high / low) ** 2
    co = np.log(close / open_) ** 2
    gk = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    # rolling mean of the per-day GK variance, annualisation not needed for signal
    return np.sqrt(gk.clip(lower=0.0).rolling(n, min_periods=n).mean())


# ---- trend / directional ---------------------------------------------------
def adx(high, low, close, n: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = true_range(high, low, close)
    atr_n = _wilder(tr, n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr_n.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder(minus_dm, n) / atr_n.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_val = _wilder(dx, n)
    return plus_di, minus_di, adx_val


def _psar_core(high, low, af_step: float, af_max: float):
    """Iterative Parabolic SAR. Returns (sar, uptrend, bear_flip) arrays.

    bear_flip[i] is True on the bar where the engine transitions uptrend ->
    downtrend (a low pierces the SAR). Sequential => look-ahead safe.
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)
    sar = np.full(n, np.nan)
    up_arr = np.ones(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    if n < 2:
        return sar, up_arr, bear
    uptrend = True
    af = af_step
    ep = h[0]
    sar[0] = l[0]
    for i in range(1, n):
        prev_sar = sar[i - 1]
        cur = prev_sar + af * (ep - prev_sar)
        if uptrend:
            cur = min(cur, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < cur:                      # flip to downtrend
                uptrend = False
                bear[i] = True
                cur = ep                        # SAR jumps to prior EP
                ep = l[i]
                af = af_step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            cur = max(cur, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > cur:                      # flip to uptrend
                uptrend = True
                cur = ep
                ep = h[i]
                af = af_step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)
        sar[i] = cur
        up_arr[i] = uptrend
    return sar, up_arr, bear


def parabolic_sar(high, low, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Parabolic SAR value series."""
    sar, _, _ = _psar_core(high, low, af_step, af_max)
    return pd.Series(sar, index=high.index)


def parabolic_sar_bear_flip(high, low, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Bearish-flip signal taken from the SAR engine's own trend transition
    (faithful to 'SAR flips above price'), not a close-vs-SAR proxy."""
    _, _, bear = _psar_core(high, low, af_step, af_max)
    return pd.Series(bear, index=high.index)


def ichimoku(high, low, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
    """Returns (tenkan, kijun, senkou_a, senkou_b).

    Senkou spans are shifted FORWARD by `kijun` (the standard +26 displacement),
    so the cloud value at date t derives from data at t-26 — known information,
    hence look-ahead SAFE. (Chikou span is deliberately omitted: it would be
    look-ahead.)
    """
    conv = (high.rolling(tenkan, min_periods=tenkan).max() +
            low.rolling(tenkan, min_periods=tenkan).min()) / 2.0
    base = (high.rolling(kijun, min_periods=kijun).max() +
            low.rolling(kijun, min_periods=kijun).min()) / 2.0
    span_a = ((conv + base) / 2.0).shift(kijun)
    span_b = ((high.rolling(senkou, min_periods=senkou).max() +
               low.rolling(senkou, min_periods=senkou).min()) / 2.0).shift(kijun)
    return conv, base, span_a, span_b


# ---- volume ----------------------------------------------------------------
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def cmf(high, low, close, volume, n: int = 20) -> pd.Series:
    rng = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)
    return mfv.rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum().replace(0.0, np.nan)


def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (volume * close.pct_change().fillna(0.0)).cumsum()


def force_index(close: pd.Series, volume: pd.Series, n: int = 13) -> pd.Series:
    raw = close.diff() * volume
    return ema(raw, n)


# ---- helpers for divergence / pivot logic ----------------------------------
def rolling_argmax_is_now(s: pd.Series, n: int) -> pd.Series:
    """True at t if s[t] is the max of the trailing window of length n."""
    return s == s.rolling(n, min_periods=n).max()


def zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - mean) / sd.replace(0.0, np.nan)


def crosses_below(a: pd.Series, b, ) -> pd.Series:
    """True at t where a crosses from >= b to < b (a[t-1] >= b[t-1], a[t] < b[t])."""
    if np.isscalar(b):
        prev_ok = a.shift(1) >= b
        now = a < b
    else:
        prev_ok = a.shift(1) >= b.shift(1)
        now = a < b
    return (prev_ok & now).fillna(False)


def crosses_above(a: pd.Series, b) -> pd.Series:
    if np.isscalar(b):
        prev_ok = a.shift(1) <= b
        now = a > b
    else:
        prev_ok = a.shift(1) <= b.shift(1)
        now = a > b
    return (prev_ok & now).fillna(False)
