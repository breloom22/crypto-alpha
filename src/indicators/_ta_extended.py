"""
_ta_extended.py — extended technical-analysis primitives for the v3 strategy
engine (signal groups S2–S6).

Every function here is LOOK-AHEAD SAFE: the value at index t depends only on
data at indices <= t. We rely on pandas rolling / ewm / shift(+k) (all
backward-looking) and on explicit forward iteration for the recursive
indicators (Supertrend, KAMA, McGinley, VIDYA, Heikin-Ashi, Fisher, ...).

pandas_ta is intentionally NOT used (broken on numpy>=2.0). All functions
take/return pandas objects indexed by date and reuse the base primitives in
``_ta`` where possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _ta

# Re-export the base primitives most callers want from one place.
sma = _ta.sma
ema = _ta.ema
rsi = _ta.rsi
roc = _ta.roc
atr = _ta.atr
true_range = _ta.true_range
zscore = _ta.zscore
bollinger = _ta.bollinger
keltner = _ta.keltner

_EPS = 1e-12


# ===========================================================================
# generic helpers
# ===========================================================================
def wma(s: pd.Series, n: int) -> pd.Series:
    """Linearly-weighted moving average (weights 1..n, newest = n)."""
    w = np.arange(1, n + 1, dtype=float)
    wsum = w.sum()
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / wsum, raw=True)


def rolling_percentile(s: pd.Series, n: int) -> pd.Series:
    """Percentile rank (0..1) of the current value within the trailing window
    of length n (fraction of window values <= current value)."""
    return s.rolling(n, min_periods=n).apply(
        lambda x: float((x <= x[-1]).mean()), raw=True)


def rolling_mad(s: pd.Series, n: int) -> pd.Series:
    """Rolling median absolute deviation about the rolling median."""
    return s.rolling(n, min_periods=n).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)


def consecutive_directional(close: pd.Series) -> pd.Series:
    """Signed run length of consecutive up/down closes: +k after k up days,
    -k after k down days, 0 on no change. Look-ahead safe (sequential)."""
    diff = close.diff().to_numpy()
    out = np.zeros(len(diff))
    run = 0
    for i in range(len(diff)):
        d = diff[i]
        if np.isnan(d) or d == 0:
            run = 0
        elif d > 0:
            run = run + 1 if run > 0 else 1
        else:
            run = run - 1 if run < 0 else -1
        out[i] = run
    return pd.Series(out, index=close.index)


def _double_smooth(s: pd.Series, r: int, t: int) -> pd.Series:
    return ema(ema(s, r), t)


# ===========================================================================
# S2 — alternative momentum
# ===========================================================================
def fisher_transform(high: pd.Series, low: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers Fisher Transform of the median price (recursive smoothing)."""
    med = (high + low) / 2.0
    roll_min = med.rolling(period, min_periods=period).min()
    roll_max = med.rolling(period, min_periods=period).max()
    rng = (roll_max - roll_min).replace(0.0, np.nan)
    raw = 2.0 * ((med - roll_min) / rng - 0.5)
    raw = raw.to_numpy()
    n = len(raw)
    val = np.full(n, np.nan)
    fish = np.full(n, np.nan)
    prev_val = 0.0
    prev_fish = 0.0
    for i in range(n):
        if np.isnan(raw[i]):
            continue
        v = 0.33 * raw[i] + 0.67 * prev_val
        v = min(max(v, -0.999), 0.999)
        f = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * prev_fish
        val[i] = v
        fish[i] = f
        prev_val, prev_fish = v, f
    return pd.Series(fish, index=high.index)


def inverse_fisher_rsi(close: pd.Series, rsi_period: int = 14) -> pd.Series:
    """Inverse Fisher Transform of (0.1 * (RSI - 50)). Output in (-1, 1)."""
    r = 0.1 * (rsi(close, rsi_period) - 50.0)
    e = np.exp(2.0 * r)
    return (e - 1.0) / (e + 1.0)


def connors_rsi(close: pd.Series, rsi_period: int = 3,
                streak_period: int = 2, rank_period: int = 100) -> pd.Series:
    """Connors RSI = mean(RSI(price), RSI(streak), PercentRank(1d ROC))."""
    rsi_price = rsi(close, rsi_period)
    streak = consecutive_directional(close)
    rsi_streak = rsi(streak, streak_period)
    ret1 = close.pct_change()
    pctrank = rolling_percentile(ret1, rank_period) * 100.0
    return (rsi_price + rsi_streak + pctrank) / 3.0


def tsi(close: pd.Series, r: int = 25, s: int = 13, signal: int = 7):
    """True Strength Index + its signal line. Returns (tsi, signal)."""
    mom = close.diff()
    num = _double_smooth(mom, r, s)
    den = _double_smooth(mom.abs(), r, s)
    tsi_val = 100.0 * num / den.replace(0.0, np.nan)
    sig = ema(tsi_val, signal)
    return tsi_val, sig


def coppock(close: pd.Series, roc1: int = 14, roc2: int = 11, wma_period: int = 10) -> pd.Series:
    return wma(roc(close, roc1) + roc(close, roc2), wma_period)


def kst(close: pd.Series):
    """Know Sure Thing + signal. Returns (kst, signal)."""
    rcma1 = sma(roc(close, 10), 10)
    rcma2 = sma(roc(close, 15), 10)
    rcma3 = sma(roc(close, 20), 10)
    rcma4 = sma(roc(close, 30), 15)
    kst_val = rcma1 * 1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4
    sig = sma(kst_val, 9)
    return kst_val, sig


def aroon(high: pd.Series, low: pd.Series, period: int = 25):
    """Aroon Up/Down/Oscillator over a (period+1)-bar window. Returns
    (up, down, oscillator)."""
    win = period + 1
    up = high.rolling(win, min_periods=win).apply(
        lambda x: 100.0 * np.argmax(x) / period, raw=True)
    down = low.rolling(win, min_periods=win).apply(
        lambda x: 100.0 * np.argmin(x) / period, raw=True)
    return up, down, up - down


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Vortex Indicator. Returns (plus_vi, minus_vi)."""
    plus_vm = (high - low.shift(1)).abs()
    minus_vm = (low - high.shift(1)).abs()
    tr = true_range(high, low, close)
    tr_sum = tr.rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    plus_vi = plus_vm.rolling(n, min_periods=n).sum() / tr_sum
    minus_vi = minus_vm.rolling(n, min_periods=n).sum() / tr_sum
    return plus_vi, minus_vi


def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13):
    """Elder Ray bull/bear power + the EMA used. Returns (bull, bear, ema)."""
    e = ema(close, period)
    return high - e, low - e, e


def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator (-100..100)."""
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    su = up.rolling(period, min_periods=period).sum()
    sd = down.rolling(period, min_periods=period).sum()
    return 100.0 * (su - sd) / (su + sd).replace(0.0, np.nan)


def dpo(close: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator (look-ahead-safe variant per spec):
    close - SMA(period) shifted back by period//2 + 1."""
    return close - sma(close, period).shift(period // 2 + 1)


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                        p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    prev_close = close.shift(1)
    true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
    bp = close - true_low
    tr = true_range(high, low, close)
    def _avg(n):
        return bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    return 100.0 * (4 * _avg(p1) + 2 * _avg(p2) + _avg(p3)) / 7.0


def stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    """Stochastic RSI in 0..1."""
    r = rsi(close, rsi_period)
    lo = r.rolling(stoch_period, min_periods=stoch_period).min()
    hi = r.rolling(stoch_period, min_periods=stoch_period).max()
    return (r - lo) / (hi - lo).replace(0.0, np.nan)


def rvi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 4):
    """Relative Vigor Index (simplified SMA form) + signal. Returns (rvi, signal)."""
    num = sma(close - open_, period)
    den = sma(high - low, period)
    rvi_val = num / den.replace(0.0, np.nan)
    return rvi_val, sma(rvi_val, period)


def mass_index(high: pd.Series, low: pd.Series, ema_period: int = 9, sum_period: int = 25) -> pd.Series:
    rng = high - low
    e1 = ema(rng, ema_period)
    e2 = ema(e1, ema_period)
    ratio = e1 / e2.replace(0.0, np.nan)
    return ratio.rolling(sum_period, min_periods=sum_period).sum()


def random_walk_index(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Random Walk Index high/low. Returns (rwi_high, rwi_low)."""
    atr_n = atr(high, low, close, n)
    denom = (atr_n * np.sqrt(n)).replace(0.0, np.nan)
    rwi_high = (high - low.shift(n)) / denom
    rwi_low = (high.shift(n) - low) / denom
    return rwi_high, rwi_low


# ===========================================================================
# S3 — alternative trend
# ===========================================================================
def dema(close: pd.Series, n: int) -> pd.Series:
    e1 = ema(close, n)
    return 2.0 * e1 - ema(e1, n)


def tema(close: pd.Series, n: int) -> pd.Series:
    e1 = ema(close, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3.0 * e1 - 3.0 * e2 + e3


def hull_ma(close: pd.Series, n: int = 20) -> pd.Series:
    half = max(1, n // 2)
    sqn = max(1, int(round(np.sqrt(n))))
    return wma(2.0 * wma(close, half) - wma(close, n), sqn)


def kama(close: pd.Series, er_period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average (recursive)."""
    c = close.to_numpy(dtype=float)
    n = len(c)
    change = np.abs(c - np.concatenate([np.full(er_period, np.nan), c[:-er_period]]))
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(er_period).sum().to_numpy()
    er = np.divide(change, vol, out=np.zeros_like(c), where=vol > 0)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.full(n, np.nan)
    # seed at first index where ER is defined
    start = er_period
    if start >= n:
        return pd.Series(out, index=close.index)
    out[start] = c[start]
    for i in range(start + 1, n):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def mcginley(close: pd.Series, n: int = 14) -> pd.Series:
    """McGinley Dynamic (recursive)."""
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) == 0:
        return pd.Series(out, index=close.index)
    out[0] = c[0]
    for i in range(1, len(c)):
        prev = out[i - 1]
        if prev <= 0 or np.isnan(prev):
            out[i] = c[i]
            continue
        ratio = c[i] / prev
        out[i] = prev + (c[i] - prev) / (n * (ratio ** 4))
    return pd.Series(out, index=close.index)


def vidya(close: pd.Series, cmo_period: int = 9, n: int = 12) -> pd.Series:
    """Variable Index Dynamic Average (CMO-driven adaptive EMA, recursive)."""
    alpha = 2.0 / (n + 1)
    k = (cmo(close, cmo_period).abs() / 100.0).to_numpy()
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    prev = None
    for i in range(len(c)):
        if np.isnan(k[i]):
            continue
        if prev is None:
            prev = c[i]
        a = alpha * k[i]
        prev = a * c[i] + (1 - a) * prev
        out[i] = prev
    return pd.Series(out, index=close.index)


def linreg_slope(close: pd.Series, n: int = 20) -> pd.Series:
    """Rolling linear-regression slope (per-bar price units)."""
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    def _slope(y):
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)
    return close.rolling(n, min_periods=n).apply(_slope, raw=True)


def linreg_value(close: pd.Series, n: int = 20) -> pd.Series:
    """Endpoint value of the rolling linear regression (forecast at current bar)."""
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    def _end(y):
        slope = ((x - x_mean) * (y - y.mean())).sum() / denom
        intercept = y.mean() - slope * x_mean
        return float(intercept + slope * (n - 1))
    return close.rolling(n, min_periods=n).apply(_end, raw=True)


def linreg_channel(close: pd.Series, n: int = 20, k: float = 2.0):
    """Linear-regression mid line + k-sigma channel. Returns (mid, upper, lower)."""
    mid = linreg_value(close, n)
    resid_std = (close - mid).rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * resid_std, mid - k * resid_std


def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series):
    """Heikin-Ashi OHLC (recursive open). Returns (ha_open, ha_high, ha_low, ha_close)."""
    o = open_.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.full(n, np.nan)
    if n:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    idx = close.index
    return (pd.Series(ha_open, index=idx), pd.Series(ha_high, index=idx),
            pd.Series(ha_low, index=idx), pd.Series(ha_close, index=idx))


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, mult: float = 3.0):
    """Supertrend line + direction (+1 up / -1 down), recursive. Returns
    (supertrend, direction)."""
    hl2 = (high + low) / 2.0
    atr_n = atr(high, low, close, period)
    upper = (hl2 + mult * atr_n).to_numpy()
    lower = (hl2 - mult * atr_n).to_numpy()
    c = close.to_numpy(dtype=float)
    n = len(c)
    st = np.full(n, np.nan)
    direction = np.ones(n)  # +1 uptrend, -1 downtrend
    fu = np.full(n, np.nan)
    fl = np.full(n, np.nan)
    started = False
    for i in range(n):
        if np.isnan(upper[i]):
            continue
        if not started:
            fu[i] = upper[i]
            fl[i] = lower[i]
            direction[i] = 1 if c[i] >= lower[i] else -1
            st[i] = fl[i] if direction[i] == 1 else fu[i]
            started = True
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if c[i] > fu[i - 1]:
            direction[i] = 1
        elif c[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        st[i] = fl[i] if direction[i] == 1 else fu[i]
    idx = close.index
    return pd.Series(st, index=idx), pd.Series(direction, index=idx)


def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series):
    """Classic floor pivots from the PREVIOUS bar. Returns (pp, r1, s1)."""
    ph, pl, pc = high.shift(1), low.shift(1), close.shift(1)
    pp = (ph + pl + pc) / 3.0
    r1 = 2.0 * pp - pl
    s1 = 2.0 * pp - ph
    return pp, r1, s1


# ===========================================================================
# S4 — volatility / range
# ===========================================================================
def ttm_squeeze(high: pd.Series, low: pd.Series, close: pd.Series,
                bb_n: int = 20, bb_k: float = 2.0, kc_n: int = 20, kc_mult: float = 1.5):
    """TTM squeeze. Returns (squeeze_on: bool Series, momentum: Series).
    squeeze_on = Bollinger Bands inside Keltner Channels."""
    mid, upper, lower, _ = bollinger(close, bb_n, bb_k)
    kmid, kup, klow = keltner(high, low, close, kc_n, kc_mult)
    squeeze_on = (lower > klow) & (upper < kup)
    hh = high.rolling(bb_n, min_periods=bb_n).max()
    ll = low.rolling(bb_n, min_periods=bb_n).min()
    base = (hh + ll) / 2.0
    sma_close = sma(close, bb_n)
    detr = close - (base + sma_close) / 2.0
    momentum = linreg_value(detr, bb_n)
    return squeeze_on.fillna(False), momentum


def chaikin_volatility(high: pd.Series, low: pd.Series, n: int = 10) -> pd.Series:
    ema_hl = ema(high - low, n)
    return ema_hl.pct_change(n) * 100.0


def hist_vol(close: pd.Series, n: int = 20) -> pd.Series:
    """Rolling realised volatility (std of daily returns)."""
    return close.pct_change().rolling(n, min_periods=n).std(ddof=0)


def natr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return atr(high, low, close, n) / close.replace(0.0, np.nan) * 100.0


def yang_zhang_vol(open_: pd.Series, high: pd.Series, low: pd.Series,
                   close: pd.Series, n: int = 20) -> pd.Series:
    """Yang-Zhang volatility estimator (per-period, not annualised)."""
    o = np.log(open_ / close.shift(1))
    c = np.log(close / open_)
    rs = (np.log(high / close) * np.log(high / open_) +
          np.log(low / close) * np.log(low / open_))
    sigma_o = o.rolling(n, min_periods=n).var(ddof=1)
    sigma_c = c.rolling(n, min_periods=n).var(ddof=1)
    rs_mean = rs.rolling(n, min_periods=n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt((sigma_o + k * sigma_c + (1 - k) * rs_mean).clip(lower=0.0))


def parkinson_vol(high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
    factor = 1.0 / (4.0 * np.log(2.0))
    hl2 = np.log(high / low) ** 2
    return np.sqrt(factor * hl2.rolling(n, min_periods=n).mean())


def ulcer_index(close: pd.Series, n: int = 14) -> pd.Series:
    roll_max = close.rolling(n, min_periods=n).max()
    dd = 100.0 * (close - roll_max) / roll_max.replace(0.0, np.nan)
    return np.sqrt((dd ** 2).rolling(n, min_periods=n).mean())


def atr_trailing_stop(high: pd.Series, low: pd.Series, close: pd.Series,
                      n: int = 14, mult: float = 3.0):
    """ATR trailing stop (Chande/Kroll style), recursive. Returns
    (stop, direction) where direction = +1 if close above stop else -1."""
    loss = (mult * atr(high, low, close, n)).to_numpy()
    c = close.to_numpy(dtype=float)
    n_obs = len(c)
    stop = np.full(n_obs, np.nan)
    direction = np.ones(n_obs)
    prev_stop = None
    for i in range(n_obs):
        if np.isnan(loss[i]):
            continue
        if prev_stop is None:
            prev_stop = c[i] - loss[i]
            stop[i] = prev_stop
            direction[i] = 1
            continue
        prev_close = c[i - 1]
        if c[i] > prev_stop and prev_close > prev_stop:
            cur = max(prev_stop, c[i] - loss[i])
        elif c[i] < prev_stop and prev_close < prev_stop:
            cur = min(prev_stop, c[i] + loss[i])
        elif c[i] > prev_stop:
            cur = c[i] - loss[i]
        else:
            cur = c[i] + loss[i]
        stop[i] = cur
        direction[i] = 1 if c[i] > cur else -1
        prev_stop = cur
    idx = close.index
    return pd.Series(stop, index=idx), pd.Series(direction, index=idx)


def chandelier_exit(high: pd.Series, low: pd.Series, close: pd.Series,
                    n: int = 22, mult: float = 3.0):
    """Chandelier exit long/short stops. Returns (long_stop, short_stop)."""
    atr_n = atr(high, low, close, n)
    long_stop = high.rolling(n, min_periods=n).max() - mult * atr_n
    short_stop = low.rolling(n, min_periods=n).min() + mult * atr_n
    return long_stop, short_stop


# ===========================================================================
# S5 — alternative volume
# ===========================================================================
def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    up = tp > tp.shift(1)
    pos = rmf.where(up, 0.0)
    neg = rmf.where(~up & (tp < tp.shift(1)), 0.0)
    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + ratio)


def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, n: int = 14):
    """Ease of Movement + its SMA. Returns (emv, sma_emv)."""
    dm = ((high + low) / 2.0) - ((high.shift(1) + low.shift(1)) / 2.0)
    rng = (high - low).replace(0.0, np.nan)
    box = volume / rng
    emv = dm / box.replace(0.0, np.nan)
    return emv, sma(emv, n)


def klinger(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
            fast: int = 34, slow: int = 55, signal: int = 13):
    """Klinger Volume Oscillator + signal. Returns (kvo, signal)."""
    hlc = (high + low + close)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    vol = volume.to_numpy(dtype=float)
    hlc_v = hlc.to_numpy(dtype=float)
    n = len(h)
    vf = np.zeros(n)
    trend = 1.0
    cm = 0.0
    prev_dm = h[0] - l[0] if n else 0.0
    for i in range(n):
        dm = h[i] - l[i]
        if i == 0:
            vf[i] = 0.0
            cm = dm
            prev_dm = dm
            continue
        new_trend = 1.0 if hlc_v[i] > hlc_v[i - 1] else -1.0
        if new_trend == trend:
            cm = cm + dm
        else:
            cm = prev_dm + dm
        trend = new_trend
        if cm > 0:
            vf[i] = vol[i] * abs(2.0 * (dm / cm) - 1.0) * trend * 100.0
        else:
            vf[i] = 0.0
        prev_dm = dm
    vf_s = pd.Series(vf, index=close.index)
    kvo = ema(vf_s, fast) - ema(vf_s, slow)
    return kvo, ema(kvo, signal)


def nvi(close: pd.Series, volume: pd.Series, signal: int = 255):
    """Negative Volume Index + EMA signal. Returns (nvi, signal)."""
    ret = close.pct_change().fillna(0.0)
    mask = volume < volume.shift(1)
    masked = ret.where(mask, 0.0)
    series = 1000.0 * (1.0 + masked).cumprod()
    return series, ema(series, signal)


def pvi(close: pd.Series, volume: pd.Series, signal: int = 255):
    """Positive Volume Index + EMA signal. Returns (pvi, signal)."""
    ret = close.pct_change().fillna(0.0)
    mask = volume > volume.shift(1)
    masked = ret.where(mask, 0.0)
    series = 1000.0 * (1.0 + masked).cumprod()
    return series, ema(series, signal)


def volume_oscillator(volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    ef = ema(volume, fast)
    es = ema(volume, slow)
    return (ef - es) / es.replace(0.0, np.nan) * 100.0


def vwap_rolling(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    pv = (tp * volume).rolling(n, min_periods=n).sum()
    v = volume.rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    return pv / v


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    rng = (high - low).replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    return (clv.fillna(0.0) * volume).cumsum()


def ad_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, fast: int = 3, slow: int = 10) -> pd.Series:
    ad = ad_line(high, low, close, volume)
    return ema(ad, fast) - ema(ad, slow)


def rvol(volume: pd.Series, n: int = 20) -> pd.Series:
    """Relative volume = volume / SMA(volume, n)."""
    return volume / sma(volume, n).replace(0.0, np.nan)


# ===========================================================================
# S6 — statistical
# ===========================================================================
def hurst_exponent(close: pd.Series, n: int = 60, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst exponent via the log-log slope of dispersion vs lag."""
    lags = np.arange(2, max_lag)
    log_lags = np.log(lags)
    def _h(x):
        tau = []
        for lag in lags:
            d = x[lag:] - x[:-lag]
            s = np.std(d)
            tau.append(s if s > 0 else _EPS)
        poly = np.polyfit(log_lags, np.log(tau), 1)
        return float(poly[0])
    return close.rolling(n, min_periods=n).apply(_h, raw=True)


def rolling_autocorr(returns: pd.Series, n: int = 20, lag: int = 1) -> pd.Series:
    def _ac(x):
        a = x[lag:]
        b = x[:-lag]
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    return returns.rolling(n, min_periods=n).apply(_ac, raw=True)


def variance_ratio(close: pd.Series, k: int = 5, n: int = 60) -> pd.Series:
    """VR = var(k-day returns) / (k * var(1-day returns)) over a rolling window."""
    r1 = close.pct_change()
    rk = close.pct_change(k)
    var1 = r1.rolling(n, min_periods=n).var(ddof=0)
    vark = rk.rolling(n, min_periods=n).var(ddof=0)
    return vark / (k * var1.replace(0.0, np.nan))


def return_entropy(returns: pd.Series, n: int = 30, bins: int = 10) -> pd.Series:
    """Shannon entropy of the binned return distribution over a rolling window."""
    def _ent(x):
        hist, _ = np.histogram(x, bins=bins)
        p = hist / hist.sum() if hist.sum() > 0 else hist
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())
    return returns.rolling(n, min_periods=n).apply(_ent, raw=True)


def clv(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Close Location Value in [-1, 1]."""
    rng = (high - low).replace(0.0, np.nan)
    return ((2.0 * close - high - low) / rng)


def distance_from_ma(close: pd.Series, n: int = 50) -> pd.Series:
    """Percent distance of close from its SMA(n)."""
    m = sma(close, n)
    return (close - m) / m.replace(0.0, np.nan) * 100.0


def median_reversion(close: pd.Series, n: int = 50):
    """(close - rolling median) / rolling MAD. Robust z-score-like measure."""
    med = close.rolling(n, min_periods=n).median()
    mad = rolling_mad(close, n).replace(0.0, np.nan)
    return (close - med) / mad
