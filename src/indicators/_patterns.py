"""
_patterns.py — candlestick / price-action pattern primitives for signal group S1.

All primitives are LOOK-AHEAD SAFE: a pattern flagged at index t uses only the
bar at t and earlier bars (via .shift(+k)). They return boolean Series aligned
to df.index (NaN-safe, filled False) or float component Series.

Conventions: df has columns open, high, low, close, volume. "bull" = close>open.
The price_action signal module composes these with direction + optional trend
context; keeping the geometry here makes each pattern individually testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _ta


# ---------------------------------------------------------------------------
# candle anatomy
# ---------------------------------------------------------------------------
def body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def candle_range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"])


def upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def is_bull(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]


def is_bear(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]


def close_location(df: pd.DataFrame) -> pd.Series:
    """Where the close sits within the day's range, 0 (low) .. 1 (high)."""
    rng = candle_range(df).replace(0.0, np.nan)
    return (df["close"] - df["low"]) / rng


def roc_pct(close: pd.Series, n: int = 5) -> pd.Series:
    """n-day rate of change in percent (look-ahead safe)."""
    return close.pct_change(n) * 100.0


def uptrend(close: pd.Series, n: int = 5, thresh: float = 3.0) -> pd.Series:
    return roc_pct(close, n) > thresh


def downtrend(close: pd.Series, n: int = 5, thresh: float = 3.0) -> pd.Series:
    return roc_pct(close, n) < -thresh


def _clean(s: pd.Series, index) -> pd.Series:
    return s.reindex(index).fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# S1.01 — Engulfing
# ---------------------------------------------------------------------------
def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bear = prev_c < prev_o
    today_bull = c > o
    engulf = (o < prev_c) & (c > prev_o)
    return _clean(prev_bear & today_bull & engulf, df.index)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bull = prev_c > prev_o
    today_bear = c < o
    engulf = (o > prev_c) & (c < prev_o)
    return _clean(prev_bull & today_bear & engulf, df.index)


# ---------------------------------------------------------------------------
# S1.02 — Hammer / Shooting Star (geometry only; trend added by caller)
# ---------------------------------------------------------------------------
def hammer_shape(df: pd.DataFrame, body_mult: float = 2.0, opp_mult: float = 0.3) -> pd.Series:
    b = body(df).replace(0.0, np.nan)
    lw, uw = lower_wick(df), upper_wick(df)
    return _clean((lw >= body_mult * b) & (uw < opp_mult * b), df.index)


def shooting_star_shape(df: pd.DataFrame, body_mult: float = 2.0, opp_mult: float = 0.3) -> pd.Series:
    b = body(df).replace(0.0, np.nan)
    lw, uw = lower_wick(df), upper_wick(df)
    return _clean((uw >= body_mult * b) & (lw < opp_mult * b), df.index)


# ---------------------------------------------------------------------------
# S1.03 — Doji
# ---------------------------------------------------------------------------
def doji(df: pd.DataFrame, thresh: float = 0.1) -> pd.Series:
    rng = candle_range(df).replace(0.0, np.nan)
    return _clean((df["close"] - df["open"]).abs() / rng < thresh, df.index)


# ---------------------------------------------------------------------------
# S1.04 / S1.05 — Inside / Outside bar
# ---------------------------------------------------------------------------
def inside_bar(df: pd.DataFrame) -> pd.Series:
    h, l = df["high"], df["low"]
    return _clean((h < h.shift(1)) & (l > l.shift(1)), df.index)


def outside_bar(df: pd.DataFrame) -> pd.Series:
    h, l = df["high"], df["low"]
    return _clean((h > h.shift(1)) & (l < l.shift(1)), df.index)


# ---------------------------------------------------------------------------
# S1.06 — Three White Soldiers / Black Crows
# ---------------------------------------------------------------------------
def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    bull = c > o
    cond = bull & bull.shift(1) & bull.shift(2)
    rising = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    # each open within the previous candle's real body (no exhaustion gap)
    open_in_body = (o <= c.shift(1)) & (o >= o.shift(1))
    open_in_body_prev = (o.shift(1) <= c.shift(2)) & (o.shift(1) >= o.shift(2))
    return _clean(cond & rising & open_in_body & open_in_body_prev, df.index)


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    bear = c < o
    cond = bear & bear.shift(1) & bear.shift(2)
    falling = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    open_in_body = (o >= c.shift(1)) & (o <= o.shift(1))
    open_in_body_prev = (o.shift(1) >= c.shift(2)) & (o.shift(1) <= o.shift(2))
    return _clean(cond & falling & open_in_body & open_in_body_prev, df.index)


# ---------------------------------------------------------------------------
# S1.07 — Morning / Evening Star (3-bar)
# ---------------------------------------------------------------------------
def morning_star(df: pd.DataFrame, small_body_frac: float = 0.5) -> pd.Series:
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    b = (c - o).abs()
    avg_b = b.rolling(10, min_periods=3).mean()
    bar1_bear = (c.shift(2) < o.shift(2)) & (b.shift(2) > avg_b.shift(2))
    bar2_small = b.shift(1) < small_body_frac * b.shift(2)
    bar3_bull = c > o
    mid1 = (o.shift(2) + c.shift(2)) / 2.0
    recover = c > mid1
    return _clean(bar1_bear & bar2_small & bar3_bull & recover, df.index)


def evening_star(df: pd.DataFrame, small_body_frac: float = 0.5) -> pd.Series:
    o, c = df["open"], df["close"]
    b = (c - o).abs()
    avg_b = b.rolling(10, min_periods=3).mean()
    bar1_bull = (c.shift(2) > o.shift(2)) & (b.shift(2) > avg_b.shift(2))
    bar2_small = b.shift(1) < small_body_frac * b.shift(2)
    bar3_bear = c < o
    mid1 = (o.shift(2) + c.shift(2)) / 2.0
    fail = c < mid1
    return _clean(bar1_bull & bar2_small & bar3_bear & fail, df.index)


# ---------------------------------------------------------------------------
# S1.08 — Pin Bar
# ---------------------------------------------------------------------------
def pin_bar_bullish(df: pd.DataFrame, tail_frac: float = 0.66, body_zone: float = 0.25) -> pd.Series:
    rng = candle_range(df).replace(0.0, np.nan)
    lw = lower_wick(df)
    body_lo = df[["open", "close"]].min(axis=1)
    long_tail = lw / rng >= tail_frac
    body_top = (body_lo - df["low"]) / rng >= (1.0 - body_zone)
    return _clean(long_tail & body_top, df.index)


def pin_bar_bearish(df: pd.DataFrame, tail_frac: float = 0.66, body_zone: float = 0.25) -> pd.Series:
    rng = candle_range(df).replace(0.0, np.nan)
    uw = upper_wick(df)
    body_hi = df[["open", "close"]].max(axis=1)
    long_tail = uw / rng >= tail_frac
    body_bot = (df["high"] - body_hi) / rng >= (1.0 - body_zone)
    return _clean(long_tail & body_bot, df.index)


# ---------------------------------------------------------------------------
# S1.09 — Gaps
# ---------------------------------------------------------------------------
def gap_up(df: pd.DataFrame) -> pd.Series:
    return _clean(df["open"] > df["high"].shift(1), df.index)


def gap_down(df: pd.DataFrame) -> pd.Series:
    return _clean(df["open"] < df["low"].shift(1), df.index)


# ---------------------------------------------------------------------------
# S1.10 — Donchian / N-day breakout
# ---------------------------------------------------------------------------
def donchian_high_break(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Close exceeds the highest high of the prior n bars (excluding today)."""
    prior_high = df["high"].rolling(n, min_periods=n).max().shift(1)
    return _clean(df["close"] > prior_high, df.index)


def donchian_low_break(df: pd.DataFrame, n: int = 20) -> pd.Series:
    prior_low = df["low"].rolling(n, min_periods=n).min().shift(1)
    return _clean(df["close"] < prior_low, df.index)


# ---------------------------------------------------------------------------
# S1.11 — Narrow Range 7 (NR7)
# ---------------------------------------------------------------------------
def nr7(df: pd.DataFrame, n: int = 7) -> pd.Series:
    """Today's range is the smallest of the last n bars."""
    rng = candle_range(df)
    return _clean(rng == rng.rolling(n, min_periods=n).min(), df.index)


# ---------------------------------------------------------------------------
# S1.14 — Range contraction
# ---------------------------------------------------------------------------
def range_contraction(df: pd.DataFrame, short_n: int = 3, long_n: int = 20,
                      frac: float = 0.5) -> pd.Series:
    rng = candle_range(df)
    short_avg = rng.rolling(short_n, min_periods=short_n).mean()
    long_avg = rng.rolling(long_n, min_periods=long_n).mean()
    return _clean(short_avg < frac * long_avg, df.index)


def range_expansion(df: pd.DataFrame, lookback: int = 1, mult: float = 2.0) -> pd.Series:
    """Today's range >= mult * the prior bar's range (volatility expansion)."""
    rng = candle_range(df)
    return _clean(rng >= mult * rng.shift(lookback), df.index)
