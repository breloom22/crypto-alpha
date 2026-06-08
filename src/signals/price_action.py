"""
price_action.py — signal group S1: price action / candlestick patterns (15 signals).

Each signal consumes one asset's OHLCV frame and returns a boolean pd.Series
aligned to df.index. A True at day t means "enter at the next day's (t+1) open";
every computation here uses only data at indices <= t (rolling / shift(+k)), so
the series is LOOK-AHEAD SAFE.

The candle/breakout geometry lives in indicators._patterns (pat.*). This module
composes those primitives with direction and trend context (pat.uptrend /
pat.downtrend) exactly as the v3 spec table (S1.01–S1.15) requires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import _ta
from indicators import _patterns as pat

from .registry import signal, LONG, SHORT, BOTH


# ---------------------------------------------------------------------------
# S1.01 — Bullish / Bearish Engulfing
# LONG : prev bear candle, today bull engulfs prev body (open<prev_close, close>prev_open)
# SHORT: prev bull candle, today bear engulfs prev body (mirror)
# ---------------------------------------------------------------------------
@signal("S1.01", "Bullish/Bearish Engulfing", "S1", BOTH)
def engulfing(df, direction):
    if direction == LONG:
        return pat.bullish_engulfing(df)
    return pat.bearish_engulfing(df)


# ---------------------------------------------------------------------------
# S1.02 — Hammer / Shooting Star (trend context required)
# LONG : hammer shape (lower wick >= 2x body, upper wick < 0.3x body) in a
#        downtrend (5-day ROC < -3%)
# SHORT: shooting-star shape (upper wick >= 2x body, lower wick < 0.3x body) in
#        an uptrend (5-day ROC > 3%)
# ---------------------------------------------------------------------------
@signal("S1.02", "Hammer / Shooting Star", "S1", BOTH, trend_n=5, trend_thresh=3.0)
def hammer_shooting_star(df, direction, trend_n=5, trend_thresh=3.0):
    if direction == LONG:
        return pat.hammer_shape(df) & pat.downtrend(df["close"], trend_n, trend_thresh)
    return pat.shooting_star_shape(df) & pat.uptrend(df["close"], trend_n, trend_thresh)


# ---------------------------------------------------------------------------
# S1.03 — Doji + next-bar direction confirmation
# LONG : doji on the prior bar AND today is a bull candle (the confirmation bar)
# SHORT: doji on the prior bar AND today is a bear candle
# (Fire on the confirmation bar t so entry at t+1 open stays look-ahead safe.)
# ---------------------------------------------------------------------------
@signal("S1.03", "Doji + Direction Confirm", "S1", BOTH, thresh=0.1)
def doji_confirm(df, direction, thresh=0.1):
    prev_doji = pat.doji(df, thresh).shift(1).fillna(False)
    if direction == LONG:
        return prev_doji & pat.is_bull(df)
    return prev_doji & pat.is_bear(df)


# ---------------------------------------------------------------------------
# S1.04 — Inside Bar Breakout
# LONG : prior bar is an inside bar; today breaks above the inside bar's high
# SHORT: prior bar is an inside bar; today breaks below the inside bar's low
# ---------------------------------------------------------------------------
@signal("S1.04", "Inside Bar Breakout", "S1", BOTH)
def inside_bar_breakout(df, direction):
    prev_inside = pat.inside_bar(df).shift(1).fillna(False)
    if direction == LONG:
        return prev_inside & (df["high"] > df["high"].shift(1))
    return prev_inside & (df["low"] < df["low"].shift(1))


# ---------------------------------------------------------------------------
# S1.05 — Outside Bar (Key Reversal) with trend context
# LONG : outside bar (today high>prev high AND low<prev low) closing bullish,
#        after a downtrend
# SHORT: outside bar closing bearish, after an uptrend
# ---------------------------------------------------------------------------
@signal("S1.05", "Outside Bar Key Reversal", "S1", BOTH, trend_n=5, trend_thresh=3.0)
def outside_bar_reversal(df, direction, trend_n=5, trend_thresh=3.0):
    outside = pat.outside_bar(df)
    # trend "before" the reversal bar -> evaluate trend as of the prior bar
    prior_down = pat.downtrend(df["close"], trend_n, trend_thresh).shift(1).fillna(False)
    prior_up = pat.uptrend(df["close"], trend_n, trend_thresh).shift(1).fillna(False)
    if direction == LONG:
        return outside & pat.is_bull(df) & prior_down
    return outside & pat.is_bear(df) & prior_up


# ---------------------------------------------------------------------------
# S1.06 — Three White Soldiers / Three Black Crows
# Spec table: "3일 연속 양봉, 각각 전일 종가 위에서 시가, 종가 상승" — each bar
# opens ABOVE the prior close (not merely inside the prior body, which is what
# pat.three_white_soldiers encodes). We honor the literal spec here.
# LONG : 3 consecutive bull candles, rising closes, each open > prior close
# SHORT: 3 consecutive bear candles, falling closes, each open < prior close
# ---------------------------------------------------------------------------
@signal("S1.06", "Three White Soldiers / Black Crows", "S1", BOTH)
def three_soldiers_crows(df, direction):
    o, c = df["open"], df["close"]
    bull, bear = pat.is_bull(df), pat.is_bear(df)
    if direction == LONG:
        cond = bull & bull.shift(1) & bull.shift(2)
        rising = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
        open_above = (o > c.shift(1)) & (o.shift(1) > c.shift(2))
        return (cond & rising & open_above).fillna(False)
    cond = bear & bear.shift(1) & bear.shift(2)
    falling = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    open_below = (o < c.shift(1)) & (o.shift(1) < c.shift(2))
    return (cond & falling & open_below).fillna(False)


# ---------------------------------------------------------------------------
# S1.07 — Morning Star / Evening Star (3-bar)
# LONG : big bear -> small body (gap) -> big bull recovering >=50% of bar 1
# SHORT: mirror (big bull -> small body -> big bear failing below 50%)
# ---------------------------------------------------------------------------
@signal("S1.07", "Morning Star / Evening Star", "S1", BOTH, small_body_frac=0.5)
def morning_evening_star(df, direction, small_body_frac=0.5):
    if direction == LONG:
        return pat.morning_star(df, small_body_frac)
    return pat.evening_star(df, small_body_frac)


# ---------------------------------------------------------------------------
# S1.08 — Pin Bar
# LONG : one long lower tail (>=66% of range), body within the top 25% of range
# SHORT: one long upper tail (>=66% of range), body within the bottom 25%
# ---------------------------------------------------------------------------
@signal("S1.08", "Pin Bar", "S1", BOTH, tail_frac=0.66, body_zone=0.25)
def pin_bar(df, direction, tail_frac=0.66, body_zone=0.25):
    if direction == LONG:
        return pat.pin_bar_bullish(df, tail_frac, body_zone)
    return pat.pin_bar_bearish(df, tail_frac, body_zone)


# ---------------------------------------------------------------------------
# S1.09 — Gap Fill (mean-reversion expectation)
# LONG : gap down (open < prev low) -> expect the gap to fill upward
# SHORT: gap up   (open > prev high) -> expect the gap to fill downward
# ---------------------------------------------------------------------------
@signal("S1.09", "Gap Fill", "S1", BOTH)
def gap_fill(df, direction):
    if direction == LONG:
        return pat.gap_down(df)
    return pat.gap_up(df)


# ---------------------------------------------------------------------------
# S1.10 — N-Day Breakout (Donchian)
# LONG : close > highest high of prior N bars (N=20)
# SHORT: close < lowest low of prior N bars
# ---------------------------------------------------------------------------
@signal("S1.10", "N-Day Breakout (Donchian)", "S1", BOTH, n=20)
def donchian_breakout(df, direction, n=20):
    if direction == LONG:
        return pat.donchian_high_break(df, n)
    return pat.donchian_low_break(df, n)


# ---------------------------------------------------------------------------
# S1.11 — Narrow Range (NR7) -> next-day directional follow
# Today's range is the smallest of the last 7 bars; the spec uses the SAME
# condition for both directions (the breakout direction is realised at entry).
# ---------------------------------------------------------------------------
@signal("S1.11", "Narrow Range (NR7)", "S1", BOTH, n=7)
def narrow_range_nr7(df, direction, n=7):
    return pat.nr7(df, n)


# ---------------------------------------------------------------------------
# S1.12 — 2-Bar Reversal
# LONG : 2-day cumulative return < -4% AND day-2 close sits in the top 25% of
#        its own range (close_location >= 0.75)
# SHORT: 2-day cumulative return >  +4% AND day-2 close in the bottom 25%
#        (close_location <= 0.25)
# ---------------------------------------------------------------------------
@signal("S1.12", "2-Bar Reversal", "S1", BOTH, ret_thresh=4.0)
def two_bar_reversal(df, direction, ret_thresh=4.0):
    cum_ret = df["close"].pct_change(2) * 100.0
    cloc = pat.close_location(df)
    if direction == LONG:
        return (cum_ret < -ret_thresh) & (cloc >= 0.75)
    return (cum_ret > ret_thresh) & (cloc <= 0.25)


# ---------------------------------------------------------------------------
# S1.13 — Consecutive N-bar Reversal
# LONG : N consecutive bear candles (the N prior bars) followed by today's first
#        bull candle
# SHORT: N consecutive bull candles followed by today's first bear candle
# ---------------------------------------------------------------------------
@signal("S1.13", "Consecutive N-Bar Reversal", "S1", BOTH, n=3)
def consecutive_reversal(df, direction, n=3):
    bull = pat.is_bull(df)
    bear = pat.is_bear(df)
    if direction == LONG:
        prior = pd.Series(True, index=df.index)
        for k in range(1, n + 1):
            prior &= bear.shift(k).fillna(False)
        return prior & bull
    prior = pd.Series(True, index=df.index)
    for k in range(1, n + 1):
        prior &= bull.shift(k).fillna(False)
    return prior & bear


# ---------------------------------------------------------------------------
# S1.14 — Range Contraction -> Expansion
# Contraction held going into today (3-day avg range < 50% of 20-day avg range),
# AND today is a >=2x range-expansion bar. Direction follows today's candle:
# LONG  on a bullish expansion bar, SHORT on a bearish expansion bar.
# ---------------------------------------------------------------------------
@signal("S1.14", "Range Contraction -> Expansion", "S1", BOTH,
        short_n=3, long_n=20, frac=0.5, mult=2.0)
def contraction_expansion(df, direction, short_n=3, long_n=20, frac=0.5, mult=2.0):
    contracted = pat.range_contraction(df, short_n, long_n, frac).shift(1).fillna(False)
    expanded = pat.range_expansion(df, lookback=1, mult=mult)
    base = contracted & expanded
    if direction == LONG:
        return base & pat.is_bull(df)
    return base & pat.is_bear(df)


# ---------------------------------------------------------------------------
# S1.15 — Close-Location Reversal
# LONG : 5 consecutive bars closing in the bottom 25% of their range -> bounce
# SHORT: 5 consecutive bars closing in the top 25% of their range -> drop
# ---------------------------------------------------------------------------
@signal("S1.15", "Close-Location Reversal", "S1", BOTH, n=5)
def close_location_reversal(df, direction, n=5):
    cloc = pat.close_location(df)
    if direction == LONG:
        low_zone = (cloc <= 0.25).fillna(False)
        streak = pd.Series(True, index=df.index)
        for k in range(n):
            streak &= low_zone.shift(k).fillna(False)
        return streak
    high_zone = (cloc >= 0.75).fillna(False)
    streak = pd.Series(True, index=df.index)
    for k in range(n):
        streak &= high_zone.shift(k).fillna(False)
    return streak
