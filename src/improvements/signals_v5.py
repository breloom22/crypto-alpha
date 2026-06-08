"""
signals_v5.py — parameterized filter / signal variants for Phase-5 tuning.

The v3 filter table bakes thresholds into each FilterSpec, and strategies carry
filters as bare ids (no per-strategy params). Rather than rewire that, we
register *concrete* threshold variants the grids can swap in by id:

    F3_15 / F3_20 / F3_25     ADX<thresh ranging filter   (S2.08 grid)
    F8_30 / F8_40 / F8_50     market-breadth<thresh        (S3.12 grid)
    F9_5d / F9_20d            BTC N-day return>0 (LONG)    (S2.08 btc_filter)

Plus a weekly-pivot breakout signal (S3.12W) for the S3.12 pivot_mode axis.
All look-ahead safe; registration is idempotent.
"""
from __future__ import annotations

import filters as flt
from filters import FilterSpec, f3_ranging, f8_weak_breadth, f9_btc_direction
from signals import registry
from signals.registry import BOTH
from indicators import _ta


# --- parameterized filter variants -----------------------------------------
_F_VARIANTS = {
    "F3_15": FilterSpec("F3_15", "Ranging ADX<15", f3_ranging, params={"thresh": 15}),
    "F3_20": FilterSpec("F3_20", "Ranging ADX<20", f3_ranging, params={"thresh": 20}),
    "F3_25": FilterSpec("F3_25", "Ranging ADX<25", f3_ranging, params={"thresh": 25}),
    "F8_30": FilterSpec("F8_30", "Weak breadth<30%", f8_weak_breadth,
                        cross_asset=True, params={"thresh": 0.30}),
    "F8_40": FilterSpec("F8_40", "Weak breadth<40%", f8_weak_breadth,
                        cross_asset=True, params={"thresh": 0.40}),
    "F8_50": FilterSpec("F8_50", "Weak breadth<50%", f8_weak_breadth,
                        cross_asset=True, params={"thresh": 0.50}),
    # BTC-up confirmation (positive baked in -> used on LONG strategies)
    "F9_5d": FilterSpec("F9_5d", "BTC 5d return>0", f9_btc_direction,
                        cross_asset=True, params={"period": 5, "positive": True}),
    "F9_20d": FilterSpec("F9_20d", "BTC 20d return>0", f9_btc_direction,
                         cross_asset=True, params={"period": 20, "positive": True}),
}
flt.FILTERS.update(_F_VARIANTS)


# --- weekly pivot breakout signal (S3.12 pivot_mode='weekly') --------------
def _weekly_pivots(df):
    """Classic floor pivots from the PREVIOUS completed week, broadcast to daily.
    Look-ahead safe: a daily bar in ISO week W uses week W-1's HLC, known before
    W begins."""
    wk = df.resample("W-SUN").agg(high=("high", "max"), low=("low", "min"),
                                  close=("close", "last"))
    ph, pl, pc = wk["high"].shift(1), wk["low"].shift(1), wk["close"].shift(1)
    pp = (ph + pl + pc) / 3.0
    r1 = 2.0 * pp - pl
    s1 = 2.0 * pp - ph
    wk_per = df.index.to_period("W-SUN")
    out = []
    for series in (pp, r1, s1):
        s = series.copy()
        s.index = s.index.to_period("W-SUN")
        out.append(wk_per.map(s))
    return out  # pp, r1, s1 aligned to df.index


@registry.signal("S3.12W", "Pivot Point Breakout (weekly)", "S3", BOTH)
def pivot_point_breakout_weekly(df, direction):
    _, r1, s1 = _weekly_pivots(df)
    import pandas as pd
    r1 = pd.Series(r1, index=df.index)
    s1 = pd.Series(s1, index=df.index)
    close = df["close"]
    if direction == "LONG":
        return _ta.crosses_above(close, r1)
    return _ta.crosses_below(close, s1)
