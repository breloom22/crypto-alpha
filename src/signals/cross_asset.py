"""Group S7 — Cross-asset / relative-strength signals (12 signals).

Every signal here is ``cross_asset=True`` with signature
``(data, symbol, direction, **params)`` and returns a boolean entry Series for
``symbol`` (it may read the whole {symbol: DataFrame} universe to do so).

Conventions (see docs/v3_signal_contract.md):
* a True at day t means "enter at t+1 open" — only data at indices <= t may be
  used (rolling / ewm / shift(+k) are all backward-looking; never shift(-k)).
* ALT/BTC ratio signals return all-False when ``symbol == "BTC"`` (ratio vs self
  is trivial).
* Market-wide signals (breadth thrust/collapse, market-cap momentum, correlation
  regimes) compute ONE market series on the market calendar, then broadcast
  (reindex) it onto the requested symbol's index.
* "크로스업/크로스다운" → ``_ta.crosses_above`` / ``_ta.crosses_below``.
* Guard div-by-zero with ``.replace(0, np.nan)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import _ta
from indicators import _ta_extended as tae
from indicators.base import market_frame, market_index, ASSETS, SECTORS

from .registry import signal, LONG, SHORT, BOTH


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _false(idx) -> pd.Series:
    return pd.Series(False, index=idx)


def _btc_close_on(data, idx) -> pd.Series:
    """BTC close reindexed onto ``idx`` (the requested symbol's calendar)."""
    return data["BTC"]["close"].reindex(idx)


def _ratio(data, symbol) -> pd.Series:
    """ALT/BTC close ratio aligned to ``symbol``'s index (BTC denom guarded)."""
    idx = data[symbol].index
    btc = _btc_close_on(data, idx).replace(0, np.nan)
    return data[symbol]["close"] / btc


def _wide(data, field, idx) -> pd.DataFrame:
    """date x symbol frame of one OHLCV field, reindexed to the market calendar."""
    return market_frame(data, field).reindex(idx)


def _breadth_above_ema(data, idx, ema_period) -> pd.Series:
    """Fraction of assets trading above their own EMA(n), on the market calendar.

    Look-ahead safe: EMA(t) uses only close[<=t]; assets not yet listed are NaN
    and excluded from both numerator and denominator (never imputed)."""
    close = _wide(data, "close", idx)
    ema = pd.DataFrame({s: _ta.ema(data[s]["close"], ema_period)
                        for s in data}).reindex(idx)
    valid = close.notna() & ema.notna()
    above = (close > ema).where(valid)
    return above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)


def _market_returns(data, idx) -> pd.DataFrame:
    return _wide(data, "close", idx).pct_change()


# ---------------------------------------------------------------------------
# S7.01 — ALT/BTC relative-strength reversal (weak -> strong)
# ---------------------------------------------------------------------------
@signal("S7.01", "ALT/BTC RS Reversal (Weak->Strong)", "S7", (LONG,),
        cross_asset=True, period=14, level=30)
def s7_01_rs_reversal_up(data, symbol, direction, period=14, level=30):
    idx = data[symbol].index
    if symbol == "BTC":
        return _false(idx)
    r = tae.rsi(_ratio(data, symbol), period)
    return _ta.crosses_above(r, level)


# ---------------------------------------------------------------------------
# S7.02 — ALT/BTC relative-strength reversal (strong -> weak)
# ---------------------------------------------------------------------------
@signal("S7.02", "ALT/BTC RS Reversal (Strong->Weak)", "S7", (SHORT,),
        cross_asset=True, period=14, level=70)
def s7_02_rs_reversal_down(data, symbol, direction, period=14, level=70):
    idx = data[symbol].index
    if symbol == "BTC":
        return _false(idx)
    r = tae.rsi(_ratio(data, symbol), period)
    return _ta.crosses_below(r, level)


# ---------------------------------------------------------------------------
# S7.03 — Breadth Thrust (market-wide)
# % of assets above EMA(20) jumps from < 20% to > 50% within `window` days.
# ---------------------------------------------------------------------------
@signal("S7.03", "Breadth Thrust", "S7", (LONG,),
        cross_asset=True, ema_period=20, low_thr=0.20, high_thr=0.50, window=3)
def s7_03_breadth_thrust(data, symbol, direction,
                         ema_period=20, low_thr=0.20, high_thr=0.50, window=3):
    idx = market_index(data)
    breadth = _breadth_above_ema(data, idx, ema_period)
    # was deeply oversold (< low_thr) at some point in the prior `window` bars,
    # and is now above high_thr -> thrust.
    was_low = (breadth.rolling(window, min_periods=1).min().shift(1) < low_thr)
    now_high = breadth > high_thr
    fire = (was_low & now_high).fillna(False)
    return fire.reindex(data[symbol].index)


# ---------------------------------------------------------------------------
# S7.04 — Breadth Collapse (market-wide)
# % above EMA(20) drops from > 80% to < 50%.
# ---------------------------------------------------------------------------
@signal("S7.04", "Breadth Collapse", "S7", (SHORT,),
        cross_asset=True, ema_period=20, high_thr=0.80, low_thr=0.50, window=3)
def s7_04_breadth_collapse(data, symbol, direction,
                           ema_period=20, high_thr=0.80, low_thr=0.50, window=3):
    idx = market_index(data)
    breadth = _breadth_above_ema(data, idx, ema_period)
    was_high = (breadth.rolling(window, min_periods=1).max().shift(1) > high_thr)
    now_low = breadth < low_thr
    fire = (was_high & now_low).fillna(False)
    return fire.reindex(data[symbol].index)


# ---------------------------------------------------------------------------
# S7.05 — Pair-Spread Mean Reversion
# ALT/BTC spread z-score(30) < -2  -> LONG numerator (the ALT).
# ---------------------------------------------------------------------------
@signal("S7.05", "Pair Spread Mean Reversion", "S7", (LONG,),
        cross_asset=True, period=30, z_thr=-2.0)
def s7_05_pair_spread_reversion(data, symbol, direction, period=30, z_thr=-2.0):
    idx = data[symbol].index
    if symbol == "BTC":
        return _false(idx)
    z = _ta.zscore(_ratio(data, symbol), period)
    return (z < z_thr).fillna(False)


# ---------------------------------------------------------------------------
# S7.06 — Pair-Spread Momentum
# ALT/BTC spread z-score(30) > +2 sustained (trend following) -> LONG numerator.
# ---------------------------------------------------------------------------
@signal("S7.06", "Pair Spread Momentum", "S7", (LONG,),
        cross_asset=True, period=30, z_thr=2.0, persist=2)
def s7_06_pair_spread_momentum(data, symbol, direction,
                               period=30, z_thr=2.0, persist=2):
    idx = data[symbol].index
    if symbol == "BTC":
        return _false(idx)
    z = _ta.zscore(_ratio(data, symbol), period)
    hot = (z > z_thr)
    # "지속 중" — z has stayed above the threshold for `persist` consecutive bars.
    sustained = hot.rolling(persist, min_periods=persist).sum() >= persist
    return sustained.fillna(False)


# ---------------------------------------------------------------------------
# S7.07 — Sector Rotation: L1 vs Meme/Infra
# L1 7d return > Meme 7d return  AND  L1 7d return > Infra 7d return.
# LONG the L1 members; SHORT the non-L1 (Meme + Infra) members.
# ---------------------------------------------------------------------------
@signal("S7.07", "Sector Rotation (L1 vs Meme/Infra)", "S7", BOTH,
        cross_asset=True, period=7)
def s7_07_sector_rotation(data, symbol, direction, period=7):
    idx = market_index(data)
    close = _wide(data, "close", idx)

    def sector_ret(members):
        members = [m for m in members if m in close.columns]
        return close[members].pct_change(period).mean(axis=1)

    l1 = sector_ret(SECTORS["L1"])
    meme = sector_ret(SECTORS["Meme"])
    infra = sector_ret(SECTORS["Infra"])
    l1_leads = ((l1 > meme) & (l1 > infra)).fillna(False)

    is_l1 = symbol in SECTORS["L1"]
    if direction == "LONG":
        fire = l1_leads if is_l1 else pd.Series(False, index=idx)
    else:  # SHORT the rest (Meme + Infra) on the same condition
        fire = l1_leads if not is_l1 else pd.Series(False, index=idx)
    return fire.reindex(data[symbol].index)


# ---------------------------------------------------------------------------
# S7.08 — Correlation Breakdown (decoupling -> follow the asset's own trend)
# 30d corr(asset returns, BTC returns) < 0.3.  LONG if the asset's own trend is
# up, SHORT if down (trend following on the decoupled asset).
# ---------------------------------------------------------------------------
@signal("S7.08", "Correlation Breakdown", "S7", BOTH,
        cross_asset=True, corr_period=30, corr_thr=0.30, trend_period=10)
def s7_08_correlation_breakdown(data, symbol, direction,
                                corr_period=30, corr_thr=0.30, trend_period=10):
    idx = data[symbol].index
    if symbol == "BTC":
        return _false(idx)
    sret = data[symbol]["close"].pct_change()
    bret = _btc_close_on(data, idx).pct_change()
    corr = sret.rolling(corr_period, min_periods=corr_period).corr(bret)
    decoupled = corr < corr_thr
    # asset's own trend: sign of its `trend_period`-day return.
    own_ret = data[symbol]["close"].pct_change(trend_period)
    if direction == "LONG":
        fire = decoupled & (own_ret > 0)
    else:
        fire = decoupled & (own_ret < 0)
    return fire.fillna(False)


# ---------------------------------------------------------------------------
# S7.09 — Lead-Lag: BTC leads ALT
# rolling corr( BTC return 2d ago , ALT return today ) > 0.5  AND  BTC was a
# green candle 2 days ago  -> LONG ALT (BTC's prior strength should pull it up).
# ---------------------------------------------------------------------------
@signal("S7.09", "Lead-Lag: BTC Leads", "S7", (LONG,),
        cross_asset=True, lag=2, corr_period=20, corr_thr=0.5)
def s7_09_lead_lag_btc(data, symbol, direction,
                       lag=2, corr_period=20, corr_thr=0.5):
    idx = data[symbol].index
    if symbol == "BTC":
        return _false(idx)
    alt_ret = data[symbol]["close"].pct_change()
    btc_close = _btc_close_on(data, idx)
    btc_ret = btc_close.pct_change()
    btc_ret_lag = btc_ret.shift(lag)               # BTC return `lag` days ago
    corr = alt_ret.rolling(corr_period, min_periods=corr_period).corr(btc_ret_lag)
    lead = corr > corr_thr
    btc_green_lag = (btc_ret.shift(lag) > 0)        # BTC green `lag` days ago
    return (lead & btc_green_lag).fillna(False)


# ---------------------------------------------------------------------------
# S7.10 — Market-Cap (volume-weighted) Momentum -> panic reversal
# volume-weighted 7d market return < -10%  -> LONG everything.
# ---------------------------------------------------------------------------
@signal("S7.10", "Market-Cap Weighted Momentum (Panic Reversal)", "S7", (LONG,),
        cross_asset=True, period=7, thr=-0.10)
def s7_10_mktcap_momentum(data, symbol, direction, period=7, thr=-0.10):
    idx = market_index(data)
    close = _wide(data, "close", idx)
    vol = _wide(data, "volume", idx)
    ret = close.pct_change(period)
    # volume weights from the same bar (volume[t] is known at t — no look-ahead).
    w = vol.where(ret.notna())
    num = (ret * w).sum(axis=1, min_count=1)
    den = w.sum(axis=1, min_count=1).replace(0, np.nan)
    vw_ret = num / den
    fire = (vw_ret < thr).fillna(False)
    return fire.reindex(data[symbol].index)


# ---------------------------------------------------------------------------
# S7.11 — Relative-Strength Ranking (bottom of the pack -> rebound)
# Among the 9 assets, the symbol's 20d return rank is bottom (8th or 9th).
# ---------------------------------------------------------------------------
@signal("S7.11", "Relative Strength Ranking (Bottom Rebound)", "S7", (LONG,),
        cross_asset=True, period=20, bottom_n=2)
def s7_11_rs_ranking_bottom(data, symbol, direction, period=20, bottom_n=2):
    idx = market_index(data)
    close = _wide(data, "close", idx)
    ret = close.pct_change(period)
    # rank ascending: 1 = worst performer. Bottom `bottom_n` ranks -> rebound bet.
    ranks = ret.rank(axis=1, ascending=True, method="min")
    if symbol not in ranks.columns:
        return _false(data[symbol].index)
    fire = (ranks[symbol] <= bottom_n).fillna(False)
    return fire.reindex(data[symbol].index)


# ---------------------------------------------------------------------------
# S7.12 — Correlation Extreme + Market Weakness -> buy the relative outperformer
# 30d cross-asset correlation > 0.9 (synchronised)  AND  market falling
# -> LONG the asset that has fallen the least (best short-window return).
# ---------------------------------------------------------------------------
@signal("S7.12", "Correlation Extreme + Weakness (Relative Strength)", "S7",
        (LONG,), cross_asset=True, corr_period=30, corr_thr=0.90,
        ret_period=7, top_n=1)
def s7_12_corr_extreme_weak(data, symbol, direction,
                            corr_period=30, corr_thr=0.90, ret_period=7, top_n=1):
    idx = market_index(data)
    ret_daily = _market_returns(data, idx)

    # average pairwise correlation over `corr_period` (look-ahead safe rolling).
    arr = ret_daily.to_numpy()
    cols_all = list(ret_daily.columns)
    avg_corr = np.full(len(ret_daily), np.nan)
    n = corr_period
    for t in range(n, len(ret_daily)):
        win = arr[t - n + 1: t + 1]
        ok = ~np.isnan(win).any(axis=0)
        w = win[:, ok]
        if w.shape[1] < 2:
            continue
        c = np.corrcoef(w, rowvar=False)
        k = c.shape[0]
        avg_corr[t] = (c.sum() - np.trace(c)) / (k * (k - 1))
    avg_corr = pd.Series(avg_corr, index=ret_daily.index)

    close = _wide(data, "close", idx)
    win_ret = close.pct_change(ret_period)
    market_ret = win_ret.mean(axis=1)

    synchronised = avg_corr > corr_thr
    falling = market_ret < 0
    # relative outperformer: highest short-window return (= least-fallen),
    # rank descending so 1 = best. Top `top_n` qualify.
    ranks = win_ret.rank(axis=1, ascending=False, method="min")
    if symbol not in ranks.columns:
        return _false(data[symbol].index)
    is_best = ranks[symbol] <= top_n

    fire = (synchronised & falling & is_best).fillna(False)
    return fire.reindex(data[symbol].index)
