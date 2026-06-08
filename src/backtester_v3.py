"""
backtester_v3.py — P&L trade-simulation engine (Part 5).

Rules (spec §5.2), applied per (strategy, asset):
  * entry  = the bar AFTER a signal, filled at that bar's OPEN
  * exit   = condition bar's CLOSE, except intraday SL/TP/trailing/ATR/chandelier
             stops which fill at their trigger level (checked via High/Low)
  * same-bar SL & TP -> SL wins (conservative)
  * slippage = 0.1% per side (entry + exit)
  * non-overlapping positions per asset; 3-bar cooldown after each exit
  * date_mask restricts ENTRY dates to a window (IS / OOS / walk-forward)

Look-ahead safety lives in the signals/indicators; here entries are always
taken at the next open and trailing stops trail the peak through the *prior*
bar, so no within-bar future information leaks into an exit decision.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from indicators import _ta
from indicators import _ta_extended as tae

SLIPPAGE = 0.001       # per side
COOLDOWN = 3           # bars
MIN_TRADES = 10        # below this a (strategy,asset) result is statistically void


@dataclass
class Trade:
    """entry_price / exit_price are PRE-slippage reference levels (the raw entry
    open and the exit trigger/close). pnl_pct is the authoritative return AFTER
    0.1%/side slippage — do not recompute it from the two price columns."""
    strategy_id: str
    symbol: str
    direction: str
    entry_date: "pd.Timestamp"
    entry_price: float
    exit_date: "pd.Timestamp"
    exit_price: float
    exit_reason: str
    holding_days: int
    pnl_pct: float


# ---------------------------------------------------------------------------
def _simulate_one(direction, e, o, h, l, c, atr_arr, chand_long, chand_short,
                  rsi_arr, opp_long, opp_short, cfg, slippage, n):
    """Simulate a single trade from entry bar index e. Returns
    (exit_index, exit_level, reason, pnl_pct).

    Stop/target LEVELS are built from the raw entry open (spec §5.2:
    'entry x (1+sl_pct)'). Slippage is charged on the realized entry and exit
    FILLS, so pnl_pct is the authoritative slippage-adjusted return. The
    returned exit_level is the pre-slippage trigger/close (mirrors entry_price =
    raw open) — do NOT recompute return from the two reported price levels."""
    is_long = direction == "LONG"
    entry_ref = o[e]                                   # raw open -> level basis
    entry_fill = entry_ref * (1 + slippage) if is_long else entry_ref * (1 - slippage)
    atr_e = atr_arr[e] if atr_arr is not None else np.nan
    has_atr = atr_arr is not None and not np.isnan(atr_e)

    def _ret(level):
        fill = level * (1 - slippage) if is_long else level * (1 + slippage)
        return (fill - entry_fill) / entry_fill if is_long else (entry_fill - fill) / entry_fill

    # max hold (E1 / E7)
    max_hold = max(7, 2 * cfg.atr_period) if cfg.time_dynamic else cfg.max_hold

    # running extreme through the PRIOR bar (initialised to the entry reference)
    peak = entry_ref             # highest high seen so far (LONG)
    trough = entry_ref           # lowest low seen so far (SHORT)
    breakeven_armed = False
    profit_trail_armed = False

    for d in range(e, n):
        # ---- assemble stop levels (using extremes through prior bar) -------
        stops = []  # (level, reason)
        if is_long:
            if cfg.stop_loss_pct is not None:
                stops.append((entry_ref * (1 + cfg.stop_loss_pct), "stop_loss"))
            if cfg.atr_stop_mult is not None and has_atr:
                stops.append((entry_ref - cfg.atr_stop_mult * atr_e, "atr_stop"))
            if cfg.trailing_pct is not None:
                stops.append((peak * (1 - cfg.trailing_pct), "trailing_stop"))
            if cfg.chandelier_mult is not None and d - 1 >= 0 and not np.isnan(chand_long[d - 1]):
                stops.append((chand_long[d - 1], "chandelier"))
            if cfg.breakeven_trigger is not None:
                if not breakeven_armed and peak >= entry_ref * (1 + cfg.breakeven_trigger):
                    breakeven_armed = True
                if breakeven_armed:
                    stops.append((entry_ref, "break_even"))
            if cfg.profit_trail_trigger is not None and has_atr:
                if not profit_trail_armed and peak >= entry_ref * (1 + cfg.profit_trail_trigger):
                    profit_trail_armed = True
                if profit_trail_armed:
                    stops.append((peak - cfg.profit_trail_atr * atr_e, "trailing_stop"))
            if cfg.stepped_trail_base is not None and has_atr:        # E13
                mult = cfg.stepped_trail_base
                if cfg.stepped_trail_at is not None and peak >= entry_ref * (1 + cfg.stepped_trail_at):
                    mult = cfg.stepped_trail_tight
                stops.append((peak - mult * atr_e, "trailing_stop"))
            eff_stop = max(stops, key=lambda x: x[0]) if stops else None

            tps = []
            if cfg.take_profit_pct is not None:
                tps.append(entry_ref * (1 + cfg.take_profit_pct))
            if cfg.atr_tp_mult is not None and has_atr:
                tps.append(entry_ref + cfg.atr_tp_mult * atr_e)
            eff_tp = min(tps) if tps else None

            if eff_stop is not None and l[d] <= eff_stop[0]:        # SL beats TP
                return d, eff_stop[0], eff_stop[1], _ret(eff_stop[0])
            if eff_tp is not None and h[d] >= eff_tp:
                return d, eff_tp, "take_profit", _ret(eff_tp)
        else:  # SHORT
            if cfg.stop_loss_pct is not None:
                stops.append((entry_ref * (1 - cfg.stop_loss_pct), "stop_loss"))
            if cfg.atr_stop_mult is not None and has_atr:
                stops.append((entry_ref + cfg.atr_stop_mult * atr_e, "atr_stop"))
            if cfg.trailing_pct is not None:
                stops.append((trough * (1 + cfg.trailing_pct), "trailing_stop"))
            if cfg.chandelier_mult is not None and d - 1 >= 0 and not np.isnan(chand_short[d - 1]):
                stops.append((chand_short[d - 1], "chandelier"))
            if cfg.breakeven_trigger is not None:
                if not breakeven_armed and trough <= entry_ref * (1 - cfg.breakeven_trigger):
                    breakeven_armed = True
                if breakeven_armed:
                    stops.append((entry_ref, "break_even"))
            if cfg.profit_trail_trigger is not None and has_atr:
                if not profit_trail_armed and trough <= entry_ref * (1 - cfg.profit_trail_trigger):
                    profit_trail_armed = True
                if profit_trail_armed:
                    stops.append((trough + cfg.profit_trail_atr * atr_e, "trailing_stop"))
            if cfg.stepped_trail_base is not None and has_atr:        # E13
                mult = cfg.stepped_trail_base
                if cfg.stepped_trail_at is not None and trough <= entry_ref * (1 - cfg.stepped_trail_at):
                    mult = cfg.stepped_trail_tight
                stops.append((trough + mult * atr_e, "trailing_stop"))
            eff_stop = min(stops, key=lambda x: x[0]) if stops else None

            tps = []
            if cfg.take_profit_pct is not None:
                tps.append(entry_ref * (1 - cfg.take_profit_pct))
            if cfg.atr_tp_mult is not None and has_atr:
                tps.append(entry_ref - cfg.atr_tp_mult * atr_e)
            eff_tp = max(tps) if tps else None

            if eff_stop is not None and h[d] >= eff_stop[0]:
                return d, eff_stop[0], eff_stop[1], _ret(eff_stop[0])
            if eff_tp is not None and l[d] <= eff_tp:
                return d, eff_tp, "take_profit", _ret(eff_tp)

        # ---- close-based exits (evaluated at this bar's close) ------------
        close = c[d]
        if cfg.opposite_signal and d > e:                         # E2
            if (is_long and opp_short[d]) or ((not is_long) and opp_long[d]):
                return d, close, "signal", _ret(close)
        if cfg.indicator_neutral and d > e and rsi_arr is not None and not np.isnan(rsi_arr[d]):  # E6
            if (is_long and rsi_arr[d] >= 50) or ((not is_long) and rsi_arr[d] <= 50):
                return d, close, "indicator_neutral", _ret(close)
        if cfg.first_profit_close:                               # E12
            if (close > entry_fill) if is_long else (close < entry_fill):
                return d, close, "first_profit", _ret(close)
        if max_hold is not None and (d - e + 1) >= max_hold:     # E1/E7
            return d, close, "max_hold", _ret(close)
        if d == n - 1:                                           # end of data
            return d, close, "eod", _ret(close)

        # ---- update running extremes for the NEXT bar --------------------
        peak = max(peak, h[d])
        trough = min(trough, l[d])

    last = n - 1
    return last, c[last], "eod", _ret(c[last])


def simulate_symbol(strategy, df, long_e, short_e, opp_long, opp_short, symbol,
                    slippage=SLIPPAGE, cooldown=COOLDOWN, date_mask=None):
    """Walk the asset, opening non-overlapping trades on entry days."""
    cfg = strategy.exit
    idx = df.index
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(c)

    need_atr = (cfg.atr_stop_mult is not None or cfg.atr_tp_mult is not None
                or cfg.profit_trail_trigger is not None or cfg.time_dynamic
                or cfg.stepped_trail_base is not None)
    atr_arr = _ta.atr(df["high"], df["low"], df["close"], cfg.atr_period).to_numpy() if need_atr else None
    if cfg.chandelier_mult is not None:
        cl, cs = tae.chandelier_exit(df["high"], df["low"], df["close"], cfg.chandelier_period, cfg.chandelier_mult)
        chand_long, chand_short = cl.to_numpy(), cs.to_numpy()
    else:
        chand_long = chand_short = None
    rsi_arr = _ta.rsi(df["close"], 14).to_numpy() if cfg.indicator_neutral else None

    le = long_e.to_numpy(bool)
    se = short_e.to_numpy(bool)
    ol = opp_long.to_numpy(bool)
    os_ = opp_short.to_numpy(bool)

    if date_mask is not None:
        ms, me = pd.Timestamp(date_mask[0]), pd.Timestamp(date_mask[1])
    else:
        ms = me = None

    # entry candidates: signal at bar i -> entry bar e=i+1, LONG before SHORT
    cands = []
    for i in range(n - 1):
        if le[i]:
            cands.append((i, "LONG"))
        if se[i]:
            cands.append((i, "SHORT"))
    cands.sort(key=lambda t: (t[0], t[1] != "LONG"))

    trades = []
    blocked_until = -1
    if cfg.cooldown is not None:                 # v5 per-strategy cooldown override
        cooldown = cfg.cooldown
    skip_n = cfg.skip_after_n_losses            # FC3 loss-streak guard (or None)
    skip_armed = False
    for i, direction in cands:
        e = i + 1
        if e <= blocked_until or e >= n:
            continue
        if ms is not None and not (ms <= idx[e] <= me):
            continue
        # FC3: after `skip_n` consecutive losing closes, skip exactly one signal.
        if skip_n and len(trades) >= skip_n and all(t.pnl_pct < 0 for t in trades[-skip_n:]):
            if not skip_armed:
                skip_armed = True
                continue
        x, exit_level, reason, pnl = _simulate_one(
            direction, e, o, h, l, c, atr_arr, chand_long, chand_short,
            rsi_arr, ol, os_, cfg, slippage, n)
        trades.append(Trade(
            strategy_id=strategy.id, symbol=symbol, direction=direction,
            entry_date=idx[e], entry_price=float(o[e]),
            exit_date=idx[x], exit_price=float(exit_level), exit_reason=reason,
            holding_days=int(x - e), pnl_pct=float(pnl)))
        skip_armed = False                       # re-arm FC3 after a real entry
        blocked_until = x + cooldown
    return trades


def backtest_strategy(strategy, data, slippage=SLIPPAGE, cooldown=COOLDOWN,
                      date_mask=None):
    trades = []
    for symbol, df in data.items():
        long_e, short_e, opp_l, opp_s = strategy.build_entries(data, symbol)
        trades += simulate_symbol(strategy, df, long_e, short_e, opp_l, opp_s,
                                  symbol, slippage, cooldown, date_mask)
    return trades


def run_backtests(strategies, data, date_mask=None, show_progress=True,
                  slippage=SLIPPAGE, cooldown=COOLDOWN) -> pd.DataFrame:
    """Backtest a list of strategies across the universe. Returns a flat trade
    DataFrame (one row per trade)."""
    try:
        from tqdm import tqdm
    except Exception:                                   # noqa: BLE001
        def tqdm(x, **k): return x
    rows = []
    it = tqdm(strategies, desc="backtest") if show_progress else strategies
    for strat in it:
        for t in backtest_strategy(strat, data, slippage, cooldown, date_mask):
            rows.append(asdict(t))
    cols = ["strategy_id", "symbol", "direction", "entry_date", "entry_price",
            "exit_date", "exit_price", "exit_reason", "holding_days", "pnl_pct"]
    return pd.DataFrame(rows, columns=cols)
