"""
spot_check.py — June 2026 detailed trade timeline (Part 8).

Lists every entry the validated strategies took in 2026-05-25 .. 2026-06-06,
with entry price, exit/current price, direction and P&L. Positions still open at
the last bar are marked MtM (mark-to-market).
"""
from __future__ import annotations

import pandas as pd

from backtester_v3 import run_backtests, backtest_strategy
from signals import registry

SPOT_START = "2026-05-25"
SPOT_END = "2026-06-06"


def spot_check(strategies, data, start=SPOT_START, end=SPOT_END) -> pd.DataFrame:
    """Trades whose ENTRY falls in [start, end], across the validated strategies."""
    last_dates = {sym: df.index.max() for sym, df in data.items()}
    trades = run_backtests(strategies, data, date_mask=None, show_progress=False)
    if trades.empty:
        return trades
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    win = trades[(trades["entry_date"] >= s) & (trades["entry_date"] <= e)].copy()
    if win.empty:
        return win
    win["status"] = win.apply(
        lambda r: "OPEN (MtM)" if (r["exit_reason"] == "eod"
                                   and r["exit_date"] == last_dates.get(r["symbol"]))
        else "closed", axis=1)
    win["pnl_pct"] = (win["pnl_pct"] * 100.0).round(3)
    win = win.sort_values(["entry_date", "strategy_id", "symbol"]).reset_index(drop=True)
    return win[["strategy_id", "symbol", "direction", "entry_date", "entry_price",
                "exit_date", "exit_price", "exit_reason", "holding_days",
                "pnl_pct", "status"]]


def spot_signals(strategies, data, start=SPOT_START, end=SPOT_END) -> pd.DataFrame:
    """Raw signal firings in the window (independent of cooldown/overlap), useful
    to see what each strategy 'saw' even when no trade was opened."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for strat in strategies:
        for sym, df in data.items():
            long_e, short_e, _, _ = strat.build_entries(data, sym)
            for d, fired in long_e.items():
                if fired and s <= d <= e:
                    rows.append({"strategy_id": strat.id, "symbol": sym,
                                 "date": d, "direction": "LONG"})
            for d, fired in short_e.items():
                if fired and s <= d <= e:
                    rows.append({"strategy_id": strat.id, "symbol": sym,
                                 "date": d, "direction": "SHORT"})
    return pd.DataFrame(rows).sort_values(["date", "strategy_id"]).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["strategy_id", "symbol", "date", "direction"])
