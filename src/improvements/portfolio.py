"""
Improvement G — multi-strategy portfolio.

Run several strategies as one book with equal capital allocation and simple
conflict handling (v4 spec §Improvement G):

  * each strategy fires independently;
  * if 2+ strategies hold the SAME asset in the SAME direction with overlapping
    holding periods, keep only the first-entered (no double sizing);
  * if they hold the SAME asset in OPPOSITE directions over overlapping periods,
    ignore BOTH (don't fight yourself);
  * portfolio return = sum of surviving trade PnL / number of strategies.

Returns a metrics dict plus the surviving combined trade frame so the dashboard
can draw the book's equity curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_v3 import run_backtests


def _combine_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Resolve same-asset overlaps into one non-conflicting book."""
    if trades.empty:
        return trades
    t = trades.sort_values(["entry_date", "strategy_id"]).reset_index(drop=True)
    rows = t.to_dict("records")
    accepted: list[dict] = []
    open_pos: dict[str, int] = {}        # symbol -> index in `accepted` (live)
    for tr in rows:
        sym = tr["symbol"]
        live_i = open_pos.get(sym)
        if (live_i is not None
                and accepted[live_i]["exit_date"] >= tr["entry_date"]):
            if accepted[live_i]["direction"] == tr["direction"]:
                continue                  # same dir overlap -> drop the newcomer
            accepted[live_i]["_drop"] = True   # opposite overlap -> ignore both
            open_pos.pop(sym, None)
            continue
        tr = dict(tr, _drop=False)
        accepted.append(tr)
        open_pos[sym] = len(accepted) - 1
    out = pd.DataFrame([a for a in accepted if not a["_drop"]])
    return out.drop(columns="_drop", errors="ignore")


def _metrics(combined: pd.DataFrame, n_strategies: int) -> dict:
    if combined.empty:
        return dict(total_return=0.0, avg_asset_return=0.0, sharpe=np.nan,
                    max_drawdown=0.0, n_trades=0, win_rate=0.0,
                    profit_factor=0.0, n_assets=0)
    c = combined.sort_values("exit_date")
    pnl = c["pnl_pct"].to_numpy(float)            # per-trade fraction
    alloc = pnl / n_strategies                    # equal capital split
    eq = np.cumsum(alloc) * 100.0                 # cumulative % (non-compounded)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    avg_hold = float(c["holding_days"].mean())
    if len(alloc) >= 2 and alloc.std(ddof=1) > 0:
        ann = np.sqrt(252.0 / max(avg_hold, 1.0))
        sharpe = float(alloc.mean() / alloc.std(ddof=1) * ann)
    else:
        sharpe = np.nan
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gl = float(-losses.sum())
    pf = float(wins.sum() / gl) if gl > 0 else (np.inf if wins.sum() > 0 else 0.0)
    # cross-asset AVERAGE return — same basis as a single strategy's reported
    # OOS return (mean over assets of that asset's summed trade PnL), so the
    # portfolio is comparable apples-to-apples with the individual strategies.
    avg_asset_return = float(combined.groupby("symbol")["pnl_pct"].sum().mean() * 100.0)
    return dict(
        total_return=float(alloc.sum() * 100.0),
        avg_asset_return=avg_asset_return,
        sharpe=sharpe,
        max_drawdown=float(dd.min()),
        n_trades=int(len(c)),
        win_rate=float(len(wins) / len(pnl)),
        profit_factor=pf,
        n_assets=int(c["symbol"].nunique()))


def run_portfolio(strategies, data, date_mask=None, name="portfolio") -> dict:
    """Backtest the strategies, combine into one book, return metrics + trades."""
    raw = run_backtests(list(strategies), data, date_mask=date_mask,
                        show_progress=False)
    combined = _combine_trades(raw)
    m = _metrics(combined, max(1, len(strategies)))
    m["name"] = name
    m["members"] = [s.id for s in strategies]
    m["raw_trades"] = len(raw)
    return {"metrics": m, "trades": combined}
