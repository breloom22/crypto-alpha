"""
oos_validator.py — IS/OOS split, survival criteria, walk-forward (Part 7).

Time split (spec §7.1):
    IS  : 2022-01-01 .. 2024-06-30   (strategy selection / exit & filter tuning)
    OOS : 2024-07-01 .. 2026-06-06   (final validation)

Survival (spec §7.2), evaluated on the OOS window:
    survived : OOS cross-asset return > 0
    valid    : OOS avg profit factor > 1.0
    robust   : OOS sharpe > IS sharpe * 0.5

Walk-forward (spec §7.3): 5 rolling IS/OOS windows; >= 3/5 OOS-positive = validated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_v3 import run_backtests
from scorer_v3 import score_trades, aggregate_strategies, per_asset_metrics

IS_WINDOW = ("2022-01-01", "2024-06-30")
OOS_WINDOW = ("2024-07-01", "2026-06-06")

WF_WINDOWS = [
    (("2022-01-01", "2023-12-31"), ("2024-01-01", "2024-06-30")),
    (("2022-07-01", "2024-06-30"), ("2024-07-01", "2024-12-31")),
    (("2023-01-01", "2024-12-31"), ("2025-01-01", "2025-06-30")),
    (("2023-07-01", "2025-06-30"), ("2025-07-01", "2025-12-31")),
    (("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-06-06")),
]


def _agg_indexed(trades, data, mask, group_of):
    pa = per_asset_metrics(trades, data, mask, group_of)
    agg = aggregate_strategies(pa)
    return agg.set_index("strategy_id") if not agg.empty else agg


def oos_validate(strategies, data, group_of, show_progress=True) -> pd.DataFrame:
    """Backtest the given strategies on IS and OOS, merge, flag survival."""
    is_trades = run_backtests(strategies, data, date_mask=IS_WINDOW,
                              show_progress=show_progress)
    oos_trades = run_backtests(strategies, data, date_mask=OOS_WINDOW,
                               show_progress=show_progress)
    is_agg = _agg_indexed(is_trades, data, IS_WINDOW, group_of)
    oos_agg = _agg_indexed(oos_trades, data, OOS_WINDOW, group_of)
    if is_agg.empty:
        return pd.DataFrame()

    rows = []
    for sid in is_agg.index:
        is_r = is_agg.loc[sid]
        has_oos = (not oos_agg.empty) and (sid in oos_agg.index)
        oos_r = oos_agg.loc[sid] if has_oos else None
        oos_return = float(oos_r["cross_asset_avg_return"]) if has_oos else np.nan
        oos_pf = float(oos_r["avg_profit_factor"]) if has_oos else np.nan
        oos_sharpe = float(oos_r["cross_asset_sharpe"]) if has_oos else np.nan
        is_sharpe = float(is_r["cross_asset_sharpe"])
        survived = bool(oos_return > 0) if has_oos else False
        valid = bool(oos_pf > 1.0) if has_oos else False
        robust = bool(has_oos and np.isfinite(is_sharpe) and is_sharpe > 0
                      and oos_sharpe > is_sharpe * 0.5)
        rows.append({
            "strategy_id": sid, "category": is_r["category"],
            "is_return": float(is_r["cross_asset_avg_return"]),
            "is_median_return": float(is_r["cross_asset_median_return"]),
            "is_sharpe": is_sharpe, "is_pf": float(is_r["avg_profit_factor"]),
            "is_trades": int(is_r["total_trades"]),
            "oos_return": oos_return, "oos_sharpe": oos_sharpe, "oos_pf": oos_pf,
            "oos_trades": int(oos_r["total_trades"]) if has_oos else 0,
            "survived": survived, "valid": valid, "robust": robust,
        })
    out = pd.DataFrame(rows).sort_values("is_return", ascending=False).reset_index(drop=True)
    return out


def walk_forward(strategies, data, group_of, show_progress=False) -> pd.DataFrame:
    """OOS cross-asset return for each of the 5 walk-forward windows."""
    per_window = {}
    for i, (_, oos) in enumerate(WF_WINDOWS, start=1):
        trades = run_backtests(strategies, data, date_mask=oos,
                               show_progress=show_progress)
        agg = _agg_indexed(trades, data, oos, group_of)
        per_window[i] = (agg["cross_asset_avg_return"] if not agg.empty
                         else pd.Series(dtype=float))
    sids = [s.id for s in strategies]
    rows = []
    for sid in sids:
        vals = {f"oos_w{i}": float(per_window[i].get(sid, np.nan))
                for i in range(1, len(WF_WINDOWS) + 1)}
        positives = sum(1 for v in vals.values() if np.isfinite(v) and v > 0)
        rows.append({"strategy_id": sid, **vals,
                     "windows_positive": positives,
                     "wf_validated": positives >= 3})
    return pd.DataFrame(rows).drop_duplicates("strategy_id").reset_index(drop=True)
