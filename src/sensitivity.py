"""
sensitivity.py — parameter robustness sweep (Part 7.5).

For each top strategy we perturb one knob at a time and re-measure the
cross-asset return:
  * the signal's primary period:  x0.7, x0.85, x1.0, x1.15, x1.3
  * stop-loss:   -3% / -5% / -7% / -10%
  * take-profit: +5% / +7% / +10% / +15%
  * max hold:    5 / 7 / 10 / 14 / 21 days
A strategy whose return stays within +/-30% of baseline across a knob's sweep is
flagged stable for that knob.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from backtester_v3 import run_backtests
from scorer_v3 import per_asset_metrics, aggregate_strategies
from signals import registry
from strategy import Strategy

PERIOD_MULTS = [0.7, 0.85, 1.0, 1.15, 1.3]
SL_VALUES = [-0.03, -0.05, -0.07, -0.10]
TP_VALUES = [0.05, 0.07, 0.10, 0.15]
HOLD_VALUES = [5, 7, 10, 14, 21]


def _xret(strat, data, mask, group_of) -> float:
    trades = run_backtests([strat], data, date_mask=mask, show_progress=False)
    pa = per_asset_metrics(trades, data, mask, group_of)
    agg = aggregate_strategies(pa)
    if agg.empty:
        return np.nan
    return float(agg.iloc[0]["cross_asset_avg_return"])


def _primary_period(spec) -> "tuple[str, int] | None":
    for k, v in spec.params.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v >= 2:
            return k, v
    return None


def sensitivity(strategies, data, group_of, mask=None, top_n=10) -> pd.DataFrame:
    rows = []
    for strat in strategies[:top_n]:
        if strat.combo_type is not None:
            continue
        spec = registry.get(strat.members[0][0])
        base_ret = _xret(strat, data, mask, group_of)

        def record(param, value, variant):
            r = _xret(variant, data, mask, group_of)
            pct = ((r - base_ret) / abs(base_ret) * 100.0
                   if np.isfinite(base_ret) and base_ret != 0 else np.nan)
            rows.append({"strategy_id": strat.id, "param": param, "value": value,
                         "return": r, "pct_change": pct,
                         "is_default": False})

        # signal primary period
        pp = _primary_period(spec)
        if pp is not None:
            pname, pval = pp
            for m in PERIOD_MULTS:
                v = max(2, int(round(pval * m)))
                variant = replace(strat, signal_params={**strat.signal_params, pname: v})
                record(f"period:{pname}", v, variant)
        # stop loss
        for sl in SL_VALUES:
            variant = replace(strat, exit=replace(strat.exit, stop_loss_pct=sl))
            record("stop_loss", sl, variant)
        # take profit
        for tp in TP_VALUES:
            variant = replace(strat, exit=replace(strat.exit, take_profit_pct=tp))
            record("take_profit", tp, variant)
        # max hold
        for mh in HOLD_VALUES:
            variant = replace(strat, exit=replace(strat.exit, max_hold=mh))
            record("max_hold", mh, variant)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # per-knob stability: spread of pct_change within +/-30%
    stab = (df.dropna(subset=["pct_change"]).groupby(["strategy_id", "param"])["pct_change"]
            .apply(lambda s: bool((s.abs() <= 30).all())).rename("stable").reset_index())
    return df.merge(stab, on=["strategy_id", "param"], how="left")
