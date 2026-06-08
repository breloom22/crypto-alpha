"""
grid_search.py — Phase-5 candidate builder + staged (greedy) grid search.

A flat param dict is turned into a :class:`StrategyV5`. Param vocabulary:

  entry        entry_delay:int, entry_confirmation:str|None, exclude_assets:tuple
  signals      nr_period / short_nr_period (S1.11 n), vortex_period (S2.08 n),
               natr_period / natr_high_thr (S4.04), supertrend_period /
               supertrend_mult (S3.01), long_signal ('S3.01'|'S4.04'),
               pivot_mode ('daily'|'weekly')
  filters      adx_threshold (F3_x), breadth_threshold (F8_x), btc_filter
  exits        sl=(mode,val) / sl_mode+sl_value, tp=(mode,val)/tp_mode+tp_value,
               max_hold, trailing ('none'|'atr_simple'|'atr_stepped') + mults,
               loss_streak_cooldown, cooldown

Search is GREEDY-staged (coordinate ascent): each stage explores its own grid on
top of the best params fixed by earlier stages — fewer combos than the full
"top-N x next-stage" product, which also guards against over-fitting on the small
samples the spec warns about. Every evaluated point is still recorded.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from exits import ExitConfig
from backtester_v3 import run_backtests
from scorer_v3 import per_asset_metrics, aggregate_strategies
from improvements.strategy_v5 import StrategyV5

# signal-param key -> (signal id it targets, kwarg name)
SIG_PARAM_MAP = {
    "nr_period": ("S1.11", "n"),
    "short_nr_period": ("S1.11", "n"),
    "vortex_period": ("S2.08", "n"),
    "natr_period": ("S4.04", "n"),
    "natr_high_thr": ("S4.04", "high_thr"),
    "supertrend_period": ("S3.01", "period"),
    "supertrend_mult": ("S3.01", "mult"),
}


# ---------------------------------------------------------------------------
def base_defaults(base) -> dict:
    ex = base.exit
    d = dict(entry_delay=0, entry_confirmation=None, exclude_assets=(),
             cooldown=None, loss_streak_cooldown=None,
             max_hold=ex.max_hold or 10, trailing="none")
    if ex.stop_loss_pct is not None:
        d["sl_mode"], d["sl_value"] = "fixed", ex.stop_loss_pct
    elif ex.atr_stop_mult is not None:
        d["sl_mode"], d["sl_value"] = "atr", ex.atr_stop_mult
    else:
        d["sl_mode"], d["sl_value"] = "fixed", -0.05
    if ex.take_profit_pct is not None:
        d["tp_mode"], d["tp_value"] = "fixed", ex.take_profit_pct
    elif ex.atr_tp_mult is not None:
        d["tp_mode"], d["tp_value"] = "atr", ex.atr_tp_mult
    else:
        d["tp_mode"], d["tp_value"] = "none", None
    return d


def _build_members(base, params):
    members = list(base.members)
    if params.get("long_signal"):
        members = [(params["long_signal"], d) if d == "LONG" else (sid, d)
                   for sid, d in members]
    if params.get("pivot_mode"):
        newsid = "S3.12W" if params["pivot_mode"] == "weekly" else "S3.12"
        members = [(newsid if sid in ("S3.12", "S3.12W") else sid, d)
                   for sid, d in members]
    present = {sid for sid, _ in members}
    mp: dict = {}
    for pk, (sid, kw) in SIG_PARAM_MAP.items():
        if params.get(pk) is not None and sid in present:
            mp.setdefault(sid, {})[kw] = params[pk]
    return tuple(members), mp


def _build_filters(base, params):
    filters = [f for f in base.filters]
    if params.get("adx_threshold") is not None:
        filters = [f for f in filters if not f.startswith("F3")]
        filters.append(f"F3_{int(params['adx_threshold'])}")
    if params.get("breadth_threshold") is not None:
        filters = [f for f in filters if not f.startswith("F8")]
        filters.append(f"F8_{int(round(params['breadth_threshold'] * 100))}")
    bf = params.get("btc_filter")
    if bf == "btc_5d_pos":
        filters.append("F9_5d")
    elif bf == "btc_20d_pos":
        filters.append("F9_20d")
    return tuple(filters)


def _build_exit(params):
    kw = dict(label="v5")
    if params.get("max_hold") is not None:
        kw["max_hold"] = int(params["max_hold"])
    sl_mode, sl_val = params.get("sl_mode", "fixed"), params.get("sl_value")
    if sl_mode == "atr" and sl_val is not None:
        kw["atr_stop_mult"] = float(sl_val)
    elif sl_val is not None:
        kw["stop_loss_pct"] = float(sl_val)
    tp_mode, tp_val = params.get("tp_mode", "fixed"), params.get("tp_value")
    if tp_mode == "atr" and tp_val is not None:
        kw["atr_tp_mult"] = float(tp_val)
    elif tp_mode == "fixed" and tp_val is not None:
        kw["take_profit_pct"] = float(tp_val)
    # tp_mode in ('none','trailing_only') -> no take-profit
    trailing = params.get("trailing", "none")
    if trailing == "atr_simple":
        m = float(params.get("trail_mult", 2.0))
        kw.update(stepped_trail_base=m, stepped_trail_tight=m, stepped_trail_at=10.0)
    elif trailing == "atr_stepped":
        kw.update(stepped_trail_base=float(params.get("trail_base", 2.0)),
                  stepped_trail_tight=float(params.get("trail_tight", 1.0)),
                  stepped_trail_at=float(params.get("trail_at", 0.05)))
    if params.get("loss_streak_cooldown") is not None:
        kw["skip_after_n_losses"] = int(params["loss_streak_cooldown"])
    if params.get("cooldown") is not None:
        kw["cooldown"] = int(params["cooldown"])
    return ExitConfig(**kw)


def build_candidate(base, params: dict, cid: str) -> StrategyV5:
    p = dict(params)
    if "sl" in p:
        p["sl_mode"], p["sl_value"] = p.pop("sl")
    if "tp" in p:
        p["tp_mode"], p["tp_value"] = p.pop("tp")
    if "min_atr_ratio" in p:                 # S1 grid: ATR(5)/ATR(20) guard -> FC2
        if p.pop("min_atr_ratio"):
            p["_add_fc2"] = True
    members, mp = _build_members(base, p)
    filters = list(_build_filters(base, p))
    if p.get("_add_fc2"):
        filters.append("FC2")
    return StrategyV5(
        id=cid, members=members, direction=base.direction, exit=_build_exit(p),
        filters=tuple(filters), combo_type=base.combo_type, group=base.group,
        entry_delay=int(p.get("entry_delay", 0)),
        entry_confirmation=p.get("entry_confirmation"),
        exclude_assets=tuple(p.get("exclude_assets", ())),
        member_params=mp)


# ---------------------------------------------------------------------------
def _score(cands, data, window, group_of):
    trades = run_backtests(cands, data, date_mask=window, show_progress=False)
    pa = per_asset_metrics(trades, data, window, group_of)
    agg = aggregate_strategies(pa)
    agg = agg.set_index("strategy_id") if not agg.empty else agg
    out = {}
    for c in cands:
        if not agg.empty and c.id in agg.index:
            r = agg.loc[c.id]
            out[c.id] = dict(ret=float(r["cross_asset_avg_return"]),
                             sharpe=float(r["cross_asset_sharpe"]),
                             pf=float(r["avg_profit_factor"]),
                             trades=int(r["total_trades"]),
                             assets=int(r["n_assets"]))
        else:
            out[c.id] = dict(ret=np.nan, sharpe=np.nan, pf=np.nan, trades=0, assets=0)
    return out


def _combos(stage: dict):
    keys = list(stage.keys())
    for vals in itertools.product(*[stage[k] for k in keys]):
        yield dict(zip(keys, vals))


def staged_search(base, stages, data, group_of, is_window):
    """Greedy-staged IS grid. Returns evaluated dict {cid: (params, is_metrics)}
    and the final best param set."""
    defaults = base_defaults(base)
    fixed: dict = {}
    evaluated: dict = {}
    n = [0]
    for stage in stages:
        cands, cmap = [], {}
        for combo in _combos(stage):
            p = {**defaults, **fixed, **combo}
            cid = f"{base.id}#g{n[0]}"; n[0] += 1
            c = build_candidate(base, p, cid)
            group_of[cid] = c.group
            cands.append(c)
            cmap[cid] = (p, combo)
        metrics = _score(cands, data, is_window, group_of)
        best, best_ret = None, -np.inf
        for cid, (p, combo) in cmap.items():
            m = metrics[cid]
            evaluated[cid] = (p, m)
            if np.isfinite(m["ret"]) and m["ret"] > best_ret:
                best, best_ret = combo, m["ret"]
        if best is not None:
            fixed.update(best)
    return evaluated, {**defaults, **fixed}
