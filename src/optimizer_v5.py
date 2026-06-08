"""
optimizer_v5.py — per-strategy deep parameter tuning (Phase 5).

For each of the 5 carriers, a bespoke staged grid (grid_search.staged_search) is
run on IS, the IS-gated top-10 are validated on OOS, and the best v5 point is the
one that beats BOTH the v3 original AND the Phase-4 best on OOS. Winners then get
5-window walk-forward + 1000x random benchmark — the same v3 framework.

The grids encode each strategy's own diagnosis (0~1 day whipsaw, asset bleeders,
hold-length edge), per crypto_alpha_v5_per_strategy_tuning.md.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import oos_validator as oosv
import random_benchmark as rb
from scorer_v3 import per_asset_metrics

from improvements import reconstruct as RC
from improvements import signals_v5            # noqa: F401  (registers F3_*/F8_*/F9_*/S3.12W)
from improvements import whipsaw_guard         # noqa: F401  (registers FC2 used by S1 grid)
from improvements.grid_search import (staged_search, build_candidate,
                                      base_defaults, _score)

IS, OOS = oosv.IS_WINDOW, oosv.OOS_WINDOW

# --------------------------------------------------------------------------
# per-strategy staged grids (greedy coordinate ascent)
# --------------------------------------------------------------------------
GRIDS = {
    # Strategy 1 — Tier S: asymmetric ATR centred (A4 made +78% in Phase 4)
    "SEQ_S1.11_S+S4.04_L": [
        {"sl": [("atr", 1.0), ("atr", 1.5), ("atr", 2.0), ("fixed", -0.05)],
         "tp": [("atr", 3.0), ("atr", 4.0), ("atr", 5.0), ("atr", 7.0), ("fixed", 0.10)],
         "max_hold": [10, 14, 21]},
        {"entry_delay": [0, 1], "short_nr_period": [5, 7, 9]},
        {"trailing": ["none", "atr_stepped"], "trail_base": [2.0],
         "trail_tight": [1.0], "trail_at": [0.05]},
        {"min_atr_ratio": [None, 1.0], "cooldown": [3, 5]},
    ],
    # Strategy 2 — S3.12: long hold + TP-removal/trailing, breadth, weekly pivot
    "S3.12_S_aggr_F8": [
        {"sl": [("fixed", -0.07), ("fixed", -0.10), ("atr", 2.0), ("atr", 2.5)],
         "tp": [("fixed", 0.15), ("fixed", 0.25), ("none", None)],
         "max_hold": [14, 21, 28, 42]},
        {"trailing": ["none", "atr_simple", "atr_stepped"],
         "trail_mult": [2.0, 2.5, 3.0], "trail_base": [2.5], "trail_tight": [1.0],
         "trail_at": [0.07]},
        {"breadth_threshold": [0.30, 0.40, 0.50], "pivot_mode": ["daily", "weekly"],
         "entry_delay": [0, 1]},
    ],
    # Strategy 3 — S2.08_F3: SMALL sample (105) -> keep grid small, conservative
    "S2.08_L_aggr_F3": [
        {"vortex_period": [10, 14, 18], "adx_threshold": [15, 20, 25],
         "sl": [("fixed", -0.07), ("atr", 2.0)], "max_hold": [14, 21]},
        {"tp": [("fixed", 0.15), ("atr", 4.0)], "trailing": ["none", "atr_simple"],
         "trail_mult": [2.0], "btc_filter": [None, "btc_5d_pos"]},
    ],
    # Strategy 4 — S1.11_S: entry timing + wide SL + DOGE bleeder + loss streak
    "S1.11_S_aggr": [
        {"sl": [("fixed", -0.07), ("fixed", -0.10), ("fixed", -0.12), ("atr", 2.0), ("atr", 2.5)],
         "tp": [("fixed", 0.15), ("atr", 4.0)], "max_hold": [14, 21, 28]},
        {"entry_delay": [0, 1, 2],
         "entry_confirmation": [None, "close_below_open", "gap_down"],
         "nr_period": [5, 7, 9]},
        {"exclude_assets": [(), ("DOGE",), ("DOGE", "BTC")],
         "loss_streak_cooldown": [None, 3, 5], "cooldown": [3, 5]},
        {"trailing": ["none", "atr_stepped"], "trail_base": [2.0, 2.5],
         "trail_tight": [1.0], "trail_at": [0.05]},
    ],
    # Strategy 5 — SEQ+S3.01: long-leg swap (S4.04 made WF 5/5) + XRP bleeder
    "SEQ_S1.11_S+S3.01_L": [
        {"long_signal": ["S3.01", "S4.04"], "short_nr_period": [5, 7, 9],
         "sl": [("fixed", -0.05), ("atr", 1.5)], "max_hold": [7, 10, 14]},
        {"natr_high_thr": [0.85, 0.90], "tp": [("fixed", 0.10), ("atr", 4.0)]},
        {"exclude_assets": [(), ("XRP",)], "cooldown": [3, 5]},
    ],
}

V5_STRATEGIES = list(GRIDS.keys())

_DESC_KEYS = ["sl_mode", "sl_value", "tp_mode", "tp_value", "max_hold",
              "entry_delay", "entry_confirmation", "trailing", "trail_mult",
              "trail_base", "trail_tight", "trail_at", "cooldown",
              "loss_streak_cooldown", "exclude_assets", "adx_threshold",
              "breadth_threshold", "btc_filter", "vortex_period", "nr_period",
              "short_nr_period", "long_signal", "pivot_mode", "natr_period",
              "natr_high_thr", "supertrend_period", "supertrend_mult", "min_atr_ratio"]


def _desc(base, p) -> str:
    d = base_defaults(base)
    out = []
    for k in _DESC_KEYS:
        if k in p and p.get(k) is not None and p.get(k) != d.get(k):
            out.append(f"{k}={p[k]}")
    return "; ".join(out) if out else "(=base)"


def _eval_base(base, data, group_of):
    group_of[base.id] = base.group
    is_m = _score([base], data, IS, group_of)[base.id]
    oos_m = _score([base], data, OOS, group_of)[base.id]
    return is_m, oos_m


def run_strategy(base_id, data, group_of, v4_best_oos=None):
    base = RC.reconstruct(base_id)
    evaluated, best_params = staged_search(base, GRIDS[base_id], data, group_of, IS)
    base_is, base_oos = _eval_base(base, data, group_of)

    rows = []
    for cid, (p, m) in evaluated.items():
        rows.append(dict(id=cid, is_return=m["ret"], is_sharpe=m["sharpe"],
                         is_pf=m["pf"], is_trades=m["trades"], desc=_desc(base, p)))
    is_df = pd.DataFrame(rows).sort_values("is_return", ascending=False).reset_index(drop=True)

    # IS-gated top-10 -> OOS
    gated = is_df[is_df["is_return"] > base_is["ret"]].head(10)
    cands = [build_candidate(base, evaluated[cid][0], cid) for cid in gated["id"]]
    for c in cands:
        group_of[c.id] = c.group
    oos_metrics = _score(cands, data, OOS, group_of) if cands else {}
    oos_rows = []
    for cid in gated["id"]:
        p, m = evaluated[cid]
        om = oos_metrics[cid]
        oos_rows.append(dict(id=cid, is_return=m["ret"], oos_return=om["ret"],
                             oos_sharpe=om["sharpe"], oos_pf=om["pf"],
                             oos_trades=om["trades"], desc=_desc(base, p)))
    oos_df = (pd.DataFrame(oos_rows).sort_values("oos_return", ascending=False)
              .reset_index(drop=True) if oos_rows else pd.DataFrame())

    thresh = base_oos["ret"]
    if v4_best_oos is not None and np.isfinite(v4_best_oos):
        thresh = max(thresh, v4_best_oos)
    best = None
    if not oos_df.empty:
        win = oos_df[oos_df["oos_return"] > thresh]
        if not win.empty:
            best = win.iloc[0]
    best_p = evaluated[best["id"]][0] if best is not None else None
    return dict(base=base, is_df=is_df, oos_df=oos_df, base_is=base_is,
                base_oos=base_oos, best=best, best_params=best_p,
                v4_best_oos=v4_best_oos)


# --------------------------------------------------------------------------
def load_v4_best(results_v4_dir) -> dict:
    """{base_id: v4_best_oos_return} from Phase-4 output, if available."""
    path = os.path.join(results_v4_dir, "best_per_base.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["base_id"], df["best_oos_return"]))


def validate_v5_winners(winner_cands, data, group_of):
    """Walk-forward + random benchmark on the v5 winner StrategyV5 objects."""
    if not winner_cands:
        return pd.DataFrame(), pd.DataFrame()
    for c in winner_cands:
        group_of[c.id] = c.group
    wf = oosv.walk_forward(winner_cands, data, group_of, show_progress=False)
    trades = oosv.run_backtests(winner_cands, data, date_mask=OOS, show_progress=False)
    pa = per_asset_metrics(trades, data, OOS, group_of)
    rnd = rb.random_benchmark(winner_cands, data, pa, trades=trades,
                              date_mask=OOS, n_sims=1000)
    return wf, rnd


def holding_period_wr(strategies, data, window=OOS, group_of=None):
    """Per-strategy win-rate by holding-day bucket (the 0~1 day diagnosis)."""
    trades = oosv.run_backtests(list(strategies), data, date_mask=window,
                                show_progress=False)
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    buckets = [(-1, 0, "0d"), (0, 1, "1d"), (1, 3, "2-3d"),
               (3, 5, "4-5d"), (5, 10, "6-10d"), (10, 999, "10d+")]

    def bucket(h):
        for lo, hi, lab in buckets:
            if lo < h <= hi:
                return lab
        return "10d+"
    t["bucket"] = t["holding_days"].map(bucket)
    g = t.groupby(["strategy_id", "bucket"]).agg(
        n=("pnl_pct", "size"),
        wr=("pnl_pct", lambda s: float((s > 0).mean())),
        avg=("pnl_pct", lambda s: float(s.mean() * 100))).reset_index()
    return g
