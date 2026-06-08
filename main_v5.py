"""
main_v5.py — Phase 5: per-strategy deep parameter tuning pipeline.

For each of the 5 carriers: bespoke staged grid on IS -> IS-gated top-10 on OOS ->
keep only points that beat BOTH the v3 original AND the Phase-4 best -> winners get
walk-forward + random benchmark. Produces a v3/v4/v5 comparison and a holding-period
win-rate analysis (the 0~1 day diagnosis).

Usage:
  python main_v5.py                         # all 5 strategies, full pipeline
  python main_v5.py --strategy S1.11_S_aggr # one strategy
  python main_v5.py --stage                 # IS grids only (fast)
  python main_v5.py --compare               # v3/v4/v5 comparison table only
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import setup_console, DATA_DIR                       # noqa: E402
from data_loader import load_data, integrity_report            # noqa: E402
import spot_check as spot                                       # noqa: E402
import optimizer_v5 as o5                                       # noqa: E402
from improvements.grid_search import build_candidate            # noqa: E402
import dashboard_v5 as dash                                     # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results_v5")
RESULTS_V4 = os.path.join(ROOT, "results_v4")


def banner(m):
    print(f"\n{'=' * 70}\n  {m}\n{'=' * 70}")


def _safe_name(sid):
    return sid.replace("+", "_").replace(".", "").replace("/", "")


def _save(df, name, sub=""):
    d = os.path.join(RESULTS, sub) if sub else RESULTS
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    df.to_csv(p, index=False)
    return p


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--stage", action="store_true", help="IS grids only")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    banner("Phase 5: Per-Strategy Parameter Tuning (v5)")
    data = load_data(DATA_DIR)
    assert not integrity_report(data)["any_nan"].any(), "NaNs in OHLCV!"
    v4_best = o5.load_v4_best(RESULTS_V4)
    targets = ([args.strategy] if args.strategy else o5.V5_STRATEGIES)
    group_of = {}

    results = {}
    grid_frames = []
    for sid in targets:
        banner(f"Grid search — {sid}  (v4 best OOS {v4_best.get(sid, float('nan')):.1f}%)")
        r = o5.run_strategy(sid, data, group_of, v4_best.get(sid))
        results[sid] = r
        gdf = r["is_df"].copy(); gdf.insert(0, "base", sid)
        grid_frames.append(gdf)
        _save(r["is_df"], f"{_safe_name(sid)}_is_grid.csv", sub="grid_search")
        if not r["oos_df"].empty:
            _save(r["oos_df"], f"{_safe_name(sid)}_oos.csv", sub="oos_validation")
        top = r["is_df"].head(3)[["is_return", "is_trades", "desc"]]
        print("IS top3:")
        print(top.round(2).to_string(index=False))
        if r["best"] is not None:
            print(f"v5 best (beats v3 {r['base_oos']['ret']:.1f}% & v4 "
                  f"{r['v4_best_oos']:.1f}%): OOS {r['best']['oos_return']:.1f}%  → {r['best']['desc']}")
        else:
            print(f"v5 best: none beat v4 ({v4_best.get(sid, float('nan')):.1f}%) → keep v4")

    combined_grid = pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame()

    if args.stage:
        banner("IS grids saved (--stage)")
        print(f"📁 {os.path.join(RESULTS, 'grid_search')}")
        return

    # ---- v3 / v4 / v5 comparison -----------------------------------------
    rows = []
    best_cands, base_strats = {}, {}
    for sid, r in results.items():
        base_strats[sid] = r["base"]
        v5_ret = v5_sh = np.nan
        desc = "(none — keep v4)"
        improved = False
        if r["best"] is not None:
            v5_ret = float(r["best"]["oos_return"])
            v5_sh = float(r["best"]["oos_sharpe"])
            desc = r["best"]["desc"]
            improved = True
            best_cands[sid] = build_candidate(r["base"], r["best_params"], f"{sid}#v5")
        rows.append(dict(
            base_id=sid, v3_oos=r["base_oos"]["ret"], v3_sharpe=r["base_oos"]["sharpe"],
            v4_best_oos=r["v4_best_oos"], v5_best_oos=v5_ret, v5_sharpe=v5_sh,
            improved_over_v4=improved, v5_change=desc))
    comp = pd.DataFrame(rows)
    _save(comp, "comparison_v5.csv")

    banner("v3 / v4 / v5 comparison (OOS return %)")
    print(comp[["base_id", "v3_oos", "v4_best_oos", "v5_best_oos",
                "improved_over_v4", "v5_change"]].round(2).to_string(index=False))

    if args.compare:
        print(f"\n📁 CSVs in {RESULTS}")
        return

    # ---- holding-period WR (0~1 day diagnosis), base vs v5-best -----------
    banner("Holding-period win-rate — base vs v5-best")
    wr_frames = []
    for sid, r in results.items():
        strs = [r["base"]] + ([best_cands[sid]] if sid in best_cands else [])
        wr = o5.holding_period_wr(strs, data, group_of=group_of)
        wr["base_id"] = sid
        wr_frames.append(wr)
    trade_wr = pd.concat(wr_frames, ignore_index=True) if wr_frames else pd.DataFrame()
    _save(trade_wr, "trade_analysis_v5.csv")
    if not trade_wr.empty:
        piv = trade_wr.pivot_table(index="strategy_id", columns="bucket",
                                   values="wr", aggfunc="mean")
        order = ["0d", "1d", "2-3d", "4-5d", "6-10d", "10d+"]
        piv = piv.reindex(columns=[c for c in order if c in piv.columns])
        print((piv * 100).round(0).to_string())

    # ---- walk-forward + random on v5 winners -----------------------------
    wf_df = rnd_df = pd.DataFrame()
    winners = list(best_cands.values())
    if winners:
        banner(f"Walk-forward + random benchmark — {len(winners)} v5 winners")
        wf_df, rnd_df = o5.validate_v5_winners(winners, data, group_of)
        _save(wf_df, "walk_forward_v5.csv")
        _save(rnd_df, "random_benchmark_v5.csv")
        if not wf_df.empty:
            print(wf_df.round(2).to_string(index=False))
        if not rnd_df.empty:
            print(rnd_df.round(3).to_string(index=False))

    # ---- June 2026 spot --------------------------------------------------
    banner("June 2026 spot check (base + v5-best)")
    spot_strats = list(base_strats.values()) + winners
    sp = spot.spot_check(spot_strats, data)
    _save(sp, "june2026_spot_v5.csv")
    print(sp.head(20).to_string(index=False) if not sp.empty else "(no entries)")

    # ---- dashboard -------------------------------------------------------
    out = os.path.join(RESULTS, "dashboard_v5.html")
    dash.build_dashboard(out, data, combined_grid, comp, trade_wr, wf_df, rnd_df,
                         sp, base_strats, best_cands)
    print(f"\n📁 Dashboard: {out}")

    _summary(comp, wf_df, rnd_df)


def _summary(comp, wf_df, rnd_df):
    banner("Phase 5: Per-Strategy Parameter Tuning — Results")
    wf_pos = {}
    if wf_df is not None and not wf_df.empty:
        wf_pos = dict(zip(wf_df["strategy_id"], wf_df["windows_positive"]))
    p_of = {}
    if rnd_df is not None and not rnd_df.empty:
        p_of = dict(zip(rnd_df["strategy_id"], rnd_df["p_value"]))
    for _, r in comp.iterrows():
        sid = r["base_id"]
        print(f"\n🏆 {sid}")
        print(f"   v3 original: OOS {r['v3_oos']:+6.1f}%  Sharpe {r['v3_sharpe']:.2f}")
        print(f"   v4 best:     OOS {r['v4_best_oos']:+6.1f}%")
        if r["improved_over_v4"]:
            wf = wf_pos.get(f"{sid}#v5")
            p = p_of.get(f"{sid}#v5")
            extra = (f"  WF {int(wf)}/5" if wf is not None else "") + \
                    (f"  p={p:.3f}" if p is not None and np.isfinite(p) else "")
            print(f"   v5 best:     OOS {r['v5_best_oos']:+6.1f}%  Sharpe {r['v5_sharpe']:.2f}{extra}")
            print(f"   Key change:  {r['v5_change']}")
        else:
            print(f"   v5 best:     no point beat v4 — keep Phase-4 winner")
    print(f"\n📁 Dashboard: {os.path.join(RESULTS, 'dashboard_v5.html')}")


if __name__ == "__main__":
    main()
