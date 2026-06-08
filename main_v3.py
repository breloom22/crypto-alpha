"""
main_v3.py — Crypto Alpha Strategy Discovery v3 pipeline orchestrator.

Stages:
  1. load data + integrity gate
  2. populate signal registry (S1-S8)
  3. generate Layer 1 (+2) strategies, IS backtest, score
  4. Layer 3 (filters) / 4 (AND) / 5 (SEQ) on IS-top, re-score
  5. OOS validation + walk-forward on final IS-top
  6. random-entry benchmark, parameter sensitivity, June-2026 spot check
  7. dashboard + CSVs + terminal summary

Usage:
  python main_v3.py                 # full pipeline
  python main_v3.py --quick         # Layer 1 only, no OOS/WF/random/sensitivity
  python main_v3.py --layers 1,2    # stop after the given layers (IS only)
  python main_v3.py --spot-only     # June-2026 spot check on a quick IS-top
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import setup_console, DATA_DIR, RESULTS_DIR, ensure_dirs   # noqa: E402
from data_loader import load_data, integrity_report                  # noqa: E402
from signals import registry                                         # noqa: E402
from strategy import StrategyGenerator                               # noqa: E402
from filters import FILTERS                                          # noqa: E402
import backtester_v3 as bt                                           # noqa: E402
import scorer_v3 as sc                                               # noqa: E402
import oos_validator as oos                                          # noqa: E402
import random_benchmark as rb                                        # noqa: E402
import sensitivity as sens_mod                                       # noqa: E402
import spot_check as spot                                            # noqa: E402
import dashboard_v3 as dash                                          # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v3")


def _save(df, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    return path


def _strat_map(*lists):
    m = {}
    for lst in lists:
        for s in lst:
            m[s.id] = s
    return m


def _top_objects(ranked, smap, n, direction=None):
    out = []
    for sid in ranked["strategy_id"]:
        s = smap.get(sid)
        if s is None:
            continue
        if direction and s.direction != direction:
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


def banner(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}")


def main():
    setup_console()
    ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--layers", default=None, help="e.g. '1,2'")
    ap.add_argument("--spot-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    banner("Crypto Alpha Strategy Discovery v3")
    data = load_data(DATA_DIR)
    rep = integrity_report(data)
    print(rep.to_string(index=False))
    assert not rep["any_nan"].any(), "NaNs in OHLCV!"

    specs = list(registry.load_all().values())
    print(f"\nSignals registered: {len(specs)} "
          f"({', '.join(sorted({s.group for s in specs}))})")
    gen = StrategyGenerator(specs)

    # ---- layer selection --------------------------------------------------
    explicit_layers = args.layers is not None
    requested = (set(args.layers.split(",")) if explicit_layers
                 else {"1", "2", "3", "4", "5"})
    if args.quick or args.spot_only:
        requested = {"1"}

    # ---- Layer 1 (+2) -----------------------------------------------------
    l1 = gen.generate_layer1()
    strategies = list(l1)
    if "2" in requested:
        strategies += gen.generate_layer2(l1)
    smap = _strat_map(strategies)
    group_of = {s.id: s.group for s in strategies}
    banner(f"Layer 1+2: {len(strategies)} strategies — IS backtest")
    is_trades = bt.run_backtests(strategies, data, date_mask=oos.IS_WINDOW)
    pa_is, ranked = sc.score_trades(is_trades, data, oos.IS_WINDOW, group_of)
    print(f"IS trades: {len(is_trades):,}; scored strategies: {len(ranked)}")

    # ---- Layers 3/4/5 (post-IS) ------------------------------------------
    do345 = bool({"3", "4", "5"} & requested)
    if do345 and not ranked.empty:
        top30 = _top_objects(ranked, smap, 30)
        top20 = _top_objects(ranked, smap, 20)
        top_short = _top_objects(ranked, smap, 5, "SHORT")
        top_long = _top_objects(ranked, smap, 5, "LONG")
        l3 = gen.generate_layer3(top30, list(FILTERS.keys()))
        l4 = gen.generate_layer4(top20)
        l5 = gen.generate_layer5(top_short, top_long)
        extra = l3 + l4 + l5
        smap.update(_strat_map(extra))
        group_of.update({s.id: s.group for s in extra})
        banner(f"Layer 3/4/5: +{len(extra)} strategies (F={len(l3)}, AND={len(l4)}, SEQ={len(l5)}) — IS backtest")
        extra_trades = bt.run_backtests(extra, data, date_mask=oos.IS_WINDOW)
        is_trades = pd.concat([is_trades, extra_trades], ignore_index=True)
        pa_is, ranked = sc.score_trades(is_trades, data, oos.IS_WINDOW, group_of)
        print(f"IS trades (all layers): {len(is_trades):,}; scored: {len(ranked)}")

    _save(is_trades, "all_trades.csv")
    _save(ranked, "strategy_scores_is.csv")

    banner("IS Top 10")
    cols = ["rank", "strategy_id", "category", "composite_score",
            "cross_asset_median_return", "cross_asset_sharpe", "avg_win_rate",
            "avg_profit_factor", "total_trades"]
    print(ranked[cols].head(10).round(3).to_string(index=False))

    if args.layers and not (args.quick or args.spot_only):
        print("\n(stopped after requested layers)")
        return

    final_top = _top_objects(ranked, smap, 30)

    # ---- spot-only shortcut ----------------------------------------------
    if args.spot_only:
        banner("June 2026 Spot Check (quick IS-top)")
        sp = spot.spot_check(final_top, data)
        _save(sp, "june2026_spot.csv")
        print(sp.head(30).to_string(index=False) if not sp.empty else "(no entries in window)")
        return

    if args.quick:
        banner("Quick mode — dashboard from IS only")
        empty = pd.DataFrame()
        sp = spot.spot_check(final_top, data)
        _save(sp, "june2026_spot.csv")
        dash.build_dashboard(os.path.join(RESULTS, "dashboard.html"), data, ranked,
                             empty, empty, empty, empty, pa_is, is_trades, sp)
        print(f"Dashboard: {os.path.join(RESULTS, 'dashboard.html')}")
        return

    # ---- OOS validation + walk-forward -----------------------------------
    banner(f"OOS Validation — final IS-top {len(final_top)}")
    oos_df = oos.oos_validate(final_top, data, group_of)
    _save(oos_df, "strategy_scores_oos.csv")
    survivors = [smap[sid] for sid in oos_df[oos_df["survived"]]["strategy_id"] if sid in smap]
    print(oos_df.head(10)[["strategy_id", "is_return", "oos_return", "oos_sharpe",
                           "oos_pf", "survived", "valid", "robust"]].round(3).to_string(index=False))
    print(f"\nSurvivors (OOS return>0): {len(survivors)}/{len(final_top)}")

    banner("Walk-Forward (5 windows)")
    wf_df = oos.walk_forward(final_top, data, group_of)
    _save(wf_df, "walk_forward.csv")
    print(wf_df.head(10).round(2).to_string(index=False))
    print(f"WF validated (>=3/5): {int(wf_df['wf_validated'].sum())}/{len(wf_df)}")

    # ---- random benchmark + sensitivity + spot ---------------------------
    bench_targets = (survivors if survivors else final_top[:10])[:12]
    banner(f"Random-Entry Benchmark — {len(bench_targets)} strategies x 1000 sims")
    bench_trades = bt.run_backtests(bench_targets, data, date_mask=oos.OOS_WINDOW,
                                    show_progress=False)
    pa_oos = sc.per_asset_metrics(bench_trades, data, oos.OOS_WINDOW, group_of)
    random_df = rb.random_benchmark(bench_targets, data, pa_oos, trades=bench_trades,
                                    date_mask=oos.OOS_WINDOW, n_sims=1000)
    _save(random_df, "random_benchmark.csv")
    if not random_df.empty:
        print(random_df.round(3).to_string(index=False))

    banner("Parameter Sensitivity (top 10)")
    sens_df = sens_mod.sensitivity(final_top, data, group_of, mask=oos.IS_WINDOW, top_n=10)
    _save(sens_df, "sensitivity.csv")
    if not sens_df.empty:
        stab = sens_df.groupby("strategy_id")["stable"].mean()
        print(f"Mean knob-stability across top-10: {stab.mean():.0%}")

    banner("June 2026 Spot Check")
    sp = spot.spot_check(bench_targets, data)
    _save(sp, "june2026_spot.csv")
    print(sp.head(30).to_string(index=False) if not sp.empty else "(no entries in window)")

    # ---- dashboard --------------------------------------------------------
    group_perf = ranked.groupby("group").agg(
        cross_asset_avg_return=("cross_asset_avg_return", "mean"),
        cross_asset_sharpe=("cross_asset_sharpe", "mean")).reset_index()
    group_perf = group_perf.rename(columns={})  # keep names dashboard expects
    dash.build_dashboard(os.path.join(RESULTS, "dashboard.html"), data, ranked,
                         oos_df, wf_df, random_df, sens_df, pa_is, is_trades, sp,
                         group_perf=group_perf.rename(columns={
                             "cross_asset_avg_return": "ret", "cross_asset_sharpe": "sharpe"}))

    _terminal_summary(ranked, oos_df, wf_df, random_df, sp, sc.best_by_category(ranked))
    print(f"\n📁 Dashboard: {os.path.join(RESULTS, 'dashboard.html')}")


def _terminal_summary(ranked, oos_df, wf_df, random_df, sp, best_cat):
    banner("Crypto Alpha Strategy Discovery v3 — Results")
    if not ranked.empty:
        print("\n🏆 IS Top 5:")
        for _, r in ranked.head(5).iterrows():
            print(f" #{int(r['rank'])} [{r['strategy_id']}] {r['category']}  "
                  f"med.ret {r['cross_asset_median_return']:+.1f}%  sharpe {r['cross_asset_sharpe']:.2f}  "
                  f"WR {r['avg_win_rate']:.0%}  PF {r['avg_profit_factor']:.2f}  n={int(r['total_trades'])}")
    if not oos_df.empty:
        surv = int(oos_df["survived"].sum())
        rob = int(oos_df["robust"].sum())
        print(f"\n✅ OOS: survivors {surv}/{len(oos_df)}, robust {rob}/{len(oos_df)}")
        for _, r in oos_df[oos_df["survived"]].head(5).iterrows():
            tag = "✓ Robust" if r["robust"] else ("✓ Valid" if r["valid"] else "")
            print(f"   [{r['strategy_id']}] OOS {r['oos_return']:+.1f}%  sharpe {r['oos_sharpe']:.2f} {tag}")
    if not wf_df.empty:
        print(f"\n🔁 Walk-Forward validated (>=3/5): {int(wf_df['wf_validated'].sum())}/{len(wf_df)}")
    if random_df is not None and not random_df.empty:
        sig = int(random_df["significant"].sum())
        print(f"\n🎲 vs Random Entry: {sig}/{len(random_df)} beat the 95th pctile (p<0.05)")
        for _, r in random_df[random_df["significant"]].head(5).iterrows():
            print(f"   [{r['strategy_id']}] {r['actual_return']:+.1f}% vs random p95 {r['random_p95']:+.1f}%")
    if not best_cat.empty:
        print("\n🏷️  Best by category:")
        for _, r in best_cat.iterrows():
            print(f"   {r['category']:14} [{r['strategy_id']}] {r['cross_asset_median_return']:+.1f}%")
    if sp is not None and not sp.empty:
        print(f"\n📅 June 2026: {len(sp)} entries in {sp['symbol'].nunique()} assets")


if __name__ == "__main__":
    main()
