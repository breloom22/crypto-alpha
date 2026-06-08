"""
main_v4.py — Phase 4: Strategy Optimization pipeline.

Takes the 7 validation-passing v3 strategies and tries to *improve their
trade-level behaviour* (exits / filters / SEQ-long-leg / asset universe) without
changing their entry signals, then re-validates on the SAME IS/OOS/walk-forward/
random framework. An improvement is adopted ONLY if it beats the original on OOS
after passing the IS gate.

Usage:
  python main_v4.py                 # full pipeline (singles + combos + portfolio)
  python main_v4.py --module A      # only Improvement A (singles), print compare
  python main_v4.py --portfolio     # only the portfolio (Improvement G)
  python main_v4.py --compare       # singles -> comparison table only (no WF/random)
  python main_v4.py --quick         # full compare but skip WF/random/dashboard
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import setup_console, DATA_DIR                       # noqa: E402
from data_loader import load_data, integrity_report            # noqa: E402
import oos_validator as oosv                                    # noqa: E402
import spot_check as spot                                       # noqa: E402
import optimizer as opt                                         # noqa: E402
from improvements import reconstruct as RC                      # noqa: E402
from improvements import asset_weighting as D                   # noqa: E402
import dashboard_v4 as dash                                     # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v4")


def banner(msg):
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}")


def _save(df, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    df.to_csv(path, index=False)
    return path


def _winner_candidates(best_df, comparison, all_cands_by_id):
    """Candidate objects for the adopted best-per-base winners (for WF/random)."""
    out = []
    for _, r in best_df[best_df["improved"]].iterrows():
        c = all_cands_by_id.get(r["best_id"])
        if c is not None:
            out.append(c)
    return out


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=None, help="restrict to one module A-F")
    ap.add_argument("--portfolio", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    banner("Phase 4: Strategy Optimization (v4)")
    data = load_data(DATA_DIR)
    rep = integrity_report(data)
    assert not rep["any_nan"].any(), "NaNs in OHLCV!"
    reduced = D.reduced_universe(data)
    bases = RC.reconstruct_targets()
    group_of = {b.id: b.group for b in bases}
    print(f"Bases: {len(bases)} | universe {list(data)} | reduced {list(reduced)}")

    # ---- portfolio-only shortcut -----------------------------------------
    if args.portfolio:
        banner("Portfolios (Improvement G) — OOS")
        pf_df, books = opt.run_portfolios(data)
        _save(pf_df, "portfolio_results.csv")
        print(pf_df.round(2).to_string(index=False))
        return

    # ---- generate + evaluate single-module candidates --------------------
    singles = opt.generate_singles(bases)
    if args.module:
        m = args.module.upper()
        singles = [c for c in singles if c.module == m]
        print(f"(restricted to module {m}: {len(singles)} candidates)")
    banner(f"IS+OOS evaluation — {len(singles)} single-module candidates + {len(bases)} bases")
    ev_singles = opt.evaluate(singles, bases, data, reduced, group_of)

    # ---- combos (skip when a single module is isolated) ------------------
    combos = []
    if not args.module:
        combos = opt.generate_combos(bases, singles, ev_singles)
        banner(f"IS+OOS evaluation — {len(combos)} promising combos")
        ev_combos = opt.evaluate(combos, bases, data, reduced, group_of)
        eval_df = pd.concat([ev_singles, ev_combos[ev_combos["module"] != "base"]],
                            ignore_index=True)
    else:
        eval_df = ev_singles

    all_cands_by_id = {c.id: c for c in (singles + combos)}

    # ---- comparison + module effectiveness -------------------------------
    comparison = opt.build_comparison(eval_df)
    mod_eff = opt.module_effectiveness(comparison)
    best_df = opt.best_per_base(comparison, eval_df, bases)

    # restricted modes (--module/--compare) see only a subset of candidates, so
    # they MUST NOT clobber the canonical full-run CSVs (Phase 5 reads them).
    sfx = (f"_module{args.module.upper()}" if args.module
           else ("_compare" if args.compare else ""))
    _save(eval_df, f"improvement_scores{sfx}.csv")
    _save(comparison, f"comparison_table{sfx}.csv")
    _save(mod_eff, f"module_effectiveness{sfx}.csv")
    _save(best_df, f"best_per_base{sfx}.csv")

    banner("Module effectiveness (avg OOS return change, pp)")
    print(mod_eff.round(2).to_string(index=False))

    banner("Best improvement per base (IS-gated, OOS-adopted)")
    print(best_df.round(2).to_string(index=False))

    if args.compare or args.module:
        banner("Top adopted improvements (by OOS delta)")
        adopted = comparison[(comparison["verdict"] == "adopt")
                             & comparison["is_gate_pass"]]
        cols = ["strategy_id", "module", "base_oos_return", "cand_oos_return",
                "delta_oos_return", "cand_oos_sharpe", "verdict"]
        print(adopted[cols].head(20).round(2).to_string(index=False))
        print(f"\n📁 CSVs in {RESULTS}")
        return

    # ---- walk-forward + random on the winners ----------------------------
    winners = _winner_candidates(best_df, comparison, all_cands_by_id)
    wf_df = rnd_df = pd.DataFrame()
    if not args.quick and winners:
        banner(f"Walk-forward + random benchmark — {len(winners)} winners")
        wf_df, rnd_df = opt.validate_winners(winners, data, reduced, group_of)
        _save(wf_df, "walk_forward_v4.csv")
        _save(rnd_df, "random_benchmark_v4.csv")
        if not wf_df.empty:
            print(wf_df.round(2).to_string(index=False))
        if not rnd_df.empty:
            print(rnd_df.round(3).to_string(index=False))

    # ---- portfolios ------------------------------------------------------
    banner("Portfolios (Improvement G) — OOS")
    pf_df, books = opt.run_portfolios(data)
    _save(pf_df, "portfolio_results.csv")
    print(pf_df.round(2).to_string(index=False))

    # ---- June 2026 spot check (bases + best improved) --------------------
    banner("June 2026 spot check")
    spot_strats = list(bases) + [all_cands_by_id[r["best_id"]].strategy
                                 for _, r in best_df[best_df["improved"]].iterrows()
                                 if r["best_id"] in all_cands_by_id]
    sp = spot.spot_check(spot_strats, data)
    _save(sp, "june2026_spot_v4.csv")
    print(sp.head(25).to_string(index=False) if not sp.empty else "(no entries)")

    # ---- dashboard -------------------------------------------------------
    if not args.quick:
        best_strats = {r["base_id"]: all_cands_by_id[r["best_id"]].strategy
                       for _, r in best_df[best_df["improved"]].iterrows()
                       if r["best_id"] in all_cands_by_id}
        out = os.path.join(RESULTS, "dashboard_v4.html")
        dash.build_dashboard(out, data, eval_df, comparison, mod_eff, best_df,
                             wf_df, rnd_df, pf_df, books, sp, best_strats=best_strats)
        print(f"\n📁 Dashboard: {out}")

    _terminal_summary(eval_df, comparison, mod_eff, best_df, pf_df, rnd_df)


def _terminal_summary(eval_df, comparison, mod_eff, best_df, pf_df, rnd_df):
    banner("Phase 4: Strategy Optimization — Results")
    n_cand = int((eval_df["module"] != "base").sum())
    print(f"\n📊 Improvements tested: {n_cand} variants of {best_df.shape[0]} base strategies\n")
    print("🔬 Module effectiveness (avg OOS return change):")
    names = {"A": "ATR SL/TP", "B": "Break-Even", "C": "Whipsaw Guard",
             "D": "Asset Select", "E": "Alt LONG", "F": "Stepped Trail",
             "combo": "Combos"}
    for _, r in mod_eff.iterrows():
        print(f"  {r['module']:5} {names.get(r['module'], ''):14} "
              f"{r['avg_delta_oos']:+6.1f} pp   "
              f"(adopt {r['n_adopt']}/{r['n_candidates']})")
    print("\n🏆 Best improvements (vs original):")
    for _, r in best_df.iterrows():
        tag = f"[{r['best_id'].replace(r['base_id'], '').lstrip('+')}]" if r["improved"] else "[no improvement — keep original]"
        print(f"  {r['base_id']} (Tier {r['tier']}):")
        print(f"    Original: OOS {r['base_oos_return']:+5.1f}%  Sharpe {r['base_oos_sharpe']:.2f}")
        print(f"    Best v4:  OOS {r['best_oos_return']:+5.1f}%  Sharpe {r['best_oos_sharpe']:.2f}  {tag}")
        if r["improved"]:
            print(f"    Change:   {r['delta']:+.1f}pp")
    if pf_df is not None and not pf_df.empty:
        print("\n📦 Portfolio results (OOS):")
        for _, r in pf_df.iterrows():
            if "error" in r and isinstance(r.get("error"), str):
                continue
            print(f"  {r['name']:12} avg-asset {r['avg_asset_return']:+6.1f}%  "
                  f"Sharpe {r['sharpe']:.2f}  MaxDD {r['max_drawdown']:.1f}%  n={int(r['n_trades'])}")
    if rnd_df is not None and not rnd_df.empty:
        sig = int(rnd_df["significant"].sum())
        print(f"\n🎲 vs Random Entry: {sig}/{len(rnd_df)} winners stay significant (p<0.05)")


if __name__ == "__main__":
    main()
