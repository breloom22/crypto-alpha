"""
main.py — end-to-end crypto crash-detector pipeline.

    python main.py                  # full pipeline (downloads data if missing)
    python main.py --download       # force re-download
    python main.py --quick          # skip in/out-sample + sensitivity (faster)

Outputs to results/:
    crash_events.csv, crash_clusters.csv, simultaneous_crashes.csv,
    backtest_full.csv, individual_scores.csv, in_out_sample.csv,
    sensitivity.csv, ensemble_scores.csv, dashboard.html
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils import setup_console, ensure_dirs, DATA_DIR, RESULTS_DIR  # noqa: E402


def _rp(name):
    return os.path.join(RESULTS_DIR, name)


def run(download=False, quick=False):
    setup_console()
    ensure_dirs()
    import pandas as pd
    from data_loader import load_data, download_all, integrity_report
    from crash_detector import detect_all, crash_starts_by_type, save as save_crashes
    from indicators.base import build_indicators
    from backtester import run_backtests, evaluation_base_rate
    from scorer import (compute_composite, aggregate_indicators, best_performers,
                        in_out_sample, sensitivity)
    from ensemble import search as ensemble_search
    from visualizer import build_dashboard

    bar = "═" * 64

    # ---- Phase 0: data -----------------------------------------------------
    print(f"\n{bar}\n  Phase 0 · Data\n{bar}")
    data = (download_all(data_dir=DATA_DIR) if download
            else load_data(DATA_DIR, download_if_missing=True))
    rep = integrity_report(data)
    print(rep.to_string(index=False))
    assert not rep["any_nan"].any(), "integrity gate failed: NaNs in OHLCV"
    period = f"{rep['start'].min()} ~ {rep['end'].max()}"

    # ---- Phase 1: crash events --------------------------------------------
    print(f"\n{bar}\n  Phase 1 · Crash events\n{bar}")
    events, clusters, simultaneous = detect_all(data)
    save_crashes(events, clusters, simultaneous, RESULTS_DIR)
    starts = crash_starts_by_type(events, clusters)
    print(f"event days: {len(events)} · crash legs: {len(clusters)} · "
          f"simultaneous-crash weeks: {len(simultaneous)}")
    assert len(clusters) > 0, "integrity gate failed: no crash events found"

    # ---- Phase 2: indicators ----------------------------------------------
    print(f"\n{bar}\n  Phase 2 · Indicators\n{bar}")
    reg = build_indicators()
    print(f"built {len(reg)} indicators across 5 categories")
    assert len(reg) == 35, f"integrity gate failed: expected 35 indicators, got {len(reg)}"

    # ---- Phase 3: backtest -------------------------------------------------
    print(f"\n{bar}\n  Phase 3 · Backtest\n{bar}")
    bt = run_backtests(data, reg, starts)
    bt.to_csv(_rp("backtest_full.csv"), index=False)
    print(f"backtested {bt['indicator'].nunique()} indicators x {bt['symbol'].nunique()} "
          f"assets x {bt['crash_type'].nunique()} crash-types = {len(bt)} rows")

    # ---- Phase 4: scoring & ranking ---------------------------------------
    print(f"\n{bar}\n  Phase 4 · Scoring & ranking\n{bar}")
    base_rate = evaluation_base_rate(data, starts)        # random-signal precision
    scores = compute_composite(aggregate_indicators(bt))
    scores["base_rate"] = base_rate
    scores["precision_lift"] = scores["precision"] / base_rate if base_rate else float("nan")
    scores.to_csv(_rp("individual_scores.csv"), index=False)
    best = best_performers(bt)
    print(f"Base rate (P[crash in next {21}d] for a random signal): {base_rate:.3f}")
    print(f"Top indicator precision {scores['precision'].iloc[0]:.3f} "
          f"= {scores['precision_lift'].iloc[0]:.2f}x base rate")
    print("Best per category:")
    print(best["by_category"].round(3).to_string(index=False))

    # ---- Phase 6/7: validation & sensitivity ------------------------------
    if not quick:
        print(f"\n{bar}\n  Phase 6/7 · In/Out-of-sample + sensitivity\n{bar}")
        io = in_out_sample(data, reg, starts)
        io.to_csv(_rp("in_out_sample.csv"), index=False)
        stable = (io["rank_delta"].abs() <= 5).mean()
        print(f"rank stability (|Δrank|<=5 in vs out): {stable:.0%} of indicators")
        sens = sensitivity(data, reg, starts, scores["indicator"].head(10).tolist())
        sens.to_csv(_rp("sensitivity.csv"), index=False)
        print(f"sensitivity: swept primary param of top-10 ({sens['indicator'].nunique()} indicators)")
    else:
        io = sens = None
        if os.path.exists(_rp("sensitivity.csv")):
            sens = pd.read_csv(_rp("sensitivity.csv"))

    # ---- Phase 5: ensembles ------------------------------------------------
    print(f"\n{bar}\n  Phase 5 · Ensemble search\n{bar}")
    ensemble = ensemble_search(data, reg, starts, scores)
    ensemble.to_csv(_rp("ensemble_scores.csv"), index=False)
    print(f"evaluated {len(ensemble)} ensembles")

    # ---- Phase 6: dashboard ------------------------------------------------
    print(f"\n{bar}\n  Phase 6 · Dashboard\n{bar}")
    if sens is None:
        sens = pd.DataFrame(columns=["indicator", "param", "value", "is_default", "f1"])
    out = build_dashboard(
        data, clusters, scores, bt, ensemble, sens, reg, _rp("dashboard.html"),
        meta={"assets": len(data), "period": period, "n_clusters": len(clusters),
              "base_rate": base_rate})
    print(f"wrote {out}")

    _summary(data, period, events, clusters, simultaneous, scores, ensemble, io, base_rate)
    return dict(data=data, clusters=clusters, scores=scores, ensemble=ensemble)


def _summary(data, period, events, clusters, simultaneous, scores, ensemble, io, base_rate=None):
    b = "═" * 51
    print(f"\n{b}")
    print("  Crypto Crash Detector — Backtest Results")
    print(b)
    print(f"\n📊 Data: {len(data)} assets, {period}")
    print(f"💥 Crash Events: {len(events)} event-days "
          f"({len(clusters)} crash legs, {len(simultaneous)} simultaneous-crash weeks)")
    if base_rate is not None:
        print(f"🎲 Base rate (random-signal precision): {base_rate:.2f} "
              f"— indicators must beat this to carry signal")
    print("\n🏆 Top 10 Indicators:")
    for _, r in scores.head(10).iterrows():
        lead = r["avg_lead_time"]
        lead_s = f"{lead:4.1f}d" if lead == lead else "  n/a"
        lift = r["precision_lift"] if "precision_lift" in r else float("nan")
        print(f" #{int(r['rank']):>2} [{r['indicator']:<4}] {r['name'][:30]:<30} "
              f"— F1 {r['f1']:.2f}  P {r['precision']:.2f}  R {r['recall']:.2f}  "
              f"Lead {lead_s}  Lift {lift:.1f}x")
    e = ensemble.iloc[0]
    print("\n🔗 Best Ensemble:")
    print(f" [{e['members']}] {e['strategy']} — F1 {e['f1']:.2f}, "
          f"Precision {e['precision']:.2f}, Recall {e['recall']:.2f}, "
          f"Lead {e['avg_lead_time']:.1f}d")
    if io is not None:
        stable = (io["rank_delta"].abs() <= 5).mean()
        print(f"\n🔬 Out-of-sample: {stable:.0%} of indicators keep rank within ±5 "
              f"(2022-24 → 2025-26)")
    print(f"\n📁 Full results: {os.path.join('results', 'dashboard.html')}")
    print(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crypto crash detector pipeline")
    ap.add_argument("--download", action="store_true", help="force data re-download")
    ap.add_argument("--quick", action="store_true", help="skip in/out-sample + sensitivity")
    args = ap.parse_args()
    run(download=args.download, quick=args.quick)
