"""
ensemble.py — Phase 5: combination (ensemble) search over top indicators.

Combines the top-N indicators in groups of 2-3 under five rules:

  AND        — every member fires the same day            (conservative)
  MAJORITY   — strict majority of members fire             (>k/2)
  OR         — any member fires                            (aggressive)
  WEIGHTED   — sum of F1-weighted member signals >= half the total weight
  SEQUENTIAL — members fire in rank order, each within `seq_window` days of the
               previous; warning on the last member's day

Each combined signal is backtested (per asset, crash_type ALL) and aggregated
across assets. Top ensembles are reported by mean F1.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from backtester import backtest_signal, _signal_series_for, pool_metrics

STRATEGIES = ["AND", "MAJORITY", "OR", "WEIGHTED", "SEQUENTIAL"]
SEQ_WINDOW = 5
TOP_N = 10


# ---- combined-signal builders ----------------------------------------------
def _matrix(series_list):
    m = pd.concat(series_list, axis=1).fillna(False).astype(int)
    m.columns = range(len(series_list))
    return m


def combine(strategy, series_list, weights=None, seq_window=SEQ_WINDOW):
    m = _matrix(series_list)
    k = m.shape[1]
    if strategy == "AND":
        out = m.sum(axis=1) == k
    elif strategy == "OR":
        out = m.sum(axis=1) >= 1
    elif strategy == "MAJORITY":
        out = m.sum(axis=1) > (k / 2.0)
    elif strategy == "WEIGHTED":
        w = np.asarray(weights, dtype=float)
        score = (m.to_numpy() * w).sum(axis=1)
        out = pd.Series(score >= 0.5 * w.sum(), index=m.index)
    elif strategy == "SEQUENTIAL":
        acc = series_list[0].fillna(False)
        for nxt in series_list[1:]:
            prior = acc.shift(1).rolling(seq_window, min_periods=1).max().fillna(0) > 0
            acc = (nxt.fillna(False) & prior)
        out = acc
    else:
        raise ValueError(strategy)
    return out.reindex(m.index).fillna(False).astype(bool)


# ---- driver ----------------------------------------------------------------
def build_member_signals(data, registry, top_ids):
    cache, sigs = {}, {}
    for iid in top_ids:
        sigs[iid] = {sym: _signal_series_for(registry[iid], data, sym, cache)
                     for sym in data}
    return sigs


def search(data, registry, crash_starts, scores: pd.DataFrame,
           top_n=TOP_N, sizes=(2, 3), seq_window=SEQ_WINDOW,
           forward_window=21, min_lead_time=1, show_progress=True) -> pd.DataFrame:
    top_ids = scores["indicator"].head(top_n).tolist()
    f1_by_id = dict(zip(scores["indicator"], scores["f1"]))
    member_sigs = build_member_signals(data, registry, top_ids)

    combos = []
    for s in sizes:
        combos.extend(itertools.combinations(top_ids, s))

    try:
        from tqdm import tqdm
        iterator = tqdm(combos, desc="ensemble") if show_progress else combos
    except Exception:                                    # noqa: BLE001
        iterator = combos

    rows = []
    for combo in iterator:
        weights = [max(f1_by_id.get(i, 0.0), 1e-6) for i in combo]
        for strat in STRATEGIES:
            # strict majority of a 2-member set == AND, so skip the duplicate
            if strat == "MAJORITY" and len(combo) < 3:
                continue
            per_asset = []
            for sym in data:
                series_list = [member_sigs[i][sym] for i in combo]
                combined = combine(strat, series_list, weights, seq_window)
                m = backtest_signal(combined, crash_starts.get(sym, {}).get("ALL", []),
                                    forward_window, min_lead_time)
                per_asset.append(m)
            pooled = pool_metrics(pd.DataFrame(per_asset))   # consistent denominators
            rows.append({
                "members": "+".join(combo), "size": len(combo), "strategy": strat,
                "f1": pooled["f1"], "precision": pooled["precision"],
                "recall": pooled["recall"], "false_alarm_rate": pooled["false_alarm_rate"],
                "avg_lead_time": pooled["avg_lead_time"],
                "total_signals": pooled["total_signals"],
                "detected_crashes": pooled["detected_crashes"],
                "total_crashes": pooled["total_crashes"],
            })
    out = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


if __name__ == "__main__":
    from utils import setup_console, DATA_DIR, RESULTS_DIR
    setup_console()
    from data_loader import load_data
    from crash_detector import detect_all, crash_starts_by_type
    from indicators.base import build_indicators
    from scorer import compute_composite, aggregate_indicators
    from backtester import run_backtests

    data = load_data(DATA_DIR)
    events, clusters, _ = detect_all(data)
    starts = crash_starts_by_type(events, clusters)
    reg = build_indicators()
    bt = run_backtests(data, reg, starts, show_progress=False)
    scores = compute_composite(aggregate_indicators(bt))

    ens = search(data, reg, starts, scores)
    ens.to_csv(os.path.join(RESULTS_DIR, "ensemble_scores.csv"), index=False)

    print(f"\nEvaluated {len(ens)} ensembles. Top 10 by F1:")
    cols = ["rank", "members", "strategy", "f1", "precision", "recall",
            "false_alarm_rate", "avg_lead_time"]
    print(ens[cols].head(10).round(3).to_string(index=False))

    print("\nBest single-indicator F1 for reference:",
          round(scores["f1"].max(), 3), "(", scores["name"].iloc[0], ")")
