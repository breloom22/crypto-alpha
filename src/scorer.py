"""
scorer.py — Phase 4/6/7: composite scoring, rankings, in/out-sample split,
parameter sensitivity.

composite_score =
    0.30 * precision
  + 0.25 * recall
  + 0.20 * (1 - false_alarm_rate)
  + 0.15 * normalized_lead_time      (min-max across indicators)
  + 0.10 * consistency_across_assets (share of 9 assets with F1 > 0.3)

All per-indicator aggregates use crash_type == 'ALL', averaged across assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester import run_backtests, pool_metrics, evaluation_base_rate, CRASH_TYPES

WEIGHTS = {"precision": 0.30, "recall": 0.25, "low_false_alarm": 0.20,
           "lead": 0.15, "consistency": 0.10}
CONSISTENCY_F1 = 0.30
IN_SAMPLE = ("2022-01-01", "2024-12-31")
OUT_SAMPLE = ("2025-01-01", "2026-06-06")


# ---------------------------------------------------------------------------
def aggregate_indicators(bt: pd.DataFrame) -> pd.DataFrame:
    """Per-indicator aggregates over assets (crash_type == ALL).

    precision/recall/F1/false-alarm are POOLED from summed counts (robust,
    consistent denominators). consistency = share of assets with a defined
    per-asset F1 above the threshold (NaN F1 from zero-crash windows excluded).
    """
    allc = bt[bt.crash_type == "ALL"].copy()
    rows = []
    for (iid, cat, name), g in allc.groupby(["indicator", "category", "name"]):
        pooled = pool_metrics(g)
        valid = g.dropna(subset=["f1"])
        consistency = float((valid["f1"] > CONSISTENCY_F1).mean()) if len(valid) else 0.0
        rows.append({
            "indicator": iid, "category": cat, "name": name,
            "precision": pooled["precision"], "recall": pooled["recall"],
            "f1": pooled["f1"], "false_alarm_rate": pooled["false_alarm_rate"],
            "avg_lead_time": pooled["avg_lead_time"],
            "median_lead_time": pooled["median_lead_time"],
            "total_signals": pooled["total_signals"],
            "true_positives": pooled["true_positives"],
            "detected_crashes": pooled["detected_crashes"],
            "total_crashes": pooled["total_crashes"],
            "consistency_across_assets": consistency,
        })
    return pd.DataFrame(rows)


def compute_composite(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()
    lead = df["avg_lead_time"].fillna(df["avg_lead_time"].min())
    lo, hi = lead.min(), lead.max()
    df["normalized_lead_time"] = (lead - lo) / (hi - lo) if hi > lo else 0.5
    df["composite_score"] = (
        WEIGHTS["precision"] * df["precision"].fillna(0)
        + WEIGHTS["recall"] * df["recall"].fillna(0)
        + WEIGHTS["low_false_alarm"] * (1 - df["false_alarm_rate"].fillna(1))
        + WEIGHTS["lead"] * df["normalized_lead_time"]
        + WEIGHTS["consistency"] * df["consistency_across_assets"].fillna(0)
    )
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def best_performers(bt: pd.DataFrame) -> dict:
    """Best indicator by category, by asset, and by crash-type (by F1)."""
    out = {}
    allc = bt[bt.crash_type == "ALL"]
    # by category
    by_cat = (allc.groupby(["category", "indicator", "name"])["f1"].mean().reset_index())
    out["by_category"] = (by_cat.sort_values(["category", "f1"], ascending=[True, False])
                          .groupby("category").head(1).reset_index(drop=True))
    # by asset
    by_asset = (allc.groupby(["symbol", "indicator", "name"])["f1"].mean().reset_index())
    out["by_asset"] = (by_asset.sort_values(["symbol", "f1"], ascending=[True, False])
                       .groupby("symbol").head(1).reset_index(drop=True))
    # by crash type (use type-specific recall as discriminator; rank by f1)
    by_type = (bt[bt.crash_type != "ALL"]
               .groupby(["crash_type", "indicator", "name"])["f1"].mean().reset_index())
    out["by_crash_type"] = (by_type.sort_values(["crash_type", "f1"], ascending=[True, False])
                            .groupby("crash_type").head(1).reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
def in_out_sample(data, registry, crash_starts, **bt_kw) -> pd.DataFrame:
    """Re-score on in-sample vs out-of-sample windows; compare ranks."""
    bt_in = run_backtests(data, registry, crash_starts, date_mask=IN_SAMPLE,
                          show_progress=False, **bt_kw)
    bt_out = run_backtests(data, registry, crash_starts, date_mask=OUT_SAMPLE,
                           show_progress=False, **bt_kw)
    sc_in = compute_composite(aggregate_indicators(bt_in))[["indicator", "name", "composite_score", "f1", "precision", "recall"]]
    sc_out = compute_composite(aggregate_indicators(bt_out))[["indicator", "composite_score", "f1", "precision", "recall"]]
    sc_in = sc_in.rename(columns={c: f"{c}_in" for c in ["composite_score", "f1", "precision", "recall"]})
    sc_out = sc_out.rename(columns={c: f"{c}_out" for c in ["composite_score", "f1", "precision", "recall"]})
    m = sc_in.merge(sc_out, on="indicator")
    m["rank_in"] = m["composite_score_in"].rank(ascending=False, method="min").astype(int)
    m["rank_out"] = m["composite_score_out"].rank(ascending=False, method="min").astype(int)
    m["rank_delta"] = m["rank_out"] - m["rank_in"]
    return m.sort_values("composite_score_in", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
def _primary_param(ind):
    return next(iter(ind.params))


def _sweep_values(v):
    """±20% sweep (plus midpoints) around a default param value."""
    mults = [0.8, 0.9, 1.0, 1.1, 1.2]
    if isinstance(v, bool):
        return [v]
    if isinstance(v, int) and abs(v) >= 2:        # keep count/period params integral
        return sorted({max(2, int(round(v * m))) for m in mults})
    if isinstance(v, (int, float)):
        return sorted({round(float(v) * m, 4) for m in mults})
    return [v]


def sensitivity(data, registry, crash_starts, top_ids, **bt_kw) -> pd.DataFrame:
    """For each top indicator, sweep its primary parameter ±20% and re-score."""
    from collections import OrderedDict
    rows = []
    for iid in top_ids:
        base = registry[iid]
        pname = _primary_param(base)
        for val in _sweep_values(base.params[pname]):
            variant = base.with_params(**{pname: val})
            bt = run_backtests(data, OrderedDict([(iid, variant)]), crash_starts,
                               show_progress=False, **bt_kw)
            agg = aggregate_indicators(bt)
            if agg.empty:
                continue
            r = agg.iloc[0]
            rows.append({
                "indicator": iid, "name": base.name, "param": pname, "value": val,
                "is_default": val == base.params[pname],
                "f1": round(float(r["f1"]), 4),
                "precision": round(float(r["precision"]), 4),
                "recall": round(float(r["recall"]), 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils import setup_console, DATA_DIR, RESULTS_DIR
    setup_console()
    import os
    from data_loader import load_data
    from crash_detector import detect_all, crash_starts_by_type
    from indicators.base import build_indicators

    data = load_data(DATA_DIR)
    events, clusters, _ = detect_all(data)
    starts = crash_starts_by_type(events, clusters)
    reg = build_indicators()

    bt = run_backtests(data, reg, starts)
    scores = compute_composite(aggregate_indicators(bt))
    scores.to_csv(os.path.join(RESULTS_DIR, "individual_scores.csv"), index=False)
    bt.to_csv(os.path.join(RESULTS_DIR, "backtest_full.csv"), index=False)

    print("\n=== Top 10 indicators (composite score) ===")
    cols = ["rank", "indicator", "name", "composite_score", "f1", "precision",
            "recall", "avg_lead_time", "consistency_across_assets"]
    print(scores[cols].head(10).round(3).to_string(index=False))

    top10 = scores["indicator"].head(10).tolist()
    print("\n=== In/Out-of-sample stability ===")
    io = in_out_sample(data, reg, starts)
    io.to_csv(os.path.join(RESULTS_DIR, "in_out_sample.csv"), index=False)
    print(io.head(12)[["indicator", "rank_in", "rank_out", "rank_delta",
                       "f1_in", "f1_out"]].round(3).to_string(index=False))

    print("\n=== Parameter sensitivity (top 10, primary param ±20%) ===")
    sens = sensitivity(data, reg, starts, top10)
    sens.to_csv(os.path.join(RESULTS_DIR, "sensitivity.csv"), index=False)
    print(sens.round(3).to_string(index=False))
