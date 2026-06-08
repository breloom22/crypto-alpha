"""
crash_detector.py — Phase 1: objective crash-event identification.

Defines crash events with four parallel criteria (any one => event day):

    CRASH_A : 7-day rolling return  <= -15%   (one-week plunge)
    CRASH_B : 3-day rolling return  <= -10%   (three-day plunge)
    CRASH_C : drawdown from 52-week high >= 20%   (structural decline)
    CRASH_D : single-day return     <=  -8%   (one-day crash)

Event days within 14 days of one another are merged into a single "crash
leg" (cluster). The first day of each cluster is its representative onset date,
which the backtester treats as the moment a leading signal must precede.

All computations use only past/contemporaneous data (no look-ahead).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---- criteria thresholds (from spec) ---------------------------------------
RET_7D_THR = -0.15
RET_3D_THR = -0.10
DD_52W_THR = -0.20
RET_1D_THR = -0.08
WIN_52W = 365            # daily crypto calendar ~ 52 weeks
MIN_C_OBS = 180          # require ~26 weeks of history before CRASH_C can fire,
                         # so the "52-week high" is meaningful (not a 20-day high)
CLUSTER_GAP_DAYS = 14    # <= this gap between event days => same cluster
CRASH_TYPES = ["A", "B", "C", "D"]


def identify_crash_events(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Identify crash event days for one asset and assign clusters.

    Returns one row per *event day* with columns:
        date, symbol, crash_A..D (bool), crash_type ("A|B|.."),
        severity (worst decline among fired criteria, fraction),
        ret_1d, ret_3d, ret_7d, dd_52w,
        cluster_id, cluster_start, cluster_peak_decline
    """
    c = df["close"]
    ret_1d = c.pct_change(1)
    ret_3d = c.pct_change(3)
    ret_7d = c.pct_change(7)
    # 52-week high requires a substantial window first: dd_52w stays NaN (so
    # CRASH_C cannot fire) during each asset's first ~26 weeks of data, avoiding
    # spurious "structural decline" onsets measured against a 20-day-old high.
    # CRASH_A/B/D still capture genuine early plunges. (Look-ahead safe: larger
    # min_periods only adds leading NaN.)
    roll_max = c.rolling(WIN_52W, min_periods=MIN_C_OBS).max()
    dd_52w = c / roll_max - 1.0

    # Each criterion is an ONSET-EDGE event: it fires on the day the condition
    # first becomes true (crosses into crash territory), not on every day it
    # persists. This is essential for CRASH_C — a *level* condition (drawdown vs
    # 52-week high) that would otherwise stay true for months and collapse an
    # entire bear market into one giant cluster, hiding distinct crash legs
    # (e.g. the 2026-06 plunge). Edge = state(t) AND NOT state(t-1): look-ahead
    # safe (uses only t and t-1).
    def _edge(state: pd.Series) -> pd.Series:
        return state & (~state.shift(1, fill_value=False))

    crash_A = _edge(ret_7d <= RET_7D_THR)
    crash_B = _edge(ret_3d <= RET_3D_THR)
    crash_C = _edge(dd_52w <= DD_52W_THR)
    crash_D = _edge(ret_1d <= RET_1D_THR)
    any_event = crash_A | crash_B | crash_C | crash_D

    ev = pd.DataFrame({
        "crash_A": crash_A, "crash_B": crash_B,
        "crash_C": crash_C, "crash_D": crash_D,
        "ret_1d": ret_1d, "ret_3d": ret_3d, "ret_7d": ret_7d, "dd_52w": dd_52w,
    })[any_event.fillna(False)].copy()

    if ev.empty:
        return pd.DataFrame(columns=[
            "date", "symbol", "crash_A", "crash_B", "crash_C", "crash_D",
            "crash_type", "severity", "ret_1d", "ret_3d", "ret_7d", "dd_52w",
            "cluster_id", "cluster_start", "cluster_peak_decline"])

    # crash_type label + severity (most-negative decline among fired criteria)
    def _row_type(r):
        return "|".join(t for t in CRASH_TYPES if r[f"crash_{t}"])

    def _row_sev(r):
        cands = []
        if r["crash_A"]: cands.append(r["ret_7d"])
        if r["crash_B"]: cands.append(r["ret_3d"])
        if r["crash_C"]: cands.append(r["dd_52w"])
        if r["crash_D"]: cands.append(r["ret_1d"])
        return min(cands) if cands else np.nan

    ev["crash_type"] = ev.apply(_row_type, axis=1)
    ev["severity"] = ev.apply(_row_sev, axis=1)

    # ---- cluster event days (gap <= CLUSTER_GAP_DAYS) -----------------------
    dates = ev.index.to_series()
    gap = dates.diff().dt.days
    cluster_local = (gap > CLUSTER_GAP_DAYS).fillna(False).cumsum()
    ev["cluster_id"] = [f"{symbol}_{i}" for i in cluster_local]

    # per-cluster onset + peak drawdown within the cluster window
    cluster_start = {}
    cluster_peak = {}
    for cid, grp in ev.groupby("cluster_id"):
        start, end = grp.index.min(), grp.index.max()
        cluster_start[cid] = start
        window = c.loc[start:end]
        run_peak = window.cummax()
        peak_dd = float((window / run_peak - 1.0).min()) if len(window) else float(grp["severity"].min())
        # also consider the single worst severity day in the leg
        cluster_peak[cid] = min(peak_dd, float(grp["severity"].min()))
    ev["cluster_start"] = ev["cluster_id"].map(cluster_start)
    ev["cluster_peak_decline"] = ev["cluster_id"].map(cluster_peak)

    ev = ev.reset_index().rename(columns={"index": "date"})
    if "date" not in ev.columns:
        ev = ev.rename(columns={ev.columns[0]: "date"})
    ev.insert(1, "symbol", symbol)
    return ev


def build_clusters(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse event-day table to one row per crash leg (cluster)."""
    if events.empty:
        return pd.DataFrame(columns=[
            "symbol", "cluster_id", "cluster_start", "cluster_end",
            "n_event_days", "types", "peak_decline",
            "has_A", "has_B", "has_C", "has_D"])
    rows = []
    for cid, g in events.groupby("cluster_id"):
        types = set()
        for t in g["crash_type"]:
            types.update(t.split("|"))
        rows.append({
            "symbol": g["symbol"].iloc[0],
            "cluster_id": cid,
            "cluster_start": g["date"].min(),
            "cluster_end": g["date"].max(),
            "n_event_days": len(g),
            "types": "|".join(sorted(types)),
            "peak_decline": float(g["cluster_peak_decline"].iloc[0]),
            **{f"has_{t}": (t in types) for t in CRASH_TYPES},
        })
    out = pd.DataFrame(rows).sort_values(["symbol", "cluster_start"]).reset_index(drop=True)
    return out


def detect_all(data: dict[str, pd.DataFrame]):
    """Run crash detection across all assets.

    Returns (events, clusters, simultaneous) DataFrames.
    """
    all_ev = []
    for sym, df in data.items():
        ev = identify_crash_events(df, sym)
        if not ev.empty:
            all_ev.append(ev)
    events = pd.concat(all_ev, ignore_index=True) if all_ev else pd.DataFrame()
    clusters = build_clusters(events)

    # ---- simultaneous crashes: weeks with >=3 distinct assets in event ------
    simultaneous = pd.DataFrame()
    if not events.empty:
        ev = events.copy()
        ev["year_week"] = ev["date"].dt.strftime("%G-W%V")
        wk = (ev.groupby("year_week")
                .agg(n_assets=("symbol", "nunique"),
                     assets=("symbol", lambda s: "|".join(sorted(set(s)))),
                     week_start=("date", "min"),
                     worst_decline=("severity", "min"))
                .reset_index())
        simultaneous = wk[wk["n_assets"] >= 3].sort_values("week_start").reset_index(drop=True)

    return events, clusters, simultaneous


def save(events, clusters, simultaneous, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    events.to_csv(os.path.join(results_dir, "crash_events.csv"), index=False)
    clusters.to_csv(os.path.join(results_dir, "crash_clusters.csv"), index=False)
    if not simultaneous.empty:
        simultaneous.to_csv(os.path.join(results_dir, "simultaneous_crashes.csv"), index=False)


def crash_starts_by_type(events: pd.DataFrame, clusters: pd.DataFrame) -> dict:
    """Map symbol -> {ALL, A, B, C, D: [onset dates]} for the backtester.

    ALL  -> each crash leg's onset (cluster_start).
    A..D -> within each leg that contains type t, the FIRST day on which type t
            actually fired (not the leg's overall onset, which may belong to a
            different criterion). This makes Hit-Rate-by-Crash-Type accurate.
    """
    out: dict[str, dict[str, list]] = {}
    for sym, g in clusters.groupby("symbol"):
        ev = events[events["symbol"] == sym]
        d = {"ALL": sorted(g["cluster_start"].tolist())}
        for t in CRASH_TYPES:
            fired = ev[ev[f"crash_{t}"]]
            d[t] = sorted(fired.groupby("cluster_id")["date"].min().tolist())
        out[sym] = d
    return out


def _print_summary(data, events, clusters, simultaneous):
    print("\n=== Phase 1: Crash events ===")
    print(f"Total event days: {len(events)}   |   clusters (legs): {len(clusters)}")
    by_sym = clusters.groupby("symbol").size().reindex(data.keys(), fill_value=0)
    print("\nClusters per asset:")
    for s, n in by_sym.items():
        print(f"  {s:5} {n:3d}")
    typ = {t: int(clusters[f"has_{t}"].sum()) for t in CRASH_TYPES}
    print(f"\nClusters containing each crash type: {typ}")
    if not simultaneous.empty:
        print(f"\nSimultaneous-crash weeks (>=3 assets): {len(simultaneous)}")
        print(simultaneous.head(8).to_string(index=False))


if __name__ == "__main__":
    from utils import setup_console, DATA_DIR, RESULTS_DIR
    setup_console()
    from data_loader import load_data
    data = load_data(DATA_DIR)
    events, clusters, simultaneous = detect_all(data)
    save(events, clusters, simultaneous, RESULTS_DIR)
    _print_summary(data, events, clusters, simultaneous)
