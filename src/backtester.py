"""
backtester.py — Phase 3: leading-signal evaluation engine.

A signal is a *leading* warning if a crash onset follows it within
[min_lead_time, forward_window] days:

      signal day  ───►  (1 .. N days)  ───►  crash onset
      ───●────────────────────────────────────●───────

Metrics (per indicator x asset x crash-type):
  precision         = TP_signals / total_signals     (signal-centric)
  recall            = detected_crashes / total_crashes (crash-centric)
  f1                = harmonic mean of precision & recall
  avg/median lead   = days from a TP signal to the crash it predicted
  false_alarm_rate  = FP / total_signals  ( = 1 - precision )
  hit-rate-by-type  = recall computed with type-specific crash onsets

All evaluation is vectorized via searchsorted. Look-ahead safety lives in the
indicators; here we only compare event timings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORWARD_WINDOW = 21
MIN_LEAD_TIME = 1
CRASH_TYPES = ["ALL", "A", "B", "C", "D"]


def _ords(index_or_dates) -> np.ndarray:
    """Convert dates -> integer day numbers (sorted not guaranteed)."""
    di = pd.DatetimeIndex(index_or_dates)
    return di.values.astype("datetime64[D]").astype("int64")


def backtest_signal(signal: pd.Series, crash_onsets,
                    forward_window: int = FORWARD_WINDOW,
                    min_lead_time: int = MIN_LEAD_TIME) -> dict:
    """Evaluate one boolean signal series against a list of crash onset dates."""
    sig_ord = np.sort(_ords(signal.index[signal.values.astype(bool)]))
    crash_ord = np.sort(_ords(crash_onsets)) if len(crash_onsets) else np.array([], dtype="int64")
    total_signals = int(sig_ord.size)
    total_crashes = int(crash_ord.size)

    # ---- precision (signal-centric): each signal followed by a crash? -------
    tp = fp = 0
    leads = np.array([], dtype="int64")
    if total_signals and total_crashes:
        lo = sig_ord + min_lead_time
        hi = sig_ord + forward_window
        idx = np.searchsorted(crash_ord, lo, side="left")
        valid = idx < total_crashes
        nearest = crash_ord[np.clip(idx, 0, total_crashes - 1)]
        within = valid & (nearest <= hi)
        tp = int(within.sum())
        fp = total_signals - tp
        leads = (nearest[within] - sig_ord[within])
    else:
        fp = total_signals

    # ---- recall (crash-centric): each crash preceded by a signal? -----------
    detected = 0
    if total_signals and total_crashes:
        lo = crash_ord - forward_window
        hi = crash_ord - min_lead_time
        idx = np.searchsorted(sig_ord, lo, side="left")
        valid = idx < total_signals
        nearest = sig_ord[np.clip(idx, 0, total_signals - 1)]
        within = valid & (nearest <= hi)
        detected = int(within.sum())
    missed = total_crashes - detected

    precision = tp / total_signals if total_signals else 0.0
    recall = (detected / total_crashes) if total_crashes else np.nan
    if total_crashes and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = (np.nan if not total_crashes else 0.0)
    false_alarm = fp / total_signals if total_signals else np.nan

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "avg_lead_time": float(leads.mean()) if leads.size else np.nan,
        "median_lead_time": float(np.median(leads)) if leads.size else np.nan,
        "false_alarm_rate": false_alarm,
        "total_signals": total_signals, "true_positives": tp, "false_positives": fp,
        "detected_crashes": detected, "missed_crashes": missed,
        "total_crashes": total_crashes,
    }


def pool_metrics(df: pd.DataFrame) -> dict:
    """Aggregate per-(asset/window) backtest rows by POOLING raw counts, so
    precision/recall/false-alarm are computed once over the same denominator
    (avoids the NaN-skip inconsistency of mean-of-ratios; preserves
    false_alarm = 1 - precision)."""
    tp = int(df["true_positives"].sum())
    fp = int(df["false_positives"].sum())
    det = int(df["detected_crashes"].sum())
    tc = int(df["total_crashes"].sum())
    ts = int(df["total_signals"].sum())
    precision = tp / ts if ts else 0.0
    recall = det / tc if tc else np.nan
    far = fp / ts if ts else 0.0
    if tc:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = np.nan
    return {
        "precision": precision, "recall": recall, "f1": f1, "false_alarm_rate": far,
        "avg_lead_time": df["avg_lead_time"].mean(), "median_lead_time": df["median_lead_time"].mean(),
        "total_signals": ts, "true_positives": tp, "false_positives": fp,
        "detected_crashes": det, "total_crashes": tc,
    }


def evaluation_base_rate(data, crash_starts, crash_type="ALL",
                         forward_window: int = FORWARD_WINDOW,
                         min_lead_time: int = MIN_LEAD_TIME) -> float:
    """Pooled base rate = P(a crash onset falls in the next [min_lead, forward]
    days) for a randomly-timed signal. This is the precision an indicator must
    beat to carry information. Computed as the precision of an always-on signal."""
    tp = ts = 0
    for sym, df in data.items():
        always = pd.Series(True, index=df.index)
        m = backtest_signal(always, crash_starts.get(sym, {}).get(crash_type, []),
                            forward_window, min_lead_time)
        tp += m["true_positives"]
        ts += m["total_signals"]
    return tp / ts if ts else float("nan")


def _signal_series_for(ind, data, symbol, market_cache: dict):
    """Boolean signal series for (indicator, asset), reusing market signals."""
    if ind.cross_asset:
        if ind.id not in market_cache:
            market_cache[ind.id] = ind.signals(data)
        return market_cache[ind.id].reindex(data[symbol].index).fillna(False)
    return ind.signals(data[symbol])


def run_backtests(data, registry, crash_starts,
                  forward_window: int = FORWARD_WINDOW,
                  min_lead_time: int = MIN_LEAD_TIME,
                  date_mask=None, show_progress: bool = True) -> pd.DataFrame:
    """Backtest every indicator x asset x crash-type.

    date_mask: optional (start, end) ISO tuple to restrict evaluation window
               (used for in-sample / out-of-sample splits).
    """
    try:
        from tqdm import tqdm
    except Exception:                                    # noqa: BLE001
        def tqdm(x, **k): return x

    if date_mask is not None:
        mask_start, mask_end = pd.Timestamp(date_mask[0]), pd.Timestamp(date_mask[1])
    market_cache: dict = {}
    rows = []
    items = list(registry.items())
    for iid, ind in (tqdm(items, desc="backtest") if show_progress else items):
        for sym, df in data.items():
            sig = _signal_series_for(ind, data, sym, market_cache)
            starts = crash_starts.get(sym, {})
            if date_mask is not None:
                sig = sig.loc[mask_start:mask_end]
            for ctype in CRASH_TYPES:
                onsets = starts.get(ctype, [])
                if date_mask is not None:
                    onsets = [d for d in onsets if mask_start <= pd.Timestamp(d) <= mask_end]
                m = backtest_signal(sig, onsets, forward_window, min_lead_time)
                rows.append({
                    "indicator": iid, "category": ind.category, "name": ind.name,
                    "symbol": sym, "crash_type": ctype, **m,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from utils import setup_console, DATA_DIR, RESULTS_DIR
    setup_console()
    from data_loader import load_data
    from crash_detector import detect_all, crash_starts_by_type
    from indicators.base import build_indicators

    data = load_data(DATA_DIR)
    events, clusters, _ = detect_all(data)
    starts = crash_starts_by_type(events, clusters)
    reg = build_indicators()
    res = run_backtests(data, reg, starts)

    # quick look: best by F1 on crash_type ALL, aggregated across assets
    allc = res[res.crash_type == "ALL"]
    agg = (allc.groupby(["indicator", "name"])
           .agg(f1=("f1", "mean"), precision=("precision", "mean"),
                recall=("recall", "mean"), lead=("avg_lead_time", "mean"))
           .sort_values("f1", ascending=False))
    print("\n=== Top 12 indicators by mean F1 (crash_type=ALL) ===")
    print(agg.head(12).round(3).to_string())
    print(f"\nTotal backtest rows: {len(res)}")
