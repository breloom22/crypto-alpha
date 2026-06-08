"""
random_benchmark.py — random-entry significance test (Part 7.4).

For each strategy we hold its exit rules and per-asset trade *frequency* fixed,
then draw that many entry days at random (1000 simulations). If the strategy's
real cross-asset return exceeds the 95th percentile of the random distribution,
its edge is unlikely to be luck (p < 0.05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import _ta
from indicators import _ta_extended as tae
from backtester_v3 import _simulate_one, run_backtests
from scorer_v3 import per_asset_metrics


def _prep_arrays(df, cfg):
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    need_atr = (cfg.atr_stop_mult is not None or cfg.atr_tp_mult is not None
                or cfg.profit_trail_trigger is not None or cfg.time_dynamic
                or cfg.stepped_trail_base is not None)
    atr_arr = _ta.atr(df["high"], df["low"], df["close"], cfg.atr_period).to_numpy() if need_atr else None
    if cfg.chandelier_mult is not None:
        cl, cs = tae.chandelier_exit(df["high"], df["low"], df["close"],
                                     cfg.chandelier_period, cfg.chandelier_mult)
        chand_long, chand_short = cl.to_numpy(), cs.to_numpy()
    else:
        chand_long = chand_short = None
    rsi_arr = _ta.rsi(df["close"], 14).to_numpy() if cfg.indicator_neutral else None
    return o, h, l, c, atr_arr, chand_long, chand_short, rsi_arr


def _valid_entry_bars(df, date_mask):
    n = len(df)
    bars = np.arange(1, n)            # entry bar = signal+1, so >=1
    if date_mask is not None:
        ms, me = pd.Timestamp(date_mask[0]), pd.Timestamp(date_mask[1])
        idx = df.index
        bars = np.array([b for b in bars if ms <= idx[b] <= me], dtype=int)
    return bars


def random_benchmark(strategies, data, actual_per_asset: pd.DataFrame,
                     trades: pd.DataFrame = None,
                     date_mask=None, n_sims=1000, seed=42) -> pd.DataFrame:
    """actual_per_asset: per_asset_metrics() rows for these strategies on the
    SAME window (real per-asset trade counts + the actual cross-asset return).
    trades: the raw trade rows for the same run, used to replay each strategy's
    exact per-(asset, direction) LONG/SHORT mix so the random null distribution
    matches the strategy's true exposure (critical for BOTH/SEQ strategies)."""
    rng = np.random.default_rng(seed)
    actual_ret = (actual_per_asset.groupby("strategy_id")["total_return"].mean()
                  if not actual_per_asset.empty else pd.Series(dtype=float))
    # per-(strategy, asset) LONG/SHORT trade counts from the real trades
    dir_count: dict = {}
    if trades is not None and not trades.empty:
        for (sid, sym, d), g in trades.groupby(["strategy_id", "symbol", "direction"]):
            dir_count.setdefault(sid, {}).setdefault(sym, {})[d] = len(g)

    rows = []
    for strat in strategies:
        cfg = strat.exit
        per_sym = dir_count.get(strat.id, {})
        if not per_sym:
            continue
        prep = {sym: (_prep_arrays(data[sym], cfg), _valid_entry_bars(data[sym], date_mask),
                      len(data[sym])) for sym in per_sym}

        sim_returns = np.empty(n_sims)
        for s in range(n_sims):
            per_asset_means = []
            for sym, dcounts in per_sym.items():
                (o, h, l, c, atr_arr, cl, cs, rsi_arr), bars, n = prep[sym]
                # replay the same LONG/SHORT mix this strategy took on this asset
                dirs = [dd for dd, k in dcounts.items() for _ in range(k)]
                if len(bars) == 0 or not dirs:
                    continue
                k = min(len(dirs), len(bars))
                picks = rng.choice(bars, size=k, replace=False)
                ol = np.zeros(n, dtype=bool); osa = np.zeros(n, dtype=bool)
                pnls = []
                for e, dd in zip(picks, dirs[:k]):
                    _, _, _, pnl = _simulate_one(
                        dd, int(e), o, h, l, c, atr_arr, cl, cs,
                        rsi_arr, ol, osa, cfg, 0.001, n)
                    pnls.append(pnl * 100.0)
                if pnls:
                    per_asset_means.append(float(np.sum(pnls)))
            sim_returns[s] = np.mean(per_asset_means) if per_asset_means else 0.0
        p95 = float(np.percentile(sim_returns, 95))
        areturn = float(actual_ret.get(strat.id, np.nan))
        p_value = float(np.mean(sim_returns >= areturn)) if np.isfinite(areturn) else np.nan
        rows.append({
            "strategy_id": strat.id, "actual_return": areturn,
            "random_mean": float(sim_returns.mean()),
            "random_p95": p95, "p_value": p_value,
            "significant": bool(np.isfinite(areturn) and areturn > p95),
            "n_sims": n_sims,
        })
    return pd.DataFrame(rows)
