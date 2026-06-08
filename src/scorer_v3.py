"""
scorer_v3.py — metrics, cross-asset aggregation, composite ranking (Part 6).

Pipeline:
  trades_df  --per_asset_metrics-->  (strategy, asset) rows
             --aggregate_strategies-> per-strategy cross-asset rows
             --compute_composite---->  ranked table with composite_score
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester_v3 import MIN_TRADES

CATEGORY_BY_GROUP = {
    "S1": "Price Action", "S2": "Alt Momentum", "S3": "Alt Trend",
    "S4": "Volatility", "S5": "Alt Volume", "S6": "Statistical",
    "S7": "Cross-Asset", "S8": "Non-standard",
}


# ---------------------------------------------------------------------------
def buy_hold_returns(data, date_mask=None) -> "dict[str, float]":
    """Buy & Hold total return (%) per asset over the (optional) window."""
    out = {}
    for sym, df in data.items():
        sub = df
        if date_mask is not None:
            sub = df.loc[pd.Timestamp(date_mask[0]):pd.Timestamp(date_mask[1])]
        if len(sub) < 2:
            out[sym] = np.nan
            continue
        out[sym] = float((sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0) * 100.0)
    return out


def _max_consec_losses(pnls: np.ndarray) -> int:
    run = best = 0
    for p in pnls:
        if p < 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _bh_span(df, start, end) -> float:
    """Buy & Hold total return (%) over [start, end] for one asset."""
    sub = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(sub) < 2:
        return np.nan
    return float((sub["close"].iloc[-1] / sub["close"].iloc[0] - 1.0) * 100.0)


def per_asset_metrics(trades: pd.DataFrame, data, date_mask=None,
                      group_of=None) -> pd.DataFrame:
    """One row per (strategy_id, symbol). group_of: dict strategy_id -> group.

    The Buy & Hold benchmark for alpha is measured over the SAME span the
    strategy was actually exposed: from the window start to the asset's last
    realized exit (capped at data end), so the two legs cover equal calendar
    time (spec §5.3 'same period') and the benchmark never gets truncated
    before the strategy's trades have closed.
    """
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (sid, sym), g in trades.groupby(["strategy_id", "symbol"], sort=False):
        pnl = g["pnl_pct"].to_numpy() * 100.0   # percent
        n = len(pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        wr = len(wins) / n if n else 0.0
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        pf = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
        expectancy = wr * avg_win - (1 - wr) * abs(avg_loss)
        avg_hold = float(g["holding_days"].mean())
        # span-aligned Buy & Hold for this (strategy, asset)
        win_start = date_mask[0] if date_mask is not None else data[sym].index.min()
        span_end = min(pd.Timestamp(g["exit_date"].max()), data[sym].index.max())
        bh_ret = _bh_span(data[sym], win_start, span_end)
        rets = g["pnl_pct"].to_numpy()
        if n >= 2 and rets.std(ddof=1) > 0:
            ann = np.sqrt(252.0 / max(avg_hold, 1.0))
            sharpe = float(rets.mean() / rets.std(ddof=1) * ann)
        else:
            sharpe = np.nan
        total_ret = float(pnl.sum())
        rows.append({
            "strategy_id": sid, "symbol": sym,
            "group": (group_of or {}).get(sid, ""),
            "total_return": total_ret, "n_trades": n, "win_rate": wr,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "profit_factor": pf, "expectancy": expectancy,
            "sharpe": sharpe, "max_consec_losses": _max_consec_losses(pnl),
            "max_single_loss": float(pnl.min()) if n else 0.0,
            "avg_holding_days": avg_hold,
            "bh_return": bh_ret,
            "alpha_vs_bh": total_ret - bh_ret,
        })
    return pd.DataFrame(rows)


def aggregate_strategies(pa: pd.DataFrame, min_trades: int = MIN_TRADES) -> pd.DataFrame:
    """Cross-asset aggregation per strategy. Only (strategy,asset) cells with
    >= min_trades count toward the statistics (spec §6.2, §11.4)."""
    if pa.empty:
        return pd.DataFrame()
    valid = pa[pa["n_trades"] >= min_trades]
    rows = []
    for sid, g in valid.groupby("strategy_id", sort=False):
        if g.empty:
            continue
        # cap a perfect (all-winning) profit factor at a finite sentinel so it
        # ranks as BEST rather than collapsing to NaN after a mean(skipna).
        pf = g["profit_factor"].replace(np.inf, 100.0)
        rets = g["total_return"]
        rows.append({
            "strategy_id": sid,
            "group": g["group"].iloc[0],
            "category": CATEGORY_BY_GROUP.get(g["group"].iloc[0], g["group"].iloc[0]),
            "n_assets": int(g["symbol"].nunique()),
            "total_trades": int(g["n_trades"].sum()),
            "cross_asset_avg_return": float(rets.mean()),
            "cross_asset_median_return": float(rets.median()),
            "worst_case_return": float(rets.min()),
            "positive_asset_count": int((rets > 0).sum()),
            "cross_asset_sharpe": float(g["sharpe"].mean(skipna=True)),
            "consistency": float((g["profit_factor"] > 1.0).mean()),
            "avg_profit_factor": float(pf.mean(skipna=True)),
            "avg_expectancy": float(g["expectancy"].mean()),
            "avg_win_rate": float(g["win_rate"].mean()),
            "avg_holding_days": float(g["avg_holding_days"].mean()),
            "avg_alpha_vs_bh": float(g["alpha_vs_bh"].mean(skipna=True)),
        })
    return pd.DataFrame(rows)


def _norm(s: pd.Series) -> pd.Series:
    """Min-max to [0,1] over the FINITE values; undefined (NaN/inf) -> neutral
    0.5 (never demoted to the worst rank, which fillna(min) would wrongly do)."""
    s = s.astype(float)
    finite = s[np.isfinite(s)]
    if finite.empty:
        return pd.Series(0.5, index=s.index)
    lo, hi = finite.min(), finite.max()
    if hi <= lo:
        return pd.Series(0.5, index=s.index)
    out = (s - lo) / (hi - lo)
    return out.where(np.isfinite(out), 0.5)


def compute_composite(agg: pd.DataFrame) -> pd.DataFrame:
    """Composite score per spec §6.3. Returns ranked DataFrame."""
    if agg.empty:
        return agg.assign(composite_score=[], rank=[]) if "composite_score" not in agg else agg
    df = agg.copy()
    df["composite_score"] = (
        0.25 * _norm(df["cross_asset_median_return"])
        + 0.20 * _norm(df["cross_asset_sharpe"])
        + 0.20 * _norm(df["consistency"])
        + 0.15 * _norm(df["avg_profit_factor"])
        + 0.10 * _norm(df["avg_expectancy"])
        + 0.10 * _norm(df["positive_asset_count"] / 9.0)
    )
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def best_by_category(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return ranked
    return (ranked.sort_values(["category", "composite_score"], ascending=[True, False])
            .groupby("category").head(1).reset_index(drop=True))


def score_trades(trades: pd.DataFrame, data, date_mask=None, group_of=None):
    """Convenience: trades -> (per_asset, ranked)."""
    pa = per_asset_metrics(trades, data, date_mask, group_of)
    ranked = compute_composite(aggregate_strategies(pa))
    return pa, ranked
