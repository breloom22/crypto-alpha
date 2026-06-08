"""
optimizer.py — Phase 4 candidate generator + comparison engine.

Pipeline (v4 spec Steps 1-6):
  1. reconstruct the 7 base strategies (improvements.reconstruct)
  2. generate single-module candidates  A,B,C,E,F  (Strategy objects) and the
     D candidates (same strategy, evaluated on a reduced universe / weighted)
  3. backtest every candidate on IS *and* OOS, score on the SAME v3 framework
  4. IS gate: a candidate is "IS-improved" iff its IS return beats its base's
  5. build a small set of promising combos from each base's best A / C / F
     winner, gate them too
  6. OOS verdict (adopt / hold / reject) vs the original, module effectiveness,
     and the original-vs-best comparison table

Nothing here re-derives signals: candidates are the v3 base strategies with
modified exits / filters / SEQ-long-legs / evaluation universe only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, field

import numpy as np
import pandas as pd

from strategy import Strategy
from backtester_v3 import run_backtests, MIN_TRADES
from scorer_v3 import per_asset_metrics, aggregate_strategies

import oos_validator as oosv
import random_benchmark as rb

from improvements import reconstruct as RC
from improvements import dynamic_sl_tp as A
from improvements import break_even as B
from improvements import whipsaw_guard as C
from improvements import stepped_trailing as F
from improvements import alt_long_signals as E
from improvements import asset_weighting as D
from improvements import portfolio as G

STOP_REASONS = ("stop_loss", "atr_stop")          # for SL-hit-rate diagnostics

# Portfolios to evaluate (v4 spec §Improvement G). Members are strategy IDs;
# any reconstructable id (incl. SEQ not in the 7 targets) is allowed.
PORTFOLIOS = {
    "P1_Top3": ["SEQ_S1.11_S+S4.04_L", "S2.08_L_aggr_F9", "S3.12_S_aggr_F8"],
    "P2_DirSplit": ["SEQ_S1.11_S+S4.04_L", "S2.08_L_aggr_F3", "S7.04_S_trail"],
    "P3_SEQonly": ["SEQ_S1.11_S+S4.04_L", "SEQ_S1.11_S+S3.01_L", "SEQ_S5.02_S+S3.01_L"],
    "P4_All7": list(RC.TARGET_IDS),
}


# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    id: str
    base_id: str
    module: str                 # 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'combo' | 'base'
    strategy: Strategy
    universe: str = "full"      # 'full' | 'ex_weak'  (Improvement D1)
    weighted: bool = False      # Improvement D2 (tier-weighted cross-asset mean)


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------
def _single_candidates(base: Strategy) -> "list[Candidate]":
    out: list[Candidate] = []
    for s in A.variants(base):
        out.append(Candidate(s.id, base.id, "A", s))
    for s in B.variants(base):
        out.append(Candidate(s.id, base.id, "B", s))
    for s in C.variants(base):
        out.append(Candidate(s.id, base.id, "C", s))
    for s in F.variants(base):
        out.append(Candidate(s.id, base.id, "F", s))
    for s in E.variants(base):                     # SEQ only -> may be empty
        out.append(Candidate(s.id, base.id, "E", s))
    # Improvement D — same strategy, different evaluation
    out.append(Candidate(f"{base.id}+D1", base.id, "D",
                         replace(base, id=f"{base.id}+D1"), universe="ex_weak"))
    out.append(Candidate(f"{base.id}+D2", base.id, "D",
                         replace(base, id=f"{base.id}+D2"), weighted=True))
    return out


def generate_singles(bases) -> "list[Candidate]":
    out = []
    for b in bases:
        out += _single_candidates(b)
    return out


def _best_variant(eval_df: pd.DataFrame, base_id: str, module: str,
                  metric="is_return"):
    """The best (by IS return) candidate id for one base + module, or None."""
    sub = eval_df[(eval_df["base_id"] == base_id) & (eval_df["module"] == module)]
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return None
    return sub.sort_values(metric, ascending=False).iloc[0]["id"]


def generate_combos(bases, cand_by_id, eval_df) -> "list[Candidate]":
    """Promising combos from each base's best A / C(mask) / F single-module
    winners (spec principle: combine only AFTER independent module results)."""
    out = []
    by_id = {c.id: c for c in cand_by_id}
    for b in bases:
        bid = b.id
        best_a = _best_variant(eval_df, bid, "A")
        # best C among *mask* filters only (FC3 is path-dependent; keep combos clean)
        csub = eval_df[(eval_df["base_id"] == bid) & (eval_df["module"] == "C")
                       & (~eval_df["id"].str.endswith("+FC3"))]
        csub = csub[np.isfinite(csub["is_return"])]
        best_c = (csub.sort_values("is_return", ascending=False).iloc[0]["id"]
                  if not csub.empty else None)
        best_f = _best_variant(eval_df, bid, "F")

        a_strat = by_id[best_a].strategy if best_a else None
        # A + B  (best ATR stop/target, then break-even + extended hold = B2)
        if a_strat is not None:
            ab = next((s for s in B.variants(a_strat) if s.id.endswith("+B2")), None)
            if ab is not None:
                out.append(Candidate(ab.id, bid, "combo", ab))
            # A + C(mask)
            if best_c is not None:
                fid = best_c.rsplit("+", 1)[-1]            # e.g. "FC1"
                ac = next((s for s in C.variants(a_strat) if s.id.endswith("+" + fid)), None)
                if ac is not None:
                    out.append(Candidate(ac.id, bid, "combo", ac))
                    # A + C + D1 (reduced universe on top of the A+C strategy)
                    out.append(Candidate(f"{ac.id}+D1", bid, "combo",
                                         replace(ac, id=f"{ac.id}+D1"), universe="ex_weak"))
        # F + best C(mask)
        if best_f and best_c is not None:
            f_strat = by_id[best_f].strategy
            fid = best_c.rsplit("+", 1)[-1]
            fc = next((s for s in C.variants(f_strat) if s.id.endswith("+" + fid)), None)
            if fc is not None:
                out.append(Candidate(fc.id, bid, "combo", fc))
    # de-dup by id
    seen, uniq = set(), []
    for c in out:
        if c.id not in seen:
            seen.add(c.id)
            uniq.append(c)
    return uniq


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _metrics(trades: pd.DataFrame, chosen_data, window, group_of, cand: Candidate):
    """Score one candidate's trades into (return, sharpe, pf, n_trades, n_assets,
    sl_hit_rate). `return` honours D2 weighting when requested."""
    sid = cand.strategy.id
    n_raw = int(len(trades))
    sl_hit = (float(trades["exit_reason"].isin(STOP_REASONS).mean())
              if n_raw else np.nan)
    pa = per_asset_metrics(trades, chosen_data, window, group_of)
    agg = aggregate_strategies(pa)
    if agg.empty or sid not in set(agg["strategy_id"]):
        return dict(ret=np.nan, sharpe=np.nan, pf=np.nan, n_trades=n_raw,
                    n_assets=0, sl_hit=sl_hit)
    row = agg[agg["strategy_id"] == sid].iloc[0]
    ret = (D.weighted_cross_asset_return(pa, sid) if cand.weighted
           else float(row["cross_asset_avg_return"]))
    return dict(ret=ret, sharpe=float(row["cross_asset_sharpe"]),
                pf=float(row["avg_profit_factor"]),
                n_trades=int(row["total_trades"]), n_assets=int(row["n_assets"]),
                sl_hit=sl_hit)


def _eval_window(cand: Candidate, full, reduced, group_of, window):
    data = reduced if cand.universe == "ex_weak" else full
    trades = run_backtests([cand.strategy], data, date_mask=window,
                           show_progress=False)
    m = _metrics(trades, data, window, group_of, cand)
    return m, trades


def evaluate(candidates, bases, full, reduced, group_of,
             is_window=oosv.IS_WINDOW, oos_window=oosv.OOS_WINDOW,
             progress=True) -> pd.DataFrame:
    """IS + OOS metrics for every candidate AND every base, in one frame."""
    base_cands = [Candidate(b.id, b.id, "base", b) for b in bases]
    all_cands = base_cands + list(candidates)
    for cand in all_cands:                              # ensure group lookups work
        group_of.setdefault(cand.strategy.id, cand.strategy.group)
    rows = []
    it = all_cands
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(all_cands, desc="eval IS+OOS")
        except Exception:
            pass
    for cand in it:
        is_m, _ = _eval_window(cand, full, reduced, group_of, is_window)
        oos_m, _ = _eval_window(cand, full, reduced, group_of, oos_window)
        rows.append(dict(
            id=cand.id, base_id=cand.base_id, module=cand.module,
            universe=cand.universe, weighted=cand.weighted,
            is_return=is_m["ret"], is_sharpe=is_m["sharpe"], is_pf=is_m["pf"],
            is_trades=is_m["n_trades"],
            oos_return=oos_m["ret"], oos_sharpe=oos_m["sharpe"], oos_pf=oos_m["pf"],
            oos_trades=oos_m["n_trades"], oos_assets=oos_m["n_assets"],
            oos_sl_hit=oos_m["sl_hit"], is_sl_hit=is_m["sl_hit"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# comparison / verdict
# ---------------------------------------------------------------------------
def _verdict(d_ret, c_sharpe, b_sharpe, c_pf, b_pf, c_tr, b_tr):
    if not np.isfinite(d_ret) or d_ret <= 0:
        return "reject"
    better_risk = ((np.isfinite(c_sharpe) and np.isfinite(b_sharpe) and c_sharpe > b_sharpe)
                   or (np.isfinite(c_pf) and np.isfinite(b_pf) and c_pf > b_pf))
    collapsed = (b_tr > 0 and c_tr < 0.5 * b_tr)
    if better_risk and not collapsed:
        return "adopt"
    return "hold"


def build_comparison(eval_df: pd.DataFrame) -> pd.DataFrame:
    """One row per non-base candidate vs its base, with IS gate + OOS verdict."""
    base = eval_df[eval_df["module"] == "base"].set_index("base_id")
    rows = []
    for _, c in eval_df[eval_df["module"] != "base"].iterrows():
        if c["base_id"] not in base.index:
            continue
        b = base.loc[c["base_id"]]
        is_gate = bool(np.isfinite(c["is_return"]) and c["is_return"] > b["is_return"])
        d_ret = c["oos_return"] - b["oos_return"]
        verdict = _verdict(d_ret, c["oos_sharpe"], b["oos_sharpe"],
                           c["oos_pf"], b["oos_pf"], c["oos_trades"], b["oos_trades"])
        rows.append(dict(
            strategy_id=c["id"], base_id=c["base_id"], module=c["module"],
            base_oos_return=b["oos_return"], cand_oos_return=c["oos_return"],
            delta_oos_return=d_ret,
            base_oos_sharpe=b["oos_sharpe"], cand_oos_sharpe=c["oos_sharpe"],
            base_oos_pf=b["oos_pf"], cand_oos_pf=c["oos_pf"],
            base_oos_trades=int(b["oos_trades"]), cand_oos_trades=int(c["oos_trades"]),
            is_return_base=b["is_return"], is_return_cand=c["is_return"],
            is_gate_pass=is_gate, verdict=verdict))
    out = pd.DataFrame(rows)
    return out.sort_values(["base_id", "delta_oos_return"],
                           ascending=[True, False]).reset_index(drop=True)


def module_effectiveness(comparison: pd.DataFrame) -> pd.DataFrame:
    """Average OOS-return change (pp) contributed by each module, over ALL its
    candidates (the honest average, not cherry-picked winners)."""
    rows = []
    for mod, g in comparison.groupby("module"):
        rows.append(dict(
            module=mod, n_candidates=len(g),
            avg_delta_oos=float(g["delta_oos_return"].mean()),
            median_delta_oos=float(g["delta_oos_return"].median()),
            n_adopt=int((g["verdict"] == "adopt").sum()),
            n_hold=int((g["verdict"] == "hold").sum()),
            n_reject=int((g["verdict"] == "reject").sum()),
            n_is_gate=int(g["is_gate_pass"].sum())))
    return pd.DataFrame(rows).sort_values("avg_delta_oos", ascending=False).reset_index(drop=True)


def best_per_base(comparison: pd.DataFrame, eval_df: pd.DataFrame, bases) -> pd.DataFrame:
    """For each base, the adopted candidate with the highest OOS return (or the
    original if nothing beats it on OOS)."""
    base = eval_df[eval_df["module"] == "base"].set_index("base_id")
    rows = []
    for b in bases:
        bid = b.id
        br = base.loc[bid]
        adopted = comparison[(comparison["base_id"] == bid)
                             & (comparison["verdict"] == "adopt")
                             & (comparison["is_gate_pass"])]
        if not adopted.empty:
            best = adopted.sort_values("cand_oos_return", ascending=False).iloc[0]
            rows.append(dict(
                base_id=bid, tier=RC.TARGET_TIER.get(bid, ""),
                best_id=best["strategy_id"], improved=True,
                base_oos_return=br["oos_return"], best_oos_return=best["cand_oos_return"],
                delta=best["cand_oos_return"] - br["oos_return"],
                base_oos_sharpe=br["oos_sharpe"], best_oos_sharpe=best["cand_oos_sharpe"],
                base_oos_pf=br["oos_pf"], best_oos_pf=best["cand_oos_pf"]))
        else:
            rows.append(dict(
                base_id=bid, tier=RC.TARGET_TIER.get(bid, ""),
                best_id=bid, improved=False,
                base_oos_return=br["oos_return"], best_oos_return=br["oos_return"],
                delta=0.0, base_oos_sharpe=br["oos_sharpe"],
                best_oos_sharpe=br["oos_sharpe"], base_oos_pf=br["oos_pf"],
                best_oos_pf=br["oos_pf"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# walk-forward + random on the OOS winners
# ---------------------------------------------------------------------------
def validate_winners(winner_cands, full, reduced, group_of) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """5-window walk-forward + 1000x random-entry on the adopted winners.
    D1 winners (reduced universe) are validated on the reduced universe."""
    if not winner_cands:
        return pd.DataFrame(), pd.DataFrame()
    wf_rows = []
    for universe in ("full", "ex_weak"):
        grp = [c.strategy for c in winner_cands if c.universe == universe]
        if not grp:
            continue
        data = reduced if universe == "ex_weak" else full
        wf = oosv.walk_forward(grp, data, group_of, show_progress=False)
        wf_rows.append(wf)
    wf_df = pd.concat(wf_rows, ignore_index=True) if wf_rows else pd.DataFrame()

    # random benchmark (full-universe winners only — null model is per-asset)
    rnd_rows = []
    full_winners = [c.strategy for c in winner_cands if c.universe == "full"]
    if full_winners:
        trades = run_backtests(full_winners, full, date_mask=oosv.OOS_WINDOW,
                               show_progress=False)
        pa = per_asset_metrics(trades, full, oosv.OOS_WINDOW, group_of)
        rnd = rb.random_benchmark(full_winners, full, pa, trades=trades,
                                  date_mask=oosv.OOS_WINDOW, n_sims=1000)
        rnd_rows.append(rnd)
    rnd_df = pd.concat(rnd_rows, ignore_index=True) if rnd_rows else pd.DataFrame()
    return wf_df, rnd_df


# ---------------------------------------------------------------------------
# portfolios
# ---------------------------------------------------------------------------
def run_portfolios(full, date_mask=oosv.OOS_WINDOW, portfolios=PORTFOLIOS) -> "tuple[pd.DataFrame, dict]":
    rows, trade_books = [], {}
    for name, ids in portfolios.items():
        try:
            strats = [RC.reconstruct(i) for i in ids]
        except Exception as e:                                   # noqa: BLE001
            rows.append(dict(name=name, error=str(e)))
            continue
        res = G.run_portfolio(strats, full, date_mask=date_mask, name=name)
        m = res["metrics"]
        rows.append(dict(name=name, members="; ".join(ids),
                         total_return=m["total_return"],
                         avg_asset_return=m["avg_asset_return"],
                         sharpe=m["sharpe"], max_drawdown=m["max_drawdown"],
                         n_trades=m["n_trades"], win_rate=m["win_rate"],
                         profit_factor=m["profit_factor"], n_assets=m["n_assets"]))
        trade_books[name] = res["trades"]
    return pd.DataFrame(rows), trade_books
