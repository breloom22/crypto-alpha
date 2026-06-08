"""
dashboard_v4.py — Phase 4 optimization dashboard (8 charts, spec §출력).

Defensive: each chart builds in isolation so one failure degrades to a note
instead of killing the report. Mirrors dashboard_v3's _safe pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from backtester_v3 import run_backtests
import oos_validator as oosv

CAT = px.colors.qualitative.Set2
MOD_NAMES = {"A": "A ATR SL/TP", "B": "B Break-Even", "C": "C Whipsaw",
             "D": "D Asset Sel", "E": "E Alt LONG", "F": "F Stepped Trail",
             "combo": "Combos"}


def _html(fig, first=False):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


def _ph(title, msg):
    return f"<div style='padding:20px;color:#888'><h3>{title}</h3><p>{msg}</p></div>"


def _safe(fn, title, first=False):
    try:
        return _html(fn(), first=first)
    except Exception as e:                                       # noqa: BLE001
        return _ph(title, f"(chart unavailable: {type(e).__name__}: {e})")


def build_dashboard(out_path, data, eval_df, comparison, mod_eff, best_df,
                    wf_df, rnd_df, pf_df, portfolio_books, spot_df,
                    best_strats=None):
    best_strats = best_strats or {}

    # 1. Original vs best-improved OOS return ------------------------------
    def c1():
        d = best_df.copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["base_id"], y=d["base_oos_return"],
                             name="Original", marker_color="#9aa5b1"))
        fig.add_trace(go.Bar(x=d["base_id"], y=d["best_oos_return"],
                             name="Best v4", marker_color="#2e86de"))
        fig.update_layout(barmode="group", height=520,
                          title="1. Original vs Best-Improved OOS Return (per base)",
                          yaxis_title="OOS return %", xaxis_tickangle=-30)
        return fig

    # 2. Module effectiveness ---------------------------------------------
    def c2():
        d = mod_eff.copy()
        d["label"] = d["module"].map(lambda m: MOD_NAMES.get(m, m))
        d["color"] = np.where(d["avg_delta_oos"] >= 0, "#27ae60", "#c0392b")
        fig = go.Figure(go.Bar(x=d["label"], y=d["avg_delta_oos"],
                               marker_color=d["color"],
                               text=d["n_adopt"].astype(str) + "/" + d["n_candidates"].astype(str),
                               textposition="outside"))
        fig.update_layout(height=480,
                          title="2. Module Effectiveness — avg OOS return change (pp), label=adopt/total",
                          yaxis_title="avg Δ OOS return (pp)")
        fig.add_hline(y=0, line_color="grey")
        return fig

    # 3. SL hit-rate: fixed vs ATR ----------------------------------------
    def c3():
        base = eval_df[eval_df["module"] == "base"][["base_id", "oos_sl_hit"]]
        a = eval_df[eval_df["module"] == "A"].copy()
        a_by_base = a.groupby("base_id")["oos_sl_hit"].mean().reset_index()
        m = base.merge(a_by_base, on="base_id", suffixes=("_fixed", "_atr"))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=m["base_id"], y=m["oos_sl_hit_fixed"] * 100,
                             name="Original (fixed/by-rule)", marker_color="#e67e22"))
        fig.add_trace(go.Bar(x=m["base_id"], y=m["oos_sl_hit_atr"] * 100,
                             name="ATR SL (avg A1-A4)", marker_color="#16a085"))
        fig.update_layout(barmode="group", height=480, xaxis_tickangle=-30,
                          title="3. Stop-Hit Rate — Original vs ATR (Improvement A)",
                          yaxis_title="% of trades exited on a stop")
        return fig

    # 4. Asset-level return change: original vs D -------------------------
    def c4():
        # original best base by OOS, asset-level returns on full vs reduced
        sid = best_df.sort_values("base_oos_return", ascending=False).iloc[0]["base_id"]
        from improvements.reconstruct import reconstruct
        from scorer_v3 import per_asset_metrics
        strat = reconstruct(sid)
        t = run_backtests([strat], data, date_mask=oosv.OOS_WINDOW, show_progress=False)
        pa = per_asset_metrics(t, data, oosv.OOS_WINDOW, {sid: strat.group})
        pa = pa.sort_values("total_return")
        fig = go.Figure(go.Bar(x=pa["symbol"], y=pa["total_return"],
                               marker_color=np.where(pa["total_return"] >= 0, "#27ae60", "#c0392b")))
        fig.update_layout(height=460,
                          title=f"4. Per-Asset OOS Return — {sid} (D drops BTC/SOL)",
                          yaxis_title="summed trade return %")
        return fig

    # 5. Equity curves: original Tier-S vs best vs portfolios -------------
    def c5():
        fig = go.Figure()
        # original Tier-S
        from improvements.reconstruct import reconstruct
        ts = reconstruct("SEQ_S1.11_S+S4.04_L")
        t = run_backtests([ts], data, date_mask=oosv.OOS_WINDOW, show_progress=False)
        if not t.empty:
            t = t.sort_values("exit_date")
            fig.add_trace(go.Scatter(x=t["exit_date"], y=(t["pnl_pct"] * 100).cumsum(),
                                     mode="lines", name="Original Tier-S"))
        for name, book in (portfolio_books or {}).items():
            if book is None or book.empty:
                continue
            b = book.sort_values("exit_date")
            n_strat = max(1, b["strategy_id"].nunique())
            fig.add_trace(go.Scatter(x=b["exit_date"],
                                     y=(b["pnl_pct"] * 100 / n_strat).cumsum(),
                                     mode="lines", name=name))
        fig.update_layout(height=560, title="5. OOS Equity Curves — Tier-S vs Portfolios",
                          yaxis_title="cumulative return % (non-compounded)")
        return fig

    # 6. Monthly return heatmap: original vs best -------------------------
    def c6():
        from improvements.reconstruct import reconstruct
        # pick the base with the biggest adopted improvement
        imp = best_df[best_df["improved"]]
        sid = (imp.sort_values("delta", ascending=False).iloc[0]["base_id"]
               if not imp.empty else best_df.iloc[0]["base_id"])
        pairs = [("original", reconstruct(sid))]
        if sid in best_strats:
            pairs.append(("best v4", best_strats[sid]))
        frames = []
        for label, strat in pairs:
            t = run_backtests([strat], data, date_mask=oosv.OOS_WINDOW, show_progress=False)
            if t.empty:
                continue
            t = t.copy()
            t["ym"] = pd.to_datetime(t["exit_date"]).dt.to_period("M").astype(str)
            s = t.groupby("ym")["pnl_pct"].sum() * 100
            s.name = label
            frames.append(s)
        if not frames:
            raise ValueError("no monthly data")
        piv = pd.concat(frames, axis=1).T
        fig = px.imshow(piv, color_continuous_scale="RdYlGn", aspect="auto",
                        color_continuous_midpoint=0,
                        title=f"6. Monthly OOS Return — {sid}: original vs best v4")
        fig.update_layout(height=340)
        return fig

    # 7. IS vs OOS consistency scatter (overfit detector) ----------------
    def c7():
        d = eval_df[np.isfinite(eval_df["is_return"]) & np.isfinite(eval_df["oos_return"])].copy()
        fig = px.scatter(d, x="is_return", y="oos_return", color="module",
                         color_discrete_sequence=CAT, hover_name="id",
                         title="7. IS vs OOS Return (overfit check — points below diagonal degraded)")
        lim = pd.concat([d["is_return"], d["oos_return"]]).abs().max() or 1
        fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                      line=dict(dash="dash", color="grey"))
        fig.update_layout(height=560)
        return fig

    # 8. June 2026 spot — entries on BTC ----------------------------------
    def c8():
        btc = data.get("BTC")
        fig = go.Figure()
        if btc is not None:
            sub = btc.loc["2026-04-01":]
            fig.add_trace(go.Scatter(x=sub.index, y=sub["close"], mode="lines",
                                     name="BTC close", line=dict(color="black")))
        if spot_df is not None and not spot_df.empty:
            bt = spot_df[spot_df["symbol"] == "BTC"]
            for d, sym, col in [("LONG", "triangle-up", "green"),
                                ("SHORT", "triangle-down", "red")]:
                sel = bt[bt["direction"] == d]
                fig.add_trace(go.Scatter(x=sel["entry_date"], y=sel["entry_price"],
                                         mode="markers", name=f"{d} entry",
                                         marker=dict(symbol=sym, size=11, color=col)))
        fig.update_layout(height=480, title="8. June 2026 Spot — BTC entries (base + best)")
        return fig

    charts = [("1", c1), ("2", c2), ("3", c3), ("4", c4),
              ("5", c5), ("6", c6), ("7", c7), ("8", c8)]
    parts = [_safe(fn, t, first=(i == 0)) for i, (t, fn) in enumerate(charts)]

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Phase 4 — Strategy Optimization</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#fafafa}}
h1{{background:#16213e;color:#fff;padding:18px 24px;margin:0}}
.grid{{padding:12px}} .card{{background:#fff;margin:12px 0;border-radius:8px;
box-shadow:0 1px 4px rgba(0,0,0,.08);padding:6px}}</style></head>
<body><h1>🛠️ Phase 4 — Strategy Optimization (v4)</h1><div class="grid">
{''.join(f'<div class="card">{p}</div>' for p in parts)}
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
