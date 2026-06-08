"""
dashboard_v5.py — Phase 5 per-strategy tuning dashboard (8 charts).

Defensive: each chart isolates failures into a placeholder note. Inputs are the
combined IS grid, the v3/v4/v5 comparison, the holding-period WR table, WF/random
results, the June spot frame, and {base_id: Strategy} / {base_id: StrategyV5}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from backtester_v3 import run_backtests
import oos_validator as oosv

CAT = px.colors.qualitative.Set2
STOP = ("stop_loss", "atr_stop")


def _html(fig, first=False):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


def _ph(t, m):
    return f"<div style='padding:20px;color:#888'><h3>{t}</h3><p>{m}</p></div>"


def _safe(fn, t, first=False):
    try:
        return _html(fn(), first=first)
    except Exception as e:                                       # noqa: BLE001
        return _ph(t, f"(chart unavailable: {type(e).__name__}: {e})")


def _oos_trades(strat, data):
    return run_backtests([strat], data, date_mask=oosv.OOS_WINDOW, show_progress=False)


def build_dashboard(out_path, data, combined_grid, comp, trade_wr, wf_df, rnd_df,
                    spot_df, base_strats, best_cands):

    # 1. v3 / v4 / v5 comparison -------------------------------------------
    def c1():
        d = comp.copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["base_id"], y=d["v3_oos"], name="v3 original",
                             marker_color="#9aa5b1"))
        fig.add_trace(go.Bar(x=d["base_id"], y=d["v4_best_oos"], name="v4 best",
                             marker_color="#74b9ff"))
        fig.add_trace(go.Bar(x=d["base_id"], y=d["v5_best_oos"], name="v5 best",
                             marker_color="#0984e3"))
        fig.update_layout(barmode="group", height=520, xaxis_tickangle=-25,
                          title="1. OOS Return — v3 vs v4 vs v5 (per strategy)",
                          yaxis_title="OOS return %")
        return fig

    # 2. IS grid search space ----------------------------------------------
    def c2():
        d = combined_grid[np.isfinite(combined_grid["is_return"])]
        fig = px.scatter(d, x="is_trades", y="is_return", color="base",
                         color_discrete_sequence=CAT, hover_name="desc",
                         title="2. IS Grid Search Space — return vs #trades (per strategy)")
        fig.update_layout(height=560, xaxis_title="IS trades", yaxis_title="IS return %")
        return fig

    # 3. holding-period win rate -------------------------------------------
    def c3():
        order = ["0d", "1d", "2-3d", "4-5d", "6-10d", "10d+"]
        d = trade_wr.copy()
        d["bucket"] = pd.Categorical(d["bucket"], categories=order, ordered=True)
        d = d.sort_values("bucket")
        fig = px.bar(d, x="bucket", y=d["wr"] * 100, color="strategy_id",
                     barmode="group", title="3. Win-Rate by Holding Period (0~1d diagnosis)")
        fig.update_layout(height=520, yaxis_title="win rate %", xaxis_title="holding days")
        return fig

    # 4. SL hit-rate base vs v5-best ---------------------------------------
    def c4():
        rows = []
        for sid, base in base_strats.items():
            tb = _oos_trades(base, data)
            rows.append(dict(base_id=sid, kind="v3 base",
                             sl=float(tb["exit_reason"].isin(STOP).mean()) * 100 if len(tb) else np.nan))
            if sid in best_cands:
                tv = _oos_trades(best_cands[sid], data)
                rows.append(dict(base_id=sid, kind="v5 best",
                                 sl=float(tv["exit_reason"].isin(STOP).mean()) * 100 if len(tv) else np.nan))
        d = pd.DataFrame(rows)
        fig = px.bar(d, x="base_id", y="sl", color="kind", barmode="group",
                     color_discrete_sequence=["#e17055", "#00b894"],
                     title="4. Stop-Hit Rate — v3 base vs v5 best")
        fig.update_layout(height=480, xaxis_tickangle=-25, yaxis_title="% trades exited on a stop")
        return fig

    # 5. equity curves base vs v5-best -------------------------------------
    def c5():
        fig = go.Figure()
        for sid, cand in best_cands.items():
            for label, strat, dash_ in [(f"{sid} v3", base_strats[sid], "dot"),
                                        (f"{sid} v5", cand, "solid")]:
                t = _oos_trades(strat, data)
                if t.empty:
                    continue
                t = t.sort_values("exit_date")
                fig.add_trace(go.Scatter(x=t["exit_date"], y=(t["pnl_pct"] * 100).cumsum(),
                                         mode="lines", name=label, line=dict(dash=dash_)))
        fig.update_layout(height=560, title="5. OOS Equity — v3 base (dotted) vs v5 best (solid)",
                          yaxis_title="cumulative return %")
        return fig

    # 6. per-asset return base vs v5-best (exclusion effect) ---------------
    def c6():
        from scorer_v3 import per_asset_metrics
        # pick a strategy that excludes an asset if any, else the biggest improver
        sid = None
        for s, c in best_cands.items():
            if getattr(c, "exclude_assets", ()):
                sid = s; break
        if sid is None and best_cands:
            sid = next(iter(best_cands))
        if sid is None:
            raise ValueError("no v5 winners")
        frames = []
        for label, strat in [("v3 base", base_strats[sid]), ("v5 best", best_cands[sid])]:
            t = _oos_trades(strat, data)
            pa = per_asset_metrics(t, data, oosv.OOS_WINDOW, {strat.id: strat.group})
            s = pa.set_index("symbol")["total_return"].rename(label)
            frames.append(s)
        d = pd.concat(frames, axis=1).reset_index().melt(id_vars="symbol",
                                                         var_name="kind", value_name="ret")
        fig = px.bar(d, x="symbol", y="ret", color="kind", barmode="group",
                     color_discrete_sequence=["#b2bec3", "#0984e3"],
                     title=f"6. Per-Asset OOS Return — {sid} (v3 vs v5)")
        fig.update_layout(height=480, yaxis_title="summed trade return %")
        return fig

    # 7. June 2026 spot -----------------------------------------------------
    def c7():
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
        fig.update_layout(height=480, title="7. June 2026 Spot — BTC entries (base + v5)")
        return fig

    # 8. walk-forward windows ----------------------------------------------
    def c8():
        if wf_df is None or wf_df.empty:
            raise ValueError("no walk-forward data")
        cols = [c for c in wf_df.columns if c.startswith("oos_w")]
        d = wf_df.set_index("strategy_id")[cols]
        fig = px.imshow(d, color_continuous_scale="RdYlGn", aspect="auto",
                        color_continuous_midpoint=0,
                        title="8. Walk-Forward OOS Return by Window — v5 winners")
        fig.update_layout(height=420)
        return fig

    charts = [("1", c1), ("2", c2), ("3", c3), ("4", c4),
              ("5", c5), ("6", c6), ("7", c7), ("8", c8)]
    parts = [_safe(fn, t, first=(i == 0)) for i, (t, fn) in enumerate(charts)]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Phase 5 — Per-Strategy Tuning</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#fafafa}}
h1{{background:#0b3d2e;color:#fff;padding:18px 24px;margin:0}}
.grid{{padding:12px}} .card{{background:#fff;margin:12px 0;border-radius:8px;
box-shadow:0 1px 4px rgba(0,0,0,.08);padding:6px}}</style></head>
<body><h1>🎯 Phase 5 — Per-Strategy Parameter Tuning (v5)</h1><div class="grid">
{''.join(f'<div class="card">{p}</div>' for p in parts)}
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
