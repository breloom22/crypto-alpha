"""
dashboard_v3.py — interactive Plotly dashboard (Part 9.2, 12 charts).

Defensive by design: each chart is built in isolation so one failure degrades to
a placeholder note rather than killing the whole report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

CAT_COLORS = px.colors.qualitative.Set2


def _fig_html(fig, first=False):
    return fig.to_html(full_html=False,
                       include_plotlyjs="cdn" if first else False)


def _placeholder(title, msg):
    return f"<div style='padding:20px;color:#888'><h3>{title}</h3><p>{msg}</p></div>"


def _safe(fn, title, first=False):
    try:
        return _fig_html(fn(), first=first)
    except Exception as e:                                  # noqa: BLE001
        return _placeholder(title, f"(chart unavailable: {type(e).__name__}: {e})")


# ---------------------------------------------------------------------------
def build_dashboard(out_path, data, ranked_is, oos_df, wf_df, random_df,
                    sens_df, per_asset_is, trades_full, spot_df,
                    group_perf=None):
    top5 = ranked_is["strategy_id"].head(5).tolist() if not ranked_is.empty else []
    top3 = ranked_is["strategy_id"].head(3).tolist() if not ranked_is.empty else []

    # 1. IS ranking ---------------------------------------------------------
    def c1():
        d = ranked_is.head(30)
        fig = px.bar(d, x="composite_score", y="strategy_id", color="category",
                     orientation="h", color_discrete_sequence=CAT_COLORS,
                     title="1. IS Composite Ranking (Top 30)")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=750)
        return fig

    # 2. IS vs OOS scatter --------------------------------------------------
    def c2():
        d = oos_df.head(30)
        fig = px.scatter(d, x="is_return", y="oos_return", text="strategy_id",
                         color="category" if "category" in d else None,
                         color_discrete_sequence=CAT_COLORS,
                         title="2. IS vs OOS Return (Top 30)")
        lim = pd.concat([d["is_return"], d["oos_return"]]).abs().max() or 1
        fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                      line=dict(dash="dash", color="grey"))
        fig.update_traces(textposition="top center", marker=dict(size=9))
        fig.update_layout(height=600)
        return fig

    # 3. Return vs Sharpe ---------------------------------------------------
    def c3():
        d = ranked_is.copy()
        d["size"] = d["total_trades"].clip(lower=1)
        fig = px.scatter(d, x="cross_asset_avg_return", y="cross_asset_sharpe",
                         size="size", color="category",
                         color_discrete_sequence=CAT_COLORS,
                         hover_name="strategy_id",
                         title="3. Return vs Sharpe (size = #trades)")
        fig.update_layout(height=600)
        return fig

    # 4. Equity curve top 5 + BTC B&H --------------------------------------
    def c4():
        fig = go.Figure()
        for sid in top5:
            t = trades_full[trades_full["strategy_id"] == sid].sort_values("exit_date")
            if t.empty:
                continue
            eq = (t["pnl_pct"] * 100).cumsum()
            fig.add_trace(go.Scatter(x=t["exit_date"], y=eq, mode="lines", name=sid))
        btc = data.get("BTC")
        if btc is not None:
            bh = (btc["close"] / btc["close"].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=btc.index, y=bh, mode="lines",
                                     name="BTC B&H", line=dict(dash="dot", color="black")))
        fig.update_layout(title="4. Equity Curve — Top 5 (cum %) vs BTC B&H",
                          height=600, yaxis_title="cumulative return %")
        return fig

    # 5. Asset x strategy heatmap ------------------------------------------
    def c5():
        top20 = ranked_is["strategy_id"].head(20).tolist()
        d = per_asset_is[per_asset_is["strategy_id"].isin(top20)]
        piv = d.pivot_table(index="strategy_id", columns="symbol",
                            values="total_return", aggfunc="mean").reindex(top20)
        fig = px.imshow(piv, color_continuous_scale="RdYlGn", aspect="auto",
                        title="5. Asset x Strategy Return Heatmap (Top 20)",
                        color_continuous_midpoint=0)
        fig.update_layout(height=700)
        return fig

    # 6. Trade return distribution top 5 -----------------------------------
    def c6():
        fig = go.Figure()
        for sid in top5:
            t = trades_full[trades_full["strategy_id"] == sid]
            if t.empty:
                continue
            fig.add_trace(go.Histogram(x=t["pnl_pct"] * 100, name=sid, opacity=0.55,
                                       nbinsx=40))
        fig.update_layout(barmode="overlay", title="6. Trade Return Distribution — Top 5",
                          height=550, xaxis_title="trade return %")
        return fig

    # 7. Win rate vs profit factor -----------------------------------------
    def c7():
        d = ranked_is.copy()
        d["pf"] = d["avg_profit_factor"].clip(upper=5)
        fig = px.scatter(d, x="avg_win_rate", y="pf", color="category",
                         color_discrete_sequence=CAT_COLORS, hover_name="strategy_id",
                         title="7. Win Rate vs Profit Factor")
        fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
        fig.update_layout(height=600, yaxis_title="profit factor (capped 5)")
        return fig

    # 8. Monthly return heatmap top 5 --------------------------------------
    def c8():
        t = trades_full[trades_full["strategy_id"].isin(top5)].copy()
        t["ym"] = pd.to_datetime(t["exit_date"]).dt.to_period("M").astype(str)
        t["year"] = pd.to_datetime(t["exit_date"]).dt.year
        t["month"] = pd.to_datetime(t["exit_date"]).dt.month
        piv = t.pivot_table(index="year", columns="month",
                            values="pnl_pct", aggfunc="sum") * 100
        fig = px.imshow(piv, color_continuous_scale="RdYlGn", aspect="auto",
                        color_continuous_midpoint=0,
                        title="8. Monthly Return Heatmap — Top 5 (sum %)")
        fig.update_layout(height=450)
        return fig

    # 9. Walk-forward results ----------------------------------------------
    def c9():
        cols = [c for c in wf_df.columns if c.startswith("oos_w")]
        d = wf_df.set_index("strategy_id")[cols].head(15)
        fig = px.imshow(d, color_continuous_scale="RdYlGn", aspect="auto",
                        color_continuous_midpoint=0,
                        title="9. Walk-Forward OOS Return by Window (Top 15)")
        fig.update_layout(height=600)
        return fig

    # 10. Parameter sensitivity top 3 --------------------------------------
    def c10():
        d = sens_df[sens_df["strategy_id"].isin(top3)]
        fig = px.line(d, x="value", y="return", color="param",
                      facet_col="strategy_id", markers=True,
                      title="10. Parameter Sensitivity (Top 3)")
        fig.update_xaxes(matches=None)
        fig.update_layout(height=500)
        return fig

    # 11. Signal-group avg performance -------------------------------------
    def c11():
        gp = group_perf
        if gp is None or gp.empty:
            gp = ranked_is.groupby("group").agg(
                ret=("cross_asset_avg_return", "mean"),
                sharpe=("cross_asset_sharpe", "mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=gp["group"], y=gp["ret"], name="avg return %",
                             marker_color="steelblue"))
        fig.add_trace(go.Scatter(x=gp["group"], y=gp["sharpe"], name="avg sharpe",
                                 yaxis="y2", mode="lines+markers", line=dict(color="firebrick")))
        fig.update_layout(title="11. Signal Group Avg Performance (S1-S8)",
                          yaxis=dict(title="avg return %"),
                          yaxis2=dict(title="avg sharpe", overlaying="y", side="right"),
                          height=500)
        return fig

    # 12. 2026-06 timeline --------------------------------------------------
    def c12():
        btc = data.get("BTC")
        fig = go.Figure()
        if btc is not None:
            sub = btc.loc["2026-04-01":]
            fig.add_trace(go.Scatter(x=sub.index, y=sub["close"], mode="lines",
                                     name="BTC close", line=dict(color="black")))
        if spot_df is not None and not spot_df.empty:
            bt = spot_df[spot_df["symbol"] == "BTC"]
            longs = bt[bt["direction"] == "LONG"]
            shorts = bt[bt["direction"] == "SHORT"]
            fig.add_trace(go.Scatter(x=longs["entry_date"], y=longs["entry_price"],
                                     mode="markers", name="LONG entry",
                                     marker=dict(symbol="triangle-up", size=12, color="green")))
            fig.add_trace(go.Scatter(x=shorts["entry_date"], y=shorts["entry_price"],
                                     mode="markers", name="SHORT entry",
                                     marker=dict(symbol="triangle-down", size=12, color="red")))
        fig.update_layout(title="12. June 2026 Timeline — BTC + spot entries", height=500)
        return fig

    charts = [
        ("1. IS Ranking", c1), ("2. IS vs OOS", c2), ("3. Return vs Sharpe", c3),
        ("4. Equity Curve", c4), ("5. Asset Heatmap", c5), ("6. Return Dist", c6),
        ("7. WR vs PF", c7), ("8. Monthly Heatmap", c8), ("9. Walk-Forward", c9),
        ("10. Sensitivity", c10), ("11. Group Perf", c11), ("12. June Timeline", c12),
    ]
    parts = []
    for i, (title, fn) in enumerate(charts):
        parts.append(_safe(fn, title, first=(i == 0)))

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Crypto Alpha Strategy Discovery v3</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#fafafa}}
h1{{background:#1a1a2e;color:#fff;padding:18px 24px;margin:0}}
.grid{{padding:12px}} .card{{background:#fff;margin:12px 0;border-radius:8px;
box-shadow:0 1px 4px rgba(0,0,0,.08);padding:6px}}</style></head>
<body><h1>📊 Crypto Alpha Strategy Discovery v3</h1><div class="grid">
{''.join(f'<div class="card">{p}</div>' for p in parts)}
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
