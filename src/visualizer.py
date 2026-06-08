"""
visualizer.py — Phase 6: interactive Plotly dashboard (dashboard.html).

Seven charts (self-contained HTML, works offline):
  1. Crash-event heatmap            (asset x month, colour = severity)
  2. Indicator ranking bar          (composite score, colour = category)
  3. Precision-Recall scatter       (size = lead time)
  4. Top-5 signals over BTC price   (markers + shaded crash legs)
  5. Ensemble F1 vs false-alarm     (top-10 ensembles)
  6. Per-asset F1 heatmap           (indicator x asset)
  7. Parameter sensitivity          (top-3 indicators, value/default vs F1)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

CATEGORY_COLORS = {
    "momentum": "#1f77b4", "trend": "#ff7f0e", "volatility": "#2ca02c",
    "volume": "#d62728", "cross_asset": "#9467bd",
}
PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]


# ---- chart 1 ----------------------------------------------------------------
def chart_event_heatmap(clusters, assets):
    cl = clusters.copy()
    cl["month"] = pd.to_datetime(cl["cluster_start"]).dt.to_period("M").astype(str)
    months = pd.period_range(cl["cluster_start"].min().to_period("M") if len(cl) else "2022-01",
                             cl["cluster_start"].max().to_period("M") if len(cl) else "2026-06",
                             freq="M").astype(str).tolist()
    z = []
    for a in assets:
        row = []
        sub = cl[cl.symbol == a]
        worst = sub.groupby("month")["peak_decline"].min()
        for m in months:
            row.append(-worst[m] * 100 if m in worst.index else np.nan)
        z.append(row)
    fig = go.Figure(go.Heatmap(
        z=z, x=months, y=assets, colorscale="Reds", zmin=0, zmax=60,
        colorbar=dict(title="Peak<br>decline %"),
        hovertemplate="%{y} %{x}<br>peak decline: -%{z:.1f}%<extra></extra>"))
    fig.update_layout(title="1 · Crash-event heatmap (peak decline by asset & month)",
                      xaxis_title="month", yaxis_title="asset", height=420)
    return fig


# ---- chart 2 ----------------------------------------------------------------
def chart_ranking_bar(scores):
    s = scores.sort_values("composite_score", ascending=True)
    colors = [CATEGORY_COLORS.get(c, "#888") for c in s["category"]]
    fig = go.Figure(go.Bar(
        x=s["composite_score"], y=s["indicator"], orientation="h",
        marker_color=colors,
        customdata=np.stack([s["name"], s["category"], s["f1"]], axis=-1),
        hovertemplate="<b>%{y}</b> %{customdata[0]}<br>category: %{customdata[1]}"
                      "<br>composite: %{x:.3f}<br>F1: %{customdata[2]:.3f}<extra></extra>"))
    # legend proxies
    for cat, col in CATEGORY_COLORS.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=col, name=cat, showlegend=True))
    fig.update_layout(title="2 · Indicator ranking by composite score (colour = category)",
                      xaxis_title="composite score", height=720, barmode="overlay",
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig


# ---- chart 3 ----------------------------------------------------------------
def chart_pr_scatter(scores, base_rate=None):
    lead = scores["avg_lead_time"].fillna(0.0)        # robust: never NaN size
    size = 6 + (lead - lead.min()) / (lead.max() - lead.min() + 1e-9) * 28
    fig = go.Figure()
    for cat, col in CATEGORY_COLORS.items():
        sub = scores[scores.category == cat]
        if sub.empty:
            continue
        ls = lead.loc[sub.index]
        sz = size.loc[sub.index]
        fig.add_trace(go.Scatter(
            x=sub["recall"], y=sub["precision"], mode="markers+text",
            text=sub["indicator"], textposition="top center", name=cat,
            marker=dict(size=sz, color=col, line=dict(width=1, color="white"), opacity=0.8),
            customdata=np.stack([sub["name"], ls], axis=-1),
            hovertemplate="<b>%{text}</b> %{customdata[0]}<br>recall %{x:.2f}"
                          " · precision %{y:.2f}<br>lead %{customdata[1]:.1f}d<extra></extra>"))
    if base_rate is not None and base_rate == base_rate:
        fig.add_hline(y=base_rate, line_dash="dash", line_color="grey",
                      annotation_text=f"base rate {base_rate:.2f} (random signal)",
                      annotation_position="top left")
    fig.update_layout(title="3 · Precision vs Recall (bubble size = avg lead time)",
                      xaxis_title="recall", yaxis_title="precision", height=560)
    return fig


# ---- chart 4 ----------------------------------------------------------------
def chart_btc_overlay(data, clusters, registry, scores, top_n=5):
    btc = data["BTC"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=btc.index, y=btc["close"], mode="lines",
                             name="BTC close", line=dict(color="#333", width=1.4)))
    # shade BTC crash legs
    for _, r in clusters[clusters.symbol == "BTC"].iterrows():
        fig.add_vrect(x0=r["cluster_start"], x1=max(r["cluster_end"], r["cluster_start"]),
                      fillcolor="red", opacity=0.10, line_width=0, layer="below")
    top_ids = scores["indicator"].head(top_n).tolist()
    for i, iid in enumerate(top_ids):
        ind = registry[iid]
        sig = (ind.signals(data).reindex(btc.index).fillna(False)
               if ind.cross_asset else ind.signals(btc))
        d = btc.index[sig.values]
        fig.add_trace(go.Scatter(
            x=d, y=btc.loc[d, "close"], mode="markers", name=f"{iid} {ind.name}",
            marker=dict(size=7, color=PALETTE[i % len(PALETTE)], symbol="triangle-down",
                        line=dict(width=0.5, color="white")),
            hovertemplate=f"{iid} signal<br>%{{x|%Y-%m-%d}}<extra></extra>"))
    fig.update_layout(title="4 · Top-5 indicator signals over BTC price (red bands = crash legs)",
                      xaxis_title="date", yaxis_title="BTC close (USDT)", height=560,
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ---- chart 5 ----------------------------------------------------------------
def chart_ensemble(ensemble, top_n=10):
    e = ensemble.head(top_n)
    rec = e["recall"].fillna(0)
    size = 8 + (rec - rec.min()) / (rec.max() - rec.min() + 1e-9) * 26
    fig = go.Figure(go.Scatter(
        x=e["false_alarm_rate"], y=e["f1"], mode="markers+text",
        text=[m.replace("+", "+<br>") for m in e["members"]], textposition="top center",
        marker=dict(size=size, color=e["recall"], colorscale="Viridis",
                    colorbar=dict(title="recall"), line=dict(width=1, color="white")),
        customdata=np.stack([e["members"], e["strategy"], e["recall"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b> [%{customdata[1]}]<br>F1 %{y:.3f}"
                      " · false-alarm %{x:.2f} · recall %{customdata[2]:.2f}<extra></extra>"))
    fig.update_layout(title="5 · Top-10 ensembles: F1 vs False-Alarm Rate (colour/size = recall)",
                      xaxis_title="false alarm rate", yaxis_title="F1", height=560)
    return fig


# ---- chart 6 ----------------------------------------------------------------
def chart_asset_heatmap(backtest_full, assets):
    allc = backtest_full[backtest_full.crash_type == "ALL"]
    piv = allc.pivot_table(index="symbol", columns="indicator", values="f1")
    order = (allc.groupby("indicator")["f1"].mean().sort_values(ascending=False).index.tolist())
    piv = piv.reindex(index=[a for a in assets if a in piv.index], columns=order)
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index, colorscale="RdYlGn", zmin=0, zmax=0.8,
        colorbar=dict(title="F1"),
        hovertemplate="%{x} on %{y}<br>F1 %{z:.3f}<extra></extra>"))
    fig.update_layout(title="6 · Per-asset F1 heatmap (which indicator works on which asset)",
                      xaxis_title="indicator", yaxis_title="asset", height=460)
    return fig


# ---- chart 7 ----------------------------------------------------------------
def chart_sensitivity(sensitivity, scores, top_n=3):
    top_ids = scores["indicator"].head(top_n).tolist()
    fig = go.Figure()
    for i, iid in enumerate(top_ids):
        sub = sensitivity[sensitivity.indicator == iid].copy()
        if sub.empty:
            continue
        default_val = sub.loc[sub.is_default, "value"]
        dv = float(default_val.iloc[0]) if len(default_val) else sub["value"].median()
        sub = sub.sort_values("value")
        ratio = sub["value"] / dv
        fig.add_trace(go.Scatter(
            x=ratio, y=sub["f1"], mode="lines+markers",
            name=f"{iid} ({sub['param'].iloc[0]})", line=dict(color=PALETTE[i]),
            hovertemplate=f"{iid}<br>value %{{customdata:.2f}} (×%{{x:.2f}})<br>F1 %{{y:.3f}}<extra></extra>",
            customdata=sub["value"]))
    fig.add_vline(x=1.0, line_dash="dash", line_color="grey")
    fig.update_layout(title="7 · Parameter sensitivity (top-3 indicators, value ÷ default vs F1)",
                      xaxis_title="parameter value ÷ default", yaxis_title="F1", height=480)
    return fig


# ---- assemble ---------------------------------------------------------------
def build_dashboard(data, clusters, scores, backtest_full, ensemble, sensitivity,
                    registry, out_path, meta=None):
    assets = list(data.keys())
    figs = [
        ("Crash events", chart_event_heatmap(clusters, assets)),
        ("Ranking", chart_ranking_bar(scores)),
        ("Precision-Recall", chart_pr_scatter(scores, (meta or {}).get("base_rate"))),
        ("BTC overlay", chart_btc_overlay(data, clusters, registry, scores)),
        ("Ensembles", chart_ensemble(ensemble)),
        ("Per-asset F1", chart_asset_heatmap(backtest_full, assets)),
        ("Sensitivity", chart_sensitivity(sensitivity, scores)),
    ]
    parts = []
    for i, (_, fig) in enumerate(figs):
        fig.update_layout(template="plotly_white", margin=dict(l=60, r=30, t=60, b=40))
        parts.append(pio.to_html(fig, include_plotlyjs=(i == 0), full_html=False,
                                 config={"displayModeBar": True, "responsive": True}))

    meta = meta or {}
    br = meta.get("base_rate")
    br_s = f"{br:.2f}" if isinstance(br, (int, float)) and br == br else "n/a"
    top_p = scores["precision"].iloc[0]
    lift_s = f"{top_p / br:.1f}×" if isinstance(br, (int, float)) and br == br and br > 0 else "n/a"
    header = f"""
    <div class="hdr">
      <h1>Crypto Crash Detector — Backtest Dashboard</h1>
      <p>{meta.get('assets', len(assets))} assets · {meta.get('period','')} ·
         {meta.get('n_clusters','?')} crash legs · {len(scores)} indicators ·
         {len(ensemble)} ensembles · base rate (random-signal precision) {br_s}</p>
      <p class="top">Best indicator: <b>{scores['indicator'].iloc[0]}</b>
         {scores['name'].iloc[0]} (composite {scores['composite_score'].iloc[0]:.3f},
         F1 {scores['f1'].iloc[0]:.3f}, precision {top_p:.2f} = {lift_s} base rate) ·
         Best ensemble: <b>{ensemble['members'].iloc[0]}</b> [{ensemble['strategy'].iloc[0]}]
         (F1 {ensemble['f1'].iloc[0]:.3f})</p>
    </div>"""
    css = """
    <style>
      body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
      .hdr{background:#1a1f36;color:#fff;padding:22px 32px}
      .hdr h1{margin:0 0 6px 0;font-size:24px}
      .hdr p{margin:2px 0;color:#c7cdd6;font-size:14px}
      .hdr .top{color:#ffd479}
      .card{background:#fff;margin:18px auto;max-width:1180px;border-radius:10px;
            box-shadow:0 1px 4px rgba(0,0,0,.12);padding:8px 12px}
      .note{max-width:1180px;margin:0 auto 22px;color:#666;font-size:12px;padding:0 14px}
      footer{text-align:center;color:#999;font-size:12px;padding:24px}
    </style>"""
    notes = [
        "Severity = peak drawdown within each crash leg. White = no crash that month.",
        "Composite = 0.30·precision + 0.25·recall + 0.20·(1−false alarm) + 0.15·lead + 0.10·consistency.",
        "Upper-right is better: high precision & recall. Bubble size = mean lead time (days).",
        "Triangles mark each indicator's warning days; red bands are actual BTC crash legs.",
        "Lower-right (high F1, low false alarm) is the sweet spot for ensembles.",
        "Green = indicator predicts that asset's crashes well (F1); red = poorly.",
        "Flat lines = robust to parameter changes; steep = overfit-prone. Dashed line = default.",
    ]
    body = [css, header]
    for (title, _), html, note in zip(figs, parts, notes):
        body.append(f'<div class="card">{html}</div><div class="note">{note}</div>')
    body.append('<footer>Generated by crypto_crash_detector · OHLCV-only · '
                'statistical signal study, not a trading system</footer>')
    full = "<!DOCTYPE html><html><head><meta charset='utf-8'>" \
           "<title>Crypto Crash Detector</title></head><body>" + "".join(body) + "</body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full)
    return out_path
