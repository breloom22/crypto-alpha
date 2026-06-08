"""Category E — Cross-asset / market-level indicators (X1..X7).

These operate on the full {symbol: DataFrame} dict and emit ONE market-wide
boolean series (indexed by the market calendar). The backtester evaluates that
single series against every asset's crash events. Look-ahead safe: all rolling
/ pct_change / ewm windows look backward; assets not yet listed contribute NaN
and are skipped, never imputed forward from the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _ta
from .base import CrossAssetIndicator, ASSETS, SECTORS, market_index, market_frame


def _edge(state: pd.Series) -> pd.Series:
    return (state.astype("boolean") & ~state.astype("boolean").shift(1, fill_value=False)).fillna(False).astype(bool)


def _wide(data, field, idx):
    return market_frame(data, field).reindex(idx)


class X1_AltWeakVsBTC(CrossAssetIndicator):
    _id, _name = "X1", "Broad Alt Weakness vs BTC"
    _defaults = {"period": 20, "threshold": -0.10}

    def compute(self, data):
        idx = market_index(data)
        close = _wide(data, "close", idx)
        alts = [a for a in ASSETS if a != "BTC" and a in close.columns]
        ratio = close[alts].div(close["BTC"], axis=0)
        return ratio.pct_change(self.params["period"]).mean(axis=1)

    def generate_signals(self, data):
        m = self.compute(data)
        return _edge(m < self.params["threshold"])


class X2_MultiRSIOverbought(CrossAssetIndicator):
    _id, _name = "X2", "Multi-Asset RSI Overbought"
    _defaults = {"period": 14, "rsi_threshold": 70, "count_threshold": 5}

    def compute(self, data):
        idx = market_index(data)
        rsi = pd.DataFrame({s: _ta.rsi(df["close"], self.params["period"])
                            for s, df in data.items()}).reindex(idx)
        return (rsi > self.params["rsi_threshold"]).sum(axis=1)

    def generate_signals(self, data):
        cnt = self.compute(data)
        return _edge(cnt >= self.params["count_threshold"])


class X3_CorrelationSpike(CrossAssetIndicator):
    _id, _name = "X3", "Cross-Asset Correlation Spike"
    _defaults = {"period": 30, "corr_threshold": 0.9}

    def compute(self, data):
        idx = market_index(data)
        ret = _wide(data, "close", idx).pct_change()
        n = self.params["period"]
        arr = ret.to_numpy()
        out = np.full(len(ret), np.nan)
        for t in range(n, len(ret)):
            win = arr[t - n + 1: t + 1]
            # keep assets with a full, finite window
            cols = ~np.isnan(win).any(axis=0)
            w = win[:, cols]
            if w.shape[1] < 2:
                continue
            c = np.corrcoef(w, rowvar=False)
            k = c.shape[0]
            off = (c.sum() - np.trace(c)) / (k * (k - 1))
            out[t] = off
        return pd.Series(out, index=ret.index)

    def generate_signals(self, data):
        m = self.compute(data)
        return _edge(m > self.params["corr_threshold"])


class X4_BreadthDeterioration(CrossAssetIndicator):
    _id, _name = "X4", "Market Breadth Deterioration"
    _defaults = {"ema_period": 20, "high_thr": 0.80, "low_thr": 0.50, "recent": 10}

    def compute(self, data):
        idx = market_index(data)
        close = _wide(data, "close", idx)
        ema = pd.DataFrame({s: _ta.ema(data[s]["close"], self.params["ema_period"])
                            for s in data}).reindex(idx)
        valid = close.notna() & ema.notna()
        above = (close > ema).where(valid)
        return above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)

    def generate_signals(self, data):
        breadth = self.compute(data)
        p = self.params
        was_high = (breadth.rolling(p["recent"], min_periods=1).max().shift(1) >= p["high_thr"])
        now_low = breadth <= p["low_thr"]
        return _edge((was_high & now_low).fillna(False))


class X5_SectorWeakness(CrossAssetIndicator):
    _id, _name = "X5", "All-Sector Simultaneous Weakness"
    _defaults = {"period": 7}

    def compute(self, data):
        idx = market_index(data)
        close = _wide(data, "close", idx)
        sector_ret = {}
        for sec, members in SECTORS.items():
            members = [m for m in members if m in close.columns]
            r = close[members].pct_change(self.params["period"])
            sector_ret[sec] = r.mean(axis=1)
        return pd.DataFrame(sector_ret)

    def generate_signals(self, data):
        sr = self.compute(data)
        all_neg = (sr < 0).all(axis=1)
        return _edge(all_neg.fillna(False))


class X6_VolWeightedMomentum(CrossAssetIndicator):
    _id, _name = "X6", "Volume-Weighted Market Momentum"
    _defaults = {"period": 5}

    def compute(self, data):
        idx = market_index(data)
        close = _wide(data, "close", idx)
        vol = _wide(data, "volume", idx)
        ret = close.pct_change()
        num = (ret * vol).sum(axis=1, min_count=1)
        den = vol.where(ret.notna()).sum(axis=1, min_count=1)
        return num / den.replace(0, np.nan)

    def generate_signals(self, data):
        vw = self.compute(data)
        neg = (vw < 0).astype(int)
        run = neg.rolling(self.params["period"], min_periods=self.params["period"]).sum()
        return _edge((run >= self.params["period"]).fillna(False))


class X7_VolatilityRegimeShift(CrossAssetIndicator):
    _id, _name = "X7", "Volatility Regime Shift"
    _defaults = {"atr_period": 14, "regime_lookback": 60}

    def compute(self, data):
        idx = market_index(data)
        # normalised ATR (ATR/close) per asset, averaged across the market
        natr = pd.DataFrame({
            s: _ta.atr(df["high"], df["low"], df["close"], self.params["atr_period"]) / df["close"]
            for s, df in data.items()}).reindex(idx)
        return natr.mean(axis=1)

    def generate_signals(self, data):
        m = self.compute(data)
        lb = self.params["regime_lookback"]
        regime = (m.rolling(lb, min_periods=20).mean() + m.rolling(lb, min_periods=20).std())
        return _ta.crosses_above(m, regime)


def get_indicators():
    return [X1_AltWeakVsBTC(), X2_MultiRSIOverbought(), X3_CorrelationSpike(),
            X4_BreadthDeterioration(), X5_SectorWeakness(), X6_VolWeightedMomentum(),
            X7_VolatilityRegimeShift()]
