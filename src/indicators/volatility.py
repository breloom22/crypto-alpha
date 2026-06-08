"""Category C — Volatility indicators (V1..V7). Look-ahead safe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _ta
from .base import Indicator


def _edge(state: pd.Series) -> pd.Series:
    return (state & ~state.shift(1, fill_value=False)).fillna(False)


class V1_BBLowerBreak(Indicator):
    _id, _name, _category = "V1", "Bollinger Lower Break", "volatility"
    _defaults = {"period": 20, "std": 2.0}

    def compute(self, df):
        mid, up, low, _ = _ta.bollinger(df["close"], self.params["period"], self.params["std"])
        return df["close"] - low

    def generate_signals(self, df):
        _, _, low, _ = _ta.bollinger(df["close"], self.params["period"], self.params["std"])
        return _ta.crosses_below(df["close"], low)


class V2_BBSqueezeExpansion(Indicator):
    _id, _name, _category = "V2", "BB Squeeze -> Expansion", "volatility"
    _defaults = {"period": 20, "squeeze_pctile": 10, "squeeze_lookback": 126, "recent": 10}

    def compute(self, df):
        _, _, _, width = _ta.bollinger(df["close"], self.params["period"], 2.0)
        return width

    def generate_signals(self, df):
        p = self.params
        _, _, _, width = _ta.bollinger(df["close"], p["period"], 2.0)
        q = width.rolling(p["squeeze_lookback"], min_periods=20).quantile(p["squeeze_pctile"] / 100.0)
        is_squeeze = width <= q
        squeeze_recent = is_squeeze.shift(1).rolling(p["recent"], min_periods=1).max().fillna(0) > 0
        breakout = _ta.crosses_above(width, width.rolling(p["period"], min_periods=p["period"]).mean())
        return (squeeze_recent & breakout).fillna(False)


class V3_ATRSpike(Indicator):
    _id, _name, _category = "V3", "ATR Spike", "volatility"
    _defaults = {"period": 14, "multiplier": 1.5, "avg_window": 20}

    def compute(self, df):
        return _ta.atr(df["high"], df["low"], df["close"], self.params["period"])

    def generate_signals(self, df):
        a = self.compute(df)
        baseline = self.params["multiplier"] * a.rolling(self.params["avg_window"], min_periods=self.params["avg_window"]).mean()
        return _ta.crosses_above(a, baseline)


class V4_GarmanKlassSpike(Indicator):
    _id, _name, _category = "V4", "Garman-Klass Vol Spike (2σ)", "volatility"
    _defaults = {"period": 20, "z_threshold": 2.0}

    def compute(self, df):
        hl = np.log(df["high"] / df["low"]) ** 2
        co = np.log(df["close"] / df["open"]) ** 2
        gk = np.sqrt((0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co).clip(lower=0.0))
        return gk

    def generate_signals(self, df):
        gk = self.compute(df)
        z = _ta.zscore(gk, self.params["period"])
        return _edge((z > self.params["z_threshold"]).fillna(False))


class V5_RangeExpansion(Indicator):
    _id, _name, _category = "V5", "Daily Range Expansion (2σ)", "volatility"
    _defaults = {"lookback": 30, "z_threshold": 2.0}

    def compute(self, df):
        return (df["high"] - df["low"]) / df["close"]

    def generate_signals(self, df):
        rng = self.compute(df)
        z = _ta.zscore(rng, self.params["lookback"])
        return _edge((z > self.params["z_threshold"]).fillna(False))


class V6_KeltnerLowerBreak(Indicator):
    _id, _name, _category = "V6", "Keltner Lower Break", "volatility"
    _defaults = {"period": 20, "atr_mult": 2.0}

    def compute(self, df):
        mid, up, low = _ta.keltner(df["high"], df["low"], df["close"], self.params["period"], self.params["atr_mult"])
        return df["close"] - low

    def generate_signals(self, df):
        _, _, low = _ta.keltner(df["high"], df["low"], df["close"], self.params["period"], self.params["atr_mult"])
        return _ta.crosses_below(df["close"], low)


class V7_ConsecutiveDownCandles(Indicator):
    _id, _name, _category = "V7", "Consecutive Down Candles", "volatility"
    _defaults = {"min_consecutive": 3}

    def compute(self, df):
        down = (df["close"] < df["open"]).astype(int)
        # consecutive-down streak length
        grp = (down == 0).cumsum()
        return down.groupby(grp).cumsum()

    def generate_signals(self, df):
        streak = self.compute(df)
        k = self.params["min_consecutive"]
        return ((streak >= k) & (streak.shift(1) < k)).fillna(False)


def get_indicators():
    return [V1_BBLowerBreak(), V2_BBSqueezeExpansion(), V3_ATRSpike(),
            V4_GarmanKlassSpike(), V5_RangeExpansion(), V6_KeltnerLowerBreak(),
            V7_ConsecutiveDownCandles()]
