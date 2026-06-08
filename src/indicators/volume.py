"""Category D — Volume indicators (VOL1..VOL6). Look-ahead safe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _ta
from .base import Indicator


def _edge(state: pd.Series) -> pd.Series:
    return (state & ~state.shift(1, fill_value=False)).fillna(False)


class VOL1_VolumeSpike(Indicator):
    _id, _name, _category = "VOL1", "Abnormal Volume Spike (down day)", "volume"
    _defaults = {"lookback": 30, "z_threshold": 2.0}

    def compute(self, df):
        return _ta.zscore(df["volume"], self.params["lookback"])

    def generate_signals(self, df):
        z = self.compute(df)
        down = df["close"] < df["close"].shift(1)
        return _edge(((z > self.params["z_threshold"]) & down).fillna(False))


class VOL2_OBVDivergence(Indicator):
    _id, _name, _category = "VOL2", "OBV Bearish Divergence", "volume"
    _defaults = {"lookback": 20}

    def compute(self, df):
        return _ta.obv(df["close"], df["volume"])

    def generate_signals(self, df):
        n = self.params["lookback"]
        obv = self.compute(df)
        close = df["close"]
        price_hh = close == close.rolling(n, min_periods=n).max()
        prev_price_high = close.rolling(n, min_periods=n).max().shift(n)
        prev_obv_high = obv.rolling(n, min_periods=n).max().shift(n)
        # edge-trigger: one signal per divergence onset (the condition can hold
        # for several consecutive bars), keeping VOL2 comparable to the rest.
        return _edge((price_hh & (close > prev_price_high) & (obv < prev_obv_high)).fillna(False))


class VOL3_CMFNegFlip(Indicator):
    _id, _name, _category = "VOL3", "CMF Negative Flip", "volume"
    _defaults = {"period": 20}

    def compute(self, df):
        return _ta.cmf(df["high"], df["low"], df["close"], df["volume"], self.params["period"])

    def generate_signals(self, df):
        c = self.compute(df)
        return _ta.crosses_below(c, 0.0)


class VOL4_VPTBreakdown(Indicator):
    _id, _name, _category = "VOL4", "VPT EMA Breakdown", "volume"
    _defaults = {"vpt_ema": 14}

    def compute(self, df):
        return _ta.vpt(df["close"], df["volume"])

    def generate_signals(self, df):
        v = self.compute(df)
        v_ema = _ta.ema(v, self.params["vpt_ema"])
        return _ta.crosses_below(v, v_ema)


class VOL5_ForceIndexNegFlip(Indicator):
    _id, _name, _category = "VOL5", "Force Index Negative Flip", "volume"
    _defaults = {"period": 13}

    def compute(self, df):
        return _ta.force_index(df["close"], df["volume"], self.params["period"])

    def generate_signals(self, df):
        fi = self.compute(df)
        return _ta.crosses_below(fi, 0.0)


class VOL6_DownVolumeRatio(Indicator):
    _id, _name, _category = "VOL6", "Down/Up Volume Ratio Spike", "volume"
    _defaults = {"period": 5, "ratio_threshold": 2.0}

    def compute(self, df):
        n = self.params["period"]
        down = df["close"] < df["close"].shift(1)
        up = df["close"] > df["close"].shift(1)
        dvol = df["volume"].where(down, 0.0).rolling(n, min_periods=n).sum()
        uvol = df["volume"].where(up, 0.0).rolling(n, min_periods=n).sum()
        ratio = dvol / uvol.replace(0.0, np.nan)
        # all-down window (no up-volume) is the MOST bearish case -> +inf, not NaN
        ratio = ratio.where(~((uvol == 0) & (dvol > 0)), np.inf)
        return ratio

    def generate_signals(self, df):
        ratio = self.compute(df)
        # fire on onset of ratio > threshold (level edge), so the NaN->high and
        # all-down (+inf) transitions are captured, unlike a strict numeric cross.
        state = (ratio > self.params["ratio_threshold"]).fillna(False)
        return _edge(state)


def get_indicators():
    return [VOL1_VolumeSpike(), VOL2_OBVDivergence(), VOL3_CMFNegFlip(),
            VOL4_VPTBreakdown(), VOL5_ForceIndexNegFlip(), VOL6_DownVolumeRatio()]
